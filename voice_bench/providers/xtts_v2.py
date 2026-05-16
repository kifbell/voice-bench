"""Coqui XTTS-v2 provider.

Runs the model in-process on a GPU. License: CPML (non-commercial / research only).

Install: ``pip install coqui-tts`` (the idiap fork of the unmaintained ``TTS`` package).
Model weights (~1.8 GB) are auto-downloaded to ``~/.cache/tts/`` or ``$TTS_HOME`` on first use.

XTTS always conditions on a speaker reference. For the TTS task we use one of the bundled
"studio" speakers (e.g. ``"Claribel Dervla"``) so generation is reproducible and does not
require an external reference WAV; for the cloning task we pass ``speaker_wav=ref_wav_path``.

Output is 24 kHz mono float32, which we cast to int16 PCM as the canonical storage format.
"""
import time
from pathlib import Path

import numpy as np

from voice_bench.providers.base import GenerationResult

# Default model spec for sidecar / model_id field.
DEFAULT_MODEL_ID = "xtts_v2"

# Default "voice" for the TTS task: a bundled studio speaker name (no external ref WAV).
DEFAULT_VOICE_ID = "Claribel Dervla"


def _float_to_pcm16_bytes(wav: np.ndarray) -> bytes:
    """Clip a float waveform in [-1, 1] and serialize to little-endian int16 PCM bytes."""
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    return pcm.tobytes()


class XttsV2Provider:
    """In-process Coqui XTTS-v2 inference.

    Lazily loads weights on first call. ``cleanup()`` frees the model and CUDA cache.
    """

    name = "xtts_v2"
    supports_cloning = True
    SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        device: str = "cuda",
        language: str = "en",
        use_deepspeed: bool = False,
    ) -> None:
        self._device = device
        self._language = language
        self._use_deepspeed = use_deepspeed
        self._model = None  # lazily loaded
        self._config = None
        # Cache: speaker name -> (gpt_cond_latent, speaker_embedding). Reusing built-in
        # studio speakers across calls avoids recomputing their embeddings.
        self._studio_speaker_cache: dict[str, tuple] = {}
        # Cache for cloning: ref wav path (str) -> (gpt_cond_latent, speaker_embedding).
        self._clone_latent_cache: dict[str, tuple] = {}

    # -- public API -----------------------------------------------------------

    def tts(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        seed: int | None = None,
    ) -> GenerationResult:
        self._ensure_loaded()
        gpt_cond, spk_emb = self._get_studio_speaker_latents(voice_id)
        wav, elapsed = self._inference(
            text=text,
            gpt_cond_latent=gpt_cond,
            speaker_embedding=spk_emb,
            seed=seed,
        )
        return GenerationResult(
            audio_pcm=_float_to_pcm16_bytes(wav),
            sample_rate=self.SAMPLE_RATE,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="tts",
            model_id=model_id,
            voice_id=voice_id,
            character_count=len(text),
            seed=seed,
            reference_wav_path=None,
        )

    def clone(
        self,
        text: str,
        reference_wav_path: Path,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        seed: int | None = None,
        reference_text: str | None = None,
    ) -> GenerationResult:
        del reference_text  # XTTS infers prosody from the wav alone, no transcript needed.
        self._ensure_loaded()
        ref_key = str(Path(reference_wav_path).resolve())
        gpt_cond, spk_emb = self._get_clone_latents(ref_key)
        wav, elapsed = self._inference(
            text=text,
            gpt_cond_latent=gpt_cond,
            speaker_embedding=spk_emb,
            seed=seed,
        )
        return GenerationResult(
            audio_pcm=_float_to_pcm16_bytes(wav),
            sample_rate=self.SAMPLE_RATE,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="cloning",
            model_id=model_id,
            voice_id=f"clone:{Path(reference_wav_path).stem}",
            character_count=len(text),
            seed=seed,
            reference_wav_path=ref_key,
        )

    def cleanup(self) -> None:
        """Free the model and CUDA cache. Idempotent."""
        if self._model is not None:
            del self._model
            self._model = None
        self._config = None
        self._studio_speaker_cache.clear()
        self._clone_latent_cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- internals ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imports are deferred so importing this module on a CPU-only dev box
        # doesn't pull in torch/CUDA.
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        manager = ModelManager()
        model_dir, _, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
        config = XttsConfig()
        config.load_json(str(Path(model_dir) / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(model_dir),
            use_deepspeed=self._use_deepspeed,
        )
        if self._device.startswith("cuda"):
            model.cuda()
        self._model = model
        self._config = config

    def _get_studio_speaker_latents(self, speaker_name: str) -> tuple:
        if speaker_name in self._studio_speaker_cache:
            return self._studio_speaker_cache[speaker_name]
        assert self._model is not None
        # XTTS bundles ~58 studio speakers; speaker_manager.speakers maps name -> latents.
        speakers = getattr(self._model, "speaker_manager", None)
        if speakers is None or not getattr(speakers, "speakers", None):
            raise RuntimeError(
                "XTTS-v2 speaker_manager has no studio speakers loaded; "
                "check that speakers_xtts.pth is present in the model dir."
            )
        if speaker_name not in speakers.speakers:
            available = list(speakers.speakers.keys())[:8]
            raise KeyError(
                f"Unknown XTTS studio speaker {speaker_name!r}. "
                f"First few available: {available}..."
            )
        entry = speakers.speakers[speaker_name]
        gpt_cond = entry["gpt_cond_latent"]
        spk_emb = entry["speaker_embedding"]
        self._studio_speaker_cache[speaker_name] = (gpt_cond, spk_emb)
        return gpt_cond, spk_emb

    def _get_clone_latents(self, ref_key: str) -> tuple:
        if ref_key in self._clone_latent_cache:
            return self._clone_latent_cache[ref_key]
        assert self._model is not None
        # ``get_conditioning_latents`` accepts a list of WAV paths; using one is fine.
        gpt_cond, spk_emb = self._model.get_conditioning_latents(audio_path=[ref_key])
        self._clone_latent_cache[ref_key] = (gpt_cond, spk_emb)
        return gpt_cond, spk_emb

    def _inference(
        self,
        *,
        text: str,
        gpt_cond_latent,
        speaker_embedding,
        seed: int | None,
    ) -> tuple[np.ndarray, float]:
        assert self._model is not None and self._config is not None
        if seed is not None:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        started = time.perf_counter()
        out = self._model.inference(
            text=text,
            language=self._language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        elapsed = time.perf_counter() - started
        wav = out["wav"]  # numpy or torch.Tensor depending on version
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        return wav, elapsed
