"""F5-TTS provider (SWivid/F5-TTS).

Runs the model in-process on a GPU. License: code MIT, weights CC-BY-NC 4.0
(non-commercial / research only).

Install: ``pip install f5-tts``. Weights auto-download from HuggingFace on first use.

F5-TTS is reference-only by architecture -- every call requires a reference WAV and its
transcript. For the TTS task we use one fixed bundled reference clip from the manifest
(treated as "default voice"), so the result is reproducible per provider. For the cloning
task we condition on the per-speaker reference WAV + its ``.normalized.txt`` transcript.

Output is 24 kHz mono float32, cast to int16 PCM as the canonical storage format.
"""
import time
from pathlib import Path

import numpy as np

from voice_bench.providers.base import GenerationResult


DEFAULT_MODEL_ID = "F5TTS_v1_Base"


def _float_to_pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    return pcm.tobytes()


class F5TtsProvider:
    """In-process F5-TTS inference.

    The "default voice" for the speaker-free ``tts()`` task is a fixed reference WAV +
    its transcript, passed at construction time (so the provider has no surprise reads).
    """

    name = "f5_tts"
    supports_cloning = True
    SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        device: str = "cuda",
        default_ref_wav: Path | None = None,
        default_ref_text: str | None = None,
    ) -> None:
        self._device = device
        self._default_ref_wav = Path(default_ref_wav) if default_ref_wav else None
        self._default_ref_text = default_ref_text
        self._model = None  # lazily loaded

    # -- public API -----------------------------------------------------------

    def tts(
        self,
        text: str,
        voice_id: str,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        seed: int | None = None,
    ) -> GenerationResult:
        del voice_id  # F5 has no speaker-free mode; voice is implicit in default_ref_*.
        if self._default_ref_wav is None or self._default_ref_text is None:
            raise RuntimeError(
                "F5TtsProvider.tts() requires default_ref_wav and default_ref_text to be "
                "set at construction time (F5 is reference-only)."
            )
        wav, elapsed = self._infer(
            ref_file=self._default_ref_wav,
            ref_text=self._default_ref_text,
            gen_text=text,
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
            voice_id=f"default_ref:{self._default_ref_wav.stem}",
            character_count=len(text),
            seed=seed,
            reference_wav_path=str(self._default_ref_wav.resolve()),
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
        ref_path = Path(reference_wav_path)
        ref_text = reference_text or _read_normalized_txt_alongside(ref_path)
        if not ref_text:
            raise RuntimeError(
                f"F5-TTS clone() needs the reference transcript. "
                f"Pass reference_text=, or place a .normalized.txt next to {ref_path}."
            )
        wav, elapsed = self._infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=text,
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
            voice_id=f"clone:{ref_path.stem}",
            character_count=len(text),
            seed=seed,
            reference_wav_path=str(ref_path.resolve()),
        )

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- internals ------------------------------------------------------------

    def _ensure_loaded(self, model_id: str) -> None:
        if self._model is not None:
            return
        # Patch F5's audio loader before the first inference call. F5's
        # `infer_batch_process` does `torchaudio.load(ref_file)`, and torchaudio>=2.9
        # dispatches via torchcodec which needs libavutil from ffmpeg at runtime;
        # Octopus worker pods do not ship ffmpeg. soundfile has no native deps
        # beyond libsndfile and matches torchaudio.load's (Tensor, int) return
        # contract (channels, frames).
        import f5_tts.infer.utils_infer as _f5_ui
        import numpy as np
        import soundfile as sf
        import torch

        def _sf_load(uri, *args, **kwargs):
            del args, kwargs
            data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
            tensor = torch.from_numpy(np.ascontiguousarray(data.T))
            return tensor, int(sr)

        _f5_ui.torchaudio.load = _sf_load

        from f5_tts.api import F5TTS
        self._model = F5TTS(model=model_id, device=self._device)

    def _infer(
        self,
        *,
        ref_file: Path,
        ref_text: str,
        gen_text: str,
        seed: int | None,
    ) -> tuple[np.ndarray, float]:
        # F5 is reference-only; load lazily with the requested model id.
        self._ensure_loaded(DEFAULT_MODEL_ID)
        assert self._model is not None
        started = time.perf_counter()
        wav, sr, _spect = self._model.infer(
            ref_file=str(ref_file),
            ref_text=ref_text,
            gen_text=gen_text,
            seed=seed if seed is not None else -1,
        )
        elapsed = time.perf_counter() - started
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        if sr != self.SAMPLE_RATE:
            # F5-TTS v1 emits 24 kHz; resample defensively if a future build changes that.
            import librosa  # local import keeps the cpu-only dev path lighter
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        return wav, elapsed


def _read_normalized_txt_alongside(wav_path: Path) -> str | None:
    """Look for a sibling ``<utt_id>.normalized.txt`` (LibriTTS-R convention)."""
    candidate = wav_path.with_suffix("").with_suffix(".normalized.txt")
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None
