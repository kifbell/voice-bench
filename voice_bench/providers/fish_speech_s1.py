"""Fish Audio S1 mini provider (embedded).

Identical inference protocol to FishSpeechS2Pro: both are dual_ar architectures
sharing the tools/server/model_manager.py + TTSInferenceEngine code. Only the
checkpoint differs.

The launcher must:
  1. Clone fish-speech at commit 3c7cd3f0 (last S1-compatible main snapshot,
     2025-06-12, before the v2.0.0-beta S2 Pro rewrite landed)
  2. Set FISH_SPEECH_REPO_DIR + FISH_SPEECH_CHECKPOINT_DIR (provider-specific)
"""
import io
import os
import time
from pathlib import Path

import numpy as np

from voice_bench.providers._common import (
    SAMPLE_RATE_CANONICAL,
    float_to_pcm16_bytes,
    read_normalized_txt_alongside,
    resample_to_canonical,
)
from voice_bench.providers.base import GenerationResult


DEFAULT_MODEL_ID = "s1-mini"


class FishSpeechS1Provider:
    name = "fish_speech_s1"
    supports_cloning = True

    def __init__(
        self,
        *,
        fish_repo_path: str | None = None,
        ckpt_dir: str | None = None,
        device: str = "cuda",
    ):
        self._fish_repo = fish_repo_path or os.environ.get("FISH_SPEECH_REPO_DIR")
        self._ckpt_dir = ckpt_dir or os.environ.get("FISH_SPEECH_CHECKPOINT_DIR")
        if not self._fish_repo or not self._ckpt_dir:
            raise RuntimeError(
                "FishSpeechS1Provider requires fish_repo_path and ckpt_dir "
                "(or FISH_SPEECH_REPO_DIR + FISH_SPEECH_CHECKPOINT_DIR env vars)."
            )
        self._device = device
        self._engine = None
        self._inference_wrapper = None
        self._serve_tts_request_cls = None

    def tts(self, text, voice_id, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del voice_id
        wav, elapsed = self._infer(text=text, ref_wav=None, ref_text=None, seed=seed)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="tts",
            model_id=model_id,
            voice_id="default",
            character_count=len(text),
            seed=seed,
            reference_wav_path=None,
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        ref_path = Path(reference_wav_path)
        ref_text = reference_text or read_normalized_txt_alongside(ref_path)
        if not ref_text:
            raise RuntimeError(
                f"FishSpeech S1 clone() needs the reference transcript. "
                f"Pass reference_text=, or place a .normalized.txt next to {ref_path}."
            )
        wav, elapsed = self._infer(text=text, ref_wav=ref_path, ref_text=ref_text, seed=seed)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
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

    def cleanup(self):
        if self._engine is not None:
            del self._engine
            self._engine = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self):
        if self._engine is not None:
            return
        import pyrootutils
        pyrootutils.set_root(
            path=self._fish_repo,
            project_root_env_var=True,
            dotenv=False,
            pythonpath=True,
            cwd=False,
        )
        from tools.server.model_manager import ModelManager
        from tools.server.inference import inference_wrapper
        from fish_speech.utils.schema import ServeTTSRequest

        self._engine = ModelManager(
            mode="tts",
            device=self._device,
            half=False,
            compile=False,
            llama_checkpoint_path=str(self._ckpt_dir),
            decoder_checkpoint_path=str(Path(self._ckpt_dir) / "codec.pth"),
            decoder_config_name="modded_dac_vq",
        )
        self._inference_wrapper = inference_wrapper
        self._serve_tts_request_cls = ServeTTSRequest

    def _infer(self, *, text, ref_wav, ref_text, seed):
        self._ensure_loaded()
        from fish_speech.utils.schema import ServeReferenceAudio

        references = []
        if ref_wav is not None:
            references = [
                ServeReferenceAudio(audio=Path(ref_wav).read_bytes(), text=ref_text)
            ]
        req = self._serve_tts_request_cls(
            text=text,
            references=references,
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            format="wav",
            seed=seed,
            use_memory_cache="off",
            normalize=True,
        )

        started = time.perf_counter()
        chunks = list(self._inference_wrapper(req, self._engine.tts_inference_engine))
        elapsed = time.perf_counter() - started

        if not chunks:
            raise RuntimeError("Fish-speech S1 inference yielded no chunks")
        import soundfile as sf
        wav_bytes = chunks[-1]
        if not isinstance(wav_bytes, (bytes, bytearray)):
            raise RuntimeError(
                f"Fish-speech S1 final chunk is not bytes: {type(wav_bytes).__name__}"
            )
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        wav = data.mean(axis=1).astype(np.float32)
        wav = resample_to_canonical(wav, sr)
        return wav, elapsed
