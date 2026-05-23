"""CosyVoice 2 provider (Alibaba FunAudioLLM).

Latest open-weight CosyVoice as of May 2026. CosyVoice 3 has a paper but the open weights
trail the paper; if the v3 release is on HF use ``model_id`` to point there, otherwise
v2 (CosyVoice2-0.5B) is the safe default.

Install: clone the FunAudioLLM/CosyVoice repo, ``pip install -r requirements.txt`` from
it, then ``pip install -e .`` so the cosyvoice package is importable. Weights are pulled
through ``modelscope`` or ``huggingface_hub`` on first use.

License (code + weights): Apache-2.0 as of v2.

CosyVoice 2's zero-shot path needs ref WAV + ref transcript (same as F5). For the TTS
task with no per-speaker conditioning we use a fixed default reference clip plus its
transcript, passed at construction time -- matches the F5 wiring.
"""
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


DEFAULT_MODEL_ID = "CosyVoice2-0.5B"


class CosyVoice2Provider:
    name = "cosyvoice2"
    supports_cloning = True
    NATIVE_SAMPLE_RATE = 24000  # CosyVoice 2 emits 24 kHz already

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_dir: str | None = None,
        default_ref_wav: Path | None = None,
        default_ref_text: str | None = None,
        load_jit: bool = False,
        load_trt: bool = False,
        fp16: bool = False,
    ):
        # ``device`` is recorded but CosyVoice 2 picks the device internally via torch's
        # default; cuda is used if available. We pass it for parity with the other providers.
        self._device = device
        self._model_dir = model_dir or f"pretrained_models/{DEFAULT_MODEL_ID}"
        self._default_ref_wav = Path(default_ref_wav) if default_ref_wav else None
        self._default_ref_text = default_ref_text
        self._load_jit = load_jit
        self._load_trt = load_trt
        self._fp16 = fp16
        self._model = None

    def tts(self, text, voice_id, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del voice_id  # CosyVoice 2 is reference-only; voice carried in default_ref_*.
        if self._default_ref_wav is None or self._default_ref_text is None:
            raise RuntimeError(
                "CosyVoice2Provider.tts() needs default_ref_wav and default_ref_text "
                "(CosyVoice 2 zero-shot path is reference-only)."
            )
        wav, elapsed = self._infer(
            ref_wav=self._default_ref_wav,
            ref_text=self._default_ref_text,
            gen_text=text,
            seed=seed,
        )
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
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

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        ref_path = Path(reference_wav_path)
        ref_text = reference_text or read_normalized_txt_alongside(ref_path)
        if not ref_text:
            raise RuntimeError(
                f"CosyVoice 2 clone() needs the reference transcript. "
                f"Pass reference_text=, or place a .normalized.txt next to {ref_path}."
            )
        wav, elapsed = self._infer(ref_wav=ref_path, ref_text=ref_text, gen_text=text, seed=seed)
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
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- internals -------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self._model = CosyVoice2(
            self._model_dir,
            load_jit=self._load_jit,
            load_trt=self._load_trt,
            fp16=self._fp16,
        )

    def _infer(self, *, ref_wav, ref_text, gen_text, seed):
        self._ensure_loaded()
        if seed is not None:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # CosyVoice 2's inference_zero_shot wants prompt speech at 16 kHz mono as a
        # torch.Tensor with shape [1, T] (or [T]).
        import soundfile as sf
        import torch
        data, sr = sf.read(str(ref_wav), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        prompt = torch.from_numpy(np.ascontiguousarray(mono))[None, :]
        if sr != 16000:
            import torchaudio.functional as F
            prompt = F.resample(prompt, sr, 16000)

        started = time.perf_counter()
        chunks = []
        for out in self._model.inference_zero_shot(
            gen_text,
            ref_text,
            prompt,
            stream=False,
        ):
            chunks.append(out["tts_speech"])
        elapsed = time.perf_counter() - started

        # Concatenate the streaming chunks (stream=False yields a single chunk in practice).
        wav_t = torch.cat(chunks, dim=-1).squeeze()
        wav = wav_t.detach().cpu().numpy().astype(np.float32)
        wav = resample_to_canonical(wav, self.NATIVE_SAMPLE_RATE)
        return wav, elapsed
