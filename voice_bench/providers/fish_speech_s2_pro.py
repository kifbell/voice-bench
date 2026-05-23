"""Fish Audio Speech S2 Pro provider.

S2 Pro is Fish Audio's commercial product. Unlike Fish Speech S1 it has no
open-weight release as of May 2026 -- it is reachable only via Fish Audio's HTTP API
(https://fish.audio). This provider therefore behaves like the ElevenLabs one: a thin
HTTP client around the fish-audio-sdk.

Install: ``pip install fish-audio-sdk`` (or fall back to a raw ``requests`` POST against
the same endpoint). Requires ``FISH_AUDIO_API_KEY`` in the environment.

Zero-shot voice cloning: pass a reference WAV (the SDK uploads it as part of the
request) and target text. Reference transcript is optional but recommended.

Output: MP3 / WAV bytes at 24 kHz (configurable). We always request 24 kHz mono PCM-16.

NOTE: this provider costs money per request. Use --pilot on the first run, audit the
generated sample, then confirm before running --full.
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


DEFAULT_MODEL_ID = "speech-s2-pro"


class FishSpeechS2ProProvider:
    name = "fish_speech_s2_pro"
    supports_cloning = True

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("FISH_AUDIO_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "FishSpeechS2ProProvider needs FISH_AUDIO_API_KEY in the environment "
                "or passed via api_key= ."
            )
        self._session = None

    def tts(self, text, voice_id, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del voice_id  # The Fish API picks a default voice when no reference is supplied.
        wav, elapsed = self._infer(
            text=text, ref_wav=None, ref_text=None, model_id=model_id, seed=seed
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
            voice_id="default",
            character_count=len(text),
            seed=seed,
            reference_wav_path=None,
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        ref_path = Path(reference_wav_path)
        ref_text = reference_text or read_normalized_txt_alongside(ref_path)
        wav, elapsed = self._infer(
            text=text,
            ref_wav=ref_path,
            ref_text=ref_text,
            model_id=model_id,
            seed=seed,
        )
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
        self._session = None

    # -- internals -------------------------------------------------------------

    def _ensure_session(self):
        if self._session is not None:
            return
        try:
            from fish_audio_sdk import Session
            self._session = Session(self._api_key)
        except ImportError as e:
            raise RuntimeError(
                "fish-audio-sdk is not installed. Run `pip install fish-audio-sdk`."
            ) from e

    def _infer(self, *, text, ref_wav, ref_text, model_id, seed):
        self._ensure_session()
        # Lazy import to keep this module CPU-importable on dev boxes without the SDK.
        from fish_audio_sdk import TTSRequest, ReferenceAudio

        request_kwargs: dict = {"text": text, "format": "wav", "mp3_bitrate": 0}
        # The SDK currently exposes ``backend=`` for picking the model family (s1 vs s2-pro etc).
        request_kwargs["backend"] = model_id
        if seed is not None:
            request_kwargs["seed"] = seed

        if ref_wav is not None:
            with open(ref_wav, "rb") as f:
                audio_bytes = f.read()
            request_kwargs["references"] = [
                ReferenceAudio(audio=audio_bytes, text=ref_text or "")
            ]

        started = time.perf_counter()
        # ``session.tts`` returns a streaming iterator of bytes chunks.
        buf = io.BytesIO()
        for chunk in self._session.tts(TTSRequest(**request_kwargs)):
            buf.write(chunk)
        elapsed = time.perf_counter() - started

        # Decode the WAV bytes the API returned. soundfile handles arbitrary container SRs.
        import soundfile as sf
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        wav = data.mean(axis=1).astype(np.float32)
        wav = resample_to_canonical(wav, sr)
        return wav, elapsed
