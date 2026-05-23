"""OpenAI TTS provider.

Cloud API, no GPU. Requires OPENAI_API_KEY env var.

Per research_plan.md: model=tts-1 (cheaper, $15/1M chars), voice=alloy (neutral).
Optional tts-1-hd for one extra point on the Pareto plot.

No voice cloning: OpenAI doesn't offer it. clone() falls back to the default voice
and tags task='cloning' so downstream metric joins line up with other providers.

Output: PCM-16 24 kHz mono via response_format='pcm' -- canonical contract is identical,
no resampling needed.
"""
import os
import time
from pathlib import Path

import numpy as np

from voice_bench.providers._common import (
    SAMPLE_RATE_CANONICAL,
    float_to_pcm16_bytes,
    resample_to_canonical,
)
from voice_bench.providers.base import GenerationResult


DEFAULT_MODEL_ID = "tts-1"
DEFAULT_VOICE_ID = "alloy"


class OpenaiTtsProvider:
    name = "openai_tts"
    supports_cloning = False

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OpenaiTtsProvider needs OPENAI_API_KEY env var (or pass via constructor)."
            )
        self._client = None

    def tts(self, text, voice_id=DEFAULT_VOICE_ID, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del seed  # No seed control in OpenAI TTS.
        wav, elapsed = self._synthesize(text=text, voice_id=voice_id, model_id=model_id)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="tts",
            model_id=model_id,
            voice_id=voice_id,
            character_count=len(text),
            seed=None,
            reference_wav_path=None,
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        del reference_wav_path, reference_text, seed
        # OpenAI has no cloning -- fall back to default voice, tag task='cloning'
        # so downstream parquet joins still find a row per (provider, task, utt_id).
        wav, elapsed = self._synthesize(text=text, voice_id=DEFAULT_VOICE_ID, model_id=model_id)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="cloning",
            model_id=model_id,
            voice_id=DEFAULT_VOICE_ID,
            character_count=len(text),
            seed=None,
            reference_wav_path=None,
        )

    def cleanup(self):
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key)

    def _synthesize(self, *, text, voice_id, model_id):
        self._ensure_client()
        started = time.perf_counter()
        # response_format='pcm' returns raw int16 PCM at 24 kHz mono. No header.
        resp = self._client.audio.speech.create(
            model=model_id,
            voice=voice_id,
            input=text,
            response_format="pcm",
        )
        pcm_bytes = resp.read()
        elapsed = time.perf_counter() - started
        wav = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # OpenAI documents 24 kHz native for tts-1/tts-1-hd. Resample defensively.
        wav = resample_to_canonical(wav, SAMPLE_RATE_CANONICAL)
        return wav, elapsed
