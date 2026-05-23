"""Fish Audio S1 mini provider via fish-speech main api_server.

S1 mini and S2 Pro share the same Dual-AR architecture (fishaudio/s1-mini's
config.model_type is "dual_ar"). The fish-speech main repo's tools/api_server.py
loads either checkpoint through --llama-checkpoint-path. So this provider is
just an HTTP client to a server the launcher starts with the s1-mini weights.

Same protocol as FishSpeechS2ProProvider: POST /v1/tts with JSON body
{"text": ..., "references": [{"audio": <base64>, "text": ...}]}.
"""
import base64
import io
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
DEFAULT_SERVER_URL = "http://127.0.0.1:8080"


class FishSpeechS1Provider:
    name = "fish_speech_s1"
    supports_cloning = True

    def __init__(self, *, server_url: str = DEFAULT_SERVER_URL, request_timeout: float = 600.0):
        self._server_url = server_url.rstrip("/")
        self._timeout = request_timeout
        self._session = None

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
        if self._session is not None:
            self._session.close()
            self._session = None

    def _ensure_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()

    def _infer(self, *, text, ref_wav, ref_text, seed):
        self._ensure_session()
        body: dict = {
            "text": text,
            "format": "wav",
            "references": [],
            "use_memory_cache": "off",
            "normalize": True,
        }
        if seed is not None:
            body["seed"] = seed
        if ref_wav is not None:
            audio_b64 = base64.b64encode(Path(ref_wav).read_bytes()).decode("ascii")
            body["references"] = [{"audio": audio_b64, "text": ref_text}]

        url = f"{self._server_url}/v1/tts"
        started = time.perf_counter()
        resp = self._session.post(url, json=body, timeout=self._timeout)
        elapsed = time.perf_counter() - started
        if not resp.ok:
            raise RuntimeError(
                f"fish-speech api_server returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        import soundfile as sf
        buf = io.BytesIO(resp.content)
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        wav = data.mean(axis=1).astype(np.float32)
        wav = resample_to_canonical(wav, sr)
        return wav, elapsed
