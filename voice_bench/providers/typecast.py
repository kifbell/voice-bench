"""Typecast (Neosapience) Voice Cloning provider — cloning ONLY.

API: https://api.typecast.ai (X-API-KEY header)
SDK: pip install typecast-python (typecast.Typecast for TTS only).

Cloning isn't in the public SDK -- use the REST endpoint
POST /v1/instant-voice-cloning to upload a reference WAV; it returns a voice_id.
Then synthesise text with that voice via SDK's text_to_speech.

License: paid tier required for IVC; Free tier returns 402.
"""
import io
import os
import time
from pathlib import Path

import numpy as np
import requests

from voice_bench.providers._common import (
    SAMPLE_RATE_CANONICAL,
    float_to_pcm16_bytes,
    read_normalized_txt_alongside,
    resample_to_canonical,
)
from voice_bench.providers.base import GenerationResult


DEFAULT_MODEL_ID = "ssfm-v30"  # Typecast's latest TTS model
TYPECAST_API_HOST = "https://api.typecast.ai"


class TypecastProvider:
    name = "typecast"
    supports_cloning = True

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("TYPECAST_API_KEY")
        if not self._api_key:
            raise RuntimeError("TypecastProvider needs TYPECAST_API_KEY")
        self._session = None
        self._sdk_client = None
        self._clone_voice_cache: dict[str, str] = {}

    def _ensure(self):
        if self._session is not None:
            return
        self._session = requests.Session()
        self._session.headers.update({"X-API-KEY": self._api_key})
        from typecast import Typecast
        self._sdk_client = Typecast(api_key=self._api_key)

    def tts(self, text, voice_id, model_id=DEFAULT_MODEL_ID, *, seed=None):
        raise NotImplementedError(
            "TypecastProvider is voice-cloning only; --tasks tts is disabled. "
            "Run with --tasks cloning."
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        del seed
        self._ensure()
        ref_path = Path(reference_wav_path)
        clone_voice_id = self._clone_voice_cache.get(str(ref_path))
        if not clone_voice_id:
            clone_voice_id = self._create_clone_voice(ref_path)
            self._clone_voice_cache[str(ref_path)] = clone_voice_id

        wav, sr, elapsed = self._synthesize(text=text, voice_id=clone_voice_id, model_id=model_id)
        wav_24k = resample_to_canonical(wav, sr)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav_24k),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="cloning",
            model_id=model_id,
            voice_id=f"clone:{ref_path.stem}",
            character_count=len(text),
            seed=None,
            reference_wav_path=str(ref_path.resolve()),
        )

    def cleanup(self):
        # Best-effort: delete clone voices to free slot quota.
        if self._session is None:
            return
        for ref, vid in list(self._clone_voice_cache.items()):
            try:
                # DELETE /v1/voices/{voice_id} — undocumented but common shape.
                self._session.delete(
                    f"{TYPECAST_API_HOST}/v1/voices/{vid}",
                    timeout=30,
                )
            except Exception:
                pass
        self._clone_voice_cache.clear()
        self._session = None
        self._sdk_client = None

    # -- internals -------------------------------------------------------------

    def _create_clone_voice(self, ref_path: Path) -> str:
        """POST /v1/instant-voice-cloning to upload reference WAV, returns voice_id."""
        # Endpoint shape: multipart POST with audio file. The exact field name varies
        # between API versions; try a couple of common shapes.
        # Per https://typecast.ai/docs/api-reference/voices/instant-cloning#instant-cloning
        # the canonical endpoint sits under /voices/instant-cloning. We try the few
        # common multipart field names since the doc doesn't pin one.
        candidates = (
            ("/v1/voices/instant-cloning", "voice_file"),
            ("/v1/voices/instant-cloning", "audio"),
            ("/v1/voices/instant-cloning", "file"),
            ("/v1/instant-voice-cloning", "voice_file"),
            ("/v1/instant-voice-cloning", "audio"),
        )
        last_err: Exception | None = None
        for endpoint, file_field in candidates:
            try:
                with open(ref_path, "rb") as f:
                    files = {file_field: (ref_path.name, f, "audio/wav")}
                    data = {"name": f"voicebench_{ref_path.stem}"}
                    r = self._session.post(
                        f"{TYPECAST_API_HOST}{endpoint}",
                        files=files,
                        data=data,
                        timeout=120,
                    )
                if r.status_code in (200, 201):
                    j = r.json()
                    for k in ("voice_id", "id", "uuid"):
                        if k in j:
                            return j[k]
                    if "voice" in j and isinstance(j["voice"], dict):
                        for k in ("voice_id", "id", "uuid"):
                            if k in j["voice"]:
                                return j["voice"][k]
                    raise RuntimeError(f"unexpected response: {j!r}")
                last_err = RuntimeError(f"HTTP {r.status_code} from {endpoint}: {r.text[:300]}")
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Typecast IVC failed across {len(candidates)} endpoint candidates: {last_err}")

    def _synthesize(self, *, text: str, voice_id: str, model_id: str) -> tuple[np.ndarray, int, float]:
        from typecast.models import TTSRequest, TTSModel
        # Map model id to enum -- the SDK demands the enum value.
        model_enum = TTSModel.SSFM_V30 if model_id == "ssfm-v30" else TTSModel.SSFM_V21
        req = TTSRequest(text=text, voice_id=voice_id, model=model_enum)

        started = time.perf_counter()
        resp = self._sdk_client.text_to_speech(req)
        elapsed = time.perf_counter() - started

        import soundfile as sf
        data, sr = sf.read(io.BytesIO(resp.audio_data), dtype="float32", always_2d=True)
        wav = data.mean(axis=1).astype(np.float32)
        return wav, int(sr), elapsed
