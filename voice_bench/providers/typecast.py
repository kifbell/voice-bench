"""Typecast (Neosapience) TTS provider — TTS + Instant Voice Cloning.

API: https://typecast.ai
SDK: pip install --upgrade typecast-python
Auth: TYPECAST_API_KEY env var

Output format: WAV bytes; we decode via soundfile and resample to 24 kHz int16 PCM.

License: paid tier required for sustained API usage; free tier may have IVC.
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


DEFAULT_MODEL_ID = "typecast-tts"
DEFAULT_VOICE_ID = ""  # picked at runtime from list_voices() if empty


class TypecastProvider:
    name = "typecast"
    supports_cloning = True

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("TYPECAST_API_KEY")
        if not self._api_key:
            raise RuntimeError("TypecastProvider needs TYPECAST_API_KEY")
        self._client = None
        self._cached_default_voice: str | None = None
        # Per-speaker IVC voice cache so we re-use one clone across multiple targets
        # for the same speaker (Typecast may charge / rate-limit IVC creation).
        self._clone_voice_cache: dict[str, str] = {}

    def _ensure_client(self):
        if self._client is not None:
            return
        # Try multiple import paths -- typecast-python's exact entry point varies.
        try:
            from typecast import Typecast  # most likely
            self._client = Typecast(api_key=self._api_key)
            return
        except Exception:
            pass
        try:
            from typecast import Client
            self._client = Client(api_key=self._api_key)
            return
        except Exception:
            pass
        # Fall back to raw HTTP client we'll wire up if SDK is unusable.
        import requests
        self._client = _RawTypecastClient(self._api_key, requests.Session())

    def _default_voice(self) -> str:
        if self._cached_default_voice:
            return self._cached_default_voice
        # Try to list voices and pick a neutral English one.
        try:
            voices = self._client.list_voices() if hasattr(self._client, "list_voices") else None
            if voices:
                # Prefer the first en-US female "neutral" voice.
                for v in voices:
                    name = (v.get("name") if isinstance(v, dict) else getattr(v, "name", "")) or ""
                    lang = (v.get("language") if isinstance(v, dict) else getattr(v, "language", "")) or ""
                    if "en" in lang.lower():
                        vid = v.get("id") if isinstance(v, dict) else getattr(v, "id", None)
                        if vid:
                            self._cached_default_voice = vid
                            return vid
                # Else first voice.
                first = voices[0]
                vid = first.get("id") if isinstance(first, dict) else getattr(first, "id", None)
                if vid:
                    self._cached_default_voice = vid
                    return vid
        except Exception:
            pass
        # Fall back: synth without voice_id (some SDKs default automatically).
        return ""

    def tts(self, text, voice_id=DEFAULT_VOICE_ID, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del seed
        self._ensure_client()
        vid = voice_id or self._default_voice()
        wav, sr, elapsed = self._call_tts(text=text, voice_id=vid)
        wav_24k = resample_to_canonical(wav, sr)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav_24k),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="tts",
            model_id=model_id,
            voice_id=vid,
            character_count=len(text),
            seed=None,
            reference_wav_path=None,
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        del seed
        self._ensure_client()
        ref_path = Path(reference_wav_path)
        # Re-use clone voice across multiple targets per speaker (key = reference file path).
        clone_voice_id = self._clone_voice_cache.get(str(ref_path))
        if not clone_voice_id:
            clone_voice_id = self._create_clone_voice(ref_path)
            self._clone_voice_cache[str(ref_path)] = clone_voice_id

        wav, sr, elapsed = self._call_tts(text=text, voice_id=clone_voice_id)
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
        if self._client is None:
            return
        for ref, vid in list(self._clone_voice_cache.items()):
            try:
                if hasattr(self._client, "delete_voice"):
                    self._client.delete_voice(vid)
                elif hasattr(self._client, "voices") and hasattr(self._client.voices, "delete"):
                    self._client.voices.delete(vid)
            except Exception:
                pass
        self._clone_voice_cache.clear()
        self._client = None

    # -- internals -------------------------------------------------------------

    def _call_tts(self, text: str, voice_id: str) -> tuple[np.ndarray, int, float]:
        started = time.perf_counter()
        # Try several common method names.
        for method in ("text_to_speech", "tts", "synthesize"):
            if hasattr(self._client, method):
                try:
                    result = getattr(self._client, method)(text=text, voice_id=voice_id)
                    elapsed = time.perf_counter() - started
                    return _decode_audio_response(result) + (elapsed,)
                except TypeError:
                    # Method exists but signature differs; try positional.
                    try:
                        result = getattr(self._client, method)(text, voice_id)
                        elapsed = time.perf_counter() - started
                        return _decode_audio_response(result) + (elapsed,)
                    except Exception:
                        continue
                except Exception:
                    continue
        raise RuntimeError("Typecast SDK has no recognised TTS method (tried text_to_speech/tts/synthesize)")

    def _create_clone_voice(self, ref_path: Path) -> str:
        # Try several common clone-creation method names.
        for path in (
            ("voices", "clone"),
            ("voices", "create"),
            ("clone_voice",),
            ("instant_clone",),
        ):
            obj = self._client
            try:
                for p in path:
                    obj = getattr(obj, p)
            except AttributeError:
                continue
            try:
                # Best-effort: pass file path or open file as the audio kwarg.
                result = obj(name=f"voicebench_{ref_path.stem}", audio_file=str(ref_path))
                vid = _extract_voice_id(result)
                if vid:
                    return vid
            except TypeError:
                try:
                    with open(ref_path, "rb") as f:
                        result = obj(name=f"voicebench_{ref_path.stem}", audio=f)
                    vid = _extract_voice_id(result)
                    if vid:
                        return vid
                except Exception:
                    continue
            except Exception:
                continue
        raise RuntimeError("Typecast SDK has no recognised voice-clone method")


def _extract_voice_id(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("id") or result.get("voice_id") or result.get("uuid") or ""
    for attr in ("id", "voice_id", "uuid"):
        v = getattr(result, attr, None)
        if v:
            return v
    return ""


def _decode_audio_response(resp) -> tuple[np.ndarray, int]:
    """Accept bytes / dict-with-audio-key / object-with-attribute and return (wav_f32, sr)."""
    import soundfile as sf
    if isinstance(resp, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(resp), dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(sr)
    if isinstance(resp, dict):
        for k in ("audio", "audio_bytes", "wav", "data"):
            if k in resp and isinstance(resp[k], (bytes, bytearray)):
                data, sr = sf.read(io.BytesIO(resp[k]), dtype="float32", always_2d=True)
                return data.mean(axis=1).astype(np.float32), int(sr)
        if "audio_url" in resp:
            import requests
            buf = requests.get(resp["audio_url"], timeout=60).content
            data, sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
            return data.mean(axis=1).astype(np.float32), int(sr)
    # Object with attributes.
    for attr in ("audio", "audio_bytes", "wav", "data"):
        v = getattr(resp, attr, None)
        if isinstance(v, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(v), dtype="float32", always_2d=True)
            return data.mean(axis=1).astype(np.float32), int(sr)
    raise RuntimeError(f"Typecast: cannot decode audio response of type {type(resp).__name__}")


class _RawTypecastClient:
    """Tiny HTTP wrapper as fallback if the typecast-python SDK API differs.

    Endpoints guessed from public Typecast docs; will need adjustment based on
    actual response shapes from the real server.
    """

    BASE = "https://api.typecast.ai"

    def __init__(self, api_key: str, session):
        self.api_key = api_key
        self.session = session

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def text_to_speech(self, *, text: str, voice_id: str):
        body = {"text": text, "voice_id": voice_id, "audio_format": "wav"}
        r = self.session.post(f"{self.BASE}/v1/text-to-speech", json=body, headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.content
