"""Resemble AI provider — TTS + Rapid/Instant Voice Cloning.

SDK: pip install resemble; from resemble import Resemble
Auth: Resemble.api_key("...")
TTS: Resemble.v2.clips.create_sync(...)
Cloning: Resemble.v2.voices.create + Resemble.v2.voices.build (async, poll)

Output: Clip object has .audio_src URL or raw bytes; varies. SDK details:
https://docs.resemble.ai/api-reference/text-to-speech/synthesize

We default to a project + voice picked at runtime from the account's catalogue.
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


DEFAULT_MODEL_ID = "resemble-v2"


class ResembleProvider:
    name = "resemble"
    supports_cloning = True

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
        if not self._api_key:
            raise RuntimeError("ResembleProvider needs RESEMBLE_API_KEY")
        self._initialised = False
        self._default_voice_uuid: str | None = None
        self._project_uuid: str | None = None
        # Per-speaker cache so we don't rebuild a voice for each utterance.
        self._clone_voices: dict[str, str] = {}

    def _ensure(self):
        if self._initialised:
            return
        from resemble import Resemble
        Resemble.api_key(self._api_key)
        self._initialised = True
        self._select_default_voice_and_project()

    def _select_default_voice_and_project(self):
        from resemble import Resemble
        # Project: pick the first project, or create one named "voicebench".
        try:
            projects = Resemble.v2.projects.all(1, 50).get("items", [])
        except Exception:
            projects = []
        if not projects:
            res = Resemble.v2.projects.create(
                name="voicebench",
                description="voice-bench experiment runs",
                public=False,
                archived=False,
            )
            self._project_uuid = res["item"]["uuid"]
        else:
            self._project_uuid = projects[0]["uuid"]

        # Voice: pick first ready voice from the account's catalogue.
        try:
            voices = Resemble.v2.voices.all(1, 50).get("items", [])
        except Exception:
            voices = []
        for v in voices:
            if v.get("status") == "ready" or not v.get("status"):
                self._default_voice_uuid = v["uuid"]
                break
        if not self._default_voice_uuid and voices:
            self._default_voice_uuid = voices[0]["uuid"]

    def tts(self, text, voice_id="", model_id=DEFAULT_MODEL_ID, *, seed=None):
        del seed
        self._ensure()
        vid = voice_id or self._default_voice_uuid
        if not vid:
            raise RuntimeError("Resemble: no default voice available; account has no ready voices")
        wav, sr, elapsed = self._synthesize(voice_uuid=vid, text=text)
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
        self._ensure()
        ref_path = Path(reference_wav_path)
        cached = self._clone_voices.get(str(ref_path))
        if cached:
            clone_uuid = cached
        else:
            clone_uuid = self._build_clone_voice(ref_path, reference_text)
            self._clone_voices[str(ref_path)] = clone_uuid

        wav, sr, elapsed = self._synthesize(voice_uuid=clone_uuid, text=text)
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
        if not self._initialised:
            return
        from resemble import Resemble
        for ref, vid in list(self._clone_voices.items()):
            try:
                Resemble.v2.voices.delete(vid)
            except Exception:
                pass
        self._clone_voices.clear()

    # -- internals -------------------------------------------------------------

    def _synthesize(self, *, voice_uuid: str, text: str) -> tuple[np.ndarray, int, float]:
        from resemble import Resemble
        started = time.perf_counter()
        # Sync clip creation -- blocks until WAV is ready, returns URL or bytes.
        try:
            resp = Resemble.v2.clips.create_sync(
                project_uuid=self._project_uuid,
                voice_uuid=voice_uuid,
                body=text,
                title=text[:60],
            )
        except TypeError:
            resp = Resemble.v2.clips.create_sync(self._project_uuid, voice_uuid, text)
        elapsed = time.perf_counter() - started

        item = resp.get("item", resp) if isinstance(resp, dict) else resp
        audio_src = item.get("audio_src") if isinstance(item, dict) else getattr(item, "audio_src", None)
        if not audio_src:
            raise RuntimeError(f"Resemble clip response missing audio_src: {resp!r}")

        import requests, soundfile as sf
        buf = requests.get(audio_src, timeout=60).content
        data, sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(sr), elapsed

    def _build_clone_voice(self, ref_path: Path, reference_text: str | None) -> str:
        """Create voice, upload ref recording, build, poll until ready."""
        from resemble import Resemble
        # Create empty voice.
        res = Resemble.v2.voices.create(
            name=f"voicebench_{ref_path.stem}",
            dataset_url=None,
            consent="I have permission to clone this voice for research benchmark.",
        )
        voice_uuid = res["item"]["uuid"]
        # Upload reference recording.
        ref_text = reference_text or read_normalized_txt_alongside(ref_path) or ref_path.stem
        try:
            with open(ref_path, "rb") as f:
                Resemble.v2.recordings.create(
                    voice_uuid=voice_uuid,
                    file=f,
                    name=ref_path.stem,
                    text=ref_text,
                    is_active=True,
                    emotion="neutral",
                )
        except TypeError:
            with open(ref_path, "rb") as f:
                Resemble.v2.recordings.create(voice_uuid, f, ref_path.stem, ref_text, True, "neutral")

        # Trigger build.
        try:
            Resemble.v2.voices.build(voice_uuid)
        except AttributeError:
            pass  # some versions auto-build on first recording

        # Poll until ready (timeout 5 min).
        deadline = time.time() + 300
        while time.time() < deadline:
            status_resp = Resemble.v2.voices.get(voice_uuid)
            status = (status_resp.get("item", {}) or {}).get("status")
            if status == "ready":
                return voice_uuid
            if status in ("failed", "error"):
                raise RuntimeError(f"Resemble voice {voice_uuid} build failed: {status_resp!r}")
            time.sleep(10)
        raise RuntimeError(f"Resemble voice {voice_uuid} did not become ready within 5 min")
