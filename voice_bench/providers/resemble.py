"""Resemble AI Rapid Voice Clone provider — cloning ONLY.

Uses the Rapid Voice Clone flow (voice_type='rapid'):
  1. Upload reference WAV to a public URL (catbox.moe -- anonymous, no API key).
  2. POST voices.create(name, dataset_url=<url>, voice_type='rapid', consent=...).
  3. Poll voices.get(uuid) until voice_status == 'Ready' (~30-60 s).
  4. Synthesise via clips.create_sync(voice_uuid, project_uuid, body=text).
  5. Cleanup deletes the voice to free the slot.

Free Flex tier gives 1 voice clone slot; additional are $2/voice/mo. Our caller
limits speakers via --max-speakers since slot budget is the bottleneck.

Note: Resemble's `voices.create` returns nested under `.item` on success, but
on failure (e.g. no slots) returns `{success: false, message: ...}` -- handle both.
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


DEFAULT_MODEL_ID = "resemble-rapid"
# We need a public HTTP URL Resemble can pull from. Catbox started returning 0-byte
# files when used repeatedly; tmpfiles.org is more reliable for short-lived uploads.
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"


class ResembleProvider:
    name = "resemble"
    supports_cloning = True

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
        if not self._api_key:
            raise RuntimeError("ResembleProvider needs RESEMBLE_API_KEY")
        self._initialised = False
        self._project_uuid: str | None = None
        self._clone_voices: dict[str, str] = {}  # ref_path -> voice_uuid

    def _ensure(self):
        if self._initialised:
            return
        from resemble import Resemble
        Resemble.api_key(self._api_key)
        self._initialised = True
        self._select_or_create_project()

    def _select_or_create_project(self):
        from resemble import Resemble
        try:
            projects = Resemble.v2.projects.all(1, 50).get("items", [])
        except Exception:
            projects = []
        if projects:
            self._project_uuid = projects[0]["uuid"]
            return
        # No project yet -- create one.
        res = Resemble.v2.projects.create(
            name="voicebench",
            description="voice-bench benchmark project",
            public=False,
            archived=False,
        )
        self._project_uuid = res["item"]["uuid"]

    def tts(self, text, voice_id, model_id=DEFAULT_MODEL_ID, *, seed=None):
        raise NotImplementedError(
            "ResembleProvider is cloning-only; run with --tasks cloning."
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        del seed
        self._ensure()
        ref_path = Path(reference_wav_path)
        clone_uuid = self._clone_voices.get(str(ref_path))
        if not clone_uuid:
            clone_uuid = self._create_rapid_clone(ref_path)
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

    def _upload_public(self, ref_path: Path) -> str:
        """Anonymous upload to tmpfiles.org; returns direct-download public URL.

        tmpfiles wraps the upload in a viewer page; the API URL needs a /dl/ injection
        to become a direct binary download (so Resemble can curl it).
        """
        with open(ref_path, "rb") as f:
            files = {"file": (ref_path.name, f, "audio/wav")}
            r = requests.post(TMPFILES_UPLOAD_URL, files=files, timeout=120)
        r.raise_for_status()
        viewer_url = r.json()["data"]["url"]
        direct = viewer_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        return direct

    def _create_rapid_clone(self, ref_path: Path) -> str:
        """Create Resemble rapid voice clone with dataset_url, explicitly trigger build, poll.

        Per https://docs.resemble.ai/voice-creation/voices/clone-overview :
        Rapid clones DO NOT auto-train from a dataset_url -- you must call build()
        after voices.create() to start training.

        Training takes <1 minute per docs; status reaches 'finished' when ready.
        """
        from resemble import Resemble
        dataset_url = self._upload_public(ref_path)
        res = Resemble.v2.voices.create(
            name=f"vb_{ref_path.stem}"[:30],
            dataset_url=dataset_url,
            voice_type="rapid",
            consent="I have the legal right to clone this voice for academic research benchmarking.",
        )
        if not res.get("success"):
            raise RuntimeError(f"Resemble voices.create failed: {res!r}")
        voice_uuid = res["item"]["uuid"]

        # Explicitly trigger training. Docs say rapid clones need this call even when
        # dataset_url is provided; only professional voices auto-train.
        build_res = Resemble.v2.voices.build(voice_uuid)
        if isinstance(build_res, dict) and build_res.get("success") is False:
            raise RuntimeError(f"Resemble voices.build({voice_uuid}) failed: {build_res!r}")

        # Poll until status reaches 'finished' (or 'Ready'). Docs claim <1 min for rapid.
        deadline = time.time() + 600
        last_status = None
        while time.time() < deadline:
            status_resp = Resemble.v2.voices.get(voice_uuid)
            item = status_resp.get("item") or {}
            # Two possible fields per API version: `status` and `voice_status`.
            status = item.get("status") or item.get("voice_status")
            if status != last_status:
                last_status = status
            if status in ("finished", "Ready"):
                return voice_uuid
            if status in ("failed", "error", "Failed", "Error"):
                raise RuntimeError(f"Resemble voice {voice_uuid} build failed: {status_resp!r}")
            time.sleep(5)
        raise RuntimeError(
            f"Resemble voice {voice_uuid} not ready within 10 min; last status={last_status}"
        )

    def _synthesize(self, *, voice_uuid: str, text: str) -> tuple[np.ndarray, int, float]:
        """Use clips.create_direct (Resemble's synthesis cluster) -- not clips.create_sync.

        create_sync (app.resemble.ai/api/v2/projects/{uuid}/clips) returns
        {'success': False, 'message': 'Synthesis failed!'} for our account.
        create_direct (f.cluster.resemble.ai/synthesize) actually returns the audio
        as base64-encoded WAV in audio_content. Both endpoints exist in the SDK.
        """
        from resemble import Resemble
        import base64
        started = time.perf_counter()
        resp = Resemble.v2.clips.create_direct(
            project_uuid=self._project_uuid,
            voice_uuid=voice_uuid,
            data=text,
        )
        elapsed = time.perf_counter() - started
        if not resp.get("success"):
            raise RuntimeError(f"Resemble clips.create_direct failed: {resp!r}")
        audio_b64 = resp.get("audio_content")
        if not audio_b64:
            raise RuntimeError(f"Resemble create_direct missing audio_content: {resp!r}")
        wav_bytes = base64.b64decode(audio_b64)
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(sr), elapsed
