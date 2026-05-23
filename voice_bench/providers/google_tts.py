"""Google Cloud Text-to-Speech provider.

Cloud API, no GPU needed. Authenticates via OAuth user credentials JSON pointed to by
GOOGLE_APPLICATION_CREDENTIALS env var (standard gcloud convention).

Voice cloning: Google's Custom Voice (Instant Custom Voice) is in restricted preview and
needs separate per-project allowlisting; we don't have access. clone() here just falls
back to the standard neural voice.

Free tier: 1M chars/mo Standard, 200K chars/mo WaveNet/Neural2 -- enough for 800-file pilot
even on Neural2.

Output: Google returns LINEAR16 (signed 16-bit PCM). We request 24 kHz directly.
"""
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


DEFAULT_MODEL_ID = "google-neural2"
DEFAULT_VOICE_ID = "en-US-Neural2-F"
DEFAULT_LANG = "en-US"


class GoogleTtsProvider:
    name = "google_tts"
    supports_cloning = False

    def __init__(self):
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError(
                "GoogleTtsProvider needs GOOGLE_APPLICATION_CREDENTIALS env var "
                "pointing to a credentials JSON."
            )
        self._client = None

    def tts(self, text, voice_id=DEFAULT_VOICE_ID, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del seed
        wav, elapsed = self._synthesize(text=text, voice_id=voice_id)
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
        # No custom voice access -- fall back to default neural voice. Sidecar still tags
        # task=cloning so downstream joins line up with other providers.
        wav, elapsed = self._synthesize(text=text, voice_id=DEFAULT_VOICE_ID)
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
        from google.cloud import texttospeech
        self._client = texttospeech.TextToSpeechClient()

    def _synthesize(self, *, text, voice_id):
        self._ensure_client()
        from google.cloud import texttospeech
        input_msg = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=DEFAULT_LANG, name=voice_id
        )
        audio_cfg = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE_CANONICAL,
        )
        started = time.perf_counter()
        resp = self._client.synthesize_speech(
            input=input_msg, voice=voice, audio_config=audio_cfg
        )
        elapsed = time.perf_counter() - started
        # Google returns a WAV header + LINEAR16 PCM. Skip the 44-byte WAV header. Some
        # SDK versions return raw PCM if format is forced; soundfile handles both.
        import io
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(resp.audio_content), dtype="float32", always_2d=True)
        wav = data.mean(axis=1).astype(np.float32)
        wav = resample_to_canonical(wav, sr)
        return wav, elapsed
