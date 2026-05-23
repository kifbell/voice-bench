"""Azure Cognitive Services Text-to-Speech provider.

Cloud API, no GPU needed. Requires:
  AZURE_SPEECH_KEY -- subscription key
  AZURE_SPEECH_ENDPOINT -- region endpoint, e.g. https://francecentral.api.cognitive.microsoft.com/

Voice cloning ("Custom Neural Voice") needs a separately-trained voice deployment, which we
don't have. The clone() method here just falls back to the default voice; if you ever
provision a CNV deployment, set AZURE_CUSTOM_VOICE_NAME and clone() will use it.

Output: Azure returns 24 kHz mono PCM-16 with our requested format. Free tier: 500K chars/mo
for Standard voices, 0.5M chars/mo for Neural -- plenty for an 800-file pilot.
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


DEFAULT_MODEL_ID = "azure-neural"
DEFAULT_VOICE_ID = "en-US-AvaMultilingualNeural"


class AzureTtsProvider:
    name = "azure_tts"
    supports_cloning = False

    def __init__(
        self,
        *,
        speech_key: str | None = None,
        endpoint: str | None = None,
    ):
        self._speech_key = speech_key or os.environ.get("AZURE_SPEECH_KEY")
        self._endpoint = endpoint or os.environ.get("AZURE_SPEECH_ENDPOINT")
        if not self._speech_key or not self._endpoint:
            raise RuntimeError(
                "AzureTtsProvider needs AZURE_SPEECH_KEY + AZURE_SPEECH_ENDPOINT "
                "(or pass via constructor)."
            )
        # Azure region is parsed from the endpoint: https://<region>.api.cognitive.microsoft.com
        host = self._endpoint.replace("https://", "").replace("http://", "")
        self._region = host.split(".")[0]
        self._config = None

    def tts(self, text, voice_id=DEFAULT_VOICE_ID, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del seed  # Azure has no seed control.
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
        # Without a Custom Neural Voice deployment, treat clone() as TTS with default voice.
        # Mark task=cloning in sidecars so downstream metric joins still find it.
        custom_voice = os.environ.get("AZURE_CUSTOM_VOICE_NAME")
        voice_id = custom_voice or DEFAULT_VOICE_ID
        wav, elapsed = self._synthesize(text=text, voice_id=voice_id)
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="cloning",
            model_id=model_id,
            voice_id=voice_id,
            character_count=len(text),
            seed=None,
            reference_wav_path=None,
        )

    def cleanup(self):
        self._config = None

    def _ensure_config(self):
        if self._config is not None:
            return
        import azure.cognitiveservices.speech as speechsdk
        self._config = speechsdk.SpeechConfig(subscription=self._speech_key, region=self._region)
        # Request 24 kHz mono PCM raw bytes so the conversion to our canonical 24k int16 PCM
        # is just a reshape -- no resampling, no MP3 decode.
        self._config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )

    def _synthesize(self, *, text, voice_id):
        self._ensure_config()
        import azure.cognitiveservices.speech as speechsdk
        self._config.speech_synthesis_voice_name = voice_id

        # PullAudioOutputStream lets us read raw bytes back without writing a file.
        stream = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=stream)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._config, audio_config=audio_config
        )

        started = time.perf_counter()
        result = synthesizer.speak_text_async(text).get()
        elapsed = time.perf_counter() - started

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancel = getattr(result, "cancellation_details", None)
            err = f"reason={result.reason}"
            if cancel is not None:
                err += f"; error={cancel.error_details}"
            raise RuntimeError(f"Azure TTS failed: {err}")

        # The Azure SDK already buffered all bytes in `result.audio_data` (raw 24 kHz int16).
        pcm = np.frombuffer(result.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        wav = resample_to_canonical(pcm, SAMPLE_RATE_CANONICAL)
        return wav, elapsed
