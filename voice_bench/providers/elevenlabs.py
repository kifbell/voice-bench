import time
from pathlib import Path

from elevenlabs.client import ElevenLabs

from voice_bench.providers.base import GenerationResult


class ElevenLabsProvider:
    name = "elevenlabs"
    supports_cloning = True
    DEFAULT_MODEL = "eleven_multilingual_v2"

    def __init__(self, api_key: str) -> None:
        self._client = ElevenLabs(api_key=api_key)
        self._clone_voice_cache: dict[str, str] = {}

    def tts(
        self,
        text: str,
        voice_id: str,
        model_id: str = DEFAULT_MODEL,
        *,
        seed: int | None = None,
    ) -> GenerationResult:
        return self._synthesize(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            seed=seed,
            task="tts",
            reference_wav_path=None,
        )

    def clone(
        self,
        text: str,
        reference_wav_path: Path,
        model_id: str = DEFAULT_MODEL,
        *,
        seed: int | None = None,
        reference_text: str | None = None,
    ) -> GenerationResult:
        del reference_text  # ElevenLabs IVC does not need the reference transcript.
        ref_key = str(Path(reference_wav_path).resolve())
        voice_id = self._get_or_create_clone_voice(ref_key, reference_wav_path)
        return self._synthesize(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            seed=seed,
            task="cloning",
            reference_wav_path=ref_key,
        )

    def cleanup_clone_voice(self, reference_wav_path: Path | str) -> bool:
        ref_key = str(Path(reference_wav_path).resolve())
        voice_id = self._clone_voice_cache.pop(ref_key, None)
        if voice_id is None:
            return False
        try:
            self._client.voices.delete(voice_id=voice_id)
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        for ref_key in list(self._clone_voice_cache.keys()):
            self.cleanup_clone_voice(ref_key)

    def list_voice_ids(self, limit: int = 10) -> list[tuple[str, str]]:
        response = self._client.voices.get_all()
        return [(v.voice_id, v.name) for v in response.voices[:limit]]

    def _synthesize(
        self,
        *,
        text: str,
        voice_id: str,
        model_id: str,
        seed: int | None,
        task: str,
        reference_wav_path: str | None,
    ) -> GenerationResult:
        started = time.perf_counter()
        audio_iter = self._client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format="pcm_24000",
            seed=seed,
        )
        audio_pcm = b"".join(audio_iter)
        elapsed = time.perf_counter() - started
        return GenerationResult(
            audio_pcm=audio_pcm,
            sample_rate=24000,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task=task,
            model_id=model_id,
            voice_id=voice_id,
            character_count=len(text),
            seed=seed,
            reference_wav_path=reference_wav_path,
        )

    def _get_or_create_clone_voice(self, ref_key: str, ref_path: Path) -> str:
        if ref_key in self._clone_voice_cache:
            return self._clone_voice_cache[ref_key]
        path = Path(ref_path)
        with path.open("rb") as fh:
            voice = self._client.voices.ivc.create(
                name=f"voicebench_{path.stem}",
                files=[(path.name, fh, "audio/wav")],
                description="voicebench experiment — ephemeral, will be deleted",
            )
        self._clone_voice_cache[ref_key] = voice.voice_id
        return voice.voice_id
