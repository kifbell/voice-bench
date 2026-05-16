from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    audio_pcm: bytes
    sample_rate: int
    channels: int
    sample_width: int
    latency_seconds: float
    provider: str
    task: str  # "tts" | "cloning"
    model_id: str
    voice_id: str
    character_count: int
    seed: int | None
    reference_wav_path: str | None  # cloning only


class Provider(Protocol):
    name: str
    supports_cloning: bool

    def tts(
        self,
        text: str,
        voice_id: str,
        model_id: str,
        *,
        seed: int | None = None,
    ) -> GenerationResult: ...

    def clone(
        self,
        text: str,
        reference_wav_path: Path,
        model_id: str,
        *,
        seed: int | None = None,
    ) -> GenerationResult: ...

    def cleanup(self) -> None: ...
