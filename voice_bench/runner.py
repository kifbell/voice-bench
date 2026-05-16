import json
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from voice_bench.providers.base import GenerationResult


@dataclass(frozen=True)
class GenerationContext:
    utt_id: str
    speaker_id: str
    text: str


def _audio_path(out_root: Path, provider: str, task: str, utt_id: str) -> Path:
    return out_root / "audio" / provider / task / f"{utt_id}.wav"


def _sidecar_path(out_root: Path, provider: str, task: str, utt_id: str) -> Path:
    return out_root / "sidecars" / provider / task / f"{utt_id}.json"


def is_done(out_root: Path, provider: str, task: str, utt_id: str) -> bool:
    return _audio_path(out_root, provider, task, utt_id).exists() and \
        _sidecar_path(out_root, provider, task, utt_id).exists()


def _write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int, sample_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def save_generation(
    result: GenerationResult,
    ctx: GenerationContext,
    out_root: Path,
) -> dict:
    wav_path = _audio_path(out_root, result.provider, result.task, ctx.utt_id)
    json_path = _sidecar_path(out_root, result.provider, result.task, ctx.utt_id)

    _write_wav(wav_path, result.audio_pcm, result.sample_rate, result.channels, result.sample_width)

    record = {
        "utt_id": ctx.utt_id,
        "speaker_id": ctx.speaker_id,
        "text": ctx.text,
        "provider": result.provider,
        "task": result.task,
        "model_id": result.model_id,
        "voice_id": result.voice_id,
        "character_count": result.character_count,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "sample_width": result.sample_width,
        "latency_seconds": result.latency_seconds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": result.seed,
        "reference_wav_path": result.reference_wav_path,
        "wav_path": str(wav_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2))
    return record


def load_sidecar(out_root: Path, provider: str, task: str, utt_id: str) -> dict:
    return json.loads(_sidecar_path(out_root, provider, task, utt_id).read_text())
