"""WER round-trip: synthesised audio → faster-whisper → compare to target text.

Whisper-medium chosen to match TTSDS-style baseline. On CPU we use int8 to keep
inference near real-time; on GPU we switch to float16 which is faster and avoids
the int8 quantization roundtrip overhead.
"""
import functools
import os
from pathlib import Path

import jiwer


_DEVICE = os.environ.get("VOICEBENCH_DEVICE", "cpu")
_COMPUTE_TYPE = os.environ.get(
    "VOICEBENCH_WHISPER_COMPUTE_TYPE",
    "float16" if _DEVICE.startswith("cuda") else "int8",
)


@functools.lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    # faster-whisper accepts "cuda" or "cpu" (no index parsing).
    device = "cuda" if _DEVICE.startswith("cuda") else "cpu"
    return WhisperModel("medium", device=device, compute_type=_COMPUTE_TYPE)


def transcribe(wav_path: Path | str, language: str = "en") -> str:
    segments, _ = _model().transcribe(
        str(wav_path),
        language=language,
        beam_size=1,
    )
    return " ".join(s.text for s in segments).strip()


_NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def wer(reference_text: str, hypothesis_text: str) -> float:
    return float(jiwer.wer(
        reference_text,
        hypothesis_text,
        reference_transform=_NORMALISE,
        hypothesis_transform=_NORMALISE,
    ))
