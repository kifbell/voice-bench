"""WER round-trip: synthesised audio → faster-whisper → compare to target text.

Whisper-medium chosen to match TTSDS-style baseline. int8 quantisation keeps CPU
inference near real-time on Apple Silicon.
"""
import functools
from pathlib import Path

import jiwer


@functools.lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel("medium", device="cpu", compute_type="int8")


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
