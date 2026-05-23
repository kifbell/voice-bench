"""Helpers shared by OS-model providers."""
import numpy as np


SAMPLE_RATE_CANONICAL = 24000  # GenerationResult.audio_pcm contract


def float_to_pcm16_bytes(wav):
    """Clip a float waveform in [-1, 1] and serialize to little-endian int16 PCM bytes."""
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    return pcm.tobytes()


def resample_to_canonical(wav, src_sr):
    """Resample a 1-D numpy waveform to SAMPLE_RATE_CANONICAL (24 kHz). No-op if already 24k."""
    if src_sr == SAMPLE_RATE_CANONICAL:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=src_sr, target_sr=SAMPLE_RATE_CANONICAL)


def read_normalized_txt_alongside(wav_path):
    """LibriTTS convention: <utt_id>.normalized.txt next to <utt_id>.wav."""
    from pathlib import Path
    p = Path(wav_path)
    candidate = p.with_suffix("").with_suffix(".normalized.txt")
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None
