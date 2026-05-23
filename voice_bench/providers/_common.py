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


def patch_torchaudio_legacy_apis():
    """Restore torchaudio APIs removed in 2.9+ that older fish-speech depends on.

    fish-speech 3c7cd3f0 calls torchaudio.list_audio_backends() which was dropped.
    Stub it to ['soundfile'] so the reference_loader.py backend selection still works.
    """
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


def patch_torchaudio_load_to_soundfile():
    """Replace torchaudio.load + torchaudio._torchcodec.load_with_torchcodec with soundfile.

    Octopus worker pods don't ship ffmpeg; torchcodec dlopens libavutil at runtime
    and fails. Patch the public torchaudio.load AND the underlying load_with_torchcodec
    helper that torchaudio>=2.9 dispatches to internally, in case some callers reach
    it directly. soundfile handles both path strings and file-like (BytesIO) objects.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _sf_load(uri, *args, **kwargs):
        del args, kwargs
        target = uri if hasattr(uri, "read") else str(uri)
        data, sr = sf.read(target, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)

    torchaudio.load = _sf_load
    # Also patch the internal torchcodec dispatcher; some callers (torchaudio>=2.9)
    # bypass torchaudio.load and go straight here.
    try:
        import torchaudio._torchcodec as _tc
        _tc.load_with_torchcodec = _sf_load
    except ImportError:
        pass
