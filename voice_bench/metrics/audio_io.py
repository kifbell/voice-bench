from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 16000


def load_16k(wav_path: Path | str) -> np.ndarray:
    """Load audio as 16 kHz mono float32 — universal input for judge models."""
    y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    return y.astype(np.float32)
