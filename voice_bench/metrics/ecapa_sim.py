"""Speaker similarity via ECAPA-TDNN (VoxCeleb-trained, SpeechBrain).

Older but most-cited baseline for speaker verification; used here as a second
independent measure of speaker identity (H2 in the plan: do similarity metrics
agree or diverge?).
"""
import functools
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


_DEVICE = os.environ.get("VOICEBENCH_DEVICE", "cpu")


@functools.lru_cache(maxsize=1)
def _model():
    from speechbrain.inference.speaker import EncoderClassifier
    cache_dir = Path(".cache/speechbrain/spkrec-ecapa-voxceleb")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache_dir),
        run_opts={"device": _DEVICE},
    )


@torch.no_grad()
def embed(audio_16k: np.ndarray) -> np.ndarray:
    sig = torch.from_numpy(audio_16k).unsqueeze(0).to(_DEVICE)
    emb = _model().encode_batch(sig)
    return emb.squeeze().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0, 0])
