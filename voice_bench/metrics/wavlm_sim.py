"""Speaker similarity via microsoft/wavlm-base-plus-sv x-vector head.

Use AutoModelForAudioXVector (not AutoModel) — the x-vector classification head
produces the 512-d speaker embedding. Cosine in [-1, 1]; cloned voices score
≈0.4–0.85, cross-speaker ≈0.0–0.4 (calibrate on GT pairs).
"""
import functools
import os

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


_DEVICE = os.environ.get("VOICEBENCH_DEVICE", "mps")


@functools.lru_cache(maxsize=1)
def _model():
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    model = AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().to(_DEVICE)
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    return model, fe


@torch.no_grad()
def embed(audio_16k: np.ndarray) -> np.ndarray:
    model, fe = _model()
    inputs = fe(audio_16k, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    return model(**inputs).embeddings.squeeze(0).cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0, 0])
