"""UTMOSv2 naturalness predictor.

Single-fold (fold=0) checkpoint, ~781 MB. Predicts MOS in [1, 5] range.
On M1 MPS, ~1.3s/clip steady-state after warmup.
"""
import functools
import os
from pathlib import Path


_DEVICE = os.environ.get("VOICEBENCH_DEVICE", "mps")


@functools.lru_cache(maxsize=1)
def _model():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import utmosv2
    return utmosv2.create_model(pretrained=True, fold=0, device=_DEVICE)


def score(wav_path: Path | str) -> float:
    pred = _model().predict(input_path=str(wav_path), device=_DEVICE, verbose=False)
    return float(pred)
