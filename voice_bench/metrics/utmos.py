"""UTMOSv2 naturalness predictor.

Single-fold (fold=0) checkpoint, ~781 MB. Predicts MOS in [1, 5] range.
On M1 MPS, ~1.3s/clip steady-state after warmup. On a CUDA box use
``VOICEBENCH_DEVICE=cuda:0`` -- UTMOSv2 parses ``"<backend>:<index>"`` so a
bare ``"cuda"`` warns ``Could not parse CUDA device string 'cuda'`` and falls
back to device 0 internally; explicit ``cuda:0`` keeps the logs clean.
"""
import functools
import os
from pathlib import Path


# Normalize the device string: UTMOSv2 expects 'cuda:N', not bare 'cuda'.
_RAW_DEVICE = os.environ.get("VOICEBENCH_DEVICE", "cpu")
_DEVICE = "cuda:0" if _RAW_DEVICE == "cuda" else _RAW_DEVICE


@functools.lru_cache(maxsize=1)
def _model():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import utmosv2
    return utmosv2.create_model(pretrained=True, fold=0, device=_DEVICE)


def score(wav_path: Path | str) -> float:
    pred = _model().predict(input_path=str(wav_path), device=_DEVICE, verbose=False)
    # UTMOSv2 returns either a scalar or a numpy 0-d / 1-element array depending on version.
    import numpy as np
    pred_arr = np.asarray(pred).ravel()
    if pred_arr.size == 0 or not np.isfinite(pred_arr[0]):
        raise RuntimeError(f"UTMOSv2 returned non-finite prediction: {pred!r}")
    return float(pred_arr[0])
