"""UTMOSv2 naturalness predictor.

Single-fold (fold=0) checkpoint, ~781 MB. Predicts MOS in [1, 5] range.
On M1 MPS, ~1.3s/clip steady-state after warmup.

Forced to CPU regardless of VOICEBENCH_DEVICE: UTMOSv2 on torch>=2.9 / CUDA 13
silently returns NaN for every prediction (no exception, just non-finite
output). CPU inference is still fast enough (~1-2 s per clip on a modern x86
core, several minutes for a thousand-clip run) and the alternative is putting
NaN into every utmos row. Revisit when UTMOSv2 ships a torch-2.9-compatible
build.
"""
import functools
from pathlib import Path


_DEVICE = "cpu"


@functools.lru_cache(maxsize=1)
def _model():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    import utmosv2
    return utmosv2.create_model(pretrained=True, fold=0, device=_DEVICE)


def score(wav_path: Path | str) -> float:
    import numpy as np
    pred = _model().predict(input_path=str(wav_path), device=_DEVICE, verbose=False)
    pred_arr = np.asarray(pred).ravel()
    if pred_arr.size == 0 or not np.isfinite(pred_arr[0]):
        raise RuntimeError(f"UTMOSv2 returned non-finite prediction: {pred!r}")
    return float(pred_arr[0])
