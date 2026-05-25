"""NISQA — multi-dimensional speech quality predictor.

Predicts MOS (1-5) plus 4 sub-scores (noisiness, coloration, discontinuity,
loudness). Trained on the NISQA-Corpus and validated against POLQA. From the
research plan, this is the second naturalness predictor (alongside UTMOSv2)
used to verify H2: do naturalness metrics agree?

Repo: https://github.com/gabrielmittag/NISQA
Weights (~100 MB) auto-downloaded to ~/.cache/nisqa/ on first use, or to
$NISQA_WEIGHTS_DIR if set.
"""
import functools
import os
from pathlib import Path


_DEVICE = os.environ.get("VOICEBENCH_DEVICE", "cpu")


_MODEL_ARGS_BASE = {
    "mode": "predict_file",
    "pretrained_model": None,  # filled in _model()
    "deg": None,  # per-call
    "data_dir": None,
    "output_dir": None,
    "csv_file": None,
    "csv_deg": None,
    "num_workers": 0,
    "bs": 1,
    "ms_channel": None,
    "ms_sr": None,
    "tr_bs_val": 1,
    "tr_num_workers": 0,
}


@functools.lru_cache(maxsize=1)
def _model():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from nisqa.NISQA_model import nisqaModel

    weights_dir = Path(os.environ.get("NISQA_WEIGHTS_DIR", Path.home() / ".cache" / "nisqa"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = weights_dir / "nisqa.tar"
    if not weights.exists():
        import urllib.request
        url = "https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar"
        urllib.request.urlretrieve(url, str(weights))

    args = dict(_MODEL_ARGS_BASE)
    args["pretrained_model"] = str(weights)
    return nisqaModel(args), args


def score(wav_path: Path | str) -> dict:
    """Return dict with nisqa_mos/noi/col/dis/loud (1-5)."""
    model, args = _model()
    args["deg"] = str(wav_path)
    df = model.predict()
    row = df.iloc[0]
    return {
        "nisqa_mos": float(row["mos_pred"]),
        "nisqa_noi": float(row["noi_pred"]),
        "nisqa_col": float(row["col_pred"]),
        "nisqa_dis": float(row["dis_pred"]),
        "nisqa_loud": float(row["loud_pred"]),
    }
