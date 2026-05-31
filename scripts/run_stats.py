"""Statistical analysis on metrics.parquet.

Outputs results/stats.json with: bootstrap means+CIs per (provider, task, metric),
Spearman ρ between metrics (utterance-level, system-level), Wilcoxon paired tests
between providers, and Pareto-frontier membership.

Gracefully handles degenerate cases (single provider, single task) for pilot
validation. Skipped tests are recorded in `stats.json['skipped']`.
"""
import argparse
import itertools
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, spearmanr, wilcoxon


METRICS_NATURALNESS = ["utmos"]      # add 'nisqa_mos' when NISQA is plugged in
METRICS_INTELLIG = ["whisper_wer"]
METRICS_SIM = ["wavlm_sim", "ecapa_sim"]
ALL_METRICS = METRICS_NATURALNESS + METRICS_INTELLIG + METRICS_SIM


def _bootstrap_mean(values: np.ndarray, n_resamples: int = 1000) -> dict:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return {"mean": float(values.mean()) if len(values) else math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": int(len(values))}
    res = bootstrap((values,), np.mean, n_resamples=n_resamples, confidence_level=0.95, method="percentile")
    return {
        "mean": float(values.mean()),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n": int(len(values)),
    }


def per_provider_summary(df: pd.DataFrame) -> dict:
    out = {}
    for (prov, task), g in df.groupby(["provider", "task"]):
        out[f"{prov}|{task}"] = {
            m: _bootstrap_mean(g[m].values) if m in g.columns else None
            for m in ALL_METRICS
        }
    return out


def metric_correlations(df: pd.DataFrame) -> dict:
    """Utterance-level Spearman ρ between metrics (within each task)."""
    out = {}
    for task, g in df.groupby("task"):
        out[task] = {}
        for m1, m2 in itertools.combinations(ALL_METRICS, 2):
            if m1 not in g.columns or m2 not in g.columns:
                continue
            x = g[[m1, m2]].dropna()
            if len(x) < 3:
                out[task][f"{m1}__{m2}"] = {"rho": math.nan, "p": math.nan, "n": int(len(x))}
                continue
            rho, p = spearmanr(x[m1], x[m2])
            out[task][f"{m1}__{m2}"] = {"rho": float(rho), "p": float(p), "n": int(len(x))}
    return out


def pairwise_wilcoxon(df: pd.DataFrame, metric: str) -> list[dict]:
    out = []
    providers = sorted(df.provider.unique())
    if len(providers) < 2:
        return out
    for task, g in df.groupby("task"):
        pivot = g.pivot_table(index="utt_id", columns="provider", values=metric)
        for p1, p2 in itertools.combinations(providers, 2):
            if p1 not in pivot.columns or p2 not in pivot.columns:
                continue
            paired = pivot[[p1, p2]].dropna()
            if len(paired) < 5:
                out.append({"task": task, "metric": metric, "p1": p1, "p2": p2, "n": int(len(paired)), "stat": math.nan, "p": math.nan})
                continue
            stat, pval = wilcoxon(paired[p1], paired[p2])
            out.append({
                "task": task, "metric": metric, "p1": p1, "p2": p2,
                "n": int(len(paired)), "stat": float(stat), "p": float(pval),
                "mean_diff": float((paired[p1] - paired[p2]).mean()),
            })
    return out


def pareto_membership(df: pd.DataFrame, axes: list[str], maximise: list[bool]) -> dict:
    """For each task, return providers on Pareto-frontier in given axes."""
    out = {}
    for task, g in df.groupby("task"):
        means = g.groupby("provider")[axes].mean().dropna(how="any")
        if len(means) < 2:
            out[task] = {"frontier": list(means.index), "note": "degenerate (≤1 provider)"}
            continue
        signed = means.values * np.where(maximise, 1, -1)
        frontier = []
        for i, prov in enumerate(means.index):
            dominated = False
            for j in range(len(signed)):
                if i == j:
                    continue
                if np.all(signed[j] >= signed[i]) and np.any(signed[j] > signed[i]):
                    dominated = True
                    break
            if not dominated:
                frontier.append(prov)
        out[task] = {"frontier": frontier, "axes": axes, "maximise": maximise, "means": means.to_dict()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.parquet")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--out", default="results/stats.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    df = pd.read_parquet(root / args.metrics)

    providers = sorted(df.provider.unique())
    tasks = sorted(df.task.unique())
    print(f"Loaded {len(df)} rows: providers={providers}, tasks={tasks}")

    stats: dict = {
        "n_rows": int(len(df)),
        "providers": providers,
        "tasks": tasks,
        "skipped": [],
    }
    if Path(root / args.calibration).exists():
        stats["calibration"] = json.loads((root / args.calibration).read_text())

    stats["per_provider"] = per_provider_summary(df)
    stats["metric_correlations"] = metric_correlations(df)

    pairwise = []
    for metric in ALL_METRICS:
        if metric not in df.columns:
            continue
        pairwise.extend(pairwise_wilcoxon(df, metric))
    if not pairwise:
        stats["skipped"].append("pairwise_wilcoxon: needs ≥2 providers")
    stats["pairwise_wilcoxon"] = pairwise

    pareto = {}
    if "utmos" in df.columns and "wavlm_sim" in df.columns:
        pareto["utmos_vs_wavlm"] = pareto_membership(df, ["utmos", "wavlm_sim"], [True, True])
    if "utmos" in df.columns and "ecapa_sim" in df.columns:
        pareto["utmos_vs_ecapa"] = pareto_membership(df, ["utmos", "ecapa_sim"], [True, True])
    stats["pareto"] = pareto

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")
    print(f"  Skipped: {stats['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
