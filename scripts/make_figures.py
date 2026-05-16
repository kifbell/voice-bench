"""Render benchmark figures from metrics.parquet + calibration.json + stats.json.

Designed to degrade gracefully on degenerate data (1 provider / 1 task).
Each figure is wrapped in try/except so partial output is preserved.

Produces:
  fig1_summary_table.png    — per (provider, task) mean ± 95% CI for each metric
  fig2_calibration.png      — similarity per provider vs same/cross GT anchors
  fig3_metric_scatter.png   — utterance-level metric pairs (intra-task)
  fig4_wer_distribution.png — WER histogram per (provider, task)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")


def fig_summary_table(df: pd.DataFrame, stats: dict, out: Path) -> None:
    rows = []
    for prov_task, metrics in stats["per_provider"].items():
        prov, task = prov_task.split("|")
        row = {"provider": prov, "task": task}
        for m, v in metrics.items():
            if v is None or np.isnan(v.get("mean", np.nan)):
                row[m] = "—"
            else:
                row[m] = f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] (n={v['n']})"
        rows.append(row)
    table_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, 1 + len(table_df.columns)), 1 + 0.5 * len(table_df)))
    ax.axis("off")
    t = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    plt.title("Provider summary — mean [95% bootstrap CI]", pad=10)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def fig_calibration(df: pd.DataFrame, calibration: dict, out: Path) -> None:
    sim_metrics = [("wavlm_sim", "wavlm"), ("ecapa_sim", "ecapa")]
    n_axes = sum(1 for m, _ in sim_metrics if m in df.columns)
    if n_axes == 0:
        return
    fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 4.5), squeeze=False)
    axes = axes[0]
    ai = 0
    for metric_col, cal_key in sim_metrics:
        if metric_col not in df.columns:
            continue
        ax = axes[ai]
        ai += 1
        # Filter only cloning rows where similarity exists
        sub = df[(df.task == "cloning") & df[metric_col].notna()]
        if sub.empty:
            ax.set_title(f"{metric_col} — no cloning rows yet")
            continue
        sns.stripplot(data=sub, x="provider", y=metric_col, ax=ax, size=8, alpha=0.7, jitter=0.15)
        cal = calibration.get(cal_key, {})
        if cal:
            ax.axhline(cal["same_mean"], color="green", linestyle="--", linewidth=1,
                       label=f"same-speaker mean ({cal['same_mean']:.2f})")
            ax.axhline(cal["cross_mean"], color="red", linestyle="--", linewidth=1,
                       label=f"cross-speaker mean ({cal['cross_mean']:.2f})")
            ax.axhspan(cal["cross_p95"], cal["same_p05"], color="grey", alpha=0.1,
                       label="ambiguous band")
            ax.legend(fontsize=8, loc="lower left")
        ax.set_ylabel(f"cosine ({metric_col})")
        ax.set_title(f"{cal_key.upper()} similarity per provider")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def fig_metric_scatter(df: pd.DataFrame, out: Path) -> None:
    """Per-task: pairwise scatter of available numeric metrics."""
    metrics = [m for m in ("utmos", "whisper_wer", "wavlm_sim", "ecapa_sim") if m in df.columns]
    if len(metrics) < 2:
        return
    tasks = sorted(df.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), squeeze=False)
    for ti, task in enumerate(tasks):
        ax = axes[0, ti]
        sub = df[df.task == task]
        if "utmos" in sub.columns and "ecapa_sim" in sub.columns:
            x_col, y_col = "utmos", "ecapa_sim"
        elif "utmos" in sub.columns and "wavlm_sim" in sub.columns:
            x_col, y_col = "utmos", "wavlm_sim"
        elif "utmos" in sub.columns and "whisper_wer" in sub.columns:
            x_col, y_col = "utmos", "whisper_wer"
        else:
            x_col, y_col = metrics[0], metrics[1]
        valid = sub[[x_col, y_col, "provider", "utt_id"]].dropna()
        if valid.empty:
            ax.set_title(f"{task}: no data for {x_col} × {y_col}")
            continue
        sns.scatterplot(data=valid, x=x_col, y=y_col, hue="provider", s=80, alpha=0.7, ax=ax)
        ax.set_title(f"{task}: {x_col} vs {y_col} (n={len(valid)})")
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def fig_wer_distribution(df: pd.DataFrame, out: Path) -> None:
    if "whisper_wer" not in df.columns or df.whisper_wer.dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df.whisper_wer.notna()].copy()
    sub["bucket"] = sub.provider + " | " + sub.task
    sns.stripplot(data=sub, x="bucket", y="whisper_wer", size=8, alpha=0.7, jitter=0.15, ax=ax)
    ax.set_ylabel("Whisper round-trip WER")
    ax.set_xlabel("")
    ax.set_title("Intelligibility per provider × task")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.parquet")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--stats", default="results/stats.json")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    df = pd.read_parquet(root / args.metrics)
    stats = json.loads((root / args.stats).read_text())
    calibration = json.loads((root / args.calibration).read_text()) if Path(root / args.calibration).exists() else {}

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("fig1_summary_table.png", lambda p: fig_summary_table(df, stats, p)),
        ("fig2_calibration.png", lambda p: fig_calibration(df, calibration, p)),
        ("fig3_metric_scatter.png", lambda p: fig_metric_scatter(df, p)),
        ("fig4_wer_distribution.png", lambda p: fig_wer_distribution(df, p)),
    ]
    produced = []
    for name, fn in figures:
        p = out_dir / name
        try:
            fn(p)
            if p.exists():
                produced.append(p)
                print(f"  ✓ {p}")
            else:
                print(f"  ⊘ {p} (skipped — no data)")
        except Exception as e:
            print(f"  ✗ {p}: {type(e).__name__}: {e}")
    print(f"\n{len(produced)} figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
