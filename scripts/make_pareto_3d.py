"""Render 3D Pareto-front visualisations for both tasks.

For each task produces two files:
  - report/figures/pareto_3d_<task>.pdf      -- matplotlib static (single viewpoint)
  - report/figures/pareto_3d_<task>.html     -- plotly interactive

3D projection axes:
  TTS:      UTMOSv2 (higher=better) x Whisper-WER (lower=better) x cost_usd (lower=better)
  Cloning:  UTMOSv2 (higher=better) x WavLM-sim (higher=better) x cost_usd (lower=better)

Pareto-optimal points get a thick gold edge; others get a thin black edge.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go


COMMERCIAL = {"azure_tts", "google_tts", "openai_tts", "elevenlabs", "typecast", "resemble"}
FAKE_CLONING_PROVIDERS = {"azure_tts", "google_tts", "openai_tts"}


def _provider_color(prov: str) -> str:
    palette = {
        "azure_tts": "#1f77b4",
        "google_tts": "#2ca02c",
        "openai_tts": "#ff7f0e",
        "elevenlabs": "#d62728",
        "typecast": "#9467bd",
        "resemble": "#8c564b",
        "xtts_v2": "#e377c2",
        "f5_tts": "#7f7f7f",
        "cosyvoice2": "#17becf",
        "fish_speech_s1": "#bcbd22",
        "fish_speech_s2_pro": "#aec7e8",
    }
    return palette.get(prov, "#666666")


def _backfill_cost(df: pd.DataFrame, ratecards_path: Path) -> pd.DataFrame:
    if not ratecards_path or not ratecards_path.exists():
        return df
    rc = json.loads(ratecards_path.read_text())
    rates = rc.get("rates_usd_per_million_chars", {})
    if "cost_usd" not in df.columns:
        df["cost_usd"] = np.nan
    for prov, rate in rates.items():
        mask = df["provider"] == prov
        if mask.any() and df.loc[mask, "cost_usd"].isna().all():
            df.loc[mask, "cost_usd"] = df.loc[mask, "character_count"].fillna(0) * rate / 1_000_000.0
    return df


def _pareto_set(rows, axis_directions):
    """Return set of provider names that are Pareto-optimal.

    rows: dict prov -> {axis: value}
    axis_directions: dict axis -> +1 (maximize) / -1 (minimize)
    """
    optimal = set()
    for prov, p in rows.items():
        dominated = False
        for prov2, q in rows.items():
            if prov == prov2:
                continue
            # q weakly dominates p iff q is >=/<= on all axes and strictly better on one
            weak_all = True
            strict_any = False
            for ax, direction in axis_directions.items():
                if direction == +1:
                    if q[ax] < p[ax]:
                        weak_all = False
                        break
                    if q[ax] > p[ax]:
                        strict_any = True
                else:
                    if q[ax] > p[ax]:
                        weak_all = False
                        break
                    if q[ax] < p[ax]:
                        strict_any = True
            if weak_all and strict_any:
                dominated = True
                break
        if not dominated:
            optimal.add(prov)
    return optimal


def render_task(df: pd.DataFrame, task: str, out_pdf: Path, out_html: Path, full_dim_optimal=None):
    """Render 3D scatter on the canonical 3 axes; highlight full-dim Pareto optimal
    set (passed via `full_dim_optimal`) if provided; otherwise compute 3D-only Pareto."""
    if task == "tts":
        axes = {"utmos": +1, "whisper_wer": -1, "cost_usd": -1}
        ax_labels = {"utmos": "UTMOSv2 (higher=better)", "whisper_wer": "WER (lower=better)", "cost_usd": "Cost USD (lower=better)"}
    elif task == "cloning":
        axes = {"utmos": +1, "wavlm_sim": +1, "cost_usd": -1}
        ax_labels = {"utmos": "UTMOSv2 (higher=better)", "wavlm_sim": "WavLM-sim (higher=better)", "cost_usd": "Cost USD (lower=better)"}
    else:
        raise ValueError(f"unknown task: {task}")

    cols = list(axes.keys())
    sub = df[df["task"] == task]
    if task == "cloning":
        sub = sub[~sub["provider"].isin(FAKE_CLONING_PROVIDERS)]
    sys_means = sub.groupby("provider")[cols].mean().dropna()
    if sys_means.empty:
        print(f"  no data for task {task}, skipping")
        return

    rows = {prov: dict(zip(cols, vals)) for prov, vals in sys_means.iterrows()}
    three_d_optimal = _pareto_set(rows, axes)
    if full_dim_optimal is None:
        full_dim_optimal = three_d_optimal
        front_kind = "3D"
    else:
        front_kind = "full-dim"
    print(f"  {task}: 3D-optimal={sorted(three_d_optimal)}; full-dim-optimal={sorted(full_dim_optimal)}")
    optimal = full_dim_optimal

    # --- matplotlib static ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax_x, ax_y, ax_z = cols
    for prov, p in rows.items():
        color = _provider_color(prov)
        marker = "o" if prov in COMMERCIAL else "s"
        is_opt = prov in optimal
        edge = "gold" if is_opt else "black"
        lw = 2.5 if is_opt else 0.5
        size = 220 if is_opt else 120
        ax.scatter([p[ax_x]], [p[ax_y]], [p[ax_z]], c=color, marker=marker, s=size,
                   edgecolors=edge, linewidths=lw, depthshade=False)
        ax.text(p[ax_x], p[ax_y], p[ax_z], "  " + prov, fontsize=7)

    ax.set_xlabel(ax_labels[ax_x])
    ax.set_ylabel(ax_labels[ax_y])
    ax.set_zlabel(ax_labels[ax_z])
    ax.set_title(f"3D Pareto projection — {task}")

    # Reasonable viewpoint that minimises overlap for both tasks
    ax.view_init(elev=22, azim=-55)

    # Add legend for markers + Pareto-optimal indicator
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markersize=10, label="Commercial"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4",
                   markersize=10, label="Open source"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="gold", markeredgewidth=2,
                   markersize=10, label="Pareto-optimal"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # --- plotly interactive ---
    traces = []
    for prov, p in rows.items():
        color = _provider_color(prov)
        is_opt = prov in optimal
        is_commercial = prov in COMMERCIAL
        traces.append(go.Scatter3d(
            x=[p[ax_x]], y=[p[ax_y]], z=[p[ax_z]],
            mode="markers+text",
            text=[prov],
            textposition="top center",
            textfont=dict(size=10),
            name=prov + (" *" if is_opt else ""),
            marker=dict(
                size=12 if is_opt else 8,
                color=color,
                symbol="circle" if is_commercial else "square",
                line=dict(color="gold" if is_opt else "black", width=3 if is_opt else 1),
            ),
            hovertemplate=(
                f"<b>{prov}</b><br>"
                f"{ax_labels[ax_x]}: %{{x:.4f}}<br>"
                f"{ax_labels[ax_y]}: %{{y:.4f}}<br>"
                f"{ax_labels[ax_z]}: %{{z:.5f}}<br>"
                f"{'Pareto-optimal' if is_opt else 'Dominated'}<extra></extra>"
            ),
        ))

    fig3d = go.Figure(data=traces)
    fig3d.update_layout(
        title=f"3D Pareto projection — {task} (Pareto-optimal points have gold border, marked with *)",
        scene=dict(
            xaxis=dict(title=ax_labels[ax_x]),
            yaxis=dict(title=ax_labels[ax_y]),
            zaxis=dict(title=ax_labels[ax_z]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        showlegend=False,
        width=900,
        height=700,
    )
    fig3d.write_html(out_html)
    print(f"  wrote {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="/mnt/gefs-me/rl/users/kifbell/diploma/voice-bench-outputs/results/metrics.parquet",
    )
    ap.add_argument(
        "--ratecards",
        default="/home/kifbell/sweets/twix/agents_workdir/cost_ratecards.json",
    )
    ap.add_argument(
        "--stats",
        default="/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json",
    )
    ap.add_argument(
        "--out-dir",
        default="report/figures",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = _backfill_cost(df, Path(args.ratecards))

    # Load full-dim Pareto front from stats.json (primary projection per task).
    stats_path = Path(args.stats)
    tts_full = None
    cloning_full = None
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        h1 = stats.get("hypotheses_verdict", {}).get("h1_no_dominant_provider", {})
        tts_full = set(h1.get("h1.tts", {}).get("providers_on_frontier", []))
        cloning_full = set(h1.get("h1.cloning", {}).get("providers_on_frontier", []))

    out_dir = Path(args.out_dir)
    render_task(df, "tts", out_dir / "pareto_3d_tts.pdf", out_dir / "pareto_3d_tts.html",
                full_dim_optimal=tts_full)
    render_task(df, "cloning", out_dir / "pareto_3d_cloning.pdf", out_dir / "pareto_3d_cloning.html",
                full_dim_optimal=cloning_full)


if __name__ == "__main__":
    main()
