"""Generate the 6 PDF figures for the voice-bench thesis.

  1. pareto_x3.pdf -- three Pareto projections (UTMOS x cost, sim x cost, UTMOS x sim)
  2. correlation_heatmap.pdf -- Spearman correlation between metrics (utterance-level)
  3. ranking_table.pdf -- mean +/- 95% CI per (provider, task) for each metric
  4. calibration_plot.pdf -- cloning similarity histograms with upper/lower anchors
  5. gap_distribution.pdf -- CC vs CO vs OO pair-gap distributions per metric
  6. utterance_jitter.pdf -- per-utterance score scatter between provider pairs

Reads results/metrics.parquet + results/calibration_anchors.json. Writes PDFs to
results/figures/ (local dir or wherever --out points).
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


COMMERCIAL = {"azure_tts", "google_tts", "openai_tts", "elevenlabs", "typecast", "resemble"}
OPEN_SOURCE = {"xtts_v2", "f5_tts", "cosyvoice2", "fish_speech_s1", "fish_speech_s2_pro"}
# Providers whose "cloning" endpoint outputs a fixed default voice and ignores
# the reference -- excluded from cloning-task plots entirely.
FAKE_CLONING_PROVIDERS = {"azure_tts", "google_tts", "openai_tts"}
ALL_METRICS = ("utmos", "nisqa_mos", "whisper_wer", "wavlm_sim", "ecapa_sim")


# -----------------------------------------------------------------------------
def _provider_color(prov: str) -> str:
    if prov in COMMERCIAL:
        return "#d62728"  # red for commercial
    return "#1f77b4"  # blue for open source


def _backfill_cost_from_ratecards(df: pd.DataFrame, ratecards_path: Path) -> pd.DataFrame:
    """Fill cost_usd column from ratecards JSON for providers where it's missing.

    Same logic as stat_pack.py: cost = character_count * USD-per-million-chars / 1e6.
    """
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


def _pareto_panel(ax, sub, x_col, y_col, xl, yl, title, log_x=False, full_dim_optimal=None):
    """Render one Pareto-projection panel.

    All providers are plotted. Border encoding:
      - gold thick edge: provider is on the full-dimensional Pareto front (from stats.json).
      - silver thin edge: provider is on the 2D-only Pareto front for this panel but
        NOT on the full-dim front (it would be dominated if more axes were considered).
      - black thin edge: dominated in both 2D and full-dim.
    """
    minimize_x = x_col in ("cost_usd", "whisper_wer", "latency_seconds")
    minimize_y = y_col in ("cost_usd", "whisper_wer", "latency_seconds")
    pts = sub[[x_col, y_col]].dropna().to_dict("index")
    two_d_optimal = set()
    for prov, p in pts.items():
        dominated = False
        for prov2, q in pts.items():
            if prov == prov2:
                continue
            x_be = (q[x_col] <= p[x_col]) if minimize_x else (q[x_col] >= p[x_col])
            y_be = (q[y_col] <= p[y_col]) if minimize_y else (q[y_col] >= p[y_col])
            x_st = (q[x_col] < p[x_col]) if minimize_x else (q[x_col] > p[x_col])
            y_st = (q[y_col] < p[y_col]) if minimize_y else (q[y_col] > p[y_col])
            if x_be and y_be and (x_st or y_st):
                dominated = True
                break
        if not dominated:
            two_d_optimal.add(prov)

    full_dim_optimal = full_dim_optimal or set()

    for prov, row in sub.iterrows():
        if pd.isna(row.get(x_col)) or pd.isna(row.get(y_col)):
            continue
        color = _provider_color(prov)
        marker = "o" if prov in COMMERCIAL else "s"
        if prov in full_dim_optimal:
            edge = "gold"
            lw = 2.5
            zo = 5
        elif prov in two_d_optimal:
            edge = "silver"
            lw = 1.5
            zo = 4
        else:
            edge = "black"
            lw = 0.5
            zo = 3
        ax.scatter(row[x_col], row[y_col], c=color, marker=marker, s=140,
                   edgecolors=edge, linewidths=lw, zorder=zo)
        ax.annotate(prov, (row[x_col], row[y_col]), xytext=(5, 5),
                    textcoords="offset points", fontsize=7)

    if log_x:
        positive = sub[x_col][sub[x_col] > 0]
        if not positive.empty:
            ax.set_xscale("symlog", linthresh=positive.min() * 0.5)

    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=0.3)


def _pareto_legend(fig):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markersize=10, label="Commercial"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4",
                   markersize=10, label="Open source"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="gold", markeredgewidth=2.5,
                   markersize=10, label="Full-dim Pareto front"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="silver", markeredgewidth=1.5,
                   markersize=10, label="2D-only Pareto front"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05),
               fontsize=9)


def fig_pareto_tts(df: pd.DataFrame, out: Path, full_dim_optimal: set | None = None):
    """Six 2D Pareto projections for TTS (all pairs from the 4 measured axes
    UTMOSv2 / NISQA / WER / cost). Full-dim 4D Pareto-optimal set is highlighted
    in gold; 2D-only optimal points get a silver border."""
    sys_means = df[df["task"] == "tts"].groupby("provider").agg(
        utmos=("utmos", "mean"),
        nisqa_mos=("nisqa_mos", "mean"),
        whisper_wer=("whisper_wer", "mean"),
        cost_usd=("cost_usd", "mean"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    UTMOS_L = "UTMOSv2 (higher=better)"
    NISQA_L = "NISQA (higher=better)"
    WER_L = "Whisper-WER (lower=better)"
    COST_L = "Cost per file (USD, log scale)"
    # Row 1: UTMOSv2 against the other three.
    _pareto_panel(axes[0, 0], sys_means, "nisqa_mos", "utmos",
                  NISQA_L, UTMOS_L, "UTMOSv2 vs NISQA",
                  full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[0, 1], sys_means, "whisper_wer", "utmos",
                  WER_L, UTMOS_L, "UTMOSv2 vs WER",
                  full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[0, 2], sys_means, "cost_usd", "utmos",
                  COST_L, UTMOS_L, "UTMOSv2 vs Cost",
                  log_x=True, full_dim_optimal=full_dim_optimal)
    # Row 2: the remaining three pairs (NISQA x WER, NISQA x Cost, WER x Cost).
    _pareto_panel(axes[1, 0], sys_means, "whisper_wer", "nisqa_mos",
                  WER_L, NISQA_L, "NISQA vs WER",
                  full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[1, 1], sys_means, "cost_usd", "nisqa_mos",
                  COST_L, NISQA_L, "NISQA vs Cost",
                  log_x=True, full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[1, 2], sys_means, "cost_usd", "whisper_wer",
                  COST_L, WER_L, "WER vs Cost",
                  log_x=True, full_dim_optimal=full_dim_optimal)
    _pareto_legend(fig)
    fig.suptitle("Pareto projections for Text-to-Speech task "
                 "(gold = full 4D front, silver = 2D-only)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_pareto_cloning(df: pd.DataFrame, out: Path, full_dim_optimal: set | None = None):
    """Six 2D Pareto projections for voice cloning. Fake-cloning providers
    (Azure/Google/OpenAI -- fixed-voice endpoints) are dropped globally because
    they did not perform the task being evaluated. The remaining six panels cover
    each of the five measured axes (UTMOSv2, NISQA, WavLM-sim, WER, cost) at
    least twice. Full-dim 5D Pareto-optimal set is highlighted in gold;
    2D-only optimal points get a silver border."""
    sub = df[(df["task"] == "cloning") & (~df["provider"].isin(FAKE_CLONING_PROVIDERS))]
    sys_means = sub.groupby("provider").agg(
        utmos=("utmos", "mean"),
        nisqa_mos=("nisqa_mos", "mean"),
        wavlm_sim=("wavlm_sim", "mean"),
        whisper_wer=("whisper_wer", "mean"),
        cost_usd=("cost_usd", "mean"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    UTMOS_L = "UTMOSv2 (higher=better)"
    NISQA_L = "NISQA (higher=better)"
    WAVLM_L = "WavLM-sim (higher=better)"
    WER_L = "Whisper-WER (lower=better)"
    COST_L = "Cost per file (USD, log scale)"
    # Row 1: UTMOSv2 against other quality axes + cost.
    _pareto_panel(axes[0, 0], sys_means, "wavlm_sim", "utmos",
                  WAVLM_L, UTMOS_L, "UTMOSv2 vs WavLM-sim",
                  full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[0, 1], sys_means, "whisper_wer", "utmos",
                  WER_L, UTMOS_L, "UTMOSv2 vs WER",
                  full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[0, 2], sys_means, "cost_usd", "utmos",
                  COST_L, UTMOS_L, "UTMOSv2 vs Cost",
                  log_x=True, full_dim_optimal=full_dim_optimal)
    # Row 2: speaker-similarity and cost trade-offs (incl. the narrative-critical
    # WER vs Cost panel that explains why commercial cloners stay on the front).
    _pareto_panel(axes[1, 0], sys_means, "whisper_wer", "wavlm_sim",
                  WER_L, WAVLM_L, "WavLM-sim vs WER",
                  full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[1, 1], sys_means, "cost_usd", "wavlm_sim",
                  COST_L, WAVLM_L, "WavLM-sim vs Cost",
                  log_x=True, full_dim_optimal=full_dim_optimal)
    _pareto_panel(axes[1, 2], sys_means, "cost_usd", "whisper_wer",
                  COST_L, WER_L, "WER vs Cost",
                  log_x=True, full_dim_optimal=full_dim_optimal)
    _pareto_legend(fig)
    fig.suptitle("Pareto projections for voice cloning task "
                 "(gold = full 5D front, silver = 2D-only; fixed-voice providers excluded)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_correlation_heatmap(df: pd.DataFrame, out: Path):
    """Spearman correlation between metrics at utterance-level.

    Excludes the cloning rows of fake-cloning providers (Azure/Google/OpenAI),
    matching the analysed-rows convention used by stat_pack."""
    fake_mask = (df['task'] == 'cloning') & (df['provider'].isin(FAKE_CLONING_PROVIDERS))
    df = df[~fake_mask]
    sub = df[list(ALL_METRICS)].dropna(how="all")
    corr = np.full((len(ALL_METRICS), len(ALL_METRICS)), np.nan)
    for i, m1 in enumerate(ALL_METRICS):
        for j, m2 in enumerate(ALL_METRICS):
            both = sub[[m1, m2]].dropna()
            if len(both) < 30:
                continue
            r = spearmanr(both[m1].values, both[m2].values)
            rho_val = r.statistic if hasattr(r, "statistic") else r[0]
            # If both columns are identical (i==j or perfectly correlated), spearman
            # may return a 0-d array. Force scalar.
            rho = float(np.asarray(rho_val).item()) if np.asarray(rho_val).size == 1 else float("nan")
            if not np.isnan(rho):
                corr[i, j] = rho
    corr = pd.DataFrame(corr, index=ALL_METRICS, columns=ALL_METRICS)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr.astype(float), annot=False, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                square=True, cbar_kws={"shrink": 0.7})
    # Manual annotation: seaborn annot=True silently drops cells after first NaN row.
    for i in range(len(ALL_METRICS)):
        for j in range(len(ALL_METRICS)):
            v = corr.iloc[i, j] if hasattr(corr, "iloc") else corr[i, j]
            if pd.isna(v):
                continue
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Utterance-level Spearman ρ between metrics")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_ranking_table(df: pd.DataFrame, out: Path):
    """Provider ranking table (mean per metric, separately for tts and cloning).

    Drops cloning rows of fake-cloning providers."""
    fake_mask = (df["task"] == "cloning") & (df["provider"].isin(FAKE_CLONING_PROVIDERS))
    df = df[~fake_mask]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panel_metrics = [
        ("utmos", "UTMOSv2 (1-5, higher better)"),
        ("nisqa_mos", "NISQA MOS (1-5, higher better)"),
        ("whisper_wer", "Whisper-WER (0-1, lower better)"),
        ("wavlm_sim", "WavLM speaker sim (0-1, higher better)"),
        ("ecapa_sim", "ECAPA speaker sim (0-1, higher better)"),
        ("latency_seconds", "Скорость генерации (с, меньше=быстрее)"),
    ]
    for ax, (m, title) in zip(axes.flat, panel_metrics):
        if m not in df.columns:
            ax.axis("off")
            continue
        sub = df.groupby(["provider", "task"])[m].agg(["mean", "std", "count"]).reset_index()
        sub = sub.dropna(subset=["mean"])
        if sub.empty:
            ax.axis("off")
            continue
        # Sort by tts mean, then append cloning-only providers (sorted by cloning mean)
        ascending = m in ("whisper_wer", "latency_seconds")
        tts_order = sub[sub["task"] == "tts"].sort_values("mean", ascending=ascending)["provider"].tolist()
        cloning_order = sub[sub["task"] == "cloning"].sort_values("mean", ascending=ascending)["provider"].tolist()
        cloning_only = [p for p in cloning_order if p not in tts_order]
        order = tts_order + cloning_only
        if not order:
            ax.axis("off")
            continue
        sub["provider"] = pd.Categorical(sub["provider"], categories=order, ordered=True)
        sub = sub.dropna(subset=["provider"]).sort_values("provider")
        ax_data = sub.pivot(index="provider", columns="task", values="mean")
        ax_err = sub.pivot(index="provider", columns="task", values="std")
        ax_data.plot(kind="barh", ax=ax, xerr=ax_err, capsize=2)
        ax.set_title(title)
        ax.set_ylabel("")
        ax.invert_yaxis()
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Provider ranking per metric (mean ± std, separate tts vs cloning)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_calibration_plot(df: pd.DataFrame, calib: dict, out: Path):
    """Cloning similarity histograms with upper/lower anchors per metric."""
    if not calib:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric_key, col in zip(axes, ("wavlm", "ecapa"), ("wavlm_sim", "ecapa_sim")):
        anchors = calib.get(metric_key, {})
        cloning = df[df["task"] == "cloning"]
        for prov in sorted(cloning["provider"].unique()):
            vals = cloning[cloning["provider"] == prov][col].dropna()
            if len(vals) < 5:
                continue
            color = _provider_color(prov)
            ls = "-" if prov in COMMERCIAL else "--"
            ax.hist(vals, bins=30, alpha=0.4, color=color, label=prov, histtype="step", linestyle=ls, linewidth=1.5)
        # Anchors
        if anchors:
            up = anchors.get("upper_p50")
            up_p05 = anchors.get("upper_p05")
            lo = anchors.get("lower_mean")
            lo_p95 = anchors.get("lower_p95")
            if up is not None:
                ax.axvline(up, color="green", linestyle="-", linewidth=1.5, label=f"same-speaker p50 ({up:.3f})")
            if up_p05 is not None:
                ax.axvline(up_p05, color="green", linestyle=":", linewidth=1, alpha=0.7)
            if lo is not None:
                ax.axvline(lo, color="orange", linestyle="-", linewidth=1.5, label=f"cross-speaker mean ({lo:.3f})")
            if lo_p95 is not None:
                ax.axvline(lo_p95, color="orange", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(f"{col} on cloning task")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Calibration anchors vs cloning similarity distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_gap_distribution(df: pd.DataFrame, out: Path):
    """CC vs CO vs OO pair-gap distributions for UTMOS / NISQA / WER on each task.

    For the cloning task we exclude fake-cloning providers (Azure/Google/OpenAI)
    -- they did not perform the task and shouldn't enter pair-gap statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panel_metrics = ("utmos", "nisqa_mos", "whisper_wer")
    for row_idx, task in enumerate(("tts", "cloning")):
        task_df = df[df["task"] == task]
        if task == "cloning":
            task_df = task_df[~task_df["provider"].isin(FAKE_CLONING_PROVIDERS)]
        sys_means = task_df.groupby("provider")[list(panel_metrics)].mean()
        providers = list(sys_means.index)
        for col_idx, m in enumerate(panel_metrics):
            ax = axes[row_idx, col_idx]
            cc_gaps, co_gaps, oo_gaps = [], [], []
            for p1, p2 in combinations(providers, 2):
                a, b = sys_means.loc[p1, m], sys_means.loc[p2, m]
                if pd.isna(a) or pd.isna(b):
                    continue
                gap = abs(a - b)
                if p1 in COMMERCIAL and p2 in COMMERCIAL:
                    cc_gaps.append(gap)
                elif p1 not in COMMERCIAL and p2 not in COMMERCIAL:
                    oo_gaps.append(gap)
                else:
                    co_gaps.append(gap)
            data = [cc_gaps, co_gaps, oo_gaps]
            ax.boxplot(data, labels=["CC", "CO", "OO"])
            ax.set_title(f"{task}: {m}")
            ax.set_ylabel("|gap|")
            ax.grid(True, alpha=0.3)
    fig.suptitle("System-level gap distributions: CC vs CO vs OO", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_utterance_jitter(df: pd.DataFrame, out: Path):
    """Per-utterance UTMOS scatter for first few provider pairs to visualise variance.

    Cloning task: excludes fake-cloning providers."""
    sub = df[(df["task"] == "cloning") & (~df["provider"].isin(FAKE_CLONING_PROVIDERS))]
    pivot = sub.pivot_table(index="utt_id", columns="provider", values="utmos", aggfunc="first")
    providers = list(pivot.columns)
    pairs = list(combinations(providers, 2))[:6]  # show 6 pairs
    n = len(pairs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes_flat = axes.flatten() if rows > 1 else axes
    for ax, (p1, p2) in zip(axes_flat, pairs):
        both = pivot[[p1, p2]].dropna()
        if len(both) < 10:
            ax.axis("off")
            continue
        ax.scatter(both[p1], both[p2], alpha=0.4, s=10)
        lo = min(both[p1].min(), both[p2].min())
        hi = max(both[p1].max(), both[p2].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
        ax.set_xlabel(f"{p1} UTMOS")
        ax.set_ylabel(f"{p2} UTMOS")
        ax.set_title(f"{p1} vs {p2}")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Per-utterance UTMOS jitter (cloning task)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="/mnt/gefs-me/rl/users/kifbell/diploma/voice-bench-outputs/results/metrics.parquet",
    )
    ap.add_argument(
        "--calibration",
        default="/mnt/gefs-me/rl/users/kifbell/diploma/voice-bench-outputs/results/calibration_anchors.json",
    )
    ap.add_argument(
        "--ratecards",
        default="/home/kifbell/sweets/twix/agents_workdir/cost_ratecards.json",
        help="Cost ratecards JSON for backfilling cost_usd where parquet has NaN.",
    )
    ap.add_argument(
        "--stats",
        default="/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json",
        help="stat_pack.py output: defines the full-dim Pareto fronts highlighted on 2D plots.",
    )
    ap.add_argument("--out", default="/home/kifbell/sweets/twix/agents_workdir/voice-bench-figures")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    ratecards_path = Path(args.ratecards) if getattr(args, "ratecards", None) else None
    if ratecards_path:
        df = _backfill_cost_from_ratecards(df, ratecards_path)
    calib_path = Path(args.calibration)
    calib_full = json.loads(calib_path.read_text()) if calib_path.exists() else None
    # Calibration anchors json structure is {wavlm: {upper_p50: ..., ...}, ecapa: {...}}
    # under the 'wavlm' and 'ecapa' top-level keys.

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full-dim Pareto front from stats.json (primary projection per task).
    stats_path = Path(getattr(args, "stats", None) or "/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json")
    tts_full = set()
    cloning_full = set()
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        h1 = stats.get("hypotheses_verdict", {}).get("h1_no_dominant_provider", {})
        tts_full = set(h1.get("h1.tts", {}).get("providers_on_frontier", []))
        cloning_full = set(h1.get("h1.cloning", {}).get("providers_on_frontier", []))
    fig_pareto_tts(df, out_dir / "pareto_tts.pdf", full_dim_optimal=tts_full)
    print(f"  pareto_tts.pdf written  (full-dim gold: {sorted(tts_full)})")
    fig_pareto_cloning(df, out_dir / "pareto_cloning.pdf", full_dim_optimal=cloning_full)
    print(f"  pareto_cloning.pdf written  (full-dim gold: {sorted(cloning_full)})")
    fig_correlation_heatmap(df, out_dir / "correlation_heatmap.pdf")
    print("  correlation_heatmap.pdf written")
    fig_ranking_table(df, out_dir / "ranking_table.pdf")
    print("  ranking_table.pdf written")
    fig_calibration_plot(df, calib_full, out_dir / "calibration_plot.pdf")
    print("  calibration_plot.pdf written")
    fig_gap_distribution(df, out_dir / "gap_distribution.pdf")
    print("  gap_distribution.pdf written")
    fig_utterance_jitter(df, out_dir / "utterance_jitter.pdf")
    print("  utterance_jitter.pdf written")
    print(f"\nfigures in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
