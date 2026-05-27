"""Compute stat-pack for the voice-bench experiment.

Reads results/metrics.parquet + results/calibration_anchors.json + ratecards,
writes results/stats.json with:

  1. bootstrap_ci: 95% bootstrap CI for the mean of every metric on every
     (provider, task) cell
  2. spearman_metric_pairs: system-level Spearman between metric pairs
  3. spearman_utt_level: utterance-level Spearman (all rows, paired)
  4. wilcoxon_providers: paired Wilcoxon (signed-rank) between every provider
     pair on UTMOS / NISQA / WER / WavLM-sim / ECAPA-sim, separately for tts
     and cloning
  5. mannwhitney_cc_vs_co: H3 test of whether commercial-commercial gaps are
     smaller than commercial-OS gaps
  6. pareto_frontiers: provider's Pareto-optimality flag in 3 projections
     (UTMOS x sim x cost, UTMOS x sim x latency), with bootstrap stability

Hypotheses summary (research_plan):
  H1: No dominant provider; they form a Pareto frontier in (UTMOSv2, WavLM,
      $/1k chars).
  H2: Naturalness metrics (UTMOSv2, NISQA) agree (Spearman > 0.7);
      similarity metrics (WavLM, ECAPA) diverge.
  H3: Commercial-commercial UTMOS gap < commercial-OS UTMOS gap (i.e. open
      source has caught up with commercial).
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, mannwhitneyu, spearmanr, wilcoxon


# Providers that DO NOT actually perform zero-shot voice cloning -- their
# "cloning" endpoint just generates audio with a fixed default voice and
# ignores the reference. We exclude their cloning rows from all cloning-task
# analyses (Pareto front, Spearman, H3 gap test, bootstrap CI), because the
# task they performed is structurally different from what the other systems did.
# Their TTS-task rows remain in the TTS analysis as usual.
FAKE_CLONING_PROVIDERS = ("azure_tts", "google_tts", "openai_tts")


def drop_fake_cloning(df):
    """Return a copy of df with the fake-cloning rows removed from the cloning task."""
    fake_mask = (df["task"] == "cloning") & (df["provider"].isin(FAKE_CLONING_PROVIDERS))
    return df[~fake_mask].reset_index(drop=True)


# Provider classification used by H3 (commercial vs open-source).
COMMERCIAL = {"azure_tts", "google_tts", "openai_tts", "elevenlabs", "typecast", "resemble"}
OPEN_SOURCE = {"xtts_v2", "f5_tts", "cosyvoice2", "fish_speech_s1", "fish_speech_s2_pro"}

METRICS_NATURAL = ("utmos", "nisqa_mos")
METRICS_INTELLIG = ("whisper_wer",)
METRICS_SIM = ("wavlm_sim", "ecapa_sim")
ALL_METRICS = METRICS_NATURAL + METRICS_INTELLIG + METRICS_SIM


def boot_ci_mean(values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap CI for the sample mean. Returns mean, ci_low, ci_high, n."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    if len(arr) == 1:
        return {"mean": float(arr[0]), "ci_low": float(arr[0]), "ci_high": float(arr[0]), "n": 1}
    res = bootstrap(
        (arr,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=ci,
        method="basic",
        random_state=42,
    )
    return {
        "mean": float(arr.mean()),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n": int(len(arr)),
    }


def is_pareto_optimal(points: np.ndarray, maximize: list[bool]) -> np.ndarray:
    """Return a boolean array marking Pareto-optimal points.

    points: (n, d) array of d scores per point.
    maximize: list of d booleans -- True if larger is better on that axis.
    A point is Pareto-optimal if no other point dominates it (>=on all axes,
    strict >on at least one).
    """
    n = len(points)
    # Flip axes where smaller is better so all become "larger = better".
    p = points.copy()
    for j, mx in enumerate(maximize):
        if not mx:
            p[:, j] = -p[:, j]
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        # Some other point dominates i iff exists j != i with p[j] >= p[i] on
        # all axes and strict > on at least one.
        others = np.delete(p, i, axis=0)
        dominated = np.any(np.all(others >= p[i], axis=1) & np.any(others > p[i], axis=1))
        optimal[i] = not dominated
    return optimal


def stat_pack(parquet_path: Path, calib_path: Path, ratecards_path: Path | None) -> dict:
    df = pd.read_parquet(parquet_path)
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None

    # Backfill cost_usd for any provider that's missing it, using ratecards.
    ratecards = None
    if ratecards_path and ratecards_path.exists():
        ratecards = json.loads(ratecards_path.read_text())
        rates = ratecards.get("rates_usd_per_million_chars", {})
        for prov, rate in rates.items():
            mask = df["provider"] == prov
            if mask.any() and ("cost_usd" not in df.columns or df.loc[mask, "cost_usd"].isna().all()):
                if "cost_usd" not in df.columns:
                    df["cost_usd"] = np.nan
                df.loc[mask, "cost_usd"] = df.loc[mask, "character_count"].fillna(0) * rate / 1_000_000

    # Keep a copy of the raw data for meta-bookkeeping but use a filtered version
    # for all downstream analyses: providers that did not perform real voice cloning
    # are dropped from the cloning task entirely.
    df_raw = df
    df = drop_fake_cloning(df)

    stats: dict = {
        "meta": {
            "parquet": str(parquet_path),
            "n_rows_raw": int(len(df_raw)),
            "n_rows_analysed": int(len(df)),
            "fake_cloning_providers_excluded": list(FAKE_CLONING_PROVIDERS),
            "providers": sorted(df["provider"].unique()),
            "tasks": sorted(df["task"].unique()),
            "metrics_available": [c for c in ALL_METRICS if c in df.columns],
        },
        "calibration_anchors": calib,
    }

    # ----- 1. Bootstrap CI per (provider, task, metric) -----
    boot: dict = {}
    for (prov, task), group in df.groupby(["provider", "task"]):
        key = f"{prov}__{task}"
        boot[key] = {}
        for m in ALL_METRICS + ("cost_usd", "latency_seconds"):
            if m not in df.columns:
                continue
            boot[key][m] = boot_ci_mean(group[m].values)
    stats["bootstrap_ci"] = boot

    # ----- 2. System-level Spearman between metrics (one rank per provider) -----
    sys_spearman: dict = {}
    for task in df["task"].unique():
        sys_means = df[df["task"] == task].groupby("provider")[list(ALL_METRICS)].mean()
        for m1, m2 in combinations(sys_means.columns, 2):
            a, b = sys_means[m1].dropna(), sys_means[m2].dropna()
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            rho, p = spearmanr(a.loc[common], b.loc[common])
            sys_spearman[f"{task}__{m1}__vs__{m2}"] = {
                "rho": float(rho) if not np.isnan(rho) else None,
                "p_value": float(p) if not np.isnan(p) else None,
                "n_providers": int(len(common)),
            }
    stats["spearman_metric_pairs__system_level"] = sys_spearman

    # ----- 2b. Utterance-level Spearman -----
    utt_spearman: dict = {}
    for m1, m2 in combinations(ALL_METRICS, 2):
        if m1 not in df.columns or m2 not in df.columns:
            continue
        sub = df[[m1, m2]].dropna()
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub[m1], sub[m2])
        utt_spearman[f"{m1}__vs__{m2}"] = {
            "rho": float(rho),
            "p_value": float(p),
            "n_rows": int(len(sub)),
        }
    stats["spearman_metric_pairs__utterance_level"] = utt_spearman

    # ----- 3. Wilcoxon paired between provider pairs (per task, per metric) -----
    wilcox: dict = {}
    for task in df["task"].unique():
        wilcox[task] = {}
        sub = df[df["task"] == task]
        for m in ALL_METRICS:
            if m not in df.columns:
                continue
            pivot = sub.pivot_table(index="utt_id", columns="provider", values=m, aggfunc="first")
            wilcox[task][m] = {}
            for p1, p2 in combinations(sorted(pivot.columns), 2):
                paired = pivot[[p1, p2]].dropna()
                if len(paired) < 10:
                    continue
                diffs = paired[p1].values - paired[p2].values
                if np.allclose(diffs, 0):
                    continue
                try:
                    stat, pval = wilcoxon(paired[p1], paired[p2])
                    wilcox[task][m][f"{p1}__vs__{p2}"] = {
                        "stat": float(stat),
                        "p_value": float(pval),
                        "n_pairs": int(len(paired)),
                        "median_diff": float(np.median(diffs)),
                    }
                except Exception as e:
                    wilcox[task][m][f"{p1}__vs__{p2}"] = {"error": str(e)[:100]}
    stats["wilcoxon_paired"] = wilcox

    # ----- 4. H3 Mann-Whitney: commercial-commercial vs commercial-OS gaps -----
    h3: dict = {}
    for task in df["task"].unique():
        sys_means = df[df["task"] == task].groupby("provider")[list(ALL_METRICS)].mean()
        h3[task] = {}
        for m in ALL_METRICS:
            if m not in sys_means.columns:
                continue
            providers = list(sys_means.index)
            cc_gaps, co_gaps, oo_gaps = [], [], []
            for p1, p2 in combinations(providers, 2):
                a, b = sys_means.loc[p1, m], sys_means.loc[p2, m]
                if pd.isna(a) or pd.isna(b):
                    continue
                gap = abs(a - b)
                p1_comm = p1 in COMMERCIAL
                p2_comm = p2 in COMMERCIAL
                if p1_comm and p2_comm:
                    cc_gaps.append(gap)
                elif (not p1_comm) and (not p2_comm):
                    oo_gaps.append(gap)
                else:
                    co_gaps.append(gap)
            if len(cc_gaps) >= 3 and len(co_gaps) >= 3:
                try:
                    stat, pval = mannwhitneyu(cc_gaps, co_gaps, alternative="less")
                    h3[task][m] = {
                        "stat": float(stat),
                        "p_value": float(pval),
                        "alternative": "cc_gap_less_than_co_gap",
                        "cc_gaps_mean": float(np.mean(cc_gaps)),
                        "co_gaps_mean": float(np.mean(co_gaps)),
                        "oo_gaps_mean": float(np.mean(oo_gaps)) if oo_gaps else None,
                        "n_cc_pairs": len(cc_gaps),
                        "n_co_pairs": len(co_gaps),
                        "n_oo_pairs": len(oo_gaps),
                    }
                except Exception as e:
                    h3[task][m] = {"error": str(e)[:100]}
    stats["h3_commercial_vs_oss"] = h3

    # ----- 5. Pareto frontier (3 projections) -----
    pareto: dict = {}
    for task in df["task"].unique():
        sys_means = df[df["task"] == task].groupby("provider").agg(
            utmos=("utmos", "mean"),
            nisqa_mos=("nisqa_mos", "mean"),
            wavlm_sim=("wavlm_sim", "mean"),
            ecapa_sim=("ecapa_sim", "mean"),
            whisper_wer=("whisper_wer", "mean"),
            cost_usd=("cost_usd", "mean"),
            latency_seconds=("latency_seconds", "mean"),
        )
        pareto[task] = {}
        # Three projections from research_plan: UTMOS x sim x cost, UTMOS x sim x latency,
        # and the WER-aware version sim x WER x cost. Sim is the WavLM-based one.
        if task == "tts":
            # TTS task has no per-speaker reference; WER replaces WavLM-sim.
            # Primary 4D = UTMOS + NISQA + WER + cost. Plus the sub-projections
            # for sensitivity-of-Pareto analysis.
            projections = [
                # Primary projection that drives H1.tts verdict.
                ("utmos_nisqa_wer_cost", ["utmos", "nisqa_mos", "whisper_wer", "cost_usd"], [True, True, False, False]),
                # Sensitivity: how the frontier shrinks/grows when axes are removed.
                ("utmos_wer_cost", ["utmos", "whisper_wer", "cost_usd"], [True, False, False]),
                ("utmos_nisqa_cost", ["utmos", "nisqa_mos", "cost_usd"], [True, True, False]),
                ("utmos_wer", ["utmos", "whisper_wer"], [True, False]),
                ("utmos_cost", ["utmos", "cost_usd"], [True, False]),
                ("wer_cost", ["whisper_wer", "cost_usd"], [False, False]),
            ]
        else:
            # Cloning primary 5D = UTMOS + NISQA + WavLM + WER + cost.
            projections = [
                # Primary projection that drives H1.cloning verdict.
                ("utmos_nisqa_sim_wer_cost", ["utmos", "nisqa_mos", "wavlm_sim", "whisper_wer", "cost_usd"], [True, True, True, False, False]),
                # Sensitivity: drop one axis at a time, or fall back to the canonical 3D.
                ("utmos_sim_cost", ["utmos", "wavlm_sim", "cost_usd"], [True, True, False]),
                ("utmos_sim_wer_cost", ["utmos", "wavlm_sim", "whisper_wer", "cost_usd"], [True, True, False, False]),
                ("utmos_nisqa_sim_cost", ["utmos", "nisqa_mos", "wavlm_sim", "cost_usd"], [True, True, True, False]),
                ("utmos_sim_latency", ["utmos", "wavlm_sim", "latency_seconds"], [True, True, False]),
                ("utmos_cost", ["utmos", "cost_usd"], [True, False]),
                ("sim_cost", ["wavlm_sim", "cost_usd"], [True, False]),
            ]
        for name, cols, maxim in projections:
            sub = sys_means[cols].dropna()
            if len(sub) < 2:
                continue
            points = sub.values
            optimal = is_pareto_optimal(points, maxim)
            pareto[task][name] = {
                "providers": list(sub.index),
                "points": [list(map(float, row)) for row in points],
                "is_pareto_optimal": [bool(x) for x in optimal],
                "axes": cols,
                "maximize": maxim,
            }
    stats["pareto_frontier"] = pareto

    # ----- Hypotheses summary at top level for quick read -----
    stats["hypotheses_verdict"] = build_verdicts(stats)
    return stats


SPEARMAN_THRESHOLD = 0.7  # H2 agreement threshold; matches research_plan.


def build_verdicts(stats: dict) -> dict:
    """Six verdicts in the structure of hypotheses.md."""
    verdicts: dict = {}

    spear = stats.get("spearman_metric_pairs__system_level", {})
    h3 = stats.get("h3_commercial_vs_oss", {})
    pareto = stats.get("pareto_frontier", {})

    def find_spear(prefix_task: str, m1: str, m2: str):
        # Spearman dict keys look like "tts__utmos__vs__nisqa_mos".
        forward = f"{prefix_task}__{m1}__vs__{m2}"
        reverse = f"{prefix_task}__{m2}__vs__{m1}"
        return spear.get(forward) or spear.get(reverse)

    # H1.tts: Pareto on (UTMOS, WER, cost). H1 cloning: (UTMOS, WavLM-sim, cost).
    h1: dict = {}
    for task, proj_name in (("tts", "utmos_nisqa_wer_cost"), ("cloning", "utmos_nisqa_sim_wer_cost")):
        info = pareto.get(task, {}).get(proj_name)
        if not info:
            continue
        on_front = [p for p, ok in zip(info["providers"], info["is_pareto_optimal"]) if ok]
        h1[f"h1.{task}"] = {
            "projection": proj_name,
            "axes": info["axes"],
            "maximize": info["maximize"],
            "providers_on_frontier": on_front,
            "n_on_frontier": len(on_front),
            "n_total": len(info["providers"]),
            "supports": (len(on_front) >= 2),  # frontier non-trivial -> no single dominator
        }
    verdicts["h1_no_dominant_provider"] = h1

    # H2.tts.naturalness: UTMOS vs NISQA agree on tts.
    # H2.cloning.naturalness: same on cloning.
    h2_nat: dict = {}
    for task in ("tts", "cloning"):
        s = find_spear(task, "utmos", "nisqa_mos")
        if not s:
            continue
        h2_nat[f"h2.{task}.naturalness"] = {
            "rho": s.get("rho"),
            "p_value": s.get("p_value"),
            "n_providers": s.get("n_providers"),
            "threshold": SPEARMAN_THRESHOLD,
            "supports": (s.get("rho") is not None and s["rho"] >= SPEARMAN_THRESHOLD),
        }
    verdicts["h2_naturalness_agreement"] = h2_nat

    # H2.cloning.similarity: WavLM vs ECAPA DIVERGE (rho < threshold).
    # TTS doesn't have this -- similarity metrics undefined there.
    h2_sim: dict = {}
    s = find_spear("cloning", "wavlm_sim", "ecapa_sim")
    if s:
        h2_sim["h2.cloning.similarity"] = {
            "rho": s.get("rho"),
            "p_value": s.get("p_value"),
            "n_providers": s.get("n_providers"),
            "threshold": SPEARMAN_THRESHOLD,
            "supports": (s.get("rho") is not None and s["rho"] < SPEARMAN_THRESHOLD),
        }
    verdicts["h2_similarity_divergence"] = h2_sim

    # H3 per (task, metric): CC gaps < CO gaps via Mann-Whitney U.
    # Mapping from hypotheses.md:
    #   H3.tts: utmos, nisqa_mos, whisper_wer
    #   H3.cloning: utmos, nisqa_mos, whisper_wer, wavlm_sim, ecapa_sim
    h3_v: dict = {}
    tts_metrics = ("utmos", "nisqa_mos", "whisper_wer")
    cloning_metrics = ("utmos", "nisqa_mos", "whisper_wer", "wavlm_sim", "ecapa_sim")
    for task, metrics in (("tts", tts_metrics), ("cloning", cloning_metrics)):
        for m in metrics:
            res = h3.get(task, {}).get(m, {})
            if "p_value" not in res:
                continue
            h3_v[f"h3.{task}.{m}"] = {
                "cc_gap_mean": res["cc_gaps_mean"],
                "co_gap_mean": res["co_gaps_mean"],
                "oo_gap_mean": res.get("oo_gaps_mean"),
                "n_cc_pairs": res["n_cc_pairs"],
                "n_co_pairs": res["n_co_pairs"],
                "p_value": res["p_value"],
                "supports": res["p_value"] < 0.05,
            }
    verdicts["h3_commercial_vs_oss_gap"] = h3_v

    return verdicts


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
    )
    ap.add_argument(
        "--out",
        default="/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json",
    )
    args = ap.parse_args()

    stats = stat_pack(Path(args.parquet), Path(args.calibration), Path(args.ratecards))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"wrote {out_path}")
    print(f"n_rows raw: {stats['meta']['n_rows_raw']}, analysed: {stats['meta']['n_rows_analysed']}")
    print(f"providers: {stats['meta']['providers']}")
    print()
    print("=== Hypotheses verdict ===")
    print(json.dumps(stats.get("hypotheses_verdict", {}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
