"""Generate report/values.tex with all dynamic numerical values as LaTeX commands.

Reads voice-bench-stats.json and writes a single .tex file with \newcommand
definitions for every value that needs to appear in the thesis. The .tex file
is included from report.tex via \input{values.tex}.

Naming convention: \valNAME where NAME is camelCase and self-describing.
All numbers are pre-formatted (rounded to appropriate precision).
"""
import argparse
import json
import math
from pathlib import Path


def _fmt(x, prec=3):
    if x is None:
        return "--"
    try:
        if math.isnan(x):
            return "--"
    except (TypeError, ValueError):
        return "--"
    if abs(x) < 1e-4 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{prec}f}"


def _pval(p):
    if p is None:
        return "--"
    if p < 1e-6:
        return "$< 10^{-6}$"
    if p < 0.001:
        return "$< 0.001$"
    return f"{p:.3f}"


def _bool(b):
    return "\\textbf{подтверждена}" if b else "\\textbf{не подтверждена}"


def _safe_cmd_name(s):
    parts = s.replace("_", " ").replace(".", " ").title().split()
    return "".join(parts)


def emit_commands(stats):
    meta = stats["meta"]
    yield "valNRowsAnalysed", str(meta.get("n_rows_analysed", meta.get("n_rows", "--")))
    yield "valNProviders", str(len(meta["providers"]))
    yield "valNSpeakers", "20"
    yield "valNUtterancesPerSpeaker", "20"

    hv = stats["hypotheses_verdict"]

    PROV_DISPLAY = {
        "azure_tts": "Azure Neural",
        "google_tts": "Google Neural2",
        "openai_tts": "OpenAI tts-1",
        "elevenlabs": "ElevenLabs",
        "typecast": "Typecast",
        "resemble": "Resemble",
        "xtts_v2": "XTTSv2",
        "f5_tts": "F5-TTS",
        "cosyvoice2": "CosyVoice2",
        "fish_speech_s1": "Fish-Speech~S1",
        "fish_speech_s2_pro": "Fish-Speech~S2~Pro",
    }

    def _pretty(provs):
        return ", ".join(PROV_DISPLAY.get(p, p.replace("_", "\\_")) for p in provs)

    h1 = hv["h1_no_dominant_provider"]
    yield "valHOneTtsFrontierSize", str(h1["h1.tts"]["n_on_frontier"])
    yield "valHOneTtsFrontierTotal", str(h1["h1.tts"]["n_total"])
    yield "valHOneTtsFrontierMembers", _pretty(h1["h1.tts"]["providers_on_frontier"])
    yield "valHOneTtsVerdict", _bool(h1["h1.tts"]["supports"])
    yield "valHOneCloningFrontierSize", str(h1["h1.cloning"]["n_on_frontier"])
    yield "valHOneCloningFrontierTotal", str(h1["h1.cloning"]["n_total"])
    yield "valHOneCloningFrontierMembers", _pretty(h1["h1.cloning"]["providers_on_frontier"])
    yield "valHOneCloningVerdict", _bool(h1["h1.cloning"]["supports"])

    h2nat = hv["h2_naturalness_agreement"]
    yield "valHTwoTtsNatRho", _fmt(h2nat["h2.tts.naturalness"]["rho"])
    yield "valHTwoTtsNatPval", _pval(h2nat["h2.tts.naturalness"]["p_value"])
    yield "valHTwoTtsNatN", str(h2nat["h2.tts.naturalness"]["n_providers"])
    yield "valHTwoTtsNatVerdict", _bool(h2nat["h2.tts.naturalness"]["supports"])
    yield "valHTwoCloningNatRho", _fmt(h2nat["h2.cloning.naturalness"]["rho"])
    yield "valHTwoCloningNatPval", _pval(h2nat["h2.cloning.naturalness"]["p_value"])
    yield "valHTwoCloningNatN", str(h2nat["h2.cloning.naturalness"]["n_providers"])
    yield "valHTwoCloningNatVerdict", _bool(h2nat["h2.cloning.naturalness"]["supports"])

    h2sim = hv["h2_similarity_divergence"]
    yield "valHTwoCloningSimRho", _fmt(h2sim["h2.cloning.similarity"]["rho"])
    yield "valHTwoCloningSimPval", _pval(h2sim["h2.cloning.similarity"]["p_value"])
    yield "valHTwoCloningSimN", str(h2sim["h2.cloning.similarity"]["n_providers"])
    yield "valHTwoCloningSimVerdict", _bool(h2sim["h2.cloning.similarity"]["supports"])

    h3 = hv["h3_tost_equivalence"]
    for key, sub in h3.items():
        cmd_suffix = _safe_cmd_name(key.replace("h3.", ""))
        yield f"valHThree{cmd_suffix}Delta", _fmt(sub["delta"])
        yield f"valHThree{cmd_suffix}MeanCommercial", _fmt(sub["mean_commercial"])
        yield f"valHThree{cmd_suffix}MeanOpenSource", _fmt(sub["mean_open_source"])
        yield f"valHThree{cmd_suffix}MeanDiff", _fmt(sub["mean_diff"])
        yield f"valHThree{cmd_suffix}Pval", _pval(sub["p_value"])
        yield f"valHThree{cmd_suffix}Verdict", _bool(sub["supports"])

    cal = stats.get("calibration_anchors", {})
    if cal:
        for metric in ("wavlm", "ecapa"):
            a = cal.get(metric, {})
            cap = metric.capitalize()
            yield f"valCalib{cap}UpperPFifty", _fmt(a.get("upper_p50"), prec=3)
            yield f"valCalib{cap}LowerMean", _fmt(a.get("lower_mean"), prec=3)
            yield f"valCalib{cap}UpperN", str(a.get("upper_n", "--"))
            yield f"valCalib{cap}LowerN", str(a.get("lower_n", "--"))

    bc = stats["bootstrap_ci"]
    METRICS = ["utmos", "nisqa_mos", "whisper_wer", "wavlm_sim", "ecapa_sim", "cost_usd", "latency_seconds"]
    higher_is_better = {"utmos": True, "nisqa_mos": True, "whisper_wer": False, "wavlm_sim": True, "ecapa_sim": True, "cost_usd": False, "latency_seconds": False}
    for task in ("tts", "cloning"):
        for m in METRICS:
            cells = []
            for k, v in bc.items():
                if not k.endswith(f"__{task}"):
                    continue
                prov = k.rsplit("__", 1)[0]
                val = v.get(m, {}).get("mean")
                if val is None:
                    continue
                cells.append((prov, val))
            if not cells:
                continue
            cells.sort(key=lambda x: x[1], reverse=higher_is_better[m])
            best_prov, best_val = cells[0]
            worst_prov, worst_val = cells[-1]
            task_cap = task.capitalize()
            m_cap = _safe_cmd_name(m)
            prec = 4 if m in ("cost_usd",) else 3
            yield f"valBest{task_cap}{m_cap}Provider", PROV_DISPLAY.get(best_prov, best_prov.replace("_", "\\_"))
            yield f"valBest{task_cap}{m_cap}Value", _fmt(best_val, prec=prec)
            yield f"valWorst{task_cap}{m_cap}Provider", PROV_DISPLAY.get(worst_prov, worst_prov.replace("_", "\\_"))
            yield f"valWorst{task_cap}{m_cap}Value", _fmt(worst_val, prec=prec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json")
    ap.add_argument("--out", default="report/values.tex")
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())

    lines = [
        "% Auto-generated by scripts/generate_values.py. DO NOT EDIT.",
        f"% Source: {args.stats}",
        "",
    ]
    seen = set()
    for cmd, value in emit_commands(stats):
        if cmd in seen:
            raise ValueError(f"duplicate command: {cmd}")
        seen.add(cmd)
        lines.append(f"\\newcommand{{\\{cmd}}}{{{value}}}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(seen)} commands to {out_path}")


if __name__ == "__main__":
    main()
