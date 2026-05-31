"""Generate LaTeX tables for the thesis from voice-bench-stats.json.

Writes one .tex file per table into report/tables/:
  - ranking_tts.tex / ranking_cloning.tex   -- provider x metric mean (95% CI) per task
  - hypotheses.tex                          -- H1/H2/H3 verdicts summary
  - hypotheses_h3.tex                       -- H3 TOST per-metric breakdown (Section 5.3)
  - costs.tex                               -- per-provider USD/1M chars + actual spend
  - sample_sizes.tex                        -- (provider, task) -> N (non-NaN per metric)
  - pareto.tex                              -- Pareto frontier membership per projection
  - calibration.tex                         -- WavLM/ECAPA calibration anchors

Each table is a standalone \begin{table}...\end{table} environment with
\label{} so report.tex can \ref it.
"""
import argparse
import json
import math
from pathlib import Path
import pandas as pd


PROVIDER_DISPLAY = {
    "azure_tts": "Azure Neural",
    "google_tts": "Google Neural2",
    "openai_tts": "OpenAI tts-1",
    "elevenlabs": "ElevenLabs",
    "typecast": "Typecast",
    "resemble": "Resemble",
    "xtts_v2": "XTTSv2",
    "f5_tts": "F5-TTS",
    "cosyvoice2": "CosyVoice2",
    "fish_speech_s1": "Fish-Speech S1",
    "fish_speech_s2_pro": "Fish-Speech S2 Pro",
}

COMMERCIAL = {"azure_tts", "google_tts", "openai_tts", "elevenlabs", "typecast", "resemble"}

# Providers whose "cloning" endpoint outputs a fixed default voice and ignores
# the reference -- excluded from cloning-task analytics globally.
FAKE_CLONING_PROVIDERS = {"azure_tts", "google_tts", "openai_tts"}

METRIC_DISPLAY = {
    "utmos": "UTMOSv2",
    "nisqa_mos": "NISQA",
    "whisper_wer": "WER",
    "wavlm_sim": "WavLM",
    "ecapa_sim": "ECAPA",
    "cost_usd": "Cost (USD)",
    "latency_seconds": "Скорость (с)",
}

METRIC_PREC = {
    "utmos": 2,
    "nisqa_mos": 2,
    "whisper_wer": 3,
    "wavlm_sim": 3,
    "ecapa_sim": 3,
    "cost_usd": 4,
    "latency_seconds": 2,
}


def fmt(x, prec=3):
    if x is None:
        return "--"
    try:
        if math.isnan(x):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"{x:.{prec}f}"


def fmt_pval(p):
    if p is None:
        return "--"
    if p < 1e-6:
        return "$< 10^{-6}$"
    if p < 0.001:
        return "$< 0.001$"
    return f"{p:.3f}"


def fmt_mean_ci(cell, prec=3):
    if not cell or cell.get("mean") is None:
        return "--"
    m = cell["mean"]
    lo = cell.get("ci_low")
    hi = cell.get("ci_high")
    if lo is None or hi is None:
        return f"{m:.{prec}f}"
    half = (hi - lo) / 2.0
    return f"{m:.{prec}f} $\\pm$ {half:.{prec}f}"


def disp(prov):
    return PROVIDER_DISPLAY.get(prov, prov.replace("_", "\\_"))


def write_table(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  {path}")


# --- Ranking tables (one per task) ---

def gen_ranking(stats, task, out_path):
    bc = stats["bootstrap_ci"]
    metrics = ["utmos", "nisqa_mos", "whisper_wer"]
    if task == "cloning":
        metrics += ["wavlm_sim", "ecapa_sim"]
    metrics += ["cost_usd", "latency_seconds"]

    providers_in_task = []
    for k in bc:
        if k.endswith(f"__{task}"):
            providers_in_task.append(k.rsplit("__", 1)[0])
    # Sort by UTMOS desc
    providers_in_task.sort(key=lambda p: -(bc.get(f"{p}__{task}", {}).get("utmos", {}).get("mean") or 0))

    col_spec = "l" + "r" * len(metrics)
    header = " & ".join(["Провайдер"] + [METRIC_DISPLAY[m] for m in metrics]) + " \\\\"

    rows = []
    for prov in providers_in_task:
        cell = bc.get(f"{prov}__{task}", {})
        row_vals = [disp(prov)]
        for m in metrics:
            row_vals.append(fmt_mean_ci(cell.get(m, {}), prec=METRIC_PREC[m]))
        rows.append(" & ".join(row_vals) + " \\\\")

    task_ru = "TTS" if task == "tts" else "Voice Cloning"
    body = "\n".join(rows)
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Средние значения метрик с 95\\% бутстрап-доверительными интервалами по провайдерам (задача: {task_ru}). N=400 на ячейку для большинства комбинаций.}}
\\label{{tab:ranking_{task}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    write_table(out_path, content)


# --- Hypotheses verdict table ---

def gen_hypotheses(stats, out_path):
    hv = stats["hypotheses_verdict"]
    rows = []

    h1 = hv["h1_no_dominant_provider"]
    rows.append(("H1.tts", "Pareto frontier (TTS) >1 провайдера",
                 f"{h1['h1.tts']['n_on_frontier']}/{h1['h1.tts']['n_total']} на Pareto",
                 "--", h1['h1.tts']['supports']))
    rows.append(("H1.cloning", "Pareto frontier (Cloning) >1 провайдера",
                 f"{h1['h1.cloning']['n_on_frontier']}/{h1['h1.cloning']['n_total']} на Pareto",
                 "--", h1['h1.cloning']['supports']))

    h2nat = hv["h2_naturalness_agreement"]
    rows.append(("H2.tts.nat", "UTMOSv2 и NISQA согласуются (TTS), $\\rho \\geq 0.7$",
                 f"$\\rho = {fmt(h2nat['h2.tts.naturalness']['rho'])}$",
                 fmt_pval(h2nat['h2.tts.naturalness']['p_value']),
                 h2nat['h2.tts.naturalness']['supports']))
    rows.append(("H2.cloning.nat", "UTMOSv2 и NISQA согласуются (Cloning), $\\rho \\geq 0.7$",
                 f"$\\rho = {fmt(h2nat['h2.cloning.naturalness']['rho'])}$",
                 fmt_pval(h2nat['h2.cloning.naturalness']['p_value']),
                 h2nat['h2.cloning.naturalness']['supports']))

    h2sim = hv["h2_similarity_divergence"]
    rows.append(("H2.cloning.sim", "WavLM и ECAPA расходятся (Cloning), $\\rho < 0.7$",
                 f"$\\rho = {fmt(h2sim['h2.cloning.similarity']['rho'])}$",
                 fmt_pval(h2sim['h2.cloning.similarity']['p_value']),
                 h2sim['h2.cloning.similarity']['supports']))

    h3 = hv["h3_tost_equivalence"]
    for key, sub in h3.items():
        descr = f"TOST: $|\\bar{{x}}_C - \\bar{{x}}_{{OS}}| < \\delta = {fmt(sub['delta'])}$"
        stat = (f"$\\bar{{x}}_C = {fmt(sub['mean_commercial'])}$, "
                f"$\\bar{{x}}_{{OS}} = {fmt(sub['mean_open_source'])}$, "
                f"$\\Delta = {fmt(sub['mean_diff'])}$")
        rows.append((key.replace("_", "\\_"), descr, stat,
                     fmt_pval(sub['p_value']),
                     sub['supports']))

    body_rows = []
    for label, descr, stat, pval, supports in rows:
        verdict = "\\checkmark" if supports else "$\\times$"
        body_rows.append(f"{label} & {descr} & {stat} & {pval} & {verdict} \\\\")

    body = "\n".join(body_rows)
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Сводная таблица проверки гипотез. Колонка «Итог» --- $\\checkmark$ = подтверждена, $\\times$ = не подтверждена.}}
\\label{{tab:hypotheses}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llllc}}
\\toprule
ID & Формулировка & Статистика & $p$-value & Итог \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    write_table(out_path, content)


# --- H3-only TOST table (Section 5.3) ---

# Display labels for the (task, metric) keys used in stats["hypotheses_verdict"]["h3_tost_equivalence"].
H3_TASK_DISPLAY = {"tts": "TTS", "cloning": "Cloning"}
H3_METRIC_DISPLAY = {
    "utmos": "UTMOSv2",
    "nisqa_mos": "NISQA",
    "whisper_wer": "Whisper-WER",
    "wavlm_sim": "WavLM-sim",
    "ecapa_sim": "ECAPA-sim",
}


def _h3_row_label(key):
    # keys look like "h3.tts.utmos", "h3.cloning.wavlm_sim", ...
    parts = key.split(".", 2)
    if len(parts) != 3:
        return key.replace("_", "\\_")
    _, task, metric = parts
    task_label = H3_TASK_DISPLAY.get(task, task)
    metric_label = H3_METRIC_DISPLAY.get(metric, metric.replace("_", "-"))
    return f"{task_label}, {metric_label}"


def gen_hypotheses_h3(stats, out_path):
    """H3-only breakdown for the body of Section 5.3.

    Reports every (task, metric) TOST cell: group means, observed Delta,
    pre-registered delta, one-sided p-value, and verdict. The broader
    H1/H2/H3 summary lives in hypotheses.tex (rendered in the discussion).
    """
    h3 = stats["hypotheses_verdict"]["h3_tost_equivalence"]

    body_rows = []
    for key, sub in h3.items():
        label = _h3_row_label(key)
        mean_c = fmt(sub["mean_commercial"])
        mean_os = fmt(sub["mean_open_source"])
        delta_obs = fmt(sub["mean_diff"])
        delta_pre = fmt(sub["delta"])
        pval = fmt_pval(sub["p_value"])
        verdict = "\\checkmark" if sub["supports"] else "$\\times$"
        body_rows.append(
            f"{label} & ${mean_c}$ & ${mean_os}$ & ${delta_obs}$ & ${delta_pre}$ & {pval} & {verdict} \\\\"
        )

    body = "\n".join(body_rows)
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Результаты теста H3 по каждой (задача, метрика)-комбинации. Двусторонний TOST на провайдер-уровневых средних (commercial vs open-source): $H_{{0}}: |\\bar{{x}}_{{C}} - \\bar{{x}}_{{OS}}| \\geq \\delta$. Колонки: $\\bar{{x}}_{{C}}$ --- среднее по commercial-провайдерам, $\\bar{{x}}_{{OS}}$ --- по open-source; $\\Delta$ --- наблюдаемая разность средних; $\\delta$ --- пре-регистрированный порог эквивалентности (см. Раздел~\\ref{{ssec:h3}}). Эквивалентность считается подтверждённой ($\\checkmark$) при $p < 0.05$. Для метрик speaker similarity провайдеры без zero-shot эндпоинта (Azure, Google, OpenAI) исключены.}}
\\label{{tab:hypotheses_h3}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrlc}}
\\toprule
(Задача, Метрика) & $\\bar{{x}}_{{C}}$ & $\\bar{{x}}_{{OS}}$ & $\\Delta$ & $\\delta$ & $p$-value & Итог \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    write_table(out_path, content)


# --- Costs table ---

def gen_costs(ratecards, stats, out_path):
    rates = ratecards["rates_usd_per_million_chars"]
    actual = ratecards.get("actual_spend_usd", {})
    bc = stats["bootstrap_ci"]

    providers = sorted(rates.keys(),
                       key=lambda p: (p not in COMMERCIAL, rates[p]))
    rows = []
    for prov in providers:
        rate = rates[prov]
        rate_str = "0 (open-source)" if rate == 0 else f"{rate:.2f}"
        spend = actual.get(prov)
        spend_str = f"\\${spend:.2f}" if spend is not None else "--"
        kind = "коммерч." if prov in COMMERCIAL else "open-source"
        # Mean cost per file (cloning task, or tts if no cloning)
        c_cell = bc.get(f"{prov}__cloning", {}).get("cost_usd", {})
        if c_cell.get("mean") is None:
            c_cell = bc.get(f"{prov}__tts", {}).get("cost_usd", {})
        per_file = fmt(c_cell.get("mean"), prec=5)
        rows.append(f"{disp(prov)} & {kind} & {rate_str} & {per_file} & {spend_str} \\\\")

    body = "\n".join(rows)
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Стоимость синтеза по провайдерам. «Цена / 1M симв.» --- официальная цена по pricing page. «\\$ / файл» --- средняя стоимость одного из наших файлов ($\\sim$110 симв.). «Фактич. расход» --- сумма реально потраченных средств за весь эксперимент (800 файлов на коммерческого провайдера).}}
\\label{{tab:costs}}
\\begin{{tabular}}{{llrrr}}
\\toprule
Провайдер & Тип & Цена / 1M симв., USD & \\$ / файл & Фактич. расход \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    write_table(out_path, content)


# --- Sample sizes table ---

def gen_sample_sizes(parquet_path, out_path):
    df = pd.read_parquet(parquet_path)
    metrics = ["utmos", "nisqa_mos", "whisper_wer", "wavlm_sim", "ecapa_sim"]
    task_label = {"tts": "TTS", "cloning": "Cloning"}
    rows_data = []
    providers = sorted(df["provider"].unique())
    for prov in providers:
        for task in ("tts", "cloning"):
            sub = df[(df["provider"] == prov) & (df["task"] == task)]
            if sub.empty:
                continue
            if task == "cloning" and prov in FAKE_CLONING_PROVIDERS:
                # Fake-cloning providers: their cloning rows are excluded from
                # all analytics; drop them from the sample-sizes table too.
                continue
            row = [disp(prov), task_label[task], str(len(sub))]
            for m in metrics:
                if m in sub.columns:
                    n_ok = sub[m].notna().sum()
                    row.append(str(n_ok))
                else:
                    row.append("--")
            rows_data.append(row)

    # Collapse provider name on consecutive rows belonging to the same provider:
    # show the name only on the first row, leave subsequent ones blank.
    prev_prov = None
    for r in rows_data:
        if r[0] == prev_prov:
            r[0] = ""
        else:
            prev_prov = r[0]

    body_rows = [" & ".join(r) + " \\\\" for r in rows_data]
    body = "\n".join(body_rows)
    header = " & ".join(["Провайдер", "Задача", "Файлов"] + [METRIC_DISPLAY[m] for m in metrics]) + " \\\\"
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Размеры выборок: количество файлов с непустым значением каждой метрики, по (провайдер $\\times$ задача). Прочерк --- метрика неприменима. Cloning-строки Azure / Google / OpenAI отсутствуют, поскольку эти провайдеры не имеют voice-cloning эндпоинта и не выполняли задачу клонирования голоса; их файлы исключены из всех cloning-аналитик.}}
\\label{{tab:sample_sizes}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrrr}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    write_table(out_path, content)


# --- Pareto table ---

def gen_pareto(stats, out_path):
    par = stats["pareto_frontier"]
    rows = []
    for task in ("tts", "cloning"):
        if task not in par:
            continue
        first = True
        for proj_name, proj in par[task].items():
            providers = proj.get("providers", [])
            optimal = proj.get("is_pareto_optimal", [])
            axes = proj.get("axes", [])
            members = [p for p, o in zip(providers, optimal) if o]
            members_str = ", ".join(disp(p) for p in members)
            axes_str = ", ".join(METRIC_DISPLAY.get(a, a) for a in axes)
            if first:
                # Primary projection: bold the entire row, prefix with star for visibility.
                row = f"\\textbf{{{task.upper()}}} & \\textbf{{{axes_str}}} & \\textbf{{{len(members)}/{len(providers)}}} & \\textbf{{{members_str}}} \\\\"
                first = False
            else:
                row = f"{task.upper()} & {axes_str} & {len(members)}/{len(providers)} & {members_str} \\\\"
            rows.append(row)

    body = "\n".join(rows)
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Sensitivity-анализ Парето-фронта: размер и состав фронта в зависимости от выбора измеренных осей. Первая строка для каждой задачи (выделена) --- основная проекция, по которой формулируется verdict H1. Последующие строки --- альтернативные проекции с удалением части осей; они показывают, как нетривиальность фронта зависит от количества измеряемых характеристик качества.}}
\\label{{tab:pareto}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lllL}}
\\toprule
Задача & Оси проекции & На фронте / всего & Парето-оптимальные \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    # Note: L is not a standard tabular spec, replace with p{} or use array package.
    # Use 'p{6cm}' instead.
    content = content.replace("{lllL}", "{lllp{6cm}}")
    write_table(out_path, content)


# --- Calibration anchors table ---

def gen_calibration(stats, out_path):
    cal = stats.get("calibration_anchors", {})
    if not cal:
        return
    rows = []
    for metric in ("wavlm", "ecapa"):
        a = cal.get(metric, {})
        rows.append(
            f"{metric.upper()} & "
            f"{fmt(a.get('upper_p50'), 3)} & "
            f"{fmt(a.get('upper_mean'), 3)} $\\pm$ {fmt(a.get('upper_std'), 3)} & "
            f"{a.get('upper_n','--')} & "
            f"{fmt(a.get('lower_mean'), 3)} $\\pm$ {fmt(a.get('lower_std'), 3)} & "
            f"{a.get('lower_n','--')} \\\\"
        )
    body = "\n".join(rows)
    content = f"""% Auto-generated. Do not edit.
\\begin{{table}}[ht]
\\centering
\\caption{{Калибровочные якоря для метрик speaker similarity. «Same-speaker» --- cosine similarity между двумя независимыми утверждениями одного диктора (верхняя граница). «Cross-speaker» --- схожесть между разными дикторами (нижняя граница).}}
\\label{{tab:calibration}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Метрика & Same-speaker $p_{{50}}$ & Same-speaker mean $\\pm$ std & $N_{{up}}$ & Cross-speaker mean $\\pm$ std & $N_{{lo}}$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    write_table(out_path, content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json")
    ap.add_argument("--parquet", default="/mnt/gefs-me/rl/users/kifbell/diploma/voice-bench-outputs/results/metrics.parquet")
    ap.add_argument("--ratecards", default="/home/kifbell/sweets/twix/agents_workdir/cost_ratecards.json")
    ap.add_argument("--out-dir", default="report/tables")
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())
    ratecards = json.loads(Path(args.ratecards).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_ranking(stats, "tts", out_dir / "ranking_tts.tex")
    gen_ranking(stats, "cloning", out_dir / "ranking_cloning.tex")
    gen_hypotheses(stats, out_dir / "hypotheses.tex")
    gen_hypotheses_h3(stats, out_dir / "hypotheses_h3.tex")
    gen_costs(ratecards, stats, out_dir / "costs.tex")
    gen_sample_sizes(args.parquet, out_dir / "sample_sizes.tex")
    gen_pareto(stats, out_dir / "pareto.tex")
    gen_calibration(stats, out_dir / "calibration.tex")
    print(f"all tables in {out_dir}")


if __name__ == "__main__":
    main()
