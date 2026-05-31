# Hypotheses

Six hypotheses for the voice-bench experiment, split by task (TTS vs voice
cloning). Each pair (H1/H2/H3) is the same question phrased twice -- once
for the speaker-free TTS task and once for the per-speaker cloning task --
because the metric set, the provider set, and the dominant trade-offs are
different between the two.

The variable `X` (the threshold for Spearman ρ in H2) is a free parameter
to discuss with the advisor; placeholder values are recorded next to each
hypothesis based on the typical "strong agreement" convention used in the
TTSDS-style benchmarks.

## TTS task (speaker-free generation, 9 providers × 400 utterances)

**Providers in the TTS task**: xtts_v2, f5_tts, cosyvoice2, fish_speech_s1,
fish_speech_s2_pro, elevenlabs, azure_tts, google_tts, openai_tts.
Typecast and Resemble are excluded because we only ran them on the cloning
task (cloning-only providers).

Metric axes available for TTS:
- UTMOSv2 (naturalness, 1-5)
- NISQA-MOS (naturalness, 1-5)
- Whisper-WER (intelligibility, 0-1)
- cost_usd, latency_seconds

Speaker similarity (WavLM, ECAPA) is **not defined** for TTS because there is
no per-speaker reference -- each provider uses its own default voice.

### H1.tts -- No dominant provider on the TTS Pareto frontier

In the three-axis space (UTMOSv2 ↑, Whisper-WER ↓, $/1k chars ↓), the 9 TTS
providers form a Pareto frontier on which no single provider dominates on
all three axes simultaneously.

WavLM-similarity is replaced by Whisper-WER because the TTS task has no
reference for cloning similarity; intelligibility is the closest universal
quality axis.

**Test**: 3-D Pareto optimality at provider level (system-level means),
with 1000-resample bootstrap to flag providers on the frontier in ≥95% of
resamples.

### H2.tts -- Naturalness metric agreement, similarity-N/A

System-level rankings produced by UTMOSv2 and NISQA-MOS on the TTS task
agree (Spearman ρ_system ≥ X, with X = 0.7 as the proposed threshold).

Whisper-WER may or may not align with naturalness rankings -- left as
descriptive, not part of the hypothesis claim.

Speaker-similarity axis is not part of this hypothesis for TTS (no
reference).

**Test**: Spearman ρ on the 9 providers × 2 metrics matrix.

### H3.tts -- Equivalence: commercial does not beat OSS on TTS by more than δ

On the standardised LibriTTS-R test-clean benchmark, commercial TTS providers
do **not** outperform open-source models on average quality metrics by more
than a methodologically meaningful margin δ. The thresholds δ are chosen per
metric to be just below "user-noticeable" effect sizes:

- δ_utmos = 0.2 (UTMOS units, 1-5 scale)
- δ_nisqa = 0.2 (NISQA-MOS units, 1-5 scale)
- δ_wer = 0.05 (5 percentage points absolute WER)

Tested separately per quality metric:
- H3.tts.utmos
- H3.tts.nisqa_mos
- H3.tts.whisper_wer

**Test**: per metric, Two One-Sided Tests (TOST) for equivalence on the
provider-level means of the commercial group vs the open-source group.

  H0_TOST: |mean(commercial) - mean(open_source)| >= δ  (non-equivalence)
  H1_TOST: |mean(commercial) - mean(open_source)| <  δ  (equivalence)

Equivalence is established when both one-sided p-values are below α = 0.05.
Higher-is-better metrics (UTMOS, NISQA) and lower-is-better (WER) use the
same symmetric ±δ interval; the directional interpretation is folded into
the metric sign.

---

## Cloning task (voice cloning, 11 providers × 280-400 utterances)

**Providers in the cloning task**: all 9 from TTS + typecast + resemble.

- 5 real cloners with per-speaker IVC: xtts_v2, f5_tts, cosyvoice2,
  fish_speech_s1, fish_speech_s2_pro, elevenlabs (6 real cloners actually),
  typecast, resemble.
- 3 fake cloners (Azure / Google / OpenAI) -- they have no IVC, so the
  "cloning" column is generated with the same default voice as TTS and
  the per-speaker speaker-similarity values are vs an unrelated default.

For analyses that depend on WavLM-sim or ECAPA-sim, only the 6+2=8 real
cloners are included. For non-similarity analyses all 11 are kept.

Metric axes available for cloning:
- UTMOSv2 (naturalness, 1-5)
- NISQA-MOS (naturalness, 1-5)
- Whisper-WER (intelligibility, 0-1)
- WavLM-XVector similarity (speaker similarity, 0-1)
- ECAPA-TDNN similarity (speaker similarity, 0-1)
- cost_usd, latency_seconds

### H1.cloning -- No dominant provider on the cloning Pareto frontier

In the three-axis space (UTMOSv2 ↑, WavLM-sim ↑, $/1k chars ↓), the real
cloners form a Pareto frontier on which no single provider dominates on
all three axes simultaneously.

This is the canonical formulation from the original research plan; only
cloners with real IVC are included (Azure/Google/OpenAI excluded since
their "cloning" similarity is artificially-low default-voice noise).

**Test**: 3-D Pareto optimality at provider level (system-level means),
with 1000-resample bootstrap to flag providers on the frontier in ≥95%.

### H2.cloning -- Naturalness agreement; similarity divergence

Two claims:

H2.cloning.naturalness: System-level rankings by UTMOSv2 and NISQA-MOS on
the cloning task agree (Spearman ρ ≥ X, X = 0.7 proposed).

H2.cloning.similarity: System-level rankings by WavLM-sim and ECAPA-sim on
the cloning task **diverge** (Spearman ρ < X). This is the original H2
claim from the research plan: similarity metrics measure different aspects
of speaker identity, so a single similarity number under-determines clone
quality and multi-metric reporting is necessary.

**Test**: Spearman ρ on cloner providers, separately for the naturalness
and similarity pairs. Both rho values reported with bootstrap CI.

### H3.cloning -- Equivalence: commercial does not beat OSS on cloning by more than δ

Same TOST-based equivalence claim as H3.tts but on the cloning task. Per
metric, the equivalence margin δ is:
- δ_utmos = 0.2
- δ_nisqa = 0.2
- δ_wer = 0.05
- δ_wavlm_sim = 0.05 (cosine similarity units)
- δ_ecapa_sim = 0.05 (cosine similarity units)

Tested per metric:
- H3.cloning.utmos
- H3.cloning.nisqa_mos
- H3.cloning.whisper_wer
- H3.cloning.wavlm_sim (real cloners only)
- H3.cloning.ecapa_sim (real cloners only)

For similarity metrics the test considers only the subset of providers with
real zero-shot cloning support (fake cloners excluded).

**Test**: per metric, TOST with margin δ on the provider-level means of the
commercial vs open-source group. Equivalence supported when both one-sided
p-values are below α = 0.05.

---

## Cross-task observations (descriptive, not hypotheses)

Things we will look at without formal claim:

- Per-provider TTS↔cloning UTMOS delta: how much does each provider lose
  on its own UTMOSv2 when forced to clone vs its default voice? Identifies
  whether the cloning gap is similar across providers.
- Calibration anchors: same-speaker WavLM p50 / ECAPA p50 from GT pairs
  give the "ceiling" against which cloning sim is interpreted. Reported
  on the calibration plot.
- Cost vs naturalness slope: log-log slope per task tells us whether more
  expensive providers actually deliver more UTMOS. Descriptive only.
- Latency tier: bucket providers into <1 s / 1-3 s / 3-10 s / 10+ s and
  observe how task choice (TTS vs cloning) shifts a provider's latency.
