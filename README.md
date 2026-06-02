# voice-bench

A reproducible benchmark comparing commercial and open-source text-to-speech (TTS) and
zero-shot voice-cloning systems on the LibriTTS-R `test-clean` corpus, scored entirely with
automated judge models and evaluated against three pre-registered statistical hypotheses.

## Overview

voice-bench answers a practical question: across modern speech-synthesis providers, is there a
single dominant system, or does each occupy a distinct trade-off between naturalness,
intelligibility, speaker similarity, and cost? To answer it without human listening tests, the
benchmark generates audio from every provider for the same texts and reference voices, then scores
the outputs with five established automated metrics and tests three hypotheses with bootstrap
confidence intervals, rank correlations, and equivalence tests.

Two tasks are evaluated. In the **TTS** task, each provider synthesises text with a single fixed
default voice (speaker-free). In the **voice-cloning** task, each provider performs zero-shot
synthesis conditioned on a short reference clip of a target speaker. The same automated judges score
both tasks; speaker-similarity metrics apply only to cloning.

## What is evaluated

### Systems

Eleven providers participate. Nine run in the TTS task; all eleven run in the cloning task (Typecast
and Resemble are cloning-only). Open-source models run in-process on a GPU; commercial providers are
called over their APIs.

| Provider           | Type        | Model / default voice            | Access     | Tasks            |
| ------------------ | ----------- | -------------------------------- | ---------- | ---------------- |
| ElevenLabs         | commercial  | `eleven_multilingual_v2`         | API key    | TTS + cloning    |
| OpenAI TTS         | commercial  | `tts-1` (`alloy`)                | API key    | TTS              |
| Azure TTS          | commercial  | `en-US-AvaMultilingualNeural`    | API key    | TTS              |
| Google TTS         | commercial  | `en-US-Neural2-F`                | credentials| TTS              |
| Typecast           | commercial  | `ssfm-v30`                       | API key    | cloning          |
| Resemble           | commercial  | `resemble-v2`                    | API key    | cloning          |
| XTTS-v2            | open-source | Coqui `xtts_v2`                  | local GPU  | TTS + cloning    |
| F5-TTS             | open-source | `F5TTS_v1_Base`                  | local GPU  | TTS + cloning    |
| CosyVoice2         | open-source | `CosyVoice2-0.5B`                | local GPU  | TTS + cloning    |
| Fish-Speech S1     | open-source | `s1-mini`                        | local GPU  | TTS + cloning    |
| Fish-Speech S2 Pro | open-source | `s2-pro`                         | local GPU  | TTS + cloning    |

Three providers — Azure TTS, Google TTS, and OpenAI TTS — have no genuine zero-shot cloning
endpoint. Their cloning-task `clone()` calls fall back to a fixed default voice and ignore the
reference clip. They still produce audio in the cloning task, but their rows are excluded from every
speaker-similarity analysis, leaving **eight real cloners**.

### Metrics

Five automated judges are computed. No human evaluation is involved. All models consume 16 kHz mono
audio (resampled on the fly from the 24 kHz WAVs stored on disk).

| Metric        | Measures          | Scale          | Model                              | Tasks         |
| ------------- | ----------------- | -------------- | ---------------------------------- | ------------- |
| UTMOSv2       | naturalness (MOS) | 1–5, higher ↑  | sarulab-speech UTMOSv2 (fold 0)    | TTS + cloning |
| NISQA         | naturalness (MOS) | 1–5, higher ↑  | `gabrielmittag/NISQA`              | TTS + cloning |
| Whisper-WER   | intelligibility   | 0–1, lower ↓   | faster-whisper + jiwer WER         | TTS + cloning |
| WavLM x-vector| speaker similarity| −1–1, higher ↑ | `microsoft/wavlm-base-plus-sv`     | cloning only  |
| ECAPA-TDNN    | speaker similarity| −1–1, higher ↑ | `speechbrain/spkrec-ecapa-voxceleb`| cloning only  |

NISQA also reports four diagnostic sub-scores (noisiness, coloration, discontinuity, loudness).
Whisper-WER is a round-trip measure: each clip is transcribed and the transcript is compared, after
lowercasing and punctuation stripping, against the target text. UTMOSv2 is forced onto CPU because it
emits NaNs on recent CUDA builds.

Absolute cosine-similarity values are only interpretable relative to reference points, so two
calibration anchors are computed from ground-truth LibriTTS-R audio: a same-speaker upper bound
(reference vs. other utterances of the same speaker) and a cross-speaker lower bound (references of
different speakers), independently for WavLM and ECAPA.

## Hypotheses

Each hypothesis is tested separately for the TTS and cloning tasks. Formal definitions and the
chosen thresholds live in `hypotheses.md`.

**H1 — No single dominant provider.** In the trade-off space of naturalness, intelligibility,
speaker similarity (cloning only), and cost, providers form a Pareto frontier on which no system
dominates all axes at once. Pareto-optimality is assessed at provider-level means, with bootstrap
resampling to flag stable frontier membership.

**H2 — Naturalness metrics agree, similarity metrics diverge.** System-level rankings from UTMOSv2
and NISQA agree (Spearman ρ ≥ 0.7), while the two speaker-similarity metrics, WavLM x-vector and
ECAPA-TDNN, diverge (ρ < 0.7) — evidence that a single similarity number under-determines clone
quality.

**H3 — Commercial and open-source groups are statistically equivalent.** Using two one-sided tests
(TOST) with pre-registered equivalence margins, the difference between the commercial-group mean and
the open-source-group mean stays within δ on every quality metric (δ = 0.2 MOS for UTMOSv2 and NISQA;
δ = 0.05 for Whisper-WER, WavLM, and ECAPA).

## Dataset and sampling

The benchmark draws from LibriTTS-R `test-clean`. Sampling is deterministic (seed 42) and
gender-balanced: 20 speakers (10 female, 10 male) are selected, each contributing one reference clip
and 20 distinct target texts, for **400 (speaker, text) pairs**. A speaker is eligible only if it has
at least 21 usable utterances, one of which (5–10 s) becomes the cloning reference; target
utterances fall in the 2–15 s range. Target texts come from the LibriTTS-R `.normalized.txt` files so
numbers and abbreviations are consistently pre-expanded across providers. The committed
`data/samples.json` is this manifest; `data/SPEAKERS.txt` supplies the gender metadata.

## Repository structure

```
voice_bench/            # importable library
  dataset.py            # scan LibriTTS-R, build the gender-balanced manifest
  runner.py             # write 24 kHz WAV + JSON sidecar, idempotent resume
  providers/            # one adapter per provider (base.py defines the Provider protocol)
  metrics/              # judge models: utmos, nisqa, whisper_wer, wavlm_sim, ecapa_sim, audio_io
scripts/                # pipeline entry points (see below)
data/
  samples.json          # committed experiment manifest (20 speakers x 20 targets, seed 42)
  SPEAKERS.txt          # LibriTTS-R speaker metadata used for gender balancing
results/                # parquet, stats JSON, calibration anchors, figures
hypotheses.md           # formal H1/H2/H3 definitions and thresholds
requirements*.txt       # base / metrics-only / GPU dependency tiers
.env.example            # API-key and device template
```

Generated audio lands under `outputs/` (gitignored).

## Installation

Python 3.10+ is required. All API-based providers and all metric computation run on CPU; a CUDA GPU
is needed only for the local open-source models.

```bash
git clone git@github.com:kifbell/voice-bench.git && cd voice-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Three dependency tiers are provided. `requirements.txt` covers generation, metrics, and analysis.
`requirements-metrics.txt` is the CPU-only judge-model subset, useful on a scoring-only worker.
`requirements-gpu.txt` adds the local TTS engines (`coqui-tts` for XTTS-v2, `f5-tts` for F5-TTS) and
should be installed on the GPU worker:

```bash
pip install -r requirements-gpu.txt
```

Several provider SDKs are lazily imported and not pinned in any requirements file; install only the
ones you need:

```bash
pip install azure-cognitiveservices-speech   # Azure TTS
pip install google-cloud-texttospeech        # Google TTS
pip install typecast-python                  # Typecast
pip install resemble                         # Resemble
pip install statsmodels                      # required by the H3 TOST analysis
```

CosyVoice2 and Fish-Speech are installed from their upstream repositories and pointed at via
environment variables (CosyVoice2 weights download on first run; Fish-Speech needs its repo and
checkpoint directories set, see Configuration). Note the `numpy<2` pin — SpeechBrain 1.0 still
depends on the NumPy 1 ABI.

## Configuration

Copy the template and fill in credentials for the providers you intend to run:

```bash
cp .env.example .env
```

The full set of environment variables read by the code (the source of truth, broader than
`.env.example`) is:

| Variable                         | Unlocks                          |
| -------------------------------- | -------------------------------- |
| `ELEVENLABS_API_KEY`             | ElevenLabs                       |
| `OPENAI_API_KEY`                 | OpenAI TTS                       |
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_ENDPOINT` | Azure TTS            |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google TTS (service-account JSON path) |
| `TYPECAST_API_KEY`               | Typecast                         |
| `RESEMBLE_API_KEY`               | Resemble                         |
| `FISH_SPEECH_REPO_DIR` + `FISH_SPEECH_CHECKPOINT_DIR` | Fish-Speech S1 / S2 Pro |
| `VOICEBENCH_DEVICE`              | device for judge models (`cpu`, `cuda`, `mps`; UTMOSv2 always CPU) |

The open-source providers XTTS-v2, F5-TTS, and CosyVoice2 need no key. `REPLICATE_API_TOKEN` appears
in `.env.example` but is a placeholder — no Replicate provider is currently wired in.

Before spending API credits, verify connectivity:

```bash
python scripts/probe_apis.py
```

## Data setup

Download LibriTTS-R `test-clean` (~1.5 GB) from OpenSLR and extract it so the layout is
`LibriTTS_R/test-clean/<speaker>/<chapter>/*.wav` plus the matching `*.normalized.txt`:

```bash
curl -O https://www.openslr.org/resources/141/test_clean.tar.gz
tar -xzf test_clean.tar.gz
```

## Quickstart

A minimal end-to-end run on a small slice:

```bash
# Build the manifest (or reuse the committed data/samples.json)
python scripts/build_manifest.py

# Generate a pilot (5 speakers x 1 target) for one provider
python scripts/generate.py --provider elevenlabs --pilot

# Score the generated audio
python scripts/compute_metrics.py --sidecars-root outputs/sidecars --out results/metrics.parquet
```

## Full pipeline

The benchmark runs as an ordered sequence. Generation is idempotent — completed
`(provider, task, utterance)` triples are skipped on re-run — so each provider can be generated
independently and resumed after interruption.

**1. Build the manifest.** Writes `data/samples.json` (already committed; rebuild only to change the
sampling).

```bash
python scripts/build_manifest.py --n-speakers 20 --n-targets 20 --seed 42
```

**2. Generate audio**, once per provider. `--tasks` selects TTS, cloning, or both; `--max-speakers`
and `--max-targets-per-speaker` cap the run; `--pilot` is a 5×1 smoke slice.

```bash
python scripts/generate.py --provider <name> --tasks tts cloning
```

`<name>` is one of `elevenlabs`, `openai_tts`, `azure_tts`, `google_tts`, `typecast`, `resemble`,
`xtts_v2`, `f5_tts`, `cosyvoice2`, `fish_speech_s1`, `fish_speech_s2_pro`. Outputs land under
`outputs/audio/<provider>/<task>/<utt_id>.wav` with a JSON sidecar alongside, plus running
`outputs/costs.csv` and `outputs/failures.csv` logs.

**3. Compute similarity calibration anchors** from the ground-truth manifest audio:

```bash
python scripts/compute_calibration.py --out results/calibration_anchors.json
```

**4. Score the generated audio** with all five judges. Per-metric failures become NaN rather than
aborting the run.

```bash
python scripts/compute_metrics.py --sidecars-root outputs/sidecars --out results/metrics.parquet
```

**5. Run the statistics.** `stat_pack.py` consumes the metrics parquet, the calibration anchors, and
a cost rate-card, and writes `results/stats.json` containing bootstrap confidence intervals,
system- and utterance-level Spearman correlations, paired Wilcoxon tests, the H3 TOST equivalence
results, and Pareto-frontier membership.

```bash
python scripts/stat_pack.py \
  --parquet results/metrics.parquet \
  --calibration results/calibration_anchors.json \
  --ratecards cost_ratecards.json \
  --out results/stats.json
```

The rate-card is a small JSON of per-provider pricing you supply, of the form
`{"rates_usd_per_million_chars": {"elevenlabs": 66.0, "google_tts": 16.0, ...}}`, used for the cost
axis of the H1 Pareto analysis. A lighter alternative, `run_stats.py`, computes the bootstrap CIs,
correlations, Wilcoxon tests, and Pareto membership without the cost back-fill or H3 TOST; pass it
`--calibration results/calibration_anchors.json` explicitly.

**6. Render the figures** (Pareto projections, correlation heatmap, per-provider rankings,
calibration plot, and commercial-vs-open-source gap distribution) into `results/figures/`:

```bash
python scripts/make_figures.py \
  --parquet results/metrics.parquet \
  --calibration results/calibration_anchors.json \
  --stats results/stats.json \
  --out results/figures
```

## Outputs

| Artifact                                   | Contents                                                       |
| ------------------------------------------ | -------------------------------------------------------------- |
| `outputs/audio/<provider>/<task>/*.wav`    | generated speech, 24 kHz mono                                  |
| `outputs/sidecars/<provider>/<task>/*.json`| per-clip metadata: text, model/voice id, latency, reference    |
| `outputs/costs.csv`, `outputs/failures.csv`| per-generation cost log and error log                          |
| `results/metrics.parquet`                  | one row per generated clip with all judge scores               |
| `results/calibration_anchors.json`         | same-speaker and cross-speaker similarity bounds               |
| `results/stats.json`                       | bootstrap CIs, Spearman, Wilcoxon, H3 TOST, Pareto membership  |
| `results/figures/`                         | analysis figures                                               |

## Scaling and parallelism

Scoring is the heaviest stage and supports parallel execution. Split it across workers with
`--shard-index`/`--shard-count`, then recombine with `merge_shards.py`:

```bash
python scripts/compute_metrics.py --out results/shard_0.parquet --shard-index 0 --shard-count 4
# ... shards 1..3 on other workers, then:
python scripts/merge_shards.py --main results/metrics.parquet \
  --shards results/shard_0.parquet results/shard_1.parquet results/shard_2.parquet results/shard_3.parquet
```

To add a single judge to an existing parquet without recomputing the rest (for example backfilling
NISQA), use `--add-judge nisqa`. Individual judges can be skipped with `--skip-utmos`, `--skip-wer`,
or `--skip-sim`.

## Extending

To add a provider, implement the `Provider` protocol in `voice_bench/providers/base.py`
(`tts`, `clone`, and `cleanup`) and register a `ProviderSpec` in the `PROVIDERS` map at the top of
`scripts/generate.py`; that registration is the entire wiring change. New metrics are added as a
module under `voice_bench/metrics/` and hooked into the `JUDGES` tuple in `scripts/compute_metrics.py`.

## License

Code is MIT. LibriTTS-R audio is CC-BY 4.0 and is not redistributed here. The local open-source
engines carry their own model-weight licenses — notably XTTS-v2 under the Coqui Public Model License
(non-commercial) and F5-TTS weights under CC-BY-NC 4.0 — so review them before any non-research use.
