# voice-bench

Comparative evaluation of commercial and open-source TTS / voice-cloning APIs on the LibriTTS-R `test-clean` benchmark.

Companion code for an in-progress HSE DSBA thesis. Implements the methodology described in `research_plan.md`: generate audio through each provider for the same texts and reference voices, then score the outputs with standard automated judges (UTMOSv2, NISQA, Whisper-WER, WavLM-XVector, ECAPA-TDNN) plus calibration anchors from ground-truth pairs.

## Status

- Phase 1 (dataset + manifest) — done
- Phase 2 (ElevenLabs provider, TTS + Instant Voice Cloning) — done
- Phase 3 (pilot generation, 10 files) — done
- Phase 5–8 (metrics, calibration, stats, figures) — pipeline written and validated on pilot
- Phase 4 (full generation, 800 files) — gated by API budget decisions
- OpenAI / Replicate providers — pending API access

## Layout

```
voice_bench/
  providers/      # provider integrations (base.py, elevenlabs.py)
  metrics/        # judge models (utmos, whisper_wer, wavlm_sim, ecapa_sim)
  analysis/       # (statpack scripts live under scripts/)
  dataset.py      # LibriTTS-R scan + manifest builder
  runner.py       # save WAV + JSON sidecar, idempotent
scripts/
  build_manifest.py     # → data/samples.json
  generate.py           # provider generation, --pilot for smoke
  compute_metrics.py    # → results/metrics.parquet
  compute_calibration.py # → results/calibration.json
  run_stats.py          # → results/stats.json
  make_figures.py       # → results/figures/*.png
research_plan.md         # full methodology + hypothesis spec
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/sarulab-speech/UTMOSv2.git

# Download LibriTTS-R test-clean (~1.4 GB extracted)
mkdir -p data && cd data
curl -O https://openslr.elda.org/resources/141/test_clean.tar.gz
tar -xzf test_clean.tar.gz && mv LibriTTS_R ../  # repo expects ./LibriTTS_R/test-clean
cd ..

# API keys
cp .env.example .env  # fill in ELEVENLABS_API_KEY (and OPENAI_API_KEY, REPLICATE_API_TOKEN when ready)

# Pipeline
python scripts/build_manifest.py
python scripts/generate.py --pilot       # smoke (10 files)
python scripts/generate.py               # full (800 files, ~30 min wall-clock + char budget)
python scripts/compute_calibration.py
python scripts/compute_metrics.py
python scripts/run_stats.py
python scripts/make_figures.py
```

## Hypotheses tested

- **H1.** Providers occupy distinct points on the Pareto frontier in (naturalness, speaker-similarity, cost) — no single dominant API.
- **H2.** Naturalness predictors (UTMOSv2 ± NISQA) agree across systems; speaker-similarity metrics (WavLM-XVector vs ECAPA-TDNN) diverge.
- **H3.** On per-(task, metric) TOST tests with pre-registered equivalence margins δ, commercial and open-source provider groups are statistically equivalent in mean (no group has an advantage > δ).

## License

Code: MIT. LibriTTS-R audio is CC-BY 4.0 (not redistributed in this repo).

