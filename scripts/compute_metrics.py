"""Compute judge-model metrics for all generated audio.

Reads outputs/sidecars/**/*.json, runs each metric, writes results/metrics.parquet.
Resumable: skips rows already in parquet.

Per-metric failures are caught and recorded as NaN; the run continues. This makes
the pipeline robust to occasional bad files without losing all completed work.
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_bench.metrics.audio_io import load_16k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecars-root", default="outputs/sidecars")
    ap.add_argument("--out", default="results/metrics.parquet")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--skip-utmos", action="store_true")
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N rows (debug)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    sidecars_root = root / args.sidecars_root
    out_path = root / args.out

    sidecars = sorted(sidecars_root.rglob("*.json"))
    if not sidecars:
        print(f"No sidecars found under {sidecars_root}", file=sys.stderr)
        return 1

    # Resumability: load existing parquet, build skip-set keyed by (provider, task, utt_id).
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(zip(existing.provider, existing.task, existing.utt_id))
        print(f"Resuming: {len(done)} rows already in {out_path}")
    else:
        existing = pd.DataFrame()
        done = set()

    # Lazy imports of metric modules so we don't load models if disabled.
    if not args.skip_utmos:
        from voice_bench.metrics import utmos
    if not args.skip_wer:
        from voice_bench.metrics import whisper_wer
    if not args.skip_sim:
        from voice_bench.metrics import wavlm_sim, ecapa_sim

    # Reference embedding caches, keyed by speaker_id (within the cloning task).
    ref_wavlm: dict[str, "np.ndarray"] = {}
    ref_ecapa: dict[str, "np.ndarray"] = {}

    rows: list[dict] = []
    pending = [sc for sc in sidecars if (
        (json.loads(sc.read_text())["provider"],
         json.loads(sc.read_text())["task"],
         json.loads(sc.read_text())["utt_id"]) not in done
    )]

    # Cheaper variant: parse once
    pending = []
    for sc in sidecars:
        meta = json.loads(sc.read_text())
        if (meta["provider"], meta["task"], meta["utt_id"]) not in done:
            pending.append((sc, meta))
    print(f"To process: {len(pending)} / {len(sidecars)}")

    if args.limit:
        pending = pending[: args.limit]

    for sc_path, meta in tqdm(pending, desc="metrics"):
        wav_path = meta["wav_path"]
        target_text = meta["text"]
        speaker_id = meta["speaker_id"]
        task = meta["task"]
        ref_path = meta.get("reference_wav_path")

        row = {
            "provider": meta["provider"],
            "task": task,
            "speaker_id": speaker_id,
            "utt_id": meta["utt_id"],
            "model_id": meta.get("model_id"),
            "voice_id": meta.get("voice_id"),
            "character_count": meta.get("character_count"),
            "latency_seconds": meta.get("latency_seconds"),
        }

        try:
            audio16 = load_16k(wav_path)
        except Exception as e:  # pragma: no cover
            row["error_load"] = str(e)[:200]
            rows.append(row)
            continue

        if not args.skip_utmos:
            try:
                row["utmos"] = utmos.score(wav_path)
            except Exception as e:
                row["utmos"] = math.nan
                row["error_utmos"] = str(e)[:200]

        if not args.skip_wer:
            try:
                hyp = whisper_wer.transcribe(wav_path)
                row["whisper_hyp"] = hyp
                row["whisper_wer"] = whisper_wer.wer(target_text, hyp)
            except Exception as e:
                row["whisper_wer"] = math.nan
                row["error_wer"] = str(e)[:200]

        if not args.skip_sim and task == "cloning" and ref_path:
            try:
                if speaker_id not in ref_wavlm:
                    ref_audio = load_16k(ref_path)
                    ref_wavlm[speaker_id] = wavlm_sim.embed(ref_audio)
                    ref_ecapa[speaker_id] = ecapa_sim.embed(ref_audio)
                gen_w = wavlm_sim.embed(audio16)
                row["wavlm_sim"] = wavlm_sim.cosine(ref_wavlm[speaker_id], gen_w)
                gen_e = ecapa_sim.embed(audio16)
                row["ecapa_sim"] = ecapa_sim.cosine(ref_ecapa[speaker_id], gen_e)
            except Exception as e:
                row["wavlm_sim"] = math.nan
                row["ecapa_sim"] = math.nan
                row["error_sim"] = str(e)[:200]

        rows.append(row)

        if len(rows) >= args.save_every:
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            existing.to_parquet(out_path)
            rows = []

    if rows:
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(out_path)

    final = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    print(f"\nDone. {len(final)} total rows in {out_path}")
    if not final.empty:
        cols = [c for c in ("utmos", "whisper_wer", "wavlm_sim", "ecapa_sim") if c in final.columns]
        if cols:
            print("\nSummary by (provider, task):")
            print(final.groupby(["provider", "task"])[cols].agg(["mean", "std", "count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
