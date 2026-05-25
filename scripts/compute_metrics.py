"""Compute judge-model metrics for generated audio.

Reads outputs/sidecars/**/*.json, runs each metric, writes results/metrics.parquet.

Two modes:
  * Full mode (default): iterate every sidecar, compute every enabled judge,
    skip rows already in parquet (by provider/task/utt_id key).
  * Add-judge mode (--add-judge nisqa): iterate every sidecar where the target
    judge column is MISSING or NaN, compute only that judge, append the new
    columns to the existing parquet row instead of inserting a new row.

Shard mode (--shard-index N --shard-count M): split work across N pods. Each
shard processes only sidecars where hash(provider/task/utt_id) % M == N.

Per-metric failures are caught and recorded as NaN; the run continues.
"""
import argparse
import hashlib
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


JUDGES = ("utmos", "whisper_wer", "wavlm_sim", "ecapa_sim", "nisqa")


def _judge_cols(judge: str) -> tuple[str, ...]:
    """Columns produced by a judge -- used to detect missing rows."""
    if judge == "utmos":
        return ("utmos",)
    if judge == "whisper_wer":
        return ("whisper_wer", "whisper_hyp")
    if judge == "wavlm_sim":
        return ("wavlm_sim",)
    if judge == "ecapa_sim":
        return ("ecapa_sim",)
    if judge == "nisqa":
        return ("nisqa_mos", "nisqa_noi", "nisqa_col", "nisqa_dis", "nisqa_loud")
    raise ValueError(judge)


def _row_key(meta: dict) -> tuple[str, str, str]:
    return (meta["provider"], meta["task"], meta["utt_id"])


def _hash_shard(key: tuple, total_shards: int) -> int:
    h = hashlib.sha256(":".join(key).encode("utf-8")).hexdigest()
    return int(h, 16) % total_shards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecars-root", default="outputs/sidecars")
    ap.add_argument("--out", default="results/metrics.parquet")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--skip-utmos", action="store_true")
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument(
        "--add-judge",
        choices=JUDGES,
        default=None,
        help="Run only this judge on rows where the column is missing.",
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    sidecars_root = root / args.sidecars_root if not Path(args.sidecars_root).is_absolute() else Path(args.sidecars_root)
    out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    sidecars = sorted(sidecars_root.rglob("*.json"))
    if not sidecars:
        print(f"No sidecars found under {sidecars_root}", file=sys.stderr)
        return 1

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        print(f"Loaded existing parquet: {len(existing)} rows")
    else:
        existing = pd.DataFrame()

    add_judge = args.add_judge

    if add_judge:
        # Add-judge mode: find rows in parquet missing the judge column(s), then
        # match those keys back to sidecars to re-compute only that judge.
        judge_cols = _judge_cols(add_judge)
        if not existing.empty and all(c in existing.columns for c in judge_cols):
            need_mask = existing[list(judge_cols)].isna().any(axis=1)
            need_keys = set(zip(
                existing.loc[need_mask, "provider"],
                existing.loc[need_mask, "task"],
                existing.loc[need_mask, "utt_id"],
            ))
        else:
            # Judge column doesn't exist yet -- all rows in parquet need it,
            # plus any sidecar not yet in parquet.
            if not existing.empty:
                need_keys = set(zip(existing.provider, existing.task, existing.utt_id))
            else:
                need_keys = set()
            # Also include sidecars not yet in parquet (cold rows).
            for sc in sidecars:
                m = json.loads(sc.read_text())
                need_keys.add(_row_key(m))
        print(f"Add-judge mode: {add_judge} needed for {len(need_keys)} rows")
    else:
        # Full mode: skip rows already in parquet.
        done = set()
        if not existing.empty:
            done = set(zip(existing.provider, existing.task, existing.utt_id))
        need_keys = None
        print(f"Full mode: {len(done)} rows already in parquet")

    # Build pending list with shard filter.
    pending = []
    for sc in sidecars:
        meta = json.loads(sc.read_text())
        key = _row_key(meta)
        if args.shard_count > 1 and _hash_shard(key, args.shard_count) != args.shard_index:
            continue
        if add_judge:
            if key not in need_keys:
                continue
        else:
            if key in done:
                continue
        pending.append((sc, meta))
    print(f"Shard {args.shard_index}/{args.shard_count}: pending {len(pending)} sidecars")

    if args.limit:
        pending = pending[: args.limit]

    # Lazy import of judges actually needed.
    nisqa_mod = utmos_mod = whisper_mod = wavlm_mod = ecapa_mod = None
    if add_judge == "nisqa" or (not add_judge and not args.skip_utmos and False):
        from voice_bench.metrics import nisqa as nisqa_mod  # noqa: F401
    if not add_judge:
        if not args.skip_utmos:
            from voice_bench.metrics import utmos as utmos_mod  # noqa: F401
        if not args.skip_wer:
            from voice_bench.metrics import whisper_wer as whisper_mod  # noqa: F401
        if not args.skip_sim:
            from voice_bench.metrics import wavlm_sim as wavlm_mod  # noqa: F401
            from voice_bench.metrics import ecapa_sim as ecapa_mod  # noqa: F401
    else:
        # Add-judge mode: only load the requested judge.
        if add_judge == "utmos":
            from voice_bench.metrics import utmos as utmos_mod  # noqa
        elif add_judge == "whisper_wer":
            from voice_bench.metrics import whisper_wer as whisper_mod  # noqa
        elif add_judge == "wavlm_sim":
            from voice_bench.metrics import wavlm_sim as wavlm_mod  # noqa
        elif add_judge == "ecapa_sim":
            from voice_bench.metrics import ecapa_sim as ecapa_mod  # noqa
        elif add_judge == "nisqa":
            from voice_bench.metrics import nisqa as nisqa_mod  # noqa

    ref_wavlm: dict = {}
    ref_ecapa: dict = {}

    new_rows: list[dict] = []
    updated_keys: set = set()

    def flush():
        nonlocal existing, new_rows
        if not new_rows:
            return
        df_new = pd.DataFrame(new_rows)
        if add_judge:
            # Merge into existing: update rows by key, add row if missing.
            existing = _merge_update(existing, df_new)
        else:
            existing = pd.concat([existing, df_new], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(out_path)
        new_rows = []

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
        }
        if not add_judge:
            row.update({
                "model_id": meta.get("model_id"),
                "voice_id": meta.get("voice_id"),
                "character_count": meta.get("character_count"),
                "latency_seconds": meta.get("latency_seconds"),
            })

        try:
            audio16 = load_16k(wav_path) if (not add_judge or add_judge in ("wavlm_sim", "ecapa_sim")) else None
        except Exception as e:
            row["error_load"] = str(e)[:200]
            new_rows.append(row)
            continue

        if add_judge:
            if add_judge == "utmos":
                try:
                    row["utmos"] = utmos_mod.score(wav_path)
                except Exception as e:
                    row["utmos"] = math.nan
                    row["error_utmos"] = str(e)[:200]
            elif add_judge == "nisqa":
                try:
                    row.update(nisqa_mod.score(wav_path))
                except Exception as e:
                    for c in _judge_cols("nisqa"):
                        row[c] = math.nan
                    row["error_nisqa"] = str(e)[:200]
            elif add_judge == "whisper_wer":
                try:
                    hyp = whisper_mod.transcribe(wav_path)
                    row["whisper_hyp"] = hyp
                    row["whisper_wer"] = whisper_mod.wer(target_text, hyp)
                except Exception as e:
                    row["whisper_wer"] = math.nan
                    row["error_wer"] = str(e)[:200]
            elif add_judge in ("wavlm_sim", "ecapa_sim"):
                if task != "cloning" or not ref_path:
                    row[add_judge] = math.nan
                else:
                    try:
                        if add_judge == "wavlm_sim":
                            if speaker_id not in ref_wavlm:
                                ref_wavlm[speaker_id] = wavlm_mod.embed(load_16k(ref_path))
                            row["wavlm_sim"] = wavlm_mod.cosine(ref_wavlm[speaker_id], wavlm_mod.embed(audio16))
                        else:
                            if speaker_id not in ref_ecapa:
                                ref_ecapa[speaker_id] = ecapa_mod.embed(load_16k(ref_path))
                            row["ecapa_sim"] = ecapa_mod.cosine(ref_ecapa[speaker_id], ecapa_mod.embed(audio16))
                    except Exception as e:
                        row[add_judge] = math.nan
                        row[f"error_{add_judge.split('_')[0]}"] = str(e)[:200]
            new_rows.append(row)
            updated_keys.add(_row_key(meta))
        else:
            # Full mode.
            if not args.skip_utmos:
                try:
                    row["utmos"] = utmos_mod.score(wav_path)
                except Exception as e:
                    row["utmos"] = math.nan
                    row["error_utmos"] = str(e)[:200]
            if not args.skip_wer:
                try:
                    hyp = whisper_mod.transcribe(wav_path)
                    row["whisper_hyp"] = hyp
                    row["whisper_wer"] = whisper_mod.wer(target_text, hyp)
                except Exception as e:
                    row["whisper_wer"] = math.nan
                    row["error_wer"] = str(e)[:200]
            if not args.skip_sim and task == "cloning" and ref_path:
                try:
                    if speaker_id not in ref_wavlm:
                        ref_audio = load_16k(ref_path)
                        ref_wavlm[speaker_id] = wavlm_mod.embed(ref_audio)
                        ref_ecapa[speaker_id] = ecapa_mod.embed(ref_audio)
                    gen_w = wavlm_mod.embed(audio16)
                    row["wavlm_sim"] = wavlm_mod.cosine(ref_wavlm[speaker_id], gen_w)
                    gen_e = ecapa_mod.embed(audio16)
                    row["ecapa_sim"] = ecapa_mod.cosine(ref_ecapa[speaker_id], gen_e)
                except Exception as e:
                    row["wavlm_sim"] = math.nan
                    row["ecapa_sim"] = math.nan
                    row["error_sim"] = str(e)[:200]
            new_rows.append(row)

        if len(new_rows) >= args.save_every:
            flush()

    flush()

    final = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    print(f"\nDone. {len(final)} total rows in {out_path}")
    if not final.empty:
        cols = [c for c in ("utmos", "nisqa_mos", "whisper_wer", "wavlm_sim", "ecapa_sim") if c in final.columns]
        if cols:
            print("\nSummary by (provider, task):")
            print(final.groupby(["provider", "task"])[cols].agg(["mean", "std", "count"]))
    return 0


def _merge_update(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Update existing rows by (provider, task, utt_id) key; insert new rows."""
    if existing.empty:
        return new_df
    key_cols = ["provider", "task", "utt_id"]
    # Set index for vectorised join.
    e = existing.set_index(key_cols)
    n = new_df.set_index(key_cols)
    # For columns present in n, overwrite values in e where rows match.
    for col in n.columns:
        if col in e.columns:
            e[col] = e[col].astype("object")
        else:
            e[col] = pd.Series(index=e.index, dtype="object")
        # Use index intersection: rows in both update; rows only in n append.
        common = e.index.intersection(n.index)
        e.loc[common, col] = n.loc[common, col].values
    only_new = n.index.difference(e.index)
    if len(only_new):
        new_subset = n.loc[only_new].reset_index()
        # Align column set.
        for col in e.columns:
            if col not in new_subset.columns:
                new_subset[col] = pd.NA
        e_reset = e.reset_index()
        return pd.concat([e_reset, new_subset[e_reset.columns]], ignore_index=True)
    return e.reset_index()


if __name__ == "__main__":
    raise SystemExit(main())
