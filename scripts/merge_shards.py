"""Merge sharded metrics parquets back into the main parquet.

Each shard's parquet has the (provider, task, utt_id) key plus only the columns
its --add-judge produced. Merge updates the main parquet by:
  * For each row in shard: find matching row in main by key, set the new columns.
  * If the key isn't in main, append a new row.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

KEY = ["provider", "task", "utt_id"]


def merge_in(main: pd.DataFrame, shard: pd.DataFrame) -> pd.DataFrame:
    """Update main with shard's column values where keys match; append missing."""
    if main.empty:
        return shard
    new_cols = [c for c in shard.columns if c not in KEY]
    for c in new_cols:
        if c not in main.columns:
            main[c] = pd.NA
    main = main.set_index(KEY)
    shard = shard.set_index(KEY)
    common = main.index.intersection(shard.index)
    for c in new_cols:
        main.loc[common, c] = shard.loc[common, c].values
    new_only = shard.index.difference(main.index)
    if len(new_only):
        # Build rows aligned to main's columns.
        appended = shard.loc[new_only].reset_index()
        for c in main.reset_index().columns:
            if c not in appended.columns:
                appended[c] = pd.NA
        main = pd.concat(
            [main.reset_index(), appended[main.reset_index().columns]],
            ignore_index=True,
        )
        return main
    return main.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="Path to main metrics parquet (in-place updated)")
    ap.add_argument("--shards", nargs="+", required=True, help="Paths to shard parquets")
    ap.add_argument("--backup", default=None, help="If given, write a backup of main here first")
    args = ap.parse_args()

    main_path = Path(args.main)
    if not main_path.exists():
        print(f"main parquet missing: {main_path}", file=sys.stderr)
        return 1
    main = pd.read_parquet(main_path)
    print(f"loaded main: {len(main)} rows, cols: {list(main.columns)}")

    if args.backup:
        backup_path = Path(args.backup)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        main.to_parquet(backup_path)
        print(f"backup -> {backup_path}")

    for sp in args.shards:
        sdf = pd.read_parquet(sp)
        # Drop columns we never want to copy back.
        drop = [c for c in ("model_id", "voice_id", "character_count", "latency_seconds") if c in sdf.columns]
        if drop:
            sdf = sdf.drop(columns=drop)
        print(f"merging shard {sp}: {len(sdf)} rows, new cols: {[c for c in sdf.columns if c not in KEY]}")
        main = merge_in(main, sdf)

    main.to_parquet(main_path)
    print(f"wrote {len(main)} rows -> {main_path}")
    print(f"final columns: {list(main.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
