import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_bench.dataset import build_manifest, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="LibriTTS_R/test-clean")
    ap.add_argument("--speakers-txt", default="data/SPEAKERS.txt")
    ap.add_argument("--out", default="data/samples.json")
    ap.add_argument("--n-speakers", type=int, default=20)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gender-balance", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    manifest = build_manifest(
        dataset_root=root / args.dataset_root,
        speakers_txt=root / args.speakers_txt,
        n_speakers=args.n_speakers,
        n_targets=args.n_targets,
        seed=args.seed,
        gender_balance=not args.no_gender_balance,
    )

    out = root / args.out
    write_manifest(manifest, out)

    n_selected = manifest["n_speakers_selected"]
    n_eligible = manifest["n_eligible_speakers"]
    n_targets = manifest["n_targets_per_speaker"]
    pairs = n_selected * n_targets
    avg_chars = sum(
        sum(len(t["text"]) for t in spk["targets"]) / max(len(spk["targets"]), 1)
        for spk in manifest["speakers"]
    ) / max(n_selected, 1)
    total_chars = sum(sum(len(t["text"]) for t in spk["targets"]) for spk in manifest["speakers"])

    print(f"Manifest written: {out}")
    print(f"  Eligible speakers: {n_eligible}")
    print(f"  Selected:         {n_selected} (requested {args.n_speakers})")
    print(f"  Gender-balanced:  {manifest['gender_balanced']}")
    print(f"  Pairs:            {pairs} ({n_selected} × {n_targets})")
    print(f"  Avg target chars: {avg_chars:.1f}")
    print(f"  Total chars TTS:  {total_chars} (×1 task)")
    print(f"  Total chars BOTH: {total_chars * 2} (TTS + cloning)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
