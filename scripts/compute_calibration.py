"""Compute speaker-similarity calibration anchors from GT (LibriTTS-R).

For each speaker in the manifest, sample multiple utterance pairs; compute
WavLM and ECAPA cosine. Aggregate:
  - upper: same-speaker similarity (what 'perfect cloning' would approach)
  - lower: cross-speaker similarity (chance level)

Writes results/calibration.json. Without these anchors, raw similarity
numbers in metrics.parquet are uninterpretable.
"""
import argparse
import json
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_bench.dataset import load_manifest
from voice_bench.metrics.audio_io import load_16k
from voice_bench.metrics import wavlm_sim, ecapa_sim


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/samples.json")
    ap.add_argument("--out", default="results/calibration.json")
    ap.add_argument("--same-pairs-per-speaker", type=int, default=10,
                    help="Number of (utt_i, utt_j) pairs to sample within each speaker.")
    ap.add_argument("--cross-pairs", type=int, default=200,
                    help="Number of (speaker_a, speaker_b) cross-speaker pairs to sample.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(__file__).resolve().parent.parent
    manifest = load_manifest(root / args.manifest)
    out_path = root / args.out

    # Build per-speaker list of utterance WAV paths (reference + targets).
    per_speaker: dict[str, list[Path]] = {}
    for spk in manifest["speakers"]:
        utts = [Path(spk["reference"]["wav_path"])] + [Path(t["wav_path"]) for t in spk["targets"]]
        per_speaker[spk["speaker_id"]] = utts

    # Pre-embed everything we'll need. Each speaker contributes ≤ same_pairs*2 utterances
    # for the within-speaker anchor, plus one utterance reused for cross-speaker.
    # In practice we embed every utterance we touch and cache by path.
    cache_w: dict[Path, np.ndarray] = {}
    cache_e: dict[Path, np.ndarray] = {}

    def emb(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if path not in cache_w:
            audio = load_16k(path)
            cache_w[path] = wavlm_sim.embed(audio)
            cache_e[path] = ecapa_sim.embed(audio)
        return cache_w[path], cache_e[path]

    # ---- Upper anchor: same-speaker pairs ----
    same_wavlm, same_ecapa = [], []
    print(f"Same-speaker pairs ({args.same_pairs_per_speaker} per speaker × {len(per_speaker)} speakers):")
    for spk_id, utts in tqdm(per_speaker.items()):
        all_pairs = list(combinations(range(len(utts)), 2))
        rng.shuffle(all_pairs)
        for i, j in all_pairs[: args.same_pairs_per_speaker]:
            w_i, e_i = emb(utts[i])
            w_j, e_j = emb(utts[j])
            same_wavlm.append(wavlm_sim.cosine(w_i, w_j))
            same_ecapa.append(ecapa_sim.cosine(e_i, e_j))

    # ---- Lower anchor: cross-speaker pairs ----
    cross_wavlm, cross_ecapa = [], []
    speakers = list(per_speaker.keys())
    print(f"\nCross-speaker pairs ({args.cross_pairs}):")
    for _ in tqdm(range(args.cross_pairs)):
        a, b = rng.sample(speakers, 2)
        ua = rng.choice(per_speaker[a])
        ub = rng.choice(per_speaker[b])
        w_a, e_a = emb(ua)
        w_b, e_b = emb(ub)
        cross_wavlm.append(wavlm_sim.cosine(w_a, w_b))
        cross_ecapa.append(ecapa_sim.cosine(e_a, e_b))

    calibration = {
        "manifest": str(out_path.parent / "..").replace("..", args.manifest),
        "seed": args.seed,
        "same_pairs_per_speaker": args.same_pairs_per_speaker,
        "cross_pairs": args.cross_pairs,
        "n_speakers": len(per_speaker),
        "wavlm": {
            "same_mean": float(np.mean(same_wavlm)),
            "same_p05": _pct(same_wavlm, 5),
            "same_p50": _pct(same_wavlm, 50),
            "same_p95": _pct(same_wavlm, 95),
            "cross_mean": float(np.mean(cross_wavlm)),
            "cross_p05": _pct(cross_wavlm, 5),
            "cross_p50": _pct(cross_wavlm, 50),
            "cross_p95": _pct(cross_wavlm, 95),
            "n_same": len(same_wavlm),
            "n_cross": len(cross_wavlm),
        },
        "ecapa": {
            "same_mean": float(np.mean(same_ecapa)),
            "same_p05": _pct(same_ecapa, 5),
            "same_p50": _pct(same_ecapa, 50),
            "same_p95": _pct(same_ecapa, 95),
            "cross_mean": float(np.mean(cross_ecapa)),
            "cross_p05": _pct(cross_ecapa, 5),
            "cross_p50": _pct(cross_ecapa, 50),
            "cross_p95": _pct(cross_ecapa, 95),
            "n_same": len(same_ecapa),
            "n_cross": len(cross_ecapa),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, indent=2))
    print(f"\nWritten to {out_path}")
    print(f"  WavLM  upper (same-speaker)   mean={calibration['wavlm']['same_mean']:.3f}  p05={calibration['wavlm']['same_p05']:.3f}")
    print(f"  WavLM  lower (cross-speaker)  mean={calibration['wavlm']['cross_mean']:.3f}  p95={calibration['wavlm']['cross_p95']:.3f}")
    print(f"  ECAPA  upper (same-speaker)   mean={calibration['ecapa']['same_mean']:.3f}  p05={calibration['ecapa']['same_p05']:.3f}")
    print(f"  ECAPA  lower (cross-speaker)  mean={calibration['ecapa']['cross_mean']:.3f}  p95={calibration['ecapa']['cross_p95']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
