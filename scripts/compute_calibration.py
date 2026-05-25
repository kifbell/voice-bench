"""Calibration anchors for speaker similarity metrics.

Computes upper (same-speaker) and lower (different-speaker) bounds for WavLM
and ECAPA-TDNN cosine similarity, using only GT LibriTTS-R WAV files from the
manifest. This lets us interpret absolute cloning-task similarity scores:

  cloning_sim ~ upper_p50  -> "indistinguishable from real speaker"
  cloning_sim ~ lower_mean -> "random different person"

Upper anchor (same-speaker):
  Per speaker, compute similarity between reference WAV and N random target
  WAVs from the same speaker in the manifest.

Lower anchor (cross-speaker):
  For all (speaker_A, speaker_B) pairs in the manifest, similarity between
  their reference WAVs.

Writes results/calibration_anchors.json with:
  {
    "wavlm": {"upper_p50": ..., "upper_p05": ..., "lower_mean": ..., "lower_p95": ...,
              "upper_samples": [...], "lower_samples": [...]},
    "ecapa": {...},
  }
"""
import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/samples.json")
    ap.add_argument("--out", default="results/calibration_anchors.json")
    ap.add_argument(
        "--device",
        default=os.environ.get("VOICEBENCH_DEVICE", "cpu"),
        help="Computation device for the encoders.",
    )
    ap.add_argument(
        "--targets-per-speaker",
        type=int,
        default=5,
        help="How many target WAVs to compare against reference per speaker.",
    )
    args = ap.parse_args()

    os.environ["VOICEBENCH_DEVICE"] = args.device

    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / args.manifest).read_text())
    speakers = manifest["speakers"]

    from voice_bench.metrics import wavlm_sim, ecapa_sim
    from voice_bench.metrics.audio_io import load_16k

    cache_wavlm: dict[str, np.ndarray] = {}
    cache_ecapa: dict[str, np.ndarray] = {}

    def embed(wav_path: str) -> tuple[np.ndarray, np.ndarray]:
        if wav_path in cache_wavlm:
            return cache_wavlm[wav_path], cache_ecapa[wav_path]
        audio = load_16k(wav_path)
        w = wavlm_sim.embed(audio)
        e = ecapa_sim.embed(audio)
        cache_wavlm[wav_path] = w
        cache_ecapa[wav_path] = e
        return w, e

    upper_wavlm: list[float] = []
    upper_ecapa: list[float] = []
    print(f"Computing upper bound (same-speaker) across {len(speakers)} speakers ...")
    for spk in speakers:
        ref = spk["reference"]["wav_path"]
        ref_w, ref_e = embed(ref)
        targets = spk["targets"][: args.targets_per_speaker]
        for tgt in targets:
            t_w, t_e = embed(tgt["wav_path"])
            upper_wavlm.append(wavlm_sim.cosine(ref_w, t_w))
            upper_ecapa.append(ecapa_sim.cosine(ref_e, t_e))

    print(f"Computing lower bound (cross-speaker) across "
          f"{len(speakers) * (len(speakers) - 1) // 2} speaker pairs ...")
    lower_wavlm: list[float] = []
    lower_ecapa: list[float] = []
    for a, b in combinations(speakers, 2):
        a_w, a_e = embed(a["reference"]["wav_path"])
        b_w, b_e = embed(b["reference"]["wav_path"])
        lower_wavlm.append(wavlm_sim.cosine(a_w, b_w))
        lower_ecapa.append(ecapa_sim.cosine(a_e, b_e))

    def summarise(name: str, upper: list[float], lower: list[float]) -> dict:
        u = np.asarray(upper)
        l_ = np.asarray(lower)
        return {
            "metric": name,
            "upper_n": int(u.size),
            "upper_p50": float(np.percentile(u, 50)),
            "upper_p05": float(np.percentile(u, 5)),
            "upper_p95": float(np.percentile(u, 95)),
            "upper_mean": float(u.mean()),
            "upper_std": float(u.std(ddof=1)),
            "lower_n": int(l_.size),
            "lower_p50": float(np.percentile(l_, 50)),
            "lower_p05": float(np.percentile(l_, 5)),
            "lower_p95": float(np.percentile(l_, 95)),
            "lower_mean": float(l_.mean()),
            "lower_std": float(l_.std(ddof=1)),
            "upper_samples": [round(x, 4) for x in upper],
            "lower_samples": [round(x, 4) for x in lower],
        }

    result = {
        "manifest_path": str(args.manifest),
        "speakers": [s["speaker_id"] for s in speakers],
        "n_speakers": len(speakers),
        "targets_per_speaker": args.targets_per_speaker,
        "device": args.device,
        "wavlm": summarise("wavlm_sim", upper_wavlm, lower_wavlm),
        "ecapa": summarise("ecapa_sim", upper_ecapa, upper_ecapa) if False else summarise("ecapa_sim", upper_ecapa, lower_ecapa),
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")
    print("\nSummary:")
    for m in ("wavlm", "ecapa"):
        r = result[m]
        print(f"  {r['metric']}: "
              f"upper p50={r['upper_p50']:.3f} (p05={r['upper_p05']:.3f}), "
              f"lower mean={r['lower_mean']:.3f} (p95={r['lower_p95']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
