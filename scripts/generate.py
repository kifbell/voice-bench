"""Generate audio for the experiment manifest using ElevenLabs.

For each speaker, iterate over targets:
  - tts: synthesize with a fixed preset voice
  - cloning: create IVC voice from speaker's reference, synthesize, cache voice
After each speaker, delete the IVC voice (1-slot-at-a-time).

Idempotent: any (provider, task, utt_id) already saved to disk is skipped.

Phase 3 pilot: --pilot N restricts to first N (speaker, target) pairs.
"""
import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_bench.dataset import load_manifest
from voice_bench.providers.elevenlabs import ElevenLabsProvider
from voice_bench.runner import GenerationContext, is_done, save_generation


ELEVEN_PRESET_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel: neutral female
ELEVEN_MODEL = "eleven_multilingual_v2"
SEED = 42


def _append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/samples.json")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="Limit to first N speakers from manifest")
    ap.add_argument("--max-targets-per-speaker", type=int, default=None,
                    help="Limit to first M targets per speaker")
    ap.add_argument("--pilot", action="store_true",
                    help="Shortcut: --max-speakers 5 --max-targets-per-speaker 1")
    ap.add_argument("--tasks", nargs="+", default=["tts", "cloning"],
                    choices=["tts", "cloning"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY missing", file=sys.stderr)
        return 1

    manifest = load_manifest(root / args.manifest)
    out_root = root / args.out_root

    max_speakers = args.max_speakers
    max_targets = args.max_targets_per_speaker
    if args.pilot:
        max_speakers = max_speakers or 5
        max_targets = max_targets or 1

    speakers = manifest["speakers"]
    if max_speakers is not None:
        speakers = speakers[:max_speakers]

    flat: list[tuple[dict, dict]] = []
    for spk in speakers:
        targets = spk["targets"]
        if max_targets is not None:
            targets = targets[:max_targets]
        for tgt in targets:
            flat.append((spk, tgt))

    n_speakers = len({s["speaker_id"] for s, _ in flat})
    print(f"Plan: {n_speakers} speakers × ~{len(flat) // max(n_speakers, 1)} targets = "
          f"{len(flat)} pairs × {len(args.tasks)} tasks = {len(flat) * len(args.tasks)} generations\n")

    provider = ElevenLabsProvider(api_key=api_key)

    costs_csv = out_root / "costs.csv"
    failures_csv = out_root / "failures.csv"
    n_new = n_cached = n_failed = 0
    prev_speaker = None

    try:
        for spk, tgt in flat:
            speaker_id = spk["speaker_id"]
            utt_id = tgt["utt_id"]
            text = tgt["text"]
            ref_path = Path(spk["reference"]["wav_path"])
            ctx = GenerationContext(utt_id=utt_id, speaker_id=speaker_id, text=text)

            # When we move to a new speaker, free the previous one's IVC slot.
            if prev_speaker is not None and prev_speaker != speaker_id:
                # the previous speaker's reference path is implicit in the cache;
                # cleanup the cache entry corresponding to prev_speaker's ref
                prev_ref = next(
                    (s["reference"]["wav_path"] for s in manifest["speakers"]
                     if s["speaker_id"] == prev_speaker),
                    None,
                )
                if prev_ref:
                    if provider.cleanup_clone_voice(Path(prev_ref)):
                        print(f"  [cleanup] freed IVC slot for speaker {prev_speaker}")
            prev_speaker = speaker_id

            for task in args.tasks:
                if is_done(out_root, provider.name, task, utt_id):
                    n_cached += 1
                    continue

                try:
                    if task == "tts":
                        result = provider.tts(
                            text, voice_id=ELEVEN_PRESET_VOICE,
                            model_id=ELEVEN_MODEL, seed=SEED,
                        )
                    else:
                        result = provider.clone(
                            text, reference_wav_path=ref_path,
                            model_id=ELEVEN_MODEL, seed=SEED,
                        )
                    save_generation(result, ctx, out_root)
                    _append_csv(costs_csv, {
                        "provider": result.provider,
                        "task": result.task,
                        "speaker_id": speaker_id,
                        "utt_id": utt_id,
                        "character_count": result.character_count,
                        "latency_seconds": f"{result.latency_seconds:.3f}",
                        "model_id": result.model_id,
                        "voice_id": result.voice_id,
                    }, fieldnames=["provider", "task", "speaker_id", "utt_id",
                                   "character_count", "latency_seconds", "model_id", "voice_id"])
                    n_new += 1
                    print(f"  [{n_new + n_cached:4d}] {task:8s} spk={speaker_id:>5s} {utt_id} "
                          f"chars={result.character_count:3d} latency={result.latency_seconds:.2f}s")
                except Exception as e:
                    n_failed += 1
                    _append_csv(failures_csv, {
                        "provider": provider.name,
                        "task": task,
                        "speaker_id": speaker_id,
                        "utt_id": utt_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:500],
                    }, fieldnames=["provider", "task", "speaker_id", "utt_id",
                                   "error_type", "error_message"])
                    print(f"  FAILED {task} spk={speaker_id} {utt_id}: {type(e).__name__}: {str(e)[:200]}")
    finally:
        provider.cleanup()  # belt-and-suspenders: nuke any remaining IVC voices

    print(f"\nDone. new={n_new}, cached={n_cached}, failed={n_failed}")
    if args.pilot:
        print("\n=== PILOT CHECKPOINT ===")
        print("Inspect WAVs in outputs/audio/elevenlabs/{tts,cloning}/")
        print("Inspect sidecars in outputs/sidecars/elevenlabs/{tts,cloning}/")
        print("If satisfied, re-run without --pilot to generate the full manifest.")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
