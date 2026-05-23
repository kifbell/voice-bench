"""Generate audio for the experiment manifest, for a single provider.

For each speaker, iterate over targets:
  - tts: synthesize with a provider-default voice (no per-speaker conditioning)
  - cloning: condition on the speaker's reference WAV (and transcript for F5-TTS)

Idempotent: any (provider, task, utt_id) already saved to disk is skipped.

Pilot mode (--pilot N) restricts the run to the first N speakers x 1 target each,
for the research_plan CHECKPOINT step.
"""
import argparse
import csv
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_bench.dataset import load_manifest
from voice_bench.runner import GenerationContext, is_done, save_generation


# Provider-specific defaults. Adding a new provider here is the entire wiring change.
@dataclass(frozen=True)
class ProviderSpec:
    name: str
    default_voice_id: str
    default_model_id: str
    needs_api_key: str | None  # env var name, or None for in-process providers
    needs_reference_text: bool  # F5-TTS needs the reference transcript at clone()
    has_clone_voice_slot: bool  # ElevenLabs IVC has a 1-slot quota that needs cleanup


PROVIDERS: dict[str, ProviderSpec] = {
    "elevenlabs": ProviderSpec(
        name="elevenlabs",
        default_voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel: neutral female
        default_model_id="eleven_multilingual_v2",
        needs_api_key="ELEVENLABS_API_KEY",
        needs_reference_text=False,
        has_clone_voice_slot=True,
    ),
    "xtts_v2": ProviderSpec(
        name="xtts_v2",
        default_voice_id="Claribel Dervla",  # bundled XTTS studio speaker
        default_model_id="xtts_v2",
        needs_api_key=None,
        needs_reference_text=False,
        has_clone_voice_slot=False,
    ),
    "f5_tts": ProviderSpec(
        name="f5_tts",
        default_voice_id="default_ref_en",  # picked from manifest at runtime
        default_model_id="F5TTS_v1_Base",
        needs_api_key=None,
        needs_reference_text=True,
        has_clone_voice_slot=False,
    ),
    "cosyvoice2": ProviderSpec(
        name="cosyvoice2",
        default_voice_id="default_ref_en",
        default_model_id="CosyVoice2-0.5B",
        needs_api_key=None,
        needs_reference_text=True,
        has_clone_voice_slot=False,
    ),
    "fish_speech_s1": ProviderSpec(
        name="fish_speech_s1",
        default_voice_id="default_ref_en",
        default_model_id="openaudio-s1-mini",
        needs_api_key=None,
        needs_reference_text=True,
        has_clone_voice_slot=False,
    ),
    "fish_speech_s2_pro": ProviderSpec(
        name="fish_speech_s2_pro",
        default_voice_id="default",
        default_model_id="speech-s2-pro",
        needs_api_key="FISH_AUDIO_API_KEY",
        needs_reference_text=False,
        has_clone_voice_slot=False,
    ),
}


SEED = 42


def _append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _build_provider(spec: ProviderSpec, manifest: dict, api_key: str | None):
    """Instantiate the provider class for ``spec``. Imports lazily so missing optional
    deps (coqui-tts, f5-tts) don't break the other providers."""
    if spec.name == "elevenlabs":
        from voice_bench.providers.elevenlabs import ElevenLabsProvider
        if api_key is None:
            raise RuntimeError("ELEVENLABS_API_KEY missing")
        return ElevenLabsProvider(api_key=api_key)
    if spec.name == "xtts_v2":
        from voice_bench.providers.xtts_v2 import XttsV2Provider
        return XttsV2Provider()
    if spec.name == "f5_tts":
        from voice_bench.providers.f5_tts import F5TtsProvider
        # The "default voice" for F5's speaker-free TTS task: pick the very first
        # speaker's reference clip from the manifest, treat its transcript as the
        # default reference text. Documented in README.
        first = manifest["speakers"][0]
        ref = first["reference"]
        return F5TtsProvider(
            default_ref_wav=Path(ref["wav_path"]),
            default_ref_text=ref["text"],
        )
    if spec.name == "cosyvoice2":
        from voice_bench.providers.cosyvoice2 import CosyVoice2Provider
        first = manifest["speakers"][0]
        ref = first["reference"]
        return CosyVoice2Provider(
            default_ref_wav=Path(ref["wav_path"]),
            default_ref_text=ref["text"],
        )
    if spec.name == "fish_speech_s1":
        from voice_bench.providers.fish_speech_s1 import FishSpeechS1Provider
        first = manifest["speakers"][0]
        ref = first["reference"]
        return FishSpeechS1Provider(
            default_ref_wav=Path(ref["wav_path"]),
            default_ref_text=ref["text"],
        )
    if spec.name == "fish_speech_s2_pro":
        from voice_bench.providers.fish_speech_s2_pro import FishSpeechS2ProProvider
        if api_key is None:
            raise RuntimeError("FISH_AUDIO_API_KEY missing")
        return FishSpeechS2ProProvider(api_key=api_key)
    raise ValueError(f"Unknown provider: {spec.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PROVIDERS.keys()), required=True)
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

    spec = PROVIDERS[args.provider]
    api_key = os.environ.get(spec.needs_api_key) if spec.needs_api_key else None
    if spec.needs_api_key and not api_key:
        print(f"{spec.needs_api_key} missing", file=sys.stderr)
        return 1

    manifest = load_manifest(root / args.manifest)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = root / out_root

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
    print(f"Provider: {spec.name}")
    print(f"Plan: {n_speakers} speakers × ~{len(flat) // max(n_speakers, 1)} targets = "
          f"{len(flat)} pairs × {len(args.tasks)} tasks = {len(flat) * len(args.tasks)} generations")
    print(f"Output root: {out_root}\n")

    provider = _build_provider(spec, manifest, api_key)

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
            ref_text = spk["reference"].get("text") if spec.needs_reference_text else None
            ctx = GenerationContext(utt_id=utt_id, speaker_id=speaker_id, text=text)

            # Providers with a single-voice-slot clone API (ElevenLabs IVC) free the
            # previous speaker's slot when we move on.
            if spec.has_clone_voice_slot and prev_speaker is not None and prev_speaker != speaker_id:
                prev_ref = next(
                    (s["reference"]["wav_path"] for s in manifest["speakers"]
                     if s["speaker_id"] == prev_speaker),
                    None,
                )
                if prev_ref and hasattr(provider, "cleanup_clone_voice"):
                    if provider.cleanup_clone_voice(Path(prev_ref)):
                        print(f"  [cleanup] freed clone slot for speaker {prev_speaker}")
            prev_speaker = speaker_id

            for task in args.tasks:
                if is_done(out_root, provider.name, task, utt_id):
                    n_cached += 1
                    continue

                try:
                    if task == "tts":
                        result = provider.tts(
                            text,
                            voice_id=spec.default_voice_id,
                            model_id=spec.default_model_id,
                            seed=SEED,
                        )
                    else:
                        clone_kwargs = {
                            "model_id": spec.default_model_id,
                            "seed": SEED,
                        }
                        if spec.needs_reference_text:
                            clone_kwargs["reference_text"] = ref_text
                        result = provider.clone(
                            text,
                            reference_wav_path=ref_path,
                            **clone_kwargs,
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
                    traceback.print_exc(limit=2)
    finally:
        # Belt-and-suspenders: nuke any remaining clone voices / free the GPU.
        try:
            provider.cleanup()
        except Exception:
            pass

    print(f"\nDone. new={n_new}, cached={n_cached}, failed={n_failed}")
    if args.pilot:
        print("\n=== PILOT CHECKPOINT ===")
        print(f"Inspect WAVs in {out_root}/audio/{spec.name}/{{tts,cloning}}/")
        print(f"Inspect sidecars in {out_root}/sidecars/{spec.name}/{{tts,cloning}}/")
        print("If satisfied, re-run without --pilot to generate the full manifest.")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
