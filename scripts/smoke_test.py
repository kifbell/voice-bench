import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_bench.providers.base import SynthRequest
from voice_bench.providers.elevenlabs import ElevenLabsProvider
from voice_bench.runner import generate_and_save


SAMPLE_TEXTS = [
    ("smoke_001", "The quick brown fox jumps over the lazy dog."),
    ("smoke_002", "She sells sea shells by the sea shore on Sunday mornings."),
    ("smoke_003", "In 2026, voice synthesis evaluation requires both objective metrics and human listening tests."),
]

MODEL_ID = "eleven_multilingual_v2"
SEED = 42


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY missing in .env", file=sys.stderr)
        return 1

    provider = ElevenLabsProvider(api_key=api_key)

    voices = provider.list_voice_ids(limit=3)
    if not voices:
        print("No voices returned from ElevenLabs account", file=sys.stderr)
        return 1
    voice_id, voice_name = voices[0]
    print(f"Using voice: {voice_name} ({voice_id})")

    out_root = Path(__file__).resolve().parent.parent / "outputs"

    for utt_id, text in SAMPLE_TEXTS:
        req = SynthRequest(text=text, voice_id=voice_id, model_id=MODEL_ID, seed=SEED)
        rec = generate_and_save(provider, req, utt_id=utt_id, out_root=out_root)
        print(
            f"  {utt_id}: {rec.character_count} chars, "
            f"{rec.latency_seconds:.2f}s -> {rec.wav_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
