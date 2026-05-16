"""Probe which APIs are reachable with the keys in .env.

Spends minimal money: <$0.001 of OpenAI tokens, zero ElevenLabs credits
(subscription endpoint is free), no Replicate calls if token absent.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def probe_elevenlabs(api_key: str) -> dict:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    info = {"reachable": False}
    sub = client.user.subscription.get()
    info["reachable"] = True
    info["tier"] = sub.tier
    info["character_count"] = sub.character_count
    info["character_limit"] = sub.character_limit
    info["voice_limit"] = sub.voice_limit
    info["professional_voice_limit"] = sub.professional_voice_limit
    info["can_extend_voice_limit"] = sub.can_extend_voice_limit
    info["can_use_instant_voice_cloning"] = sub.can_use_instant_voice_cloning
    info["can_use_professional_voice_cloning"] = sub.can_use_professional_voice_cloning
    return info


def probe_openai_tts(api_key: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    info = {"reachable": False, "models": {}}

    for model in ("tts-1", "gpt-4o-mini-tts"):
        try:
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice="alloy",
                input="hi",
                response_format="wav",
            ) as resp:
                body = resp.read()
            info["models"][model] = {"ok": True, "bytes": len(body)}
            info["reachable"] = True
        except Exception as e:
            info["models"][model] = {"ok": False, "error": str(e)[:200]}
    return info


def probe_replicate(api_key: str | None) -> dict:
    if not api_key:
        return {"reachable": False, "reason": "REPLICATE_API_TOKEN not set in .env"}
    try:
        import replicate
    except ImportError:
        return {"reachable": False, "reason": "replicate package not installed"}
    client = replicate.Client(api_token=api_key)
    info = {"reachable": False}
    account = client.accounts.current()
    info["reachable"] = True
    info["username"] = account.username
    info["type"] = account.type
    return info


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    print("=" * 60)
    print("ElevenLabs")
    print("=" * 60)
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("  no key")
    else:
        try:
            info = probe_elevenlabs(key)
            for k, v in info.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("OpenAI TTS")
    print("=" * 60)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("  no key")
    else:
        try:
            info = probe_openai_tts(key)
            for k, v in info.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("Replicate")
    print("=" * 60)
    key = os.environ.get("REPLICATE_API_TOKEN")
    try:
        info = probe_replicate(key)
        for k, v in info.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
