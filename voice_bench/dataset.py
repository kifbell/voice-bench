import json
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Utterance:
    utt_id: str  # e.g. "237_126133_000014_000000"
    speaker_id: str
    chapter_id: str
    wav_path: str  # absolute path on disk
    text: str  # from .normalized.txt
    duration_sec: float


@dataclass(frozen=True)
class SpeakerInfo:
    speaker_id: str
    gender: str | None  # "F" | "M" | None if not in SPEAKERS.txt


def parse_speakers_txt(path: Path) -> dict[str, SpeakerInfo]:
    """Parse LibriTTS-R SPEAKERS.txt (pipe-separated, ';'-comments)."""
    result: dict[str, SpeakerInfo] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        spk_id, gender = parts[0], parts[1]
        if gender not in ("F", "M"):
            gender = None
        result[spk_id] = SpeakerInfo(speaker_id=spk_id, gender=gender)
    return result


def scan_dataset(root: Path) -> dict[str, list[Utterance]]:
    """Walk LibriTTS-R test-clean (or similar subset). Returns {speaker_id: utterances}."""
    speakers: dict[str, list[Utterance]] = {}
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        utts: list[Utterance] = []
        for chap_dir in sorted(p for p in spk_dir.iterdir() if p.is_dir()):
            for wav in sorted(chap_dir.glob("*.wav")):
                txt = wav.with_suffix("").with_suffix(".normalized.txt")
                if not txt.exists():
                    continue
                with wave.open(str(wav)) as w:
                    duration = w.getnframes() / w.getframerate()
                utts.append(Utterance(
                    utt_id=wav.stem,
                    speaker_id=spk_dir.name,
                    chapter_id=chap_dir.name,
                    wav_path=str(wav.resolve()),
                    text=txt.read_text(encoding="utf-8").strip(),
                    duration_sec=duration,
                ))
        if utts:
            speakers[spk_dir.name] = utts
    return speakers


def build_manifest(
    dataset_root: Path,
    speakers_txt: Path | None = None,
    *,
    n_speakers: int = 20,
    n_targets: int = 20,
    ref_min_sec: float = 5.0,
    ref_max_sec: float = 10.0,
    target_min_sec: float = 2.0,
    target_max_sec: float = 15.0,
    seed: int = 42,
    gender_balance: bool = True,
) -> dict:
    rng = random.Random(seed)
    raw = scan_dataset(dataset_root)
    gender_lookup = parse_speakers_txt(speakers_txt) if speakers_txt and speakers_txt.exists() else {}

    eligible: list[tuple[str, list[Utterance], list[Utterance], str | None]] = []
    for spk_id, utts in raw.items():
        refs = [u for u in utts if ref_min_sec <= u.duration_sec <= ref_max_sec]
        targets_pool = [u for u in utts if target_min_sec <= u.duration_sec <= target_max_sec]
        if not refs:
            continue
        if len(targets_pool) < n_targets + 1:
            continue
        gender = gender_lookup.get(spk_id).gender if spk_id in gender_lookup else None
        eligible.append((spk_id, refs, targets_pool, gender))

    if gender_balance and gender_lookup:
        females = [e for e in eligible if e[3] == "F"]
        males = [e for e in eligible if e[3] == "M"]
        rng.shuffle(females)
        rng.shuffle(males)
        half = n_speakers // 2
        picked = females[:half] + males[: n_speakers - half]
        if len(picked) < n_speakers:
            remaining = [e for e in eligible if e not in picked]
            rng.shuffle(remaining)
            picked.extend(remaining[: n_speakers - len(picked)])
        selected = picked
    else:
        rng.shuffle(eligible)
        selected = eligible[:n_speakers]

    rng.shuffle(selected)

    speakers_out = []
    for spk_id, refs, targets_pool, gender in selected:
        ref = rng.choice(refs)
        candidates = [u for u in targets_pool if u.utt_id != ref.utt_id]
        targets = rng.sample(candidates, min(n_targets, len(candidates)))
        speakers_out.append({
            "speaker_id": spk_id,
            "gender": gender,
            "reference": asdict(ref),
            "targets": [asdict(t) for t in targets],
        })

    return {
        "seed": seed,
        "dataset_root": str(dataset_root.resolve()),
        "n_speakers_requested": n_speakers,
        "n_speakers_selected": len(speakers_out),
        "n_targets_per_speaker": n_targets,
        "ref_duration_range_sec": [ref_min_sec, ref_max_sec],
        "target_duration_range_sec": [target_min_sec, target_max_sec],
        "gender_balanced": gender_balance and bool(gender_lookup),
        "n_eligible_speakers": len(eligible),
        "speakers": speakers_out,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())
