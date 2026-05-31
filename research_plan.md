# План исследования: сравнительная оценка TTS и Voice Cloning API

> Документ предназначен для автономной реализации Claude Code. Каждая фаза имеет acceptance-критерии и (где есть денежные траты или потенциальные ошибки) **CHECKPOINT** — точки, где надо остановиться и подтвердить с пользователем перед продолжением.

---

## 1. Контекст и research question

**RQ.** Как современные коммерческие TTS- и voice-cloning-API (ElevenLabs, OpenAI TTS, Replicate-hosted F5-TTS и XTTS-v2) сравниваются между собой на стандартизованном тесте LibriTTS-R `test-clean` по совместной оси (naturalness, intelligibility, speaker similarity, latency, cost) — и существует ли доминирующий провайдер либо разные API занимают разные точки на Pareto-фронте?

**Гипотезы:**
- **H1.** Доминирующего провайдера нет; провайдеры формируют Pareto-фронт в (UTMOSv2, WavLM-cosine, $/1k chars).
- **H2.** Naturalness-метрики (UTMOSv2, NISQA) согласованы между собой (Spearman ρ > 0.7); speaker-similarity-метрики (WavLM, ECAPA-TDNN) расходятся — обосновывает многометричное репортирование.
- **H3.** Разрыв между лучшим коммерческим API и лучшим open-source меньше, чем разрыв между коммерческими API друг от друга.

**Что мы НЕ делаем в этом эксперименте.**
- Не запускаем human listening tests (нет пользователей).
- Не обучаем и не fine-tune'им модели.
- Не сравниваем многоязычность (только английский, LibriTTS-R).
- Не пытаемся валидировать automated metrics против ground-truth MOS (берём литературу как данность — TTSDS2 уже это сделал).

---

## 2. Структура репозитория

```
research/
├── config.yaml                 # ВСЕ гиперпараметры в одном месте
├── .env                        # API-ключи (НЕ коммитить, в .gitignore)
├── .env.example                # шаблон с пустыми ключами
├── requirements.txt            # пинованные версии
├── pyproject.toml              # опционально, для poetry
├── data/
│   ├── librittsr/              # распакованный test-clean
│   └── samples.json            # манифест: какие спикеры и фразы выбраны
├── src/
│   ├── generation/
│   │   ├── providers/
│   │   │   ├── base.py         # абстрактный Provider
│   │   │   ├── elevenlabs.py
│   │   │   ├── openai_tts.py
│   │   │   ├── replicate_xtts.py
│   │   │   └── replicate_f5.py
│   │   ├── generate.py         # main script
│   │   └── audio_io.py         # ресемплинг, формат-нормализация
│   ├── metrics/
│   │   ├── utmos.py
│   │   ├── nisqa.py
│   │   ├── whisper_wer.py
│   │   ├── wavlm_sim.py
│   │   ├── ecapa_sim.py
│   │   └── compute_all.py      # main script
│   └── analysis/
│       ├── stat_pack.py
│       └── figures.py
├── outputs/
│   ├── audio/{provider}/{task}/{speaker}_{utterance}.wav
│   ├── sidecars/{provider}/{task}/{speaker}_{utterance}.json
│   ├── costs.csv               # лог денежных трат построчно
│   └── failures.csv            # лог упавших запросов
├── results/
│   ├── metrics.parquet         # все метрики, одна строка на (provider, speaker, utterance, task)
│   ├── stats.json              # bootstrap CIs, p-values, корреляции
│   └── figures/*.pdf
└── README.md
```

---

## 3. Окружение

**Python 3.11** (3.12 ломает совместимость SpeechBrain).

`requirements.txt`:
```
# generation
elevenlabs==1.1.2
openai==1.43.0
replicate==0.34.1
python-dotenv==1.0.1
tenacity==9.0.0          # retry с exponential backoff

# audio
soundfile==0.12.1
librosa==0.10.2
faster-whisper==1.0.3
torch==2.3.1
torchaudio==2.3.1

# metrics models
transformers==4.44.2
speechbrain==1.0.0
git+https://github.com/sarulab-speech/UTMOSv2.git
git+https://github.com/gabrielmittag/NISQA.git

# stat & viz
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.0
seaborn==0.13.2
pyarrow==17.0.0          # для parquet
tqdm==4.66.4
```

`.env.example`:
```
ELEVENLABS_API_KEY=
OPENAI_API_KEY=
REPLICATE_API_TOKEN=
TOTAL_BUDGET_USD=100      # hard cap, скрипт остановится при превышении
```

---

## 4. Phase 1 — Подготовка датасета

### 4.1 Скачивание LibriTTS-R `test-clean`

```bash
mkdir -p data/librittsr
cd data/librittsr
wget https://openslr.elda.org/resources/141/test_clean.tar.gz
tar -xzf test_clean.tar.gz
rm test_clean.tar.gz
```

Структура после распаковки: `data/librittsr/LibriTTS_R/test-clean/{speaker_id}/{chapter_id}/...`.

### 4.2 Фильтрация спикеров

**Важно: распределение utterance-ов по спикерам в `test-clean` скошено.** Из ~39 спикеров несколько имеют <15 фраз. Мы требуем **≥21 фразы** (1 reference + 20 target texts), плюс reference длиной 5–10 секунд.

```python
# src/generation/build_manifest.py
import json, random
from pathlib import Path
import soundfile as sf

ROOT = Path("data/librittsr/LibriTTS_R/test-clean")
SEED = 42
N_SPEAKERS = 20
N_TARGETS = 20
REF_MIN_SEC, REF_MAX_SEC = 5.0, 10.0

random.seed(SEED)

# Соберём всех спикеров с их utterance-ами
speakers = {}
for spk_dir in ROOT.iterdir():
    if not spk_dir.is_dir(): continue
    utts = []
    for chap_dir in spk_dir.iterdir():
        for wav in chap_dir.glob("*.wav"):
            txt = wav.with_suffix(".normalized.txt")
            if not txt.exists(): continue
            info = sf.info(str(wav))
            utts.append({
                "id": wav.stem,
                "wav": str(wav.relative_to(ROOT)),
                "txt": str(txt.relative_to(ROOT)),
                "duration_sec": info.frames / info.samplerate,
            })
    if len(utts) >= N_TARGETS + 1:
        speakers[spk_dir.name] = utts

# Фильтр: спикер должен иметь хотя бы одну utterance подходящей длины для reference
eligible = []
for spk_id, utts in speakers.items():
    refs = [u for u in utts if REF_MIN_SEC <= u["duration_sec"] <= REF_MAX_SEC]
    if refs and len(utts) >= N_TARGETS + 1:
        eligible.append((spk_id, refs, utts))

# Сэмплинг 20 спикеров — желательно с гендерным балансом
# В LibriTTS-R speaker-info указан в SPEAKERS.txt (есть в doc.tar.gz)
# Если нет SPEAKERS.txt — берём 20 случайных
random.shuffle(eligible)
selected = eligible[:N_SPEAKERS]

manifest = []
for spk_id, refs, utts in selected:
    ref = random.choice(refs)
    targets = random.sample([u for u in utts if u["id"] != ref["id"]], N_TARGETS)
    manifest.append({"speaker_id": spk_id, "reference": ref, "targets": targets})

with open("data/samples.json", "w") as f:
    json.dump({"seed": SEED, "speakers": manifest}, f, indent=2)

print(f"Eligible speakers: {len(eligible)}; selected: {len(selected)}")
print(f"Total (speaker, target) pairs: {len(selected) * N_TARGETS}")
```

**Acceptance:** `data/samples.json` содержит 20 спикеров × 20 текстов = 400 (speaker, target) пар. Должно быть не меньше eligible-спикеров чем `N_SPEAKERS`, иначе понизить `N_SPEAKERS` или `N_TARGETS`.

**Если eligible < 20:** уменьшить `N_TARGETS` до 15 или `N_SPEAKERS` до 15. Документировать в README.

### 4.3 Хватает ли test-clean?

Да, при условии фильтрации:
- ~39 спикеров → после фильтра ≥21 utterance + хотя бы один ref-clip 5–10 сек → ожидаемо 25–32 eligible
- Из них берём 20 — есть запас
- 400 (спикер, текст) пар × 4 провайдера × 2 задачи (TTS + cloning, для cloning без OpenAI) = достаточно для статистики

**Если нужно больше мощности** (например, окажется, что различия слабые) — добавить `dev-clean` тем же скриптом и расширить до 40 спикеров. Это +1.3 GB скачивания, никаких других изменений.

---

## 5. Phase 2 — Интеграция провайдеров

### 5.1 Абстракция

```python
# src/generation/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GenerationResult:
    audio_path: Path        # WAV, 24 kHz mono PCM-16
    latency_ms: int
    cost_usd: float
    provider: str
    model_version: str      # для воспроизводимости
    raw_response: dict      # лог исходного ответа API

class Provider(ABC):
    name: str
    supports_cloning: bool

    @abstractmethod
    def tts(self, text: str, voice: str | None = None) -> GenerationResult:
        """TTS без клонирования. voice=None → дефолтный пресет."""
        ...

    @abstractmethod
    def clone(self, text: str, reference_wav: Path) -> GenerationResult:
        """Voice cloning. Может бросить NotImplementedError для OpenAI."""
        ...
```

### 5.2 Конкретные провайдеры

**ElevenLabs** (TTS + cloning):
- TTS: дефолтный голос `Rachel` (`21m00Tcm4TlvDq8ikWAM`) — нейтральный женский preset
- Cloning: Instant Voice Cloning через `/v1/voices/add`, потом `/v1/text-to-speech/{voice_id}`
- Модель: `eleven_multilingual_v2`
- Cost: ~$66 / 1M chars на Scale-плане → ~$0.066 / 1k chars

**OpenAI TTS** (только TTS, без cloning):
- Модель: `tts-1` (cheaper) — для основного эксперимента
- Опционально `tts-1-hd` для отдельной точки на графике
- Голос: `alloy` (нейтральный preset)
- Cost: $15 / 1M chars
- `clone()` бросает `NotImplementedError`

**Replicate XTTS-v2**:
- Модель: `lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e`
- Поддерживает и TTS (с дефолтным голосом), и cloning (с reference)
- Cost: ~$0.05–0.10 за прогон, зависит от длины

**Replicate F5-TTS**:
- Модель: `nyxynyx/f5-tts:1234abcd...` (взять актуальный hash с replicate.com)
- Cost: ~$0.09 за прогон

### 5.3 Retry и обработка ошибок

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _call_api(...):
    ...
```

Обработка:
- HTTP 429 → ждать `Retry-After` если есть, иначе exponential backoff
- HTTP 400 (content filter, неподдерживаемый текст) → залогировать в `failures.csv`, не ретраить, продолжить со следующим текстом
- Timeout > 60 сек → cancel + лог
- Невалидный аудиоответ (длина < 1 сек или > 60 сек) → отметить как failure

### 5.4 Нормализация аудио

Каждый провайдер возвращает свой формат (ElevenLabs — MP3, OpenAI — MP3, Replicate — обычно WAV). Привести к единому формату **до сохранения**:

```python
# src/generation/audio_io.py
import soundfile as sf
import librosa

CANONICAL_SR = 24000  # storage format

def save_canonical(audio_bytes: bytes, src_format: str, out_path: Path) -> None:
    """Конвертирует любой формат в WAV 24kHz mono PCM-16."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=CANONICAL_SR, mono=True)
    sf.write(out_path, y, CANONICAL_SR, subtype="PCM_16")
```

Метрические модели потом будут ресемплить в 16 kHz отдельно — это нормально, лучше хранить в исходном качестве провайдера (24 kHz) и ресемплить downstream.

---

## 6. Phase 3 — Pilot generation [CHECKPOINT]

**КРИТИЧЕСКАЯ ТОЧКА ОСТАНОВКИ.** Перед полным прогоном Claude Code должен:

1. Запустить генерацию на **N=5 (speaker, text) пар** через все 4 провайдера (TTS) и 3 (cloning) — всего ~35 файлов.
2. Проверить:
   - Все файлы созданы и читаются soundfile-ом
   - Длительность в разумных пределах (1–20 сек)
   - Сэмплрейт = 24 kHz, mono
   - Sidecar JSON-ы валидны
3. Залогировать pilot-cost.
4. **Остановиться и показать пользователю:**
   - Количество успешных vs failed
   - Pilot-cost
   - Экстраполированный full-cost (× 80 для коммерческих, × 80 для open-source)
   - Образцы аудио (путь к 2–3 файлам)
5. **Не запускать полную генерацию без явного `proceed` от пользователя.**

---

## 7. Phase 4 — Полная генерация

После CHECKPOINT'а:

- **TTS-задача**: 4 провайдера × 20 спикеров × 20 текстов = 1600 файлов
  - У OpenAI/ElevenLabs здесь используется их дефолтный preset-голос
  - У F5/XTTS — их дефолтный встроенный голос (если есть; если нет — придётся либо взять один из embedded референсов модели, либо включить TTS-режим через клонирование одного фиксированного speaker'а — задокументировать выбор)
- **Cloning-задача**: 3 провайдера (ex-OpenAI) × 20 спикеров × 20 текстов = 1200 файлов
  - Reference clip = `reference` из манифеста для каждого спикера
- **Итого: 2800 файлов**

### Resumability

Проверять перед каждым запросом:
```python
out_path = f"outputs/audio/{provider}/{task}/{speaker}_{utt}.wav"
if Path(out_path).exists():
    continue
```

Падение на 1500-м файле → перезапустить, продолжит с 1501-го.

### Cost cap

```python
import csv
total_spent = sum_costs_from("outputs/costs.csv")
if total_spent >= BUDGET:
    raise BudgetExceeded(f"Spent {total_spent}, cap {BUDGET}")
```

### Параллелизм

- Внутри одного провайдера — `asyncio` с 4–8 параллельными запросами (зависит от rate-лимита)
- Между провайдерами — параллельно (отдельные процессы или asyncio.gather)
- Replicate cold-start: первые 1–2 запроса будут медленными (~60 сек), потом обычно 5–15 сек

**Ожидаемое wall-clock на полную генерацию: 4–8 часов.**

---

## 8. Phase 5 — Установка моделей-судей

Эти модели **запускаются локально на CPU**. Они не платные; «скачать» означает один раз `pip install` или `huggingface-cli download`, затем кэш в `~/.cache/`.

### 8.1 UTMOSv2 — naturalness predictor

```bash
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

Веса (~300 MB) скачиваются автоматически при первом импорте в `~/.cache/utmosv2/`.

### 8.2 NISQA — multidimensional speech quality

```bash
pip install nisqa
# или git clone https://github.com/gabrielmittag/NISQA.git
```

Веса (~100 MB) в `~/.cache/nisqa/`.

### 8.3 faster-whisper — для WER round-trip

```bash
pip install faster-whisper
```

При первом использовании скачается `medium` (~1.5 GB) в `~/.cache/huggingface/`.

### 8.4 WavLM speaker verification

```python
from transformers import AutoModel, AutoFeatureExtractor
model = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv")
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
```

Веса (~400 MB) в `~/.cache/huggingface/`.

### 8.5 ECAPA-TDNN

```python
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="~/.cache/speechbrain/spkrec-ecapa-voxceleb",
)
```

Веса (~80 MB).

**Итого ~3 GB диска. Скачивается один раз. После — никаких сетевых вызовов.**

---

## 9. Phase 6 — Pipeline метрик

### 9.1 Универсальная подготовка аудио

Все модели-судьи ожидают **16 kHz mono float32**:

```python
def load_16k(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y.astype(np.float32)
```

### 9.2 UTMOSv2

```python
# src/metrics/utmos.py
import utmosv2
import torch

model = utmosv2.create_model(pretrained=True)
model.eval()

@torch.no_grad()
def score(wav_path: Path) -> float:
    return float(model.predict(input_path=str(wav_path)))
```

### 9.3 NISQA

```python
# src/metrics/nisqa.py
from nisqa.NISQA_model import nisqaModel
import argparse

args = argparse.Namespace(
    mode="predict_file",
    pretrained_model="weights_nisqa.tar",
    ms_channel=None,
    output_dir=None,
)
model = nisqaModel(args)

def score(wav_path: Path) -> dict:
    args.deg = str(wav_path)
    result = model.predict()
    # result содержит mos_pred, noi_pred, col_pred, dis_pred, loud_pred
    return result
```

### 9.4 Whisper-WER

```python
# src/metrics/whisper_wer.py
from faster_whisper import WhisperModel
import jiwer

model = WhisperModel("medium", device="cpu", compute_type="int8")

def transcribe(wav_path: Path) -> str:
    segments, _ = model.transcribe(str(wav_path), language="en", beam_size=1)
    return " ".join(s.text for s in segments).strip()

def wer(reference_text: str, hypothesis_text: str) -> float:
    # Нормализация важна — Whisper добавляет пунктуацию
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    return jiwer.wer(transform(reference_text), transform(hypothesis_text))
```

### 9.5 WavLM speaker similarity

```python
# src/metrics/wavlm_sim.py
import torch, numpy as np
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

model = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

@torch.no_grad()
def embed(wav: np.ndarray) -> np.ndarray:
    inputs = fe(wav, sampling_rate=16000, return_tensors="pt")
    out = model(**inputs).embeddings
    return out.cpu().numpy()[0]

def cosine(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    return float(cosine_similarity([emb_a], [emb_b])[0, 0])
```

### 9.6 ECAPA-TDNN

```python
# src/metrics/ecapa_sim.py
from speechbrain.inference.speaker import EncoderClassifier
import torch

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="~/.cache/speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"},
)

def embed(wav: np.ndarray) -> np.ndarray:
    sig = torch.tensor(wav).unsqueeze(0)
    emb = classifier.encode_batch(sig)
    return emb.squeeze().cpu().numpy()
```

### 9.7 Калибровочные якоря (обязательно для интерпретации similarity)

```python
# Считаем upper-bound: similarity между двумя разными utterance одного спикера из GT
# Считаем lower-bound: средняя similarity между разными спикерами
def compute_calibration_anchors(samples_manifest):
    """
    Возвращает {'upper_p50': ..., 'upper_p05': ..., 'lower_mean': ...}
    для WavLM и ECAPA отдельно.
    """
    ...
```

Без этих якорей абсолютные числа similarity бессмысленны.

### 9.8 Compute-all script

```python
# src/metrics/compute_all.py
import pandas as pd
from tqdm import tqdm
from pathlib import Path

PARQUET = Path("results/metrics.parquet")

rows = []
# Загрузить существующие, чтобы не пересчитывать
if PARQUET.exists():
    existing = pd.read_parquet(PARQUET)
    done = set(zip(existing.provider, existing.task, existing.speaker_id, existing.utt_id))
else:
    existing = pd.DataFrame()
    done = set()

for wav_path in tqdm(all_wavs):
    key = (provider, task, speaker_id, utt_id)
    if key in done: continue
    wav16 = load_16k(wav_path)
    row = {
        "provider": provider, "task": task, "speaker_id": speaker_id, "utt_id": utt_id,
        "utmos": utmos.score(wav_path),
        "nisqa_mos": nisqa.score(wav_path)["mos_pred"],
        "nisqa_noi": nisqa.score(wav_path)["noi_pred"],
        "whisper_wer": whisper_wer.wer(target_text, whisper_wer.transcribe(wav_path)),
        "wavlm_sim": wavlm_sim.cosine(ref_wavlm[speaker_id], wavlm_sim.embed(wav16)),
        "ecapa_sim": ecapa_sim.cosine(ref_ecapa[speaker_id], ecapa_sim.embed(wav16)),
        "latency_ms": sidecar["latency_ms"],
        "cost_usd": sidecar["cost_usd"],
    }
    rows.append(row)
    # Сохраняем батчами, чтобы не потерять прогресс
    if len(rows) % 100 == 0:
        pd.concat([existing, pd.DataFrame(rows)]).to_parquet(PARQUET)

pd.concat([existing, pd.DataFrame(rows)]).to_parquet(PARQUET)
```

**Ожидаемое wall-clock**: ~3 часа на CPU, ~1.5 часа на M-series Mac.

---

## 10. Phase 7 — Stat-pack

```python
# src/analysis/stat_pack.py
from scipy.stats import spearmanr, wilcoxon, bootstrap
from statsmodels.stats.weightstats import ttost_ind
import numpy as np

df = pd.read_parquet("results/metrics.parquet")

# 10.1 Bootstrap CI для среднего каждой метрики на каждом провайдере
for provider in df.provider.unique():
    for metric in ["utmos", "nisqa_mos", "whisper_wer", "wavlm_sim", "ecapa_sim"]:
        data = df[df.provider == provider][metric].dropna().values
        ci = bootstrap((data,), np.mean, n_resamples=1000, confidence_level=0.95)
        ...

# 10.2 Spearman ρ между метриками (system-level)
system_means = df.groupby("provider")[["utmos", "nisqa_mos", "wavlm_sim", "ecapa_sim"]].mean()
rho_naturalness = spearmanr(system_means["utmos"], system_means["nisqa_mos"])
rho_similarity = spearmanr(system_means["wavlm_sim"], system_means["ecapa_sim"])

# 10.3 Wilcoxon для парных сравнений провайдеров на utterance-level
# Парность: одна и та же (speaker, utt) у разных провайдеров → парная выборка
pivot = df.pivot_table(index=["speaker_id", "utt_id"], columns="provider", values="utmos")
for p1, p2 in itertools.combinations(pivot.columns, 2):
    stat, pval = wilcoxon(pivot[p1].dropna(), pivot[p2].dropna())
    ...

# 10.4 Pareto frontier с bootstrap
# Точка провайдера на фронте если ни одна другая не доминирует во всех осях
def is_pareto_optimal(points, idx):
    return not np.any(np.all(points >= points[idx], axis=1) & np.any(points > points[idx], axis=1))

# Bootstrap: какие провайдеры на фронте в ≥95% ресемплов?
...

# 10.5 H3: TOST на групповых средних commercial vs open-source
# Для каждой (задача, метрика) пары проверяем эквивалентность с порогом δ.
commercial = {"elevenlabs", "openai_tts", "azure_tts", "google_tts", "typecast", "resemble"}
open_source = {"xtts_v2", "f5_tts", "cosyvoice2", "fish_speech_s1", "fish_speech_s2_pro"}
delta = {"utmos": 0.20, "nisqa_mos": 0.20, "whisper_wer": 0.05,
         "wavlm_sim": 0.05, "ecapa_sim": 0.05}
for metric, d in delta.items():
    c = system_means.loc[list(system_means.index & commercial), metric].dropna().values
    o = system_means.loc[list(system_means.index & open_source), metric].dropna().values
    # ttost_ind возвращает (p, (t1,p1,df1), (t2,p2,df2)); p = max(p1,p2).
    p, _, _ = ttost_ind(c, o, low=-d, upp=d, usevar="unequal")
    equivalent = p < 0.05
```

Сохранять всё в `results/stats.json`.

---

## 11. Phase 8 — Визуализация

Минимальный набор figures:

1. **Pareto plot 2D** (×3 проекции: UTMOS×Sim, UTMOS×Cost, Sim×Cost) — точки = провайдеры, размер = N
2. **Metric correlation heatmap** — 5 метрик × 5 метрик, Spearman ρ, цветом
3. **Provider ranking table** — для каждой метрики средние + 95% CI, упорядоченные
4. **Calibration plot** — speaker sim для каждого провайдера + горизонтальные линии для upper/lower bounds
5. **Gap distribution** — boxplot CC vs CO разрывов
6. **Per-utterance jitter** — для каждой пары провайдеров scatter их utterance-level scores

Все в PDF в `results/figures/`. Использовать `matplotlib` + `seaborn`.

---

## 12. Чеклист «что могло пойти не так»

Хронологически:

- **Spec mismatch у Replicate-моделей**: хеш модели может устареть → проверять `replicate.com/{model}` на актуальный version-hash перед запуском
- **ElevenLabs Instant Voice Cloning лимит на voices**: бесплатный/недорогой тариф разрешает ~10 custom voices одновременно → возможно придётся удалять voice после генерации всех 20 utterance этим клоном
- **OpenAI content filter**: некоторые LibriTTS-тексты (особенно про насилие/религию из старых книг) могут отклоняться → ловить, логировать, считать как missing
- **Replicate cold-start**: первый запрос ~60 сек, не путать с латентностью модели → измерять latency *среднюю на не-первые* запросы
- **Whisper hallucination на тишине**: иногда выдаёт «Thanks for watching!» или «Subtitles by...» → фильтровать short outputs, проверять trailing silence
- **WavLM ожидает 16 kHz, ECAPA — 16 kHz, NISQA — 48 kHz, UTMOSv2 — 16 kHz**: ресемплить под каждую метрику, **не сохранять ресемплированную версию обратно** — это потеря качества
- **Sample rate у провайдеров**: ElevenLabs/OpenAI выдают 24 kHz, Replicate-модели — 24 kHz, иногда 22.05. Сохранять в **исходном sample rate** провайдера, ресемплить только для метрик
- **NaN в метриках**: на пустых файлах модели падают → wrap в try/except, NaN в parquet
- **Cost не обновляется в реальном времени**: ElevenLabs counts по символам, но фактический счёт виден в dashboard с лагом → доверять только своему локальному счётчику

---

## 13. Что я (как пользователь) забыл предусмотреть — добавлено в план выше

1. **Гендерный баланс спикеров** (10M / 10F) — без него возможны систематические перекосы в speaker similarity между провайдерами, по-разному работающими с мужскими/женскими голосами
2. **Default voice selection per provider для TTS-задачи** — без явной фиксации Claude Code может рандомно выбирать, что ломает воспроизводимость
3. **Калибровочные якоря для similarity** — без них абсолютные числа cosine не интерпретируются
4. **Resumability** — критично для генерации (платно!) и для метрик (долго)
5. **Cost cap и pilot-checkpoint** — защита от случайного списания $500
6. **Audio sample-rate canonicalisation** — без неё разные провайдеры дают несравнимые метрики
7. **Sidecar JSON metadata** — отдельный файл рядом с каждым WAV, содержит provider, model_version, timestamps, cost, raw_response → критично для воспроизводимости и отладки
8. **Reference clip length constraint** — слишком короткий ref (<3 сек) ломает XTTS, слишком длинный (>15 сек) ломает OpenAI; диапазон 5–10 сек безопасен для всех
9. **OpenAI exclusion from cloning** — явно задокументировать, что OpenAI участвует только в TTS-задаче
10. **Text normalization** — использовать `.normalized.txt`, а не `.original.txt`, чтобы числа и аббревиатуры были одинаково развёрнуты для всех провайдеров
11. **Failure logging** — отдельный CSV с {provider, speaker, utt, error_type, error_message} → потом считать failure rate как ещё одну метрику
12. **Replicate hash pinning** — модели обновляются, хеш надо зафиксировать в config.yaml
13. **Seed fixing** — `random.seed(42)` для воспроизводимости sampling спикеров и фраз

---

## 14. Финальный deliverable

После выполнения всех фаз должно быть:

- `data/samples.json` — манифест эксперимента
- `outputs/audio/` — 2800 WAV-файлов + sidecar JSON-ы
- `outputs/costs.csv` — построчный лог трат
- `outputs/failures.csv` — лог упавших запросов
- `results/metrics.parquet` — таблица метрик
- `results/stats.json` — bootstrap CI, p-values, корреляции
- `results/figures/*.pdf` — 6 figures
- `README.md` с инструкцией «как воспроизвести с нуля»

---

## 15. Порядок исполнения Claude Code

1. Phase 1 — setup repo + download dataset + build manifest (~30 мин)
2. Phase 2 — provider integrations (~2 ч)
3. Phase 3 — PILOT + **CHECKPOINT** ⛔
4. Phase 4 — full generation (~6 ч wall-clock, фоновый)
5. Phase 5 — install judge models (~15 мин)
6. Phase 6 — metrics pipeline (~3 ч wall-clock)
7. Phase 7 — stat-pack (~30 мин)
8. Phase 8 — figures (~30 мин)
9. Финальный README с reproduction instructions

**Активная работа: ~1 рабочий день. Wall-clock с фоновыми прогонами: 3–5 дней.**
