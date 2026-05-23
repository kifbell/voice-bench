"""Fish Audio OpenAudio S1 (Fish Speech S1 / fishaudio/s1-mini).

Two-stage TTS: an autoregressive LLM emits acoustic tokens, then a firefly-gan vocoder
decodes them to waveform. The model needs both checkpoints downloaded ahead of time
into ``--model-dir``.

Install:
    git clone https://github.com/fishaudio/fish-speech.git /tmp/fish-speech
    cd /tmp/fish-speech && pip install -e .
    huggingface-cli download fishaudio/s1-mini --local-dir checkpoints/s1-mini

Zero-shot voice cloning at inference: pass ref WAV + ref transcript + target text. The
provider's tts() task uses a fixed default reference clip (same shape as F5 / CosyVoice 2
providers).

License: code under Apache-2.0; s1-mini weights under Apache-2.0 (verify the
exact LICENSE in the HF repo). The full S1 (non-mini) may be CC-BY-NC -- prefer the mini
checkpoint for the benchmark.

Output: float32 mono at ~44.1 kHz; we resample to 24 kHz before packing PCM-16.
"""
import os
import time
from pathlib import Path

import numpy as np

from voice_bench.providers._common import (
    SAMPLE_RATE_CANONICAL,
    float_to_pcm16_bytes,
    read_normalized_txt_alongside,
    resample_to_canonical,
)
from voice_bench.providers.base import GenerationResult


DEFAULT_MODEL_ID = "s1-mini"


class FishSpeechS1Provider:
    name = "fish_speech_s1"
    supports_cloning = True
    # firefly-gan default. Verified empirically; falls through resample_to_canonical
    # so wrong guesses degrade quality but don't crash.
    NATIVE_SAMPLE_RATE = 44100

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint_dir: str | Path | None = None,
        default_ref_wav: Path | None = None,
        default_ref_text: str | None = None,
    ):
        self._device = device
        # If unset, expect ``HF_HOME``-style auto-resolution -- the inference engine
        # will look up ``fishaudio/s1-mini`` via huggingface_hub.
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else None
        )
        self._default_ref_wav = Path(default_ref_wav) if default_ref_wav else None
        self._default_ref_text = default_ref_text
        self._engine = None
        self._engine_module = None

    def tts(self, text, voice_id, model_id=DEFAULT_MODEL_ID, *, seed=None):
        del voice_id
        if self._default_ref_wav is None or self._default_ref_text is None:
            raise RuntimeError(
                "FishSpeechS1Provider.tts() needs default_ref_wav and default_ref_text "
                "(Fish Speech zero-shot is reference-only)."
            )
        wav, elapsed = self._infer(
            ref_wav=self._default_ref_wav,
            ref_text=self._default_ref_text,
            gen_text=text,
            seed=seed,
        )
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="tts",
            model_id=model_id,
            voice_id=f"default_ref:{self._default_ref_wav.stem}",
            character_count=len(text),
            seed=seed,
            reference_wav_path=str(self._default_ref_wav.resolve()),
        )

    def clone(self, text, reference_wav_path, model_id=DEFAULT_MODEL_ID, *, seed=None, reference_text=None):
        ref_path = Path(reference_wav_path)
        ref_text = reference_text or read_normalized_txt_alongside(ref_path)
        if not ref_text:
            raise RuntimeError(
                f"Fish Speech clone() needs the reference transcript. "
                f"Pass reference_text=, or place a .normalized.txt next to {ref_path}."
            )
        wav, elapsed = self._infer(
            ref_wav=ref_path, ref_text=ref_text, gen_text=text, seed=seed
        )
        return GenerationResult(
            audio_pcm=float_to_pcm16_bytes(wav),
            sample_rate=SAMPLE_RATE_CANONICAL,
            channels=1,
            sample_width=2,
            latency_seconds=elapsed,
            provider=self.name,
            task="cloning",
            model_id=model_id,
            voice_id=f"clone:{ref_path.stem}",
            character_count=len(text),
            seed=seed,
            reference_wav_path=str(ref_path.resolve()),
        )

    def cleanup(self):
        if self._engine is not None:
            del self._engine
            self._engine = None
        self._engine_module = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- internals -------------------------------------------------------------

    def _ensure_loaded(self):
        if self._engine is not None:
            return
        # The fish_speech package has reorganised its inference module across versions.
        # Try the names in order of recency.
        candidate_paths = [
            ("fish_speech.inference_engine", "TTSInferenceEngine"),
            ("fish_speech.utils.inference_engine", "TTSInferenceEngine"),
            ("tools.api_server", "AppState"),  # last-resort placeholder
        ]
        last_err = None
        for module_path, cls_name in candidate_paths:
            try:
                module = __import__(module_path, fromlist=[cls_name])
                cls = getattr(module, cls_name)
                self._engine_module = module_path
                # Look for llama / decoder checkpoint paths under the configured dir.
                ckpt_dir = self._checkpoint_dir or Path(
                    os.environ.get(
                        "FISH_SPEECH_CHECKPOINT_DIR",
                        "checkpoints/s1-mini",
                    )
                )
                # Different versions of fish_speech use different kwargs; try the most
                # common shape and let TypeError signal a mismatch we can adapt to.
                try:
                    self._engine = cls(
                        llama_checkpoint_path=str(ckpt_dir),
                        decoder_checkpoint_path=str(ckpt_dir),
                        device=self._device,
                    )
                except TypeError:
                    # Newer API: just pass the directory.
                    self._engine = cls(checkpoint_path=str(ckpt_dir), device=self._device)
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Failed to load fish_speech inference engine. "
            f"Tried {[m for m, _ in candidate_paths]}; last error: {last_err}"
        )

    def _infer(self, *, ref_wav, ref_text, gen_text, seed):
        self._ensure_loaded()
        if seed is not None:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Engine API differs by version. Use a duck-typed call -- pass the most common
        # kwargs and accept whichever return shape it gives us.
        started = time.perf_counter()
        result = self._engine.inference(
            text=gen_text,
            reference_audio=str(ref_wav),
            reference_text=ref_text,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            seed=seed if seed is not None else -1,
        )
        # ``result`` is either:
        #   * a list / generator of objects with a ``.audio`` numpy array
        #   * a single (numpy, sr) tuple
        #   * a numpy array (assume native SR)
        chunks = []
        if hasattr(result, "__iter__") and not isinstance(result, (np.ndarray, tuple)):
            for chunk in result:
                audio = getattr(chunk, "audio", None)
                if audio is None and isinstance(chunk, (tuple, list)) and chunk:
                    audio = chunk[0]
                if audio is None:
                    raise RuntimeError(
                        f"unrecognized fish_speech inference chunk type: {type(chunk)!r}"
                    )
                chunks.append(np.asarray(audio, dtype=np.float32).squeeze())
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        elif isinstance(result, tuple) and len(result) == 2:
            wav = np.asarray(result[0], dtype=np.float32).squeeze()
        else:
            wav = np.asarray(result, dtype=np.float32).squeeze()
        elapsed = time.perf_counter() - started

        wav = resample_to_canonical(wav, self.NATIVE_SAMPLE_RATE)
        return wav, elapsed
