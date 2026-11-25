"""Speech-to-text engine — loads the Whisper model produced by Whisper_Song_ASR_FineTuning.ipynb."""
from __future__ import annotations

import io
import json
import os
import threading
import time

import numpy as np

SR = 16000


def decode_audio(data: bytes, filename: str = "audio.wav") -> np.ndarray:
    """Bytes -> 16 kHz mono float32. The browser already sends 16 kHz WAV; other formats go through librosa."""
    import soundfile as sf
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        import librosa, tempfile
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
        finally:
            os.unlink(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


class ASREngine:
    """Loads lazily in a background thread so the search UI is usable immediately."""

    def __init__(self, model_dir: str | None, fallback: str = "openai/whisper-small", max_new_tokens: int = 128):
        self.model_dir = model_dir if model_dir and os.path.isdir(model_dir) else None
        self.model_name = self.model_dir or fallback
        self.is_finetuned = self.model_dir is not None
        self.max_new_tokens = max_new_tokens
        self.pipe = None
        self.error = None
        self.loading = False
        self.load_seconds = None
        self._lock = threading.Lock()
        self.report = {}
        if self.model_dir and os.path.exists(os.path.join(self.model_dir, "asr_metrics.json")):
            self.report = json.load(open(os.path.join(self.model_dir, "asr_metrics.json")))

    def start(self):
        if self.pipe is None and not self.loading:
            self.loading = True
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import torch
            from transformers import pipeline
            t0 = time.time()
            cuda = torch.cuda.is_available()
            self.pipe = pipeline("automatic-speech-recognition", model=self.model_name,
                                 dtype=torch.float16 if cuda else torch.float32,
                                 device=0 if cuda else -1)
            self.device = "cuda" if cuda else "cpu"
            self.load_seconds = round(time.time() - t0, 1)
        except Exception as e:  # surfaced in /api/status
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.loading = False

    @property
    def ready(self):
        return self.pipe is not None

    def transcribe(self, audio: np.ndarray) -> dict:
        if not self.ready:
            raise RuntimeError(self.error or "The speech model is still loading. Try again in a few seconds.")
        t0 = time.time()
        with self._lock:
            out = self.pipe({"raw": audio, "sampling_rate": SR},
                            chunk_length_s=30 if len(audio) > 30 * SR else None,
                            generate_kwargs={"language": "english", "task": "transcribe",
                                             "max_new_tokens": self.max_new_tokens})
        return {"text": out["text"].strip(), "ms": round(1000 * (time.time() - t0)),
                "seconds": round(len(audio) / SR, 2)}

    def info(self):
        return {"model": os.path.basename(self.model_name.rstrip("/")) if self.model_dir else self.model_name,
                "finetuned": self.is_finetuned, "ready": self.ready, "loading": self.loading,
                "error": self.error, "load_seconds": self.load_seconds}
