"""Smart Song Search — web platform (FastAPI backend + static HTML/CSS/JS frontend).

    python server.py                      # http://localhost:8000
    python server.py --port 9000 --lyrics-csv music_dataset.csv

Put the two zips from the notebooks in ./artifacts (they are unpacked on first start):
    artifacts/song_search_model.zip   (SBERT_Song_Search_FineTuning.ipynb)
    artifacts/whisper_song_asr.zip    (Whisper_Song_ASR_FineTuning.ipynb)
"""
from __future__ import annotations

import argparse
import glob
import os
import zipfile

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_engine import ASREngine, decode_audio
from search_engine import METHODS, SearchEngine

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")


# ----------------------------------------------------------------------------- artifacts
def _find_dir(root, must_have):
    for d, _, files in os.walk(root):
        if all(m in files for m in must_have):
            return d
    return None


def prepare_artifacts(art_root):
    os.makedirs(art_root, exist_ok=True)
    sbert_zip = os.path.join(art_root, "song_search_model.zip")
    sbert_dir = os.path.join(art_root, "song_search_artifacts")
    if os.path.exists(sbert_zip) and not os.path.isdir(sbert_dir):
        print("Unpacking", sbert_zip)
        with zipfile.ZipFile(sbert_zip) as z:
            z.extractall(sbert_dir)
    asr_zip = os.path.join(art_root, "whisper_song_asr.zip")
    if os.path.exists(asr_zip) and not _find_dir(art_root, ["generation_config.json", "preprocessor_config.json"]) \
            and not _find_dir(art_root, ["generation_config.json", "processor_config.json"]):
        print("Unpacking", asr_zip)
        with zipfile.ZipFile(asr_zip) as z:
            z.extractall(art_root)
    search_dir = _find_dir(art_root, ["config.json", "songs.csv", "chunks.csv", "chunk_embeddings.npy"])
    asr_dir = None
    for d, _, files in os.walk(art_root):
        if "generation_config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files) \
                and any(f in files for f in ("preprocessor_config.json", "processor_config.json")):
            asr_dir = d
            break
    return search_dir, asr_dir


# ----------------------------------------------------------------------------- app
class SearchRequest(BaseModel):
    query: str = Field(..., max_length=2000)
    method: str = "hybrid"
    top_k: int = Field(10, ge=1, le=50)


def create_app(art_root, lyrics_csv=None, asr_fallback="openai/whisper-small", no_asr=False, cors="*"):
    search_dir, asr_dir = prepare_artifacts(art_root)
    if not search_dir:
        raise SystemExit(f"No search index found in {art_root}. Put song_search_model.zip there (see README).")
    if lyrics_csv is None:
        cands = glob.glob(os.path.join(HERE, "*.csv")) + glob.glob(os.path.join(art_root, "*.csv"))
        lyrics_csv = cands[0] if cands else None
    print("Search index :", search_dir)
    print("Speech model :", asr_dir or f"(not found, fallback {asr_fallback})")
    engine = SearchEngine(search_dir, lyrics_csv=lyrics_csv)
    asr = None if no_asr else ASREngine(asr_dir, fallback=asr_fallback)
    if asr:
        asr.start()

    app = FastAPI(title="Smart Song Search", version="1.0")
    # allow a frontend opened from another address (VS Code Live Server, another port, file://)
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in cors.split(",")],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/status")
    def status():
        return {"search": engine.info(), "asr": asr.info() if asr else {"ready": False, "disabled": True},
                "methods": METHODS}

    @app.post("/api/search")
    def search(req: SearchRequest):
        if req.method not in METHODS:
            raise HTTPException(400, f"Unknown method '{req.method}'. Use one of {list(METHODS)}.")
        return engine.search(req.query, req.method, req.top_k)

    @app.post("/api/voice-search")
    async def voice_search(audio: UploadFile = File(...), method: str = Form("hybrid"), top_k: int = Form(10)):
        if asr is None:
            raise HTTPException(503, "Voice search is turned off on this server (started with --no-asr).")
        if not asr.ready:
            raise HTTPException(503, asr.error or "The speech model is still loading. Try again in a few seconds.")
        data = await audio.read()
        if len(data) < 1000:
            raise HTTPException(400, "The recording is empty. Record at least two seconds of singing.")
        try:
            y = decode_audio(data, audio.filename or "audio.wav")
        except Exception as e:
            raise HTTPException(400, f"Could not read the audio file ({e}). Use WAV, MP3, OGG or FLAC.")
        if len(y) < 16000 * 0.8:
            raise HTTPException(400, "The recording is shorter than one second. Sing a full line and try again.")
        tr = asr.transcribe(y)
        res = engine.search(tr["text"], method if method in METHODS else "hybrid", max(1, min(int(top_k), 50)))
        return {"transcript": tr["text"], "asr_ms": tr["ms"], "audio_seconds": tr["seconds"], **res}

    @app.get("/api/songs/{idx}")
    def song(idx: int):
        if not 0 <= idx < engine.n_songs:
            raise HTTPException(404, "Song not found.")
        s = engine.songs.iloc[idx]
        text, source = engine.lyrics(idx)
        return {"song_idx": idx, "title": s["Title"], "artist": s["Artist"],
                "album": "" if str(s.get("Album")) == "nan" else str(s.get("Album")),
                "year": None if str(s.get("Year")) == "nan" else int(float(s["Year"])),
                "lyrics": text, "lyrics_source": source}

    @app.get("/api/examples")
    def examples(n: int = 4):
        return engine.examples(max(1, min(n, 8)))

    @app.get("/api/metrics")
    def metrics():
        return {"retrieval": engine.metrics(), "asr": asr.report if asr else {}}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    # serve the frontend at / as well (index.html, styles.css, config.js, app.js); API routes above win
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="frontend")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smart Song Search web platform")
    ap.add_argument("--artifacts", default=os.environ.get("SSS_ARTIFACTS", os.path.join(HERE, "artifacts")))
    ap.add_argument("--lyrics-csv", default=os.environ.get("SSS_LYRICS_CSV"),
                    help="optional original dataset CSV, to show full original lyrics")
    ap.add_argument("--asr-fallback", default="openai/whisper-small",
                    help="model used when no fine-tuned Whisper is found")
    ap.add_argument("--no-asr", action="store_true", help="text search only")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cors", default="*", help="allowed frontend origins, comma separated (default: any)")
    a = ap.parse_args()
    app = create_app(a.artifacts, a.lyrics_csv, a.asr_fallback, a.no_asr, a.cors)
    print(f"\n  Smart Song Search is running at http://{a.host}:{a.port}\n")
    uvicorn.run(app, host=a.host, port=a.port)
