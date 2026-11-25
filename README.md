# Smart Song Search: web platform

Find a song from a line of lyrics. You can type the line, sing it into the microphone or upload a recording.
The backend is a Python FastAPI server that loads both fine-tuned models. The frontend is plain HTML, CSS and JavaScript.

```
Browser (HTML/CSS/JS)                         Python server (FastAPI)
  Type lyrics ───────── POST /api/search ───▶ normalize_text ─▶ SBERT + TF-IDF/BM25 hybrid ─▶ ranked songs
  Sing / Upload ─ 16 kHz WAV ─ POST /api/voice-search ─▶ fine-tuned Whisper ─▶ transcript ─▶ same search
```

## 1. Train the two models in Colab

| Notebook | Output (in `MyDrive/smart_song_search/`) |
|---|---|
| `SBERT_Song_Search_FineTuning.ipynb` (yours) | `song_search_model.zip` |
| `Whisper_Song_ASR_FineTuning.ipynb` | `whisper_song_asr.zip` |

Run the SBERT notebook first. Section 14 of the Whisper notebook (the end-to-end test) uses its output.

## 2. Put the zips in `artifacts/`

```
smart-song-search/
├── artifacts/
│   ├── song_search_model.zip
│   └── whisper_song_asr.zip
├── server.py  search_engine.py  asr_engine.py
└── static/  index.html  styles.css  app.js
```
Both zips are unpacked automatically the first time the server starts.

## 3. Install and run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py --lyrics-csv music_dataset.csv
```
Then open **http://localhost:8000**.

| Option | What it does |
|---|---|
| `--lyrics-csv music_dataset.csv` | Shows the original lyrics in the lyric panel. Without it, the panel shows the normalised text rebuilt from the index. A `.csv` placed next to `server.py` is picked up automatically. |
| `--no-asr` | Text search only. The server starts faster and uses less memory. |
| `--asr-fallback openai/whisper-base` | The model to download when no fine-tuned Whisper is found. |
| `--port 9000`, `--host 0.0.0.0` | Change the port, or open the server to your network. |

The microphone only works on `localhost` or over HTTPS, because browsers block it on other plain-HTTP addresses.
For a demo on other devices, use a tunnel such as `ngrok http 8000`.

## API

| Method | Path | Body |
|---|---|---|
| GET | `/api/status` | Models loaded, number of songs, hybrid settings |
| POST | `/api/search` | `{"query": "...", "method": "hybrid", "top_k": 10}`. Methods: `hybrid`, `sbert`, `tfidf`, `bm25` |
| POST | `/api/voice-search` | multipart `audio` file, plus `method` and `top_k` |
| GET | `/api/songs/{song_idx}` | Title, artist, album, year, lyrics |
| GET | `/api/examples?n=3` | Random lines from held-out test songs |
| GET | `/api/metrics` | `test_results.csv` (SBERT notebook) and `asr_metrics.json` (Whisper notebook) |

Interactive docs: http://localhost:8000/docs

## Notes
- The search runs exactly as in the notebook: the same `normalize_text`, chunk max-pooling, min-max hybrid with the `hybrid_alpha` and `hybrid_lexical` values from `config.json`, and one result per song group.
- `bm25.pkl` and `tfidf.pkl` are Python pickles, so only load artifacts that you produced yourself.
- On a laptop CPU, whisper-small takes about 1 to 3 seconds per 10-second clip. The first start loads the model in the background, and the "Voice model" pill turns green when it is ready.
