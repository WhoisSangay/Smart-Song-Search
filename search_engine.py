"""Lyric search engine — loads the artifacts produced by SBERT_Song_Search_FineTuning.ipynb.

Expected folder (unzipped song_search_model.zip):
    songs.csv, chunks.csv, chunk_embeddings.npy, bm25.pkl, tfidf.pkl, config.json, sbert_song_search/
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# ----------------------------------------------------------------------------------------------
# Text normalisation — identical to the notebooks so queries, transcripts and lyrics match.
# ----------------------------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace(chr(0x2019), "'").replace(chr(0x2018), "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class BM25:
    """Same class as in the SBERT notebook (needed to unpickle bm25.pkl)."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.cv = CountVectorizer(token_pattern=r"(?u)\b[\w']+\b")
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        N = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        avgdl = dl.mean()
        dfreq = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (N - dfreq + 0.5) / (dfreq + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(N), np.diff(tf.indptr))
        W = tf.copy()
        W.data = self.idf[tf.indices] * tf.data * (self.k1 + 1) / (
            tf.data + self.k1 * (1 - self.b + self.b * dl[rows] / avgdl))
        self.W_T = W.T.tocsr()
        return self

    def scores(self, queries):
        Q = self.cv.transform(queries)
        Q.data[:] = 1
        return (Q @ self.W_T).toarray()


class _ArtifactUnpickler(pickle.Unpickler):
    # bm25.pkl was pickled from the notebook's __main__; point it at the class above.
    def find_class(self, module, name):
        if name == "BM25":
            return BM25
        return super().find_class(module, name)


def _minmax(S: np.ndarray) -> np.ndarray:
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - lo) / np.where(hi - lo == 0, 1, hi - lo)


def _clean_title(t) -> str:
    zero_width = "".join(map(chr, (0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF)))
    return str(t).translate({ord(c): None for c in zero_width}).strip()


METHODS = {
    "hybrid": "Hybrid (SBERT + lexical)",
    "sbert": "Fine-tuned SBERT",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
}


class SearchEngine:
    def __init__(self, art_dir: str, lyrics_csv: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import torch

        t0 = time.time()
        self.dir = art_dir
        self.cfg = json.load(open(os.path.join(art_dir, "config.json")))
        self.songs = pd.read_csv(os.path.join(art_dir, "songs.csv"))
        self.chunks = pd.read_csv(os.path.join(art_dir, "chunks.csv"))
        self.emb = np.load(os.path.join(art_dir, "chunk_embeddings.npy")).astype(np.float32)
        with open(os.path.join(art_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = _ArtifactUnpickler(f).load()
        with open(os.path.join(art_dir, "tfidf.pkl"), "rb") as f:
            t = _ArtifactUnpickler(f).load()
            self.tfidf_vec, self.tfidf_mat = t["vectorizer"], t["matrix"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = os.path.join(art_dir, self.cfg.get("model_dir", "sbert_song_search"))
        self.model = SentenceTransformer(model_dir, device=self.device)

        self.alpha = float(self.cfg.get("hybrid_alpha", 0.5))
        self.lexical = self.cfg.get("hybrid_lexical", "BM25")
        self.n_songs = len(self.songs)
        self.chunk_song = self.chunks["song_idx"].to_numpy()
        self.group = self.songs["group_id"].to_numpy()
        self.songs["Title"] = self.songs["Title"].map(_clean_title)
        # chunks are stored song by song -> fast max-pooling with reduceat
        self._sorted = bool(np.all(np.diff(self.chunk_song) >= 0)) and len(np.unique(self.chunk_song)) == self.n_songs
        self._offsets = np.searchsorted(self.chunk_song, np.arange(self.n_songs)) if self._sorted else None
        self._song_chunk_rows = self.chunks.groupby("song_idx").indices

        self._raw_lyrics = self._load_raw_lyrics(lyrics_csv)
        self.load_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ lyrics
    def _load_raw_lyrics(self, path):
        if not path or not os.path.exists(path):
            return {}
        raw = pd.read_csv(path)
        raw["Title"] = raw["Title"].map(_clean_title)
        return {(a, t): l for a, t, l in zip(raw["Artist"], raw["Title"], raw["Lyric"].astype(str))}

    def _rebuilt_words(self, idx: int):
        words = []
        for r in self._song_chunk_rows.get(idx, []):
            w = str(self.chunks.at[r, "text"]).split()
            k = min(len(words), len(w))
            while k > 0 and words[-k:] != w[:k]:
                k -= 1
            words += w[k:]
        return words

    def lyrics(self, idx: int):
        row = self.songs.iloc[idx]
        raw = self._raw_lyrics.get((row["Artist"], row["Title"]))
        if raw:
            return raw, "original"
        return " ".join(self._rebuilt_words(idx)), "normalised"

    # ------------------------------------------------------------------ scoring
    def _sbert_chunk_sims(self, queries):
        q = self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        return q @ self.emb.T

    def _pool(self, sims):
        if self._sorted:
            return np.maximum.reduceat(sims, self._offsets, axis=1)
        out = np.full((sims.shape[0], self.n_songs), -1.0, dtype=np.float32)
        for j in range(sims.shape[0]):
            np.maximum.at(out[j], self.chunk_song, sims[j])
        return out

    def _tfidf(self, qs):
        return (self.tfidf_vec.transform(qs) @ self.tfidf_mat.T).toarray()

    def _lexical(self, qs, name):
        return self._tfidf(qs) if name == "TF-IDF" else self.bm25.scores(qs)

    def search(self, query: str, method: str = "hybrid", top_k: int = 10):
        t0 = time.time()
        q = normalize_text(query)
        if not q:
            return {"query": query, "normalized": "", "results": [], "ms": 0}
        chunk_sims = self._sbert_chunk_sims([q])          # also used to pick the lyric snippet
        sb = self._pool(chunk_sims)
        if method == "sbert":
            scores = sb[0]
        elif method == "tfidf":
            scores = self._tfidf([q])[0]
        elif method == "bm25":
            scores = self.bm25.scores([q])[0]
        else:
            method = "hybrid"
            scores = (self.alpha * _minmax(sb) + (1 - self.alpha) * _minmax(self._lexical([q], self.lexical)))[0]

        q_words = set(q.split())
        results, seen = [], set()
        for i in np.argsort(-scores):
            g = self.group[i]
            if g in seen:
                continue
            seen.add(g)
            rows = self._song_chunk_rows.get(int(i), [])
            best_row = rows[int(np.argmax(chunk_sims[0, rows]))] if len(rows) else None
            snippet = str(self.chunks.at[best_row, "text"]) if best_row is not None else ""
            s = self.songs.iloc[int(i)]
            results.append({
                "rank": len(results) + 1, "song_idx": int(i),
                "title": s["Title"], "artist": s["Artist"],
                "album": "" if pd.isna(s.get("Album")) else str(s["Album"]),
                "year": None if pd.isna(s.get("Year")) else int(s["Year"]),
                "score": round(float(scores[i]), 4),
                "semantic": round(float(sb[0, i]), 4),
                "snippet": snippet,
                "matched": sorted(q_words & set(snippet.split())),
            })
            if len(results) >= top_k:
                break
        return {"query": query, "normalized": q, "method": method, "method_label": METHODS[method],
                "results": results, "ms": round(1000 * (time.time() - t0), 1)}

    # ------------------------------------------------------------------ extras
    def examples(self, n=4, seed=None):
        rng = np.random.default_rng(seed)
        pool = self.songs.index[self.songs.get("split", pd.Series(["test"] * self.n_songs)) == "test"].to_numpy()
        if len(pool) == 0:
            pool = self.songs.index.to_numpy()
        out = []
        for i in rng.choice(pool, size=min(n * 3, len(pool)), replace=False):
            w = self._rebuilt_words(int(i))
            if len(w) < 12:
                continue
            L = int(rng.integers(6, 10))
            st = int(rng.integers(0, len(w) - L))
            out.append({"text": " ".join(w[st:st + L]), "title": self.songs.at[i, "Title"],
                        "artist": self.songs.at[i, "Artist"]})
            if len(out) == n:
                break
        return out

    def metrics(self):
        path = os.path.join(self.dir, "test_results.csv")
        rows = []
        if os.path.exists(path):
            res = pd.read_csv(path)
            keep = [c for c in ["method", "query_set", "Hit@1", "Hit@5", "Hit@10", "MRR@10", "nDCG@10", "ms_per_query"] if c in res]
            rows = res[keep].round(4).to_dict(orient="records")
        return {"rows": rows, "hybrid": {"lexical": self.lexical, "alpha": self.alpha},
                "best_val_mrr": self.cfg.get("best_val_mrr@10"), "epochs_run": self.cfg.get("epochs_run"),
                "base_model": self.cfg.get("model_name")}

    def info(self):
        return {"n_songs": self.n_songs, "n_chunks": int(len(self.chunks)), "n_artists": int(self.songs["Artist"].nunique()),
                "hybrid_lexical": self.lexical, "hybrid_alpha": self.alpha, "device": self.device,
                "base_model": self.cfg.get("model_name"), "load_seconds": self.load_seconds,
                "has_original_lyrics": bool(self._raw_lyrics)}
