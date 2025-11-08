from __future__ import annotations

from typing import Dict, Any, List, Tuple
import json
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.paths import DATA, OUT, ensure_dirs
from src.common.artifacts import save_artifact, load_latest

TASK = "nlp"
OUT_DIR = OUT / TASK

MOVIES_FILE = DATA / "movies.csv"
TAGS_FILE = DATA / "tags.csv"
NLP_FILE = DATA / "NLP-mergedDB.csv"

TEXT_COL_ENV = os.getenv("NLP_TEXT_COL")  # optional override if using NLP-mergedDB.csv


def _load_text() -> Tuple[pd.Series, pd.DataFrame]:
    if NLP_FILE.exists():
        df = pd.read_csv(NLP_FILE)
        text_col = TEXT_COL_ENV or ("text" if "text" in df.columns else None)
        if not text_col:
            for c in ("overview", "content", "tagline", "description", "plot", "synopsis"):
                if c in df.columns:
                    text_col = c
                    break
        if text_col:
            movies = pd.read_csv(MOVIES_FILE)
            if "movieId" in df.columns:
                merged = df.merge(movies[["movieId", "title"]], on="movieId", how="left")
                movie_ids = merged.get("movieId", pd.Series(range(len(merged))))
            else:
                merged = df
                movie_ids = pd.Series(range(len(merged)))
            return merged[text_col].fillna("").astype(str), pd.DataFrame({"movieId": movie_ids, "title": merged.get("title")})
        # fall through to tag-based text if no suitable column

    tags = pd.read_csv(TAGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    agg = tags.groupby("movieId")["tag"].apply(lambda s: " ".join(map(str, s))).reset_index()
    base = movies.merge(agg, on="movieId", how="left").fillna("")
    text = (base["title"].astype(str) + " " + base["genres"].astype(str) + " " + base["tag"].astype(str)).str.strip()
    return text, base[["movieId", "title"]]


def _build_tfidf(texts: pd.Series) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts.values)
    return vec, X


def train() -> None:
    ensure_dirs(OUT_DIR)
    texts, meta = _load_text()
    vec, X = _build_tfidf(texts)
    model = {
        "kind": "tfidf",
        "vectorizer": vec,
        "X": X,
        "movie_ids": meta["movieId"].astype(int).tolist(),
        "titles": meta.get("title").astype(str).fillna("").tolist(),
    }
    save_artifact(model, TASK, model_key="tfidf")
    (OUT_DIR / "status.txt").write_text("trained\n")


def evaluate() -> None:
    ensure_dirs(OUT_DIR)
    texts, _ = _load_text()
    n_docs = int(len(texts))
    avg_len = float(np.mean(texts.str.split().map(len))) if len(texts) else 0.0
    (OUT_DIR / "metrics.json").write_text(json.dumps({"n_docs": n_docs, "avg_tokens": avg_len}, indent=2))


def _load_model() -> Dict[str, Any]:
    texts, meta = _load_text()
    vec, X = _build_tfidf(texts)
    return {
        "vectorizer": vec,
        "X": X,
        "movie_ids": meta["movieId"].astype(int).tolist(),
        "titles": meta.get("title").astype(str).fillna("").tolist(),
    }


def predict(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs(OUT_DIR)
    payload = payload or {}
    top_n = int(payload.get("top_n", 10))

    model = _load_model()
    vec: TfidfVectorizer = model["vectorizer"]
    X: sparse.csr_matrix = model["X"]
    ids: List[int] = model["movie_ids"]
    titles: List[str] = model["titles"]

    items: List[Dict[str, Any]] = []

    if "query" in payload and payload["query"]:
        q = payload["query"]
        qv = vec.transform([q])
        sims = cosine_similarity(qv, X).ravel()
        idxs = np.argsort(-sims)[:top_n]
        for i in idxs:
            items.append({"movieId": int(ids[i]), "title": titles[i], "score": float(sims[i])})

    elif "movie_id" in payload:
        mid = int(payload["movie_id"])
        if mid in ids:
            i = ids.index(mid)
            sims = cosine_similarity(X[i], X).ravel()
            sims[i] = -1.0
            idxs = np.argsort(-sims)[:top_n]
            for j in idxs:
                items.append({"movieId": int(ids[j]), "title": titles[j], "score": float(sims[j])})
        else:
            items = []
    else:
        items = []

    out_csv = OUT_DIR / "nlp_results.csv"
    pd.DataFrame(items).to_csv(out_csv, index=False)
    return {"top_n": top_n, "csv": str(out_csv), "items": items}
