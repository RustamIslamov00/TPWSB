from __future__ import annotations

from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.paths import DATA, OUT, ensure_dirs
from src.common.artifacts import save_artifact, load_latest

TASK = "classification"
OUT_DIR = OUT / TASK

RATINGS_FILE = DATA / "ratings.csv"


def _load_data() -> pd.DataFrame:
    return pd.read_csv(RATINGS_FILE)


def _make_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = (df["rating"] >= 4.0).astype(int).values
    pairs = [("u", int(u)) for u in df["userId"].values]
    pairs += [("m", int(m)) for m in df["movieId"].values]
    # FeatureHasher expects a sequence of feature mappings per sample; build efficiently
    X_dicts = [{"u": int(u), "m": int(m)} for u, m in zip(df["userId"].values, df["movieId"].values)]
    hasher = FeatureHasher(n_features=2**18, input_type="dict")
    X = hasher.transform(X_dicts)
    return X, y


def train() -> None:
    ensure_dirs(OUT_DIR)
    df = _load_data()
    X, y = _make_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(X_tr, y_tr)
    save_artifact({"model": clf}, TASK, model_key="logreg")
    (OUT_DIR / "status.txt").write_text("trained\n")


def evaluate() -> None:
    ensure_dirs(OUT_DIR)
    df = _load_data()
    X, y = _make_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    yhat = (p >= 0.5).astype(int)
    acc = float(accuracy_score(y_te, yhat))
    try:
        auc = float(roc_auc_score(y_te, p))
    except Exception:
        auc = float("nan")
    (OUT_DIR / "metrics.json").write_text(json.dumps({"accuracy": acc, "roc_auc": auc}, indent=2))


def predict(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs(OUT_DIR)
    payload = payload or {}
    user_id = int(payload.get("user_id", 1))
    movie_ids: List[int] = payload.get("movie_ids") or []
    if not movie_ids:
        df_all = _load_data()
        seen = set(df_all[df_all.userId == user_id].movieId.values.tolist())
        all_m = sorted(set(df_all.movieId.values.tolist()))
        movie_ids = [m for m in all_m if m not in seen][:50]

    X_dicts = [{"u": user_id, "m": int(m)} for m in movie_ids]
    hasher = FeatureHasher(n_features=2**18, input_type="dict")
    X = hasher.transform(X_dicts)

    # Build fresh model from full data for now
    df = _load_data()
    X_all, y_all = _make_xy(df)
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(X_all, y_all)

    p = clf.predict_proba(X)[:, 1]
    rows = [{"movieId": int(m), "prob_like": float(s)} for m, s in zip(movie_ids, p)]
    rows.sort(key=lambda x: x["prob_like"], reverse=True)
    out_csv = OUT_DIR / f"pred_user_{user_id}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return {"user_id": user_id, "n": len(rows), "csv": str(out_csv), "items": rows[:10]}
