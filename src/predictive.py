from __future__ import annotations

from typing import Dict, Any, List, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

from src.common.paths import DATA, OUT, ensure_dirs
from src.common.artifacts import save_artifact, load_latest

TASK = "predictive"
OUT_DIR = OUT / TASK

RATINGS_FILE = DATA / "ratings.csv"
MOVIES_FILE = DATA / "movies.csv"


def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.read_csv(RATINGS_FILE)
    m = pd.read_csv(MOVIES_FILE)
    return r, m


def _build_item_cf(ratings: pd.DataFrame) -> Dict[str, Any]:
    mat = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    item_means = mat.mean(axis=0)
    centered = mat.subtract(item_means, axis=1).fillna(0.0)
    sims = cosine_similarity(centered.T)
    item_ids = centered.columns.to_list()
    id2idx = {int(mid): i for i, mid in enumerate(item_ids)}
    return {
        "kind": "itemcf",
        "item_ids": item_ids,
        "id2idx": id2idx,
        "sims": sims.astype(np.float32),
        "item_means": item_means.astype(np.float32).to_dict(),
    }


def _predict_rating_itemcf(model: Dict[str, Any], user_ratings: pd.Series, target_mid: int) -> float:
    id2idx = model["id2idx"]
    if target_mid not in id2idx:
        return float(model["item_means"].get(str(target_mid), np.nan))
    target_idx = id2idx[target_mid]
    sims = model["sims"][target_idx]
    num = 0.0
    den = 0.0
    for mid, r in user_ratings.dropna().items():
        mid = int(mid)
        if mid in id2idx and mid != target_mid:
            s = float(sims[id2idx[mid]])
            if s != 0.0 and np.isfinite(s):
                num += s * float(r)
                den += abs(s)
    if den == 0.0:
        return float(model["item_means"].get(str(target_mid), np.nan))
    return float(num / den)


def _train_test_split(r: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" in r.columns:
        r = r.sort_values("timestamp")
    tr_parts, te_parts = [], []
    for _, grp in r.groupby("userId"):
        n = len(grp)
        k = max(1, int(n * test_ratio))
        tr_parts.append(grp.iloc[:-k])
        te_parts.append(grp.iloc[-k:])
    tr = pd.concat(tr_parts, ignore_index=True) if tr_parts else r
    te = pd.concat(te_parts, ignore_index=True) if te_parts else r.iloc[0:0]
    return tr, te


def train() -> None:
    ensure_dirs(OUT_DIR)
    ratings, _ = _load_data()
    model = _build_item_cf(ratings)
    save_artifact(model, TASK, model_key="itemcf")
    (OUT_DIR / "status.txt").write_text("trained\n")


def evaluate() -> None:
    ensure_dirs(OUT_DIR)
    ratings, _ = _load_data()
    tr, te = _train_test_split(ratings)
    model = _build_item_cf(tr)

    preds, truths = [], []
    mat_tr = tr.pivot_table(index="userId", columns="movieId", values="rating")
    for uid, grp in te.groupby("userId"):
        user_ratings = mat_tr.loc[uid] if uid in mat_tr.index else pd.Series(dtype=float)
        for _, row in grp.iterrows():
            p = _predict_rating_itemcf(model, user_ratings, int(row.movieId))
            if np.isfinite(p):
                preds.append(float(p))
                truths.append(float(row.rating))
    rmse = float(np.sqrt(mean_squared_error(truths, preds))) if preds else float("nan")
    (OUT_DIR / "metrics.json").write_text(json.dumps({"rmse": rmse}, indent=2))


def predict(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs(OUT_DIR)
    p = payload or {}
    top_n = int(p.get("top_n", 10))
    user_id = int(p.get("user_id", 1))

    ratings, movies = _load_data()
    _ = load_latest(TASK, model_key_prefix="itemcf")
    model = _build_item_cf(ratings)

    mat = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    if user_id not in mat.index:
        out_csv = OUT_DIR / f"recs_user_{user_id}.csv"
        pd.DataFrame(columns=["movieId", "title", "score"]).to_csv(out_csv, index=False)
        return {"user_id": user_id, "top_n": top_n, "csv": str(out_csv), "items": []}

    user_ratings = mat.loc[user_id]
    candidates = [int(mid) for mid in mat.columns if np.isnan(user_ratings.get(mid))]
    scores: List[Tuple[int, float]] = []
    for mid in candidates:
        s = _predict_rating_itemcf(model, user_ratings, mid)
        if np.isfinite(s):
            scores.append((mid, float(s)))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[: max(0, top_n)]

    mv = movies.set_index("movieId")
    rows: List[Dict[str, Any]] = []
    for mid, sc in top:
        title = mv.loc[mid, "title"] if mid in mv.index else None
        try:
            fs = float(sc)
            if not np.isfinite(fs):
                continue
        except Exception:
            continue
        rows.append({"movieId": int(mid), "title": title, "score": fs})

    df = pd.DataFrame(rows, columns=["movieId", "title", "score"])
    out_csv = OUT_DIR / f"recs_user_{user_id}.csv"
    df.to_csv(out_csv, index=False)
    return {"user_id": int(user_id), "top_n": int(top_n), "csv": str(out_csv), "items": rows}
