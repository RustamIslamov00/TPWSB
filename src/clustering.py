from __future__ import annotations

from typing import Dict, Any, Tuple, List
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.common.paths import DATA, OUT, ensure_dirs
from src.common.artifacts import save_artifact, load_latest

TASK = "clustering"
OUT_DIR = OUT / TASK

RATINGS_FILE = DATA / "ratings.csv"
MOVIES_FILE = DATA / "movies.csv"


def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.read_csv(RATINGS_FILE)
    m = pd.read_csv(MOVIES_FILE)
    return r, m


def _select_top_movies(ratings: pd.DataFrame, top_m: int = 500) -> List[int]:
    cnt = ratings.groupby("movieId").size().sort_values(ascending=False)
    return cnt.head(top_m).index.astype(int).tolist()


def _build_matrix(ratings: pd.DataFrame, cols: List[int]) -> pd.DataFrame:
    mat = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    mat = mat.reindex(columns=cols)
    return mat.fillna(0.0)


def train(payload: Dict[str, Any] | None = None) -> None:
    ensure_dirs(OUT_DIR)
    payload = payload or {}
    k = int(payload.get("k", 8))
    top_m = int(payload.get("top_m", 500))

    ratings, _ = _load_data()
    cols = _select_top_movies(ratings, top_m=top_m)
    X = _build_matrix(ratings, cols)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X.values)

    model = {
        "kind": "kmeans_user",
        "k": k,
        "top_cols": cols,
        "labels_": km.labels_.tolist(),
        "centers": km.cluster_centers_.astype(np.float32).tolist(),
    }
    save_artifact(model, TASK, model_key="kmeans")
    (OUT_DIR / "status.txt").write_text("trained\n")


def evaluate() -> None:
    ensure_dirs(OUT_DIR)
    ratings, _ = _load_data()
    cols = _select_top_movies(ratings)
    X = _build_matrix(ratings, cols)
    k = 8
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X.values)
    inertia = float(km.inertia_)
    (OUT_DIR / "metrics.json").write_text(json.dumps({"k": k, "inertia": inertia}, indent=2))


def predict(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs(OUT_DIR)
    payload = payload or {}
    user_id = int(payload.get("user_id", 1))
    top_n = int(payload.get("top_n", 10))

    ratings, movies = _load_data()
    cols = _select_top_movies(ratings)
    X = _build_matrix(ratings, cols)

    if user_id not in X.index:
        return {"user_id": user_id, "cluster": None, "items": []}

    km = KMeans(n_clusters=8, n_init=10, random_state=42)
    km.fit(X.values)
    idx = X.index.get_loc(user_id)
    cluster = int(km.labels_[idx])

    # Recommend top movies in this cluster not yet rated by the user
    df = ratings.copy()
    df = df[df["movieId"].isin(cols)]
    df["cluster"] = km.labels_[X.index.get_indexer(df["userId"].values)]

    seen = set(df[df.userId == user_id].movieId.values.tolist())
    cluster_mean = (df[df.cluster == cluster]
                    .groupby("movieId")["rating"].mean()
                    .sort_values(ascending=False))

    rec_ids = [int(mid) for mid in cluster_mean.index if int(mid) not in seen][:top_n]

    mv = movies.set_index("movieId")
    items = [{"movieId": mid, "title": (mv.loc[mid, "title"] if mid in mv.index else None)} for mid in rec_ids]

    out_csv = OUT_DIR / f"cluster_recs_user_{user_id}.csv"
    pd.DataFrame(items).to_csv(out_csv, index=False)
    return {"user_id": user_id, "cluster": cluster, "csv": str(out_csv), "items": items}
