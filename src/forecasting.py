from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.paths import DATA, OUT, ensure_dirs
from src.common.artifacts import save_artifact, load_latest

TASK = "forecasting"
OUT_DIR = OUT / TASK
DATA_FILE = DATA / "monthly_time_series.csv"

# column hints via env, with sane fallbacks in _load_data
DATE_COL = os.getenv("FCAST_DATE_COL")
VALUE_COL = os.getenv("FCAST_VALUE_COL")


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    # date column
    date_col = None
    if DATE_COL and DATE_COL in df.columns:
        date_col = DATE_COL
    elif "date" in df.columns:
        date_col = "date"

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    # value column
    val_col = None
    if VALUE_COL and VALUE_COL in df.columns:
        val_col = VALUE_COL
    elif "avg_rating" in df.columns:
        val_col = "avg_rating"
    elif "value" in df.columns:
        val_col = "value"
    else:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(nums) == 1:
            val_col = nums[0]

    if not val_col:
        raise ValueError("set FCAST_VALUE_COL or provide a detectable numeric column, e.g. avg_rating")

    df = df.rename(columns={val_col: "value"})
    if date_col:
        df = df.rename(columns={date_col: "date"})
    return df.reset_index(drop=True)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(df) * 0.8)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def _predict_naive_last(history: pd.Series, horizon: int) -> List[float]:
    last_val = float(history.iloc[-1])
    return [last_val] * int(horizon)


def train() -> None:
    ensure_dirs(OUT_DIR)
    _ = _load_data()
    model = {"kind": "naive_last"}
    save_artifact(model, TASK, model_key="naive")
    (OUT_DIR / "status.txt").write_text("trained\n")


def evaluate() -> None:
    ensure_dirs(OUT_DIR)
    df = _load_data()
    tr, te = _split(df)
    y_hist = tr["value"]
    y_true = te["value"].to_list()
    preds = _predict_naive_last(y_hist, len(y_true))
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    pd.DataFrame({
        "date": te.get("date", pd.Series(range(len(te)))).values,
        "y_true": y_true,
        "y_pred": preds,
    }).to_csv(OUT_DIR / "pred_vs_true.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(pd.Series({"rmse": rmse, "mae": mae}).to_json())


def predict(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs(OUT_DIR)
    df = _load_data()
    horizon = int((payload or {}).get("horizon", 12))
    _ = load_latest(TASK)
    preds = _predict_naive_last(df["value"], horizon)
    out_csv = OUT_DIR / "forecast_next.csv"
    pd.DataFrame({"step": range(1, horizon + 1), "y_pred": preds}).to_csv(out_csv, index=False)
    return {"horizon": horizon, "csv": str(out_csv)}
