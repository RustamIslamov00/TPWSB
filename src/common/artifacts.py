"""
Artifact helpers: save/load model pickle and meta JSON with versioned names.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import sklearn

from src.common.paths import MODELS, ensure_dirs


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _fname(task: str, model_key: str, ts: Optional[str] = None) -> str:
    ts = ts or _ts()
    return f"{task}_{model_key}_v{ts}"


def save_artifact(obj: Any, task: str, model_key: str, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    name = _fname(task, model_key)
    base = MODELS / task
    ensure_dirs(base)
    pkl_path = base / f"{name}.pkl"
    meta_path = base / f"{name}.meta.json"

    joblib.dump(obj, pkl_path)
    meta = {
        "created": datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "libs": {"sklearn": getattr(sklearn, "__version__", "unknown")},
        "task": task,
        "model_key": model_key,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2))
    return pkl_path


def load_latest(task: str, model_key_prefix: Optional[str] = None) -> Path:
    base = MODELS / task
    candidates = sorted(base.glob("*.pkl"))
    if model_key_prefix:
        candidates = [p for p in candidates if p.name.startswith(f"{task}_{model_key_prefix}_v")]
    if not candidates:
        raise FileNotFoundError(f"No artifacts found for task={task}")
    return candidates[-1]
