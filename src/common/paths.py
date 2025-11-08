from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

_MARKERS = {".git", "pyproject.toml", "requirements.txt", "src", "data", "models"}

def _walk_up(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        for m in _MARKERS:
            if (parent / m).exists():
                return parent
    return Path.cwd()

def project_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    return _walk_up(here.parent)

ROOT: Path = project_root()
DATA: Path = (ROOT / "data").resolve()
MODELS: Path = (ROOT / "models").resolve()
OUT: Path = (ROOT / "out").resolve()

def ensure_dirs(*paths: Iterable[Path] | Path) -> None:
    def _iter():
        for p in paths:
            if isinstance(p, Iterable) and not isinstance(p, (str, bytes, Path)):
                for x in p:
                    yield Path(x)
            else:
                yield Path(p)  # type: ignore[arg-type]
    for p in _iter():
        Path(p).mkdir(parents=True, exist_ok=True)
