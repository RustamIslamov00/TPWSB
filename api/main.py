from __future__ import annotations

from typing import Any, Dict
import importlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Movie IDSS API", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    payload: Dict[str, Any] | None = None

TASKS = {
    "forecasting": "forecasting",
    "predictive": "predictive",
    "classification": "classification",
    "clustering": "clustering",
    "nlp": "nlp",
}

import importlib, sys

def _get_module(task: str):
    fullname = f"src.{task}"
    try:
        if fullname in sys.modules:
            mod = importlib.reload(sys.modules[fullname])
        else:
            mod = importlib.import_module(fullname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module import error: {e}")
    for fn in ("train", "evaluate", "predict"):
        if not hasattr(mod, fn):
            raise HTTPException(status_code=500, detail=f"Task '{task}' missing function: {fn}")
    return mod

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/train/{task}")
def train(task: str) -> dict:
    key = TASKS.get(task)
    if not key:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task}")
    mod = _get_module(key)
    try:
        mod.train()
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
    return {"task": task, "result": "trained"}


@app.post("/evaluate/{task}")
def evaluate(task: str) -> dict:
    key = TASKS.get(task)
    if not key:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task}")
    mod = _get_module(key)
    try:
        mod.evaluate()
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
    return {"task": task, "result": "evaluated"}


@app.post("/predict/{task}")
def predict(task: str, req: PredictRequest) -> dict:
    key = TASKS.get(task)
    if not key:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task}")
    mod = _get_module(key)
    try:
        result = mod.predict(req.payload or {})
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
    return {"task": task, "result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
