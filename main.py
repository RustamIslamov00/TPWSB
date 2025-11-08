from __future__ import annotations

import argparse
from typing import Dict, Any, Callable

from src import forecasting, predictive, classification, clustering, nlp

TaskFn = Callable[[Dict[str, Any] | None], Dict[str, Any] | None]

TASKS: dict[str, dict[str, TaskFn]] = {
    "forecasting": {
        "train": lambda _: forecasting.train(),
        "evaluate": lambda _: forecasting.evaluate(),
        "predict": lambda p=None: forecasting.predict(p or {"horizon": 6}),
    },
    "predictive": {
        "train": lambda _: predictive.train(),
        "evaluate": lambda _: predictive.evaluate(),
        "predict": lambda p=None: predictive.predict(p or {"user_id": 1, "top_n": 10}),
    },
    "classification": {
        "train": lambda _: classification.train(),
        "evaluate": lambda _: classification.evaluate(),
        "predict": lambda p=None: classification.predict(p or {"user_id": 1, "movie_ids": [1,2,3,4,5]}),
    },
    "clustering": {
        "train": lambda _: clustering.train(),
        "evaluate": lambda _: clustering.evaluate(),
        "predict": lambda p=None: clustering.predict(p or {"user_id": 1, "top_n": 10}),
    },
    "nlp": {
        "train": lambda _: nlp.train(),
        "evaluate": lambda _: nlp.evaluate(),
        "predict": lambda p=None: nlp.predict(p or {"query": "space adventure", "top_n": 5}),
    },
}


def run_task(task: str, mode: str) -> None:
    fn = TASKS[task][mode]
    res = fn(None)
    if res is not None:
        print({"task": task, "mode": mode, "result": res})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=[*TASKS.keys(), "all"], default="all")
    ap.add_argument("--mode", choices=["train", "evaluate", "predict", "all"], default="all")
    args = ap.parse_args()

    tasks = list(TASKS.keys()) if args.task == "all" else [args.task]
    modes = ["train", "evaluate", "predict"] if args.mode == "all" else [args.mode]

    for t in tasks:
        for m in modes:
            run_task(t, m)


if __name__ == "__main__":
    main()
