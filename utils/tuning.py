from __future__ import annotations

from typing import Callable

import optuna
import pandas as pd

from models.base import Recommender
from utils.data import EvalSplit
from utils.evaluation import fit_evaluate


def tune(
    build: Callable[[optuna.Trial], Recommender],
    split: EvalSplit,
    n_trials: int,
    metric: str = "ndcg",
    k: int = 10,
    seed: int = 42,
    enqueue: list[dict] | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> optuna.Study:
    sampler = sampler or optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 4))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for params in enqueue or []:
        study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        model = build(trial)
        res, fit_time = fit_evaluate(model, split, k)
        for name, value in res.metrics.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("fit_time", fit_time)
        return res.metrics[metric]

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in df.columns]
    return df.sort_values("value", ascending=False)
