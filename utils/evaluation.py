from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from models.base import Recommender
from utils.data import EvalSplit


@dataclass
class EvalResult:
    metrics: dict[str, float]
    per_user: pd.DataFrame
    top: np.ndarray = field(repr=False)


def ranking_metrics(top: np.ndarray, relevant_rows) -> dict[str, np.ndarray]:
    n, k = top.shape
    rel = relevant_rows.tocoo()
    n_items = relevant_rows.shape[1]
    hits = np.isin(np.arange(n, dtype=np.int64)[:, None] * n_items + top, rel.row.astype(np.int64) * n_items + rel.col)
    n_rel = np.asarray(relevant_rows.getnnz(axis=1), dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(1)
    idcg = np.cumsum(discounts)[np.minimum(n_rel, k).astype(int) - 1]
    first_hit = np.where(hits.any(1), hits.argmax(1) + 1, np.inf)
    return {
        "ndcg": dcg / idcg,
        "recall": hits.sum(1) / n_rel,
        "hitrate": hits.any(1).astype(np.float64),
        "mrr": 1.0 / first_hit,
    }


def evaluate(model: Recommender, split: EvalSplit, k: int = 10, batch_size: int = 2048) -> EvalResult:
    tops = []
    for start in range(0, len(split.users), batch_size):
        users = split.users[start : start + batch_size]
        tops.append(model.recommend(users, k, exclude=split.history, candidates=split.candidates))
    top = np.vstack(tops)
    per_user = pd.DataFrame(ranking_metrics(top, split.relevant[split.users]), index=split.users)
    metrics = per_user.mean().to_dict()
    metrics["coverage"] = len(np.unique(top)) / split.candidates.sum()
    pop = np.bincount(split.train.interactions.item_idx, minlength=split.train.n_items)
    metrics["mean_rec_popularity"] = float(pop[top].mean())
    return EvalResult(metrics, per_user, top)


def fit_evaluate(model: Recommender, split: EvalSplit, k: int = 10) -> tuple[EvalResult, float]:
    t0 = time.perf_counter()
    model.fit(split.train)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - t0
    return evaluate(model, split, k), fit_time


def bootstrap_ci(per_user: dict[str, pd.Series], n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    names = list(per_user)
    values = np.column_stack([per_user[m].to_numpy() for m in names])
    rng = np.random.default_rng(seed)
    means = np.stack([values[rng.integers(0, len(values), len(values))].mean(0) for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame({"mean": values.mean(0), "ci_low": lo, "ci_high": hi}, index=names)


def paired_bootstrap_diff(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    diff = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_le_0": float((means <= 0).mean()),
    }
