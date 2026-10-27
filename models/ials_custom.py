from __future__ import annotations

import numpy as np

from .base import SECONDS_PER_DAY, TrainData
from .ials import IALS


def custom_target(
    data: TrainData, beta: float, half_life_days: float, floor: float, mean_shrink: float = 5.0
) -> np.ndarray:
    df = data.interactions
    users = df.user_idx.to_numpy()
    ratings = df.rating.to_numpy(dtype=np.float64)
    mu = ratings.mean()
    sums = np.bincount(users, weights=ratings, minlength=data.n_users)
    counts = np.bincount(users, minlength=data.n_users)
    rbar = (sums + mean_shrink * mu) / (counts + mean_shrink)
    satisfaction = 1.0 + beta * (ratings - rbar[users]) / 4.0
    age_days = (data.now - df.ts.to_numpy(dtype=np.float64)) / SECONDS_PER_DAY
    freshness = floor + (1.0 - floor) * np.exp2(-age_days / half_life_days)
    return satisfaction * freshness


class IALSCustomTarget(IALS):
    def __init__(
        self,
        beta: float = 0.5,
        half_life_days: float = 365.0,
        floor: float = 0.2,
        mean_shrink: float = 5.0,
        **ials_kwargs,
    ):
        super().__init__(**ials_kwargs)
        self.beta = beta
        self.half_life_days = half_life_days
        self.floor = floor
        self.mean_shrink = mean_shrink

    def targets(self, data: TrainData) -> tuple[np.ndarray, np.ndarray]:
        conf, _ = super().targets(data)
        return conf, custom_target(data, self.beta, self.half_life_days, self.floor, self.mean_shrink)
