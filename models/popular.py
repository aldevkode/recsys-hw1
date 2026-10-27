from __future__ import annotations

import numpy as np
import torch

from .base import SECONDS_PER_DAY, Recommender, TrainData, get_device


class Popular(Recommender):
    def __init__(self, window_days: float | None = 90, device: str | None = None):
        self.window_days = window_days
        self.device = get_device(device)

    def fit(self, data: TrainData) -> "Popular":
        df = data.interactions
        all_time = np.bincount(df.item_idx, minlength=data.n_items).astype(np.float64)
        if self.window_days is not None:
            df = df[df.ts >= data.now - self.window_days * SECONDS_PER_DAY]
        recent = np.bincount(df.item_idx, minlength=data.n_items).astype(np.float64)
        pop = recent + all_time / (all_time.max() + 1.0)
        self.pop = torch.as_tensor(pop, dtype=torch.float32, device=self.device)
        return self

    def score(self, users: np.ndarray) -> torch.Tensor:
        return self.pop.unsqueeze(0).expand(len(users), -1)
