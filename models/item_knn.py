from __future__ import annotations

import numpy as np
import torch

from .base import SECONDS_PER_DAY, Recommender, TrainData, dense_rows, get_device, item_gram


class ItemKNN(Recommender):
    def __init__(
        self,
        k: int = 100,
        shrink: float = 10.0,
        alpha: float = 0.5,
        iuf: bool = False,
        half_life_days: float | None = None,
        device: str | None = None,
    ):
        self.k = k
        self.shrink = shrink
        self.alpha = alpha
        self.iuf = iuf
        self.half_life_days = half_life_days
        self.device = get_device(device)

    def fit(self, data: TrainData) -> "ItemKNN":
        x = data.csr()
        x.data[:] = 1.0
        if self.half_life_days is None:
            self.history = x
        else:
            age_days = (data.now - data.interactions.ts.to_numpy(dtype=np.float64)) / SECONDS_PER_DAY
            self.history = data.csr(np.exp2(-age_days / self.half_life_days))
        user_weight = 1.0 / np.log2(2.0 + np.asarray(x.sum(axis=1)).ravel()) if self.iuf else None
        co = item_gram(x, self.device, user_weight)
        norm = torch.diagonal(co).clone()
        co.fill_diagonal_(0.0)
        denom = norm.pow(self.alpha).unsqueeze(1) * norm.pow(1.0 - self.alpha).unsqueeze(0) + self.shrink
        sim = co.div_(denom.clamp_min_(1e-12))
        del denom
        k = min(self.k, data.n_items - 1)
        vals, idx = torch.topk(sim, k, dim=0)
        self.sim = torch.zeros_like(sim).scatter_(0, idx, vals)
        del sim, co
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray) -> torch.Tensor:
        return dense_rows(self.history, users, self.device) @ self.sim
