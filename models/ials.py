from __future__ import annotations

import numpy as np
import torch

from .base import Recommender, TrainData, get_device


def cg_solve(
    fixed: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    conf_minus_one: torch.Tensor,
    conf_times_pref: torch.Tensor,
    x0: torch.Tensor,
    reg: float,
    steps: int,
) -> torch.Tensor:
    gram = fixed.T @ fixed + reg * torch.eye(fixed.shape[1], device=fixed.device, dtype=fixed.dtype)
    f_obs = fixed[cols]

    def matvec(v: torch.Tensor) -> torch.Tensor:
        out = v @ gram
        coef = (v[rows] * f_obs).sum(1) * conf_minus_one
        return out.index_add_(0, rows, coef.unsqueeze(1) * f_obs)

    b = torch.zeros_like(x0).index_add_(0, rows, conf_times_pref.unsqueeze(1) * f_obs)
    x = x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = (r * r).sum(1)
    for _ in range(steps):
        ap = matvec(p)
        a = rs / (p * ap).sum(1).clamp_min(1e-20)
        x.addcmul_(a.unsqueeze(1), p)
        r.addcmul_(a.unsqueeze(1), ap, value=-1.0)
        rs_new = (r * r).sum(1)
        p = r + (rs_new / rs.clamp_min(1e-20)).unsqueeze(1) * p
        rs = rs_new
    return x


class IALS(Recommender):
    def __init__(
        self,
        factors: int = 64,
        reg: float = 1.0,
        alpha: float = 10.0,
        iterations: int = 15,
        cg_steps: int = 3,
        init_std: float = 0.01,
        seed: int = 42,
        device: str | None = None,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.init_std = init_std
        self.seed = seed
        self.device = get_device(device)

    def targets(self, data: TrainData) -> tuple[np.ndarray, np.ndarray]:
        n = len(data.interactions)
        return np.full(n, 1.0 + self.alpha), np.ones(n)

    def fit(self, data: TrainData) -> "IALS":
        conf, pref = self.targets(data)
        dev = self.device
        users = torch.as_tensor(data.interactions.user_idx.to_numpy(), dtype=torch.long, device=dev)
        items = torch.as_tensor(data.interactions.item_idx.to_numpy(), dtype=torch.long, device=dev)
        c_minus_1 = torch.as_tensor(conf - 1.0, dtype=torch.float32, device=dev)
        c_times_p = torch.as_tensor(conf * pref, dtype=torch.float32, device=dev)

        gen = torch.Generator(device=dev).manual_seed(self.seed)
        x = torch.randn(data.n_users, self.factors, generator=gen, device=dev) * self.init_std
        y = torch.randn(data.n_items, self.factors, generator=gen, device=dev) * self.init_std
        for _ in range(self.iterations):
            x = cg_solve(y, users, items, c_minus_1, c_times_p, x, self.reg, self.cg_steps)
            y = cg_solve(x, items, users, c_minus_1, c_times_p, y, self.reg, self.cg_steps)
        self.user_factors, self.item_factors = x, y
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray) -> torch.Tensor:
        u = torch.as_tensor(users, dtype=torch.long, device=self.device)
        return self.user_factors[u] @ self.item_factors.T
