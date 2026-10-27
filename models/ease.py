from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .base import Recommender, TrainData, dense_rows, get_device, item_gram


def ease_weights(x: sp.csr_matrix, reg: float, device: torch.device, block: int = 4096) -> torch.Tensor:
    gram = item_gram(x, device).double()
    gram.diagonal().add_(reg)
    chol = torch.linalg.cholesky(gram)
    del gram
    n = chol.shape[0]
    b = torch.empty(n, n, dtype=torch.float32, device=chol.device)
    for start in range(0, n, block):
        cols = torch.arange(start, min(start + block, n), device=chol.device)
        eye = torch.zeros(n, len(cols), dtype=chol.dtype, device=chol.device)
        eye[cols, torch.arange(len(cols), device=chol.device)] = 1.0
        p = torch.cholesky_solve(eye, chol)
        p_diag = p[cols, torch.arange(len(cols), device=chol.device)]
        b[:, cols] = (p / -p_diag).float()
        b[cols, cols] = 0.0
    return b


class EASE(Recommender):
    def __init__(self, reg: float = 500.0, device: str | None = None):
        self.reg = reg
        self.device = get_device(device)

    def fit(self, data: TrainData) -> "EASE":
        x = data.csr()
        x.data[:] = 1.0
        self.x = x
        self.b = ease_weights(x, self.reg, self.device)
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray) -> torch.Tensor:
        return dense_rows(self.x, users, self.device) @ self.b
