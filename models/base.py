from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

SECONDS_PER_DAY = 86_400


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainData:
    interactions: pd.DataFrame
    n_users: int
    n_items: int
    now: int
    item_meta: pd.DataFrame | None = None

    def csr(self, values: np.ndarray | None = None) -> sp.csr_matrix:
        df = self.interactions
        data = np.ones(len(df), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
        m = sp.csr_matrix(
            (data, (df.user_idx.to_numpy(), df.item_idx.to_numpy())),
            shape=(self.n_users, self.n_items),
        )
        m.sum_duplicates()
        return m


def sparse_to_torch(m: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = m.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, coo.shape, device=device, check_invariants=False).coalesce()


def item_gram(x: sp.csr_matrix, device: torch.device, user_weights: np.ndarray | None = None) -> torch.Tensor:
    xw = x if user_weights is None else sp.diags(user_weights.astype(np.float32)) @ x
    return torch.sparse.mm(sparse_to_torch(x.T, device), sparse_to_torch(xw, device)).to_dense()


def dense_rows(m: sp.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(m[rows].toarray().astype(np.float32)).to(device)


class Recommender:
    def fit(self, data: TrainData) -> "Recommender":
        raise NotImplementedError

    def score(self, users: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def recommend(
        self,
        users: np.ndarray,
        k: int = 10,
        exclude: sp.csr_matrix | None = None,
        candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        scores = self.score(users).float().clone()
        if candidates is not None:
            scores[:, ~torch.as_tensor(candidates, device=scores.device)] = -torch.inf
        if exclude is not None:
            sub = exclude[users].tocoo()
            scores[torch.as_tensor(sub.row, device=scores.device), torch.as_tensor(sub.col, device=scores.device)] = -torch.inf
        return torch.topk(scores, k, dim=1).indices.cpu().numpy()
