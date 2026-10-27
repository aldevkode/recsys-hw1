from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .base import SECONDS_PER_DAY, Recommender, TrainData, get_device, sparse_to_torch

USER_FIELDS = ("user_id", "u_cat", "u_last_cat", "u_store")
ITEM_FIELDS = ("item_id", "i_cat", "i_store", "i_main_cat", "i_title", "i_rated")
USER_CONTEXT_FIELDS = ("c_user_activity", "c_user_recency")
ITEM_CONTEXT_FIELDS = ("c_item_age", "c_item_pop", "c_item_pop_recent")
ALL_FIELDS = USER_FIELDS + ITEM_FIELDS + USER_CONTEXT_FIELDS + ITEM_CONTEXT_FIELDS
STOPWORDS = {"the", "a", "an", "of", "and", "for", "with", "to", "in", "on", "by", "or", "from", "at", "is"}

AGE_EDGES_DAYS = (90, 365, 730, 1460, 2920)
RECENCY_EDGES_DAYS = (30, 90, 180, 365, 730)
MAX_LOG2_BUCKET = 12
CONTEXT_BUCKETS = {
    "c_user_activity": [str(b) for b in range(MAX_LOG2_BUCKET + 1)],
    "c_user_recency": ["first"] + [f"<{e}d" for e in RECENCY_EDGES_DAYS] + [f">={RECENCY_EDGES_DAYS[-1]}d"],
    "c_item_age": ["na", "future"] + [f"<{e}d" for e in AGE_EDGES_DAYS] + [f">={AGE_EDGES_DAYS[-1]}d"],
    "c_item_pop": [str(b) for b in range(MAX_LOG2_BUCKET + 1)],
    "c_item_pop_recent": [str(b) for b in range(MAX_LOG2_BUCKET + 1)],
}


class _Features:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []

    def _index(self, field: str, values) -> np.ndarray:
        return np.array([self.vocab.setdefault(f"{field}={v}", len(self.vocab)) for v in values], dtype=np.int64)

    def add_lists(self, field: str, values: list[list[str]]) -> None:
        lengths = np.array([len(v) for v in values])
        self.rows.append(np.repeat(np.arange(len(values)), lengths))
        self.cols.append(self._index(field, [v for vs in values for v in vs]))
        self.vals.append(np.repeat(1.0 / np.maximum(lengths, 1), lengths))

    def add_matrix(self, field: str, mat: sp.spmatrix, names: list[str]) -> None:
        coo = mat.tocoo()
        self.rows.append(coo.row.astype(np.int64))
        self.cols.append(self._index(field, names)[coo.col])
        self.vals.append(coo.data.astype(np.float64))

    def to_csr(self, n_rows: int, n_cols: int) -> sp.csr_matrix:
        if not self.rows:
            return sp.csr_matrix((n_rows, n_cols), dtype=np.float32)
        return sp.csr_matrix(
            (np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
            shape=(n_rows, n_cols),
            dtype=np.float32,
        )


def _multi_hot(values: list[list[str]]) -> tuple[sp.csr_matrix, list[str]]:
    names = sorted({v for vs in values for v in vs})
    pos = {v: k for k, v in enumerate(names)}
    rows = np.repeat(np.arange(len(values)), [len(vs) for vs in values])
    cols = np.array([pos[v] for vs in values for v in vs], dtype=np.int64)
    m = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(values), len(names)))
    return m, names


class FMFeatureBuilder:
    def __init__(
        self,
        fields: tuple[str, ...] = ALL_FIELDS,
        min_store_items: int = 5,
        min_token_items: int = 20,
        max_token_share: float = 0.3,
        last_n: int = 5,
    ):
        self.fields = tuple(fields)
        self.min_store_items = min_store_items
        self.min_token_items = min_token_items
        self.max_token_share = max_token_share
        self.last_n = last_n

    def item_values(self, data: TrainData) -> dict[str, list[list[str]]]:
        meta = data.item_meta.reindex(range(data.n_items))
        n = data.n_items
        out: dict[str, list[list[str]]] = {"item_id": [[str(i)] for i in range(n)]}

        cats = []
        for cs in meta.categories:
            cs = list(cs) if isinstance(cs, (list, np.ndarray)) else []
            cats.append([">".join(cs[: d + 1]) for d in range(len(cs))])
        out["i_cat"] = cats

        store_counts = meta.store.value_counts()
        frequent = set(store_counts[store_counts >= self.min_store_items].index)
        out["i_store"] = [[s if s in frequent else "other"] if isinstance(s, str) and s else ["na"] for s in meta.store]
        out["i_main_cat"] = [[s] if isinstance(s, str) and s else ["na"] for s in meta.main_category]
        out["i_rated"] = [[s] if isinstance(s, str) and s else ["na"] for s in meta.rated]

        tokens = [
            sorted({t for t in re.findall(r"[a-z0-9]+", t.lower()) if len(t) > 1 and t not in STOPWORDS})
            for t in meta.title.fillna("").astype(str)
        ]
        doc_freq = Counter(t for ts in tokens for t in ts)
        vocab = {t for t, c in doc_freq.items() if self.min_token_items <= c <= self.max_token_share * n}
        out["i_title"] = [[t for t in ts if t in vocab] for ts in tokens]
        return out

    def build(self, data: TrainData) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str]]:
        vocab: dict[str, int] = {}
        items, users = _Features(vocab), _Features(vocab)
        item_values = self.item_values(data)
        for field in ITEM_FIELDS:
            if field in self.fields:
                items.add_lists(field, item_values[field])

        df = data.interactions
        x = data.csr()
        x.data[:] = 1.0
        last = df.sort_values("ts", kind="stable").groupby("user_idx").tail(self.last_n)
        x_last = sp.csr_matrix(
            (np.ones(len(last), dtype=np.float32), (last.user_idx, last.item_idx)), shape=(data.n_users, data.n_items)
        )
        if "user_id" in self.fields:
            users.add_lists("user_id", [[str(u)] for u in range(data.n_users)])
        for field, source, hist in (("u_cat", "i_cat", x), ("u_last_cat", "i_cat", x_last), ("u_store", "i_store", x)):
            if field in self.fields:
                item_hot, names = _multi_hot(item_values[source])
                profile = (hist @ item_hot).tocsr()
                total = np.asarray(profile.sum(axis=1)).ravel()
                users.add_matrix(field, sp.diags(1.0 / np.maximum(total, 1.0)) @ profile, names)

        names = [""] * len(vocab)
        for key, idx in vocab.items():
            names[idx] = key
        return users.to_csr(data.n_users, len(vocab)), items.to_csr(data.n_items, len(vocab)), names


class _EventIndex:
    def __init__(self, entity: np.ndarray, ts: np.ndarray, device: torch.device):
        keys = np.sort((entity.astype(np.int64) << 32) | ts.astype(np.int64))
        self.keys = torch.as_tensor(keys, device=device)

    def _pos(self, entity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.keys, (entity << 32) | t.clamp_min(0))

    def count_before(self, entity: torch.Tensor, t: torch.Tensor, window: int | None = None) -> torch.Tensor:
        lo = self._pos(entity, torch.zeros_like(t) if window is None else t - window)
        return self._pos(entity, t) - lo

    def days_since_last(self, entity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        prev = self.keys[(self._pos(entity, t) - 1).clamp_min(0)]
        valid = (self._pos(entity, t) > 0) & ((prev >> 32) == entity)
        days = (t - (prev & 0xFFFFFFFF)).double() / SECONDS_PER_DAY
        return torch.where(valid, days, torch.full_like(days, float("nan")))


def _log2_bucket(count: torch.Tensor) -> torch.Tensor:
    return torch.floor(torch.log2(1.0 + count.double())).clamp_max(MAX_LOG2_BUCKET).long()


class FactorizationMachine(Recommender):
    def __init__(
        self,
        fields: tuple[str, ...] = ALL_FIELDS,
        factors: int = 64,
        lr: float = 3e-3,
        reg: float = 1e-5,
        epochs: int = 20,
        n_neg: int = 4,
        batch_size: int = 4096,
        recent_days: int = 90,
        init_std: float = 0.01,
        seed: int = 42,
        device: str | None = None,
        **builder_kwargs,
    ):
        self.fields = tuple(fields)
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.n_neg = n_neg
        self.batch_size = batch_size
        self.recent_days = recent_days
        self.init_std = init_std
        self.seed = seed
        self.device = get_device(device)
        self.builder = FMFeatureBuilder(fields=self.fields, **builder_kwargs)

    def _context(self, fields: tuple[str, ...], entity: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
        cols = []
        for field in fields:
            if field not in self.fields:
                continue
            if field == "c_user_activity":
                bucket = _log2_bucket(self.user_events.count_before(entity, t))
            elif field == "c_user_recency":
                days = self.user_events.days_since_last(entity, t)
                edges = torch.tensor(RECENCY_EDGES_DAYS, dtype=days.dtype, device=days.device)
                bucket = torch.where(days.isnan(), 0, 1 + torch.bucketize(days.nan_to_num(0.0), edges, right=True))
            elif field == "c_item_age":
                days = (t.double() - self.available_ts[entity]) / SECONDS_PER_DAY
                edges = torch.tensor(AGE_EDGES_DAYS, dtype=days.dtype, device=days.device)
                bucket = 2 + torch.bucketize(days.nan_to_num(0.0), edges, right=True)
                bucket = torch.where(days < 0, 1, bucket)
                bucket = torch.where(days.isnan(), 0, bucket)
            elif field == "c_item_pop":
                bucket = _log2_bucket(self.item_events.count_before(entity, t))
            else:
                bucket = _log2_bucket(self.item_events.count_before(entity, t, self.recent_days * SECONDS_PER_DAY))
            cols.append(self.context_offset[field] + bucket)
        return torch.stack(cols, -1) if cols else None

    def _static_items(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.sparse.mm(self.item_x, self.v)
        sq = torch.sparse.mm(self.item_x_sq, self.v.pow(2)).sum(1)
        lin = torch.sparse.mm(self.item_x, self.w.unsqueeze(1)).squeeze(1)
        return s, sq, lin

    def _item_side(self, static, items: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, sq, lin = (z[items] for z in static)
        ctx = self._context(ITEM_CONTEXT_FIELDS, items, t)
        if ctx is not None:
            v_ctx = self.v[ctx]
            s = s + v_ctx.sum(-2)
            sq = sq + v_ctx.pow(2).sum((-2, -1))
            lin = lin + self.w[ctx].sum(-1)
        return s, lin + 0.5 * (s.pow(2).sum(-1) - sq)

    def _user_side(self, users: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        starts = self.u_indptr[users]
        lengths = self.u_indptr[users + 1] - starts
        offsets = torch.cumsum(lengths, 0) - lengths
        total = int(lengths.sum())
        pos = torch.arange(total, device=users.device) + torch.repeat_interleave(starts - offsets, lengths, output_size=total)
        s = F.embedding_bag(self.u_indices[pos], self.v, offsets, mode="sum", per_sample_weights=self.u_values[pos])
        ctx = self._context(USER_CONTEXT_FIELDS, users, t)
        return s if ctx is None else s + self.v[ctx].sum(-2)

    def fit(self, data: TrainData) -> "FactorizationMachine":
        dev = self.device
        user_x, item_x, names = self.builder.build(data)
        self.context_offset = {}
        for field in USER_CONTEXT_FIELDS + ITEM_CONTEXT_FIELDS:
            if field in self.fields:
                self.context_offset[field] = len(names)
                names = names + [f"{field}={b}" for b in CONTEXT_BUCKETS[field]]
        self.feature_names = names
        user_x.sort_indices()
        self.u_indptr = torch.as_tensor(user_x.indptr, dtype=torch.long, device=dev)
        self.u_indices = torch.as_tensor(user_x.indices, dtype=torch.long, device=dev)
        self.u_values = torch.as_tensor(user_x.data, dtype=torch.float32, device=dev)
        item_x = sp.hstack([item_x, sp.csr_matrix((data.n_items, len(names) - item_x.shape[1]), dtype=np.float32)]).tocsr()
        self.item_x = sparse_to_torch(item_x, dev)
        self.item_x_sq = sparse_to_torch(item_x.multiply(item_x), dev)

        df = data.interactions
        self.user_events = _EventIndex(df.user_idx.to_numpy(), df.ts.to_numpy(), dev)
        self.item_events = _EventIndex(df.item_idx.to_numpy(), df.ts.to_numpy(), dev)
        available = pd.to_datetime(data.item_meta.reindex(range(data.n_items)).date_first_available, errors="coerce")
        available_ts = (available - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=np.float64)
        self.available_ts = torch.as_tensor(available_ts, device=dev)
        self.now = data.now

        gen = torch.Generator(device=dev).manual_seed(self.seed)
        self.v = torch.nn.Parameter(torch.randn(len(names), self.factors, generator=gen, device=dev) * self.init_std)
        self.w = torch.nn.Parameter(torch.zeros(len(names), device=dev))
        opt = torch.optim.Adam([self.v, self.w], lr=self.lr)

        pos_u = torch.as_tensor(df.user_idx.to_numpy(), dtype=torch.long, device=dev)
        pos_i = torch.as_tensor(df.item_idx.to_numpy(), dtype=torch.long, device=dev)
        pos_t = torch.as_tensor(df.ts.to_numpy(), dtype=torch.long, device=dev)
        candidates = torch.as_tensor(np.unique(df.item_idx.to_numpy()), dtype=torch.long, device=dev)
        n = len(pos_u)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen, device=dev)
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                u, i, t = pos_u[b], pos_i[b], pos_t[b]
                j = candidates[torch.randint(len(candidates), (len(b), self.n_neg), generator=gen, device=dev)]
                static = self._static_items()
                su = self._user_side(u, t)
                si, term_i = self._item_side(static, i, t)
                sj, term_j = self._item_side(static, j, t.unsqueeze(1).expand_as(j))
                score_pos = (su * si).sum(-1) + term_i
                score_neg = (su.unsqueeze(1) * sj).sum(-1) + term_j
                bpr = -F.logsigmoid(score_pos.unsqueeze(1) - score_neg).mean()
                l2 = (su.pow(2).sum() + si.pow(2).sum() + sj.pow(2).sum() / self.n_neg) / len(b)
                opt.zero_grad(set_to_none=True)
                (bpr + self.reg * l2).backward()
                opt.step()
        with torch.no_grad():
            items = torch.arange(data.n_items, device=dev)
            self.item_now, self.item_term_now = self._item_side(self._static_items(), items, torch.full_like(items, self.now))
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray) -> torch.Tensor:
        u = torch.as_tensor(users, dtype=torch.long, device=self.device)
        return self._user_side(u, torch.full_like(u, self.now)) @ self.item_now.T + self.item_term_now.unsqueeze(0)
