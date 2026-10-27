from __future__ import annotations

import gzip
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from models.base import TrainData

REVIEWS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz"
)
META_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/"
    "meta_Video_Games.jsonl"
)
REVIEWS_FILE = "Video_Games.jsonl.gz"
META_FILE = "meta_Video_Games.jsonl"


def download_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for url, name in ((REVIEWS_URL, REVIEWS_FILE), (META_URL, META_FILE)):
        path = raw_dir / name
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)


def load_reviews(path: Path) -> pd.DataFrame:
    cols = ("user_id", "parent_asin", "rating", "timestamp", "helpful_vote", "verified_purchase")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(tuple(d.get(c) for c in cols) + (len(d.get("text") or ""),))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp", "helpful_vote", "verified_purchase", "text_len"])
    df["timestamp"] = df.timestamp.astype("int64")
    return df


def load_meta(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            details = d.get("details") or {}
            rows.append(
                {
                    "item_id": d["parent_asin"],
                    "title": d.get("title"),
                    "main_category": d.get("main_category"),
                    "store": d.get("store"),
                    "categories": list(d.get("categories") or []),
                    "price": d.get("price"),
                    "rated": details.get("Rated"),
                    "date_first_available": details.get("Date First Available"),
                    "release_date": details.get("Release date"),
                    "n_features": len(d.get("features") or []),
                    "description_len": sum(len(x) for x in d.get("description") or []),
                    "average_rating": d.get("average_rating"),
                    "rating_number": d.get("rating_number"),
                }
            )
    meta = pd.DataFrame(rows)
    meta["price"] = pd.to_numeric(meta.price, errors="coerce")
    for col in ("date_first_available", "release_date"):
        meta[col] = pd.to_datetime(meta[col], format="%B %d, %Y", errors="coerce")
    return meta


def load_cached(raw_dir: Path, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    reviews_cache, meta_cache = cache_dir / "reviews.parquet", cache_dir / "meta.parquet"
    if not reviews_cache.exists():
        load_reviews(raw_dir / REVIEWS_FILE).to_parquet(reviews_cache)
    if not meta_cache.exists():
        load_meta(raw_dir / META_FILE).to_parquet(meta_cache)
    reviews = pd.read_parquet(reviews_cache)
    meta = pd.read_parquet(meta_cache)
    meta["categories"] = meta.categories.apply(list)
    return reviews, meta


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    while True:
        user_cnt = df.user_id.map(df.user_id.value_counts())
        item_cnt = df.item_id.map(df.item_id.value_counts())
        keep = (user_cnt >= k) & (item_cnt >= k)
        if keep.all():
            return df
        df = df[keep]


def clean_interactions(reviews: pd.DataFrame, meta: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = reviews.drop_duplicates()
    df = df.sort_values(["timestamp", "user_id", "item_id"], kind="stable")
    df = df.drop_duplicates(["user_id", "item_id"], keep="first")
    df = df[df.item_id.isin(set(meta.item_id))]
    df = k_core(df, k)
    return df.reset_index(drop=True)


@dataclass
class Encoded:
    interactions: pd.DataFrame
    item_meta: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def encode(df: pd.DataFrame, meta: pd.DataFrame) -> Encoded:
    user_ids = np.sort(df.user_id.unique())
    item_ids = np.sort(df.item_id.unique())
    out = df.copy()
    out["user_idx"] = np.searchsorted(user_ids, out.user_id)
    out["item_idx"] = np.searchsorted(item_ids, out.item_id)
    out["ts"] = out.timestamp // 1000
    out = out.sort_values(["ts", "user_idx", "item_idx"], kind="stable").reset_index(drop=True)
    item_meta = meta.set_index("item_id").loc[item_ids].reset_index()
    item_meta.index.name = "item_idx"
    return Encoded(out, item_meta, user_ids, item_ids)


@dataclass
class EvalSplit:
    name: str
    train: TrainData
    window: pd.DataFrame
    users: np.ndarray
    relevant: sp.csr_matrix
    history: sp.csr_matrix
    candidates: np.ndarray

    def summary(self) -> dict:
        warm_users = self.window.user_idx.isin(np.unique(self.train.interactions.user_idx))
        warm_items = self.candidates[self.window.item_idx.to_numpy()]
        return {
            "взаимодействий в обучении": len(self.train.interactions),
            "пользователей в обучении": self.train.interactions.user_idx.nunique(),
            "товаров-кандидатов": int(self.candidates.sum()),
            "взаимодействий в окне": len(self.window),
            "пользователей в окне": self.window.user_idx.nunique(),
            "доля событий окна с новым пользователем": round(1 - warm_users.mean(), 3),
            "доля событий окна с новым товаром": round(1 - warm_items.mean(), 3),
            "оцениваемых пользователей": len(self.users),
            "релевантных пар": int(self.relevant.nnz),
            "релевантных на пользователя": round(self.relevant.nnz / max(len(self.users), 1), 2),
        }


def make_split(
    data: Encoded,
    name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
    positive_threshold: float = 4.0,
) -> EvalSplit:
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp()) if end is not None else np.iinfo(np.int64).max
    df = data.interactions
    train_df = df[df.ts < start_ts][["user_idx", "item_idx", "rating", "ts"]].reset_index(drop=True)
    window = df[(df.ts >= start_ts) & (df.ts < end_ts)].reset_index(drop=True)
    train = TrainData(train_df, data.n_users, data.n_items, start_ts, data.item_meta)

    candidates = np.zeros(data.n_items, dtype=bool)
    candidates[train_df.item_idx.unique()] = True
    known_users = np.zeros(data.n_users, dtype=bool)
    known_users[train_df.user_idx.unique()] = True
    rel = window[
        (window.rating >= positive_threshold)
        & known_users[window.user_idx.to_numpy()]
        & candidates[window.item_idx.to_numpy()]
    ]
    relevant = sp.csr_matrix(
        (np.ones(len(rel), dtype=np.float32), (rel.user_idx, rel.item_idx)), shape=(data.n_users, data.n_items)
    )
    relevant.sum_duplicates()
    relevant.data[:] = 1.0
    history = train.csr()
    history.data[:] = 1.0
    users = np.unique(rel.user_idx.to_numpy())
    return EvalSplit(name, train, window, users, relevant, history, candidates)
