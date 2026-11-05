"""Load the existing split files for LightGCN and verify item-index alignment.

Index contract: LightGCN row i is item_idx i, which is also row i of
data/processed/item_content_semantic_ids.pkl. The Transformer's token for the
same item is i + 1 because its token 0 is padding.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class SplitData:
    train: np.ndarray  # unique (user_idx, item_idx) edges, shape (n, 2)
    validation: np.ndarray  # (user_idx, item_idx), one row per user
    test: np.ndarray
    n_users: int
    n_items: int
    item_id_sha256: str
    item_train_degree: np.ndarray

    @property
    def cold_item_mask(self):
        return self.item_train_degree == 0


def _pairs(frame):
    return frame[["user_idx", "item_idx"]].to_numpy(dtype=np.int64, copy=True)


def item_ids_by_index(*frames):
    """Return item_id strings ordered by item_idx, requiring a 1:1 contiguous map."""
    mapping = pd.concat([f[["item_idx", "item_id"]] for f in frames]).drop_duplicates()
    if mapping.item_idx.duplicated().any() or mapping.item_id.duplicated().any():
        raise ValueError("item_idx and item_id are not one-to-one across the splits")
    mapping = mapping.sort_values("item_idx")
    if not np.array_equal(mapping.item_idx.to_numpy(), np.arange(len(mapping))):
        raise ValueError("item_idx is not contiguous from 0")
    return mapping.item_id.to_numpy(dtype=str)


def load_split_data(processed_dir="data/processed", subsample_users=0, seed=42):
    """Read the splits, check alignment against the content file, build edges.

    Subsampling keeps n_users, n_items and all indices unchanged.
    """
    processed = Path(processed_dir)
    train = pd.read_parquet(processed / "train.parquet")
    validation = pd.read_parquet(processed / "validation.parquet")
    test = pd.read_parquet(processed / "test.parquet")

    item_ids = item_ids_by_index(train, validation, test)
    content = pd.read_pickle(processed / "item_content_semantic_ids.pkl")
    if not np.array_equal(content.item_id.to_numpy(dtype=str), item_ids):
        raise ValueError("content/semantic-ID rows are not aligned with item_idx")

    n_items = len(item_ids)
    n_users = int(max(f.user_idx.max() for f in (train, validation, test))) + 1
    digest = hashlib.sha256("\n".join(item_ids.tolist()).encode()).hexdigest()

    if subsample_users:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_users, size=subsample_users, replace=False)
        train = train[train.user_idx.isin(keep)]
        validation = validation[validation.user_idx.isin(keep)]
        test = test[test.user_idx.isin(keep)]

    edges = np.unique(_pairs(train), axis=0)
    return SplitData(
        train=edges,
        validation=_pairs(validation),
        test=_pairs(test),
        n_users=n_users,
        n_items=n_items,
        item_id_sha256=digest,
        item_train_degree=np.bincount(edges[:, 1], minlength=n_items),
    )


def seen_matrix(n_users, n_items, *pair_arrays):
    """Sparse user x item matrix marking the given (user, item) pairs."""
    pairs = np.concatenate(pair_arrays)
    values = np.ones(len(pairs), dtype=bool)
    return sparse.csr_matrix((values, (pairs[:, 0], pairs[:, 1])), shape=(n_users, n_items))


def degree_buckets(degree):
    return {
        "items_total": int(degree.size),
        "items_degree_0": int((degree == 0).sum()),
        "items_degree_1_4": int(((degree >= 1) & (degree <= 4)).sum()),
        "items_degree_5_plus": int((degree >= 5).sum()),
    }
