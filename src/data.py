import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_reviews(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [(r["reviewerID"], r["asin"], r["unixReviewTime"]) for r in map(json.loads, filter(str.strip, f))]
    return pd.DataFrame(rows, columns=["user", "item", "time"])


def load_items(path):
    items = Path(path).read_text().split()
    return items, {a: i for i, a in enumerate(items)}


def user_sequences(df):
    df = df.iloc[np.argsort(df["time"].to_numpy(), kind="stable")]
    return {u: g.tolist() for u, g in df.groupby("user", sort=False)["item"]}


def temporal_windows(df, idx, q1, q2, train_end="val"):
    # train before the q1 time quantile, validate on each user's first interaction in [q1, q2), test on the first after q2
    df = df.iloc[np.argsort(df["time"].to_numpy(), kind="stable")]
    t1, t2 = np.quantile(df["time"].to_numpy(), [q1, q2])
    users = {}
    for u, a, t in zip(df["user"].to_numpy(), df["item"].to_numpy(), df["time"].to_numpy()):
        users.setdefault(u, []).append((idx[a], t))
    out = {k: [] for k in ("train", "val_hist", "val_target", "test_hist", "test_target")}
    for seq in users.values():
        pre = [i for i, t in seq if t < t1]
        win = [i for i, t in seq if t1 <= t < t2]
        post = [i for i, t in seq if t >= t2]
        train = pre if train_end == "val" else pre + win
        if len(train) >= 2:
            out["train"].append(train)
        if pre and win:
            out["val_hist"].append(pre)
            out["val_target"].append(win[0])
        if (pre or win) and post:
            out["test_hist"].append(pre + win)
            out["test_target"].append(post[0])
    return out


@dataclass
class Split:
    train: list
    val_hist: list
    val_target: list
    test_hist: list
    test_target: list
    count: np.ndarray


def build_split(df, idx, max_len, holdout=frozenset(), temporal=None, train_end="val"):
    n = len(idx)
    if temporal:
        w = temporal_windows(df, idx, *temporal, train_end)
        count = np.bincount([i for t in w["train"] for i in t], minlength=n)
        return Split([t[-(max_len + 1):] for t in w["train"]], [h[-max_len:] for h in w["val_hist"]], w["val_target"],
                     [h[-max_len:] for h in w["test_hist"]], w["test_target"], count)
    seqs = [[idx[a] for a in s] for s in user_sequences(df).values()]
    train = [[i for i in s[:-2] if i not in holdout][-(max_len + 1):] for s in seqs]
    count = np.bincount([i for s in seqs for i in s[:-2]], minlength=n)
    return Split([t for t in train if len(t) >= 2], [s[:-2][-max_len:] for s in seqs], [s[-2] for s in seqs],
                 [s[:-1][-max_len:] for s in seqs], [s[-1] for s in seqs], count)


def pad(rows, length, fill=0):
    out = torch.full((len(rows), length), fill, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out
