import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import build_split, load_items, load_reviews, temporal_windows
from src.graph import procrustes, transition_svd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--out", default="", help="default: <data>/graph.npy")
p.add_argument("--dim", type=int, default=128)
p.add_argument("--window", type=int, default=3)
p.add_argument("--holdout", default="", help="npy of item ids to leave out of the graph")
p.add_argument("--temporal", default="", help="q1,q2: only interactions before the q1 time quantile")
p.add_argument("--train-end", choices=["val", "test"], default="val")
p.add_argument("--prefix-frac", type=float, default=1.0, help="use only the older part of each training prefix (a stale graph)")
p.add_argument("--align-to", default="", help="rotate the result onto the basis of this graph (refresh)")
a = p.parse_args()
d = Path(a.data)
items, idx = load_items(d / "items.txt")
df = load_reviews(d / "reviews.json.gz")
holdout = set(np.load(a.holdout).tolist()) if a.holdout else set()
if a.temporal:
    q1, q2 = (float(v) for v in a.temporal.split(","))
    seqs = temporal_windows(df, idx, q1, q2, a.train_end)["train"]
else:  # training prefix only: no validation or test item enters the graph
    seqs = build_split(df, idx, 10**6).train
seqs = [[i for i in s if i not in holdout] for s in seqs]
seqs = [s[:int(np.ceil(a.prefix_frac * len(s)))] for s in seqs]
g = transition_svd(seqs, len(items), a.dim, a.window)
if a.align_to:
    g = procrustes(torch.tensor(np.load(a.align_to)), torch.tensor(g)).numpy()
out = Path(a.out or d / "graph.npy")
np.save(out, g)
print("saved", g.shape, "to", out)
