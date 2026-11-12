"""Sanity baselines through the same evaluator: popularity, and last-item similarity from a graph embedding."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evalkit import Evaluator
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--graph", nargs="*", default=[])
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
items = Path(args.items).read_text().split()
idx = {a: i for i, a in enumerate(items)}
n = len(items)
seqs = [[idx[a] for a in s] for s in build_sequences(load_interactions(args.reviews)).values()]
count = np.bincount([i for s in seqs for i in s[:-2]], minlength=n)
pop = torch.tensor(count, dtype=torch.float32, device=dev)


def run(name, score_fn, split):
    ev = Evaluator(n, count)
    for b in range(0, len(seqs), 512):
        rows = seqs[b:b + 512]
        hist = [s[:-2] if split == "val" else s[:-1] for s in rows]
        tgt = torch.tensor([s[-2] if split == "val" else s[-1] for s in rows], device=dev)
        L = max(map(len, hist))
        h = torch.full((len(rows), L), -1, dtype=torch.long)
        for i, r in enumerate(hist):
            h[i, :len(r)] = torch.tensor(r)
        last = torch.tensor([r[-1] for r in hist], device=dev)
        ev.update(score_fn(last, len(rows)), tgt, h.to(dev))
    s = ev.summary()["raw"]
    print(f"{name:34s} {split:4s} R5 {s['recall@5']:.4f} N5 {s['ndcg@5']:.4f} R10 {s['recall@10']:.4f} N10 {s['ndcg@10']:.4f}")


for split in ("val", "test"):
    run("popularity", lambda last, B: pop[None].expand(B, -1), split)
    for g in args.graph:
        e = F.normalize(torch.tensor(np.load(g), device=dev), dim=1)
        run("last-item cosine " + Path(g).parent.name, lambda last, B, e=e: e[last] @ e.T, split)
