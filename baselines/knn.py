import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import build_split, load_items, load_reviews
from src.evaluator import Evaluator
from src.graph import load_graph
from src.tokens import reduce_dim

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--graph", default="", help="graph vectors; with --holdout use the file that has imputed rows for the held-out items")
p.add_argument("--holdout", default="", help="npy of unseen item ids: adds cold-pool and warm-pool views")
p.add_argument("--out", required=True)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(a.data)
items, idx = load_items(d / "items.txt")
n = len(items)
split = build_split(load_reviews(d / "reviews.json.gz"), idx, 50)
holdout = set(np.load(a.holdout).tolist()) if a.holdout else set()
count = split.count.copy()
count[list(holdout)] = 0
content = F.normalize(torch.tensor(reduce_dim(np.load(d / "emb.npy"), 128), device=dev), dim=1)
graph = F.normalize(load_graph(a.graph or d / "graph.npy", dev)[:n], dim=1)
pop = torch.tensor(count, dtype=torch.float32, device=dev)
rng = torch.Generator(device=dev).manual_seed(0)
methods = {
    "random": lambda last3: torch.rand(len(last3), n, device=dev, generator=rng),
    "popularity": lambda last3: pop[None].expand(len(last3), -1),
    "content_knn": lambda last3: F.normalize(content[last3].sum(1), dim=-1) @ content.T,
    "graph_knn": lambda last3: graph[last3[:, -1]] @ graph.T,
    "content_plus_graph_knn": lambda last3: F.normalize(content[last3].sum(1), dim=-1) @ content.T + graph[last3[:, -1]] @ graph.T,
}
result = {}
for name, hists, targets in (("val", split.val_hist, split.val_target), ("test", split.test_hist, split.test_target)):
    for m, fn in methods.items():
        ev = Evaluator(n, count, cold_items=holdout or None, item_vecs=content)
        for b in range(0, len(hists), 512):
            rows = hists[b:b + 512]
            hist = torch.full((len(rows), max(map(len, rows))), -1, dtype=torch.long)
            for i, r in enumerate(rows):
                hist[i, :len(r)] = torch.tensor(r)
            last3 = torch.stack([torch.tensor(([r[-1]] * 3 + r)[-3:]) for r in rows]).to(dev)
            ev.update(fn(last3), torch.tensor(targets[b:b + 512], device=dev), hist.to(dev))
        s = ev.summary()
        result[f"{name}/{m}"] = {k: s[k] for k in s if k in ("raw", "beyond", "cold_pool", "warm_pool")}
        print(f"{name:4s} {m:24s} R@10 {s['raw']['recall@10']:.4f} N@10 {s['raw']['ndcg@10']:.4f}"
              + (f" | cold-pool R@10 {s['cold_pool']['recall@10']:.4f}" if holdout else ""), flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(result, indent=1))
