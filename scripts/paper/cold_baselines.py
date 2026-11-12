"""Cold-start reference points through the same evaluator: random, popularity, content nearest neighbour, graph nearest neighbour."""
import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evalkit import Evaluator
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--emb", required=True)
p.add_argument("--holdout", required=True)
p.add_argument("--graph", required=True, help="graph vectors with imputed rows for the held-out items")
p.add_argument("--out", required=True)
args = p.parse_args()
dev = "cuda"
items = Path(args.items).read_text().split()
idx = {a: i for i, a in enumerate(items)}
n = len(items)
seqs = [[idx[a] for a in s] for s in build_sequences(load_interactions(args.reviews)).values()]
holdout = set(np.load(args.holdout).tolist())
count = np.zeros(n, dtype=np.int64)
for s in seqs:
    for i in s[:-2]:
        if i not in holdout:
            count[i] += 1
x = np.load(args.emb).astype(np.float32)
pca = faiss.PCAMatrix(x.shape[1], 128)
pca.train(x)
content = F.normalize(torch.tensor(pca.apply(x), device=dev), dim=1)
graph = F.normalize(torch.tensor(np.load(args.graph), device=dev).float(), dim=1)
pop = torch.tensor(count, dtype=torch.float32, device=dev)
gen = torch.Generator(device=dev).manual_seed(0)

methods = {
    "random": lambda last3, B: torch.rand(B, n, device=dev, generator=gen),
    "popularity": lambda last3, B: pop[None].expand(B, -1),
    "content_knn": lambda last3, B: F.normalize(content[last3].sum(1), dim=-1) @ content.T,
    "graph_knn_imputed": lambda last3, B: graph[last3[:, -1]] @ graph.T,
    "content_plus_graph_knn": lambda last3, B: F.normalize(content[last3].sum(1), dim=-1) @ content.T + graph[last3[:, -1]] @ graph.T,
}
result = {}
for split in ("val", "test"):
    for name, fn in methods.items():
        ev = Evaluator(n, count, cold_items=holdout, item_vecs=content)
        for b in range(0, len(seqs), 512):
            rows = seqs[b:b + 512]
            hist = [s[:-2] if split == "val" else s[:-1] for s in rows]
            tgt = torch.tensor([s[-2] if split == "val" else s[-1] for s in rows], device=dev)
            L = max(map(len, hist))
            h = torch.full((len(rows), L), -1, dtype=torch.long)
            for i, r in enumerate(hist):
                h[i, :len(r)] = torch.tensor(r)
            last3 = torch.stack([torch.tensor(([r[-1]] * 3 + r)[-3:]) for r in hist]).to(dev)
            ev.update(fn(last3, len(rows)), tgt, h.to(dev))
        s = ev.summary()
        result[f"{split}/{name}"] = {"raw": s["raw"], "cold_pool": s["cold_pool"], "warm_pool": s["warm_pool"], "beyond": s["beyond"]}
        print(f"{split:4s} {name:24s} all R10 {s['raw']['recall@10']:.4f} N10 {s['raw']['ndcg@10']:.4f} | cold-pool R10 {s['cold_pool']['recall@10']:.4f} "
              f"(chance {s['cold_pool']['chance_recall@10']:.4f}) N10 {s['cold_pool']['ndcg@10']:.4f} | warm-pool R10 {s['warm_pool']['recall@10']:.4f}", flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1))
