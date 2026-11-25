import argparse
from pathlib import Path

import numpy as np

from src.tokens import reduce_dim

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--graph", default="", help="default: <data>/graph.npy")
p.add_argument("--cold", required=True, help="npy of item ids without interactions")
p.add_argument("--out", required=True)
a = p.parse_args()
d = Path(a.data)
x = reduce_dim(np.load(d / "emb.npy"), 128)
x /= np.linalg.norm(x, axis=1, keepdims=True)
g = np.load(a.graph or d / "graph.npy").astype(np.float32)
gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
cold = np.load(a.cold)
warm = np.setdiff1d(np.arange(len(x)), cold)
rng = np.random.default_rng(0)
dev = rng.choice(warm, len(warm) // 10, replace=False)
fit = np.setdiff1d(warm, dev)


def knn(train, k):
    def predict(rows):
        sim = x[rows] @ x[train].T
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        w = np.exp(np.take_along_axis(sim, top, 1) * 10)
        return (w[..., None] * gn[train][top]).sum(1) / w.sum(1, keepdims=True)
    return predict


def cosine(pred, true):
    return float((pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9) * true).sum(1).mean())


# pick k on held-out warm items; the cold items' true vectors are never used
scores = {k: cosine(knn(fit, k)(dev), gn[dev]) for k in (5, 20, 50)}
k = max(scores, key=scores.get)
print("held-out cosine to the true vector by k:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", k)
pred = knn(warm, k)(cold)
out = g.copy()
out[cold] = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9) * np.linalg.norm(gn[warm], axis=1).mean()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.save(a.out, out)
print("imputed", len(cold), "items")
