"""Give items with no interactions a collaborative vector predicted from their text vector.

Fit only on items that have a graph vector. Candidates (ridge, kNN) are compared on a held-out slice of those same warm items;
the cold items' true vectors are never used.
"""
import argparse
from pathlib import Path

import faiss
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--emb", required=True)
p.add_argument("--graph", required=True)
p.add_argument("--cold", required=True, help="npy of item ids without interactions")
p.add_argument("--out", required=True)
p.add_argument("--pca", type=int, default=128)
args = p.parse_args()

x = np.load(args.emb).astype(np.float32)
pca = faiss.PCAMatrix(x.shape[1], args.pca)
pca.train(x)
x = pca.apply(x)
x /= np.linalg.norm(x, axis=1, keepdims=True)
g = np.load(args.graph).astype(np.float32)
cold = np.load(args.cold)
warm = np.setdiff1d(np.arange(len(x)), cold)
gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
rng = np.random.default_rng(0)
dev = rng.choice(warm, len(warm) // 10, replace=False)
fit = np.setdiff1d(warm, dev)


def ridge(train, lam):
    X = np.hstack([x[train], np.ones((len(train), 1), np.float32)])
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ gn[train])
    return lambda idx: np.hstack([x[idx], np.ones((len(idx), 1), np.float32)]) @ w


def knn(train, k):
    def f(idx):
        sim = x[idx] @ x[train].T
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        w = np.exp(np.take_along_axis(sim, top, 1) * 10)
        return (w[..., None] * gn[train][top]).sum(1) / w.sum(1, keepdims=True)
    return f


def score(f):
    pred = f(dev)
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
    return float((pred * gn[dev]).sum(1).mean())


cands = {f"ridge{lam}": (ridge, lam) for lam in (0.1, 1.0, 10.0)} | {f"knn{k}": (knn, k) for k in (5, 20, 50)}
scores = {n: score(fn(fit, a)) for n, (fn, a) in cands.items()}
print("held-out warm cosine to the true vector:", {k: round(v, 3) for k, v in scores.items()})
best = max(scores, key=scores.get)
fn, a = cands[best]
out = g.copy()
pred = fn(warm, a)(cold)
out[cold] = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9) * np.linalg.norm(gn[warm], axis=1).mean()
Path(args.out).mkdir(parents=True, exist_ok=True)
np.save(Path(args.out) / "item_emb.npy", out.astype(np.float32))
print("chosen", best, "imputed", len(cold), "items")
