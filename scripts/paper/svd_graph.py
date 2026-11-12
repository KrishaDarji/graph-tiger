import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--out", required=True)
p.add_argument("--dim", type=int, default=64)
p.add_argument("--window", type=int, default=3)
p.add_argument("--holdout", default="")
p.add_argument("--method", choices=["svd", "rp"], default="svd", help="rp: fixed random projection of the normalised co-occurrence rows, same basis whenever the data grows")
p.add_argument("--weight", choices=["norm", "ppmi"], default="norm", help="ppmi: positive pointwise mutual information (context-smoothed) instead of degree normalisation")
p.add_argument("--directed", action="store_true", help="count a before b only; save source factors (item_emb.npy) and target factors (item_emb_tgt.npy)")
p.add_argument("--prefix-frac", type=float, default=1.0, help="use only the older part of each training prefix (a stale graph)")
args = p.parse_args()

items = Path(args.items).read_text().split()
idx = {a: i for i, a in enumerate(items)}
n = len(items)
holdout = set(np.load(args.holdout).tolist()) if args.holdout else set()
rows, cols, vals = [], [], []
for s in build_sequences(load_interactions(args.reviews)).values():
    s = [idx[a] for a in s][:-2]  # training prefix only: no validation or test item enters the graph
    s = [i for i in s if i not in holdout]
    s = s[:int(np.ceil(args.prefix_frac * len(s)))]
    for t, a in enumerate(s):
        for dist in range(1, args.window + 1):
            if t + dist < len(s):
                b = s[t + dist]
                if args.directed:
                    rows.append(a)
                    cols.append(b)
                    vals.append(1.0 / dist)
                else:
                    rows += [a, b]
                    cols += [b, a]
                    vals += [1.0 / dist] * 2
m = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
d_out = np.asarray(m.sum(1)).ravel() + 1e-9
d_in = np.asarray(m.sum(0)).ravel() + 1e-9
if args.weight == "ppmi":
    m = m.tocoo()
    ctx = d_in ** 0.75
    pmi = np.log(m.data * ctx.sum() / (d_out[m.row] * ctx[m.col]) + 1e-12)
    keep = pmi > 0
    m = sp.coo_matrix((pmi[keep], (m.row[keep], m.col[keep])), shape=m.shape).tocsr()
else:
    m = sp.diags(d_out ** -0.5) @ m @ sp.diags(d_in ** -0.5)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.method == "rp":
    R = np.random.default_rng(12345).standard_normal((n, args.dim)).astype(np.float32) / np.sqrt(args.dim)
    np.save(out / "item_emb.npy", np.asarray(m @ R, dtype=np.float32))
    print("saved random-projection memory", args.dim, "nonzero pairs", m.nnz)
    sys.exit(0)
u, sv, vt = svds(m, k=args.dim)
np.save(out / "item_emb.npy", (u * np.sqrt(sv)).astype(np.float32))
if args.directed:
    np.save(out / "item_emb_tgt.npy", (vt.T * np.sqrt(sv)).astype(np.float32))
print("saved", u.shape, "nonzero pairs", m.nnz)
