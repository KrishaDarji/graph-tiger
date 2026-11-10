import argparse
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tiger.data import add_collision_codes

p = argparse.ArgumentParser()
p.add_argument("--emb", required=True)
p.add_argument("--out", required=True)
p.add_argument("--levels", type=int, default=3)
p.add_argument("--codes", type=int, default=256)
p.add_argument("--standardize", action="store_true")
args = p.parse_args()

x = np.load(args.emb).astype(np.float32)
if args.standardize:
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
residual, codes = x.copy(), []
for level in range(args.levels):
    km = faiss.Kmeans(x.shape[1], args.codes, niter=50, seed=level, verbose=False)
    km.train(residual)
    assign = km.index.search(residual, 1)[1][:, 0]
    residual = residual - km.centroids[assign]
    codes.append(assign)
    print(level, "usage", len(np.unique(assign)) / args.codes, "residual", float((residual ** 2).sum(1).mean()))
codes = add_collision_codes(np.stack(codes, 1))
print("unique ids", len(np.unique(codes[:, :-1], axis=0)), "/", len(codes), "max collision idx", codes[:, -1].max())
np.save(args.out, codes)
