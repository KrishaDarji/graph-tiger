"""Does validation rank configs the way test does, and how big is the val-to-test gap? Usage: overfit_report.py results/paper GLOB [GLOB ...]"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

root = Path(sys.argv[1])
rows = []
for pat in sys.argv[2:]:
    for d in sorted(glob.glob(str(root / pat))):
        f = Path(d) / "result.json"
        if f.exists():
            r = json.load(open(f))
            log = json.load(open(Path(d) / "log.json")) if (Path(d) / "log.json").exists() else []
            best_ep = int(np.argmax([x.get("val_ndcg@10", 0) for x in log])) + 1 if log else -1
            rows.append((Path(d).name, r["best_val"], r["test"]["ndcg@10"], r["test"]["recall@10"], best_ep, r["epochs_run"],
                         log[-1]["loss"] if log else float("nan")))
if not rows:
    sys.exit("no runs")
print(f"{'run':36s} val_N10  test_N10  test-val  test_R10  best_ep/epochs  last_train_loss")
for n, v, t, r10, be, ep, loss in rows:
    print(f"{n:36s} {v:.4f}   {t:.4f}   {t - v:+.4f}   {r10:.4f}    {be:3d}/{ep:3d}        {loss:.3f}")
v = np.array([x[1] for x in rows])
t = np.array([x[2] for x in rows])
rho, p = spearmanr(v, t)
top_v = int(np.argmax(v))
print(f"\n{len(rows)} configs. Spearman(val, test) = {rho:.2f} (p={p:.3f}). Config picked by validation: {rows[top_v][0]}, "
      f"its test NDCG@10 {t[top_v]:.4f}; best test among all {t.max():.4f}; mean test over all {t.mean():.4f}.")
print(f"Mean test-minus-val gap {np.mean(t - v):+.4f}; gap of the validation-picked config {t[top_v] - v[top_v]:+.4f}.")
