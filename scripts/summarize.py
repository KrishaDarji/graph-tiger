# mean +- std over seeds, and a paired per-user bootstrap against a reference run
# usage: python -m scripts.summarize results/beauty plain ours [--ref plain]   (run folders are <name>_s<seed>)
import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.evaluator import paired_test

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("names", nargs="+")
p.add_argument("--ref", default="")
a = p.parse_args()
root = Path(a.root)
METRICS = ["recall@5", "ndcg@5", "recall@10", "ndcg@10", "recall@20", "ndcg@20"]


def runs(name):
    return [d for d in sorted(root.iterdir()) if re.fullmatch(re.escape(name) + r"_s\d+", d.name) and (d / "result.json").exists()]


print(f"{'run':20s} n  " + "  ".join(f"{m:>14s}" for m in METRICS))
for name in a.names:
    results = [json.load(open(d / "result.json")) for d in runs(name)]
    if results:
        print(f"{name:20s} {len(results)}  " + "  ".join(
            f"{np.mean([r['test'][m] for r in results]):.4f}±{np.std([r['test'][m] for r in results]):.4f}" for m in METRICS))
if a.ref:
    ref = {d.name.rsplit("_s", 1)[1]: np.load(d / "test_ranks_raw.npy") for d in runs(a.ref)}
    for name in a.names:
        for k in (5, 10):
            for metric in ("recall", "ndcg"):
                tests = [paired_test(np.load(d / "test_ranks_raw.npy"), ref[d.name.rsplit("_s", 1)[1]], k=k, metric=metric)
                         for d in runs(name) if d.name.rsplit("_s", 1)[1] in ref and name != a.ref]
                if tests:
                    print(f"{name:20s} vs {a.ref}: {metric}@{k:<2d} {np.mean([t['rel_diff'] for t in tests]) * 100:+6.1f}%  "
                          f"better in {sum(t['mean_diff'] > 0 for t in tests)}/{len(tests)} seeds, CI excludes 0 in {sum(t['ci95'][0] > 0 for t in tests)}/{len(tests)}")
