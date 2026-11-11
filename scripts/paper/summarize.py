import json
import re
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/paper")
keys = ["recall@5", "ndcg@5", "recall@10", "ndcg@10"]
groups = {}
for d in sorted(root.iterdir()):
    if d.name.startswith(("pgen_", "ace_")) and (d / "result.json").exists():
        row = [json.load(open(d / "result.json"))["test"][k] for k in keys]
        name = re.sub(r"_s\d+$", "", d.name)
    elif d.name.startswith("t5_") and (d / "metrics_test.json").exists():
        m = json.load(open(d / "metrics_test.json"))
        m = m["best_val"] if "test_paper_faithful" in m["best_val"] else m["last"]
        row = [m["test_paper_faithful"][k] for k in keys]
        name = re.sub(r"_s?\d+$", "", d.name) if re.search(r"_s\d+$", d.name) else d.name
        name = re.sub(r"_s\d+$", "", d.name)
    else:
        continue
    groups.setdefault(name, []).append(row)
print(f"{'run':28s} n  " + "  ".join(f"{k:>14s}" for k in keys))
for name, rows in groups.items():
    a = np.array(rows)
    print(f"{name:28s} {len(a)}  " + "  ".join(f"{m:.4f}±{s:.4f}" for m, s in zip(a.mean(0), a.std(0))))
