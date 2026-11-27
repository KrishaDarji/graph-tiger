import argparse
import json
from pathlib import Path

import torch

from src.data import build_split, load_items, load_reviews
from src.evaluator import Evaluator
from src.scoring import ease_matrix

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--out", required=True)
p.add_argument("--temporal", default="")
p.add_argument("--lams", default="100,300,1000,3000")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(a.data)
items, idx = load_items(d / "items.txt")
n = len(items)
temporal = tuple(float(v) for v in a.temporal.split(",")) if a.temporal else None
split = build_split(load_reviews(d / "reviews.json.gz"), idx, 10**6, temporal=temporal)


def run(b, hists, targets):
    ev = Evaluator(n, split.count)
    for i in range(0, len(hists), 512):
        rows = hists[i:i + 512]
        seen = torch.zeros(len(rows), n, device=dev)
        hist = torch.full((len(rows), max(map(len, rows))), -1, dtype=torch.long)
        for j, r in enumerate(rows):
            seen[j, r] = 1.0
            hist[j, :len(r)] = torch.tensor(r)
        ev.update(seen @ b, torch.tensor(targets[i:i + 512], device=dev), hist.to(dev))
    return ev.summary()


best, val_by_lam = None, {}
for lam in [float(v) for v in a.lams.split(",")]:
    b = ease_matrix(split.train, n, lam, dev)
    v = run(b, split.val_hist, split.val_target)
    val_by_lam[f"lam{lam}"] = v["raw"]["ndcg@10"]
    print("lam", lam, "val N@10", round(v["raw"]["ndcg@10"], 4), flush=True)
    if best is None or v["raw"]["ndcg@10"] > best[0]:
        best = (v["raw"]["ndcg@10"], lam, b, v)
_, lam, b, v = best
t = run(b, split.test_hist, split.test_target)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps({"lam": lam, "val": v["raw"], "test": t["raw"], "test_filtered": t["filtered"], "test_beyond": t["beyond"],
                                             "val_by_lam": val_by_lam, "n_test_users": len(split.test_target)}, indent=1))
print("test", {k: round(t["raw"][k], 4) for k in ("recall@5", "ndcg@5", "recall@10", "ndcg@10")})
