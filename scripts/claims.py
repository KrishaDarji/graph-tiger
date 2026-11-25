# is our seed-averaged metric above or below each published number? per-user bootstrap, published numbers treated as exact
# usage: python -m scripts.claims results/beauty beauty ours [test_ranks_raw.npy]
import glob
import sys
from pathlib import Path

import numpy as np

# R@5, N@5, R@10, N@10 as reported (raw full ranking, leave-last-out, Sentence-T5 text)
PUBLISHED = {
    "beauty": {"RPG": (.0550, .0381, .0809, .0464), "ACERec": (.0610, .0433, .0856, .0513),
               "TopoGR": (.0620, .0439, .0883, .0522), "DiffGRM": (.0603, .0414, .0876, .0502)},
    "sports": {"RPG": (.0314, .0216, .0463, .0263), "ACERec": (.0341, .0238, .0494, .0287),
               "TopoGR": (.0374, .0258, .0554, .0314), "DiffGRM": (.0363, .0245, .0550, .0305)},
    "toys": {"RPG": (.0592, .0401, .0869, .0490), "ACERec": (.0698, .0514, .0940, .0591), "TopoGR": (.0712, .0497, .0995, .0588)},
}
root, dataset, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
fname = sys.argv[4] if len(sys.argv) > 4 else "test_ranks_raw.npy"
runs = sorted(glob.glob(str(root / f"{prefix}_s*" / fname)))
ranks = np.stack([np.load(r) for r in runs])  # seeds x users
print(f"{dataset}: {len(runs)} seeds, {ranks.shape[1]} users")


def per_user(k, kind):
    hit = ranks < k
    return hit.mean(0) if kind == "r" else np.where(hit, 1.0 / np.log2(ranks + 2.0), 0.0).mean(0)


metrics = [("R@5", 5, "r"), ("N@5", 5, "n"), ("R@10", 10, "r"), ("N@10", 10, "n")]
values = [per_user(k, kind) for _, k, kind in metrics]
resample = np.random.default_rng(0).integers(0, ranks.shape[1], (2000, ranks.shape[1]))
boots = [np.array([v[i].mean() for i in resample]) for v in values]
print("ours   " + "  ".join(f"{m} {v.mean():.4f} [{np.quantile(b, .025):.4f},{np.quantile(b, .975):.4f}]" for (m, _, _), v, b in zip(metrics, values, boots)))
for name, pub in PUBLISHED[dataset].items():
    cells = []
    for (m, _, _), v, b, p in zip(metrics, values, boots, pub):
        verdict = "above" if np.quantile(b, .025) > p else ("below" if np.quantile(b, .975) < p else "n.s.")
        cells.append(f"{m} {(v.mean() - p) / p * 100:+.1f}% {verdict}")
    print(f"vs {name:8s} " + " | ".join(cells))
