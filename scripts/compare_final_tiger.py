"""Final test comparison: per-user metrics and paired bootstrap intervals.

Regenerates test rankings from the saved best checkpoints. Nothing is selected here: the
checkpoints and alpha were already fixed on validation. Rankings must reproduce the stored
test metrics, otherwise the script stops. Also gives the epoch 99 snapshot its test score.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/xiaolongtao_tiger/model"))
from dataloader import GenRecDataLoader
from dataset import GenRecDataset
from main import TIGER
from scripts.rerank_tiger_lightgcn import rerank
from scripts.run_tiger_t5 import load_frames, model_config, seen_items
from src.tiger_eval import CodeTable, generate_scored, score

KS = (5, 10)
METRICS = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
COMPARISONS = [("design_a", "run1"), ("run2", "run1"), ("run2", "design_a"), ("run2_through_epoch99", "run1")]
BUCKETS = (("degree_0", 0, 0), ("degree_1_4", 1, 4), ("degree_5_plus", 5, 10**9))


def per_user(rankings, targets):
    out = {m: np.zeros(len(targets)) for m in METRICS}
    for row, (ranked, target) in enumerate(zip(rankings, targets)):
        if target in ranked[:max(KS)]:
            rank = ranked.index(target)
            for k in KS:
                if rank < k:
                    out[f"recall@{k}"][row] = 1.0
                    out[f"ndcg@{k}"][row] = 1.0 / np.log2(rank + 2)
    return out


def bootstrap(a, b, keep, reps, rng):
    diff = {m: (a[m] - b[m])[keep] for m in METRICS}
    n = int(keep.sum())
    draws = {m: np.empty(reps) for m in METRICS}
    for rep in range(reps):
        sample = rng.integers(0, n, n)
        for m in METRICS:
            draws[m][rep] = diff[m][sample].mean()
    result = {}
    for m in METRICS:
        base = b[m][keep].mean()
        low, high = np.percentile(draws[m], [2.5, 97.5])
        p = min(1.0, 2 * min((draws[m] <= 0).mean(), (draws[m] >= 0).mean()))
        result[m] = {"difference": float(diff[m].mean()), "ci95": [float(low), float(high)],
                     "relative": float(diff[m].mean() / base) if base else None, "p_bootstrap": float(p)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/tiger_full")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tiger")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "results/lightgcn_768")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.runs_dir / "comparison"
    output.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, frame = load_frames(args.data_dir)
    users = (frame.user.to_numpy() - 1).tolist()
    targets = frame.target_idx.to_numpy()
    seen = seen_items(frame)
    user_emb = np.load(args.graph_dir / "user_emb.npy")
    item_emb = np.load(args.graph_dir / "item_emb.npy")
    degree = np.load(args.graph_dir / "item_train_degree.npy")[targets]

    def scored(run, checkpoint):
        codes = args.runs_dir / run / "codes.npy"
        model = TIGER(model_config()).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        data = GenRecDataset(str(args.data_dir / "test.parquet"), str(codes), "evaluation", 20)
        loader = GenRecDataLoader(data, batch_size=256, shuffle=False, num_workers=4)
        return generate_scored(model, loader, CodeTable(np.load(codes)), 30, device)

    def check(name, rankings, path, key):
        stored = json.loads(Path(path).read_text())[key]
        fresh = score(rankings, targets.tolist(), KS)
        for m in METRICS:
            if abs(stored[m] - fresh[m]) > 1e-9:
                raise SystemExit(f"{name} {m}: stored {stored[m]}, regenerated {fresh[m]}")
        print(f"{name}: reproduces stored test metrics", flush=True)

    alpha = json.loads((args.runs_dir / "content/rerank_lightgcn768/metrics_val.json").read_text())["selected_alpha"]
    plain = lambda s, a=0.0: rerank(s, users, seen, user_emb, item_emb, a)
    rankings = {}
    s = scored("content", args.runs_dir / "content/best.pt")
    rankings["run1"] = plain(s)
    rankings["design_a"] = plain(s, alpha)
    rankings["run2"] = plain(scored("graph768", args.runs_dir / "graph768/best.pt"))
    rankings["run2_through_epoch99"] = plain(scored("graph768", args.runs_dir / "graph768/snapshot_epoch99/best.pt"))
    check("run1", rankings["run1"], args.runs_dir / "content/metrics_test.json", "primary_mask_train_and_validation")
    check("design_a", rankings["design_a"], args.runs_dir / "content/rerank_lightgcn768/metrics_test.json", "reranked")
    check("run2", rankings["run2"], args.runs_dir / "graph768/metrics_test.json", "primary_mask_train_and_validation")

    values = {name: per_user(r, targets.tolist()) for name, r in rankings.items()}
    for name, v in values.items():
        fresh = score(rankings[name], targets.tolist(), KS)
        for m in METRICS:
            if abs(fresh[m] - v[m].mean()) > 1e-9:
                raise SystemExit(f"per-user {name} {m} {v[m].mean()} != {fresh[m]}")
    np.savez_compressed(output / "per_user_test.npz", user=np.array(users), target=targets,
                        **{f"{n}__{m}": values[n][m] for n in values for m in METRICS})

    rng = np.random.default_rng(args.seed)
    everyone = np.ones(len(targets), bool)
    result = {"note": "Paired bootstrap over test users; two-sided p from the bootstrap distribution, "
                      "no correction for multiple comparisons.",
              "reps": args.reps, "alpha_design_a": alpha, "n_users": len(targets),
              "scores": {n: {m: float(values[n][m].mean()) for m in METRICS} for n in values},
              "scores_by_target_degree": {n: {label: {m: float(values[n][m][(degree >= lo) & (degree <= hi)].mean())
                                                      for m in METRICS} for label, lo, hi in BUCKETS} for n in values},
              "comparisons": {}, "comparisons_by_target_degree": {}}
    for a, b in COMPARISONS:
        key = f"{a}_vs_{b}"
        result["comparisons"][key] = bootstrap(values[a], values[b], everyone, args.reps, rng)
        result["comparisons_by_target_degree"][key] = {
            label: bootstrap(values[a], values[b], (degree >= lo) & (degree <= hi), args.reps, rng)
            for label, lo, hi in BUCKETS}
        print(key, json.dumps(result["comparisons"][key]), flush=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
