"""Aggregate gen_* runs: mean +- std over seeds for every metric, plus paired per-user bootstrap against a reference config.

usage: python scripts/paper/final_report.py results/paper CONFIG[:label] ... [--ref CONFIG]
CONFIG is the run-name prefix without the seed suffix, e.g. gen_final -> gen_final_s2024, gen_final_s2025, ...
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evalkit import paired_test

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("configs", nargs="+")
p.add_argument("--ref", default="")
args = p.parse_args()
root = Path(args.root)

MAIN = ["recall@5", "ndcg@5", "recall@10", "ndcg@10"]
MORE = ["recall@20", "ndcg@20", "recall@50", "mrr@10"]
BEYOND = ["coverage@10", "gini@10", "recall@10_tail_targets", "recall@10_head_targets", "avg_log_popularity@10"]


def runs(cfg):
    out = []
    for d in sorted(root.iterdir()):
        if re.fullmatch(re.escape(cfg) + r"_s\d+", d.name) and (d / "result.json").exists():
            out.append((d, json.load(open(d / "result.json"))))
    return out


def ms(vals):
    a = np.array(vals)
    return f"{a.mean():.4f}+-{a.std():.4f}"


print("== test, raw ranking (no item removed; the TIGER / RPG / ACERec protocol) ==")
print(f"{'config':34s} n  " + "  ".join(f"{k:>13s}" for k in MAIN + MORE))
for cfg in args.configs:
    rs = runs(cfg)
    if rs:
        print(f"{cfg:34s} {len(rs)}  " + "  ".join(f"{ms([r['test'][k] for _, r in rs]):>13s}" for k in MAIN + MORE))
print("\n== test, history items removed from the ranking (second view; NOT comparable with the papers) ==")
for cfg in args.configs:
    rs = runs(cfg)
    if rs:
        print(f"{cfg:34s} {len(rs)}  " + "  ".join(f"{ms([r['test_filtered'][k] for _, r in rs]):>13s}" for k in MAIN))
print("\n== beyond accuracy and cost ==")
print(f"{'config':34s} " + "  ".join(f"{k[:16]:>16s}" for k in BEYOND) + "   params   train_min  ms/user")
for cfg in args.configs:
    rs = runs(cfg)
    if rs:
        print(f"{cfg:34s} " + "  ".join(f"{np.mean([r['test_beyond'][k] for _, r in rs]):16.4f}" for k in BEYOND)
              + f"   {rs[0][1]['params'] / 1e6:5.2f}M   {np.mean([r['train_seconds'] for _, r in rs]) / 60:6.1f}   {np.mean([r['score_ms_per_user'] for _, r in rs]):.3f}")
if args.ref:
    ref = {d.name.rsplit("_s", 1)[1]: np.load(d / "test_ranks_raw.npy") for d, _ in runs(args.ref) if (d / "test_ranks_raw.npy").exists()}
    print(f"\n== paired per-user bootstrap vs {args.ref} (same seed, raw ranking), mean over matching seeds ==")
    for cfg in args.configs:
        if cfg == args.ref:
            continue
        for k in (5, 10):
            for metric in ("recall", "ndcg"):
                diffs = []
                for d, _ in runs(cfg):
                    s = d.name.rsplit("_s", 1)[1]
                    if s in ref and (d / "test_ranks_raw.npy").exists():
                        diffs.append(paired_test(np.load(d / "test_ranks_raw.npy"), ref[s], k=k, metric=metric))
                if diffs:
                    rel = np.mean([x["rel_diff"] for x in diffs]) * 100
                    better = sum(x["mean_diff"] > 0 for x in diffs)
                    sig = sum(x["ci95"][0] > 0 for x in diffs)
                    print(f"{cfg:34s} {metric}@{k:<2d} rel {rel:+6.1f}%  better in {better}/{len(diffs)} seeds, CI excludes 0 in {sig}/{len(diffs)}")
