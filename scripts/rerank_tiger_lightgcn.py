"""Design A: keep a trained TIGER frozen and re-rank its candidates with LightGCN.

Final score = z(T5 log-probability) + alpha * z(LightGCN dot product), with z-scores taken
within each user's masked candidates. Alpha is chosen on validation NDCG@10; alpha 0 is
plain TIGER. The test split runs once, with the chosen alpha.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/xiaolongtao_tiger/model"))
from dataloader import GenRecDataLoader
from dataset import GenRecDataset
from main import TIGER
from scripts.run_tiger_t5 import load_frames, model_config, seen_items
from src.tiger_eval import CodeTable, generate_scored, score

KS = (5, 10)
ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def zscore(values):
    return (values - values.mean()) / (values.std() + 1e-8)


def rerank(scored, users, seen, user_emb, item_emb, alpha):
    rankings = []
    for (items, t5_scores), user in zip(scored, users):
        keep = [(i, s) for i, s in zip(items, t5_scores) if i not in seen.get(user, ())]
        if not keep:
            rankings.append([])
            continue
        kept_items = np.array([i for i, _ in keep])
        combined = zscore(np.array([s for _, s in keep])) + alpha * zscore(item_emb[kept_items] @ user_emb[user])
        # Stable sort, so alpha 0 keeps T5's order on ties.
        rankings.append(kept_items[np.argsort(-combined, kind="stable")].tolist())
    return rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Finished TIGER run with best.pt and codes.npy")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to best.pt in --run-dir")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "results/lightgcn")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tiger")
    parser.add_argument("--output-dir", type=Path, help="Defaults to rerank_lightgcn in --run-dir")
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--infer-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--limit-users", type=int, default=0, help="Check the code on validation users, then exit")
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "rerank_lightgcn"
    output.mkdir(parents=True, exist_ok=True)
    test_path = output / "metrics_test.json"
    if test_path.exists():
        raise SystemExit(f"{test_path} exists: the test split was already evaluated for this run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codes_path = args.run_dir / "codes.npy"
    table = CodeTable(np.load(codes_path))
    model = TIGER(model_config()).to(device)
    state = torch.load(args.checkpoint or args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    user_emb = np.load(args.graph_dir / "user_emb.npy")
    item_emb = np.load(args.graph_dir / "item_emb.npy")
    valid_frame, test_frame = load_frames(args.data_dir)

    def scored_for(name, limit=0):
        data = GenRecDataset(str(args.data_dir / f"{name}.parquet"), str(codes_path), "evaluation", args.max_len)
        if limit:
            data = Subset(data, range(limit))
        loader = GenRecDataLoader(data, batch_size=args.infer_size, shuffle=False, num_workers=4)
        return generate_scored(model, loader, table, args.beam_size, device)

    limit = args.limit_users
    scored = scored_for("valid", limit)
    frame = valid_frame.iloc[:limit] if limit else valid_frame
    users, targets = (frame.user.to_numpy() - 1).tolist(), frame.target_idx.tolist()
    seen = seen_items(valid_frame)
    history = []
    for alpha in ALPHAS:
        metrics = score(rerank(scored, users, seen, user_emb, item_emb, alpha), targets, KS)
        history.append({"alpha": alpha, **metrics})
        print(json.dumps(history[-1]), flush=True)
    best = max(history, key=lambda row: row["ndcg@10"])
    (output / "metrics_val.json").write_text(json.dumps({"selected_alpha": best["alpha"], "history": history}, indent=2) + "\n")
    if limit:
        print("code check on validation users only; test not run")
        return

    scored = scored_for("test")
    users = (test_frame.user.to_numpy() - 1).tolist()
    targets = test_frame.target_idx.to_numpy()
    seen = seen_items(test_frame)
    degree = np.load(args.graph_dir / "item_train_degree.npy")[targets]
    results = {"selected_alpha": best["alpha"]}
    for name, alpha in (("reranked", best["alpha"]), ("tiger_alpha_0", 0.0)):
        ranked = rerank(scored, users, seen, user_emb, item_emb, alpha)
        results[name] = score(ranked, targets.tolist(), KS)
        results[f"{name}_by_target_frequency"] = {
            label: score([r for r, k in zip(ranked, keep) if k], targets[keep].tolist(), KS)
            for label, keep in (("degree_0", degree == 0), ("degree_1_4", (degree >= 1) & (degree <= 4)),
                                ("degree_5_plus", degree >= 5))}
    test_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
