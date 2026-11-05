"""Train LightGCN on the train split, select on validation, test once, export."""
import argparse
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lightgcn import LightGCN, NegativeSampler, bpr_loss, build_normalized_graph
from src.lightgcn_data import degree_buckets, load_split_data, seen_matrix
from src.lightgcn_eval import evaluate_split, popularity_embeddings, score_rankings, top_k_items

KS = (5, 10)
SELECTION_METRIC = "ndcg@10"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lightgcn"))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10,
                        help="Stop after this many evaluations without improvement; 0 disables")
    parser.add_argument("--eval-chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample-users", type=int, default=0,
                        help="Pilot on this many users; indices stay unchanged")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_state():
    def run(*command):
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log")],
    )


def train(args, data, model, graph, device, log):
    edges = torch.as_tensor(data.train, device=device)
    sampler = NegativeSampler(edges[:, 0], edges[:, 1], data.n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    val_mask = seen_matrix(data.n_users, data.n_items, data.train)
    state = {"epoch": 0, "best": -1.0, "best_epoch": 0, "bad_evals": 0, "history": []}

    last_path = args.output_dir / "last.pt"
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        log.info("resumed after epoch %d", state["epoch"])

    while state["epoch"] < args.epochs:
        state["epoch"] += 1
        epoch = state["epoch"]
        start = time.time()
        model.train()
        order = torch.randperm(edges.size(0), device=device)
        total_loss = total_bpr = 0.0
        batches = order.split(args.batch_size)
        for batch in batches:
            users, positives = edges[batch, 0], edges[batch, 1]
            negatives = sampler.sample(users)
            pos_score, neg_score = model(graph, users, positives, negatives)
            ranking_loss = bpr_loss(pos_score, neg_score)
            loss = ranking_loss + args.reg * model.l2_penalty(users, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bpr += ranking_loss.item()
        log.info("epoch %d loss=%.5f bpr=%.5f time=%.1fs", epoch,
                 total_loss / len(batches), total_bpr / len(batches), time.time() - start)

        if epoch % args.eval_every and epoch != args.epochs:
            continue
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.split_embeddings(graph)
        metrics = evaluate_split(user_emb, item_emb, data.validation, val_mask, KS, args.eval_chunk_size)
        state["history"].append({"epoch": epoch, **metrics})
        log.info("val epoch %d %s", epoch, json.dumps(metrics))
        if metrics[SELECTION_METRIC] > state["best"]:
            state.update(best=metrics[SELECTION_METRIC], best_epoch=epoch, bad_evals=0)
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics_val": metrics,
                        "n_users": data.n_users, "n_items": data.n_items,
                        "config": vars_for_json(args)}, args.output_dir / "best.pt")
        else:
            state["bad_evals"] += 1
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, last_path)
        if args.patience and state["bad_evals"] >= args.patience:
            log.info("early stop at epoch %d", epoch)
            break
    return state


def vars_for_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def evaluate_test_once(args, data, user_emb, item_emb, device):
    """Evaluate the selected checkpoint under every masking protocol in one pass."""
    train_mask = seen_matrix(data.n_users, data.n_items, data.train)
    history_mask = seen_matrix(data.n_users, data.n_items, data.train, data.validation)
    users, targets = data.test[:, 0], data.test[:, 1]
    chunk = args.eval_chunk_size
    k = max(KS)
    primary = top_k_items(user_emb, item_emb, users, history_mask, k, chunk)
    warm = ~data.cold_item_mask[targets]
    pop_user, pop_item = popularity_embeddings(data.item_train_degree, data.n_users, device)
    return {
        # Primary protocol: mask train + validation items.
        "lightgcn": score_rankings(primary, targets, KS),
        "lightgcn_warm_targets_only": score_rankings(primary[warm], targets[warm], KS),
        # Same masking as scripts/run_popularity_baseline.py (train only).
        "lightgcn_mask_train_only": evaluate_split(user_emb, item_emb, data.test, train_mask, KS, chunk),
        # Same masking as scripts/evaluate_tiger_checkpoint.py (none).
        "lightgcn_no_mask": evaluate_split(user_emb, item_emb, data.test, None, KS, chunk),
        "popularity_same_primary_mask": evaluate_split(pop_user, pop_item, data.test, history_mask, KS, chunk),
        "n_test_users": len(targets),
        "fraction_targets_cold": float((~warm).mean()),
        "fraction_targets_inside_primary_mask": float(
            np.asarray(history_mask[users, targets]).ravel().mean()
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.output_dir / "metrics_test.json"
    if test_path.exists():
        raise SystemExit(f"{test_path} exists: the test split was already evaluated for this run.")
    if (args.output_dir / "last.pt").exists() and not args.resume:
        raise SystemExit(f"{args.output_dir} has a run in progress; pass --resume or use a new --output-dir.")
    setup_logging(args.output_dir)
    log = logging.getLogger("lightgcn")
    seed_everything(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    data = load_split_data(args.processed_dir, args.subsample_users, args.seed)
    buckets = degree_buckets(data.item_train_degree)
    config = {**vars_for_json(args), "device_used": str(device), "git": git_state(),
              "n_users": data.n_users, "n_items": data.n_items,
              "n_train_edges": len(data.train), "item_id_sha256": data.item_id_sha256,
              "selection_metric": SELECTION_METRIC, "cold_items": buckets}
    write_json(args.output_dir / "config.json", config)
    log.info("config %s", json.dumps(config))

    edges = torch.as_tensor(data.train, device=device)
    graph = build_normalized_graph(edges[:, 0], edges[:, 1], data.n_users, data.n_items, device)
    model = LightGCN(data.n_users, data.n_items, args.dim, args.layers).to(device)
    state = train(args, data, model, graph, device, log)

    write_json(args.output_dir / "metrics_val.json", {
        "selection_metric": SELECTION_METRIC, "best_epoch": state["best_epoch"],
        "history": state["history"],
        "best": next(h for h in state["history"] if h["epoch"] == state["best_epoch"]),
    })
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.split_embeddings(graph)
    np.save(args.output_dir / "item_emb.npy", item_emb.cpu().numpy())
    np.save(args.output_dir / "user_emb.npy", user_emb.cpu().numpy())
    np.save(args.output_dir / "cold_item_mask.npy", data.cold_item_mask)
    np.save(args.output_dir / "item_train_degree.npy", data.item_train_degree)

    metrics = {"best_epoch": state["best_epoch"], **evaluate_test_once(args, data, user_emb, item_emb, device)}
    write_json(test_path, metrics)
    log.info("test %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
