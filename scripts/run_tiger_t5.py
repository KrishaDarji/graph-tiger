"""Train the T5 recommender on the project's split, select on validation, test once.

Validation during training uses a fixed user sample. Selection metric is NDCG@10.
The test split runs once with the selected checkpoint, and refuses to run again.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

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
from main import TIGER, train as train_epoch
from src.tiger_eval import CodeTable, apply_mask, generate_rankings, real_item_counts, score

KS = (5, 10)
SELECTION = "ndcg@10"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tiger")
    parser.add_argument("--degree", type=Path, default=ROOT / "results/lightgcn/item_train_degree.npy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-sample-users", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--infer-size", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--no-constraints", action="store_true", help="Allow beams on tuples that are not items")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help="Exit after this epoch, for resume checks")
    parser.add_argument("--check-scoring", type=int, default=0,
                        help="Score this many validation users with an untrained model, then exit")
    return parser.parse_args()


def model_config():
    # Upstream defaults in third_party/xiaolongtao_tiger/model/main.py.
    return {"num_layers": 4, "num_decoder_layers": 4, "d_model": 128, "d_ff": 1024, "num_heads": 6,
            "d_kv": 64, "dropout_rate": 0.1, "vocab_size": 1025, "pad_token_id": 0, "eos_token_id": 0,
            "feed_forward_proj": "relu"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def seen_items(*frames):
    seen = {}
    for frame in frames:
        for user, items in zip(frame.user - 1, frame.items_seen):
            seen.setdefault(int(user), set()).update(items)
    return seen


def load_frames(data_dir):
    """Return valid and test frames with 0-based targets and the items to mask."""
    frames = {}
    for name in ("valid", "test"):
        frame = pd.read_parquet(data_dir / f"{name}.parquet")
        frame["target_idx"] = frame.target - 1
        frame["items_seen"] = [set((h - 1).tolist()) for h in frame.history]
        frames[name] = frame
    return frames["valid"], frames["test"]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.output_dir / "metrics_test.json"
    if test_path.exists():
        raise SystemExit(f"{test_path} exists: the test split was already evaluated for this run.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.output_dir / "train.log")])
    log = logging.getLogger("tiger")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    table = CodeTable(np.load(args.codes))
    valid_frame, test_frame = load_frames(args.data_dir)
    constrained = not args.no_constraints
    make = lambda name, mode: GenRecDataset(str(args.data_dir / f"{name}.parquet"), str(args.codes), mode, args.max_len)
    loader = lambda data, size, shuffle: GenRecDataLoader(data, batch_size=size, shuffle=shuffle, num_workers=args.workers)

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    write_json(args.output_dir / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": model_config(), "git_commit": git, "n_items": len(table.tokens), "selection": SELECTION,
        "constrained_beam": constrained})

    valid_data = make("valid", "evaluation")
    model = TIGER(model_config()).to(device)

    if args.check_scoring:
        users = valid_frame.user.to_numpy()[:args.check_scoring] - 1
        rankings = generate_rankings(model, loader(Subset(valid_data, range(args.check_scoring)), args.infer_size, False),
                                     table, args.beam_size, device, constrained)
        masked = apply_mask(rankings, users, seen_items(valid_frame))
        print(json.dumps({"untrained_scores_are_meaningless": score(masked, valid_frame.target_idx[:args.check_scoring], KS),
                          **real_item_counts(rankings)}, indent=2))
        return

    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(len(valid_data), size=min(args.val_sample_users, len(valid_data)), replace=False))
    sample_loader = loader(Subset(valid_data, sample.tolist()), args.infer_size, False)
    train_loader = loader(make("train", "train"), args.batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = {"epoch": 0, "best": -1.0, "best_epoch": 0, "bad": 0, "history": []}
    last_path = args.output_dir / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        log.info("resumed after epoch %d", state["epoch"])

    sample_users = valid_frame.user.to_numpy()[sample] - 1
    sample_targets = valid_frame.target_idx.to_numpy()[sample].tolist()
    seen_valid = seen_items(valid_frame)
    stopped_early = state["bad"] >= args.patience
    while state["epoch"] < args.epochs and not stopped_early:
        state["epoch"] += 1
        epoch = state["epoch"]
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        trained = time.time()
        rankings = generate_rankings(model, sample_loader, table, args.beam_size, device, constrained)
        metrics = score(apply_mask(rankings, sample_users, seen_valid), sample_targets, KS)
        metrics.update(real_item_counts(rankings))
        state["history"].append({"epoch": epoch, "loss": loss, **metrics})
        log.info("epoch %d loss=%.4f train=%.0fs val=%.0fs %s", epoch, loss, trained - start,
                 time.time() - trained, json.dumps(metrics))
        if metrics[SELECTION] > state["best"]:
            state.update(best=metrics[SELECTION], best_epoch=epoch, bad=0)
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        else:
            state["bad"] += 1
        temp = args.output_dir / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temp)
        temp.replace(last_path)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            log.info("stopping after epoch %d as requested", epoch)
            return
        stopped_early = state["bad"] >= args.patience

    log.info("training finished; best epoch %d", state["best_epoch"])
    model.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=device))

    # All validation users, for reporting. Selection already happened on the sample.
    rankings = generate_rankings(model, loader(valid_data, args.infer_size, False), table, args.beam_size, device, constrained)
    users = valid_frame.user.to_numpy() - 1
    write_json(args.output_dir / "metrics_val.json", {
        "selection": SELECTION, "best_epoch": state["best_epoch"], "sampled_users": len(sample),
        "history": state["history"],
        "all_users_mask_train": score(apply_mask(rankings, users, seen_valid), valid_frame.target_idx.tolist(), KS),
        **real_item_counts(rankings)})

    # The test split, once.
    test_data = make("test", "evaluation")
    rankings = generate_rankings(model, loader(test_data, args.infer_size, False), table, args.beam_size, device, constrained)
    users = test_frame.user.to_numpy() - 1
    targets = test_frame.target_idx.to_numpy()
    primary = apply_mask(rankings, users, seen_items(test_frame))
    degree = np.load(args.degree)[targets]
    buckets = {"target_degree_0": degree == 0, "target_degree_1_4": (degree >= 1) & (degree <= 4),
               "target_degree_5_plus": degree >= 5}
    pick = lambda items, keep: [item for item, flag in zip(items, keep) if flag]
    metrics = {
        "best_epoch": state["best_epoch"],
        "primary_mask_train_and_validation": score(primary, targets.tolist(), KS),
        "mask_train_only": score(apply_mask(rankings, users, seen_valid), targets.tolist(), KS),
        "no_mask": score(rankings, targets.tolist(), KS),
        "by_target_frequency": {name: {"fraction_of_targets": float(keep.mean()),
                                       **score(pick(primary, keep), targets[keep].tolist(), KS)}
                                for name, keep in buckets.items()},
        **real_item_counts(rankings),
    }
    write_json(test_path, metrics)
    log.info("test %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
