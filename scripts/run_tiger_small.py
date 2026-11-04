"""Train and evaluate the lightweight TIGER-style next-item model."""
import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tiger_small import NextItemTransformer
from src.evaluation import evaluate_rankings


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SequenceDataset(Dataset):
    def __init__(self, frame, max_length, limit=0):
        self.examples = []
        for _, group in frame.sort_values(["user_idx", "timestamp"]).groupby("user_idx"):
            items = group.item_idx.astype(int).tolist()
            for end in range(1, len(items)):
                history = items[max(0, end - max_length):end]
                target = items[end] + 1
                tokens = [item + 1 for item in history]
                tokens = [0] * (max_length - len(tokens)) + tokens
                self.examples.append((tokens, target))
                if limit and len(self.examples) >= limit:
                    return

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokens, target = self.examples[index]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Limit training examples for a quick pilot")
    parser.add_argument("--eval-limit", type=int, default=1000, help="Number of test users to evaluate; 0 means all")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/tiger_small_checkpoint.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-content", action="store_true")
    parser.add_argument("--steps-per-job", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    seed_everything(args.seed)

    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    n_items = int(max(train.item_idx.max(), test.item_idx.max()) + 1)
    dev = choose_device()
    dataset = SequenceDataset(train, args.max_length, args.limit)
    # Fixed order is required so a later Hazel job can resume at a batch index.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    content_matrix = None
    if args.use_content:
        content = pd.read_pickle("data/processed/item_content_semantic_ids.pkl")
        content_map = dict(zip(content.item_id, content.content_embedding))
        content_matrix = np.zeros((n_items + 1, len(content.content_embedding.iloc[0])), dtype=np.float32)
        for item_id, item_idx in train[["item_id", "item_idx"]].drop_duplicates().itertuples(index=False):
            if item_id in content_map:
                content_matrix[int(item_idx) + 1] = content_map[item_id]
        print(f"using content features: {content_matrix.shape[1]}")
    model = NextItemTransformer(
        n_items, args.dim, max_length=args.max_length,
        content_features=None if content_matrix is None else content_matrix.shape[1],
    ).to(dev)
    if content_matrix is not None:
        model.content_embeddings.copy_(torch.tensor(content_matrix, device=dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    start_epoch = 1
    start_batch = 0
    if args.resume and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=dev)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        start_epoch = int(state.get("epoch", 1))
        start_batch = int(state.get("batch", 0))
        print(f"resuming at epoch {start_epoch}, batch {start_batch}")
    print(f"device={dev} items={n_items:,} examples={len(dataset):,}")

    completed_steps = 0
    epoch = start_epoch
    while epoch <= args.epochs and completed_steps < args.steps_per_job:
        model.train()
        total = 0.0
        for batch_index, (tokens, targets) in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            tokens, targets = tokens.to(dev), targets.to(dev)
            logits = model(tokens)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(tokens)
            completed_steps += 1
            next_batch = batch_index + 1
            if completed_steps % args.checkpoint_every == 0:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "batch": next_batch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "n_items": n_items}, args.checkpoint)
            if completed_steps >= args.steps_per_job:
                break
        else:
            print(f"epoch {epoch}/{args.epochs} completed")
            epoch += 1
            start_batch = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "batch": 0,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "n_items": n_items}, args.checkpoint)

    print(f"processed {completed_steps} batches; checkpoint epoch={epoch} batch={next_batch if 'next_batch' in locals() else 0}")

    if epoch <= args.epochs:
        print("Job chunk complete; skipping evaluation until final epoch.")
        return

    model.eval()
    histories = train.sort_values(["user_idx", "timestamp"]).groupby("user_idx")["item_idx"].apply(list).to_dict()
    rankings, targets = [], []
    with torch.no_grad():
        eval_rows = test if not args.eval_limit else test.head(args.eval_limit)
        for row in eval_rows.itertuples():
            history = histories.get(row.user_idx, [])[-args.max_length:]
            tokens = torch.tensor([[0] * (args.max_length - len(history)) + [i + 1 for i in history]], device=dev)
            scores = model(tokens)[0]
            scores[0] = -torch.inf
            top = torch.topk(scores, k=10).indices.cpu().tolist()
            rankings.append([i - 1 for i in top])
            targets.append(int(row.item_idx))
    print("metrics", evaluate_rankings(rankings, targets))
    Path("results").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "n_items": n_items}, "results/tiger_small.pt")
    print("saved results/tiger_small.pt")


if __name__ == "__main__":
    main()

