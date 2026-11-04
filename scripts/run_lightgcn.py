"""Train a small LightGCN model with BPR loss."""
import argparse
from pathlib import Path
import sys
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_interactions, save_split
from src.lightgcn import LightGCN, build_normalized_graph


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    split_path = Path("data/processed/train.parquet")
    if not split_path.exists():
        save_split(load_interactions())
    train = pd.read_parquet(split_path)
    test = pd.read_parquet("data/processed/test.parquet")
    n_users = int(max(train.user_idx.max(), test.user_idx.max()) + 1)
    n_items = int(max(train.item_idx.max(), test.item_idx.max()) + 1)
    dev = device()
    print(f"Training LightGCN on {dev}")

    users = torch.tensor(train.user_idx.to_numpy(), dtype=torch.long, device=dev)
    items = torch.tensor(train.item_idx.to_numpy(), dtype=torch.long, device=dev)
    graph = build_normalized_graph(users, items, n_users, n_items, dev)
    model = LightGCN(n_users, n_items, args.dim, args.layers).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    positives_by_user = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    for epoch in range(1, args.epochs + 1):
        negatives = []
        for user in train.user_idx:
            candidate = int(torch.randint(n_items, (1,)).item())
            seen = positives_by_user[int(user)]
            while candidate in seen:
                candidate = int(torch.randint(n_items, (1,)).item())
            negatives.append(candidate)
        neg = torch.tensor(negatives, dtype=torch.long, device=dev)
        pos_score, neg_score = model(graph, users, items, neg)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score).clamp_min(1e-8)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}/{args.epochs} loss={loss.item():.4f}")

    Path("results").mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "n_users": n_users, "n_items": n_items}, "results/lightgcn.pt")
    print("Saved results/lightgcn.pt")


if __name__ == "__main__":
    main()

