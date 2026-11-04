"""Evaluate a completed content-aware TIGER checkpoint without training."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tiger_small import NextItemTransformer
from src.evaluation import evaluate_rankings


def main():
    checkpoint_path = Path("results/tiger_small_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    content = pd.read_pickle("data/processed/item_content_semantic_ids.pkl")
    n_items = int(max(train.item_idx.max(), test.item_idx.max()) + 1)

    content_map = dict(zip(content.item_id, content.content_embedding))
    matrix = np.zeros((n_items + 1, len(content.content_embedding.iloc[0])), dtype=np.float32)
    for item_id, item_idx in train[["item_id", "item_idx"]].drop_duplicates().itertuples(index=False):
        if item_id in content_map:
            matrix[int(item_idx) + 1] = content_map[item_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextItemTransformer(n_items, 128, max_length=20, content_features=matrix.shape[1]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.content_embeddings.copy_(torch.tensor(matrix, device=device))

    histories = train.sort_values(["user_idx", "timestamp"]).groupby("user_idx")["item_idx"].apply(list).to_dict()
    rankings, targets = [], []
    with torch.no_grad():
        for row in test.itertuples():
            history = histories.get(row.user_idx, [])[-20:]
            tokens = torch.tensor([[0] * (20 - len(history)) + [i + 1 for i in history]], device=device)
            scores = model(tokens)[0]
            scores[0] = -torch.inf
            top = torch.topk(scores, k=10).indices.cpu().tolist()
            rankings.append([i - 1 for i in top])
            targets.append(int(row.item_idx))
    print(f"checkpoint epoch={checkpoint.get('epoch')} batch={checkpoint.get('batch')}")
    print("content-aware TIGER metrics:", evaluate_rankings(rankings, targets))


if __name__ == "__main__":
    main()

