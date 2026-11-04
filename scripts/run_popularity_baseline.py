"""Evaluate a chronological popularity recommender."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_rankings


def main():
    path = Path("data/processed/interactions.parquet")
    if not path.exists():
        raise SystemExit("Run scripts/prepare_data.py first.")
    df = pd.read_parquet(path).sort_values(["user_id", "timestamp"])
    groups = df.groupby("user_id", sort=False)
    train = groups.head(-2)
    test = groups.tail(1)
    popular = train["item_id"].value_counts().index.tolist()
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    targets, rankings = [], []
    for row in test.itertuples():
        user_seen = seen.get(row.user_id, set())
        candidates = []
        for item in popular:
            if item not in user_seen:
                candidates.append(item)
                if len(candidates) == 10:
                    break
        targets.append(row.item_id)
        rankings.append(candidates)
    metrics = evaluate_rankings(rankings, targets)
    print("Popularity baseline")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

