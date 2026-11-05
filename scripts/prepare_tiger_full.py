"""Write the project's existing split in the T5 trainer's input format.

Reads data/processed split files without changing them. T5 item ID = item_idx + 1,
because 0 is the padding token. Validation history is the train items; test history
is the train items plus the validation item.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lightgcn_data import item_ids_by_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/tiger"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["user_idx", "item_idx", "item_id", "timestamp"]
    train = pd.read_parquet(args.processed_dir / "train.parquet", columns=columns)
    validation = pd.read_parquet(args.processed_dir / "validation.parquet", columns=columns)
    test = pd.read_parquet(args.processed_dir / "test.parquet", columns=columns)
    item_ids = item_ids_by_index(train, validation, test)
    content = pd.read_pickle(args.processed_dir / "item_content_semantic_ids.pkl")
    if not np.array_equal(content.item_id.to_numpy(dtype=str), item_ids):
        raise SystemExit("item_idx is not aligned with the content file")

    # The split files keep each user's chronological order.
    histories = {u: (g + 1).tolist() for u, g in train.groupby("user_idx", sort=True)["item_idx"]}
    val_target = dict(zip(validation.user_idx, validation.item_idx + 1))
    test_target = dict(zip(test.user_idx, test.item_idx + 1))
    if not histories.keys() == val_target.keys() == test_target.keys():
        raise SystemExit("Users differ across splits")

    def write(name, rows):
        pd.DataFrame(rows, columns=["user", "history", "target"]).to_parquet(args.output_dir / name, index=False)

    # The trainer rebuilds sliding windows from history + target.
    write("train.parquet", [(u + 1, s[:-1], s[-1]) for u, s in histories.items()])
    write("valid.parquet", [(u + 1, s, int(val_target[u])) for u, s in histories.items()])
    write("test.parquet", [(u + 1, s + [int(val_target[u])], int(test_target[u])) for u, s in histories.items()])
    n_windows = sum(len(s) - 1 for s in histories.values())
    print(f"users={len(histories):,} items={len(item_ids):,} training windows={n_windows:,}")


if __name__ == "__main__":
    main()
