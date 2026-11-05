"""Build the 5-core Beauty files used to reproduce the XiaoLongtaoo/TIGER result.

This dataset is only a code check. It is separate from the project's split in
data/processed. Item IDs start at 1, as the upstream dataset code expects, and
items.txt line i (from 0) is item ID i + 1.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import pandas as pd


def rows_for(sequences):
    return pd.DataFrame([{"user": user, "history": seq[:-1], "target": seq[-1]}
                         for user, seq in sequences.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=Path("data/tiger_5core/raw/reviews_Beauty_5.json.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/tiger_5core/Beauty"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    users, items, events = {}, {}, defaultdict(list)
    with gzip.open(args.reviews, "rt", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            user = users.setdefault(review["reviewerID"], len(users) + 1)
            item = items.setdefault(review["asin"], len(items) + 1)
            events[user].append((item, review["unixReviewTime"]))
    # sort() is stable, so same-day ties keep file order, as upstream does.
    sequences = {u: [i for i, _ in sorted(e, key=lambda x: x[1])] for u, e in events.items()}

    rows_for({u: s[:-2] for u, s in sequences.items()}).to_parquet(args.output_dir / "train.parquet", index=False)
    rows_for({u: s[:-1] for u, s in sequences.items()}).to_parquet(args.output_dir / "valid.parquet", index=False)
    rows_for(sequences).to_parquet(args.output_dir / "test.parquet", index=False)
    (args.output_dir / "items.txt").write_text("\n".join(items) + "\n")
    print(f"users={len(users):,} items={len(items):,} interactions={sum(map(len, sequences.values())):,}")


if __name__ == "__main__":
    main()
