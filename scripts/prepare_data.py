"""Download and prepare Amazon Beauty review data.

The downloader accepts the public Amazon review JSON-lines URL. If network
access is unavailable, pass --input with a local .json or .json.gz file.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import urllib.request

import pandas as pd

REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Beauty.json.gz"


def read_json_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--input", type=Path, help="Local review JSON/JSON.GZ file")
    parser.add_argument("--output", type=Path, default=Path("data/processed/interactions.parquet"))
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = args.input
    if args.download:
        source = raw_dir / "reviews_Beauty_5.json.gz"
        if not source.exists():
            download(REVIEWS_URL, source)
    if source is None:
        parser.error("Use --download or --input PATH")

    rows = []
    for row in read_json_lines(source):
        rows.append({
            "user_id": row.get("reviewerID"),
            "item_id": row.get("asin"),
            "timestamp": row.get("unixReviewTime", row.get("reviewTime")),
            "rating": row.get("overall"),
        })
    df = pd.DataFrame(rows).dropna(subset=["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(["user_id", "item_id", "timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp"])
    counts = df.groupby("user_id").size()
    df = df[df["user_id"].isin(counts[counts >= 3].index)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df):,} interactions to {args.output}")
    print(f"Users: {df.user_id.nunique():,}; items: {df.item_id.nunique():,}")


if __name__ == "__main__":
    main()

