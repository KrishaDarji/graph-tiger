"""Print nearest neighbours by LightGCN and by content embedding, with titles."""
import argparse
import ast
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lightgcn_data import item_ids_by_index


def load_titles(path):
    titles = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)
                titles[row.get("asin")] = row.get("title")
    return titles


def normalize(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/lightgcn"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--meta", type=Path, default=Path("data/raw/meta_Beauty.json.gz"))
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=5)
    args = parser.parse_args()

    frames = [pd.read_parquet(args.processed_dir / f"{s}.parquet", columns=["item_idx", "item_id"])
              for s in ("train", "validation", "test")]
    item_ids = item_ids_by_index(*frames)
    titles = load_titles(args.meta)
    content = pd.read_pickle(args.processed_dir / "item_content_semantic_ids.pkl")
    content_emb = normalize(np.vstack(content.content_embedding.to_numpy()))
    graph_emb = normalize(np.load(args.run_dir / "item_emb.npy"))
    cold = np.load(args.run_dir / "cold_item_mask.npy")
    degree = np.load(args.run_dir / "item_train_degree.npy")

    def describe(index):
        title = titles.get(item_ids[index]) or "(no title)"
        return f"[{index}] {item_ids[index]} deg={degree[index]} {str(title)[:80]}"

    def neighbors(matrix, query, excluded=None):
        scores = matrix @ matrix[query]
        scores[query] = -np.inf
        if excluded is not None:
            scores[excluded] = -np.inf
        return np.argsort(-scores)[:args.n_neighbors]

    for query in np.argsort(-degree)[:args.n_items]:
        print(f"\nQUERY {describe(query)}")
        print("  LightGCN (items with training edges only):")
        for index in neighbors(graph_emb, query, cold):
            print(f"    {describe(index)}")
        print("  Content:")
        for index in neighbors(content_emb, query):
            print(f"    {describe(index)}")


if __name__ == "__main__":
    main()
