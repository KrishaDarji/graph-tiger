"""Build lightweight content representations and semantic IDs for Beauty items.

This is a reproducible project-scale approximation of TIGER's semantic-ID
stage: TF-IDF text features are reduced with SVD and quantized into multiple
discrete codebooks. It can later be replaced with pretrained Sentence-T5 and
RQ-VAE without changing the downstream file interface.
"""
import argparse
import ast
import gzip
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Beauty.json.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/meta_Beauty.json.gz"))
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--codes", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=256)
    args = parser.parse_args()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if not args.raw.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, args.raw)

    rows = []
    with gzip.open(args.raw, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)
                text = " ".join([
                    str(row.get("title", "")),
                    str(row.get("brand", "")),
                    " ".join(row.get("description", []) or []),
                ]).strip()
                rows.append({"item_id": row.get("asin"), "text": text})
    metadata = pd.DataFrame(rows).dropna(subset=["item_id"]).drop_duplicates("item_id")
    interactions = pd.read_parquet("data/processed/interactions.parquet")
    items = pd.DataFrame({"item_id": interactions.item_id.unique()})
    metadata = items.merge(metadata, on="item_id", how="left").fillna({"text": "unknown product"})

    vectorizer = TfidfVectorizer(max_features=20000, min_df=2, stop_words="english")
    tfidf = vectorizer.fit_transform(metadata.text)
    n_components = min(args.components, max(2, tfidf.shape[1] - 1))
    embeddings = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(tfidf)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    codes = []
    residual = embeddings.copy()
    for code_index in range(args.codes):
        k = min(args.codebook_size, len(metadata))
        clusterer = MiniBatchKMeans(n_clusters=k, random_state=42 + code_index,
                                    batch_size=2048, n_init=3)
        labels = clusterer.fit_predict(residual)
        codes.append(labels)
        residual = residual - clusterer.cluster_centers_[labels]
    output = metadata[["item_id"]].copy()
    output["content_embedding"] = [row.tolist() for row in embeddings]
    for i, labels in enumerate(codes):
        output[f"semantic_id_{i}"] = labels
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    output.to_pickle("data/processed/item_content_semantic_ids.pkl")
    print(f"Saved {len(output):,} item representations")
    print(f"Output: data/processed/item_content_semantic_ids.pkl")


if __name__ == "__main__":
    main()

