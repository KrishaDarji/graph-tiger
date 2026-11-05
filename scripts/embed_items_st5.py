"""Encode item metadata with Sentence-T5, keeping one row per item in index order.

--dataset full writes data/processed/item_st5_emb.npy, where row i is item_idx i.
--dataset 5core writes data/tiger_5core/Beauty/item_emb.npy, where row i is item ID i + 1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.item_text import TEMPLATES, read_metadata
from src.lightgcn_data import item_ids_by_index

MODEL = "sentence-transformers/sentence-t5-base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["full", "5core"], required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/raw/meta_Beauty.json.gz"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Pilot on the first N items")
    parser.add_argument("--output-suffix", default="", help="Added to output names for pilots")
    args = parser.parse_args()

    if args.dataset == "full":
        processed = Path("data/processed")
        frames = [pd.read_parquet(processed / f"{s}.parquet", columns=["item_idx", "item_id"])
                  for s in ("train", "validation", "test")]
        item_ids = item_ids_by_index(*frames).tolist()
        template, stem = "project", processed / f"item_st5_emb{args.output_suffix}"
    else:
        folder = Path("data/tiger_5core/Beauty")
        item_ids = (folder / "items.txt").read_text().split()
        template, stem = "upstream", folder / f"item_emb{args.output_suffix}"

    metadata = read_metadata(args.meta, set(item_ids))
    missing = [i for i in item_ids if i not in metadata]
    if missing:
        raise SystemExit(f"{len(missing)} items have no metadata, for example {missing[:3]}")
    if args.limit:
        item_ids = item_ids[:args.limit]
    build = TEMPLATES[template]
    texts = [build(metadata[i]) for i in item_ids]

    model = SentenceTransformer(MODEL)
    start = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)
    seconds = time.time() - start
    if embeddings.shape != (len(item_ids), 768) or not np.isfinite(embeddings).all():
        raise SystemExit(f"Bad embeddings: shape {embeddings.shape}")

    tokens = model.tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
    limit = model.max_seq_length
    info = {
        "model": MODEL, "template": template, "n_items": len(item_ids), "dim": embeddings.shape[1],
        "seconds": round(seconds, 1), "max_seq_length": limit,
        "items_truncated": int(sum(len(t) > limit for t in tokens)),
        "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "item_id_sha256": hashlib.sha256("\n".join(item_ids).encode()).hexdigest(),
        "duplicate_text_items": len(texts) - len(set(texts)),
    }
    np.save(f"{stem}.npy", embeddings)
    Path(f"{stem}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print("example text:\n" + texts[0][:300])


if __name__ == "__main__":
    main()
