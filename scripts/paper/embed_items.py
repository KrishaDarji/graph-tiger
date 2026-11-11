import argparse
import gzip
import json
import ast
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--meta", required=True)
p.add_argument("--model", default="sentence-transformers/sentence-t5-base")
p.add_argument("--out", required=True)
p.add_argument("--bf16", action="store_true")
args = p.parse_args()

items = []
with gzip.open(args.reviews, "rt") as f:
    for line in f:
        asin = json.loads(line)["asin"]
        if asin not in items:
            items.append(asin)
index = set(items)
meta = {}
with gzip.open(args.meta, "rt") as f:
    for line in f:
        row = ast.literal_eval(line)
        if row["asin"] in index:
            meta[row["asin"]] = row


def sentence(row):
    cats = []
    for path in row.get("categories") or []:
        cats += [c for c in path if c not in cats]
    fields = [("Title", row.get("title")), ("Price", row.get("price")), ("Brand", row.get("brand")),
              ("Categories", ", ".join(cats))]
    return ". ".join(f"{k}: {v}" for k, v in fields if v)


texts = [sentence(meta[a]) for a in items]
model = SentenceTransformer(args.model, model_kwargs={"torch_dtype": torch.bfloat16} if args.bf16 else {})
emb = model.encode(texts, batch_size=32, convert_to_numpy=True)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.save(out / "emb.npy", emb.astype(np.float32))
(out / "items.txt").write_text("\n".join(items) + "\n")
print(len(items), "items", emb.shape, "norm", float(np.linalg.norm(emb, axis=1).mean()))
print(texts[0])
