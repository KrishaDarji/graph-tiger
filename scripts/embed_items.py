import argparse
import ast
import gzip
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# the item text is built exactly as in RPG's official data code (fields, cleaning and its quirks), so the encoder sees identical inputs
FIELDS = ["title", "price", "brand", "feature", "categories", "description"]


def clean_text(x):
    x = ", ".join(map(str, x)) if isinstance(x, list) else x
    x = html.unescape(x).strip()
    x = re.sub(r"<[^>]+>", "", x)
    x = re.sub(r"[\n\t]", " ", x)
    x = re.sub(r" +", " ", x)
    return re.sub(r"[^\x00-\x7F]", " ", x)


def part(raw):
    if isinstance(raw, float):
        return f"{raw}. "
    if len(raw) > 0 and isinstance(raw[0], list):
        return ", ".join(clean_text(v)[:-1] for group in raw for v in group) + ". "
    if isinstance(raw, list):
        return "".join(clean_text(v) for v in raw) + " "
    return clean_text(raw) + " "


def parse(line):
    return ast.literal_eval(line.replace(b"true", b"True").replace(b"false", b"False").decode())


p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--model", default="sentence-transformers/sentence-t5-base")
a = p.parse_args()
d = Path(a.data)

seqs = defaultdict(list)
with gzip.open(d / "reviews.json.gz", "rb") as f:
    for line in f:
        r = parse(line)
        seqs[r["reviewerID"]].append((r["asin"], int(r["unixReviewTime"])))
items = {}  # item order: users in file order, each user's items sorted by time (as RPG assigns ids)
for seq in seqs.values():
    for asin, _ in sorted(seq, key=lambda x: x[1]):
        items.setdefault(asin, len(items))
meta = {}
with gzip.open(d / "meta.json.gz", "rb") as f:
    for line in f:
        row = parse(line)
        if row["asin"] in items:
            meta[row["asin"]] = "".join(part(row[k]) for k in FIELDS if k in row)

model = SentenceTransformer(a.model)
emb = model.encode([meta[i] for i in items], batch_size=512, convert_to_numpy=True)
np.save(d / "emb.npy", emb.astype(np.float32))
(d / "items.txt").write_text("\n".join(items) + "\n")
(d / "item_text.json").write_text(json.dumps(meta))
print(len(items), "items", emb.shape)
