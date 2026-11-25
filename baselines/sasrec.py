import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data import build_split, load_items, load_reviews, pad
from src.evaluator import Evaluator

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--out", required=True)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--drop", type=float, default=0.3)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--max-len", type=int, default=50)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--seed", type=int, default=2024)
a = p.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(a.data)
items, idx = load_items(d / "items.txt")
n = len(items)
split = build_split(load_reviews(d / "reviews.json.gz"), idx, a.max_len)
val_t, test_t = (torch.tensor(t, device=dev) for t in (split.val_target, split.test_target))


class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.item = nn.Embedding(n + 1, a.dim, padding_idx=n)
        self.pos = nn.Embedding(a.max_len, a.dim)
        layer = nn.TransformerEncoderLayer(a.dim, a.heads, a.dim * 4, a.drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, a.layers)
        self.norm = nn.LayerNorm(a.dim)
        self.drop = nn.Dropout(a.drop)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, ids):
        L = ids.shape[1]
        h = self.drop(self.item(ids) * math.sqrt(a.dim) + self.pos(torch.arange(L, device=ids.device)))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), 1)
        return self.norm(self.enc(h, mask=causal, src_key_padding_mask=ids == n))

    def logits(self, h):
        return h @ self.item.weight[:n].T


model = SASRec().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)


@torch.no_grad()
def evaluate(hists, targets):
    model.eval()
    ev = Evaluator(n, split.count)
    for b in range(0, len(hists), 512):
        rows = hists[b:b + 512]
        L = max(map(len, rows))
        ids = pad(rows, L, n).to(dev)
        lens = torch.tensor([len(r) for r in rows], device=dev)
        h = model(ids)[torch.arange(len(rows)), lens - 1]
        ev.update(model.logits(h), targets[b:b + 512], pad(rows, L, -1).to(dev))
    return ev


best, bad, best_state, log = -1, 0, None, []
start = time.time()
for epoch in range(1, a.epochs + 1):
    model.train()
    order = np.random.permutation(len(split.train))
    for b in range(0, len(order), a.batch):
        rows = [split.train[i] for i in order[b:b + a.batch]]
        L = max(len(r) for r in rows) - 1
        ids = pad([r[:-1] for r in rows], L, n).to(dev)
        labels = pad([r[1:] for r in rows], L, -100).to(dev)
        for j, i in enumerate(order[b:b + a.batch]):
            if split.last_only[i]:
                labels[j, :len(rows[j]) - 2] = -100
        loss = F.cross_entropy(model.logits(model(ids)).reshape(-1, n), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    v = evaluate(split.val_hist, val_t).summary()["raw"]
    log.append({"epoch": epoch, "loss": loss.item(), "val_ndcg@10": v["ndcg@10"], "val_recall@10": v["recall@10"]})
    print(epoch, round(loss.item(), 4), round(v["recall@10"], 4), round(v["ndcg@10"], 4), f"{time.time() - start:.0f}s", flush=True)
    if v["ndcg@10"] > best:
        best, bad, best_state = v["ndcg@10"], 0, {k: t.clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience:
            break
model.load_state_dict(best_state)
ev = evaluate(split.test_hist, test_t)
test = ev.summary()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps({"best_val": best, "test": test["raw"], "test_filtered": test["filtered"], "test_beyond": test["beyond"],
                                             "params": sum(p.numel() for p in model.parameters()), "train_seconds": time.time() - start,
                                             "epochs_run": len(log), "args": vars(a)}, indent=1))
raw, filt = ev.ranks()
np.save(out / "test_ranks_raw.npy", raw)
np.save(out / "test_ranks_filtered.npy", filt)
print(json.dumps(test["raw"], indent=1))
