"""SASRec with a full-softmax loss over all items: the collaborative-only sequential baseline, scored by the same evaluator."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evalkit import Evaluator
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
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
args = p.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"

items = Path(args.items).read_text().split()
idx = {a: i for i, a in enumerate(items)}
n = len(items)
seqs = [[idx[a] for a in s] for s in build_sequences(load_interactions(args.reviews)).values()]
train = [t for t in ([s[:-2][-(args.max_len + 1):] for s in seqs]) if len(t) >= 2]
val_hist = [s[:-2][-args.max_len:] for s in seqs]
val_target = torch.tensor([s[-2] for s in seqs], device=dev)
test_hist = [s[:-1][-args.max_len:] for s in seqs]
test_target = torch.tensor([s[-1] for s in seqs], device=dev)
count = np.bincount([i for s in seqs for i in s[:-2]], minlength=n)


def pad(rows, length, fill):
    out = torch.full((len(rows), length), fill, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.dim
        self.item = nn.Embedding(n + 1, d, padding_idx=n)
        self.pos = nn.Embedding(args.max_len, d)
        layer = nn.TransformerEncoderLayer(d, args.heads, d * 4, args.drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, args.layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(args.drop)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, ids):
        L = ids.shape[1]
        h = self.drop(self.item(ids) * math.sqrt(args.dim) + self.pos(torch.arange(L, device=ids.device)))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), 1)
        h = self.enc(h, mask=causal, src_key_padding_mask=ids == n)
        return self.norm(h)

    def logits(self, h):
        return h @ self.item.weight[:n].T


model = SASRec().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


@torch.no_grad()
def evaluate(hists, targets):
    model.eval()
    ev = Evaluator(n, count)
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
for epoch in range(1, args.epochs + 1):
    model.train()
    order = np.random.permutation(len(train))
    for b in range(0, len(order), args.batch):
        rows = [train[i] for i in order[b:b + args.batch]]
        L = max(len(r) for r in rows) - 1
        ids = pad([r[:-1] for r in rows], L, n).to(dev)
        lab = pad([r[1:] for r in rows], L, -100).to(dev)
        loss = F.cross_entropy(model.logits(model(ids)).reshape(-1, n), lab.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    v = evaluate(val_hist, val_target).summary()["raw"]
    log.append({"epoch": epoch, "loss": loss.item(), "val_ndcg@10": v["ndcg@10"], "val_recall@10": v["recall@10"]})
    print(epoch, round(loss.item(), 4), round(v["recall@10"], 4), round(v["ndcg@10"], 4), f"{time.time() - start:.0f}s", flush=True)
    if v["ndcg@10"] > best:
        best, bad, best_state = v["ndcg@10"], 0, {k: t.clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            break
train_seconds = time.time() - start
model.load_state_dict(best_state)
tev = evaluate(test_hist, test_target)
test = tev.summary()
result = {"best_val": best, "test": test["raw"], "test_filtered": test["filtered"], "test_beyond": test["beyond"],
          "params": sum(p.numel() for p in model.parameters()), "train_seconds": train_seconds,
          "score_ms_per_user": 0.0, "epochs_run": len(log), "args": vars(args)}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps(result, indent=1))
raw, filt = tev.ranks()
np.save(out / "test_ranks_raw.npy", raw)
np.save(out / "test_ranks_filtered.npy", filt)
print(json.dumps(result["test"], indent=1))
