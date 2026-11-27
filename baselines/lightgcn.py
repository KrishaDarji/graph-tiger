import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data import load_items, load_reviews, user_sequences

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/beauty")
p.add_argument("--out", default="", help="folder for item_emb.npy (default: <data>/lightgcn)")
p.add_argument("--dim", type=int, default=64)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--reg", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = Path(a.data)
items, idx = load_items(d / "items.txt")
n_items = len(items)
seqs = [[idx[x] for x in s] for s in user_sequences(load_reviews(d / "reviews.json.gz")).values()]
n_users = len(seqs)
pairs = torch.tensor([(u, i) for u, s in enumerate(seqs) for i in s[:-2]], device=dev)
val = torch.tensor([s[-2] for s in seqs], device=dev)


def normalized_graph():
    users, its = pairs[:, 0], pairs[:, 1] + n_users
    rows, cols = torch.cat([users, its]), torch.cat([its, users])
    deg = torch.zeros(n_users + n_items, device=dev).scatter_add_(0, rows, torch.ones(len(rows), device=dev))
    w = 1.0 / deg[rows].sqrt() / deg[cols].sqrt()
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), w, (n_users + n_items,) * 2).coalesce()


class LightGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(n_users + n_items, a.dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, g):
        layers, cur = [self.emb.weight], self.emb.weight
        for _ in range(a.layers):
            cur = torch.sparse.mm(g, cur)
            layers.append(cur)
        return torch.split(torch.stack(layers).mean(0), [n_users, n_items])


keys = torch.sort(pairs[:, 0] * n_items + pairs[:, 1]).values


def sample_negatives(users):
    neg = torch.randint(n_items, users.shape, device=dev)
    while True:  # redraw items the user has interacted with
        q = users * n_items + neg
        bad = keys[torch.searchsorted(keys, q).clamp_max(len(keys) - 1)] == q
        if not bad.any():
            return neg
        neg[bad] = torch.randint(n_items, (int(bad.sum()),), device=dev)


g = normalized_graph()
seen = torch.zeros(n_users, n_items, dtype=torch.bool, device=dev)
seen[pairs[:, 0], pairs[:, 1]] = True
model = LightGCN().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
out = Path(a.out or d / "lightgcn")
out.mkdir(parents=True, exist_ok=True)
best, bad = -1, 0
for epoch in range(1, a.epochs + 1):
    perm = torch.randperm(len(pairs), device=dev)
    for b in range(0, len(perm), 4096):
        batch = pairs[perm[b:b + 4096]]
        u, pos = batch[:, 0], batch[:, 1]
        neg = sample_negatives(u)
        ue, ie = model.propagate(g)
        loss = F.softplus((ue[u] * ie[neg]).sum(1) - (ue[u] * ie[pos]).sum(1)).mean()
        e0 = model.emb.weight
        loss = loss + a.reg * 0.5 * torch.cat([e0[u], e0[n_users + pos], e0[n_users + neg]]).pow(2).sum() / len(u)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 5 == 0:
        with torch.no_grad():
            ue, ie = model.propagate(g)
            scores = (ue @ ie.T).masked_fill(seen, -1e9)
            recall = ((scores > scores.gather(1, val[:, None])).sum(1) < 10).float().mean().item()
        print(epoch, round(loss.item(), 4), "val recall@10", round(recall, 4), flush=True)
        if recall > best:
            best, bad = recall, 0
            np.save(out / "item_emb.npy", ie.cpu().numpy())
        else:
            bad += 1
            if bad >= 6:
                break
print("best val recall@10", best)
