import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.lightgcn import LightGCN, NegativeSampler, bpr_loss, build_normalized_graph
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--out", required=True)
p.add_argument("--dim", type=int, default=64)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--reg", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = "cuda"

items = Path(args.items).read_text().split()
item_idx = {a: i for i, a in enumerate(items)}
sequences = build_sequences(load_interactions(args.reviews))
users = list(sequences)
pairs, val = [], []
for u, seq in enumerate(sequences.values()):
    idx = [item_idx[a] for a in seq]
    pairs += [(u, i) for i in idx[:-2]]
    val.append(idx[-2])
pairs = torch.tensor(pairs, device=dev)
val = torch.tensor(val, device=dev)
n_users, n_items = len(users), len(items)
graph = build_normalized_graph(pairs[:, 0], pairs[:, 1], n_users, n_items, dev)
seen = torch.zeros(n_users, n_items, dtype=torch.bool, device=dev)
seen[pairs[:, 0], pairs[:, 1]] = True
sampler = NegativeSampler(pairs[:, 0], pairs[:, 1], n_items)
model = LightGCN(n_users, n_items, args.dim, args.layers).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)


@torch.no_grad()
def recall10():
    user_emb, item_emb = model.split_embeddings(graph)
    scores = user_emb @ item_emb.T
    scores[seen] = -1e9
    rank = (scores > scores.gather(1, val[:, None])).sum(1)
    return (rank < 10).float().mean().item()


best, bad = -1, 0
for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(pairs), device=dev)
    for i in range(0, len(perm), 4096):
        batch = pairs[perm[i:i + 4096]]
        pos, neg = model(graph, batch[:, 0], batch[:, 1], sampler.sample(batch[:, 0]))
        loss = bpr_loss(pos, neg) + args.reg * model.l2_penalty(batch[:, 0], batch[:, 1], sampler.sample(batch[:, 0]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 5 == 0:
        r = recall10()
        print(epoch, round(loss.item(), 4), "val recall@10", round(r, 4), flush=True)
        if r > best:
            best, bad = r, 0
            with torch.no_grad():
                user_emb, item_emb = model.split_embeddings(graph)
            Path(args.out).mkdir(parents=True, exist_ok=True)
            np.save(f"{args.out}/item_emb.npy", item_emb.cpu().numpy())
            np.save(f"{args.out}/user_emb.npy", user_emb.cpu().numpy())
        else:
            bad += 1
            if bad >= 6:
                break
print("best val recall@10", best)
