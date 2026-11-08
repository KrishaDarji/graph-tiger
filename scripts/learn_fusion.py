import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["scalar", "gate"], required=True)
p.add_argument("--content", default="data/processed/item_st5_emb.npy")
p.add_argument("--graph-dir", default="results/lightgcn_768")
p.add_argument("--split-dir", default="data/processed")
p.add_argument("--output", required=True)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--temp", type=float, default=0.05)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = "cuda"


def unit(x):
    return F.normalize(x, dim=1)


content = unit(torch.tensor(np.load(args.content), device=dev))
cold = torch.tensor(np.load(f"{args.graph_dir}/cold_item_mask.npy"), device=dev)
graph = torch.tensor(np.load(f"{args.graph_dir}/item_emb.npy"), device=dev)
graph = unit(graph) * (~cold)[:, None]
log_deg = torch.log1p(torch.tensor(np.load(f"{args.graph_dir}/item_train_degree.npy"), device=dev).float())[:, None]
n_items = len(content)

# history = a user's train items (last 20), target = their validation item. LightGCN never saw
# the validation edge, so fitting on it does not reward memorised graph edges.
train = pd.read_parquet(f"{args.split_dir}/train.parquet").sort_values(["user_idx", "timestamp"], kind="stable")
val = pd.read_parquet(f"{args.split_dir}/validation.parquet").set_index("user_idx").item_idx
users = np.array(sorted(val.index))
hist = np.full((len(users), 20), n_items)
for row, (_, items) in zip(range(len(users)), train.groupby("user_idx")["item_idx"]):
    items = items.to_numpy()[-20:]
    hist[row, :len(items)] = items
hist, target = torch.tensor(hist, device=dev), torch.tensor(val.loc[users].to_numpy(), device=dev)
order = torch.randperm(len(users), generator=torch.Generator().manual_seed(args.seed)).to(dev)
cut = int(0.8 * len(users))
fit, hold = order[:cut], order[cut:]


class Fusion(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "scalar":
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.gate = nn.Linear(2 * 768 + 1, 768)

    def forward(self):
        if self.mode == "scalar":
            return unit(content + self.alpha * graph)
        gate = torch.sigmoid(self.gate(torch.cat([content, graph, log_deg], 1)))
        return unit(content + gate * graph)


def scores(items, hist_batch):
    mask = (hist_batch < n_items).float()
    user = (items[hist_batch.clamp_max(n_items - 1)] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return unit(user) @ items.T


@torch.no_grad()
def holdout(items):
    hits = ndcg = 0.0
    for i in range(0, len(hold), 2048):
        rows = hold[i:i + 2048]
        s = scores(items, hist[rows])
        s.scatter_(1, hist[rows].clamp_max(n_items - 1), -1e9)
        rank = (s > s.gather(1, target[rows][:, None])).sum(1)
        hits += (rank < 10).sum().item()
        ndcg += (1 / torch.log2(rank[rank < 10].float() + 2)).sum().item()
    return {"recall@10": hits / len(hold), "ndcg@10": ndcg / len(hold)}


report = {"content_only": holdout(content), "fixed_alpha_1": holdout(unit(content + graph))}
model = Fusion(args.mode).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
best = {"ndcg@10": -1}
for epoch in range(args.epochs):
    perm = fit[torch.randperm(len(fit), device=dev)]
    for i in range(0, len(perm), args.batch):
        rows = perm[i:i + args.batch]
        loss = F.cross_entropy(scores(model(), hist[rows]) / args.temp, target[rows])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        result = holdout(model())
    print(epoch, round(loss.item(), 4), result, flush=True)
    if result["ndcg@10"] > best["ndcg@10"]:
        best = {**result, "epoch": epoch}
        torch.save(model.state_dict(), "/tmp/fusion_best.pt")
model.load_state_dict(torch.load("/tmp/fusion_best.pt"))
with torch.no_grad():
    fused = model()
    report["learned"] = best
    report["learned_alpha" if args.mode == "scalar" else "mean_gate"] = (
        model.alpha.item() if args.mode == "scalar"
        else torch.sigmoid(model.gate(torch.cat([content, graph, log_deg], 1)))[~cold].mean().item())
    assert torch.allclose(fused[cold], content[cold], atol=1e-5), "cold items must equal content"
np.save(args.output, fused.cpu().numpy().astype(np.float32))
Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))