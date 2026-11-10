import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--emb", required=True)
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=20000)
p.add_argument("--lr", type=float, default=0.4)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--levels", type=int, default=3)
p.add_argument("--codes", type=int, default=256)
p.add_argument("--beta", type=float, default=0.25)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = "cuda"


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def kmeans(x, k, iters=100):
    centers = x[torch.randperm(len(x), device=x.device)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(1)
        for j in range(k):
            members = x[assign == j]
            if len(members):
                centers[j] = members.mean(0)
    return centers


class RQVAE(nn.Module):
    def __init__(self, d_in=768):
        super().__init__()
        self.enc, self.dec = mlp([d_in, 512, 256, 128, 32]), mlp([32, 128, 256, 512, d_in])
        self.books = nn.Parameter(torch.randn(args.levels, args.codes, 32))

    def quantize(self, z):
        residual, total, loss, codes = z, 0, 0, []
        for level in range(args.levels):
            book = self.books[level]
            idx = torch.cdist(residual, book).argmin(1)
            e = book[idx]
            loss = loss + ((residual.detach() - e) ** 2).sum(1).mean() + args.beta * ((residual - e.detach()) ** 2).sum(1).mean()
            total, residual = total + e, residual - e
            codes.append(idx)
        return total, loss, torch.stack(codes, 1)

    def forward(self, x):
        z = self.enc(x)
        zq, qloss, codes = self.quantize(z)
        out = self.dec(z + (zq - z).detach())
        return ((x - out) ** 2).sum(1).mean(), qloss, codes

    @torch.no_grad()
    def init_books(self, x):
        residual = self.enc(x)
        for level in range(args.levels):
            self.books.data[level] = kmeans(residual, args.codes)
            residual = residual - self.books[level][torch.cdist(residual, self.books[level]).argmin(1)]


x = torch.tensor(np.load(args.emb), device=dev)
model = RQVAE(x.shape[1]).to(dev)
opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=0.1, eps=1e-7)
model.init_books(x[torch.randperm(len(x), device=dev)[:args.batch]])
start = time.time()
for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(x), device=dev)
    for i in range(0, len(x), args.batch):
        recon, qloss, _ = model(x[perm[i:i + args.batch]])
        opt.zero_grad()
        (recon + qloss).backward()
        opt.step()
    if epoch % 500 == 0 or epoch == 1:
        with torch.no_grad():
            _, _, codes = model(x)
        usage = [len(torch.unique(codes[:, l])) / args.codes for l in range(args.levels)]
        print(f"epoch {epoch} recon {recon.item():.4f} q {qloss.item():.4f} usage {[round(u, 2) for u in usage]} "
              f"unique ids {len(torch.unique(codes, dim=0))}/{len(x)} {time.time() - start:.0f}s", flush=True)
with torch.no_grad():
    _, _, codes = model(x)
codes = codes.cpu().numpy()
seen, extra = {}, np.zeros(len(codes), dtype=np.int64)
for i, row in enumerate(map(tuple, codes)):
    extra[i] = seen.get(row, 0)
    seen[row] = extra[i] + 1
full = np.concatenate([codes, extra[:, None]], 1)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.save(args.out, full)
print("saved", full.shape, "max collision idx", extra.max(), "items in shared prefix", int((extra > 0).sum()))
