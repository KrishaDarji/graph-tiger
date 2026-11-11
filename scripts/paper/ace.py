import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--emb", required=True)
p.add_argument("--out", required=True)
p.add_argument("--graph", choices=["none", "gate"], default="none")
p.add_argument("--graph-emb", default="")
p.add_argument("--latents", type=int, default=1, help="tokens per item after merging; 1 = mean of code embeddings")
p.add_argument("--atm", action="store_true", help="attentive token merger instead of the mean")
p.add_argument("--intent", action="store_true")
p.add_argument("--isa", type=float, default=0.0)
p.add_argument("--isa-tau", type=float, default=0.07)
p.add_argument("--isa-beta", type=float, default=0.02)
p.add_argument("--pca", type=int, default=128)
p.add_argument("--digits", type=int, default=32)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--temp", type=float, default=0.03)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--warmup", type=int, default=10000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--max-len", type=int, default=50)
p.add_argument("--dim", type=int, default=448)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--seed", type=int, default=2024)
args = p.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
K = 256
K_LAT = args.latents if args.atm else 1
SLOTS = K_LAT + int(args.intent)

items = Path(args.items).read_text().split()
idx = {a: i for i, a in enumerate(items)}
n_items = len(items)
seqs = [[idx[a] for a in s] for s in build_sequences(load_interactions(args.reviews)).values()]

x = np.load(args.emb).astype(np.float32)
if args.pca:
    pca = faiss.PCAMatrix(x.shape[1], args.pca)
    pca.train(x)
    x = pca.apply(x)
opq = faiss.index_factory(x.shape[1], f"OPQ{args.digits},PQ{args.digits}x8")
opq.train(x)
opq.add(x)
codes = faiss.vector_to_array(faiss.downcast_index(opq.index).codes).reshape(n_items, args.digits).astype(np.int64)
tokens = torch.tensor(codes + np.arange(args.digits) * K + 1, device=dev)
tokens = torch.cat([tokens, torch.zeros(1, args.digits, dtype=torch.long, device=dev)])
print("unique item IDs:", len(np.unique(codes, axis=0)), "/", n_items, flush=True)

graph = None
if args.graph != "none":
    graph = F.normalize(torch.tensor(np.load(args.graph_emb), device=dev), dim=1)
    graph = torch.cat([graph, torch.zeros(1, graph.shape[1], device=dev)])


def pad(rows, length):
    out = torch.zeros(len(rows), length, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


train = [s[:-2][-(args.max_len + 1):] for s in seqs if len(s) - 2 >= 2]
val_hist = [s[:-2][-args.max_len:] for s in seqs]
val_target = torch.tensor([s[-2] for s in seqs], device=dev)
test_hist = [s[:-1][-args.max_len:] for s in seqs]
test_target = torch.tensor([s[-1] for s in seqs], device=dev)
freq = Counter(i for s in seqs for i in s[:-2])
log_pop = torch.log(torch.tensor([freq.get(i, 0) + 1 for i in range(n_items)] + [1], dtype=torch.float, device=dev))
log_pop = log_pop - torch.logsumexp(log_pop, 0)


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=1e-12), nn.LayerNorm(d, eps=1e-12)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 1024), nn.GELU(approximate="tanh"), nn.Linear(1024, d))
        self.heads, self.drop = heads, drop

    def forward(self, h, mask):
        b, t, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        h = h + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return h + self.mlp(self.ln2(h))


class Merger(nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k, self.d = k, d
        self.pos = nn.Parameter(torch.zeros(m, d))
        self.query = nn.Linear(d, k * d)
        self.attn = nn.MultiheadAttention(d, 4, dropout=0.0, batch_first=True)
        self.ln = nn.LayerNorm(d, eps=1e-12)
        self.mlp = nn.Sequential(nn.Linear(d, 1024), nn.GELU(approximate="tanh"), nn.Linear(1024, d))

    def forward(self, e, s):
        q = self.query(s).view(-1, self.k, self.d)
        z, _ = self.attn(q, e + self.pos, e + self.pos, need_weights=False)
        z = self.ln(q + z)
        return z + self.mlp(z)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.dim
        self.wte = nn.Embedding(args.digits * K + 1, d)
        self.summary = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        if args.atm:
            self.merger = Merger(d, args.digits, K_LAT)
        self.step_pos = nn.Embedding(args.max_len, d)
        self.slot_pos = nn.Embedding(SLOTS, d)
        self.drop = nn.Dropout(0.5)
        self.blocks = nn.ModuleList([Block(d, args.heads, 0.5) for _ in range(args.layers)])
        self.ln_f = nn.LayerNorm(d, eps=1e-12)
        self.heads = nn.ModuleList([nn.Linear(d, d) for _ in range(args.digits)])
        if graph is not None:
            self.proj = nn.Linear(graph.shape[1], d)
            self.gate = nn.Linear(d + graph.shape[1], d)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        for h in self.heads:
            nn.init.zeros_(h.weight)

    def item_parts(self, ids):
        e = self.wte(tokens[ids])
        mean = e.mean(-2)
        s = self.summary(mean)
        if graph is not None:
            g = graph[ids]
            gg = torch.sigmoid(self.gate(torch.cat([mean, g], -1))) * self.proj(g)
        else:
            gg = 0.0
        return e, mean, s, gg

    def sequence(self, ids, lens):
        b, L = ids.shape
        e, mean, s, gg = self.item_parts(ids)
        if args.atm:
            z = self.merger(e.reshape(b * L, args.digits, -1), s.reshape(b * L, -1)).view(b, L, K_LAT, -1)
        else:
            z = mean.unsqueeze(2)
        if graph is not None:
            z = z + gg.unsqueeze(2)
        if args.intent:
            z = torch.cat([z, (s + gg).unsqueeze(2)], 2)
        step = torch.arange(L, device=ids.device)
        z = z + self.step_pos(step)[None, :, None] + self.slot_pos.weight[None, None]
        h = self.drop(z.reshape(b, L * SLOTS, -1))
        t = L * SLOTS
        st = step.repeat_interleave(SLOTS)
        slot = torch.arange(SLOTS, device=ids.device).repeat(L)
        allow = (st[None] < st[:, None]) | ((st[None] == st[:, None]) & (slot[None] < K_LAT)) | torch.eye(t, dtype=torch.bool, device=ids.device)
        valid = (st[None, :] < lens[:, None])
        mask = (allow[None] & valid[:, None, :]) | torch.eye(t, dtype=torch.bool, device=ids.device)[None]
        mask = mask[:, None]
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_f(h).view(b, L, SLOTS, -1)
        return h[:, :, -1], s

    def states(self, ids, lens):
        h, s = self.sequence(ids, lens)
        return torch.stack([h + F.silu(head(h)) for head in self.heads], -2), h, s

    def logits(self, s):
        e = F.normalize(self.wte.weight[1:], dim=-1).view(args.digits, K, -1)
        return torch.einsum("...nd,nkd->...nk", F.normalize(s, dim=-1), e) / args.temp


model = Model().to(dev)
print("params", sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
total = args.epochs * math.ceil(len(train) / args.batch)
sched = get_cosine_schedule_with_warmup(opt, args.warmup, total)


@torch.no_grad()
def evaluate(hists, targets, ks=(5, 10)):
    model.eval()
    ranks = []
    for b in range(0, len(hists), 512):
        rows = hists[b:b + 512]
        ids = pad(rows, max(map(len, rows))).to(dev)
        lens = torch.tensor([len(r) for r in rows], device=dev)
        st, _, _ = model.states(ids, lens)
        s = st[torch.arange(len(rows)), lens - 1]
        logp = F.log_softmax(model.logits(s).float(), -1)
        scores = sum(logp[:, d, :][:, tokens[:n_items, d] - d * K - 1] for d in range(args.digits))
        t = targets[b:b + 512]
        ranks.append((scores > scores.gather(1, t[:, None])).sum(1))
    ranks = torch.cat(ranks)
    out = {}
    for k in ks:
        hit = ranks < k
        out[f"recall@{k}"] = hit.float().mean().item()
        out[f"ndcg@{k}"] = (hit.float() / torch.log2(ranks.float() + 2)).mean().item()
    return out


best, bad, best_state = -1, 0, None
log = []
start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    order = np.random.permutation(len(train))
    for b in range(0, len(order), args.batch):
        rows = [train[i] for i in order[b:b + args.batch]]
        ids = pad([r[:-1] for r in rows], max(len(r) for r in rows) - 1).to(dev)
        labels = pad([r[1:] for r in rows], ids.shape[1]).to(dev)
        lens = torch.tensor([len(r) - 1 for r in rows], device=dev)
        mask = torch.arange(ids.shape[1], device=dev)[None] < lens[:, None]
        st, anchor, _ = model.states(ids, lens)
        target_items = labels[mask]
        target_tokens = tokens[target_items] - torch.arange(args.digits, device=dev) * K - 1
        loss = F.cross_entropy(model.logits(st[mask]).reshape(-1, K), target_tokens.reshape(-1))
        if args.isa:
            n = min(2048, len(target_items))
            pick = torch.randperm(len(target_items), device=dev)[:n]
            h = F.normalize(anchor[mask][pick], dim=-1)
            tgt = target_items[pick]
            s = F.normalize(model.summary(model.wte(tokens[tgt]).mean(-2)), dim=-1)
            lg = h @ s.T / args.isa_tau - args.isa_beta * log_pop[tgt][None]
            lg = lg.masked_fill((tgt[:, None] == tgt[None]) & ~torch.eye(n, dtype=torch.bool, device=dev), float("-inf"))
            loss = loss + args.isa * F.cross_entropy(lg, torch.arange(n, device=dev))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    v = evaluate(val_hist, val_target)
    log.append({"epoch": epoch, "loss": loss.item(), **{f"val_{k}": x for k, x in v.items()}})
    print(epoch, round(loss.item(), 4), {k: round(x, 4) for k, x in v.items()}, f"{time.time() - start:.0f}s", flush=True)
    if v["ndcg@10"] > best:
        best, bad = v["ndcg@10"], 0
        best_state = {k: t.clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            break
model.load_state_dict(best_state)
result = {"best_val": best, "test": evaluate(test_hist, test_target), "val": evaluate(val_hist, val_target),
          "args": vars(args)}
print(json.dumps(result, indent=1))
Path(args.out).mkdir(parents=True, exist_ok=True)
Path(f"{args.out}/result.json").write_text(json.dumps(result, indent=1))
Path(f"{args.out}/log.json").write_text(json.dumps(log))
