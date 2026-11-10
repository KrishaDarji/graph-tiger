import argparse
import json
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model
from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--emb", required=True)
p.add_argument("--out", required=True)
p.add_argument("--graph", choices=["none", "scalar", "gate"], default="none")
p.add_argument("--graph-emb", default="")
p.add_argument("--token-graph", type=float, default=0.0)
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
p.add_argument("--seed", type=int, default=2024)
args = p.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
K = 256

items = Path(args.items).read_text().split()
idx = {a: i for i, a in enumerate(items)}
n_items = len(items)
seqs = [[idx[a] for a in s] for s in build_sequences(load_interactions(args.reviews)).values()]

x = np.load(args.emb).astype(np.float32)
if args.pca:
    pca = faiss.PCAMatrix(x.shape[1], args.pca)
    pca.train(x)
    x = pca.apply(x)
if args.token_graph:
    g = np.load(args.graph_emb).astype(np.float32)
    x = np.concatenate([x / np.linalg.norm(x, axis=1, keepdims=True),
                        args.token_graph * g / np.linalg.norm(g, axis=1, keepdims=True)], 1)
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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.dim
        self.wte = nn.Embedding(args.digits * K + 1, d)
        self.gpt = GPT2Model(GPT2Config(vocab_size=1, n_positions=args.max_len, n_embd=d, n_layer=2, n_head=4,
                                        n_inner=1024, activation_function="gelu_new", resid_pdrop=0.0,
                                        embd_pdrop=0.5, attn_pdrop=0.5, layer_norm_epsilon=1e-12,
                                        initializer_range=0.02))
        self.heads = nn.ModuleList([nn.Linear(d, d) for _ in range(args.digits)])
        for h in self.heads:
            nn.init.zeros_(h.weight)
        if graph is not None:
            self.proj = nn.Linear(graph.shape[1], d)
            if args.graph == "scalar":
                self.alpha = nn.Parameter(torch.zeros(1))
            else:
                self.gate = nn.Linear(d + graph.shape[1], d)

    def item_emb(self, item_ids):
        e = self.wte(tokens[item_ids]).mean(-2)
        if graph is None:
            return e
        g = self.proj(graph[item_ids])
        if args.graph == "scalar":
            return e + self.alpha * g
        return e + torch.sigmoid(self.gate(torch.cat([e, graph[item_ids]], -1))) * g

    def states(self, item_ids, mask):
        h = self.gpt(inputs_embeds=self.item_emb(item_ids), attention_mask=mask).last_hidden_state
        return torch.stack([h + F.silu(head(h)) for head in self.heads], -2)

    def logits(self, s):
        e = F.normalize(self.wte.weight[1:], dim=-1).view(args.digits, K, -1)
        return torch.einsum("...nd,nkd->...nk", F.normalize(s, dim=-1), e) / args.temp


model = Model().to(dev)
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
        mask = (torch.arange(ids.shape[1], device=dev)[None] < lens[:, None]).long()
        s = model.states(ids, mask)[torch.arange(len(rows)), lens - 1]
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
        s = model.states(ids, mask.long())[mask]
        target_tokens = tokens[labels[mask]] - torch.arange(args.digits, device=dev) * K - 1
        loss = F.cross_entropy(model.logits(s).reshape(-1, K), target_tokens.reshape(-1))
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
if args.graph == "scalar":
    result["alpha"] = model.alpha.item()
print(json.dumps(result, indent=1))
Path(args.out).mkdir(parents=True, exist_ok=True)
Path(f"{args.out}/result.json").write_text(json.dumps(result, indent=1))
Path(f"{args.out}/log.json").write_text(json.dumps(log))
