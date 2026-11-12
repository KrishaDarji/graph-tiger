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
from src.evalkit import Evaluator
from src.tiger.data import build_sequences, load_interactions

p = argparse.ArgumentParser()
p.add_argument("--reviews", required=True)
p.add_argument("--items", required=True)
p.add_argument("--emb", required=True)
p.add_argument("--out", required=True)
p.add_argument("--graph", choices=["none", "gate"], default="none")
p.add_argument("--graph-emb", default="")
p.add_argument("--graph-tgt", default="", help="second graph file used for the target side (dense head); directional memory")
p.add_argument("--bridge", type=float, default=0.0, help="weight of the text-to-graph-vector bridge loss; the bridge also supplies vectors for items without one")
p.add_argument("--token-graph", type=float, default=0.0, help="weight of the graph vector inside the vectors that are quantized into item tokens")
p.add_argument("--eval-graph", default="", help="graph file swapped in after training (refresh test)")
p.add_argument("--align", action="store_true", help="rotate the refreshed graph vectors onto the basis of the training-time ones (orthogonal Procrustes)")
p.add_argument("--graph-train", action="store_true", help="fine-tune the graph vectors with the model")
p.add_argument("--graph-drop", type=float, default=0.0, help="zero the graph vector of a random item occurrence with this probability")
p.add_argument("--gammas", default="0", help="grid of weights for last-item graph similarity added to the score (picked on validation)")
p.add_argument("--latency", action="store_true", help="also time single-user requests (needs an otherwise idle GPU)")
p.add_argument("--dense-batch", action="store_true", help="dense-loss negatives are the other targets in the batch")
p.add_argument("--dense-pop", type=float, default=0.0, help="weight of the log-frequency correction of sampled negatives (1 = logQ correction)")
p.add_argument("--coldpath", action="store_true", help="items without training interactions are scored by the dense head alone, plus the offset, instead of generative + dense")
p.add_argument("--cals", default="0", help="grid of score offsets added to items without any training interaction (exposure calibration, picked on validation)")
p.add_argument("--cgammas", default="0", help="grid of weights for content similarity between the last 3 history items and each candidate (works for unseen items)")
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
p.add_argument("--ffn", type=int, default=1024)
p.add_argument("--drop", type=float, default=0.5)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--ema", type=float, default=0.0)
p.add_argument("--dense-w", type=float, default=0.0, help="weight of the item-level softmax loss (0 = off)")
p.add_argument("--dense-tau", type=float, default=0.07)
p.add_argument("--dense-neg", type=int, default=4096)
p.add_argument("--dense-beta", type=float, default=8.0, help="weight of the dense log-prob in the score used for early stopping")
p.add_argument("--holdout", default="", help="npy of item ids that never appear in training (cold-start test)")
p.add_argument("--seed", type=int, default=2024)
p.add_argument("--save", default="")
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
content = torch.nn.functional.normalize(torch.tensor(x, device=dev), dim=1)
content = torch.cat([content, torch.zeros(1, content.shape[1], device=dev)])
xq = x
if args.token_graph:
    g0 = np.load(args.graph_emb).astype(np.float32)
    g0 = g0 / np.maximum(np.linalg.norm(g0, axis=1, keepdims=True), 1e-6)
    xq = np.concatenate([x / np.linalg.norm(x, axis=1, keepdims=True), args.token_graph * g0], 1)
opq = faiss.index_factory(xq.shape[1], f"OPQ{args.digits},PQ{args.digits}x8")
opq.train(xq)
opq.add(xq)
codes = faiss.vector_to_array(faiss.downcast_index(opq.index).codes).reshape(n_items, args.digits).astype(np.int64)
tokens = torch.tensor(codes + np.arange(args.digits) * K + 1, device=dev)
tokens = torch.cat([tokens, torch.zeros(1, args.digits, dtype=torch.long, device=dev)])
print("unique item IDs:", len(np.unique(codes, axis=0)), "/", n_items, flush=True)

def load_graph(path):
    g = torch.tensor(np.load(path), device=dev).float()
    nrm = g.norm(dim=1, keepdim=True)
    g = torch.where(nrm > 1e-6, g / nrm.clamp_min(1e-12), torch.zeros_like(g))  # items without interactions stay zero
    return torch.cat([g, torch.zeros(1, g.shape[1], device=dev)])


graph = graph_tgt = None
if args.graph != "none":
    graph = load_graph(args.graph_emb)
    graph_tgt = load_graph(args.graph_tgt) if args.graph_tgt else None


def pad(rows, length, fill=0):
    out = torch.full((len(rows), length), fill, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


holdout = set(np.load(args.holdout).tolist()) if args.holdout else set()
train = [[i for i in s[:-2] if i not in holdout][-(args.max_len + 1):] for s in seqs]
train = [t for t in train if len(t) >= 2]
pool = torch.tensor([i for i in range(n_items) if i not in holdout], device=dev)
seen_train = np.zeros(n_items + 1, dtype=bool)
for t in train:
    seen_train[t] = True
cold_mask = torch.tensor(~seen_train, device=dev)
cold_mask[n_items] = False
val_hist = [s[:-2][-args.max_len:] for s in seqs]
val_target = torch.tensor([s[-2] for s in seqs], device=dev)
test_hist = [s[:-1][-args.max_len:] for s in seqs]
test_target = torch.tensor([s[-1] for s in seqs], device=dev)
count_t = None
train_count = np.bincount([i for s in seqs for i in s[:-2]], minlength=n_items)


count_t = torch.tensor(train_count, dtype=torch.float32, device=dev)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = args.dim
        self.wte = nn.Embedding(args.digits * K + 1, d)
        self.gpt = GPT2Model(GPT2Config(vocab_size=1, n_positions=args.max_len, n_embd=d, n_layer=args.layers,
                                        n_head=args.heads, n_inner=args.ffn, activation_function="gelu_new",
                                        resid_pdrop=0.0, embd_pdrop=args.drop, attn_pdrop=args.drop,
                                        layer_norm_epsilon=1e-12, initializer_range=0.02))
        self.heads = nn.ModuleList([nn.Linear(d, d) for _ in range(args.digits)])
        for h in self.heads:
            nn.init.zeros_(h.weight)
        if graph is not None:
            self.proj = nn.Linear(graph.shape[1], d)
            self.gate = nn.Linear(d + graph.shape[1], d)
            if args.graph_train:
                self.graph = nn.Parameter(graph.clone())
            else:
                self.register_buffer("graph", graph.clone())
            if graph_tgt is not None:
                self.register_buffer("graph_tgt", graph_tgt.clone())
            if args.bridge:
                self.bridge = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, graph.shape[1] * (2 if graph_tgt is not None else 1)))

    def bridged(self, e, tgt):
        out = self.bridge(e)
        gd = self.graph.shape[1]
        return F.normalize(out[..., gd:] if (tgt and graph_tgt is not None) else out[..., :gd], dim=-1)

    def item_emb(self, item_ids, tgt=False):
        e = self.wte(tokens[item_ids]).mean(-2)
        if graph is None:
            return e
        gv = (self.graph_tgt if (tgt and graph_tgt is not None) else self.graph)[item_ids]
        if args.bridge:
            use_pred = cold_mask[item_ids]
            if self.training and args.graph_drop:
                use_pred = use_pred | (torch.rand(item_ids.shape, device=gv.device) < args.graph_drop)
            gv = torch.where(use_pred[..., None], self.bridged(e, tgt), gv)
        elif self.training and args.graph_drop:
            gv = gv * (torch.rand(*gv.shape[:-1], 1, device=gv.device) >= args.graph_drop)
        return e + torch.sigmoid(self.gate(torch.cat([e, gv], -1))) * self.proj(gv)

    def hidden(self, item_ids, mask):
        return self.gpt(inputs_embeds=self.item_emb(item_ids), attention_mask=mask).last_hidden_state

    def digit_states(self, h):
        return torch.stack([h + F.silu(head(h)) for head in self.heads], -2)

    def logits(self, s):
        e = F.normalize(self.wte.weight[1:], dim=-1).view(args.digits, K, -1)
        return torch.einsum("...nd,nkd->...nk", F.normalize(s, dim=-1), e) / args.temp


model = Model().to(dev)
n_params = sum(p.numel() for p in model.parameters())
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
total = args.epochs * math.ceil(len(train) / args.batch)
sched = get_cosine_schedule_with_warmup(opt, args.warmup, total)
ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema else None


@torch.no_grad()
def score_batches(hists, targets, combos):
    """Yield (scores per (beta, gamma), target, padded history) for each batch of users."""
    model.eval()
    all_items = torch.arange(n_items, device=dev)
    dense_items = F.normalize(model.item_emb(all_items, tgt=True), dim=-1) if args.dense_w else None
    for b in range(0, len(hists), 512):
        rows = hists[b:b + 512]
        L = max(map(len, rows))
        ids = pad(rows, L).to(dev)
        lens = torch.tensor([len(r) for r in rows], device=dev)
        mask = (torch.arange(L, device=dev)[None] < lens[:, None]).long()
        h = model.hidden(ids, mask)[torch.arange(len(rows)), lens - 1]
        logp = F.log_softmax(model.logits(model.digit_states(h)).float(), -1)
        gen = sum(logp[:, d, :][:, tokens[:n_items, d] - d * K - 1] for d in range(args.digits))
        if dense_items is not None:
            dense = F.log_softmax((F.normalize(h.float(), dim=-1) @ dense_items.float().T) / args.dense_tau, -1)
        outs = []
        last3 = torch.stack([torch.cat([torch.tensor(r[-3:]), torch.full((3 - len(r[-3:]),), n_items)]) for r in rows]).to(dev)
        cv = F.normalize(content[last3].sum(1), dim=-1)
        for beta, gamma, cg, cal in combos:
            sc = gen + beta * dense if dense_items is not None else gen
            if gamma:
                gl = F.normalize(model.graph[ids[torch.arange(len(rows)), lens - 1]], dim=-1)
                sc = sc + gamma * (gl @ F.normalize(model.graph[:n_items], dim=-1).T)
            if cg:
                sc = sc + cg * (cv @ content[:n_items].T)
            if cal and args.coldpath:
                sc = torch.where(cold_mask[None, :n_items], dense + cal, sc)
            elif cal:
                sc = sc + cal * cold_mask[:n_items].float()
            outs.append(sc)
        yield outs, targets[b:b + 512], pad(rows, L, -1).to(dev)


def evaluate(hists, targets, combos=None):
    combos = combos if combos is not None else [(args.dense_beta if args.dense_w else 0.0, 0.0, 0.0, 0.0)]
    evs = [Evaluator(n_items, train_count, cold_items=holdout or None, item_vecs=content[:n_items]) for _ in combos]
    for outs, t, hist in score_batches(hists, targets, combos):
        for ev, s in zip(evs, outs):
            ev.update(s, t, hist)
    return combos, evs


def swap_in(state):
    cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return cur


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
        h = model.hidden(ids, mask.long())[mask]
        tgt = labels[mask]
        target_tokens = tokens[tgt] - torch.arange(args.digits, device=dev) * K - 1
        loss = F.cross_entropy(model.logits(model.digit_states(h)).reshape(-1, K), target_tokens.reshape(-1))
        if args.dense_w:
            cand = tgt.unique() if args.dense_batch else pool[torch.randint(0, len(pool), (args.dense_neg,), device=dev)]
            pos = F.normalize(model.item_emb(tgt, tgt=True), dim=-1)
            neg = F.normalize(model.item_emb(cand, tgt=True), dim=-1)
            hn = F.normalize(h, dim=-1)
            lg = torch.cat([(hn * pos).sum(-1, keepdim=True), hn @ neg.T], 1) / args.dense_tau
            if args.dense_pop:
                logq = torch.log(count_t + 1.0)
                lg = lg - args.dense_pop * torch.cat([logq[tgt][:, None], logq[cand][None].expand(len(h), -1)], 1)
            lg[:, 1:] = lg[:, 1:].masked_fill(cand[None] == tgt[:, None], float("-inf"))
            loss = loss + args.dense_w * F.cross_entropy(lg, torch.zeros(len(h), dtype=torch.long, device=dev))
        if args.bridge:
            uniq = tgt.unique()
            e_u = model.wte(tokens[uniq]).mean(-2)
            loss = loss + args.bridge * (1 - (model.bridged(e_u, False) * model.graph[uniq]).sum(-1)).mean()
            if graph_tgt is not None:
                loss = loss + args.bridge * (1 - (model.bridged(e_u, True) * model.graph_tgt[uniq]).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
                    else:
                        ema[k].copy_(v)
    cur = swap_in(ema) if ema is not None else None
    _, (ev,) = evaluate(val_hist, val_target)
    v = ev.summary()["raw"]
    if cur is not None:
        eval_state = {k: t.clone() for k, t in model.state_dict().items()}
        model.load_state_dict(cur)
    else:
        eval_state = None
    log.append({"epoch": epoch, "loss": loss.item(), **{f"val_{k}": x for k, x in v.items() if "@" in k}})
    print(epoch, round(loss.item(), 4), {k: round(x, 4) for k, x in v.items() if k in ("recall@5", "recall@10", "ndcg@10")},
          f"{time.time() - start:.0f}s", flush=True)
    if v["ndcg@10"] > best:
        best, bad = v["ndcg@10"], 0
        best_state = eval_state if eval_state is not None else {k: t.clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            break
train_seconds = time.time() - start
model.load_state_dict(best_state)

betas = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] if args.dense_w else [0.0]
gammas = [float(g) for g in args.gammas.split(",")] if graph is not None else [0.0]
cgammas = [float(c) for c in args.cgammas.split(",")]
cals = [float(c) for c in args.cals.split(",")]
combos = [(b, g, c, k) for b in betas for g in gammas for c in cgammas for k in cals]
_, val_evs = evaluate(val_hist, val_target, combos)
val_all = [e.summary() for e in val_evs]
pick = int(np.argmax([s["raw"]["ndcg@10"] for s in val_all]))
beta, gamma, cgamma, cal = combos[pick]
_, (tev,) = evaluate(test_hist, test_target, [(beta, gamma, cgamma, cal)])
test = tev.summary()
dense_only = None
if args.dense_w and holdout:  # score only with the content-bridged dense head (beta -> large)
    _, (dev_,) = evaluate(test_hist, test_target, [(1000.0, 0.0, 0.0, 0.0)])
    ds = dev_.summary()
    dense_only = {"raw": ds["raw"], "cold_pool": ds["cold_pool"], "warm_pool": ds["warm_pool"]}
refreshed = None
if args.eval_graph:  # same trained weights, graph vectors rebuilt from more recent interactions
    with torch.no_grad():
        new = load_graph(args.eval_graph)
        if args.align:
            old = model.graph
            both = (old.norm(dim=1) > 0) & (new.norm(dim=1) > 0)
            u, _, vh = torch.linalg.svd(new[both].T @ old[both])
            new = new @ (u @ vh)
        model.graph.copy_(new)
    _, (vev2,) = evaluate(val_hist, val_target, [(beta, gamma, cgamma, cal)])
    _, (tev2,) = evaluate(test_hist, test_target, [(beta, gamma, cgamma, cal)])
    refreshed = {"val": vev2.summary()["raw"], "test": tev2.summary()["raw"]}

torch.cuda.synchronize()
t0 = time.time()
for _ in score_batches(test_hist, test_target, [(beta, gamma, cgamma, cal)]):
    pass
torch.cuda.synchronize()
latency_ms = (time.time() - t0) / len(test_hist) * 1000

single = None
if args.latency:
    times = []
    for i in range(300):
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in score_batches(test_hist[i:i + 1], test_target[i:i + 1], [(beta, gamma, cgamma, cal)]):
            pass
        torch.cuda.synchronize()
        times.append((time.time() - t1) * 1000)
    times = np.array(times[20:])
    single = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)), "p99_ms": float(np.percentile(times, 99)),
              "includes_item_vector_rebuild": True, "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
result = {"best_val": best, "beta": beta, "gamma": gamma, "cgamma": cgamma, "cal": cal, "val": val_all[pick]["raw"], "val_filtered": val_all[pick]["filtered"],
          "test": test["raw"], "test_filtered": test["filtered"], "test_beyond": test["beyond"], "test_segments": test["segments"],
          "test_cold_pool": test.get("cold_pool"), "test_warm_pool": test.get("warm_pool"),
          "val_by_combo": {str(c): s["raw"]["ndcg@10"] for c, s in zip(combos, val_all)},
          "params": n_params, "train_seconds": train_seconds, "score_ms_per_user": latency_ms,
          "epochs_run": len(log), "args": vars(args)}
if single:
    result["single_user_latency"] = single
if refreshed:
    result["refreshed"] = refreshed
if dense_only:
    result["dense_only"] = dense_only
print(json.dumps({k: result[k] for k in ("beta", "test", "test_filtered", "params", "train_seconds", "score_ms_per_user")}, indent=1))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps(result, indent=1))
(out / "log.json").write_text(json.dumps(log))
raw, filt = tev.ranks()
if holdout:
    from src.evalkit import summarize_ranks
    cold = np.isin(test_target.cpu().numpy(), list(holdout))
    result["cold"] = {"n_users": int(cold.sum()), "metrics": summarize_ranks(raw[cold]), "warm_metrics": summarize_ranks(raw[~cold])}
    (out / "result.json").write_text(json.dumps(result, indent=1))
np.save(out / "test_ranks_raw.npy", raw)
np.save(out / "test_ranks_filtered.npy", filt)
if args.save:
    torch.save(best_state, args.save)
