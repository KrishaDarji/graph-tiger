from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .data import pad
from .evaluator import Evaluator


@dataclass(frozen=True)
class Blend:
    beta: float = 0.0   # weight of the dense head's log-probability
    gamma: float = 0.0  # weight of last-item graph similarity (graph used after generation)
    ease: float = 0.0   # weight of the z-scored EASE score
    cand: int = 0       # re-score only the top-`cand` items of the dense head
    slots: int = 0      # reserved top-10 positions for items without training interactions
    place: str = "tail"


def ease_matrix(train, n_items, lam, dev):
    x = torch.zeros(len(train), n_items, device=dev)
    for u, t in enumerate(train):
        x[u, t] = 1.0
    inv = torch.linalg.inv(x.T @ x + lam * torch.eye(n_items, device=dev))
    b = -inv / torch.diag(inv)[None, :]
    b.fill_diagonal_(0.0)
    return b


def make_context(content, split, cold_mask, cold_eval, dense, tau, ease=None):
    return SimpleNamespace(content=content, count=split.count, cold_mask=cold_mask, cold_eval=cold_eval, dense=dense, tau=tau, ease=ease)


@torch.no_grad()
def score_batches(model, ctx, hists, targets, blends, bs=512):
    # yields one [B, N] score matrix per blend for each batch of users
    model.eval()
    n, dev = model.n_items, model.wte.weight.device
    targets = torch.as_tensor(targets, device=dev)
    dense_items = F.normalize(model.item_emb(torch.arange(n, device=dev)), dim=-1) if ctx.dense else None
    gn = F.normalize(model.graph[:n], dim=-1) if model.graph is not None else None
    for b in range(0, len(hists), bs):
        rows = hists[b:b + bs]
        L = max(map(len, rows))
        ids = pad(rows, L).to(dev)
        lens = torch.tensor([len(r) for r in rows], device=dev)
        mask = (torch.arange(L, device=dev)[None] < lens[:, None]).long()
        last = torch.arange(len(rows), device=dev), lens - 1
        h = model.hidden(ids, mask)[last]
        logp = F.log_softmax(model.logits(model.digit_states(h)).float(), -1)
        gen = sum(logp[:, d, :][:, model.codes[:, d]] for d in range(model.digits))
        if ctx.dense:
            dlg = (F.normalize(h.float(), dim=-1) @ dense_items.float().T) / ctx.tau
            dense = F.log_softmax(dlg, -1)
        if any(c.slots for c in blends):
            last3 = torch.stack([torch.cat([torch.tensor(r[-3:]), torch.full((3 - len(r[-3:]),), n)]) for r in rows]).to(dev)
            content_q = F.normalize(ctx.content[last3].sum(1), dim=-1)
        if any(c.ease for c in blends):
            seen = torch.zeros(len(rows), n, device=dev).scatter_add_(1, ids, mask.float()).clamp_(max=1.0)
            z = seen @ ctx.ease
            z = (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-6)
        outs = []
        for c in blends:
            sc = gen + c.beta * dense if ctx.dense else gen
            if c.gamma:
                sc = sc + c.gamma * (gn[ids[last]] @ gn.T)
            if c.ease:
                sc = sc + c.ease * z
            if c.cand:
                sc = sc.masked_fill(dlg < dlg.topk(c.cand, dim=1).values[:, -1:], float("-inf"))
            if c.slots:  # best unseen items by content + graph nearest neighbour, placed at the head or after the model's top 10-slots
                knn = (content_q @ ctx.content[:n].T + gn[ids[last]] @ gn.T).masked_fill(~ctx.cold_mask[None, :n], float("-inf"))
                top = knn.topk(c.slots, dim=1)
                if c.place == "head":
                    sc = sc.scatter(1, top.indices, 1e4 + top.values)
                else:
                    keep = sc.scatter(1, top.indices, float("-inf")).topk(10 - c.slots, dim=1)
                    sc = sc.scatter(1, top.indices, 5e3 + top.values).scatter(1, keep.indices, 1e4 + keep.values)
            outs.append(sc)
        yield outs, targets[b:b + bs], pad(rows, L, -1).to(dev)


def evaluate(model, ctx, hists, targets, blends):
    evs = [Evaluator(model.n_items, ctx.count, cold_items=ctx.cold_eval or None, item_vecs=ctx.content[:model.n_items]) for _ in blends]
    for outs, t, hist in score_batches(model, ctx, hists, targets, blends):
        for ev, s in zip(evs, outs):
            ev.update(s, t, hist)
    return evs
