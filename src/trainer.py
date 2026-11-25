import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.optimization import get_cosine_schedule_with_warmup

from .data import pad
from .scoring import Blend, evaluate


def dense_loss(model, h, tgt, cand, tau):
    # item-level softmax: cosine(sequence state, item vector) against sampled negatives, shared item embedding
    pos = F.normalize(model.item_emb(tgt), dim=-1)
    neg = F.normalize(model.item_emb(cand), dim=-1)
    hn = F.normalize(h, dim=-1)
    lg = torch.cat([(hn * pos).sum(-1, keepdim=True), hn @ neg.T], 1) / tau
    lg[:, 1:] = lg[:, 1:].masked_fill(cand[None] == tgt[:, None], float("-inf"))
    return F.cross_entropy(lg, torch.zeros(len(h), dtype=torch.long, device=h.device))


def train(model, ctx, split, pool, a):
    dev = model.wte.weight.device
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}], lr=a.lr)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, a.epochs * math.ceil(len(split.train) / a.batch))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.ema else None
    val_blend = [Blend(beta=a.dense_beta if a.dense_w else 0.0)]
    train_rows, last_only = split.train, np.array(split.last_only)
    best, bad, best_state, last_state, log = -1, 0, None, None, []
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_rows))
        for b in range(0, len(order), a.batch):
            rows = [train_rows[i] for i in order[b:b + a.batch]]
            ids = pad([r[:-1] for r in rows], max(len(r) for r in rows) - 1).to(dev)
            labels = pad([r[1:] for r in rows], ids.shape[1]).to(dev)
            lens = torch.tensor([len(r) - 1 for r in rows], device=dev)
            pos = torch.arange(ids.shape[1], device=dev)[None]
            mask = pos < lens[:, None]
            only_last = torch.tensor(last_only[order[b:b + a.batch]], device=dev)[:, None]
            loss_mask = mask & (~only_last | (pos == lens[:, None] - 1))
            h = model.hidden(ids, mask.long())[loss_mask]
            tgt = labels[loss_mask]
            digit_logits = model.logits(model.digit_states(h))
            loss = F.cross_entropy(digit_logits.reshape(-1, digit_logits.shape[-1]), model.codes[tgt].reshape(-1))
            if a.dense_w:
                cand = pool[torch.randint(0, len(pool), (a.dense_neg,), device=dev)]
                loss = loss + a.dense_w * dense_loss(model, h, tgt, cand, a.dense_tau)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(a.ema).add_(v, alpha=1 - a.ema)
                        else:
                            ema[k].copy_(v)
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if ema is not None else None
        if ema is not None:
            model.load_state_dict(ema)
        v = evaluate(model, ctx, split.val_hist, split.val_target, val_blend)[0].summary()["raw"]
        last_state = {k: t.clone() for k, t in model.state_dict().items()}
        if raw_state is not None:
            model.load_state_dict(raw_state)
        log.append({"epoch": epoch, "loss": loss.item(), **{f"val_{k}": x for k, x in v.items() if "@" in k}})
        print(epoch, round(loss.item(), 4), {k: round(x, 4) for k, x in v.items() if k in ("recall@5", "recall@10", "ndcg@10")},
              f"{time.time() - start:.0f}s", flush=True)
        if a.stop_after and epoch >= a.stop_after:
            break
        if v["ndcg@10"] > best:
            best, bad, best_state = v["ndcg@10"], 0, last_state
        else:
            bad += 1
            if bad >= a.patience:
                break
    chosen = last_state if (a.no_select or a.stop_after) else best_state
    return chosen, best, log, time.time() - start
