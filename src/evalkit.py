"""Full-ranking evaluation for sequential recommenders, one place for every number we report.

Protocol (leave-one-out): one target item per user, the candidates are ALL items, ties count against the model.
Two views are always computed from the same scores:
  raw       no item is removed from the ranking (what RPG's and TIGER's papers describe)
  filtered  the user's history items are removed (the target itself is never removed)
"""
import numpy as np
import torch

KS = (1, 5, 10, 20, 50)


def pessimistic_rank(scores, target):
    """0-based rank of the target; every other item with an equal or higher score counts as ahead."""
    t = scores.gather(1, target[:, None])
    return (scores >= t).sum(1) - 1


def hit_gain(ranks, k):
    hit = ranks < k
    gain = torch.where(hit, 1.0 / torch.log2(ranks.float() + 2.0), torch.zeros(()))
    return hit.float(), gain


def summarize_ranks(ranks, ks=KS):
    ranks = torch.as_tensor(ranks)
    out = {}
    for k in ks:
        hit, gain = hit_gain(ranks, k)
        out[f"recall@{k}"] = hit.mean().item()
        if k > 1:
            out[f"ndcg@{k}"] = gain.mean().item()
    out["mrr@10"] = torch.where(ranks < 10, 1.0 / (ranks.float() + 1), torch.zeros(())).mean().item()
    out["mrr"] = (1.0 / (ranks.float() + 1)).mean().item()
    return out


class Evaluator:
    """Feed it score matrices batch by batch; read summary() at the end."""

    def __init__(self, n_items, train_count, topn=10, cold_items=None, item_vecs=None):
        self.n, self.topn = n_items, topn
        self.count = np.asarray(train_count, dtype=np.float64)
        self.raw, self.filt, self.targets, self.top, self.hist_len, self.ild = [], [], [], [], [], []
        self.item_vecs = item_vecs  # [N,d] L2-normalised, for intra-list diversity
        self.cold = None
        if cold_items is not None and len(cold_items):
            self.cold = torch.as_tensor(np.asarray(sorted(cold_items)), dtype=torch.long)
            self.cold_pos = torch.full((n_items,), -1, dtype=torch.long)
            self.cold_pos[self.cold] = torch.arange(len(self.cold))
            self.cold_ranks, self.warm_ranks = [], []

    @torch.no_grad()
    def update(self, scores, target, hist):
        """scores [B,N] float; target [B]; hist [B,L] item ids, padded with -1."""
        scores = scores.float()
        self.raw.append(pessimistic_rank(scores, target).cpu())
        seen = torch.zeros(scores.shape[0], self.n + 1, dtype=torch.bool, device=scores.device)
        seen.scatter_(1, torch.where(hist < 0, self.n, hist), True)
        seen = seen[:, :self.n]
        seen[torch.arange(len(target), device=scores.device), target] = False
        self.filt.append(pessimistic_rank(scores.masked_fill(seen, float("-inf")), target).cpu())
        self.targets.append(target.cpu())
        top = scores.topk(self.topn, dim=1).indices
        self.top.append(top.cpu())
        self.hist_len.append((hist >= 0).sum(1).cpu())
        if self.item_vecs is not None:
            v = self.item_vecs[top]
            sim = v @ v.transpose(1, 2)
            k = self.topn
            self.ild.append((1 - (sim.sum((1, 2)) - k) / (k * (k - 1))).cpu())
        if self.cold is not None:
            dev = scores.device
            cold, pos = self.cold.to(dev), self.cold_pos.to(dev)
            is_cold = pos[target] >= 0
            if is_cold.any():  # rank of a cold target among the cold items only
                sub = scores[is_cold][:, cold]
                self.cold_ranks.append(pessimistic_rank(sub, pos[target[is_cold]]).cpu())
            if (~is_cold).any():  # rank of a warm target among the warm items only
                w = scores[~is_cold].clone()
                w[:, cold] = float("-inf")
                self.warm_ranks.append(pessimistic_rank(w, target[~is_cold]).cpu())

    def ranks(self):
        return torch.cat(self.raw).numpy(), torch.cat(self.filt).numpy()

    def summary(self):
        raw, filt = (torch.cat(self.raw), torch.cat(self.filt))
        target = torch.cat(self.targets).numpy()
        top = torch.cat(self.top).numpy()
        out = {"raw": summarize_ranks(raw), "filtered": summarize_ranks(filt)}
        shown = np.bincount(top.ravel(), minlength=self.n).astype(np.float64)
        p = shown / shown.sum()
        order = np.sort(shown)
        gini = float((2 * np.arange(1, self.n + 1) - self.n - 1) @ order / (self.n * order.sum()))
        out["beyond"] = {
            "coverage@10": float((shown > 0).mean()),
            "gini@10": gini,
            "entropy@10": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
            "avg_log_popularity@10": float(np.log1p(self.count[top]).mean()),
            "avg_log_popularity_of_targets": float(np.log1p(self.count[target]).mean()),
        }
        hl = torch.cat(self.hist_len).numpy()
        out["segments"] = {}
        for name, lo, hi in [("len_le5", 0, 5), ("len_6_10", 6, 10), ("len_11_20", 11, 20), ("len_gt20", 21, 10**6)]:
            sel = (hl >= lo) & (hl <= hi)
            out["segments"][f"recall@10_{name}"] = float((raw.numpy()[sel] < 10).mean()) if sel.any() else float("nan")
            out["segments"][f"n_{name}"] = int(sel.sum())
        if self.ild:
            out["beyond"]["ild@10"] = float(torch.cat(self.ild).mean())
        out["beyond"]["novelty@10"] = float(-np.log2((self.count[top] + 1) / (self.count.sum() + self.n)).mean())
        if self.cold is not None:
            cr = torch.cat(self.cold_ranks) if self.cold_ranks else torch.zeros(0, dtype=torch.long)
            wr = torch.cat(self.warm_ranks) if self.warm_ranks else torch.zeros(0, dtype=torch.long)
            out["cold_pool"] = {"n_users": int(len(cr)), "pool_size": int(len(self.cold)), **summarize_ranks(cr, (1, 5, 10, 20)),
                                "chance_recall@10": 10 / len(self.cold)}
            out["warm_pool"] = {"n_users": int(len(wr)), **summarize_ranks(wr, (5, 10))}
        quart = np.digitize(self.count, np.quantile(self.count, [0.25, 0.5, 0.75]))  # 0 = least popular quarter of items
        for q, name in enumerate(["tail", "low_mid", "high_mid", "head"]):
            sel = quart[target] == q
            r = raw.numpy()[sel]
            out["beyond"][f"recall@10_{name}_targets"] = float((r < 10).mean()) if sel.any() else float("nan")
            out["beyond"][f"n_{name}_targets"] = int(sel.sum())
        return out


def paired_test(ranks_a, ranks_b, k=10, metric="ndcg", n_boot=2000, seed=0):
    """Per-user paired comparison of two models on the same users: mean difference, bootstrap 95% CI, wins/losses."""
    a, b = torch.as_tensor(ranks_a), torch.as_tensor(ranks_b)
    ha, ga = hit_gain(a, k)
    hb, gb = hit_gain(b, k)
    d = (ga - gb) if metric == "ndcg" else (ha - hb)
    d = d.numpy()
    g = np.random.default_rng(seed)
    boots = np.array([d[g.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    base = (gb if metric == "ndcg" else hb).mean().item()
    return {"mean_diff": float(d.mean()), "rel_diff": float(d.mean() / base) if base else float("nan"),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
            "users_better": int((d > 0).sum()), "users_worse": int((d < 0).sum())}
