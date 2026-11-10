import numpy as np


def rank_of_target(ranked_items, targets, histories=None, pad_index=None):
    ranked = np.asarray(ranked_items)
    targets = np.asarray(targets)
    ranks = np.full(len(targets), -1, dtype=np.int64)
    for row in range(len(targets)):
        items = ranked[row]
        if histories is not None:
            seen = set(np.asarray(histories[row]).tolist())
            seen.discard(pad_index)
            seen.discard(int(targets[row]))  # never filter the ground truth itself
            items = np.array([i for i in items if i not in seen], dtype=np.int64)
        hit = np.nonzero(items == targets[row])[0]
        if len(hit):
            ranks[row] = hit[0]
    return ranks


def summarize(ranks, ks=(5, 10)):
    ranks = np.asarray(ranks)
    out = {}
    for k in ks:
        hit = (ranks >= 0) & (ranks < k)
        out[f"recall@{k}"] = float(hit.mean())
        gain = np.where(hit, 1.0 / np.log2(np.where(hit, ranks, 0) + 2.0), 0.0)
        out[f"ndcg@{k}"] = float(gain.mean())
    return out
