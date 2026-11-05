"""Full-ranking evaluation over all items, scored in user chunks.

Metrics come from src.evaluation.evaluate_rankings, the same code used by the
popularity and Transformer baselines. A target inside the mask cannot be
recommended and counts as a miss, as in the popularity baseline.
"""
import numpy as np
import torch

from src.evaluation import evaluate_rankings


@torch.no_grad()
def top_k_items(user_emb, item_emb, users, mask=None, k=10, chunk_size=2048):
    """Return a (len(users), k) array of top items by dot product."""
    item_t = item_emb.T
    tops = []
    for start in range(0, len(users), chunk_size):
        chunk = users[start:start + chunk_size]
        scores = user_emb[torch.as_tensor(chunk, device=user_emb.device)] @ item_t
        if mask is not None:
            seen = mask[chunk].tocoo()
            scores[
                torch.as_tensor(seen.row.astype(np.int64), device=scores.device),
                torch.as_tensor(seen.col.astype(np.int64), device=scores.device),
            ] = -torch.inf
        tops.append(torch.topk(scores, k=k, dim=1).indices.cpu().numpy())
    return np.concatenate(tops)


def score_rankings(rankings, targets, ks=(5, 10)):
    return evaluate_rankings(rankings.tolist(), targets.tolist(), ks=ks)


def evaluate_split(user_emb, item_emb, pairs, mask=None, ks=(5, 10), chunk_size=2048):
    rankings = top_k_items(user_emb, item_emb, pairs[:, 0], mask, max(ks), chunk_size)
    return score_rankings(rankings, pairs[:, 1], ks)


def popularity_embeddings(item_train_degree, n_users, device):
    """Express popularity as dot-product scores so it shares the eval path."""
    items = torch.as_tensor(item_train_degree, dtype=torch.float32, device=device)[:, None]
    return torch.ones(n_users, 1, device=device), items
