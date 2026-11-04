"""Ranking metrics for next-item recommendation."""
import math


def recall_at_k(predictions, target, k):
    return float(target in predictions[:k])


def ndcg_at_k(predictions, target, k):
    try:
        rank = predictions[:k].index(target) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_rankings(rankings, targets, ks=(5, 10)):
    results = {}
    for k in ks:
        results[f"recall@{k}"] = sum(
            recall_at_k(p, t, k) for p, t in zip(rankings, targets)
        ) / max(len(targets), 1)
        results[f"ndcg@{k}"] = sum(
            ndcg_at_k(p, t, k) for p, t in zip(rankings, targets)
        ) / max(len(targets), 1)
    return results

