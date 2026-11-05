# TIGER 5-core reproduction and collision pilot

This run checks the code. It uses the 5-core Amazon Beauty data (22,363 users, 12,101 items,
198,502 interactions) rather than the project's split, and it does not use LightGCN.

Code: `third_party/xiaolongtao_tiger`, with local changes listed in `UPSTREAM.md`.
Run: `scripts/run_tiger_5core.sh`, one seed. RQ-VAE 3,000 epochs. T5 selected on validation
NDCG@20 at epoch 80, early stopped at epoch 90. About 83 minutes on one NVIDIA L40S, sharing
the GPU with the collision pilot for part of the run.

## Test metrics

| Metric | This run | Upstream repo (reported) | TIGER paper |
|---|---:|---:|---:|
| Recall@5 | 0.0351 | 0.0392 | 0.0454 |
| Recall@10 | 0.0580 | 0.0594 | 0.0648 |
| NDCG@5 | 0.0227 | 0.0257 | 0.0321 |
| NDCG@10 | 0.0301 | 0.0321 | 0.0384 |

Scored with the upstream metric code, which averages per batch. Semantic IDs: 0.49% prefix
collisions, largest group 3, 49 of 256 first-level codes used.

## Collision pilot on the project's 163,257 items

These runs train only a short RQ-VAE, not T5, so they give no recommendation results.

| Embeddings | RQ-VAE epochs | Collisions after RQ-VAE | After retry step | Largest prefix group | First-level codes used |
|---|---:|---:|---:|---:|---:|
| Content (768-D) | 225 | 9.8% | 1.05% | 11 | 254 |
| Projected content (64-D) | 20 | 13.4% | 3.10% | 49 | 137 |
| Fused, alpha = 1 (64-D) | 20 | 3.4% | 0.42% | 19 | 214 |

All items received unique IDs. The content row had about 11x more training than the others,
so it is not directly comparable to them.
