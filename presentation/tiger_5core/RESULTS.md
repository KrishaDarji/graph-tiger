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

## Semantic IDs on the project's 163,257 items

These runs give no recommendation results. Run 1 is the model in use. The others come from pilots or the dropped 64-D plan.

| Embeddings | Use | RQ-VAE epochs | Collisions after RQ-VAE | After retry step | Largest prefix group | First-level codes used |
|---|---|---:|---:|---:|---:|---:|
| Content (768-D) | **Run 1** | 225 | 9.6% | 0.97% | 14 | 247 |
| Content (768-D) | Pilot | 225 | 9.8% | 1.05% | 11 | 254 |
| Projected content (64-D) | Dropped plan | 225 | 11.0% | 1.48% | 66 | 256 |
| Projected content (64-D) | Pilot | 20 | 13.4% | 3.10% | 49 | 137 |
| Fused, alpha = 1 (64-D) | Pilot | 20 | 3.4% | 0.42% | 19 | 214 |

All items received unique IDs. Rows with different epoch counts are not directly comparable. Run 2's semantic IDs are pending.
