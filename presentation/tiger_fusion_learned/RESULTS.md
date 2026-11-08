# Learned fusion: scalar and gate

Question: can learning how much LightGCN to mix into the content vector beat the fixed weight in Run 2?

Same setup as Runs 1 and 2: 147,241 test users, train + validation items masked, one training seed per model, RQ-VAE seed 2024, T5 seed 2025, same settings and stopping rule.

## Methods

`scripts/learn_fusion.py` builds the 768-D RQ-VAE input. Cold items (no training interactions) always get content only.

| Model | Fused vector | Learned parameters |
|---|---|---|
| Learnable scalar | normalize(content + a x graph) | one number, a = 0.20 |
| Gate | normalize(content + g x graph), g = sigmoid(Linear([content, graph, log(1 + degree)])) | 1.2M, mean g on warm items 0.39 |

Both were fitted on a next-item task: a user's last 20 training items predict their validation item, scored by cosine similarity to the mean of the history vectors. LightGCN was trained on training edges only, so the validation edge is unseen by it. 80% of validation users were used to fit, 20% to pick the epoch. The test split was not used.

On that held-out 20% of users (not the T5 task):

| Input | Recall@10 | NDCG@10 |
|---|---:|---:|
| Content only | 0.0410 | 0.0234 |
| Fixed alpha = 1 (Run 2) | 0.0396 | 0.0241 |
| Learnable scalar | 0.0417 | 0.0253 |
| Gate | 0.0456 | 0.0280 |

## Test metrics

| Model | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Run 1: content only | 0.02245 | 0.03338 | 0.01519 | 0.01870 |
| Run 2: fixed add, alpha = 1 | 0.02323 | 0.03363 | 0.01596 | 0.01932 |
| Concatenation | 0.02327 | 0.03402 | 0.01556 | 0.01901 |
| MLP | 0.02333 | 0.03413 | 0.01595 | 0.01943 |
| Learnable scalar | 0.02194 | 0.03297 | 0.01480 | 0.01834 |
| Gate | 0.02078 | 0.03073 | 0.01402 | 0.01722 |

## Paired bootstrap on test users

Differences in absolute units, 95% intervals, 2,000 resamples of users, no correction for multiple comparisons. All 22 comparisons are in `comparison.json`.

| Comparison | Metric | Difference | 95% interval | Relative | p |
|---|---|---:|---:|---:|---:|
| Scalar vs Run 1 | R@10 | -0.00041 | -0.00119 to +0.00033 | -1.2% | 0.340 |
| Scalar vs Run 1 | N@10 | -0.00036 | -0.00078 to +0.00007 | -1.9% | 0.109 |
| Scalar vs Run 2 | N@10 | -0.00098 | -0.00147 to -0.00047 | -5.1% | < 0.001 |
| Gate vs Run 1 | R@10 | -0.00265 | -0.00346 to -0.00183 | -7.9% | < 0.001 |
| Gate vs Run 1 | N@10 | -0.00147 | -0.00193 to -0.00102 | -7.9% | < 0.001 |
| Gate vs Run 2 | N@10 | -0.00209 | -0.00260 to -0.00157 | -10.8% | < 0.001 |
| Gate vs MLP | N@10 | -0.00220 | -0.00266 to -0.00176 | -11.3% | < 0.001 |
| Scalar vs MLP | N@10 | -0.00109 | -0.00154 to -0.00063 | -5.6% | < 0.001 |

The scalar cannot be told apart from Run 1. The gate is worse than every other model. The MLP is the best of the fusion methods, as in `presentation/tiger_fusion/RESULTS.md`.

## By the target item's training interactions

Recall@10. The gate loses on frequently seen items.

| Interactions | Run 1 | Run 2 | Scalar | Gate |
|---|---:|---:|---:|---:|
| 0 (20.7% of targets) | 0.00003 | 0.00000 | 0.00007 | 0.00003 |
| 1-4 (25.2%) | 0.00030 | 0.00156 | 0.00056 | 0.00054 |
| 5 or more (54.1%) | 0.06156 | 0.06144 | 0.06067 | 0.05655 |

## Why it probably failed

This is a guess, not tested. The fitting task rewards similarity between consecutive items, which is not what the T5 needs from the semantic IDs. The gate has the most freedom to bend that similarity, so it helped on the task and hurt after quantization.

## Limits

- One training seed per model. The bootstrap covers variation across users, not training randomness.
- The scalar was fitted twice, with learning rates 1e-3 (a = 0.76) and 1e-2 (a = 0.20). The second was kept because it scored higher on the held-out users.
- The fitting used 80% of validation users, and T5 checkpoints were selected on a sample of 20,000 validation users, so validation scores of these two runs are slightly optimistic. Test was not used for either.
- Stored test metrics were regenerated from the saved checkpoints and matched for all seven runs in the comparison.
- Semantic IDs: collisions after retry 1.09% (scalar) and 0.56% (gate), largest shared prefix 19 and 17.

Checkpoints, codes and the fused embeddings are in the Hugging Face repo (`results/tiger_full/{lscalar768,gated768}`, `data/processed/item_fused_{lscalar,gated}.npy`).
