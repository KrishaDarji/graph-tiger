# Final TIGER results (Amazon Beauty, full data)

Question: does user-item graph information improve TIGER? On test, yes on the ranking metrics, by a small amount, and with one training seed per model.

All scores use the 147,241 test users, with train + validation items masked. Checkpoints and alpha were chosen on validation. Logs and metrics are in `logs/`.

## Models

| Model | How the graph is used | Selection |
|---|---|---|
| Run 1: simple TIGER | Not used | Epoch 89, stopped at 99 |
| Design A | LightGCN re-ranks Run 1's top 30 candidates, with TIGER frozen | 768-D LightGCN, alpha 1 |
| Run 2 | Content and 768-D LightGCN are fused before RQ-VAE, then TIGER is retrained | Epoch 113, stopped at 123 |

## Test metrics

| Model | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.00482 | 0.00807 | 0.00263 | 0.00367 |
| LightGCN 64-D | 0.01256 | 0.01978 | 0.00799 | 0.01032 |
| LightGCN 768-D | 0.01925 | 0.02881 | 0.01290 | 0.01597 |
| Run 1: simple TIGER | 0.02245 | 0.03338 | 0.01519 | 0.01870 |
| Design A | 0.02292 | 0.03399 | 0.01556 | 0.01911 |
| Run 2 | 0.02323 | 0.03363 | 0.01596 | 0.01932 |

## Paired bootstrap on test users

Differences are in absolute metric units, with 95% intervals, from 2,000 resamples of users. There is no correction for multiple comparisons.

| Comparison | Metric | Difference | 95% interval | Relative | p |
|---|---|---:|---:|---:|---:|
| Design A vs Run 1 | R@5 | +0.00048 | -0.00003 to +0.00096 | +2.1% | 0.059 |
| Design A vs Run 1 | R@10 | +0.00060 | +0.00007 to +0.00109 | +1.8% | 0.033 |
| Design A vs Run 1 | N@5 | +0.00036 | +0.00003 to +0.00069 | +2.4% | 0.034 |
| Design A vs Run 1 | N@10 | +0.00042 | +0.00013 to +0.00070 | +2.2% | 0.007 |
| Run 2 vs Run 1 | R@5 | +0.00078 | +0.00005 to +0.00152 | +3.5% | 0.039 |
| Run 2 vs Run 1 | R@10 | +0.00025 | -0.00062 to +0.00107 | +0.8% | 0.568 |
| Run 2 vs Run 1 | N@5 | +0.00077 | +0.00027 to +0.00126 | +5.1% | 0.001 |
| Run 2 vs Run 1 | N@10 | +0.00062 | +0.00010 to +0.00110 | +3.3% | 0.018 |
| Run 2 vs Design A | R@5 | +0.00031 | -0.00046 to +0.00103 | +1.3% | 0.426 |
| Run 2 vs Design A | R@10 | -0.00035 | -0.00118 to +0.00051 | -1.0% | 0.402 |
| Run 2 vs Design A | N@5 | +0.00041 | -0.00009 to +0.00091 | +2.6% | 0.121 |
| Run 2 vs Design A | N@10 | +0.00020 | -0.00029 to +0.00070 | +1.1% | 0.432 |

Run 2 and Design A cannot be separated on these results.

## Same training budget as Run 1

Run 1 stopped at epoch 99. Run 2's best checkpoint within its first 99 epochs was epoch 97.

| Model | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Run 1 (epoch 89) | 0.02245 | 0.03338 | 0.01519 | 0.01870 |
| Run 2 through epoch 99 (epoch 97) | 0.02290 | 0.03333 | 0.01569 | 0.01905 |

N@10 difference: +0.00035, 95% interval -0.00014 to +0.00088, p = 0.181. None of the four differences has an interval that excludes 0, so Run 2's gain depends on the extra training. The final checkpoint is the main result because the validation rule selected it. Both were evaluated on test.

## By the target item's training interactions

Differences from Run 1, with 95% intervals.

| Interactions | Share of targets | Run 1 R@10 | Run 2 R@10 | Design A R@10 | Run 2 R@10 difference | Run 2 N@10 difference |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 20.7% | 0.00003 | 0.00000 | 0.00003 | -0.00003 (-0.00010 to 0) | -0.00001 (-0.00003 to 0) |
| 1-4 | 25.2% | 0.00030 | 0.00156 | 0.00032 | +0.00126 (+0.00086 to +0.00170) | +0.00061 (+0.00040 to +0.00085) |
| 5 or more | 54.1% | 0.06156 | 0.06144 | 0.06266 | -0.00011 (-0.00178 to +0.00156) | +0.00086 (-0.00011 to +0.00182) |

Run 2's clearest gain is on rare items, though recall there stays very low. Design A's gain is on items with 5 or more interactions (R@10 +0.00110, interval +0.00015 to +0.00205). Neither method helps cold items.

## Semantic IDs

| | Run 1 | Run 2 |
|---|---:|---:|
| Prefix collisions after retry | 0.97% | 2.86% |
| Largest shared prefix group | 14 | 30 |
| Full IDs unique | Yes | Yes |

## Limits

- Each model has one training seed. The bootstrap covers variation across users, not training randomness, so a different seed could change a gain of this size.
- The p-values are not corrected. Across 12 tests on the main three comparisons, only Run 2 vs Run 1 on N@5 survives a Bonferroni threshold of 0.0042.
- Alpha = 1 in Run 2 was not tuned, and rare items get the same graph weight as popular ones.
- Design A was evaluated on test with both 64-D and 768-D LightGCN. The 768-D option was added after the 64-D test score was seen. Validation made the choice: N@10 0.02692 against 0.02504.
- Run 1 and Run 2 stopped by the same patience rule, at different epochs.

## Reproduce

`scripts/compare_final_tiger.py` regenerates rankings from the saved checkpoints, confirms they match the stored metrics, and writes `comparison.json`.
