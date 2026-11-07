# Fusion results: how to combine content and graph before RQ-VAE

Question: does the way content and LightGCN embeddings are combined change TIGER's results? Run 2 adds the two normalized vectors. Two alternatives were trained: concatenation, and a fusion MLP.

**Answer: no method stands out.** On test, the MLP improves on content-only TIGER by about as much as Run 2 does, and concatenation's gain can't be told apart from noise. None of the three graph methods can be separated from another. Run 2 remains the simplest choice.

All scores use the 147,241 test users, with train and validation items masked. Checkpoints were chosen on validation, and each model's test split was evaluated once. Logs and metrics are in `logs/`.

## Models

Everything after the fusion step matches Run 2: Sentence-T5 and 768-D LightGCN inputs, RQ-VAE seed 2024, T5 seed 2025, and the same settings and stopping rule.

| Model | RQ-VAE input | Selection |
|---|---|---|
| Run 1: content only | 768-D Sentence-T5 | Epoch 89, stopped at 99 |
| Run 2: normalize and add | normalize(content + normalized graph), 768-D | Epoch 113, stopped at 123 |
| Concatenation | [normalized content, normalized graph], 1536-D | Epoch 100, stopped at 110 |
| MLP | 768-D output of a fusion MLP | Epoch 117, stopped at 127 |

For items with no training interactions (26.6% of items), the graph part is 0 in every method.

**Concatenation** changes the RQ-VAE's input and output size. Its hidden layers are unchanged, so its first layer compresses 1536 to 512 instead of 768 to 512. Per dimension, the graph half varies about 4 times as much as the content half, so the graph accounts for most of the RQ-VAE's reconstruction error.

**The MLP** (1536 → 1024 → 768, unit-norm output) was trained on its own before the RQ-VAE, then frozen. A decoder had to rebuild both halves from the 768-D output, with each half's error divided by its variance so they count equally. It uses no interactions. On 5% of items held out, it rebuilds 91% of the content variance and 86% of the graph variance.

## Test metrics

| Model | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Run 1: content only | 0.02245 | 0.03338 | 0.01519 | 0.01870 |
| Design A: re-ranking | 0.02292 | 0.03399 | 0.01556 | 0.01911 |
| Run 2: normalize and add | 0.02323 | 0.03363 | 0.01596 | 0.01932 |
| Concatenation | 0.02327 | 0.03402 | 0.01556 | 0.01901 |
| MLP | 0.02333 | 0.03413 | 0.01595 | 0.01943 |

## Paired bootstrap on test users

Differences are in absolute metric units, with 95% intervals, from 2,000 resamples of users. There is no correction for multiple comparisons. Comparisons involving only Run 1, Run 2 and Design A are in `presentation/tiger_full/RESULTS.md` and are unchanged.

| Comparison | Metric | Difference | 95% interval | Relative | p |
|---|---|---:|---:|---:|---:|
| Concatenation vs Run 1 | R@5 | +0.00082 | +0.00008 to +0.00158 | +3.7% | 0.040 |
| Concatenation vs Run 1 | R@10 | +0.00064 | -0.00022 to +0.00155 | +1.9% | 0.163 |
| Concatenation vs Run 1 | N@5 | +0.00036 | -0.00015 to +0.00086 | +2.4% | 0.163 |
| Concatenation vs Run 1 | N@10 | +0.00031 | -0.00019 to +0.00082 | +1.7% | 0.226 |
| MLP vs Run 1 | R@5 | +0.00088 | +0.00018 to +0.00160 | +3.9% | 0.014 |
| MLP vs Run 1 | R@10 | +0.00075 | -0.00006 to +0.00156 | +2.2% | 0.065 |
| MLP vs Run 1 | N@5 | +0.00076 | +0.00027 to +0.00122 | +5.0% | 0.001 |
| MLP vs Run 1 | N@10 | +0.00073 | +0.00026 to +0.00117 | +3.9% | 0.002 |
| Concatenation vs Run 2 | R@10 | +0.00039 | -0.00037 to +0.00118 | +1.2% | 0.328 |
| Concatenation vs Run 2 | N@10 | -0.00030 | -0.00074 to +0.00015 | -1.6% | 0.195 |
| MLP vs Run 2 | R@10 | +0.00050 | -0.00031 to +0.00134 | +1.5% | 0.274 |
| MLP vs Run 2 | N@10 | +0.00011 | -0.00040 to +0.00062 | +0.6% | 0.696 |
| MLP vs Concatenation | R@10 | +0.00011 | -0.00077 to +0.00096 | +0.3% | 0.800 |
| MLP vs Concatenation | N@10 | +0.00041 | -0.00006 to +0.00089 | +2.2% | 0.095 |

R@5 and N@5 for the last six rows, and every comparison with Design A, are in `comparison.json`. None of those intervals excludes 0.

- **Compared with Run 1:** the MLP's ranking gains (N@5, N@10) are clear and similar in size to Run 2's (+5.1% and +3.3%). Concatenation's are not, and only its R@5 interval excludes 0.
- **Between the graph methods:** no interval excludes 0, so the three can't be ranked.

## By the target item's training interactions

Recall@10 by group:

| Interactions | Share of targets | Run 1 | Run 2 | Concatenation | MLP |
|---|---:|---:|---:|---:|---:|
| 0 | 20.7% | 0.00003 | 0.00000 | 0.00003 | 0.00000 |
| 1-4 | 25.2% | 0.00030 | 0.00156 | 0.00191 | 0.00078 |
| 5 or more | 54.1% | 0.06156 | 0.06144 | 0.06198 | 0.06272 |

Differences from Run 1, with 95% intervals:

| Interactions | Concatenation R@10 | MLP R@10 | MLP N@10 |
|---|---:|---:|---:|
| 1-4 | +0.00161 (+0.00113 to +0.00207) | +0.00048 (+0.00019 to +0.00081) | +0.00024 (+0.00008 to +0.00041) |
| 5 or more | +0.00043 (-0.00118 to +0.00193) | +0.00117 (-0.00029 to +0.00266) | +0.00124 (+0.00042 to +0.00208) |

The methods help different items:

- **Concatenation** helps rare items most. It is clearly ahead of the MLP there (R@10 +0.00113, interval +0.00065 to +0.00164), but it can't be separated from Run 2.
- **The MLP** is the only fusion method with a clear gain on popular items (N@10 +0.00124). Design A also gains there. On rare items it gains less than Run 2 (R@10 -0.00078, interval -0.00121 to -0.00030).
- **No method** helps cold items.

Rare and popular items both matter here. Rare items are 25% of targets, but recall on them stays below 0.002. Overall scores are driven by the 54% of targets with 5 or more interactions.

## Validation and test disagree

Checkpoints were selected on a sample of 20,000 validation users. The best sampled score overstates the run, because it is the highest of 100 or more noisy readings. Validation on all users, at each run's selected checkpoint, gives a different order:

| Model | Best N@10, 20,000-user sample | N@10, all validation users | Test N@10 |
|---|---:|---:|---:|
| Run 1 | 0.02492 | 0.02381 | 0.01870 |
| Run 2 | 0.02531 | 0.02525 | 0.01932 |
| Concatenation | 0.02669 | 0.02488 | 0.01901 |
| MLP | 0.02582 | 0.02441 | 0.01943 |

Concatenation led on the sample by 5%, but it is behind Run 2 on all validation users and on test. The MLP is behind Run 2 on all validation users, but ahead on test. The three graph methods change order between data sets, which is consistent with differences this small being noise.

## Semantic IDs

| | Run 1 | Run 2 | Concatenation | MLP |
|---|---:|---:|---:|---:|
| Prefix collisions after retry | 0.97% | 2.86% | 2.64% | 4.01% |
| Items sharing a prefix | 2,781 | 6,985 | 7,111 | 9,632 |
| Largest shared prefix group | 14 | 30 | 16 | 33 |
| First-level codes used, of 256 | 247 | 254 | 206 | 248 |
| Full IDs unique | Yes | Yes | Yes | Yes |

Collisions don't track results. The MLP has the most collisions and the best test N@10. Concatenation uses fewer first-level codes, which fits the graph half dominating its reconstruction error.

## Limits

- **One training seed per model.** The bootstrap covers variation across users, not training randomness. Gaps of 1-2% between graph methods are well within what a different seed could change.
- **Uncorrected p-values.** Across the 28 new tests, the Bonferroni threshold is 0.0018. Only MLP vs Run 1 on N@5 (p = 0.001) clears it.
- **The MLP's settings were chosen by reconstruction only.** The learning rate was picked from 1e-3, 3e-4 and 1e-4 on held-out reconstruction, before any recommendation result was seen. Other sizes and objectives were not tried.
- **Concatenation also changes the RQ-VAE.** Its first and last layers are larger, so the comparison with Run 2 isn't only about the input.
- **The two new models trained together on one GPU,** which only affected speed.
- **The runs' `config.json` files record commit `505d814`,** because the fusion scripts were committed afterwards. The scripts that built the inputs are in the commit that adds this file.

## Reproduce

```bash
.venv/bin/python scripts/build_concat_1536.py
.venv/bin/python scripts/build_mlp_fused_768.py
scripts/run_tiger_fusion.sh
.venv/bin/python scripts/compare_final_tiger.py --fusion --output-dir results/tiger_full/comparison_fusion
```

The MLP trains on the GPU, so a rebuild may differ slightly. The exact `item_mlp768.npy` and the MLP weights are in the team's Hugging Face repo, with both trained runs. The comparison regenerates rankings from the saved checkpoints, stops if they don't reproduce the stored test metrics, and writes `comparison.json`.
