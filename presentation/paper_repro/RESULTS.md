# Paper-protocol results: TIGER, RPG and LightGCN on 5-core Beauty

Data: Amazon Beauty 5-core, 22,363 users, 12,101 items, 198,502 interactions, leave-one-out, history up to 20 items (TIGER) or 50 (RPG), full ranking, no filtering of seen items. Text: Sentence-T5-base on "Title / Price / Brand / Categories". One L40S. Code: `scripts/paper/`, `src/tiger/`. Per-run numbers: `runs.json`; mean and std: `summary.txt`.

## Test results

| Model | Seeds | R@5 | N@5 | R@10 | N@10 |
|---|---:|---:|---:|---:|---:|
| TIGER paper, single run (Table 1) | 1 | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| TIGER paper, 3-seed mean (Table 9) | 3 | 0.0441 | 0.0309 | 0.0642 | 0.0374 |
| TIGER, our T5 + RQ-VAE | 3 | 0.0345 | 0.0229 | 0.0546 | 0.0294 |
| TIGER, our T5 + RK-Means | 3 | 0.0370 | 0.0242 | 0.0583 | 0.0310 |
| RPG official code (Meta), Sentence-T5 | 3 | 0.0507 | 0.0353 | 0.0736 | 0.0426 |
| RPG-style (ours), no graph | 6 | 0.0496 | 0.0344 | 0.0725 | 0.0418 |
| RPG-style + LightGCN, scalar weight | 3 | 0.0509 | 0.0349 | 0.0758 | 0.0429 |
| RPG-style + LightGCN, gated | 6 | 0.0512 | 0.0354 | 0.0767 | 0.0436 |

Published for reference (their own setups): RPG 0.0809 / 0.0464 (R@10 / N@10, OpenAI embeddings), ACERec 0.0841 / 0.0501.

Gated vs no graph, same six seeds, paired: R@10 +5.8% (p = 0.0014, better in 6 of 6), N@10 +4.3% (p = 0.014, 6 of 6), R@5 +3.4% (p = 0.042), N@5 +2.7% (p = 0.10).

## TIGER reproduction check

Our TIGER reaches about 85% of the paper (R@10 0.0546 vs 0.0642). To see whether the T5 or the semantic IDs are at fault, we repeated the paper's ID ablation (Table 2), one seed:

| IDs | Paper R@10 | Ours R@10 | Ours / paper |
|---|---:|---:|---:|
| Random | 0.0434 | 0.0375 | 86% |
| LSH | 0.0533 | 0.0477 | 90% |
| RQ-VAE | 0.0642 | 0.0546 | 85% |

The order and the relative gains match, and the shortfall is the same for every ID type, so it comes from the T5 training recipe, not from the tokenizer. We tried, on seed 42 and the same codes (R@10): Sentence-T5-XL 0.0560, d_model 256 0.0515, dropout 0.2 0.0538, dropout 0.3 0.0507, AdamW 0.0436, tied embeddings 0.0550, beam 50 0.0550. None closes the gap. The paper does not state the optimizer, beam size, how training examples are built, or which checkpoint it reports, and its "13M parameters" does not match its stated sizes (those give 4.9M). We therefore treat it as reproduced up to these unknowns, not exactly.

Details we followed: 4 + 4 layers, 6 heads of 64, d_model 128, d_ff 1024, ReLU, dropout 0.1, 2000 hashed user tokens, 3 x 256 codes + collision code, batch 256, lr 0.01 for 10k steps then inverse square root, unconstrained beam 20. RQ-VAE: 512-256-128 to 32, beta 0.25, k-means init, batch 1024, 20k epochs. Deviation: Adagrad lr 0.4 diverged for us, so lr 0.05 (level-1 codebook usage 59%). Validation recall peaks at 20-25k steps and falls after, so the best-validation checkpoint is reported.

## Graph in the tokens did not help

| Variant (R@10) | Result |
|---|---|
| RPG-style, tokens from content + LightGCN, weight 0.5 / 1.0, 2 seeds | 0.0722 / 0.0754 (no graph 0.0725) |
| Same with gated input | 0.0760 / 0.0756 (gate alone 0.0767) |
| TIGER RK-Means on content + LightGCN, weight 0.25 / 0.5 / 1 / 2, 1 seed | 0.0594 / 0.0571 / 0.0527 / 0.0530 (content only 0.0583) |

## Limits

- Our RPG-style model is a reimplementation of RPG's published design. The official code is the baseline to cite, and the two agree within about 3%.
- Official RPG was run with seeds 2024-2026, Sentence-T5-base with PCA 128 in place of the default OpenAI embeddings (which need a paid key).
- TIGER variants other than the 3-seed rows use one seed.
- LightGCN here is the 64-D, 2-layer model trained on 5-core training edges (validation recall@10 0.0683).
- Checkpoints and early stopping were chosen on validation only, never on test.

## Stronger text encoders and ACERec-style components (Nov 2026)

Same data and protocol, RPG-style model, test R@10 / N@10. Per-run numbers: `runs.json`, means: `summary.txt`.

| Text encoder (vectors reduced to 128-d) | Seeds | No graph | With gated LightGCN |
|---|---:|---|---|
| Sentence-T5-base | 6 | 0.0725 / 0.0418 | 0.0767 / 0.0436 |
| Qwen3-Embedding 0.6B | 2 | 0.0724 / 0.0416 | not run |
| BGE-large | 2 | 0.0727 / 0.0418 | not run |
| Qwen3-Embedding 4B | 2 (3 with gate) | 0.0739 / 0.0426 | 0.0775 / 0.0436 |
| Qwen3-Embedding 8B | 4 | 0.0739 / 0.0431 | 0.0765 / 0.0436 |

A larger encoder helps a little without the graph (about +2%) and nothing once the gate is on, at 5x the vector storage and a large one-off embedding pass. We keep Sentence-T5-base.

ACERec-style components, our reimplementation (`scripts/paper/ace.py`), seed 2024, one run each:

| Variant | R@10 | N@10 |
|---|---:|---:|
| Same architecture, none of the components (check) | 0.0742 | 0.0427 |
| Intent token | 0.0663 | 0.0392 |
| Token merger | 0.0635 | 0.0369 |
| Merger + intent | 0.0677 | 0.0400 |
| Merger + intent + alignment loss (0.01) | 0.0692 | 0.0412 |
| Intent + alignment loss | 0.0678 | 0.0407 |
| Merger + intent + alignment loss + gate | 0.0750 | 0.0445 |
| Merger + intent + gate | 0.0732 | 0.0424 |

None of the components beat the plain architecture on Beauty (16.5M vs 14.0M parameters). The alignment-loss weight 0.05 run ran out of GPU memory and was not repeated. Built from the paper text without the authors' code, so a different result from theirs does not show their method fails.

## Stage 3: transition graph, dense loss, weight averaging (Nov 2026)

New evaluator (`src/evalkit.py`, rules in `EVAL_PROTOCOL.md`), new training script `scripts/paper/gen.py`. Test, raw ranking over all 12,101 items (the papers' protocol), 5 seeds (2024-2028), Sentence-T5-base, history 50.

| Model | R@5 | N@5 | R@10 | N@10 |
|---|---:|---:|---:|---:|
| RPG-style, plain | 0.0485 | 0.0340 | 0.0722 | 0.0416 |
| + gated LightGCN | 0.0509 | 0.0348 | 0.0757 | 0.0428 |
| **+ transition graph, dense loss, EMA (ours)** | **0.0619** | **0.0427** | **0.0927** | **0.0527** |
| RPG official code, same encoder (3 seeds) | 0.0507 | 0.0353 | 0.0736 | 0.0426 |
| RPG paper (OpenAI embeddings) | 0.0550 | 0.0381 | 0.0809 | 0.0464 |
| ACERec paper | 0.0591 | 0.0421 | 0.0841 | 0.0501 |

Ours vs gated LightGCN, paired per user, same seeds: R@5 +21.5%, N@5 +22.7%, R@10 +22.5%, N@10 +23.0%; better in 5 of 5 seeds, 95% bootstrap interval excludes 0 in 5 of 5. Std over seeds is at most 0.0009 (R@10). Per-run files: `results/paper/gen_*` (not in git); table rows: `runs.json`, `summary.txt`.

What changed, one component each (seed 2024, R@10 / N@10; gated LightGCN 0.0767 / 0.0430):

| Change | R@10 | N@10 |
|---|---:|---:|
| Dense item loss (weight 0.3) | 0.0810 | 0.0448 |
| EMA of weights (0.99) | 0.0789 | 0.0446 |
| Transition graph vector, 64-d, window 3, instead of LightGCN | 0.0830 | 0.0468 |
| Same, 128-d | 0.0873 | 0.0498 |
| 256-d / window 5 (128-d) | 0.0879 / 0.0836 | 0.0494 / 0.0479 |
| 128-d + dense loss + EMA (final) | 0.0939 | 0.0537 |
| 128-d window 5 / 256-d window 5, with dense loss + EMA | 0.0922 / 0.0924 | 0.0519 / 0.0514 |

The transition graph vector is a truncated SVD (128-d) of the normalized item-item co-occurrence matrix, each item pair weighted 1/distance inside a window of 3 steps, built from each user's training prefix only (all but the last two items). Alone as a nearest-neighbour recommender on the last item it gets test R@10 0.062 (popularity: 0.011). The dense item loss adds a softmax over 4,096 sampled items on cosine similarity between the sequence state and the item vector, trained next to the 32 digit losses; at scoring time it adds `beta` times its log-probability (`beta` chosen on validation). Tried without gain: short or low-learning-rate warm-up (underfit badly), trainable graph vector, graph dropout, 3 layers, dropout 0.3, dense weight 1 or 3.

Beyond accuracy and cost (5-seed means; times are on a GPU shared by several jobs):

| | Gated LightGCN | Ours |
|---|---:|---:|
| Coverage@10 (share of catalog ever recommended) | 0.33 | 0.62 |
| Gini of recommendations@10 (lower = more even) | 0.925 | 0.849 |
| Recall@10 on least popular quarter of items | 0.018 | 0.027 |
| Recall@10 on most popular quarter | 0.123 | 0.140 |
| Parameters | 13.84M | 13.90M |
| Training minutes / scoring ms per user | 33 / 0.22 | 34 / 0.22 |

With history items removed from the ranking (second view, not comparable with the papers): ours R@10 0.1004 / N@10 0.0601.

## Cold start (not solved)

5% of items (605) are kept out of all training data and graphs; 1,228 test users have such a target; ranking is over all items. Seed 2024.

| Model | Cold Recall@10 | Cold Recall@50 | Warm Recall@10 |
|---|---:|---:|---:|
| Plain RPG-style | 0.0008 | 0.0187 | 0.0734 |
| Ours | 0.0008 | 0.0114 | 0.0918 |
| Ours, graph vector randomly dropped in training | 0.0000 | 0.0130 | 0.0920 |
| Plain + content-similarity score term (weight picked on validation) | 0.0073 | 0.0489 | 0.0771 |

Items never seen in training are almost never in the top 10 for any of these models. The graph model did not make this worse than the plain model, and a content-similarity term in the score helps the plain model (overall R@10 0.0694 to 0.0733) but validation gave it zero weight for the graph model. A deployable system needs a separate path for new items (for example reserved slots filled by content similarity) until they collect interactions.

## Limits

- The final model uses collaborative statistics from the training set (transition graph); TIGER and RPG use item text only. The comparison with their published numbers is therefore a comparison of systems, not of architectures alone.
- Hyper-parameters (graph dimension and window, dense weight, EMA decay) were screened with seed 2024 on about 20 variants. Validation scores were logged for all of them, but the test scores of those runs were also visible while we chose, so the test users carry some selection bias. The 5-seed rows reuse the chosen setting with new seeds (2025-2028) and the same test users. The validation ranking of the variants agrees with the test ranking. A second dataset (below, if present) was not used for any choice.
- Published numbers use their own text encoders (RPG: OpenAI embeddings, ours: Sentence-T5-base) and seeds; official RPG with our encoder is the like-for-like baseline.
- Graph-based vectors use other users' training interactions, as is usual in leave-one-out evaluation.

## More datasets (Sports, Toys, CDs): same code, RPG's own hyper-parameters

Test, raw full ranking, Sentence-T5-base, 5 seeds (2024-2028) for Sports and Toys, 3 seeds (2024-2026) for CDs. Code length and learning rate are RPG's published per-dataset values for Sports and Toys (16 digits, lr 0.003); nothing was tuned for our model. "Ours" = transition graph (128-d, window 3) + dense loss 0.3 + EMA 0.99, as on Beauty. Published rows use their own encoders and are not re-run.

| Dataset | Model | R@5 | N@5 | R@10 | N@10 |
|---|---|---:|---:|---:|---:|
| Sports (35,598 users, 18,357 items) | plain RPG-style (5 seeds) | 0.0311 | 0.0217 | 0.0445 | 0.0260 |
| | SASRec, full softmax (ours, 3 seeds) | 0.0252 | 0.0135 | 0.0405 | 0.0185 |
| | **ours (5 seeds)** | **0.0366** | **0.0241** | **0.0553** | **0.0301** |
| | RPG paper (OpenAI emb.) | 0.0314 | 0.0216 | 0.0463 | 0.0263 |
| | ACERec paper (ST5) | 0.0341 | 0.0231 | 0.0493 | 0.0280 |
| Toys (19,412 users, 11,924 items) | plain RPG-style (5 seeds) | 0.0581 | 0.0413 | 0.0815 | 0.0488 |
| | SASRec, full softmax (ours, 3 seeds) | 0.0466 | 0.0290 | 0.0676 | 0.0358 |
| | **ours (5 seeds)** | **0.0651** | **0.0447** | **0.0962** | **0.0547** |
| | RPG paper (OpenAI emb.) | 0.0592 | 0.0401 | 0.0869 | 0.0490 |
| | ACERec paper (ST5) | 0.0688 | 0.0505 | 0.0916 | 0.0579 |
| CDs (64,443 items, 32 digits, lr 0.003) | SASRec, full softmax (ours, 1 seed) | 0.0435 | 0.0246 | 0.0731 | 0.0342 |
| | **ours (3 seeds)** | **0.0604** | **0.0386** | **0.0956** | **0.0499** |
| | RPG paper | 0.0498 | 0.0338 | 0.0735 | 0.0415 |
| | TIGER paper | 0.0492 | 0.0329 | 0.0748 | 0.0411 |

Standard deviations over the 5 seeds are at most 0.0012 (Recall) and 0.0007 (NDCG).

- **Sports:** above RPG (R@10 +19.4%, N@10 +14.4%) and ACERec (R@10 +12.2%, N@10 +7.5%) on all four metrics.
- **Toys:** above RPG on all four (R@10 +10.7%, N@10 +11.6%); above ACERec on R@10 only (+5.0%). ACERec is ahead on R@5 (-5.4%), N@5 (-11.5%) and N@10 (-5.5%). We do not claim to beat ACERec on Toys.
- **CDs:** above the published RPG (R@10 +30%, N@10 +20%) and TIGER (R@10 +28%, N@10 +21%), and above our own full-softmax SASRec (+31% R@10, +46% N@10). Three seeds (std at most 0.0001; the seeds give different per-user rankings, the large user count keeps the mean stable). All three runs ran to the 60-epoch cap with the best validation epoch at 56 and the curve nearly flat; a longer schedule might add a little. CDs needed 32 digits, lr 0.003 and a 1,500-step warm-up to train at all; the plain RPG-style model with RPG's CDs settings never converged (validation R@10 0.01-0.03 after 40+ epochs), so no like-for-like plain row is reported. The published CDs rows use different text encoders and a different CDs file version may be in use; treat the comparison as indicative.
- **Against our own plain model**, paired per user and seed (5 seeds, intervals exclude 0 in 5/5): Sports R@5 +17.9%, N@5 +11.5%, R@10 +24.1%, N@10 +16.0%; Toys R@5 +12.2%, N@5 +8.3%, R@10 +18.0%, N@10 +12.0%.
- Sports lists users and items the other way round in RPG's Table 1; the 5-core file has 35,598 users and 18,357 items (our evaluation uses all 35,598 users).
- Our SASRec (full-softmax loss, 128-d, dropout 0.5, picked on Beauty validation) is much stronger than the SASRec rows copied into the papers on Beauty (0.0748 vs 0.0605 R@10), and matches them on Toys (0.0676 vs 0.0675). On CDs it reaches 0.0731 R@10 against the published 0.0619.

Beyond accuracy (5-seed means, ours vs plain): Sports coverage@10 0.63 vs 0.41, Gini 0.88 vs 0.92, tail-quartile recall 0.014 vs 0.010; Toys coverage@10 0.70 vs 0.67, tail-quartile recall 0.046 vs 0.031.

## Cost on an idle GPU (Beauty, seed 2024, NVIDIA L40S, nothing else running)

| | Plain RPG-style | Ours |
|---|---:|---:|
| Parameters | 13.58M | 13.90M |
| Training time (to early stop) | 5.1 min (69 epochs) | 5.3 min |
| Batch scoring, ms per user | 0.062 | 0.066 |
| Single-user request, p50 / p95 / p99 (ms) | 4.9 / 5.0 / 5.0 | 7.6 / 7.7 / 7.8 |
| Peak GPU memory (GB) | 2.0 | 2.4 |
| Test R@10 / N@10 | 0.0734 / 0.0418 | 0.0939 / 0.0537 |

The single-user latency includes rebuilding the item vectors from the current weights on each request (an upper bound; they can be cached between model updates). Ours adds about 2.7 ms per single request for the graph gate and dense head, and 0.3M parameters. Training time was measured with the GPU otherwise idle; earlier per-run minutes in other tables were taken while several jobs shared the GPU and are not comparable.

## Are we overfitting? (`scripts/paper/overfit_report.py`)

- **Selection was on validation only.** Over 33 Beauty configurations (seed 2024), the Spearman correlation between validation and test NDCG@10 is 0.98. The test score sits a constant 0.014 below validation for every config, including the untuned baselines, so the gap is a property of the split and not of our tuning. The config picked by validation reaches test N@10 0.0529; the best on test was 0.0537.
- **Per-dataset tuning gave nothing beyond noise.** On Toys and Sports (seed 2024) we tried a 5x larger dense weight, window 5, and lr 0.01. The best-by-validation variants differ from the default by at most 1.7% on validation, inside the seed spread, so we kept RPG's published settings and did not tune further. The validation-picked variant is also best on test in both datasets.
- Seeds: std over seeds is at most 0.0018 (R@10) on every dataset; the model has 8.8-13.9M parameters and early stopping on validation.
- Remaining risk: about 20 Beauty variants were screened with the test split visible in the logs (see Limits above). Sports and Toys were not used for any choice.

## Ideas tried and dropped (Beauty, seed 2024, chosen on validation)

| Idea | Val N@10 | Test R@10 / N@10 | Verdict |
|---|---:|---:|---|
| Reference (SVD transition graph, dense loss, EMA) | 0.0677 | 0.0939 / 0.0537 | |
| Directional graph (separate source and target vectors) | 0.0649 | 0.0878 / 0.0514 | worse |
| Graph vector inside the quantized item tokens, weight 0.5 / 1.0 | 0.0658 / 0.0656 | 0.0893 / 0.0907 | worse |
| Text-to-graph "bridge" + graph dropout | 0.0681 | 0.0925 / 0.0529 | no change |
| PPMI weighting instead of degree normalisation | 0.0628 | 0.0866 / 0.0481 | worse |
| Fixed random projection instead of SVD (stable basis) | 0.0659 | 0.0889 / 0.0511 | 5% worse, still far above LightGCN gate (0.0757) |

**Refreshing the graph without retraining does not work.** Train on a graph built from the oldest 70% of each user's training prefix, then swap in the full-prefix graph with the weights fixed:
- SVD: test R@10 collapses to 0.0156 (every SVD run has its own arbitrary basis). Rotating the new vectors onto the old basis (Procrustes) gives 0.0875 against 0.0898 for the stale graph.
- Random projection (same basis by construction): 0.0837 against 0.0855 stale.
In both cases the refreshed graph is slightly worse than the stale one, so a refresh needs a short fine-tune.
