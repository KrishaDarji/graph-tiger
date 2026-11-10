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