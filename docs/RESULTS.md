# Results

> Numbers below come from earlier runs and are being re-run and re-verified; treat them as provisional.

Test, full-catalog ranking, leave-one-out, Sentence-T5 text. Protocol and significance rules: `PROTOCOL.md`. Rows marked "3 seeds" are means over seeds 2024 to 2026.

## Main comparison

| | Beauty R@10 / N@10 | Sports R@10 / N@10 | Toys R@10 / N@10 |
|---|---|---|---|
| EASE (linear, ours) | 0.0501 / 0.0258 | 0.0383 / 0.0198 | 0.0548 / 0.0294 |
| SASRec, full softmax (ours) | 0.0748 / n/a | 0.0405 / 0.0185 | 0.0676 / 0.0358 |
| Plain RPG-style (ours) | 0.0722 / 0.0416 | 0.0445 / 0.0260 | 0.0815 / 0.0488 |
| **Graph-TIGER** (3 seeds) | **0.0945 / 0.0533** | **0.0544 / 0.0298** | **0.0983 / 0.0563** |
| Graph-TIGER + EASE blend (3 seeds) | 0.0955 / 0.0542 | 0.0576 / 0.0318 | 0.0983 / 0.0563 (blend weight 0) |
| RPG | 0.0809 / 0.0464 | 0.0463 / 0.0263 | 0.0869 / 0.0490 |
| ACERec | 0.0856 / 0.0513 | 0.0494 / 0.0287 | 0.0940 / 0.0591 |
| TopoGR | 0.0883 / 0.0522 | 0.0554 / 0.0314 | 0.0995 / 0.0588 |

Graph-TIGER on Toys uses batch 128, learning rate 0.005 and dense temperature 0.07 (validation-chosen). The EASE blend adds the z-scored score of a linear item-item model with a weight picked on validation; it is an ensemble with a known method and is reported separately.

**Against the published numbers** (95% bootstrap interval of our seed-averaged metric versus the published value; "above/below" = fully on one side, "n.s." = contains it):

| | vs RPG | vs ACERec | vs TopoGR |
|---|---|---|---|
| Beauty | above on all four metrics (+14 to +17%) | R@10 +10.4% above; others n.s. | R@10 +7.0% above; others n.s. |
| Beauty + EASE | above on all four (+16 to +18%) | R@5, R@10, N@10 above; N@5 n.s. | R@10 +8.2% above; others n.s. |
| Sports | above on all four (+11 to +18%) | R@5, R@10 above; NDCGs n.s. | N@5 -6.9%, N@10 -5.0% below; Recall n.s. |
| Sports + EASE | above on all four (+19 to +24%) | above on all four | n.s. on all four |
| Toys | above on all four (+13 to +16%) | R@10 above; N@5, N@10 below | R@5, N@5 below; R@10, N@10 n.s. |

Beauty is the dataset where we pass the newest published results on Recall@10. On Sports we match them only with the EASE blend. On Toys we do not: ACERec and TopoGR place hits higher (N@5 6 to 10% lower for us). Beauty + EASE is also above DiffGRM on all four metrics; Sports + EASE is above it on Recall and n.s. on NDCG.

## What each component adds (Beauty, seed 2024, R@10 / N@10)

| Model | R@10 | N@10 |
|---|---:|---:|
| Plain RPG-style | 0.0734 | 0.0418 |
| + gated LightGCN input | 0.0767 | 0.0430 |
| + dense item loss | 0.0810 | 0.0448 |
| + EMA of weights | 0.0789 | 0.0446 |
| Transition-graph vector (64-d) instead of LightGCN | 0.0830 | 0.0468 |
| Transition-graph vector (128-d, window 3) | 0.0873 | 0.0498 |
| 128-d + dense loss + EMA (Graph-TIGER) | 0.0939 | 0.0537 |
| Graph vector inside the quantised codes, weight 0.5 / 1.0 | 0.0893 / 0.0907 | |

Entering as a gated model input is better than entering through the codes. Graph-TIGER over the gated LightGCN model, paired per user over five seeds: R@10 +22.5%, N@10 +23.0%, better in 5 of 5 seeds with the interval excluding 0. Over the plain model on the other datasets (5 seeds): Sports R@10 +24.1%, Toys R@10 +18.0%.

The transition-graph vector is a 128-d truncated SVD of the item-item co-occurrence matrix (pairs within 3 steps, weight 1/distance, symmetric normalisation). The dense loss is a softmax over 4,096 sampled items on the cosine between the sequence state and the item vector, trained next to the digit losses and added to the score with a validation-chosen weight.

## Temporal split (3 seeds, R@10 / N@10)

| | Test users | Plain | Graph-TIGER | EASE |
|---|---:|---|---|---|
| Beauty | 5,585 | 0.0209 / 0.0107 | 0.0246 / 0.0122 | 0.0215 / 0.0116 |
| Sports | 9,328 | 0.0198 / 0.0103 | 0.0260 / 0.0138 | 0.0243 / 0.0128 |
| Toys | 5,021 | 0.0173 / 0.0087 | 0.0208 / 0.0101 | 0.0215 / 0.0122 |

Graph-TIGER beats the plain model in 3 of 3 seeds on every dataset (R@10 +18%, +32%, +20%), but the test sets are small and several per-seed intervals include 0. Under time drift it is about level with the linear model.

## New items and refresh

- **New items.** 14 to 20% of temporal test targets are unseen in training and the model almost never ranks them. Reserving k tail slots in the top 10 for the best unseen items (content plus graph nearest neighbours) raised overall R@10 in the temporal setting at k=2: Beauty 0.0249 to 0.0286, Toys 0.0201 to 0.0261, and lowered it on Sports (0.0277 to 0.0266). On the standard Beauty split one tail slot lifts R@10 of users with an unseen target from 0.0008 to 0.077 for a 0.9% loss in overall R@10; head placement hurts NDCG (-17% at k=1), so use the tail.
- **Refresh.** Rebuilding the graph for new data and rotating it onto the old basis (orthogonal Procrustes), then fine-tuning for 5 epochs (about one minute), recovers 74% (Beauty), 77% (Sports) and 53% (Toys) of the R@10 gap between a stale model and a full retrain. Swapping the graph without fine-tuning helps on two of three datasets and hurts on Sports.

| Temporal R@10 / N@10 | Stale | Graph swap only | Fine-tune (5 epochs) | Full retrain |
|---|---|---|---|---|
| Beauty | 0.0249 / 0.0124 | 0.0304 / 0.0149 | 0.0514 / 0.0258 | 0.0609 / 0.0305 |
| Sports | 0.0277 / 0.0148 | 0.0258 / 0.0133 | 0.0327 / 0.0181 | 0.0342 / 0.0191 |
| Toys | 0.0201 / 0.0098 | 0.0231 / 0.0107 | 0.0295 / 0.0136 | 0.0378 / 0.0182 |

The fine-tuned and retrained models also see the validation window; the stale one does not.

## Cold-start protocol (Beauty, seed 2024; R@10 ranking only the 605 held-out items)

| Method | Cold-pool R@10 |
|---|---:|
| Random | 0.0171 |
| Plain RPG-style | 0.1865 |
| Graph-TIGER, full score | 0.2288 |
| Graph-TIGER, dense head only | 0.2557 |
| Content nearest neighbour | 0.2256 |
| Graph nearest neighbour (imputed vectors) | 0.2492 |
| Content + graph nearest neighbour | 0.2720 |

No generative model puts unseen items in the global top 10 (R@10 below 0.5%). Deploy a content-plus-graph path for new items and reserve slots for it.

## Cost (Beauty, idle L40S)

| | Plain | Graph-TIGER |
|---|---:|---:|
| Parameters | 13.58M | 13.90M |
| Training to early stop | 5.1 min | 5.3 min |
| Batch scoring, ms per user | 0.062 | 0.066 |
| Single request p50 / p95 | 4.9 / 5.0 ms | 7.6 / 7.7 ms |

Re-scoring only the top-50 items of the dense head (`--cands 50`) keeps 99.7 to 100.6% of full R@10 on Beauty, Sports and Toys (exact top-K; an approximate index would add error).

## Limits

- Graph-TIGER uses collaborative statistics from the training data; TIGER, RPG and ACERec use item text only, so the comparison is between systems, not architectures alone.
- Hyper-parameters were chosen with seed 2024 on validation, but on Beauty the test scores of those runs were also visible while choosing. Sports and Toys were not used for earlier choices.
- Published numbers are treated as exact, and use their own seeds and (for some) text encoders.
- The graph uses other users' interactions that may be later than a test target (as in all leave-one-out evaluation); the temporal split removes this.
- Toys is not state of the art on NDCG; the EASE blend needs an item-by-item matrix and was not run on catalogs of 60k items.
