# Results

Test results from the verified pipeline (preprocessing checked to be identical to RPG's official code). All items are ranked, ties count against the model. Protocol and significance rules: [PROTOCOL.md](PROTOCOL.md). Every run behind these tables is a row of [runs.csv](runs.csv).

**Still running** (tables show what has finished): CDs second seed (currently Plain 1, No graph 0, Graph-TIGER 1 seeds), extra seeds of the placement study, LightGCN recipe runs, and two official-RPG seeds on Sports.

## 1. Main comparison (leave-one-out, R@5 / N@5 / R@10 / N@10)

| Model (seeds in brackets) | Beauty | Sports | Toys | CDs |
|---|---|---|---|---|
| SASRec (ours, full softmax) | .0462 / .0274 / .0722 / .0358 (3) | .0236 / .0131 / .0387 / .0179 (3) | .0472 / .0302 / .0672 / .0366 (3) | .0474 / .0268 / .0803 / .0374 (1) |
| EASE (ours) | .0303 / .0195 / .0501 / .0258 | .0235 / .0150 / .0383 / .0198 | .0356 / .0233 / .0548 / .0294 | - |
| TIGER (published) | .0454 / .0321 / .0648 / .0384 | .0264 / .0181 / .0400 / .0225 | .0521 / .0371 / .0712 / .0432 | .0492 / .0329 / .0748 / .0411 |
| RPG (published) | .0550 / .0381 / .0809 / .0464 | .0314 / .0216 / .0463 / .0263 | .0592 / .0401 / .0869 / .0490 | .0498 / .0338 / .0735 / .0415 |
| DiffGRM (published) | .0603 / .0414 / .0876 / .0502 | .0363 / .0245 / .0550 / .0305 | .0618 / .0455 / .0834 / .0524 | .0348 / .0228 / .0550 / .0293 |
| ACERec (published) | .0610 / .0433 / .0856 / .0513 | .0341 / .0238 / .0494 / .0287 | .0698 / .0514 / .0940 / .0591 | .0547 / .0370 / .0800 / .0451 |
| TopoGR (published) | .0620 / .0439 / .0883 / .0522 | .0374 / .0258 / .0554 / .0314 | .0712 / .0497 / .0995 / .0588 | .0531 / .0364 / .0783 / .0445 |
| RPG official code, Sentence-T5, our run | .0493 / .0346 / .0718 / .0418 (3) | .0286 / .0196 / .0428 / .0242 (1) | .0546 / .0379 / .0793 / .0459 (3) | - |
| Plain (our base, no graph) | .0487 / .0343 / .0699 / .0411 (3) | .0278 / .0193 / .0415 / .0237 (3) | .0542 / .0381 / .0790 / .0461 (3) | .0481 / .0329 / .0708 / .0402 (1) |
| No graph (item loss + EMA) | .0542 / .0376 / .0790 / .0456 (3) | .0331 / .0225 / .0498 / .0278 (3) | .0592 / .0405 / .0878 / .0498 (3) | pending |
| **Graph-TIGER** | .0616 / .0422 / .0925 / .0522 (3) | .0356 / .0235 / .0527 / .0290 (3) | .0629 / .0410 / .0952 / .0514 (3) | .0692 / .0455 / .1042 / .0568 (1) |
| Graph-TIGER + EASE (ensemble) | .0624 / .0429 / .0929 / .0527 | .0378 / .0254 / .0565 / .0315 | .0630 / .0415 / .0962 / .0522 | - |

## 2. Significance (paired bootstrap over users, seed-averaged; n.s. = 95% interval includes 0)

| Dataset | Comparison | Seeds | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|---|---|
| Beauty | Graph-TIGER vs Plain | 3 | +26.5% | +22.9% | +32.5% | +26.8% |
| Beauty | Graph-TIGER vs No graph | 3 | +13.6% | +12.2% | +17.1% | +14.4% |
| Beauty | No graph vs Plain | 3 | +11.4% | +9.6% | +13.1% | +10.8% |
| Sports | Graph-TIGER vs Plain | 3 | +27.9% | +21.7% | +26.8% | +22.3% |
| Sports | Graph-TIGER vs No graph | 3 | +7.5% | +4.7% | +5.8% | +4.3% |
| Sports | No graph vs Plain | 3 | +19.0% | +16.2% | +19.9% | +17.3% |
| Toys | Graph-TIGER vs Plain | 3 | +16.1% | +7.5% | +20.6% | +11.6% |
| Toys | Graph-TIGER vs No graph | 3 | +6.2% | +1.1% (n.s.) | +8.4% | +3.4% |
| Toys | No graph vs Plain | 3 | +9.3% | +6.3% | +11.3% | +7.9% |
| CDs | Graph-TIGER vs Plain | 1 | +43.9% | +38.3% | +47.1% | +41.1% |

**Against the best published value per metric** (bootstrap interval of our seed-averaged metric; published values treated as exact):

| Dataset | R@5 | N@5 | R@10 | N@10 |
|---|---|---|---|---|
| Beauty | -0.7% n.s. (TopoGR) | -3.9% n.s. (TopoGR) | +4.8% above (TopoGR) | -0.1% n.s. (TopoGR) |
| Sports | -4.7% below (TopoGR) | -8.9% below (TopoGR) | -4.9% below (TopoGR) | -7.6% below (TopoGR) |
| Toys | -11.7% below (TopoGR) | -20.2% below (ACERec) | -4.3% below (TopoGR) | -12.9% below (ACERec) |
| CDs | +26.6% above (ACERec) | +23.0% above (ACERec) | +30.2% above (ACERec) | +25.9% above (ACERec) |

## 3. Ablation: every combination of the three components (Beauty, R@10 / N@10)

| Added to the base model | Without graph (seeds) | With transition graph, 128-d (seeds) |
|---|---|---|
| none | 0.0699 / 0.0411 (3) | 0.0878 / 0.0493 (3) |
| item loss | 0.0779 / 0.0448 (3) | 0.0919 / 0.0512 (3) |
| EMA | 0.0715 / 0.0420 (3) | 0.0905 / 0.0510 (3) |
| item loss + EMA | 0.0790 / 0.0456 (3) | 0.0925 / 0.0522 (3) |

## 4. The two graphs as gated inputs at equal size (Beauty, test R@10 / N@10, validation N@10 in brackets)

| Setting | Transition graph | User-item graph (LightGCN) |
|---|---|---|
| gate only, 64-d | 0.0834 / 0.0473 [0.0605] (3) | 0.0819 / 0.0482 [0.0619] (3) |
| gate only, 128-d | 0.0878 / 0.0493 [0.0623] (3) | 0.0835 / 0.0491 [0.0629] (3) |
| 128-d + EMA | 0.0905 / 0.0510 [0.0634] (3) | 0.0849 / 0.0504 [0.0641] (2) |
| 128-d + item loss 0.1 + EMA | 0.0927 / 0.0512 [0.0629] (1) | 0.0848 / 0.0479 [0.0597] (1) |
| 128-d + item loss 0.3 + EMA | 0.0925 / 0.0522 [0.0642] (3) | 0.0814 / 0.0447 [0.0565] (3) |

With the gate alone the two graphs are close (64-d tie; at 128-d the transition graph leads on R@10 only, and validation N@10 slightly favours LightGCN). The fair comparison lets each graph choose its own training recipe on validation N@10 (rows 3 to 5; preliminary until all seeds finish). Current picks: Transition graph: item loss 0.3 + EMA, test 0.0925 / 0.0522 (3 seeds); LightGCN: EMA only, test 0.0849 / 0.0504 (2 seeds).

## 5. Where the graph enters (Beauty, no item loss or EMA, test R@10 / N@10, validation N@10 in brackets)

| Placement | Transition graph | LightGCN (64-d in A and B) |
|---|---|---|
| no graph | 0.0699 / 0.0411 [0.0561] (3) |  |
| A: re-rank the scores | 0.0808 / 0.0465 [0.0612] (1) | 0.0777 / 0.0442 [0.0592] (1) |
| B: inside the semantic IDs, weight 0.5 | 0.0789 / 0.0459 [0.0598] (1) | 0.0752 / 0.0433 [0.0595] (1) |
| B: inside the semantic IDs, weight 1.0 | 0.0835 / 0.0487 [0.0614] (1) | 0.0774 / 0.0442 [0.0598] (1) |
| C: gated input (128-d / 64-d) | 0.0878 / 0.0493 [0.0623] (3) | 0.0819 / 0.0482 [0.0619] (3) |

The gated input (C) is the best placement for both graphs. The weight in B is chosen on validation.

## 6. Temporal split (global time cut, 3 seeds, test R@10 / N@10)

| Dataset | Plain | Graph-TIGER | EASE | Graph-TIGER vs Plain (R@10, N@10) |
|---|---|---|---|---|
| Beauty | 0.0180 / 0.0088 | 0.0231 / 0.0113 | 0.0215 / 0.0116 | +28.1%, +27.3% |
| Sports | 0.0210 / 0.0107 | 0.0222 / 0.0107 | 0.0243 / 0.0128 | +5.6% (n.s.), -0.6% (n.s.) |
| Toys | 0.0204 / 0.0101 | 0.0246 / 0.0110 | 0.0215 / 0.0122 | +20.5%, +8.7% (n.s.) |

**Unseen items.** Reserving k of the top 10 for the best unseen items by content and graph neighbours (temporal split, seed 2024, overall R@10):

| Dataset | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| Beauty | 0.0240 | 0.0265 | 0.0281 | 0.0297 |
| Sports | 0.0223 | 0.0218 | 0.0222 | 0.0213 |
| Toys | 0.0253 | 0.0279 | 0.0297 | 0.0297 |

**Refresh** (seed 2024, R@10 / N@10). The fine-tuned and retrained models also see the validation window; the stale one does not. The retrain stops after a fixed number of epochs without validation selection, so it is a reference point, not an upper bound.

| Dataset | Stale | Graph swap only | Fine-tune 5 epochs | Full retrain |
|---|---|---|---|---|
| Beauty | 0.0240 / 0.0119 | 0.0304 / 0.0138 | 0.0562 / 0.0277 | 0.0625 / 0.0297 |
| Sports | 0.0223 / 0.0108 | 0.0215 / 0.0105 | 0.0301 / 0.0183 | 0.0267 / 0.0159 |
| Toys | 0.0253 / 0.0110 | 0.0275 / 0.0120 | 0.0370 / 0.0165 | 0.0398 / 0.0174 |

## 7. Cold start (Beauty, 605 held-out items, seed 2024)

| Method | R@10 among held-out items only (1059 users) | R@10 among all items, same users |
|---|---|---|
| Random | 0.0189 | - |
| Plain | 0.1020 | 0.0009 |
| Graph-TIGER, full score | 0.1992 | 0.0047 |
| Graph-TIGER, item head only | 0.2219 | - |
| Content neighbours | 0.2266 | - |
| Graph neighbours (imputed vectors) | 0.2200 | - |
| Content + graph neighbours | 0.2568 | - |

Unseen items almost never reach the global top 10, and a simple neighbour rule ranks them better than the generative model.

## 8. Cost (Beauty, idle L40S, seed 2024)

|  | Plain | Graph-TIGER |
|---|---|---|
| Parameters | 13.58M | 13.90M |
| Training to early stop | 9.1 min | 6.0 min |
| Batch scoring, ms per user | 0.052 | 0.055 |
| Single request p50 / p95, ms | 3.8 / 3.9 | 6.7 / 7.1 |

Re-scoring only the top 50 items of the item head (`--cands 50`) keeps 100.1 to 103.3% of full R@10 across the four datasets.

## 9. Agreement with official RPG (Sentence-T5 text)

| Dataset | Seeds | Official RPG, exhaustive (R@10 / N@10) | Our Plain | Exhaustive over graph decoding, R@10 |
|---|---|---|---|---|
| Beauty | 3 | 0.0718 / 0.0418 | 0.0699 / 0.0411 | +1.1 to +1.6% |
| Sports | 1 | 0.0428 / 0.0242 | 0.0415 / 0.0237 | +2.7% |
| Toys | 3 | 0.0793 / 0.0459 | 0.0790 / 0.0461 | +1.1 to +1.6% |

## Limits

- Graph-TIGER uses collaborative statistics from the training data; TIGER, RPG and ACERec use item text only.
- The item-loss weight, EMA decay, window and graph size were chosen on Beauty validation; earlier exploration also looked at Beauty test scores.
- Published numbers are treated as exact; they use their own seeds, decoding and in some cases a different text encoder.
- In leave-one-out the graph can use other users' interactions that are later than a test target; the temporal split removes this.
- Toys and Sports stay below the best published NDCG; EASE and its ensemble were not run on CDs.

## Run names in runs.csv

| Run | Meaning (flags of main.py) |
|---|---|
| plain | `--no-graph` |
| ours | `--dense-w 0.3 --ema 0.99` (transition graph is the default input) |
| nograph | `--no-graph --dense-w 0.3 --ema 0.99` |
| svd128 / svd64 | gated transition graph only (`--graph-emb` for the 64-d file) |
| lgcn256 / lgcn128 | gated LightGCN 64-d / 128-d only (256 refers to the batch size) |
| lgcn128_ours / _i01 / _ema | LightGCN 128-d with item loss 0.3 + EMA / 0.1 + EMA / EMA |
| abl_g_* / abl_n_* | with / without graph; `item` = item loss 0.3, `ema` = EMA, `i01` = item loss 0.1 + EMA |
| ab_lgcn_dense / ab_lgcn_ema | LightGCN 64-d with item loss / EMA, seed 2024 |
| ab_tok05 / ab_tok10 | Graph-TIGER plus graph inside the IDs (`--token-graph 0.5 / 1.0`) |
| pl_rerank_* | `--rerank-graph` with `--gammas` |
| pl_codes_* | `--no-graph --token-graph` weight 0.5 / 1.0 |
| t_* | temporal split (`--temporal 0.8,0.9`); `t_ft` fine-tuned, `t_retrain` retrained |
| c_* | cold-start split (`--holdout`) |
