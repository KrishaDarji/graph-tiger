# Graph-TIGER Configuration and Differences from the Paper

## Data

```yaml
Dataset: Amazon Beauty 2014, not 5-core filtered
Interactions: 786,049
Users: 147,241
Items: 163,257
Train: 491,567
Validation: 147,241
Test: 147,241
```

T5 training uses 344,326 sliding windows over the training histories.

## Current configuration

| Component | Setting |
|---|---|
| Compute | VCL, NVIDIA L40S |
| Content encoder | `sentence-t5-base`, 768-D |
| Content text | Title, brand, categories, description. `salesRank` excluded as a possible popularity leak |
| LightGCN | 2 layers, 768-D for fusion and 64-D for re-ranking, BPR, Adam lr 1e-3, batch 4096, reg 1e-4, up to 200 epochs |
| RQ-VAE | Layers 512-256-128-64, 32-D latent, 3 codebooks of 256, lr 1e-3, batch 1024, 225 epochs |
| Semantic IDs | 3 codes plus a 4th that separates collisions, chosen by best collision rate |
| T5 | 4 encoder and 4 decoder layers, d_model 128, 6 heads, d_kv 64, d_ff 1024, dropout 0.1 |
| T5 training | Adam lr 1e-4, batch 256, up to 200 epochs, early stopping after 10 epochs |
| Selection | NDCG@10 on 20,000 sampled validation users |
| Prediction | Beam search of width 30, limited to real items |
| Fusion, Run 2 | normalize(content + α × normalized graph), α = 1, graph term 0 for cold items |
| Fusion, concatenation | [normalized content, normalized graph], 1536-D, graph half 0 for cold items. Same RQ-VAE layers, so the first layer maps 1536 to 512 |
| Fusion, MLP | 1536 → 1024 → 768, ReLU, unit-norm output. Trained alone to rebuild both halves, each half's error divided by its variance, cold graph halves left out. Adam lr 1e-4, batch 1024, 5% of items held out, best of 427 epochs at 397. Then frozen |

The RQ-VAE runs 225 epochs because that matches the roughly 36,000 training steps of the 5-core check.

## Code sources

The RQ-VAE and T5 code comes from [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER) (MIT), an unofficial implementation. Local changes are listed in `third_party/xiaolongtao_tiger/UPSTREAM.md`.

## Differences from the TIGER paper

| Component | TIGER paper | This project |
|---|---|---|
| Dataset | 5-core Beauty, about 22K users and 12K items | Unfiltered Beauty, 147K users and 163K items |
| Implementation | Authors' code | Unofficial implementation |
| Content text | See paper | Title, brand, categories, description |
| Graph information | None | LightGCN, in Run 2 and in re-ranking |
| Validation during training | See paper | Sample of 20,000 users |

Published TIGER scores are reference values. They can't be compared directly with results on this dataset.

## Earlier prototype

The earlier content-aware Transformer predicted raw item IDs over 163,257 classes, using 64-D TF-IDF/SVD embeddings, and ran on Hazel. It reached Recall@10 0.01304 and NDCG@10 0.00699 with no masking.

`scripts/prepare_metadata.py`, which built those embeddings, joins each description string character by character, and TF-IDF then drops the single characters. Descriptions therefore had no effect on those embeddings. The script is unchanged so that result stays reproducible, and current runs use `scripts/embed_items_st5.py`.
