# LightGCN results (Amazon Beauty, VCL run)

Config: 2 layers, 64-D, Adam lr 1e-3, batch 4096, reg 1e-4, 200 epochs, seed 42.
Selected on validation NDCG@10 at epoch 190 (validation NDCG@10 0.01900, Recall@10 0.03394).
Runtime: about 7 minutes on one NVIDIA L40S.

## Test metrics, compared under matching masks

| Masking on test | Model | R@5 | R@10 | N@5 | N@10 |
|---|---|---:|---:|---:|---:|
| Train + validation (primary) | LightGCN | 0.01256 | 0.01978 | 0.00799 | 0.01032 |
| Train + validation (primary) | Popularity | 0.00482 | 0.00807 | 0.00263 | 0.00367 |
| Train only | LightGCN | 0.01204 | 0.01963 | 0.00752 | 0.00998 |
| Train only | Popularity (existing baseline) | 0.0048 | 0.0080 | 0.0026 | 0.0036 |
| None | LightGCN | 0.00989 | 0.01816 | 0.00555 | 0.00821 |
| None | Content-aware raw-ID Transformer (existing baseline, epoch 63) | 0.00833 | 0.01304 | 0.00547 | 0.00699 |

The Transformer number comes from the earlier Hazel run. Its checkpoint and split
files were not available on VCL, so it was not re-evaluated.

## Cold items

| Training interactions | Items |
|---|---:|
| 0 (cold) | 43,349 |
| 1-4 | 99,639 |
| 5+ | 20,269 |

20.7% of test targets and 15.6% of validation targets are cold. Test metrics on
non-cold targets only: R@10 0.02494, N@10 0.01301.

## Recommendation for fusion (not implemented)

- Set the graph term to 0 where `cold_item_mask` is true, so those items use content
  only. Their stored vectors have no graph signal: they received only negative-sample
  and L2 updates, and message passing leaves them at e0 / 3.
- L2-normalize the graph vectors of non-cold items first. Their mean norm is 2.74,
  while content vectors are unit length, so alpha is not comparable without this.
- Load content from `item_content_semantic_ids.pkl`. The Transformer's content matrix
  fills only items present in train, which leaves cold items with all zeros there.
