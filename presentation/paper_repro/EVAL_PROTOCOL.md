# Evaluation protocol

One evaluator for every number we report: `src/evalkit.py`. Full notes with sources and file/line citations: `research_notes/eval_protocols.md`.

## What the papers do

- **No seen-item masking** in TIGER's paper text, RPG's official code (`genrec/evaluator.py`), ActionPiece, GRID, LETTER or the two public TIGER reimplementations. All rank the full catalog, leave-last-out (last item test, second-to-last validation), Recall and NDCG at 5 and 10. RPG and ACERec use history 50, TIGER and ActionPiece 20.
- TIGER's baseline rows (SASRec etc.) were copied from the S3-Rec repository, whose evaluation does mask seen items. So published generative-vs-baseline gaps are slightly conservative.
- On this dataset the target is never in the user's history (no repeated user-item pairs), so masking can only raise scores.

## Our inconsistency, and the fix

| | Stage 1 (full data, Vivek and Janam) | Stage 2 (paper protocol) |
|---|---|---|
| Ranking | beam output, train and validation items removed | full catalog, nothing removed |
| Matches the papers | no (inflated) | yes |

Stage 1 also stored the unmasked score in every `metrics_test.json` (`no_mask`). Re-reading those: the masking effect there is small, so the stage-1 conclusions do not change, but the unmasked numbers are the ones to compare with anything published.

| Stage 1 run (147,241 test users) | Masked R@10 / N@10 (old headline) | Unmasked R@10 / N@10 |
|---|---|---|
| Run 1, text only | 0.0334 / 0.0187 | 0.0323 / 0.0176 |
| Run 2, graph in the codes | 0.0336 / 0.0193 | 0.0325 / 0.0180 |
| Concatenation | 0.0340 / 0.0190 | 0.0327 / 0.0176 |
| MLP fusion | 0.0341 / 0.0194 | 0.0334 / 0.0186 |
| Learned scalar | 0.0330 / 0.0183 | 0.0322 / 0.0175 |
| Learned gate | 0.0307 / 0.0172 | 0.0300 / 0.0162 |

On the 5-core data our TIGER reproduction goes from R@10 0.0546 (raw) to about 0.0598 when seen items are removed from its beam output. That is a 10% effect, which is why a masked number must never be compared with a published raw number.

## Rules for every new result

1. **Primary** column: raw ranking over all 12,101 items, nothing removed, ties count against the model. This is the column to compare with TIGER, RPG and ACERec.
2. **Second view**: same scores with history items removed. Reported next to the first, never mixed.
3. Metrics: Recall@1/5/10/20/50, NDCG@5/10/20/50, MRR@10, MRR. Beyond accuracy: coverage@10, Gini and entropy of the recommended items, average popularity of recommended items, Recall@10 by target-popularity quartile, cold-item Recall@10 with 5% of items never seen in training. Cost: parameters, training minutes, scoring milliseconds per user.
4. Validation only chooses the checkpoint (NDCG@10) and the score-mixing weights; test is read once.
5. Graph inputs (LightGCN, transition SVD) are built from each user's training prefix only (all but the last two items).
6. Seeds: 5 per headline setting, paired per-user bootstrap against the reference model.
7. Training examples: every position of the training prefix is a target (as in RPG). History 50 (RPG, ACERec setting). Chronological order with a stable sort (43% of users have a timestamp tie on their last two items, so file order decides, as in RPG and ActionPiece).
8. Published numbers quoted in tables are labelled with their source and text encoder; they are not re-run by us.

## Added on 2026-11-12

- **Datasets**: Beauty, Toys, Sports and CDs (Amazon 2014, 5-core, same files and split for every model; chronological with a stable sort; history 50). Baselines run by us (plain RPG-style, SASRec with full-softmax loss, nearest-neighbour rules) go through the same evaluator; published numbers are quoted with their encoder and not re-run. Model code lengths and learning rates follow RPG's per-dataset table unless stated.
- **More metrics** (all in `result.json`): Recall@1/5/10/20/50, NDCG@5/10/20/50, MRR; coverage@10, Gini, entropy, novelty (mean self-information of recommended items), intra-list diversity of the top 10 (cosine in text space), Recall@10 by target-popularity quartile and by history length (<=5, 6-10, 11-20, >20); parameters, training time, scoring time per user.
- **Cold start**: separate protocol with global, cold-pool and warm-pool views and nearest-neighbour baselines; see `COLD_START.md`.
- **Overfitting guard**: `scripts/paper/overfit_report.py` reports validation-vs-test agreement across all configurations tried; the final choices follow validation scores (for the Beauty screening runs the test scores were visible too, see the limits in `RESULTS.md`; Toys, Sports and CDs were not used for any choice); per-dataset tuning was dropped when it was within seed noise. A variant that improved validation by 2.4% (in-batch negatives with popularity correction) gave no gain on test (N@10 0.0533 vs 0.0537) and was not adopted.
- **Tie handling**: ties count against the model (the target is ranked below every item with an equal score).
