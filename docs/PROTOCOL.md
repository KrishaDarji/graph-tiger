# Evaluation protocol

One evaluator (`src/evaluator.py`) produces every number in this repository. Baselines are run through it; published numbers are quoted from their papers and never re-run.

## Data

Amazon Reviews 2014, 5-core (McAuley et al., SIGIR 2015; He & McAuley, WWW 2016), the files used by TIGER, RPG and ACERec.

| Dataset | Users | Items |
|---|---:|---:|
| Beauty | 22,363 | 12,101 |
| Sports and Outdoors | 35,598 | 18,357 |
| Toys and Games | 19,412 | 11,924 |

Item text: Sentence-T5-base on `Title. Price. Brand. Categories`, reduced to 128 dimensions with PCA.

## Leave-one-out split (the protocol of the published papers)

- Interactions are sorted by timestamp with a stable sort; the last item of each user is the test target, the second to last the validation target.
- History is the last 50 items. Every position of the training prefix is a training target.
- The transition graph is built from each user's training prefix only (all but the last two items).
- Validation chooses the checkpoint (NDCG@10), the dense-score weight and the EASE blend weight. Test is read once.

## Ranking

- Full catalog, nothing removed from the ranking, ties count against the model. This is the column to compare with TIGER, RPG, ACERec and later papers.
- A second view with the user's history removed is stored next to it (`test_filtered`). It is higher and is never mixed with the first.
- Metrics: Recall and NDCG at 5, 10, 20, 50; MRR; coverage, Gini, entropy, novelty, intra-list diversity; Recall@10 by target-popularity quartile and history length.

## Significance

- Seeds 2024 to 2026 (or more). Paired per-user bootstrap (2,000 resamples) between two of our models on the same users and seeds (`scripts/summarize.py --ref`).
- Against a published number: bootstrap interval of our seed-averaged per-user metric; "above" or "below" means the 95% interval lies fully on one side of the published value, otherwise "n.s." (`scripts/claims.py`). Published values are treated as exact, so their own seed noise is ignored.

Published reference values (Recall@5 / NDCG@5 / Recall@10 / NDCG@10, same protocol, Sentence-T5 text):

| | RPG (2506.05781) | ACERec (2602.13573) | TopoGR (2607.25216) | DiffGRM (2510.21805) |
|---|---|---|---|---|
| Beauty | .0550 / .0381 / .0809 / .0464 | .0610 / .0433 / .0856 / .0513 | .0620 / .0439 / .0883 / .0522 | .0603 / .0414 / .0876 / .0502 |
| Sports | .0314 / .0216 / .0463 / .0263 | .0341 / .0238 / .0494 / .0287 | .0374 / .0258 / .0554 / .0314 | .0363 / .0245 / .0550 / .0305 |
| Toys | .0592 / .0401 / .0869 / .0490 | .0698 / .0514 / .0940 / .0591 | .0712 / .0497 / .0995 / .0588 | n/a |

## Temporal split (no future information in the graph)

`--temporal 0.8,0.9`: training sequences and the graph use interactions before the 80% time quantile; validation is each user's first interaction between the 80% and 90% quantiles; test is the first interaction after the 90% quantile, with everything before it as history. Items never seen in training are cold targets by construction (14 to 20% of test targets). Reported next to, never mixed with, the leave-one-out numbers.

## Cold-start split

`--holdout`: 5% of items (605 on Beauty, fixed random choice) are removed from all training sequences and from the graph; 1,228 test users have such a target. Three views from the same scores: global rank among all items, rank among the held-out items only, rank of warm targets among warm items only. Held-out items get a graph vector predicted from their text (`scripts/impute_graph.py`); their true vectors are never used.
