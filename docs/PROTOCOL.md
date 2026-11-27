# Evaluation protocol

One evaluator (`src/evaluator.py`) produces every number of our models and baselines. Published numbers are quoted from their papers and never re-run.

## Data

Amazon Reviews 2014, 5-core (McAuley et al., SIGIR 2015; He & McAuley, WWW 2016).

| Dataset | Users | Items | Interactions |
|---|---:|---:|---:|
| Beauty | 22,363 | 12,101 | 198,502 |
| Sports and Outdoors | 35,598 | 18,357 | 296,337 |
| Toys and Games | 19,412 | 11,924 | 167,597 |
| CDs and Vinyl | 75,258 | 64,443 | 1,097,592 |

ACERec lists 1,022,334 interactions for CDs, which is exactly one fewer per user than the file contains.

## Preprocessing (as in RPG's official code)

- Item text: title, price, brand, feature, categories and description, with RPG's cleaning, encoded by Sentence-T5-base.
- Whitened PCA to 128 dimensions, then faiss `OPQ{m},IVF1,PQ{m}x8` (inner product) trained on training items only.
- With the library versions in `requirements.txt`, item order, item text, user sequences, training examples and semantic IDs are identical to the official code on all four datasets.

## Leave-one-out split

- Interactions are sorted by timestamp; the last item of each user is the test target, the second to last the validation target.
- History is the last 50 items. Longer training prefixes give additional sliding windows, trained on their last item.
- Both graphs are built from the training prefix only (all but the last two items of each user).

## Training and selection

- Per-dataset digits and learning rate follow RPG: Beauty 32 / 0.01, Sports 16 / 0.003, Toys 16 / 0.003, CDs 64 / 0.001. Batch 256, AdamW without weight decay, cosine schedule with 10,000 warm-up steps, at most 150 epochs, patience 20.
- Validation NDCG@10 chooses the checkpoint and every scoring weight (item head, re-ranking, EASE blend). Test is evaluated with the chosen settings.

## Ranking and metrics

- Full catalog, nothing removed from the ranking, ties count against the model.
- A second view with the user's history removed is stored as `test_filtered`. It is higher and is never mixed with the first.
- Recall and NDCG at 5, 10, 20, 50; MRR; coverage, Gini, entropy, novelty, intra-list diversity; Recall@10 by target popularity and history length.
- Published generative recommenders decode approximately (beam search or graph-constrained decoding). On the official RPG code, exhaustive ranking of the same model gives 1.1 to 2.7% higher test Recall@10 than its graph decoding.

## Significance

- Between our models: paired per-user bootstrap (2,000 resamples) on the same users and seeds (`scripts/summarize.py --ref`).
- Against a published number: bootstrap interval of our seed-averaged per-user metric; "above" or "below" means the 95% interval lies fully on one side, otherwise "n.s." (`scripts/claims.py`). Published values are treated as exact.

Published reference values (Recall@5 / NDCG@5 / Recall@10 / NDCG@10), checked against the papers' tables:

| | TIGER (2305.05065) | RPG (2506.05781) | ACERec (2602.13573) | TopoGR (2607.25216) | DiffGRM (2510.21805) |
|---|---|---|---|---|---|
| Beauty | .0454 / .0321 / .0648 / .0384 | .0550 / .0381 / .0809 / .0464 | .0610 / .0433 / .0856 / .0513 | .0620 / .0439 / .0883 / .0522 | .0603 / .0414 / .0876 / .0502 |
| Sports | .0264 / .0181 / .0400 / .0225 | .0314 / .0216 / .0463 / .0263 | .0341 / .0238 / .0494 / .0287 | .0374 / .0258 / .0554 / .0314 | .0363 / .0245 / .0550 / .0305 |
| Toys | .0521 / .0371 / .0712 / .0432 | .0592 / .0401 / .0869 / .0490 | .0698 / .0514 / .0940 / .0591 | .0712 / .0497 / .0995 / .0588 | .0618 / .0455 / .0834 / .0524 |
| CDs | .0492 / .0329 / .0748 / .0411 | .0498 / .0338 / .0735 / .0415 | .0547 / .0370 / .0800 / .0451 | .0531 / .0364 / .0783 / .0445 | .0348 / .0228 / .0550 / .0293 |

TIGER on CDs is the value in the RPG paper; DiffGRM on CDs is the value in the TopoGR paper. RPG's main numbers use a commercial text-embedding model.

## Temporal split

`--temporal 0.8,0.9`: training sequences and the graph use interactions before the 80% time quantile; validation is each user's first interaction between the 80% and 90% quantiles; test is the first interaction after the 90% quantile, with everything before it as history. Items never seen in training are cold targets by construction. Reported next to, never mixed with, the leave-one-out numbers.

## Cold-start split

`--holdout`: 5% of items (605 on Beauty, fixed random choice) are removed from all training sequences and from the graph; 1,059 test users have such a target. Held-out items get a graph vector predicted from their text (`scripts/impute_graph.py`); their true vectors are never used.

## Hardware and software

One NVIDIA L40S per run. Python 3.12, PyTorch 2.15.0, other versions as pinned in `requirements.txt`.
