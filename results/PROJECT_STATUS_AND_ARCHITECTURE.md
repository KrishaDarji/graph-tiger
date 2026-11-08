# Graph-TIGER Project Status and Architecture

Last updated: 2026-11-06. All planned runs are finished.

## 1. Project goal

This project studies whether collaborative user–item graph information improves a TIGER-style generative recommender on Amazon Beauty.

> Does adding graph-derived collaborative information improve next-item recommendation compared with content-only TIGER?

TIGER treats recommendation as generative retrieval: a Transformer predicts an item's semantic ID, a short tuple of discrete codes derived from item content, from a user's history.

## 2. Dataset and evaluation

Source: Amazon Beauty 2014 reviews and metadata, not 5-core filtered.

- Interactions: 786,049
- Users: 147,241
- Items: 163,257
- Training: 491,567 interactions; validation: 147,241; test: 147,241

For each user, the second-to-last item is validation and the last is test. The split is per user, not by a global date, so training can include interactions later than another user's test item.

Metrics are Recall@5/10 and NDCG@5/10, from `src/evaluation.py`. Models are selected on validation NDCG@10, and each model's test split is evaluated once.

## 3. Architecture

See `presentation/architecture/graph_tiger_architecture.png`.

```text
Product text -> Sentence-T5 (768-D) ---------------+-> RQ-VAE -> semantic IDs + lookup table
                                                   |
User-item graph -> LightGCN (768-D), Run 2 only ---+

User history as semantic IDs -> T5 -> constrained beam search -> lookup table -> top 10
```

| Component | Role | Training |
|---|---|---|
| Sentence-T5 (`sentence-t5-base`) | Encodes title, brand, categories and description | Pretrained, not trained |
| LightGCN | Collaborative user and item vectors, from training interactions only | Trained once per dimension |
| RQ-VAE | Compresses each item vector into 3 codes, plus a 4th code that separates remaining collisions | One per run, 225 epochs |
| T5 encoder-decoder | Predicts the next item's semantic ID | One per run, same settings |
| Lookup table | Translates between items and codes, and limits beam search to real items | Not trained |

## 4. Experiments

| Experiment | RQ-VAE input | Status |
|---|---|---|
| Run 1: simple TIGER | 768-D content | Done |
| Design A: re-ranking | None: frozen Run 1, with LightGCN re-scoring its top 30 candidates | Done, 768-D model selected on validation |
| Run 2: graph-enhanced TIGER | 768-D content + α × 768-D LightGCN, normalized, α = 1 | Done |

- **Design B compares Run 2 with Run 1.** The RQ-VAE input size and T5 settings match, so the graph is the main difference.
- **Design A compares re-ranked Run 1 with Run 1.** α is chosen on validation.
- Items with no training interactions get a graph term of 0 in Run 2.

## 5. Results so far

### LightGCN (64-D) test metrics, under matching masks

| Masking on test | Model | R@5 | R@10 | N@5 | N@10 |
|---|---|---:|---:|---:|---:|
| Train + validation | LightGCN | 0.01256 | 0.01978 | 0.00799 | 0.01032 |
| Train + validation | Popularity | 0.00482 | 0.00807 | 0.00263 | 0.00367 |
| None | LightGCN | 0.00989 | 0.01816 | 0.00555 | 0.00821 |
| None | Content-aware raw-ID Transformer, epoch 63 (Hazel run) | 0.00833 | 0.01304 | 0.00547 | 0.00699 |

The data was rebuilt on VCL with the repo's scripts, and the counts and popularity baseline match. The Transformer was not re-evaluated on VCL. It used the older TF-IDF embeddings.

### LightGCN (768-D) test metrics

Selected on validation NDCG@10 at epoch 195 of 200, with the same settings as the 64-D model apart from size. It beats the 64-D model on validation as well as on test. Epoch 195 is close to the limit, so it may not have fully converged.

| Masking on test | Model | R@5 | R@10 | N@5 | N@10 |
|---|---|---:|---:|---:|---:|
| Train + validation | LightGCN 768-D | 0.01925 | 0.02881 | 0.01290 | 0.01597 |
| Train only | LightGCN 768-D | 0.01863 | 0.02860 | 0.01206 | 0.01527 |
| None | LightGCN 768-D | 0.01385 | 0.02496 | 0.00734 | 0.01091 |

### TIGER code check on 5-core Beauty

This uses a different, smaller dataset and no graph. Test Recall@10 was 0.0580 and NDCG@10 was 0.0301, against 0.0594 and 0.0321 reported by the upstream repo and 0.0648 and 0.0384 in the paper. Details are in `presentation/tiger_5core/RESULTS.md`.

### Run 1 and Design A test metrics

Train + validation mask. Run 2 is added below.

| Model | R@5 | R@10 | N@5 | N@10 |
|---|---:|---:|---:|---:|
| Run 1: simple TIGER, epoch 89 | 0.02245 | 0.03338 | 0.01519 | 0.01870 |
| Design A: 768-D LightGCN, α = 1 (selected) | 0.02292 | 0.03399 | 0.01556 | 0.01911 |
| Design A: 64-D LightGCN, α = 0.5 (not selected) | 0.02244 | 0.03354 | 0.01532 | 0.01888 |
| Run 2: graph-enhanced TIGER, epoch 113 | 0.02323 | 0.03363 | 0.01596 | 0.01932 |

Validation chose the 768-D model, with NDCG@10 of 0.02692 against 0.02504 for 64-D and 0.02381 with no re-ranking. Both versions were evaluated on test, and the choice between them was made on validation alone. The test gain (2.2% on NDCG@10) is much smaller than the validation gain (13%).

Run 1 Recall@10 by the target item's training interactions: 0 interactions (20.7% of targets) 0.00003, 1-4 (25.2%) 0.00030, 5 or more (54.1%) 0.06156.

### Answer to the research question

On test, the graph gives TIGER a small gain on the ranking metrics, from one training seed per model. Paired bootstrap, compared with Run 1:

- **Run 2:** N@10 +3.3% (95% interval +0.00010 to +0.00110) and N@5 +5.1%. R@10 is unchanged (+0.8%, interval includes 0). Its gain within Run 1's 99-epoch budget is not distinguishable from 0.
- **Design A:** N@10 +2.2% (interval +0.00013 to +0.00070) and R@10 +1.8%.
- **Run 2 vs Design A:** no measurable difference.
- **Rare items:** Run 2 raises Recall@10 from 0.00030 to 0.00156 for targets with 1-4 training interactions. Neither method helps cold items.

Full tables and limits are in `presentation/tiger_full/RESULTS.md`.

## 6. Changes from the proposal

| Proposal | Current plan | Why |
|---|---|---|
| Project content and graph to a shared lower dimension | 768-D LightGCN, no projection | Keeps the RQ-VAE input the same as plain TIGER |
| Projected-content control | Not run | No projection is involved. Its 64-D files and semantic IDs are kept |
| Graph affects semantic IDs only | Design A re-ranking added | Follows advice to keep the TIGER block fixed |
| Hazel | VCL (NVIDIA L40S) | Current compute |

Slide 5 of the update deck shows LightGCN entering the Transformer. That design is not being run.

## 7. Known limitations

- 20.7% of test targets have no training interactions, and 61% of items have 1–4.
- Validation scores run well above test scores for both LightGCN and the 5-core TIGER check.
- Results come from one seed each.
- Reducing the T5 validation sample to 20,000 users adds some noise to checkpoint selection.

## 8. Fusion experiments (2026-11-07)

Two more ways to combine content and LightGCN before the RQ-VAE were trained with the same settings as Run 2: concatenation (1536-D) and a fusion MLP (768-D). Results are in `presentation/tiger_fusion/RESULTS.md`.

## 9. Remaining work

1. Extra training seeds, since several gaps between graph models are within seed noise.
2. Optionally try other α values, chosen on validation.
