# Graph-TIGER: Amazon Beauty Recommendation

**Idea.** Modern "generative" recommenders (TIGER, RPG) give every product a short code made from its text and predict the next product's code for a user. They never see who bought what together. We add that missing signal with a graph model (LightGCN) and test whether it beats the published models.

## The pieces, in plain words

| Piece | What it is | Why we use it |
|---|---|---|
| **TIGER** (Google, NeurIPS 2023) | Gives each product a 4-part code ("semantic ID") built from its title, brand and category. A small Transformer reads a user's past codes and writes the next product's code one part at a time, like writing the next word. | Similar products get similar codes, so the model can generalize and work with new products. It is the baseline we must reproduce. |
| **RQ-VAE / RK-Means** | Two ways to turn a product's text vector into its code. RQ-VAE trains a small network. RK-Means repeats k-means on the leftovers, with nothing to tune. | The code quality limits the whole recommender. RK-Means was better and more stable for us. |
| **RPG** (Meta, KDD 2025) | Same idea with a much longer code (32 parts) predicted all at once instead of one by one. | Faster, and published as clearly better than TIGER (Recall@10 0.081 vs 0.065). |
| **LightGCN** | Learns from purchase history: it draws a graph of users and the products they bought, and gives products bought by the same people similar vectors. | Text cannot tell that two different products are often bought together. This can. |
| **Gated fusion** | Adds the LightGCN vector to a product's text-based input, scaled by a learned dial between 0 (ignore the graph) and 1 (use it fully), set per product and per dimension. | The graph helps for well-known products and is useless for new ones. The dial lets the model decide. |

Key lesson from our two stages: putting the graph into the **codes** hurt or did nothing, while feeding it to the **model input** through a gate helped.

## What we did

1. **Full-data study (Vivek, Janam).** Our own split of 147,241 users and 163,257 items. TIGER on text only (Run 1), text plus graph in the codes (Run 2, concatenation, MLP, learned scalar, learned gate), and re-ranking with LightGCN. Putting the graph into the codes helped little (best: MLP fusion, +4% NDCG@10) and the learned gate was 8% worse. Details: `presentation/tiger_full`, `tiger_fusion`, `tiger_fusion_learned`.
2. **Paper-protocol study (Janam).** The published TIGER numbers use a different dataset version (5-core, 22,363 users), so stage 1 cannot be compared to them. We rebuilt TIGER from the paper on that data, added Meta's official RPG as a stronger baseline, and tested LightGCN inside RPG. Details: `presentation/paper_repro/RESULTS.md`.

## Results (test, Recall@10 / NDCG@10)

5-core Beauty, the paper's setting:

| Model | Who | R@10 | N@10 |
|---|---|---:|---:|
| TIGER, paper (3-seed mean) | Google | 0.0642 | 0.0374 |
| TIGER, our T5 + RQ-VAE (3 seeds) | Janam | 0.0546 | 0.0294 |
| TIGER, upstream code check (1 seed) | Vivek | 0.0580 | 0.0301 |
| TIGER, our T5 + RK-Means (3 seeds) | Janam | 0.0583 | 0.0310 |
| RPG, official code (3 seeds) | Meta, run by Janam | 0.0736 | 0.0426 |
| RPG-style, no graph (6 seeds) | Janam | 0.0725 | 0.0418 |
| RPG-style + LightGCN gated (6 seeds) | Janam | 0.0767 | 0.0436 |
| **RPG-style + transition graph + dense loss + EMA (5 seeds)** | Janam | **0.0927** | **0.0527** |
| RPG, paper (OpenAI text embeddings) | Meta | 0.0809 | 0.0464 |

Project data (147,241 users, not comparable with the table above), one seed each:

| Model | Who | R@10 | N@10 |
|---|---|---:|---:|
| Run 1: text only | Vivek | 0.0334 | 0.0187 |
| Run 2: graph added in the codes | Vivek | 0.0336 | 0.0193 |
| Concatenation / MLP fusion | Vivek | 0.0340 / 0.0341 | 0.0190 / 0.0194 |
| Learned scalar / learned gate | Janam | 0.0330 / 0.0307 | 0.0183 / 0.0172 |

- Our TIGER reaches about 85% of the paper, and so do the paper's own ID ablations (random and LSH IDs), so the gap is in unstated training details, not the codes. Seven fixes were tried without effect (`RESULTS.md`).
- Gated LightGCN improves our RPG-style model by 5.8% Recall@10 (p = 0.0014, better in 6 of 6 seeds) and edges past official RPG with the same text encoder (Sentence-T5-base).
- The final model (5 seeds, raw full ranking, same protocol as the papers) is above the published RPG (0.0809 / 0.0464) and ACERec (0.0841 / 0.0501) on Recall and NDCG at 5 and 10, with 13.9M parameters and about 0.2 ms per user. It uses collaborative statistics from the training data, which those content-only models do not. Details, ablations, cost and the cold-start limit: `presentation/paper_repro/RESULTS.md`; evaluation rules: `presentation/paper_repro/EVAL_PROTOCOL.md`.

Other datasets (test, raw full ranking, R@10 / N@10; Sports and Toys 5 seeds, CDs 3 seeds): Sports 0.0553 / 0.0301 (RPG paper 0.0463 / 0.0263, ACERec 0.0493 / 0.0280); Toys 0.0962 / 0.0547 (RPG 0.0869 / 0.0490, ACERec 0.0916 / 0.0579, so ACERec is ahead on N@10 and at cutoff 5); CDs 0.0956 / 0.0499 (RPG 0.0735 / 0.0415, TIGER 0.0748 / 0.0411). Plain RPG-style, SASRec and cold-start numbers: `presentation/paper_repro/RESULTS.md`, `COLD_START.md`.

## Next steps

Tried and dropped: larger text encoders (no gain once a graph input is used), ACERec-style merger, intent token and alignment loss (worse on Beauty), trainable or dropped-out graph vectors.

1. Cold start: items never seen in training are almost never recommended (Recall@10 below 1%). A content-plus-graph nearest-neighbour path ranks new items against each other best (cold-pool Recall@10 0.27 vs 0.23 for our model); a deployable system needs it as a separate slot (`COLD_START.md`).
2. Toys NDCG gap to ACERec (N@5 -11.5%, N@10 -5.5%) is unsolved; CDs needs a like-for-like baseline and longer training (all seeds hit the epoch cap).
3. Graph refresh needs fine-tuning: swapping in a new graph without retraining does not work (`RESULTS.md`).
4. Learn the transition graph end to end instead of a fixed SVD.

## Repository

- `src/` — data, evaluation (`evalkit.py` is the one evaluator for all current results), LightGCN (`lightgcn.py`) and the paper-style T5 TIGER (`src/tiger/`).
- `scripts/paper/` — paper-protocol experiments: `embed_items.py` (text vectors), `rqvae.py`, `rkmeans.py` (codes), `lightgcn5.py`, `train_t5.py` (TIGER), `pgen.py` (RPG-style model, `--graph gate` adds LightGCN), `gen.py` (current model: shared evaluator, dense loss, EMA, cold-start holdout), `svd_graph.py` (transition-graph item vectors), `baselines.py`, `sasrec.py` (full-softmax SASRec through the same evaluator), `cold_baselines.py`, `impute_graph.py`, `overfit_report.py`, `final_report.py` (5-seed tables and paired tests), `ace.py` (ACERec-style components), `run_all.sh`, `summarize.py`.
- `scripts/` (rest) — full-data experiments: data prep, `run_tiger_*.sh`, `run_lightgcn.py`, fusion builders, `learn_fusion.py`, `hf_sync.py` (private Hugging Face sync, configured in `.env`).
- `third_party/xiaolongtao_tiger/` — upstream RQ-VAE and T5 code used for Vivek's runs (changes in `UPSTREAM.md`).
- `presentation/` — result tables and logs. Large files (embeddings, checkpoints) are not in git.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt faiss-cpu
python scripts/check_device.py
```

## Run

Paper-protocol experiments (5-core Beauty, one GPU, about 8 hours in total). It downloads the data, and skips finished steps when re-run. Use `screen` so it survives a disconnect:

```bash
screen -dmS paper bash -c 'bash scripts/paper/run_all.sh > logs_paper.txt 2>&1'
```

Official RPG baseline (separate environment, its own repo https://github.com/facebookresearch/RPG_KDD2025): set `sent_emb_model: sentence-transformers/sentence-t5-base`, `sent_emb_dim: 768`, `sent_emb_pca: 128` in `genrec/models/RPG/config.yaml`, then
`python main.py --category=Beauty --lr=0.01 --temperature=0.03 --n_codebook=32 --num_beams=20 --n_edges=200 --propagation_steps=3`.

Full-data experiments on VCL (the queue resumes if re-run):

```bash
python scripts/prepare_data.py --download
python scripts/create_split.py
python scripts/embed_items_st5.py --dataset full
python scripts/prepare_tiger_full.py
python scripts/run_lightgcn.py --dim 768 --output-dir results/lightgcn_768
screen -dmS tiger_queue bash -c 'scripts/run_tiger_queue.sh'
screen -dmS tiger_graph768 bash -c 'scripts/run_tiger_graph768.sh'
```

Progress is in `results/tiger_full/queue.log`. LightGCN: `screen -dmS lightgcn bash -c 'scripts/run_lightgcn.sh > results/lightgcn_console.log 2>&1'`; outputs in `results/lightgcn/` (row `i` of `item_emb.npy` is `item_idx` `i`), results in `presentation/lightgcn/RESULTS.md`. Models and results can be pushed to or pulled from the team's private Hugging Face repo with `python scripts/hf_sync.py push|pull <paths>`.

## Earlier work (kept for reference)

A TF-IDF semantic-ID prototype (`scripts/prepare_metadata.py`) and a small raw-ID Transformer (`src/tiger_small.py`, `scripts/run_tiger_small.py`) came first and are superseded. A 5-core code check of Vivek's pipeline is in `presentation/tiger_5core/RESULTS.md`. Status and settings of the full-data runs: `results/PROJECT_STATUS_AND_ARCHITECTURE.md`, `results/configuration_and_deviations.md`.