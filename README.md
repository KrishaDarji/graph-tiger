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
| **RPG-style + LightGCN gated (6 seeds)** | Janam | **0.0767** | **0.0436** |
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
- We are not at the published state of the art yet (RPG 0.0809, ACERec 0.0841).

## Next steps

Tried and dropped: larger text encoders (Qwen3, BGE: no gain once the graph gate is on) and ACERec-style merger, intent token and alignment loss (worse than the plain model on Beauty). See `presentation/paper_repro/RESULTS.md`.

1. Tune the training recipe of the gated model (learning rate, batch size, dropout, sliding-window targets). Costs no parameters.
2. A better graph source: larger LightGCN, or a graph built from item-to-item transitions in the sequences.
3. Use the graph at scoring time as well (add a graph similarity term to the item score).
4. Cost table (parameters, training time, inference latency) against TIGER and ACERec, final tables with 5 seeds and paired tests, cold-start breakdown.

## Repository

- `src/` — data, evaluation, LightGCN (`lightgcn.py`) and the paper-style T5 TIGER (`src/tiger/`).
- `scripts/paper/` — paper-protocol experiments: `embed_items.py` (text vectors), `rqvae.py`, `rkmeans.py` (codes), `lightgcn5.py`, `train_t5.py` (TIGER), `pgen.py` (RPG-style model, `--graph gate` adds LightGCN), `ace.py` (ACERec-style components), `run_all.sh`, `summarize.py`.
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