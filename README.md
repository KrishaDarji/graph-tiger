# Graph-TIGER: Amazon Beauty Recommendation

This project reproduces a lightweight TIGER-style generative recommender on the Amazon Beauty dataset and studies whether collaborative graph information improves it.

## Current scope

The main research model is TIGER, and the graph model is an extension. Current status and results are in `results/PROJECT_STATUS_AND_ARCHITECTURE.md`, and settings are in `results/configuration_and_deviations.md`.

| Experiment | RQ-VAE input | Purpose |
|---|---|---|
| Run 1: simple TIGER | 768-D Sentence-T5 content | Baseline |
| Run 2: graph-enhanced TIGER | 768-D content + alpha x 768-D LightGCN | Graph effect on semantic IDs |
| Design A: re-ranking | None: frozen Run 1 re-scored with 64-D LightGCN | Graph effect with TIGER fixed |
| Fusion: concatenation | 1536-D: [content, LightGCN], each normalized | Fusion method |
| Fusion: MLP | 768-D from an MLP trained to rebuild content and LightGCN | Fusion method |

Fusion results are in `presentation/tiger_fusion/RESULTS.md`.

The popularity baseline, the earlier Transformer and standalone LightGCN are supporting baselines.

## Implementation files

### Data and preprocessing

- `scripts/prepare_data.py` — downloads and preprocesses Amazon Beauty interactions.
- `scripts/create_split.py` — creates the chronological train/validation/test split.
- `src/data.py` — data loading, ID encoding, and chronological splitting utilities.
- `scripts/prepare_metadata.py` — builds the older TF-IDF embeddings and codes used by the earlier prototype.
- `scripts/embed_items_st5.py` and `src/item_text.py` — encode item text with Sentence-T5.
- `scripts/prepare_tiger_full.py` — writes the existing split in the T5 input format.
- `scripts/build_fused_768.py` — builds the 768-D content + graph embeddings for Run 2.
- `scripts/build_fused_embeddings.py` — builds the 64-D projected files, which the current plan does not use.
- `scripts/build_concat_1536.py` — builds the 1536-D concatenated input for the fusion experiment.
- `scripts/build_mlp_fused_768.py` — trains the fusion MLP and writes its 768-D output.
- `scripts/run_tiger_fusion.sh` — trains TIGER on both fusion inputs at once.
- `scripts/hf_sync.py` — pushes and pulls data and results through the team's private Hugging Face repo, configured in `.env`.

### TIGER with semantic IDs

- `third_party/xiaolongtao_tiger/` — upstream RQ-VAE and T5 code, with changes listed in `UPSTREAM.md`.
- `scripts/run_tiger_model.sh` — trains one model: RQ-VAE, semantic IDs, T5, backup.
- `scripts/run_tiger_queue.sh` and `scripts/run_tiger_graph768.sh` — run the queued experiments.
- `scripts/run_tiger_t5.py` and `src/tiger_eval.py` — T5 training, lookup table, constrained beam search and scoring.
- `scripts/rerank_tiger_lightgcn.py` — Design A re-ranking.
- `scripts/analyze_semantic_ids.py` — reports collisions and codebook use.
- `scripts/prepare_tiger_5core.py` and `scripts/run_tiger_5core.sh` — the 5-core code check.
- `scripts/draw_architecture.py` — draws the architecture figure.

### Earlier raw-ID prototype

- `src/tiger_small.py` — defines the lightweight Transformer architecture.
- `scripts/run_tiger_small.py` — trains the content-aware next-item model.
- `scripts/run_tiger_small.sbatch` — Hazel Slurm training script with checkpoint/resume support.
- `scripts/evaluate_tiger_checkpoint.py` — evaluates a saved checkpoint using Recall and NDCG.

### Evaluation and graph extension

- `src/evaluation.py` — implements Recall@K and NDCG@K.
- `src/lightgcn.py` — LightGCN model and normalized user–item graph construction.
- `scripts/run_lightgcn.py` — trains the planned graph-based extension.
- `scripts/run_popularity_baseline.py` — popularity reference baseline.


## Run the current experiments on VCL

Prepare the inputs, then start the queue in `screen`. It doesn't need an open session, and it resumes if re-run. The graph model starts after Run 1 and 768-D LightGCN finish.

```bash
python scripts/embed_items_st5.py --dataset full
python scripts/prepare_tiger_full.py
python scripts/run_lightgcn.py --dim 768 --output-dir results/lightgcn_768
screen -dmS tiger_queue bash -c 'scripts/run_tiger_queue.sh'
screen -dmS tiger_graph768 bash -c 'scripts/run_tiger_graph768.sh'
```

Progress is in `results/tiger_full/queue.log`. Run 1 re-ranking needs `results/lightgcn` from the 64-D LightGCN run. A file `results/tiger_full/SKIP_<name>` drops a model from the queue, and `queue.log` still shows it as done.

## Build the earlier TF-IDF representations and codes

After the interaction data has been prepared:

```bash
python scripts/prepare_metadata.py
```

This creates `data/processed/item_content_semantic_ids.pkl`. It is a
lightweight, reproducible approximation of TIGER's semantic-ID stage using
TF-IDF, SVD, and residual quantization. The later experiment will use these
representations in the content-only and graph-enhanced models.

## Run the earlier raw-ID prototype

On the MacBook:

```bash
python scripts/run_tiger_small.py --epochs 3 --limit 50000 --eval-limit 1000

# Content-aware TIGER prototype
python scripts/run_tiger_small.py --epochs 3 --limit 50000 --eval-limit 1000 --use-content
```

For a full run on a CUDA machine:

```bash
python scripts/run_tiger_small.py --epochs 5 --batch-size 256 --eval-limit 0
```

On Hazel, copy the project and environment to the cluster, edit the account,
partition, and module setup in `scripts/run_tiger_small.sbatch`, then submit:

```bash
mkdir -p logs
sbatch scripts/run_tiger_small.sbatch
```

## Train LightGCN on VCL

Requires the split files and content file in `data/processed`. VCL has no job
scheduler, so run inside `screen`. The script resumes from `results/lightgcn/last.pt`
if a session drops. It refuses to start once `metrics_test.json` exists, so that the
test split is only evaluated once per output directory.

```bash
screen -dmS lightgcn bash -c 'scripts/run_lightgcn.sh > results/lightgcn_console.log 2>&1'
```

Outputs go to `results/lightgcn/`. LightGCN row `i` in `item_emb.npy` is `item_idx`
`i`, and row `i` of `item_content_semantic_ids.pkl`. The Transformer's token for the
same item is `i + 1`. Results and fusion guidance are in
`presentation/lightgcn/RESULTS.md`; `scripts/plot_lightgcn_curves.py` regenerates the
presentation files.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the available accelerator:

```bash
python scripts/check_device.py
```

## Run the first milestone

```bash
python scripts/prepare_data.py --download
python scripts/run_popularity_baseline.py
```

The scripts use `data/raw`, `data/processed`, and `results` inside this project. Large downloaded files are intentionally excluded from version control.

## Project proposal

The proposal is kept outside version control, in the `MLG/` context folder.

