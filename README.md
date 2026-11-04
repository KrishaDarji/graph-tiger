# Graph-TIGER: Amazon Beauty Recommendation

This project reproduces a lightweight TIGER-style generative recommender on the Amazon Beauty dataset and studies whether collaborative graph information improves it.

## Current scope

The main research model is TIGER. The graph model is an extension, not a replacement. The project is organized as:

1. Reproduce the data and chronological evaluation protocol.
2. Generate item representations and semantic IDs.
3. Train a lightweight TIGER-style next-item generator.
4. Add LightGCN information to item representations.
5. Compare baseline TIGER against graph-enhanced TIGER.

The popularity baseline and LightGCN code are supporting baselines. The primary comparison will be content-only TIGER versus graph-enhanced TIGER.

## Implementation files

### Data and preprocessing

- `scripts/prepare_data.py` — downloads and preprocesses Amazon Beauty interactions.
- `scripts/create_split.py` — creates the chronological train/validation/test split.
- `src/data.py` — data loading, ID encoding, and chronological splitting utilities.
- `scripts/prepare_metadata.py` — builds product-content embeddings and lightweight semantic-ID-like codes.

### TIGER-style model

- `src/tiger_small.py` — defines the lightweight Transformer architecture.
- `scripts/run_tiger_small.py` — trains the content-aware next-item model.
- `scripts/run_tiger_small.sbatch` — Hazel Slurm training script with checkpoint/resume support.
- `scripts/evaluate_tiger_checkpoint.py` — evaluates a saved checkpoint using Recall and NDCG.

### Evaluation and graph extension

- `src/evaluation.py` — implements Recall@K and NDCG@K.
- `src/lightgcn.py` — LightGCN model and normalized user–item graph construction.
- `scripts/run_lightgcn.py` — trains the planned graph-based extension.
- `scripts/run_popularity_baseline.py` — popularity reference baseline.

The model currently trained on Hazel is the content-aware raw-ID Transformer. The full semantic-ID TIGER model and graph-enhanced TIGER are planned extensions.

## Build content representations and semantic IDs

After the interaction data has been prepared:

```bash
python scripts/prepare_metadata.py
```

This creates `data/processed/item_content_semantic_ids.pkl`. It is a
lightweight, reproducible approximation of TIGER's semantic-ID stage using
TF-IDF, SVD, and residual quantization. The later experiment will use these
representations in the content-only and graph-enhanced models.

## Run the lightweight TIGER prototype

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

The proposal is outside this folder at:
`/Users/RadhaKrishna/Desktop/ML with Graphs/Project_proposal.pdf`

