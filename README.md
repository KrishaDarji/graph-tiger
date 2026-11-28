# Graph-TIGER

Semantic-ID generative recommenders such as TIGER and RPG give each item a short code built from its text and predict the next item's code. They never see which items are bought together. Graph-TIGER adds that signal as a gated input to an RPG-style decoder and trains it with an item-level loss.

<div align="center">
<img src="asset/overview.png" width="100%"/>
</div>

## What it does

- **Collaborative input.** Each item has a 128-d vector from a truncated SVD of the item-item transition graph (built from training prefixes only). It enters the decoder through a learned gate: `x = e + sigmoid(W[e; g] + b) * P g`, with `e` the item's code embedding and `g` the graph vector. The codes stay text-only.
- **Item-level loss.** Next to the per-digit losses, a softmax over 4,096 sampled items on the cosine between the sequence state and the item vector. At scoring time its log-probability is added with a weight chosen on validation.
- **Weight averaging** (exponential moving average, 0.99).

The same code also runs the other placements of the graph that we compare: re-ranking the scores, and building the graph into the semantic IDs, with either the transition graph or LightGCN.

## Results

Amazon Reviews 2014, 5-core, leave-one-out, all items ranked. Test Recall@10 / NDCG@10; our rows are means over 3 seeds.

| | Beauty | Sports | Toys | CDs |
|---|---|---|---|---|
| RPG (published) | .0809 / .0464 | .0463 / .0263 | .0869 / .0490 | .0735 / .0415 |
| ACERec (published) | .0856 / .0513 | .0494 / .0287 | .0940 / **.0591** | .0800 / .0451 |
| TopoGR (published) | .0883 / **.0522** | **.0554 / .0314** | **.0995** / .0588 | .0783 / .0445 |
| Plain (our base, no graph) | .0699 / .0411 | .0415 / .0237 | .0790 / .0461 | .0712 / .0405 |
| No graph (item loss + averaging) | .0790 / .0456 | .0498 / .0278 | .0878 / .0498 | .0893 / .0484 |
| **Graph-TIGER** | **.0925 / .0522** | .0527 / .0290 | .0952 / .0514 | **.1035 / .0566** |

Against the best published values we are ahead on Beauty Recall@10 and on CDs, level on the other Beauty metrics, and behind on Sports and Toys. All tables, significance tests and limits: [docs/RESULTS.md](docs/RESULTS.md). Protocol: [docs/PROTOCOL.md](docs/PROTOCOL.md). One row per run: [docs/runs.csv](docs/runs.csv).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/prepare_data.sh Beauty beauty
```

`prepare_data.sh` downloads the 5-core reviews and metadata into `data/beauty/`, embeds the item text with Sentence-T5-base and builds the transition graph. The other datasets are `Sports_and_Outdoors sports`, `Toys_and_Games toys` and `CDs_and_Vinyl cds`.

## Train and evaluate

```bash
python main.py --data data/beauty --out results/beauty/ours_s2024 \
    --digits 32 --lr 0.01 --batch 256 --dense-w 0.3 --ema 0.99 --seed 2024
```

Digits and learning rate follow RPG:

| | digits | lr |
|---|---:|---:|
| Beauty | 32 | 0.01 |
| Sports | 16 | 0.003 |
| Toys | 16 | 0.003 |
| CDs | 64 | 0.001 |

| Variant | Flags |
|---|---|
| Plain | `--no-graph` |
| No graph | `--no-graph --dense-w 0.3 --ema 0.99` |
| Graph-TIGER | `--dense-w 0.3 --ema 0.99` |
| Re-rank with the graph | `--rerank-graph data/beauty/graph.npy --gammas 0,2,5,10,20,40,80,160` |
| Graph inside the semantic IDs | `--no-graph --token-graph 1.0` |
| LightGCN as the graph | `--graph-emb data/beauty/lightgcn/item_emb.npy` after `python -m baselines.lightgcn --data data/beauty` (for equal size add `--dim 128 --out data/beauty/lightgcn128`) |

Add `--ease-ws 0,0.5,1,2` for the EASE ensemble, `--cands 50` for top-50 re-scoring, `--temporal 0.8,0.9` for the temporal split, `--holdout` for the cold-start split and `--slots 1,2,3` for reserved new-item slots (needs one of the two splits). Every run writes `result.json` and the per-user ranks.

```bash
python -m scripts.summarize results/beauty plain ours --ref plain
python -m scripts.claims results/beauty beauty ours
```

Baselines: `python -m baselines.ease|sasrec|knn --data data/beauty --out <folder>`. Graph refresh: `python -m scripts.build_graph --prefix-frac 0.7` for a stale graph, then rebuild with `--align-to` and fine-tune with `--init`.

## Layout

```
main.py            train, select on validation, evaluate, write results
src/               data and splits, item codes, graph, model, scoring, trainer, evaluator
baselines/         EASE, SASRec, nearest-neighbour rules, LightGCN
scripts/           data preparation, graph building, significance and summary tables
docs/              results, evaluation protocol, one row per run
asset/             figures
```

`data/`, `results/`, `logs/` and `ckpt/` stay local and are git-ignored.
