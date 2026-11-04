# Graph-TIGER Configuration and Architecture Differences

## Configuration used in our project

### Data

```yaml
Dataset: Amazon Beauty
Interactions: 786,049
Users: 147,241
Items: 163,257

Train: 491,567
Validation: 147,241
Test: 147,241
```

The split is chronological: earlier interactions train the model, the next interaction is validation, and the final interaction is test.

### Current content-aware model

```text
Embedding dimension: 128
Transformer layers: 2
Attention heads: 4
Dropout: 0.1
Maximum history length: 20
Batch size: 256
Learning rate: 5e-4
Optimizer: AdamW
Content embedding size: 64
Content source: TF-IDF → SVD
Device: RTX 6000 Pro using CUDA
```

Training was checkpointed in 100-batch chunks because Hazel has a two-hour job limit.

The current model predicts the next item from a raw item-ID vocabulary of 163,257 classes.

Run command:

```bash
python scripts/run_tiger_small.py \
  --use-content \
  --epochs 100 \
  --batch-size 256 \
  --dim 128 \
  --max-length 20 \
  --steps-per-job 100 \
  --checkpoint results/tiger_small_checkpoint.pt \
  --resume
```

## What differs from the TIGER paper

The TIGER paper uses semantic IDs: a tuple of discrete codewords representing each item. The Transformer predicts the next item's semantic-ID sequence rather than an arbitrary raw product ID. See the [TIGER paper](https://arxiv.org/abs/2305.05065).

| Component | TIGER paper | Our current implementation |
|---|---|---|
| Item representation | Pretrained content encoder + RQ-VAE | TF-IDF + SVD + lightweight clustering |
| Item target | Semantic-ID codeword tuple | Raw item ID |
| Decoder output | Multiple semantic-ID codeword predictions | One 163,257-class classifier |
| Sequence model | Autoregressive encoder–decoder Transformer | Small Transformer encoder |
| Dataset protocol | Paper's published preprocessing/evaluation protocol | Our chronological Amazon Beauty split |
| Graph information | Not the main TIGER model | Planned LightGCN extension |
| Scale | Large-scale paper experiments | Project-scale reproduction |
| Goal | Generative retrieval | Test whether graph information improves TIGER |

## Why we changed the architecture

Our current raw-ID model performs well enough to establish a baseline:

```text
Recall@10: 0.01304
NDCG@10:   0.00699
```

Raw product IDs are arbitrary labels. The model must learn each product independently, even when two products have similar descriptions or user behavior.

The planned architecture is:

```text
Product metadata
      ↓
Content encoder
      ↓
Semantic-ID generator
      ↓
Semantic-ID sequence
      ↓
TIGER Transformer
      ↓
Semantic-ID prediction heads
      ↓
Decode predicted IDs into products
```

Then we add graph information:

```text
User–item interactions
      ↓
LightGCN
      ↓
Collaborative item embeddings
      ↓
Fuse with content embeddings
      ↓
Graph-enhanced TIGER
```

The purpose of the architecture changes is to test the actual research question:

> Does combining semantic content representations with collaborative graph information improve generative recommendation?

Our current result should therefore be described as a **content-aware TIGER-style prototype**, while the remaining work is the full semantic-ID TIGER and graph-enhanced comparison.

