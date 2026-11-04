# Graph-TIGER Project Status and Architecture Plan

## 1. Project goal

This project studies whether collaborative user–item graph information can improve a TIGER-style generative recommender on the Amazon Beauty dataset.

The central research question is:

> Does adding graph-derived collaborative information improve next-item recommendation compared with content-only or raw-ID sequential recommendation?

The project is based on the TIGER idea of treating recommendation as generative retrieval: a Transformer predicts an item's identifier from a user's interaction history. TIGER's intended identifier is a semantic ID made from discrete codewords derived from item content.

## 2. Dataset and evaluation setup

Source: Amazon Beauty reviews and product metadata.

Prepared interaction data:

- Interactions: 786,049
- Users: 147,241
- Items: 163,257

Chronological split:

- Training: 491,567 interactions
- Validation: 147,241 interactions
- Test: 147,241 interactions

For each user, interactions are ordered by timestamp. Earlier interactions are used for training, the next interaction is used for validation, and the final interaction is used for testing. This avoids using future behavior when training the recommender.

Metrics:

- Recall@5
- Recall@10
- NDCG@5
- NDCG@10

## 3. Results completed so far

### Popularity baseline

| Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.0048 | 0.0080 | 0.0026 | 0.0036 |

### Content-aware raw-ID Transformer

The current content-aware prototype uses a small Transformer. Each history item is represented by a learned item-token embedding plus a 64-dimensional product-content embedding. The model predicts the next raw item ID from a 163,257-class output layer.

Full-test result at checkpoint epoch 63, batch 148:

| Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| Content-aware raw-ID Transformer | 0.00833 | 0.01304 | 0.00547 | 0.00699 |

The model improves over popularity on all reported metrics. The recorded epoch-31 result was also positive:

- Recall@5: 0.00850
- Recall@10: 0.01282
- NDCG@5: 0.00554
- NDCG@10: 0.00693

The epoch-63 result is the current full-test result to report. It is important to describe this as a content-aware TIGER-style prototype, not yet as a complete semantic-ID TIGER reproduction.

## 4. Current architecture

```text
User interaction history
          |
          v
Raw item-ID tokens
          |
          +-----------------------------+
          |                             |
          v                             v
Learned item embedding        64-D product-content embedding
          |                             |
          +-------------+---------------+
                        v
              Small Transformer encoder
                        |
                        v
             163,257-way raw item classifier
                        |
                        v
                 Next-item prediction
```

### What the current model demonstrates

The current model demonstrates that learned sequential modeling plus product content can outperform a popularity-only strategy on this dataset.

### Limitations of the current model

1. It predicts raw item IDs rather than semantic-ID codewords.
2. The output vocabulary is very large: 163,257 classes.
3. The generated semantic IDs are currently stored but are not used as prediction targets.
4. The model does not yet incorporate LightGCN embeddings.
5. The content representation is a lightweight TF-IDF/SVD approximation rather than the full pretrained text encoder used by the paper.

## 5. Why change the architecture?

### Change 1: Raw item IDs to semantic IDs

Current problem: raw item IDs are arbitrary labels. Product IDs do not express similarity: two similar beauty products may have completely unrelated IDs.

Planned change: represent each item using a short tuple of discrete semantic codewords derived from its content.

```text
Product text -> content embedding -> discrete semantic ID
```

Why this matters:

- Similar products can share codeword structure.
- The model predicts a compact structured identifier instead of one arbitrary class.
- The model can generalize better from known products to related products.
- The architecture becomes faithful to the central TIGER idea.

### Change 2: One raw-item output head to semantic-ID output heads

Current architecture:

```text
Transformer -> one 163,257-class classifier
```

Planned architecture:

```text
Transformer -> codeword head 1
           -> codeword head 2
           -> ...
```

Each head predicts one component of the item's semantic ID. At inference time, the predicted codeword tuple is mapped back to the corresponding product.

Why this matters:

- The output space is structured and compact.
- Training is less dependent on memorizing arbitrary item labels.
- The model can share statistical information across related products.

### Change 3: Add LightGCN collaborative information

Content describes what a product is. The user–item graph describes how products are used together by real users.

Planned graph pipeline:

```text
Training interactions only
          |
          v
      User-item graph
          |
          v
        LightGCN
          |
          v
 Collaborative item embeddings
```

The graph embeddings will be fused with the content embeddings:

```text
fused item representation
  = content representation
  + weighted collaborative representation
```

Why this matters:

- Content can miss behavioral relationships.
- Graph structure captures co-interaction and collaborative preference signals.
- The graph extension directly tests the project's research question.

### Change 4: Controlled model comparisons

The final experiment should compare:

```text
Popularity baseline
Raw-ID Transformer
Content-aware Transformer
Semantic-ID TIGER
Graph-enhanced semantic-ID TIGER
```

All models must use the same chronological split and evaluation metrics. This isolates whether improvements come from content, semantic IDs, or graph information.

## 6. Planned architecture

```text
                         Product metadata
                               |
                               v
                  Text/content representation
                               |
                               v
                         Semantic IDs
                               |
                               |
User history ------------------+
       |                       |
       v                       v
Semantic-ID sequence       User-item graph
       |                       |
       |                       v
       |                    LightGCN
       |                       |
       +-----------+-----------+
                   v
       Content + collaborative item representation
                   |
                   v
            TIGER Transformer
                   |
                   v
       Semantic-ID prediction heads
                   |
                   v
          Decode IDs to products
                   |
                   v
             Ranked recommendations
```

## 7. Remaining work

1. Modify the training targets to semantic-ID components.
2. Implement multi-head semantic-ID prediction.
3. Decode predicted semantic IDs back to product candidates.
4. Evaluate the semantic-ID TIGER model on the full test set.
5. Train LightGCN using training interactions only.
6. Fuse graph and content representations.
7. Evaluate graph-enhanced TIGER.
8. Create comparison tables and plots.
9. Analyze performance by item frequency and cold-start behavior.

## 8. Current conclusion

The project has a working and reproducible foundation. The current content-aware raw-ID Transformer beats the popularity baseline, demonstrating that the sequential/content direction is promising.

The remaining architecture changes are necessary because the research contribution is not simply training a larger Transformer. The goal is to test whether semantic item identifiers and collaborative graph information improve generative recommendation. Semantic-ID prediction addresses structured item generalization; LightGCN addresses behavioral collaboration. Together they form the planned graph-enhanced TIGER model.

