# Slides 1–6: checked explanations

## Main findings and corrections

### Takeaway
The speaker notes are generally more accurate than the slide text. Explain the intended idea while correcting the misleading claims; no deck edits requested.

### Cited Findings
- Slides2/5 text says TIGER never sees who bought what. This is false: TIGER trains on ordered user-item interaction histories. Its original semantic-ID tokenizer uses product content; this is distinct from saying the recommender uses only text. — [TIGER paper](https://arxiv.org/html/2305.05065v3)
- Slide2 notes expand NDCG incorrectly. It means **Normalized Discounted Cumulative Gain**: discounted relevance divided by ideal discounted relevance. — [Official scikit-learn metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- With one relevant target per user, local metric code gives Recall@10=1 when target rank≤10, otherwise0; NDCG=1/log2(rank+1) inside top10. Example rank3 gives1 and.5; rank11 gives0 for both. — local `src/tiger/metrics.py:25`
- 5-core means every retained user and product has at least5 reviews/interactions in the filtered subset. The publisher's2018 page confirms the definition; do **not** confuse its Beauty data/counts with2014 Beauty used here. — [Dataset publisher definition](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html)
- Source data are review-derived interactions, not independently observed transaction logs. “Purchase history” is the deck's informal shorthand. — [Dataset publisher](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html)
- Slide4's residual codes comprise3 learned positions plus collision counter. '3×256 vocabulary' omits the fourth range and user/special tokens. Similarity often produces related codes; shared prefixes are not guaranteed for every similar pair. — [TIGER paper](https://arxiv.org/html/2305.05065v3)
- OPQ learns a rotation before separate-subspace quantization, not merely slicing the raw vector. — [Faiss authors' documentation](https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing)
- RK-Means locally trains256 centers on residuals at each of3 levels, then adds a collision code. It needs fitting despite lacking a neural autoencoder. — local `scripts/paper/rkmeans.py:21`
- Slide5 '~5M' describes local reproduction, not original paper's reported~13M. Last20 history is local `src/tiger/data.py:11`; stage2 unconstrained beam20 is `scripts/paper/train_t5.py:72`, with additional constrained evaluation available. — [Original TIGER experimental settings](https://arxiv.org/html/2305.05065v3); local `src/tiger/model.py:9`, `scripts/paper/train_t5.py:244`
- LightGCN linearly aggregates graph neighbors and combines layers; it omits per-layer learned feature transforms and nonlinearities. — [LightGCN paper](https://arxiv.org/abs/2002.02126)
- Author implementation explicitly learns initial embeddings, propagates by graph matrix multiplication, averages layers, and applies BPR via softplus(negative_score−positive_score), plus L2. — [Author-linked PyTorch source](https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/code/model.py)
- Local graph is D^(-1/2)AD^(-1/2); neighbor weights are degree-normalized, not a simple arithmetic mean. Final2-round vector=(E0+E1+E2)/3. Stage2 pairs explicitly use sequence[:-2], leaving validation/test targets out of graph. — local `src/lightgcn.py:7`, `scripts/paper/lightgcn5.py:29`

### Inferences
- A user's two products can become related through item→user→item paths; this can expose complements missed by text similarity. It is a tendency learned by the objective, not a guarantee all co-interacted items are identical.
- Chronological holdout example A,B,C,D,E: train on A,B,C; validate D; test E. This is confirmed in local `src/tiger/data.py:178`. Test history includes earlier validation interaction, while the LightGCN graph remains training-only.

### Gaps
- Did not run checkpoints or independently reproduce counts/results; these explanations concern the deck, its notes, source code, and referenced methods.
- Keep total TIGER-paper-derived final wording≤200 words across all slides. Attribute local implementation details to local files; use Faiss for OPQ and metric documentation for NDCG. Other each-page word budgets200 separately.

## Proposed short paragraph for each slide

### Slide1 — Project question
Graph-TIGER asks whether understanding relationships between shoppers and products improves the next recommendation. Imagine someone bought shampoo: product text describes shampoo, while shared shopping patterns might reveal that conditioner is a useful next suggestion. The project tests three ways to add this relationship information: reorder recommendations, change product codes, or enrich the model's inputs. Its purpose is to discover where graph information actually helps.

### Slide2 — Task, data, evaluation
The task is predicting the next interaction from earlier ones. TIGER already learns from these histories; the new ingredient is an explicit user-item graph. The two dataset settings must be evaluated separately. “5-core” keeps users and products with at least five interactions. Recall@10 asks whether the correct next item appears; NDCG@10 also rewards its position: one target at rank3 scores1/ log2(4)=0.5. NDCG means Normalized Discounted Cumulative Gain.

### Slide3 — Three insertion points
An embedding is a vector of numbers describing a product; quantization converts that vector into a short discrete code. The recommender reads past products and predicts the next. MethodA changes the final ranking, MethodB changes the codes, and MethodC changes the vectors entering the model. Think of these as changing the final shortlist, changing product labels, or giving the decision-maker extra information. Their different effects reveal where graph information survives best.

### Slide4 — Semantic codes
A semantic ID is a short product address, such as(12,201,7,0). RQ-VAE and residual k-means repeatedly choose a nearby center and encode what remains unexplained; the final number separates products with identical first three numbers. RQ-VAE learns a neural autoencoder; residual k-means fits centers directly. OPQ instead learns a rotation and codes separate vector pieces. These compact codes let related products share information, although matching prefixes are not guaranteed.

### Slide5 — TIGER
TIGER's encoder reads the earlier product codes, and its decoder predicts the next code sequentially:12, then201 given12, then7, then0. Training teaches these predictions using known next items. Beam search retains several promising partial codes so an early wrong guess does not decide everything. The local model uses four encoder and four decoder layers. 'Real items only' applies to constrained decoding; the main second-stage evaluation uses unconstrained decoding.

### Slide6 — LightGCN
LightGCN draws users and products as nodes, connecting each user to their training interactions. Each node starts with a learned vector; two rounds of degree-normalized neighbor mixing share information across item→user→item paths. It averages the starting and two propagated vectors. BPR trains a known interaction above a sampled unknown one:loss=−log sigmoid(positive_score−negative_score). Only initial vectors are trained. This supplies collaborative information while keeping held-out test interactions out of the graph.
