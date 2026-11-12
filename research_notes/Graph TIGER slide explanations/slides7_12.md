# Slides 7–12: simple explanations and supporting checks

## Slide 7 — Method A: re-ranking

Re-ranking means rearranging recommendations already generated. TIGER supplies up to 30 candidates; LightGCN gives each a user–product dot-product score. After standardizing both score types, we add them with a validation-chosen weight, then keep 10. A cream might move from eighth to third because its graph score is strong. This needs no TIGER retraining, but cannot recover a correct product missing from those 30 candidates.

Evidence: scripts/rerank_tiger_lightgcn.py:1-5,33-47,58,92-97; presentation/tiger_full/RESULTS.md:9-12. Primary background: [LightGCN paper](https://arxiv.org/html/2002.02126), which defines user–item dot-product scoring.

Correction: the drawing says64-D, but the selected final re-ranking result uses768-D graph embeddings, alpha1. Correct formula is z(TIGER log probability)+alpha*z(user dot item), with both z-scores calculated within that user's retained candidates; raw score scales are not simply added. There can be fewer than30 usable candidates after seen-item filtering. No retraining refers to TIGER; LightGCN was separately trained.

## Slide 8 — Method B: graph inside product codes

Fusion means combining two descriptions of a product: text vector c and shopping-pattern vector g. We try addition, concatenation, a small neural network, a learned graph weight, or dimension-wise gates. Then we quantize the mixture and retrain TIGER. Sunscreen and moisturizer may become related through shared shoppers. However, compression can lose useful detail, and these separately trained fusion mechanisms do not optimize TIGER's final prediction loss.

Evidence: scripts/learn_fusion.py:38-76,95-114; presentation/tiger_fusion/RESULTS.md:9-25; presentation/tiger_fusion_learned/RESULTS.md:9-20. Primary foundations: [TIGER paper](https://arxiv.org/abs/2305.05065) and [InfoNCE origin](https://arxiv.org/html/1807.03748v2).

Training distinctions: fixed addition/concatenation need no fusion learning; MLP1536->1024->768 is fitted by balanced reconstruction of both source halves, then frozen. Learned scalar/gate are fitted by full-catalog cross-entropy of temperature-scaled cosine(history-mean,target-product), using80% validation users for fitting and20% holdout for selection, prior to RQ-VAE. That contrastive classification resembles InfoNCE, but do not claim the entire MLP or TIGER uses an InfoNCE fusion loss. InfoNCE explanation if asked: make the actual next item's similarity high relative to distractors; loss=-log(exp(s_positive/T)/sum_j exp(s_j/T)). Here the denominator spans catalog items, not only in-batch negatives. Cold graph terms are explicitly zero. Gate inputs include content, graph and log training degree, producing768 per-dimension weights. 'TIGER unchanged' means architecture unchanged, not checkpoint unchanged: retraining occurs.

## Slide 9 — RPG-style parallel prediction

Here, each product has 32 code tokens, but their learned vectors are averaged into one input vector. A small GPT-style Transformer reads the purchase sequence; 32 output heads predict the next product's code positions simultaneously. We score each catalog product by adding the log probabilities of its known codes. Like filling 32 answer boxes together instead of sequentially, this supports longer codes without 32 decoding passes.

Evidence: scripts/paper/pgen.py:49-64,86-120,129-148; primary [RPG paper](https://arxiv.org/abs/2506.05781) and [official implementation](https://github.com/facebookresearch/RPG_KDD2025).

Correction/distinction: this slide explains the local RPG-style model. Official RPG also uses a graph connecting similar semantic IDs for decoding; that semantic-ID graph is different from the project's LightGCN user–item graph, and local pgen.py instead directly scores the whole catalog. The32 output heads are prediction heads, not32 Transformer attention heads (local Transformer uses4 attention heads). GPT-style describes the architecture; this is not a pretrained ChatGPT model.

## Slide 10 — Method C: gated graph input

A gate is a learned volume control for graph information. Product codes remain text-derived; their averaged vector e receives a projected graph vector Pg, weighted separately in each dimension: input=e+gate*Pg. If e=.6, Pg=.4 and gate=.25 in one dimension, input=.7. Training adjusts this gate for next-item prediction, while LightGCN stays frozen. Unlike Method B, no graph information is compressed into the discrete codes.

Evidence: scripts/paper/pgen.py:66-69,98-112,123-124,155-170; [LightGCN foundation](https://arxiv.org/html/2002.02126). This gate is the project's extension; do not attribute it to the original RPG paper.

Clarifications: projection maps64-dimensional graph vectors to448-dimensional model inputs. gate=sigmoid(W[e;g]+b) has448 components; multiplication is element-wise. Gate depends on item representation and graph, not directly on the current user. Sigmoid can approach0, but 'switches off exactly where unhelpful' is a capability/hypothesis, not a demonstrated guarantee. Jointly trained means token embeddings, sequence model, projection and gate; it excludes offline OPQ and pretrained frozen LightGCN embeddings. g is L2-normalized first.

## Slide 11 — Stage-one results

These bars compare ranking quality, measured by NDCG@10, against text-only TIGER. MLP fusion raises it from.01870 to.01943: (.01943/.01870-1)*100≈3.9%; learned code gates instead hurt. NDCG rewards placing the correct product nearer the top, so a ranking can improve without many additional hits. Each model has one training seed, making these small gains preliminary; graph-in-code results on 5-core are mixed rather than steadily decreasing.

Evidence: presentation/tiger_fusion/RESULTS.md:26-60,107-114; presentation/tiger_fusion_learned/RESULTS.md:28-51; presentation/tiger_full/RESULTS.md:45-56; presentation/paper_repro/RESULTS.md:36-42.

Metric verification in local src/tiger/metrics.py or src/evaluation.py: single held-out target contributes1/log2(rank+1) if rank<=10, else0. For rank1 contribution1, rank3 contribution.5; average over users. NDCG means Normalized Discounted Cumulative Gain (the slide2 speaker note's 'Non Discounted' expansion is wrong). For a single relevant target ideal DCG is1. The bars are relative percentages, not percentage-point increases. Bootstrap resamples test users, not training seeds; multiple comparisons uncorrected. Fixed-add gain lacks clear evidence under equal training budget. MLP's .01943 mean is not evidence it reliably beats fixed-add .01932. All5-core code-fusion values do not decline monotonically (.0594,.0571,.0527,.0530).

## Slide 12 — Which products benefit?

Cold products have zero training interactions; rare ones have 1–4; this slide calls 5-or-more popular. Rare-item Recall@10 rises from.00030 to.00156 with fixed graph addition—about 5.2 times, but only 30 versus 156 correct top-ten hits per 100,000 such cases. Cold-item recall stays near zero in these experiments. With no interaction edges, their graph vectors lack learned collaborative evidence; this does not mean text-based recommendation can never recommend unseen products.

Evidence: presentation/tiger_full/RESULTS.md:59-68; presentation/tiger_fusion/RESULTS.md:63-85; presentation/tiger_fusion_learned/RESULTS.md:55-65. Primary [LightGCN paper](https://arxiv.org/html/2002.02126) explains interaction-graph propagation and trained initial ID embeddings.

Clarifications:20.7%,25.2%,54.1% are shares of TEST TARGETS, not shares of catalog products. For catalog items26.6% are cold in this stage. The rare improvement is fixed addition vs baseline, not every graph variant. Concatenation actually reaches.00191 on rare targets; MLP.00078, learned gate.00054. Frequent-item baseline.06156 and fixed-add.06144 are similar, but learned gate.05655 is worse. Because frequent targets contribute54.1%*roughly.062≈.0335, they dominate the overall.0334 recall. The graph supplies no collaborative evidence for isolated nodes; local prequantization builders explicitly zero their graph contribution. A content model still has text features, so its observed cold failure is an empirical outcome, not a mathematical necessity.

