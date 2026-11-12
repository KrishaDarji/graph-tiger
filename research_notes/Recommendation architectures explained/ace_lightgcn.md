# ACERec and LightGCN architectures

## ACERec paper v2: exact architecture and train versus inference

### Takeaway
ACERec compresses long product codes with attention, models a history of item blocks with a decoder Transformer, and predicts all target code positions in parallel. Use v2 (2 August 2026), because it adds a reconstruction objective absent from the local experiment.

### Cited Findings
- Paper configuration: Sentence-T5-base; OPQ IDs length m=32 (64 is an ablation), M=256 on Amazon2014 or 512 on Amazon2023; k=4 compressed latents; d=448; 2 decoder layers, 4 attention heads, FFN1024. [Paper Appendix A.4](https://arxiv.org/html/2602.13573v2#A1.SS4)
- Pipeline and shapes: token-embedding matrix E[m,d]; summary s=f_s(E); content-conditioned queries Q=f_q(s)[k,d]; cross-attention uses Q as queries and E+P as keys/values, then output MLP/normalization yields Z[k,d]. The attention weights softly mix all m token vectors; they are not hard clusters. [Paper Section 3.1](https://arxiv.org/html/2602.13573v2#S3.SS1)
- Append h=s after each item's k latent tokens. Apply step-wise causal decoding; current latents interact, the intent position sees current latents plus history, and future items stay masked. [Paper Section 3.2](https://arxiv.org/html/2602.13573v2#S3.SS2)
- Train with parallel token cross-entropy plus popularity-calibrated in-batch intent/target alignment plus cosine reconstruction loss from an upsampling decoder. Inference ranks existing products solely by summed code log-probabilities; there is no added item-alignment score. [Paper Sections 3.1-3.3](https://arxiv.org/html/2602.13573v2#S3)

### Inferences
- The intent token is a floating-point vector acting as a prediction summary, not a semantic code number or an explicitly labeled shopping intention.
- Four latent vectors are continuous neural representations, not four newly chosen discrete OPQ IDs. Output still predicts all 32 original code positions.

### Gaps
- No official code link was found in paper HTML or a targeted search; do not claim an official implementation exists.
- The exact internal choice of f_s/f_q beyond the paper notation is not specified in the main formulas. Local code makes a particular choice documented below.
- This one web source has a 200-word derivation limit. The concise paper-facts above approach that budget. Detailed explanations of shared mechanisms can cite the locally inspected code instead; do not produce a large uncited paraphrase of the whole paper.

## Local ACERec-inspired implementation mapping

### Takeaway
scripts/paper/ace.py implements the attention merger, per-item intent position, masked Transformer, parallel prediction heads and optional alignment. It is not a complete v2 implementation: there is no reconstruction decoder/loss, and defaults keep the special components disabled.

### Cited Findings
- Defaults: --digits32, K256, PCA128, d448, 2 layers, 4 heads, FFN1024, max history50. --atm and --intent are false unless supplied; --isa is 0. --latents defaults1, so four latents require --atm --latents4. [Options](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:24)
- Quantization trains FAISS PCA then OPQ/PQ with 8 bits per subspace; immutable integer catalog codes are offset by codebook index before entering a trainable token embedding table. The OPQ quantization is offline, not differentiated through by the recommender. [Tokenization](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:60)
- For a batch of B items, lookup yields [B,m,d]; average over m to get [B,d]. A Linear-GELU-Linear summary MLP produces s[B,d]. A separate Linear(d,k*d) reshaped to [B,k,d] makes content-conditioned queries. Cross-attention uses those queries with token vectors plus learned code-position vectors as keys/values. It then applies a query residual, layer normalization, and an MLP residual to form [B,k,d] latents. [Summary](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:129); [Merger](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:112)
- With --intent, append s to each item block, after its latent vectors. Add learned item-step and within-block slot positions. A length-L history becomes [B,L*(k+1),d]. Current block latents can attend to all current latents and earlier blocks; the intent token can attend to current latents, itself and earlier blocks. It cannot see future items. Each block's last output is its prediction anchor. [Block assembly and mask](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:165)
- Each Transformer block uses pre-normalized masked multi-head attention and a residual, followed by pre-normalized FFN and residual. [Transformer block](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:96)
- The anchor h[B,L,d] feeds m separate Linear(d,d) heads, each used as h+SiLU(head(h)). Normalize each and dot it against normalized trainable code embeddings in its own subspace, divide by temperature .03, then softmax over256. Tensor is [B,L,m,256]. The heads operate in parallel, without conditioning on earlier predicted code digits. [Prediction heads](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:191)
- Training uses every valid next-item position in retained histories. Cross-entropy compares all predicted digits to the next product's offline code. Optional ISA normalizes anchors and target summaries, computes in-batch similarities divided by .07, subtracts .02 times local log popularity, masks duplicate targets off-diagonal and applies a contrastive cross-entropy. Weighted ISA is added to MTP. No reconstruction objective is implemented. [Training](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:233)
- The local popularity correction uses log(f+1) normalized as log probabilities, unlike v2's log(f)/max log(f); do not equate the formulas exactly. [Popularity](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:91)
- Inference uses the final history anchor, predicts m code distributions once, and gathers each catalog product's known code log probabilities, summing over m before top-K ranking. This is exact ranking over the finite catalog under the factorized scoring model; it does not enumerate all256^32 possible tuples. [Evaluation](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:207)
- Optional local graph gate normalizes frozen graph embeddings, projects them to d, learns sigmoid weights from concatenated mean-token and graph vectors, and adds the resulting correction to latent vectors and intent input. This is the project's addition, not a claim about ACERec paper architecture. [Graph input](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:154)

### Inferences
- For explanations, use m32,k4,d448 as the paper's example. A three-product history is15 Transformer positions, not96. With local defaults --atm omitted and --intent omitted it is only3 mean-pooled positions; distinguish example from executable defaults.
- A soft attention matrix[k,m] per head learns which input code embeddings matter to each latent; no code position is forced to represent a named human attribute such as brand.

### Gaps
- No training was rerun and no implementation quality audit was requested.

## LightGCN architecture and how this repository uses it

### Takeaway
LightGCN learns user and product vectors from an undirected bipartite interaction graph. It repeatedly sums normalized neighbor vectors, averages all propagation depths, then trains dot-product scores to rank observed interactions above sampled unobserved ones.

### Cited Findings
- Paper deliberately removes per-layer feature transforms and nonlinearities; its propagation only aggregates neighbors, and final representations combine layers0..K. Only layer0 user/item embeddings are trainable in the basic model. [LightGCN paper Section3](https://arxiv.org/html/2002.02126#S3)
- The paper uses symmetric degree normalization, layer-weighted combinations (uniform1/(K+1) in experiments), dot-product recommendations, BPR plus L2 regularization. [Primary paper](https://arxiv.org/html/2002.02126); [Official PyG documentation](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.models.LightGCN.html)
- Local graph includes user and item nodes, symmetric edges, unique unweighted user-item pairs. Edge(u,i) weight=1/sqrt(deg(u)*deg(i)); matrix G=D^-1/2 A D^-1/2. It has no explicit self-loop. [Graph construction](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/src/lightgcn.py:7)
- Local parameters begin with a [n_users+n_items,64] trainable embedding table, initialized Normal(0,.1). Two propagation steps apply E1=G E0, E2=G E1. Final embedding is (E0+E1+E2)/3. No learned transform W or activation appears in propagation. [Model](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/src/lightgcn.py:26)
- Scoring is dot(user_final,item_final). Local BPR=mean softplus(score_negative-score_positive), equivalent to -log sigmoid(score_positive-score_negative). L2 applies to sampled layer0 user, positive and negative vectors. [Scores/loss](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/src/lightgcn.py:48)
- Negatives are uniformly sampled items, rejecting the user's observed training items. Unobserved is a training assumption, not proof the user dislikes the item. [Negative sampling](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/src/lightgcn.py:68)
- Paper-style graph trainer defaults dimension64/layers2/lr.001/reg.0001, excludes final2 interactions per user from graph edges, selects checkpoint by validation Recall@10, exports item_emb.npy and user_emb.npy. [Trainer](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/lightgcn5.py:12)
- In standalone graph inference the trainer computes all user-item dot products and filters seen training items. In the combined RPG model the saved product vectors are instead normalized and injected by a learned gate at the sequence-model input; user vectors are not directly used there. [Graph validation](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/lightgcn5.py:46); [RPG fusion](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/pgen.py:105)

### Inferences
- One propagation step lets a product collect information from its users; two steps let information travel product->user->product, reflecting shared shoppers without explicit product-product edges.
- Layer0 random vectors are optimized through the entire propagation computation by ranking loss; they are not fixed random descriptors. Averaging includes both direct and more distant neighborhood information.
- LightGCN ignores the ordering of interactions in its graph; the sequence Transformer supplies order-aware prediction in the combined model.

### Gaps
- The local64D/2-layer setting is this project's configuration, not a universal LightGCN hyperparameter requirement.
