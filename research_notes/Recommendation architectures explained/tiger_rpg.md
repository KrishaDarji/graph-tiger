# TIGER and RPG architectures

## TIGER: exact mechanism and experimental choices

### Takeaway
Two separately trained stages: content tokenizer, then sequence recommender. Numbers below describe the published experiment, not requirements for every TIGER implementation.

### Cited Findings
- Sentence-T5: metadata (title/price/brand/category) → 768D. RQ-VAE encoder: 768→512→256→128→32, ReLU hidden layers. Three 256-entry codebooks contain 32D vectors. Start residual z; choose nearest vector, subtract it, repeat. Indices form three codes; sum selected vectors and decode to reconstruct the original embedding. Loss: reconstruction squared error + codebook fitting + β-weighted commitment, β=.25. Append a fourth collision counter, **including 0 for unique triples**. Four positions have distinct 256-token ranges (1024 item tokens). — [Paper §§3–4](https://arxiv.org/html/2305.05065v3)
- Input: hashed user token (2000 buckets), then flattened chronological four-token item codes. Encoder-decoder generates next code autoregressively; beam search returns candidates. Published configuration: four encoder/four decoder layers, six 64D attention heads, input128, FFN1024, ReLU, dropout.1; paper reports ~13M parameters. Invalid generated IDs can occur. — [Paper §§3–4 and appendix](https://arxiv.org/html/2305.05065v3)
- Author slides show content embeddings→residual quantization, shared semantic-ID prefixes for similar content, and training a model on the resulting domain tokens. — [TIGER coauthor slides pp7–14](https://nikhil-dce.github.io/data/GenRetrieval%20Research.pdf)

### Inferences
- Teacher forcing means target prefix is supplied while training successive output-token predictions; at inference use generated prefixes. This is the standard operational interpretation of the paper's autoregressive seq2seq setup, rather than an explicitly named paragraph in TIGER.
- Distinguish frozen pretrained Sentence-T5 content encoder from the trainable recommender Transformer; the latter predicts code IDs rather than writing English.
- Do not infer model dimension384 merely from six heads×64: attention projections can have a different inner width. Paper explicitly reports input128; parameter-count/configuration consistency was not independently established.

### Gaps
- No official Google source implementation found. Do not label local HF T5 reproduction identical to released Google code.
- Original paper does not establish that every decode is catalog-constrained; local constrained search is an implementation choice.

## RPG: code construction, sequence model, loss, decoding

### Takeaway
RPG changes both tokenization and prediction: product quantization gives parallel code positions; one item vector enters the sequence backbone; all next-item code positions are predicted together.

### Cited Findings
- Method: OPQ replaces residual quantization. Code positions represent separate vector subvectors rather than sequential approximation residuals. Input code embeddings are pooled per item. Conditional factorization is P(c1,…,cm|history)=product_j P(cj|history). Training sums token negative log probabilities. Inference caches per-position log-softmax values; candidate score sums its selected values. Its graph contains valid catalog codes, connects top-k similar items using corresponding learned token embeddings, and searches from random b catalog items: expand neighbors, score, retain b, repeat q times. This is approximate catalog search, not freely combining argmax digits. — [Paper §2](https://arxiv.org/html/2506.05781v1)
- Config defaults: `n_codebook=32`, each256; history50; width448, 2 decoder layers,4 heads, FFN1024; dropout embedding/attention .5, residual0. Default text encoder `text-embedding-3-large`,3072D→PCA512. Commented Sentence-T5 alternative768D→PCA128. Model sizes are experiment settings, not semantic-ID definitions. — [Official config](https://raw.githubusercontent.com/facebookresearch/RPG_KDD2025/main/genrec/models/RPG/config.yaml)
- Official tokenizer fits whitened PCA and Faiss OPQ/PQ; OPQ fits training items. Digit j gets offset 256j+1, padding0;32×256+2=8194 vocabulary including EOS. Tokens remain position/codebook-specific despite “unordered” prediction. Training uses shifted next-item targets at each valid position, plus windows for longer histories. — [Official tokenizer](https://raw.githubusercontent.com/facebookresearch/RPG_KDD2025/main/genrec/models/RPG/tokenizer.py)
- Code: token embeddings mean-pool into one vector per item; fresh `GPT2Model(config)` supplies causal sequence states. Each digit has an independent `ResBlock`: x+SiLU(Wx+b), W initialized zero. Normalize head states and token embeddings, dot product/temperature, per-digit softmax; average CE across digits (paper's sum differs only by constant). No pretrained GPT-2 checkpoint is loaded. — [Official model](https://raw.githubusercontent.com/facebookresearch/RPG_KDD2025/main/genrec/models/RPG/model.py)
- Implementation builds a dense item-item similarity matrix offline, retains top-k neighbors, and supports both exhaustive item scoring (default flag false for graph decoding) and graph search. Graph search initializes with `torch.randint` (so duplicates possible), ranks neighbors using mean selected log probabilities, and repeats configured steps. This graph is semantic similarity; it is not a LightGCN user-item graph. — [Official model](https://raw.githubusercontent.com/facebookresearch/RPG_KDD2025/main/genrec/models/RPG/model.py)
- The official release's README sets Beauty32 digits, Sports/Toys16, CDs64;64 is a tested setting, not universal length. — [Authors' README](https://github.com/facebookresearch/RPG_KDD2025)
- Faiss defines product quantization over subvectors, contrasting residual/additive quantizers representing a vector as summed codebook entries. — [Faiss documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)

### Inferences
- “Parallel” concerns next-item code positions. The history backbone still models chronological dependence causally; it does not ignore sequence order.
- Graph search serves the function of approximate nearest/best-score search, but the public code constructs exact pairwise similarity offline rather than calling a conventional online Faiss ANN index.
- “Independent” is a modeling assumption conditional on the same history state, not proof OPQ digits are statistically independent.

### Gaps
- Paper appendix graph hyperparameters differ from official README examples: avoid presenting either as universal.
- Source budgets: each paper, each code file, README, author slides, and Faiss page has web wordlim200 separately. Keep final TIGER paper-derived explanation ≤200 words; RPG paper-derived mechanism ≤200 words; attribute implementation details to config/model/tokenizer sources separately. No verbatim quotation needed.
