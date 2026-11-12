# Local Graph TIGER repository

## What does this project do?

### Takeaway
This is a research experiment repository for predicting a user's next Amazon Beauty product. It tests whether adding a user-item interaction graph to text-code-based generative recommendation improves accuracy.

### Cited Findings
- README names it Graph-TIGER: Amazon Beauty Recommendation and describes the graph extension. [README](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/README.md:1)
- Product metadata is converted to Sentence-T5-base vectors; text includes title, price, brand and categories. [Embedding implementation](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/embed_items.py:34)
- The paper-protocol orchestration downloads Amazon Beauty 5-core reviews and metadata, computes text embeddings, builds RQ-VAE and residual-k-means IDs, trains LightGCN, runs TIGER, and compares RPG-style predictors with and without graph gates. [Run script](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/run_all.sh:6)
- The model trains on user item sequences and reserves the last two interactions for validation and test. [Sequence split](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/pgen.py:79)
- LightGCN learns from training-only user-item edges, excluding each sequence's last two interactions. [Graph training](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/lightgcn5.py:30)

### Inferences
- A simple beginner framing: use what each product is, what the user used before, and patterns shared by other shoppers to recommend the next product.
- README line 3's claim that TIGER/RPG never see who bought what together is misleading: their sequence prediction does use interaction history. The graph adds explicit collaborative structure; it does not introduce behavior into an otherwise behavior-free model.

### Gaps
- Training has not been rerun; this is source inspection, not a reproduction audit.

## Which papers and implementations are involved?

### Takeaway
TIGER is the starting point, RPG is the later stronger baseline and architectural influence, LightGCN supplies the graph signal, and ACERec-inspired additions are exploratory. The repo is a custom combination and study, not an official implementation of all these papers.

### Cited Findings
- Local TIGER uses an encoder-decoder T5 with 4 encoder and 4 decoder layers, while the RPG-style experiment uses a GPT2 backbone and 32 parallel prediction heads by default. [TIGER model](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/src/tiger/model.py:8); [RPG-style model](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/pgen.py:86)
- RPG-style input item representations average code-token embeddings, then optionally add a projected LightGCN vector multiplied by a learned sigmoid gate. [Gated fusion implementation](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/pgen.py:105)
- RPG-style product codes are made via PCA followed by OPQ/product quantization; 32 code components and 256 values each are default settings. [Quantization](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/pgen.py:27)
- The author labels the local RPG implementation a reimplementation, and cites a separately run official Meta baseline. [Limitations](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/presentation/paper_repro/RESULTS.md:46)
- The README identifies the official RPG repository as [facebookresearch/RPG_KDD2025](https://github.com/facebookresearch/RPG_KDD2025), run separately with Sentence-T5-base replacing OpenAI embeddings. [Local instructions](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/README.md:83)
- Bundled third-party TIGER code is copied from [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER), commit f1e30af675a8ffec47220f323581e4aed1aabafa, with local changes documented. This provenance alone does not establish it as paper authors' official code. [Upstream documentation](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/third_party/xiaolongtao_tiger/UPSTREAM.md:3)
- ACERec-inspired attentive merger, intent token and alignment loss are configurable in ace.py. Results explicitly state the implementation was built from paper text without the authors' code. [ACE options](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/scripts/paper/ace.py:24); [ACE limitations](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/presentation/paper_repro/RESULTS.md:66)

### Inferences
- For a basic introduction, explain all four roles briefly, rather than presenting this as a faithful TIGER-only reproduction.

### Gaps
- Paper metadata and official-code status require independent internet verification by the web researcher.

## What does the repo report, and where should a beginner look?

### Takeaway
The latest main claim is an improvement from adding graph features at the model input rather than changing product codes. Results are reported by the repo and have not been independently verified here.

### Cited Findings
- Two distinct settings exist: earlier full-data split with 147,241 users and 163,257 products; later paper-style Beauty 5-core with 22,363 users, 12,101 items and 198,502 interactions. Their results should not be compared directly. [README stages](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/README.md:19); [Protocol](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/presentation/paper_repro/RESULTS.md:3)
- Paper-style results report Recall@10 0.0725 without graph and 0.0767 with gated LightGCN, averaged over six seeds, relative gain 5.8%. [Results](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/presentation/paper_repro/RESULTS.md:14)
- ACERec-style additions did not outperform the plain architecture in the documented single-seed tests, with acknowledged reproduction limits. [ACE results](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/presentation/paper_repro/RESULTS.md:66)
- README maps src to models/evaluation, scripts/paper to the latest paper-style experiments, third_party to copied upstream TIGER, and presentation to result tables/logs. [Repository map](/C:/Users/janam/Desktop/MLG/graph-tiger-recsys/README.md:59)

### Inferences
- The most useful starting points are README, the overall pipeline in scripts/paper/run_all.sh, and the model input/gate in pgen.py; detailed metrics can wait until the user asks.

### Gaps
- The stated cause of the TIGER reproduction gap is the repo authors' diagnosis, not a proven exclusion of every other cause. Avoid repeating that causal claim as established fact.
