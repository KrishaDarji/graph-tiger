# Papers and official implementations

## What papers underpin the project?

### Takeaway
The project belongs to sequential recommendation: predict a user's next item from their earlier interactions. TIGER introduced generating a product's semantic code; RPG predicts longer codes in parallel; ACERec compresses long codes intelligently before sequence modeling.

### Cited Findings
- TIGER's exact title is *Recommender Systems with Generative Retrieval*, NeurIPS 2023. It represents items as discrete, content-derived Semantic IDs and trains a sequence-to-sequence Transformer to generate the next item's ID from interaction history. Beginner analogy: give each product a short meaningful address, then predict the next address. The learned tokens are not necessarily readable category names. — [Paper](https://arxiv.org/abs/2305.05065); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)
- RPG's exact title is *Generating Long Semantic IDs in Parallel for Recommendation*, KDD 2025, by Yupeng Hou et al. It uses unordered long codes and predicts all tokens in parallel, then uses a graph over similar semantic IDs for valid decoding. Its semantic-ID graph is distinct from the local project's user-item LightGCN graph. — [Paper](https://arxiv.org/abs/2506.05781); [Authors' code](https://github.com/facebookresearch/RPG_KDD2025)
- ACERec's exact title is *Unleash the Potential of Long Semantic IDs for Generative Recommendation*, 2026 preprint (arXiv submitted February 14, revised August 2, currently labeled under review). It uses an Attentive Token Merger to compress long item codes into compact latent tokens and an Intent Token to predict next items, trained with token-level generation plus item-level alignment. — [Current abstract](https://arxiv.org/abs/2602.13573); [Full current paper](https://arxiv.org/html/2602.13573v2)
- LightGCN's exact title is *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR 2020. It learns collaborative representations by propagating embeddings over user-item graph connections. — [Paper](https://arxiv.org/abs/2002.02126); [Authors' repository](https://github.com/kuandeng/LightGCN)

### Inferences
- These are related research methods, not three dependencies or mandatory stages of one pipeline. The local repo's actual use must be established from its code and notes.

### Gaps
- Do not describe current reported ACERec numbers as verified experimental facts here; this task only establishes method identity and provenance.

## Which implementations are official?

### Takeaway
RPG has clearly verified official code. The TIGER code bundled in this repo is explicitly unofficial; no official Google release was found. No author-linked ACERec implementation was found in targeted searches or its paper.

### Cited Findings
- The RPG abstract explicitly links `facebookresearch/RPG_KDD2025`; its README calls this code for the authors' KDD 2025 paper. This supports calling it official. — [Paper](https://arxiv.org/abs/2506.05781); [Code](https://github.com/facebookresearch/RPG_KDD2025)
- `XiaoLongtaoo/TIGER`, from which the local third-party folder was copied, labels itself '[Pytorch] Unofficial Implementation'. — [Repository](https://github.com/XiaoLongtaoo/TIGER)
- The LightGCN authors' TensorFlow repo explicitly links a PyTorch implementation supplied by the team. — [TensorFlow authors' code](https://github.com/kuandeng/LightGCN); [Author-linked PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)

### Inferences
- Safe wording: 'I found official RPG and LightGCN code. I did not find an official TIGER or ACERec release; your bundled TIGER implementation is unofficial.' Do not assert universally that no official implementation exists.

### Gaps
- A guessed Google Research publication URL was inaccessible and exact-title Google-domain searches returned no usable Google publication page. Use arXiv, NeurIPS, and the authors' own pages instead.
- An unrelated old `mickeysjm/AceRec` repository exists; it is a topic-model project, not this 2026 semantic-ID ACERec. Do not link it.

## What learning materials are available?

### Takeaway
For later explanations, the most credible readily available materials are the paper's NeurIPS poster/video page and an author-led tutorial. The introductory reply need not list all materials.

### Cited Findings
- The official TIGER NeurIPS poster page has paper, poster, OpenReview, and a Video section. The page was read, but video playback/transcript was not available through the text tool. — [TIGER presentation page](https://neurips.cc/virtual/2023/poster/72488)
- RPG first author Yupeng Hou presents the CIKM 2025 tutorial *Towards Large Generative Recommendation: A Tokenization Perspective*, with background, semantic IDs, and semantic-ID recommendation sections and slides. — [Tutorial](https://large-genrec.github.io/cikm2025.html)
- TIGER coauthor Nikhil Mehta hosts current generative-retrieval slides, including RQ-VAE and hierarchical Semantic IDs, on his own website. — [Author page](https://nikhil-dce.github.io/); [Slides](https://nikhil-dce.github.io/data/GenRetrieval%20Research.pdf)
- A readable third-party TIGER blog was found, but it is not the basis for technical claims: — [Give Every Item a Semantic Address](https://chenhuiyu.github.io/blog/generative-recommendation-11-tiger-en/)

### Inferences
- Explain using a tiny product-history example and plain language before introducing embeddings, quantization, and Transformers.

### Gaps
- No claim of having watched videos: only the official presentation page and text sources were inspected.
- Third-party blogs were located; technical facts above rely on papers/authors' material as required.
