# Upstream source

Source: https://github.com/XiaoLongtaoo/TIGER (MIT license, see LICENSE)
Commit: f1e30af675a8ffec47220f323581e4aed1aabafa

Copied: Python source only. Compiled files, checkpoints and the data folder were left out.

## Local changes
- `rqvae/datasets.py`: also reads `.npy` embedding files.
- `rqvae/generate_code.py`: takes checkpoint and output paths as arguments instead of
  hardcoded values; loads with `weights_only=False` for current PyTorch.
- `model/main.py`:
  - seeds before building the model, so initialization is reproducible;
  - adds `--selection_metric` (default `NDCG@20`, as upstream) and `--metrics_path`;
  - evaluates the test split once at the end instead of on every validation improvement;
  - creates output directories.

## Not copied
`data/process.ipynb`. Its embeddings are saved in metadata-file order while `model/dataset.py`
treats row i as item i + 1. `scripts/prepare_tiger_5core.py` replaces it and sorts by item.
