#!/bin/bash
# Reproduce the XiaoLongtaoo/TIGER 5-core Beauty result as a check on the code.
# Needs scripts/prepare_tiger_5core.py and scripts/embed_items_st5.py --dataset 5core first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/tiger_5core"
DATA="$ROOT/data/tiger_5core/Beauty"
PY="$ROOT/.venv/bin/python"
mkdir -p "$OUT"

cd "$ROOT/third_party/xiaolongtao_tiger/rqvae"
"$PY" main.py --num_workers 0 --data_path "$DATA/item_emb.npy" --ckpt_dir "$OUT/rqvae"
CKPT="$(ls -d "$OUT"/rqvae/*/ | tail -1)best_collision_model.pth"
"$PY" generate_code.py --ckpt_path "$CKPT" --output_file "$OUT/codes.npy"

cd "$ROOT/third_party/xiaolongtao_tiger/model"
"$PY" main.py --dataset_path "$DATA" --code_path "$OUT/codes.npy" \
  --log_path "$OUT/t5.log" --save_path "$OUT/t5.pth" --metrics_path "$OUT/metrics.json" "$@"
