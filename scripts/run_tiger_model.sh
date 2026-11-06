#!/bin/bash
# Train one model end to end: RQ-VAE, semantic IDs, T5, backup. Re-running resumes:
# finished stages are skipped and T5 continues from last.pt.
# Usage: scripts/run_tiger_model.sh NAME EMBEDDINGS.npy [extra run_tiger_t5.py args]
set -euo pipefail
NAME="$1"; EMB="$2"; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/tiger_full/$NAME"
PY="$ROOT/.venv/bin/python"
BACKUP=/mnt/ncsudrive/v/vvanera/graph-tiger-backup/tiger_full
RQVAE_EPOCHS="${RQVAE_EPOCHS:-225}"
# A marker file drops a model from the queue without touching the running job.
if [ -f "$ROOT/results/tiger_full/SKIP_$NAME" ]; then echo "skipping $NAME"; exit 0; fi
mkdir -p "$OUT"

if [ ! -f "$OUT/codes.npy" ]; then
  rm -rf "$OUT/rqvae"
  cd "$ROOT/third_party/xiaolongtao_tiger/rqvae"
  "$PY" main.py --epochs "$RQVAE_EPOCHS" --warmup_epochs "$((RQVAE_EPOCHS / 10))" --eval_step 25 --num_workers 0 \
    --data_path "$EMB" --ckpt_dir "$OUT/rqvae" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'evaluating|Best|Traceback|Error' || true
  CKPT="$(ls -d "$OUT"/rqvae/*/ | tail -1)best_collision_model.pth"
  "$PY" generate_code.py --ckpt_path "$CKPT" --output_file "$OUT/codes.tmp.npy" 2>&1 | grep -E '^Collision Rate|^Max number|still have|no duplicates|Traceback|Error' || true
  mv "$OUT/codes.tmp.npy" "$OUT/codes.npy"
  "$PY" "$ROOT/scripts/analyze_semantic_ids.py" "$OUT/codes.npy" | tee "$OUT/semantic_ids.json"
fi

cd "$ROOT"
"$PY" scripts/run_tiger_t5.py --codes "$OUT/codes.npy" --output-dir "$OUT" "$@"

# A failed backup must not stop the remaining models. The drive login needs renewing overnight.
kinit -R 2>/dev/null || true
mkdir -p "$BACKUP/$NAME" && cp -rp "$OUT/." "$BACKUP/$NAME/" && echo "backed up $NAME" || echo "BACKUP FAILED for $NAME"
