#!/bin/bash
# Run every remaining model in order, without needing an open session. Re-running is safe:
# finished work is skipped and interrupted T5 training resumes. One failed model does not
# stop the others. Progress goes to results/tiger_full/queue.log.
cd "$(dirname "$0")/.."
ROOT="$PWD"
BACKUP=/mnt/ncsudrive/v/vvanera/graph-tiger-backup/tiger_full
LOG="$ROOT/results/tiger_full/queue.log"
mkdir -p "$(dirname "$LOG")"
note() { echo "$(date +'%F %T') $*" | tee -a "$LOG"; }

run_model() {
  local name="$1" emb="$2"
  if [ -f "results/tiger_full/$name/metrics_test.json" ]; then note "$name already finished"; return; fi
  note "$name start"
  if scripts/run_tiger_model.sh "$name" "$emb" >> "results/tiger_${name}_console.log" 2>&1; then
    note "$name done"
  else
    note "$name FAILED, see results/tiger_${name}_console.log"
  fi
}

run_model content "$ROOT/data/processed/item_st5_emb.npy"
if [ -f results/tiger_full/content/best.pt ] && [ ! -f results/tiger_full/content/rerank_lightgcn/metrics_test.json ]; then
  note "rerank start"
  if .venv/bin/python scripts/rerank_tiger_lightgcn.py --run-dir results/tiger_full/content >> results/tiger_rerank_console.log 2>&1; then
    note "rerank done"
  else
    note "rerank FAILED, see results/tiger_rerank_console.log"
  fi
  kinit -R 2>/dev/null || true
  mkdir -p "$BACKUP/content" && cp -rp results/tiger_full/content/rerank_lightgcn "$BACKUP/content/" || note "rerank BACKUP FAILED"
fi
run_model proj "$ROOT/data/processed/item_content_proj.npy"
run_model fused "$ROOT/data/processed/item_fused_a1.npy"
note "queue finished"
