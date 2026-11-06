#!/bin/bash
# Wait for plain TIGER and 768-D LightGCN to finish, then train the graph-enhanced model.
# Safe to re-run: finished stages are skipped and T5 resumes.
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/results/tiger_full/queue.log"
note() { echo "$(date +'%F %T') $*" | tee -a "$LOG"; }
while [ ! -f results/lightgcn_768/metrics_test.json ] || screen -ls 2>/dev/null | grep -q tiger_queue; do sleep 60; done
note "graph768 build"
.venv/bin/python scripts/build_fused_768.py || { note "graph768 build FAILED"; exit 1; }
note "graph768 start"
if scripts/run_tiger_model.sh graph768 "$ROOT/data/processed/item_fused768_a1.npy" >> results/tiger_graph768_console.log 2>&1; then
  note "graph768 done"
else
  note "graph768 FAILED, see results/tiger_graph768_console.log"
fi
