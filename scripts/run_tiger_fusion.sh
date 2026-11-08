#!/bin/bash
# Train TIGER on the two new fusion inputs at the same time, one screen session each. They use
# the same RQ-VAE and T5 settings as Runs 1 and 2. Re-running is safe: finished stages are skipped
# and T5 resumes. Build the inputs first with build_concat_1536.py and build_mlp_fused_768.py.
cd "$(dirname "$0")/.."
ROOT="$PWD"
for run in concat1536 mlp768; do
  [ -f "data/processed/item_$run.npy" ] || { echo "missing data/processed/item_$run.npy"; exit 1; }
  screen -dmS "tiger_$run" bash -c "scripts/run_tiger_model.sh $run $ROOT/data/processed/item_$run.npy >> results/tiger_${run}_console.log 2>&1"
done
# Each run uploads its own folder when it finishes. Upload the MLP and its output here, since
# GPU training makes them hard to rebuild exactly.
.venv/bin/python scripts/hf_sync.py push results/fusion_mlp768 data/processed/item_mlp768.npy \
  || echo "UPLOAD FAILED for fusion_mlp768"
screen -ls
