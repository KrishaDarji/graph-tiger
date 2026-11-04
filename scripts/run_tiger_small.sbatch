#!/bin/bash
# Example Hazel submission. Adjust partition/account/time to your allocation.
#SBATCH --job-name=tiger-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:rtx_6000_pro:1
#SBATCH --time=01:50:00
#SBATCH --output=logs/tiger-%j.out

set -euo pipefail
source /usr/local/apps/conda/miniconda3/26.3.2/etc/profile.d/conda.sh
conda activate /share/csc591f26/kdarji/conda/envs/tiger-project

export TIGER_PROJECT_DIR=/share/csc591f26/kdarji/ML-with-Graphs/Project
export TIGER_TARGET_EPOCHS="${TIGER_TARGET_EPOCHS:-5}"
cd "$TIGER_PROJECT_DIR"
mkdir -p slurm_logs results

if [ -f results/tiger_small_checkpoint.pt ]; then
  NEXT_EPOCH=$(python -c 'import torch; print(torch.load("results/tiger_small_checkpoint.pt", map_location="cpu")["epoch"] + 1)')
else
  NEXT_EPOCH=1
fi

python scripts/run_tiger_small.py \
  --epochs "$NEXT_EPOCH" \
  --batch-size 256 --dim 128 --eval-limit 1000 \
  --steps-per-job 100 --checkpoint-every 5 \
  --checkpoint results/tiger_small_checkpoint.pt --resume

CURRENT_EPOCH=$(python -c 'import torch; print(torch.load("results/tiger_small_checkpoint.pt", map_location="cpu")["epoch"])')
if [ "$CURRENT_EPOCH" -le "$TIGER_TARGET_EPOCHS" ]; then
  sbatch --export=ALL,TIGER_TARGET_EPOCHS="$TIGER_TARGET_EPOCHS" "$0"
else
  echo "Completed TIGER training through epoch $CURRENT_EPOCH"
fi

