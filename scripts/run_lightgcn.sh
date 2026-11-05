#!/bin/bash
# Full LightGCN run on the VCL GPU machine (no scheduler). Safe to re-run after a
# dropped session or new reservation: it resumes from results/lightgcn/last.pt.
# Extra arguments pass through, e.g. scripts/run_lightgcn.sh --lr 5e-4
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/run_lightgcn.py --resume "$@"
python scripts/lightgcn_neighbors.py | tee results/lightgcn/neighbors.txt
