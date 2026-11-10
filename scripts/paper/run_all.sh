#!/bin/bash
# Paper-protocol experiments on 5-core Beauty. Run inside screen; every step is skipped if its output exists.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
pip install -q faiss-cpu
URL=https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles
RV=data/tiger_5core/raw/reviews_Beauty_5.json.gz
META=data/raw/meta_Beauty.json.gz
D=data/paper
R=results/paper
mkdir -p data/tiger_5core/raw data/raw $D $R logs
[ -f $RV ] || curl -L -o $RV $URL/reviews_Beauty_5.json.gz
[ -f $META ] || curl -L -o $META $URL/meta_Beauty.json.gz

[ -f $D/st5_base/emb.npy ] || python scripts/paper/embed_items.py --reviews $RV --meta $META --out $D/st5_base
[ -f $D/rq_base.npy ] || python scripts/paper/rqvae.py --emb $D/st5_base/emb.npy --out $D/rq_base.npy --lr 0.05 > $R/rqvae_base.log
[ -f $D/rk_base.npy ] || python scripts/paper/rkmeans.py --emb $D/st5_base/emb.npy --out $D/rk_base.npy
[ -f $D/lightgcn5/item_emb.npy ] || python scripts/paper/lightgcn5.py --reviews $RV --items $D/st5_base/items.txt --out $D/lightgcn5 > $R/lightgcn5.log
G=$D/lightgcn5/item_emb.npy

# TIGER: T5 on 4-token semantic IDs (validation peaks near 20k steps, so 30k is enough)
for codes in rq_base rk_base; do
  for s in 42 43 44; do
    [ -f $R/t5_${codes}_s$s/metrics_test.json ] || python scripts/paper/train_t5.py --interactions $RV --kcore 0 --min-user-len 5 \
      --semantic-ids $D/$codes.npy --item-order $D/st5_base/items.txt --output-dir $R/t5_${codes}_s$s --seed $s \
      --max-steps 30000 --val-metric-every 5000 --val-metric-users 22363 > logs/t5_${codes}_s$s.log 2>&1
  done
done

# RPG-style model, with and without the gated LightGCN input
for s in 2024 2025 2026 2027 2028 2029; do
  [ -f $R/pgen_base_s$s/result.json ] || python scripts/paper/pgen.py --reviews $RV --items $D/st5_base/items.txt \
    --emb $D/st5_base/emb.npy --out $R/pgen_base_s$s --seed $s > logs/pgen_base_s$s.log 2>&1
  [ -f $R/pgen_gate_s$s/result.json ] || python scripts/paper/pgen.py --reviews $RV --items $D/st5_base/items.txt \
    --emb $D/st5_base/emb.npy --graph gate --graph-emb $G --out $R/pgen_gate_s$s --seed $s > logs/pgen_gate_s$s.log 2>&1
done

python scripts/paper/summarize.py $R
