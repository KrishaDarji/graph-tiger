#!/bin/bash
# usage: bash scripts/prepare_data.sh Beauty beauty   (Amazon category name, folder name under data/)
set -e
cat=$1
dir=data/$2
url=https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles
mkdir -p "$dir"
[ -f "$dir/reviews.json.gz" ] || curl -L -o "$dir/reviews.json.gz" "$url/reviews_${cat}_5.json.gz"
[ -f "$dir/meta.json.gz" ] || curl -L -o "$dir/meta.json.gz" "$url/meta_${cat}.json.gz"
[ -f "$dir/emb.npy" ] || python -m scripts.embed_items --data "$dir"
[ -f "$dir/graph.npy" ] || python -m scripts.build_graph --data "$dir"
