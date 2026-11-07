"""Build the concatenated input: [normalize(content), normalize(graph)], 1536-D.

Each half has unit norm, so neither dominates by scale. The whole vector is not normalized
again: per-dimension values then stay on the same scale as the 768-D inputs, which keeps the
RQ-VAE's reconstruction and quantization losses balanced as in Runs 1 and 2. The graph half is
0 for items with no training edges. Rows stay in item_idx order.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_fused_768 import normalize


def load_halves(content_path, graph_dir):
    """Return unit-norm content, unit-norm graph (0 for cold items) and the cold-item mask."""
    content = normalize(np.load(content_path).astype(np.float64))
    graph = np.load(graph_dir / "item_emb.npy").astype(np.float64)
    cold = np.load(graph_dir / "cold_item_mask.npy")
    if content.shape != graph.shape or content.shape[1] != 768:
        raise SystemExit(f"Expected matching 768-D inputs, got {content.shape} and {graph.shape}")
    term = np.zeros_like(graph)
    term[~cold] = normalize(graph[~cold])
    return content, term, cold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=Path("data/processed/item_st5_emb.npy"))
    parser.add_argument("--graph-dir", type=Path, default=Path("results/lightgcn_768"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/item_concat1536.npy"))
    args = parser.parse_args()
    content, graph, cold = load_halves(args.content, args.graph_dir)
    joined = np.concatenate([content, graph], axis=1).astype(np.float32)
    temp = args.output.with_name(args.output.stem + ".tmp.npy")
    np.save(temp, joined)
    temp.replace(args.output)
    print(f"wrote {args.output} {joined.shape}; cold items {int(cold.sum())}; per-dimension variance: "
          f"content {content.var(0).mean():.3e}, graph {graph.var(0).mean():.3e}, "
          f"graph on non-cold items {graph[~cold].var(0).mean():.3e}")


if __name__ == "__main__":
    main()
