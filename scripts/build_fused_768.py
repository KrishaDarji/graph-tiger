"""Build the 768-D graph-enhanced input: normalize(content + alpha * normalize(graph)).

Both sources are 768-D, so the RQ-VAE input size matches plain TIGER. The graph term is 0
for items with no training edges. Rows stay in item_idx order.
"""
import argparse
from pathlib import Path

import numpy as np


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise SystemExit(f"{int((norms < 1e-8).sum())} rows have zero norm")
    return matrix / norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=Path("data/processed/item_st5_emb.npy"))
    parser.add_argument("--graph-dir", type=Path, default=Path("results/lightgcn_768"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("data/processed/item_fused768_a1.npy"))
    args = parser.parse_args()
    content = normalize(np.load(args.content).astype(np.float64))
    graph = np.load(args.graph_dir / "item_emb.npy").astype(np.float64)
    cold = np.load(args.graph_dir / "cold_item_mask.npy")
    if content.shape != graph.shape or content.shape[1] != 768:
        raise SystemExit(f"Expected matching 768-D inputs, got {content.shape} and {graph.shape}")
    term = np.zeros_like(graph)
    term[~cold] = normalize(graph[~cold])
    fused = normalize(content + args.alpha * term)
    if not np.allclose(fused[cold], content[cold]):
        raise SystemExit("Cold items must equal content")
    temp = args.output.with_name(args.output.stem + ".tmp.npy")
    np.save(temp, fused.astype(np.float32))
    temp.replace(args.output)
    print(f"wrote {args.output} {fused.shape}; mean cosine to content on non-cold items "
          f"{(content[~cold] * fused[~cold]).sum(1).mean():.3f}")


if __name__ == "__main__":
    main()
