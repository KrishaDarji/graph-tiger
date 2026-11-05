"""Build the projected-content control and graph-fused item embeddings.

Fused row = normalize(normalize(PCA(content)) + alpha * normalize(graph)), where the
graph term is 0 for items with no training edges. Rows stay in item_idx order.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise SystemExit(f"{int((norms < 1e-8).sum())} rows have zero norm")
    return matrix / norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=Path("data/processed/item_st5_emb.npy"))
    parser.add_argument("--graph-dir", type=Path, default=Path("results/lightgcn"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    content = np.load(args.content)
    graph = np.load(args.graph_dir / "item_emb.npy")
    cold = np.load(args.graph_dir / "cold_item_mask.npy")
    if not content.shape[0] == graph.shape[0] == cold.shape[0]:
        raise SystemExit("Embedding row counts differ")

    pca = PCA(n_components=graph.shape[1], random_state=args.seed)
    projected = normalize(pca.fit_transform(content).astype(np.float64))
    np.save(args.output_dir / "item_content_proj.npy", projected.astype(np.float32))

    graph_term = np.zeros_like(projected)
    graph_term[~cold] = normalize(graph[~cold].astype(np.float64))
    info = {"pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "n_items": int(content.shape[0]), "dim": int(graph.shape[1]),
            "cold_items": int(cold.sum()), "alphas": args.alphas, "seed": args.seed}
    for alpha in args.alphas:
        fused = normalize(projected + alpha * graph_term)
        if not np.allclose(fused[cold], projected[cold]):
            raise SystemExit("Cold items must equal projected content")
        np.save(args.output_dir / f"item_fused_a{alpha:g}.npy", fused.astype(np.float32))
    (args.output_dir / "item_fused_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
