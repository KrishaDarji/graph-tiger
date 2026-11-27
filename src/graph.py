import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import svds


def transition_svd(seqs, n_items, dim=128, window=3):
    # item pairs within `window` steps, weighted 1/distance; symmetric normalisation; truncated SVD
    rows, cols, vals = [], [], []
    for s in seqs:
        for t, a in enumerate(s):
            for d in range(1, window + 1):
                if t + d < len(s):
                    rows += [a, s[t + d]]
                    cols += [s[t + d], a]
                    vals += [1.0 / d] * 2
    m = sp.coo_matrix((vals, (rows, cols)), shape=(n_items, n_items)).tocsr()
    deg = np.asarray(m.sum(1)).ravel() + 1e-9
    u, sv, _ = svds(sp.diags(deg ** -0.5) @ m @ sp.diags(deg ** -0.5), k=dim)
    return (u * np.sqrt(sv)).astype(np.float32)


def load_graph(path, dev):
    g = torch.tensor(np.load(path), device=dev).float()
    norm = g.norm(dim=1, keepdim=True)
    g = torch.where(norm > 1e-6, g / norm.clamp_min(1e-12), torch.zeros_like(g))  # items without interactions stay zero
    return torch.cat([g, torch.zeros(1, g.shape[1], device=dev)])  # extra zero row for padding


def procrustes(old, new):
    # rotate `new` onto the basis of `old` (orthogonal Procrustes on items present in both)
    both = (old.norm(dim=1) > 0) & (new.norm(dim=1) > 0)
    u, _, vh = torch.linalg.svd(new[both].T @ old[both])
    return new @ (u @ vh)
