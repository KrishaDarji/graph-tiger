"""LightGCN model, normalized user-item graph, and BPR loss."""
import torch
from torch import nn
from torch.nn import functional as F


def build_normalized_graph(user_indices, item_indices, n_users, n_items, device):
    """Return D^-1/2 A D^-1/2 over (users + items) nodes as a sparse tensor.

    Repeated (user, item) pairs collapse to one unweighted edge. Item node j is
    n_users + j.
    """
    pairs = torch.unique(torch.stack([user_indices, item_indices], dim=1), dim=0).to(device)
    users, items = pairs[:, 0], pairs[:, 1] + n_users
    n_nodes = n_users + n_items
    rows = torch.cat([users, items])
    cols = torch.cat([items, users])
    values = torch.ones(rows.numel(), device=device)
    degree = torch.zeros(n_nodes, device=device).scatter_add_(0, rows, values)
    weights = values / degree[rows].sqrt() / degree[cols].sqrt()
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), weights, (n_nodes, n_nodes), check_invariants=False,
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_users + n_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self, graph):
        """Mean of layers 0..K, as a (n_users + n_items, dim) tensor."""
        embeddings = [self.embedding.weight]
        current = self.embedding.weight
        for _ in range(self.n_layers):
            current = torch.sparse.mm(graph, current)
            embeddings.append(current)
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def split_embeddings(self, graph):
        """Return (user_embeddings, item_embeddings); item row j is item_idx j."""
        return torch.split(self.propagate(graph), [self.n_users, self.n_items])

    def forward(self, graph, users, positives, negatives):
        user_emb, item_emb = self.split_embeddings(graph)
        user = user_emb[users]
        return (user * item_emb[positives]).sum(1), (user * item_emb[negatives]).sum(1)

    def l2_penalty(self, users, positives, negatives):
        """0.5 * squared L2 of the batch's layer-0 embeddings, per example."""
        layer0 = self.embedding.weight
        batch = torch.cat([
            layer0[users],
            layer0[self.n_users + positives],
            layer0[self.n_users + negatives],
        ])
        return 0.5 * batch.pow(2).sum() / users.numel()


def bpr_loss(positive_scores, negative_scores):
    return F.softplus(negative_scores - positive_scores).mean()


class NegativeSampler:
    """Uniform item negatives, rejecting each user's own training items."""

    def __init__(self, users, items, n_items):
        self.n_items = n_items
        self.keys = torch.sort(users * n_items + items).values

    def _is_positive(self, users, items):
        query = users * self.n_items + items
        position = torch.searchsorted(self.keys, query).clamp_max(self.keys.numel() - 1)
        return self.keys[position] == query

    def sample(self, users):
        negatives = torch.randint(self.n_items, users.shape, device=users.device)
        collisions = self._is_positive(users, negatives).nonzero(as_tuple=True)[0]
        while collisions.numel():
            negatives[collisions] = torch.randint(
                self.n_items, collisions.shape, device=users.device
            )
            still = self._is_positive(users[collisions], negatives[collisions])
            collisions = collisions[still]
        return negatives
