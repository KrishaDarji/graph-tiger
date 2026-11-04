"""Small LightGCN implementation for the project baseline."""
import torch
from torch import nn


def build_normalized_graph(user_indices, item_indices, n_users, n_items, device):
    n_nodes = n_users + n_items
    rows = torch.cat([user_indices, item_indices + n_users])
    cols = torch.cat([item_indices + n_users, user_indices])
    values = torch.ones(rows.numel(), device=device)
    degree = torch.zeros(n_nodes, device=device).scatter_add_(0, rows, values)
    weights = values / degree[rows].clamp_min(1).sqrt() / degree[cols].clamp_min(1).sqrt()
    indices = torch.stack([rows, cols])
    return torch.sparse_coo_tensor(indices, weights, (n_nodes, n_nodes)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_users + n_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self, graph):
        embeddings = [self.embedding.weight]
        current = self.embedding.weight
        for _ in range(self.n_layers):
            current = torch.sparse.mm(graph, current)
            embeddings.append(current)
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def forward(self, graph, users, positives, negatives):
        all_embeddings = self.propagate(graph)
        user = all_embeddings[users]
        positive = all_embeddings[self.n_users + positives]
        negative = all_embeddings[self.n_users + negatives]
        return (user * positive).sum(1), (user * negative).sum(1)

