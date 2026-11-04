"""Lightweight TIGER-style next-item Transformer prototype.

This is the project-scale prototype: item indices act as tokens. Semantic-ID
generation from item metadata is a later extension once the training loop is
validated.
"""
import torch
from torch import nn


class NextItemTransformer(nn.Module):
    def __init__(self, n_items, dim=128, layers=2, heads=4, dropout=0.1,
                 max_length=20, content_features=None):
        super().__init__()
        self.n_items = n_items
        self.max_length = max_length
        self.item_embedding = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.content_projection = None
        if content_features is not None:
            self.content_projection = nn.Linear(content_features, dim, bias=False)
            self.register_buffer("content_embeddings", torch.empty(n_items + 1, content_features))
        self.position_embedding = nn.Embedding(max_length, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, n_items + 1)

    def forward(self, tokens):
        length = tokens.size(1)
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.item_embedding(tokens)
        if self.content_projection is not None:
            content = self.content_projection(self.content_embeddings[tokens])
            x = x + content
        x = x + self.position_embedding(positions)
        padding = tokens.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding)
        lengths = (~padding).sum(dim=1).clamp_min(1) - 1
        pooled = x[torch.arange(x.size(0), device=x.device), lengths]
        return self.output(self.norm(pooled))

