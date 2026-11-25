import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

from .tokens import K


class Decoder(nn.Module):
    """RPG-style decoder: item = mean of its digit embeddings (+ gated graph vector), one cosine head per digit."""

    def __init__(self, codes, graph=None, dim=448, layers=2, heads=4, ffn=1024, drop=0.5, max_len=50, temp=0.03):
        super().__init__()
        self.n_items, self.digits, self.temp = len(codes), codes.shape[1], temp
        self.register_buffer("codes", codes, persistent=False)
        tokens = codes + torch.arange(self.digits, device=codes.device) * K + 1
        self.register_buffer("tokens", torch.cat([tokens, torch.zeros(1, self.digits, dtype=torch.long, device=codes.device)]), persistent=False)
        self.gpt = GPT2Model(GPT2Config(vocab_size=self.digits * K + 2, n_positions=max_len, n_embd=dim, n_layer=layers, n_head=heads, n_inner=ffn,
                                        activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=drop, attn_pdrop=drop,
                                        layer_norm_epsilon=1e-12, initializer_range=0.02))
        self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.digits)])
        for h in self.heads:
            nn.init.zeros_(h.weight)
        self.register_buffer("graph", graph.clone() if graph is not None else None)
        if graph is not None:
            self.proj = nn.Linear(graph.shape[1], dim)
            self.gate = nn.Linear(dim + graph.shape[1], dim)

    @property
    def wte(self):  # GPT-2's own token embedding (its init), as in RPG; row 0 is padding
        return self.gpt.wte

    def item_emb(self, ids):
        e = self.wte(self.tokens[ids]).mean(-2)
        if self.graph is None:
            return e
        g = self.graph[ids]
        return e + torch.sigmoid(self.gate(torch.cat([e, g], -1))) * self.proj(g)

    def hidden(self, ids, mask):
        return self.gpt(inputs_embeds=self.item_emb(ids), attention_mask=mask).last_hidden_state

    def digit_states(self, h):
        return torch.stack([h + F.silu(head(h)) for head in self.heads], -2)

    def logits(self, s):
        e = F.normalize(self.wte.weight[1:-1], dim=-1).view(self.digits, K, -1)
        return torch.einsum("...nd,nkd->...nk", F.normalize(s, dim=-1), e) / self.temp
