import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from .vocab import DECODER_START_ID


class PrefixTable:

    def __init__(self, codes, level_sizes):
        codes = np.asarray(codes, dtype=np.int64)
        self.level_sizes = tuple(level_sizes)
        self.num_levels = len(level_sizes)
        self.keys, self.masks = [], []
        for depth in range(self.num_levels):
            prefix_keys = (codes[:, :depth] * self._weights(depth)).sum(axis=1)
            unique, inverse = np.unique(prefix_keys, return_inverse=True)
            mask = np.zeros((len(unique), self.level_sizes[depth]), dtype=bool)
            mask[inverse.reshape(-1), codes[:, depth]] = True
            self.keys.append(torch.as_tensor(unique))
            self.masks.append(torch.as_tensor(mask))
        full = (codes * self._weights(self.num_levels)).sum(axis=1)
        order = np.argsort(full)
        self.full_keys = torch.as_tensor(full[order])
        self.full_items = torch.as_tensor(order)

    def to(self, device):
        self.keys = [k.to(device) for k in self.keys]
        self.masks = [m.to(device) for m in self.masks]
        self.full_keys, self.full_items = self.full_keys.to(device), self.full_items.to(device)
        return self

    def _weights(self, depth):
        return np.array(
            [int(np.prod(self.level_sizes[l + 1 : depth])) for l in range(depth)], dtype=np.int64
        )

    def _key(self, prefix, depth):
        if depth == 0:
            return torch.zeros(prefix.size(0), dtype=torch.long, device=prefix.device)
        weights = torch.as_tensor(self._weights(depth), device=prefix.device)
        return (prefix * weights).sum(dim=1)

    def allowed(self, prefix, depth):
        keys = self.keys[depth].to(prefix.device)
        query = self._key(prefix, depth)
        pos = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
        found = keys[pos] == query
        return self.masks[depth].to(prefix.device)[pos] & found.unsqueeze(1)

    def lookup(self, codes):
        keys = self.full_keys.to(codes.device)
        query = self._key(codes, self.num_levels)
        pos = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
        found = keys[pos] == query
        items = self.full_items.to(codes.device)[pos]
        return torch.where(found, items, torch.full_like(items, -1))


@torch.no_grad()
def beam_search(model, input_ids, attention_mask, vocab, beam_size, table=None, constrained=False):
    device = input_ids.device
    batch, levels = input_ids.size(0), vocab.num_levels
    encoder_hidden = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    tokens = torch.full((batch, 1, 1), DECODER_START_ID, dtype=torch.long, device=device)
    codes = torch.zeros((batch, 1, 0), dtype=torch.long, device=device)
    scores = torch.zeros((batch, 1), device=device)

    for level in range(levels):
        nb = tokens.size(1)
        lo, hi = vocab.level_range(level)
        enc = encoder_hidden.repeat_interleave(nb, dim=0)
        mask = attention_mask.repeat_interleave(nb, dim=0)
        out = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc),
            attention_mask=mask,
            decoder_input_ids=tokens.reshape(batch * nb, -1),
            use_cache=False,
        )
        logp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)[:, lo:hi]  # [B*nb, K]
        if constrained:
            allowed = table.allowed(codes.reshape(batch * nb, level), level)
            logp = logp.masked_fill(~allowed, float("-inf"))
        total = (scores.reshape(-1, 1) + logp).reshape(batch, nb * (hi - lo))
        k = min(beam_size, total.size(1))
        scores, flat = total.topk(k, dim=1)
        parent, code = flat // (hi - lo), flat % (hi - lo)
        tokens = torch.cat(
            [tokens.gather(1, parent.unsqueeze(-1).expand(-1, -1, tokens.size(2))), (code + lo).unsqueeze(-1)],
            dim=2,
        )
        codes = torch.cat(
            [codes.gather(1, parent.unsqueeze(-1).expand(-1, -1, codes.size(2))), code.unsqueeze(-1)], dim=2
        )
    return codes, scores
