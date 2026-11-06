"""Decode T5 beam output into ranked items and score with the project's metrics.

Item row i of a codes file is item_idx i. T5 token for code c at level l is
c + l * codebook_size + 1, because token 0 is padding.
"""
import numpy as np
import torch

from transformers import LogitsProcessor, LogitsProcessorList

from src.evaluation import evaluate_rankings


class CodeTable:
    """Lookup between semantic-ID token tuples and item_idx, plus valid prefixes."""

    def __init__(self, codes, codebook_size=256):
        codes = np.asarray(codes, dtype=np.int64)
        if codes.max() >= codebook_size:
            raise ValueError(f"Code {codes.max()} does not fit codebook size {codebook_size}")
        offsets = np.arange(codes.shape[1]) * codebook_size + 1
        self.tokens = codes + offsets
        keys = [tuple(row) for row in self.tokens.tolist()]
        self.token_to_item = {key: index for index, key in enumerate(keys)}
        if len(self.token_to_item) != len(keys):
            raise ValueError("Semantic IDs are not unique")
        allowed = {}
        for key in keys:
            for depth in range(len(key)):
                allowed.setdefault(key[:depth], set()).add(key[depth])
        self.allowed = {prefix: sorted(values) for prefix, values in allowed.items()}

    def prefix_allowed_tokens(self, _batch_id, input_ids):
        """For generate(): input_ids starts with the decoder start token."""
        return self.allowed.get(tuple(input_ids[1:].tolist()), [0])

    def decode(self, beams):
        """Turn (users, beams, levels) token rows into item_idx lists; drop invalid ones."""
        lookup = self.token_to_item
        return [[lookup[tuple(beam)] for beam in user if tuple(beam) in lookup]
                for user in beams]


class PrefixConstraint(LogitsProcessor):
    """Tensor version of CodeTable.prefix_allowed_tokens, fast enough for every beam."""

    def __init__(self, table, device, base=1025):
        self.powers = base ** torch.arange(table.tokens.shape[1], device=device)
        self.levels = []
        for depth in range(table.tokens.shape[1]):
            prefixes = sorted(p for p in table.allowed if len(p) == depth)
            children = [table.allowed[p] for p in prefixes]
            lengths = torch.tensor([len(c) for c in children], device=device)
            ends = lengths.cumsum(0)
            self.levels.append({
                "keys": torch.tensor([sum(t * base ** i for i, t in enumerate(p)) for p in prefixes], device=device),
                "starts": ends - lengths,
                "lengths": lengths,
                "tokens": torch.tensor([t for c in children for t in c], device=device),
            })
            order = self.levels[-1]["keys"].argsort()
            for name in ("keys", "starts", "lengths"):
                self.levels[-1][name] = self.levels[-1][name][order]

    def __call__(self, input_ids, scores):
        depth = input_ids.shape[1] - 1
        if depth >= len(self.levels):
            return scores
        level = self.levels[depth]
        keys = (input_ids[:, 1:] * self.powers[:depth]).sum(1)
        index = torch.searchsorted(level["keys"], keys).clamp_max(level["keys"].numel() - 1)
        valid = level["keys"][index] == keys
        counts = level["lengths"][index] * valid
        rows = torch.arange(keys.numel(), device=keys.device).repeat_interleave(counts)
        first = (level["starts"][index]).repeat_interleave(counts)
        group_start = (counts.cumsum(0) - counts).repeat_interleave(counts)
        columns = level["tokens"][first + torch.arange(rows.numel(), device=keys.device) - group_start]
        constrained = torch.full_like(scores, -torch.inf)
        constrained[rows, columns] = scores[rows, columns]
        return constrained


@torch.no_grad()
def generate_rankings(model, loader, table, beam_size, device, constrained=True):
    """Return one ranked item_idx list per example, in loader order."""
    model.eval()
    levels = table.tokens.shape[1]
    rankings = []
    extra = {"logits_processor": LogitsProcessorList([PrefixConstraint(table, device)])} if constrained else {}
    for batch in loader:
        input_ids = batch["history"].to(device)
        output = model.generate(input_ids=input_ids, attention_mask=batch["attention_mask"].to(device),
                                num_beams=beam_size, **extra)[:, 1:]
        if output.shape[1] < levels:  # every beam ended early on the padding token
            output = torch.nn.functional.pad(output, (0, levels - output.shape[1]))
        rows = output.reshape(input_ids.shape[0], beam_size, -1).cpu().tolist()
        rankings.extend(table.decode(rows))
    return rankings


def apply_mask(rankings, users, seen):
    """Drop each user's seen items. seen maps user_idx to a set of item_idx."""
    return [[item for item in ranking if item not in seen.get(user, ())]
            for ranking, user in zip(rankings, users)]


def score(rankings, targets, ks=(5, 10)):
    return evaluate_rankings(rankings, list(targets), ks=ks)


def real_item_counts(rankings):
    counts = np.array([len(r) for r in rankings])
    return {"mean_real_items": float(counts.mean()), "min_real_items": int(counts.min()),
            "fraction_users_under_10": float((counts < 10).mean())}


@torch.no_grad()
def generate_scored(model, loader, table, beam_size, device):
    """Return per-example (items, log-probability scores), best first, with constrained beams."""
    model.eval()
    constraint = LogitsProcessorList([PrefixConstraint(table, device)])
    results = []
    for batch in loader:
        input_ids = batch["history"].to(device)
        output = model.generate(input_ids=input_ids, attention_mask=batch["attention_mask"].to(device),
                                num_beams=beam_size, logits_processor=constraint,
                                return_dict_in_generate=True, output_scores=True)
        rows = output.sequences[:, 1:].reshape(input_ids.shape[0], beam_size, -1).cpu().tolist()
        scores = output.sequences_scores.reshape(input_ids.shape[0], beam_size).cpu().tolist()
        for beams, beam_scores in zip(rows, scores):
            kept = [(table.token_to_item[tuple(b)], s) for b, s in zip(beams, beam_scores)
                    if tuple(b) in table.token_to_item]
            results.append(([i for i, _ in kept], [s for _, s in kept]))
    return results
