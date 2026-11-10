import numpy as np
import torch

from .data import collate
from .decode import beam_search
from .metrics import rank_of_target, summarize


@torch.no_grad()
def evaluate(model, examples, index, table, device, beam_size=20, batch_size=256, ks=(5, 10),
             constrained=False, filter_history=False, use_user_token=True, limit=0, item_groups=None,
             report_filtered=False):
    model.eval()
    n = len(examples) if not limit else min(limit, len(examples))
    all_ranks, all_filtered, all_targets, invalid, total_beams = [], [], [], 0, 0
    for start in range(0, n, batch_size):
        rows = np.arange(start, min(start + batch_size, n))
        input_ids, attention_mask, _, target, hist = collate(
            examples, rows, index, torch, device, use_user_token
        )
        codes, _ = beam_search(
            model, input_ids, attention_mask, index.vocab, beam_size, table, constrained
        )
        items = table.lookup(codes.reshape(-1, codes.size(-1))).reshape(codes.shape[0], -1)
        invalid += int((items < 0).sum())
        total_beams += items.numel()
        all_targets.append(target.cpu().numpy())
        if report_filtered:  # same beams, history items removed from the ranking
            all_filtered.append(rank_of_target(
                items.cpu().numpy(), target.cpu().numpy(), hist.cpu().numpy(), index.pad_index))
        all_ranks.append(
            rank_of_target(
                items.cpu().numpy(), target.cpu().numpy(),
                hist.cpu().numpy() if filter_history else None, index.pad_index,
            )
        )
    ranks, targets = np.concatenate(all_ranks), np.concatenate(all_targets)
    metrics = summarize(ranks, ks)
    for name, mask in (item_groups or {}).items():
        sel = np.asarray(mask)[targets]
        metrics[f"group_{name}"] = {**summarize(ranks[sel], ks), "num_users": int(sel.sum())} if sel.any() else {"num_users": 0}
    if report_filtered:
        metrics["with_history_filter"] = summarize(np.concatenate(all_filtered), ks)
    metrics["invalid_beam_rate"] = invalid / max(total_beams, 1)
    metrics["num_users"] = int(n)
    return metrics


@torch.no_grad()
def validation_loss(model, examples, index, device, batch_size=512, use_user_token=True, limit=0):
    model.eval()
    n = len(examples) if not limit else min(limit, len(examples))
    total, count = 0.0, 0
    for start in range(0, n, batch_size):
        rows = np.arange(start, min(start + batch_size, n))
        input_ids, attention_mask, labels, _, _ = collate(examples, rows, index, torch, device, use_user_token)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        total += float(loss) * len(rows)
        count += len(rows)
    return total / max(count, 1)
