"""Train and evaluate the paper-faithful TIGER Transformer on Semantic IDs.

Defaults reproduce the paper's recipe (Sec. 4): 4+4 layer T5, d_model 128, 6x64 heads,
d_ff 1024, dropout 0.1, batch 256, lr 0.01 constant for 10k steps then inverse-sqrt,
200k steps, history capped at 20 items, one hashed user token (2000 buckets) prepended.

VCL has no scheduler: run inside `screen`; the job resumes from <output>/last.pt.
The test split is only evaluated at the end (and refused if metrics_test.json exists).
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tiger.data import (
    build_examples, build_sequences, collate, k_core, load_codes_npy, load_interactions,
    load_project_split, load_semantic_id_table, make_item_index, train_item_degree,
)
from src.tiger.decode import PrefixTable
from src.tiger.evaluate import evaluate, validation_loss
from src.tiger.model import ModelConfig, build_model, count_parameters
from src.tiger.optim import build_optimizer, constant_then_inverse_sqrt
from src.tiger.vocab import TigerVocab


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split-dir", type=Path,
                     help="project split files (data/processed): identical to LightGCN's train/val/test")
    src.add_argument("--interactions", type=Path,
                     help="parquet(user_id,item_id,timestamp) or reviews_*_5.json.gz (re-splits leave-one-out)")
    p.add_argument("--semantic-ids", type=Path, required=True,
                   help=".npy [N,3|4] with row i = item_idx i (needs --split-dir), or a table "
                        "with item_id + c0..c2[,c3] (or semantic_id_0..)")
    p.add_argument("--item-order", type=Path, help="txt with one item_id per line = row order of a .npy codes file")
    p.add_argument("--output-dir", type=Path, default=Path("results/tiger"))
    p.add_argument("--kcore", type=int, default=0, help="only with --interactions; paper uses 5")
    p.add_argument("--min-user-len", type=int, default=3)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--num-codebooks", type=int, default=3, help="RQ levels before the collision code")
    p.add_argument("--user-buckets", type=int, default=2000)
    p.add_argument("--no-user-token", action="store_true", help="ablation (paper Table 8)")
    # model (paper defaults)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-kv", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tie-embeddings", action="store_true")
    # optimisation (paper defaults)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--optimizer", choices=["adafactor", "adamw"], default="adafactor")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--constant-steps", type=int, default=10_000)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast on CUDA (paper trained on TPUs); --no-bf16 for fp32")
    # bookkeeping
    p.add_argument("--val-loss-every", type=int, default=1000)
    p.add_argument("--val-metric-every", type=int, default=10_000)
    p.add_argument("--val-metric-users", type=int, default=5000, help="subset used during training")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--beam-size", type=int, default=20)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate last/best on test")
    p.add_argument("--no-final-eval", action="store_true", help="benchmarks: skip the full val/test evaluation")
    p.add_argument("--time-limit-minutes", type=float, default=0,
                   help="stop cleanly (checkpoint saved) after this wall time; re-run with --resume")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    use_user = not args.no_user_token

    # ---- data -----------------------------------------------------------------------
    if args.split_dir:
        sequences, split_item_ids = load_project_split(args.split_dir)
        n_inter = sum(len(s) for s in sequences.values())
    else:
        interactions = k_core(load_interactions(args.interactions), args.kcore)
        sequences, split_item_ids = build_sequences(interactions), None
        n_inter = len(interactions)
    if args.semantic_ids.suffix == ".npy":
        order = split_item_ids if split_item_ids is not None else (
            args.item_order.read_text().split() if args.item_order else None)
        if order is None:
            raise SystemExit("a .npy Semantic-ID file needs --split-dir or --item-order")
        item_ids, codes = load_codes_npy(args.semantic_ids, order)
    else:
        item_ids, codes = load_semantic_id_table(args.semantic_ids)
    vocab = TigerVocab((args.codebook_size,) * (args.num_codebooks + 1), args.user_buckets)
    index = make_item_index(item_ids, codes, vocab, args.codebook_size)
    missing = {i for s in sequences.values() for i in s} - set(index.id_to_idx)
    if missing:
        raise SystemExit(f"{len(missing)} interaction items have no Semantic ID, e.g. {sorted(missing)[:3]}")
    train = build_examples(sequences, index, vocab, "train", min_len=args.min_user_len)
    val = build_examples(sequences, index, vocab, "val", min_len=args.min_user_len)
    test = build_examples(sequences, index, vocab, "test", min_len=args.min_user_len)
    table = PrefixTable(index.codes, vocab.level_sizes).to(device)
    degree = train_item_degree(sequences, index)
    groups = {"cold": degree == 0, "tail_1_4": (degree >= 1) & (degree <= 4), "head_5plus": degree >= 5}
    print(f"users={len(sequences):,} items={index.num_items:,} interactions={n_inter:,} "
          f"train/val/test examples={len(train):,}/{len(val):,}/{len(test):,}")
    steps_per_epoch = -(-len(train) // args.batch_size)
    print(f"steps/epoch={steps_per_epoch:,}; {args.max_steps:,} steps = "
          f"{args.max_steps / steps_per_epoch:.0f} epochs")

    # ---- model ----------------------------------------------------------------------
    cfg = ModelConfig(d_model=args.d_model, d_kv=args.d_kv, d_ff=args.d_ff, num_heads=args.num_heads,
                      num_layers=args.layers, num_decoder_layers=args.layers, dropout=args.dropout,
                      tie_embeddings=args.tie_embeddings)
    model = build_model(vocab, cfg).to(device)
    print(f"parameters={count_parameters(model) / 1e6:.2f}M vocab={vocab.vocab_size} device={device}")
    optimizer = build_optimizer(model.parameters(), args.optimizer, args.lr)
    scheduler = constant_then_inverse_sqrt(optimizer, args.constant_steps)
    config = {"args": {k: str(v) for k, v in vars(args).items()}, "model": cfg.to_dict(),
              "level_sizes": list(vocab.level_sizes), "params": count_parameters(model),
              "n_users": len(sequences), "n_items": index.num_items, "n_interactions": n_inter}
    write_json(out / "config.json", config)

    step = 0
    last_path, best_path = out / "last.pt", out / "best.pt"
    best = {"ndcg@10": -1.0, "step": 0}
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best = int(state["step"]), state["best"]
        print(f"resumed at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best": best, "config": config}, tmp)
        tmp.replace(path)

    def run_eval(examples, limit=0, **kw):
        return evaluate(model, examples, index, table, device, beam_size=args.beam_size,
                        batch_size=args.eval_batch_size, use_user_token=use_user, limit=limit,
                        item_groups=None if limit else groups, **kw)

    # ---- train ----------------------------------------------------------------------
    log = open(out / "train_log.jsonl", "a")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda")
    start = time.time()
    out_of_time = False
    while step < args.max_steps and not args.eval_only and not out_of_time:
        epoch, offset = divmod(step, steps_per_epoch)
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        model.train()
        for b in range(offset, steps_per_epoch):
            if step >= args.max_steps:
                break
            rows = order[b * args.batch_size:(b + 1) * args.batch_size]
            input_ids, attention_mask, labels, _, _ = collate(train, rows, index, torch, device, use_user)
            with autocast:
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"step": step, "lr": scheduler.get_last_lr()[0]}
            if step % 100 == 0:
                record.update(train_loss=float(loss.detach()), elapsed_s=round(time.time() - start))
            if step % args.val_loss_every == 0:
                record["val_loss"] = validation_loss(model, val, index, device, use_user_token=use_user)
                model.train()
            if step % args.val_metric_every == 0 or step == args.max_steps:
                m = run_eval(val, limit=args.val_metric_users)
                record.update({f"val_{k}": v for k, v in m.items()})
                if m["ndcg@10"] > best["ndcg@10"]:
                    best = {"ndcg@10": m["ndcg@10"], "step": step}
                    save(best_path)
                model.train()
            if len(record) > 2:
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step % 1000 == 0 or "val_ndcg@10" in record:
                    print(record, flush=True)
            if step % args.checkpoint_every == 0:
                save(last_path)
            if args.time_limit_minutes and (time.time() - start) / 60 > args.time_limit_minutes:
                out_of_time = True
                save(last_path)
                print(f"time limit reached at step {step}; checkpoint saved. Re-run with --resume.")
                break
    if not args.eval_only:
        save(last_path)
    if out_of_time:
        return  # test split is only touched once training has finished

    # ---- final evaluation (test touched once per output dir) ---------------------------
    test_path = out / "metrics_test.json"
    if test_path.exists():
        print(f"{test_path} exists; not re-evaluating the test split.")
        return
    if args.no_final_eval:
        return
    results = {}
    last_step = int(torch.load(last_path, map_location="cpu", weights_only=False)["step"])
    for name, path in (("last", last_path), ("best_val", best_path)):
        if not path.exists():
            continue
        state = torch.load(path, map_location=device, weights_only=False)
        if name == "best_val" and int(state["step"]) == last_step:
            results[name] = {"step": last_step, "same_checkpoint_as": "last"}
            continue
        model.load_state_dict(state["model"])
        t0 = time.time()
        results[name] = {
            "step": int(state["step"]),
            "val_full": run_eval(val),
            "test_paper_faithful": run_eval(test, report_filtered=True),  # incl. with_history_filter
            "test_constrained": run_eval(test, constrained=True),
        }
        results[name]["eval_seconds"] = round(time.time() - t0)
        print(name, json.dumps(results[name], indent=1))
    write_json(test_path, results)


if __name__ == "__main__":
    main()
