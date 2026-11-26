import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import build_split, load_items, load_reviews
from src.evaluator import summarize_ranks
from src.graph import load_graph, procrustes
from src.model import Decoder
from src.scoring import Blend, ease_matrix, evaluate, make_context, score_batches
from src.tokens import item_codes
from src.trainer import train


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/beauty", help="folder with reviews.json.gz, items.txt, emb.npy, graph.npy")
    p.add_argument("--reviews", default="")
    p.add_argument("--items", default="")
    p.add_argument("--emb", default="")
    p.add_argument("--graph-emb", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--save", default="", help="write the chosen weights here")
    p.add_argument("--init", default="", help="start from saved weights (fine-tune); the graph comes from --graph-emb")

    p.add_argument("--no-graph", action="store_true", help="plain RPG-style model without collaborative input")
    p.add_argument("--rerank-graph", default="", help="plain model; this graph is only used to re-rank the scores (see --gammas)")
    p.add_argument("--token-graph", type=float, default=0.0, help="weight of the graph vector inside the vectors that are quantised into codes")
    p.add_argument("--pca", type=int, default=128)
    p.add_argument("--digits", type=int, default=32)
    p.add_argument("--dim", type=int, default=448)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--drop", type=float, default=0.5)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--temp", type=float, default=0.03)

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--dense-w", type=float, default=0.0, help="weight of the item-level softmax loss (0 = off)")
    p.add_argument("--dense-tau", type=float, default=0.07)
    p.add_argument("--dense-neg", type=int, default=4096)
    p.add_argument("--dense-beta", type=float, default=8.0, help="dense weight in the score used for early stopping")
    p.add_argument("--no-select", action="store_true", help="keep the last epoch instead of the best validation epoch")
    p.add_argument("--stop-after", type=int, default=0, help="stop after this many epochs, keeping the long schedule")

    p.add_argument("--temporal", default="", help="q1,q2: train before the q1 time quantile, validate up to q2, test after")
    p.add_argument("--train-end", choices=["val", "test"], default="val", help="temporal: also train on the validation window")
    p.add_argument("--holdout", default="", help="npy of item ids removed from all training data (cold-start protocol)")

    p.add_argument("--beta-fixed", type=float, default=-1.0, help="use this dense weight instead of picking it on validation")
    p.add_argument("--gammas", default="0", help="grid of last-item graph-similarity weights added to the score")
    p.add_argument("--ease-ws", default="", help="grid of weights for blending an EASE score in")
    p.add_argument("--ease-lam", type=float, default=1000.0)
    p.add_argument("--cands", default="", help="grid of candidate sizes: re-score only the top-K items of the dense head")
    p.add_argument("--slots", default="", help="grid of top-10 slots reserved for items without training interactions")
    p.add_argument("--eval-graph", default="", help="swap this graph in after training (refresh test)")
    p.add_argument("--align", action="store_true", help="rotate the swapped-in graph onto the training-time basis")
    p.add_argument("--latency", action="store_true", help="time single-user requests")
    return p.parse_args()


def floats(s):
    return [float(v) for v in s.split(",")]


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    d = Path(a.data)
    items, idx = load_items(a.items or d / "items.txt")
    n = len(items)
    graph_path = a.graph_emb or d / "graph.npy"
    graph = load_graph(a.rerank_graph, dev) if a.rerank_graph else (None if a.no_graph else load_graph(graph_path, dev))

    holdout = set(np.load(a.holdout).tolist()) if a.holdout else set()
    temporal = tuple(floats(a.temporal)) if a.temporal else None
    split = build_split(load_reviews(a.reviews or d / "reviews.json.gz"), idx, a.max_len, holdout, temporal, a.train_end)
    codes, x = item_codes(np.load(a.emb or d / "emb.npy"), a.digits, a.pca, split.seen, np.load(graph_path) if a.token_graph else None, a.token_graph)
    print("unique item IDs:", len(np.unique(codes, axis=0)), "/", n, flush=True)
    content = torch.cat([F.normalize(torch.tensor(x, device=dev), dim=1), torch.zeros(1, x.shape[1], device=dev)])
    val_t, test_t = (torch.tensor(t, device=dev) for t in (split.val_target, split.test_target))
    pool = torch.tensor([i for i in range(n) if i not in holdout], device=dev)
    cold_mask = torch.tensor(np.append(~split.seen, False), device=dev)
    cold_eval = holdout or ({i for i in range(n) if not split.seen[i]} if temporal else set())
    print(len(split.train), "train users,", len(split.val_hist), "val,", len(split.test_hist), "test", flush=True)

    model = Decoder(torch.tensor(codes, device=dev), graph, a.dim, a.layers, a.heads, a.ffn, a.drop, a.max_len, a.temp, graph_input=not a.rerank_graph).to(dev)
    if a.init:
        state = torch.load(a.init, map_location=dev)
        if graph is not None:
            state["graph"] = model.graph.detach().clone()
        model.load_state_dict(state)
    ctx = make_context(content, split, cold_mask, cold_eval, bool(a.dense_w), a.dense_tau)

    state, best, log, train_seconds = train(model, ctx, split, pool, a)
    model.load_state_dict(state)

    betas = [a.beta_fixed] if a.beta_fixed >= 0 else ([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] if a.dense_w else [0.0])
    gammas = floats(a.gammas) if graph is not None else [0.0]
    grid = [Blend(beta=b, gamma=g) for b in betas for g in gammas]
    val_all = [e.summary() for e in evaluate(model, ctx, split.val_hist, val_t, grid)]
    pick = int(np.argmax([s["raw"]["ndcg@10"] for s in val_all]))
    chosen = grid[pick]
    test_ev = evaluate(model, ctx, split.test_hist, test_t, [chosen])[0]
    test = test_ev.summary()

    def val_and_test(blend):
        v = evaluate(model, ctx, split.val_hist, val_t, [blend])[0]
        t = evaluate(model, ctx, split.test_hist, test_t, [blend])[0]
        return v, t

    extra = {}
    if a.dense_w and cold_eval:
        s = evaluate(model, ctx, split.test_hist, test_t, [replace(chosen, beta=1000.0)])[0].summary()
        extra["dense_only"] = {k: s[k] for k in ("raw", "cold_pool", "warm_pool")}

    if a.slots and cold_eval and graph is not None:
        extra["slots"] = {}
        cold_ids = list(cold_eval)
        for place in ("head", "tail"):
            for k in [int(v) for v in a.slots.split(",")]:
                row = {}
                for name, hist, tgt in (("val", split.val_hist, val_t), ("test", split.test_hist, test_t)):
                    ev = evaluate(model, ctx, hist, tgt, [replace(chosen, slots=k, place=place)])[0]
                    ranks = ev.ranks()[0]
                    is_cold = np.isin(tgt.cpu().numpy(), cold_ids)
                    row[name] = {"all": ev.summary()["raw"], "cold_targets": summarize_ranks(ranks[is_cold]),
                                 "warm_targets": summarize_ranks(ranks[~is_cold]), "n_cold": int(is_cold.sum())}
                extra["slots"][f"{place}{k}"] = row
                print("slots", place, k, {m: round(row["test"][m]["recall@10"], 4) for m in ("all", "cold_targets", "warm_targets")}, flush=True)

    ease_ranks = None
    if a.ease_ws and n <= 20000:
        ctx.ease = ease_matrix(split.train, n, a.ease_lam, dev)
        extra["ease_blend"], pick_w = {}, None
        for w in floats(a.ease_ws):
            v, t = val_and_test(replace(chosen, ease=w))
            extra["ease_blend"][f"w{w}"] = {"val": v.summary()["raw"], "test": t.summary()["raw"]}
            print("ease blend", w, round(extra["ease_blend"][f"w{w}"]["val"]["ndcg@10"], 4), round(extra["ease_blend"][f"w{w}"]["test"]["ndcg@10"], 4), flush=True)
            if pick_w is None or extra["ease_blend"][f"w{w}"]["val"]["ndcg@10"] > extra["ease_blend"][pick_w]["val"]["ndcg@10"]:
                pick_w, ease_ranks = f"w{w}", t.ranks()[0]
        extra["ease_pick"] = pick_w
        ctx.ease = None

    if a.cands and a.dense_w:
        extra["candidates"] = {}
        for k in [int(v) for v in a.cands.split(",")]:
            extra["candidates"][str(k)] = evaluate(model, ctx, split.test_hist, test_t, [replace(chosen, cand=k)])[0].summary()["raw"]
            print("cands", k, round(extra["candidates"][str(k)]["recall@10"], 4), flush=True)

    if a.eval_graph:
        with torch.no_grad():
            new = load_graph(a.eval_graph, dev)
            model.graph.copy_(procrustes(model.graph, new) if a.align else new)
        v, t = val_and_test(chosen)
        extra["refreshed"] = {"val": v.summary()["raw"], "test": t.summary()["raw"]}

    sync()
    t0 = time.time()
    for _ in score_batches(model, ctx, split.test_hist, test_t, [chosen]):
        pass
    sync()
    score_ms = (time.time() - t0) / len(split.test_hist) * 1000

    if a.latency:
        times = []
        for i in range(300):
            sync()
            t1 = time.time()
            for _ in score_batches(model, ctx, split.test_hist[i:i + 1], test_t[i:i + 1], [chosen]):
                pass
            sync()
            times.append((time.time() - t1) * 1000)
        times = np.array(times[20:])
        extra["single_user_latency"] = {f"p{q}_ms": float(np.percentile(times, q)) for q in (50, 95, 99)}

    result = {"best_val": best, "beta": chosen.beta, "gamma": chosen.gamma, "val": val_all[pick]["raw"], "val_filtered": val_all[pick]["filtered"],
              "test": test["raw"], "test_filtered": test["filtered"], "test_beyond": test["beyond"], "test_segments": test["segments"],
              "test_cold_pool": test.get("cold_pool"), "test_warm_pool": test.get("warm_pool"),
              "val_by_blend": {f"{b.beta},{b.gamma}": s["raw"]["ndcg@10"] for b, s in zip(grid, val_all)},
              "params": sum(p.numel() for p in model.parameters()), "train_seconds": train_seconds, "score_ms_per_user": score_ms,
              "epochs_run": len(log), "args": vars(a), **extra}
    raw, filt = test_ev.ranks()
    if cold_eval:
        is_cold = np.isin(test_t.cpu().numpy(), list(cold_eval))
        result["cold"] = {"n_users": int(is_cold.sum()), "metrics": summarize_ranks(raw[is_cold]), "warm_metrics": summarize_ranks(raw[~is_cold])}
    print(json.dumps({k: result[k] for k in ("beta", "test", "test_filtered", "params", "train_seconds", "score_ms_per_user")}, indent=1))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    (out / "log.json").write_text(json.dumps(log))
    np.save(out / "test_ranks_raw.npy", raw)
    np.save(out / "test_ranks_filtered.npy", filt)
    if ease_ranks is not None:
        np.save(out / "test_ranks_raw_ease.npy", ease_ranks)
    if a.save:
        torch.save(state, a.save)


if __name__ == "__main__":
    main()
