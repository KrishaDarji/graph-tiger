"""Learn a 768-D fusion of content and graph with an MLP, then save it for the RQ-VAE.

The MLP compresses [content, graph] (1536-D) to 768-D, and a decoder rebuilds both halves from
that. Training stops here: the encoder is frozen and its unit-norm output is saved, so the
RQ-VAE stays identical to Runs 1 and 2. The decoder is only used for training.

Each half's error is divided by that half's variance, so both count equally even though the
graph half varies more. Items with no training edges have a zero graph half, and their graph
error is left out, so the model is not trained to reproduce those zeros. Epochs are selected on
reconstruction of held-out items. No interactions are used here, and LightGCN's embeddings come
from training interactions only.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_concat_1536 import load_halves

HALF = 768


class FusionAutoencoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2 * HALF, hidden), nn.ReLU(), nn.Linear(hidden, HALF))
        self.decoder = nn.Sequential(nn.Linear(HALF, hidden), nn.ReLU(), nn.Linear(hidden, 2 * HALF))

    def encode(self, x):
        return nn.functional.normalize(self.encoder(x), dim=1)

    def forward(self, x):
        # The decoder sees the unit-norm vector scaled back up, the same output the RQ-VAE gets.
        return self.decoder(self.encode(x) * HALF ** 0.5)


def half_errors(out, x, warm):
    """Mean squared error of the content half on all rows and the graph half on non-cold rows."""
    content = ((out[:, :HALF] - x[:, :HALF]) ** 2).mean()
    graph = ((out[warm, HALF:] - x[warm, HALF:]) ** 2).mean()
    return content, graph


@torch.no_grad()
def r_squared(model, x, warm, rows, content_var, graph_var, batch=8192):
    """Fraction of variance rebuilt, per half, on the given rows. 1 is perfect, 0 is the mean."""
    errors = torch.zeros(3, device=x.device)
    counts = torch.zeros(3, device=x.device)
    for chunk in rows.split(batch):
        out, target, is_warm = model(x[chunk]), x[chunk], warm[chunk]
        content = ((out[:, :HALF] - target[:, :HALF]) ** 2).mean(1)
        graph = ((out[:, HALF:] - target[:, HALF:]) ** 2).mean(1)
        errors += torch.stack([content[is_warm].sum(), content[~is_warm].sum(), graph[is_warm].sum()])
        counts += torch.stack([is_warm.sum(), (~is_warm).sum(), is_warm.sum()]).float()
    means = (errors / counts.clamp(min=1)).tolist()
    return {"content_non_cold": 1 - means[0] / content_var,
            "content_cold": 1 - means[1] / content_var,
            "graph_non_cold": 1 - means[2] / graph_var}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, default=Path("data/processed/item_st5_emb.npy"))
    parser.add_argument("--graph-dir", type=Path, default=Path("results/lightgcn_768"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/item_mlp768.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fusion_mlp768"))
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    content, graph, cold = load_halves(args.content, args.graph_dir)
    # Unit-norm vectors have very small entries, so scale them up for training.
    scale = HALF ** 0.5
    x = torch.as_tensor(np.concatenate([content, graph], axis=1) * scale, dtype=torch.float32, device=device)
    warm = torch.as_tensor(~cold, device=device)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(args.seed)).to(device)
    n_holdout = int(len(x) * args.holdout)
    held, fit = order[:n_holdout], order[n_holdout:]
    # Variance on training rows only. The graph half is measured on non-cold rows.
    fit_warm = fit[warm[fit]]
    content_var = x[fit, :HALF].var(0).mean().item()
    graph_var = x[fit_warm, HALF:].var(0).mean().item()

    model = FusionAutoencoder(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = fit[torch.randperm(len(fit), device=device)]
        for chunk in shuffled.split(args.batch_size):
            content_error, graph_error = half_errors(model(x[chunk]), x[chunk], warm[chunk])
            loss = content_error / content_var + graph_error / graph_var
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = r_squared(model, x, warm, held, content_var, graph_var)
        # Lower is better. Cold and non-cold content are weighted by their share of held-out items.
        share = warm[held].float().mean().item()
        held_loss = (1 - scores["graph_non_cold"]) + share * (1 - scores["content_non_cold"]) \
            + (1 - share) * (1 - scores["content_cold"])
        history.append({"epoch": epoch, "holdout_loss": held_loss, **scores})
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch} holdout_loss={held_loss:.4f} {json.dumps(scores)}", flush=True)
        if held_loss < best:
            best, best_epoch = held_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fused = torch.cat([model.encode(chunk) for chunk in x.split(8192)]).cpu().numpy()
    if not np.isfinite(fused).all():
        raise SystemExit("Fused embeddings contain non-finite values")

    temp = args.output.with_name(args.output.stem + ".tmp.npy")
    np.save(temp, fused.astype(np.float32))
    temp.replace(args.output)
    torch.save(best_state, args.output_dir / "best.pt")
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "best_epoch": best_epoch,
        "holdout_r_squared": history[best_epoch - 1],
        "content_var": content_var,
        "graph_var": graph_var,
        # A value near 0 would mean items had collapsed onto one vector.
        "output_per_dimension_variance": float(fused.var(0).mean()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "history.json").write_text(json.dumps(history) + "\n")
    print(f"wrote {args.output} {fused.shape}; best epoch {best_epoch}; {json.dumps(history[best_epoch - 1])}; "
          f"output variance {summary['output_per_dimension_variance']:.3e}")


if __name__ == "__main__":
    main()
