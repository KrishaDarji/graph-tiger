"""Export LightGCN logs, a per-epoch CSV and training curves for presentations."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
COPIED = ["train.log", "console.log", "config.json", "metrics_val.json",
          "metrics_test.json", "neighbors.txt"]


def parse_log(path):
    epochs = {}
    for line in path.read_text().splitlines():
        match = re.search(r" epoch (\d+) loss=([\d.]+) bpr=([\d.]+) time=([\d.]+)s", line)
        if match:
            epochs[int(match[1])] = {"epoch": int(match[1]), "loss": float(match[2]),
                                     "bpr": float(match[3]), "seconds": float(match[4])}
        match = re.search(r"val epoch (\d+) (\{.*\})", line)
        if match:
            epochs[int(match[1])].update({f"val_{k}": v for k, v in json.loads(match[2]).items()})
    return [epochs[e] for e in sorted(epochs)]


def style(ax, title, xlabel="Epoch"):
    ax.set_title(title, loc="left", color=INK, fontsize=13)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def line(ax, xs, ys, fmt):
    ax.plot(xs, ys, color=SERIES, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(xs[-1], ys[-1], "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(fmt.format(ys[-1]), (xs[-1], ys[-1]), xytext=(0, 10), textcoords="offset points",
                ha="right", color=INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/lightgcn"))
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/lightgcn"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        shutil.copy2(args.run_dir / name, args.output_dir / name)

    rows = parse_log(args.run_dir / "train.log")
    fields = ["epoch", "loss", "bpr", "seconds", "val_recall@5", "val_recall@10", "val_ndcg@5", "val_ndcg@10"]
    with open(args.output_dir / "epoch_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    line(ax, [r["epoch"] for r in rows], [r["bpr"] for r in rows], "{:.4f}")
    style(ax, "LightGCN training BPR loss by epoch")
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss_curve.png", dpi=200)

    validated = [r for r in rows if "val_ndcg@10" in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "val_ndcg@10", "Validation NDCG@10 by epoch"),
                           (axes[1], "val_recall@10", "Validation Recall@10 by epoch")):
        line(ax, [r["epoch"] for r in validated], [r[key] for r in validated], "{:.4f}")
        style(ax, title)
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_curves.png", dpi=200)
    print(f"Wrote {len(rows)} epochs to {args.output_dir}")


if __name__ == "__main__":
    main()
