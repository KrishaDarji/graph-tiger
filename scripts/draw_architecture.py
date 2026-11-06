"""Draw the Graph-TIGER architecture figure as it is currently being run."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUTPUT = Path("presentation/architecture/graph_tiger_architecture.png")
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#6f6e69"
STYLES = {
    "data": ("#f1f0ec", "#a9a8a2"),
    "fixed": ("#e3e2dd", "#6f6e69"),
    "trained": ("#dbe9fa", "#2a78d6"),
    "graph": ("#fde6dc", "#eb6834"),
    "rerank": ("#d9f3e9", "#1baf7a"),
}
W, H = 2.55, 1.05


def box(ax, x, y, title, detail, kind):
    fill, edge = STYLES[kind]
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=2))
    ax.text(x + W / 2, y + H - 0.32, title, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x + W / 2, y + 0.34, detail, ha="center", va="center", fontsize=9, color=MUTED,
            linespacing=1.25, zorder=3)


def arrow(ax, points, color=LINE, style="-", width=1.6):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=width, linestyle=style,
                solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=15, color=color,
                                 linewidth=width, linestyle=style, shrinkA=0, shrinkB=0, zorder=1))


def label(ax, x, y, text, color=MUTED, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=9, color=color, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
ax.text(0.4, 8.6, "Graph-TIGER architecture", fontsize=20, fontweight="bold", color=INK, va="center")

C1, C2, C3, C4, C5 = 0.4, 3.45, 6.75, 9.8, 13.05
R1, R2, R3, R4 = 6.45, 4.85, 2.55, 0.55

ax.text(C1, R1 + H + 0.62, "1. BUILD ITEM CODES", fontsize=9.5, fontweight="bold", color=MUTED, va="center")
ax.text(C2, R3 + H + 0.2, "2. RECOMMEND", fontsize=9.5, fontweight="bold", color=MUTED, va="center")

box(ax, C1, R1, "Product text", "title, brand,\ncategories, description", "data")
box(ax, C2, R1, "Sentence-T5", "pretrained, 768-D\ncontent vector", "fixed")
box(ax, C3, R1, "Fusion", "content + α × graph,\ncold items: content only", "graph")
box(ax, C4, R1, "RQ-VAE", "768-D input,\none trained per run", "trained")
box(ax, C5, R1, "Semantic IDs", "4 codes per item,\nstored in a lookup table", "data")
box(ax, C1, R2, "User–item graph", "training interactions only", "data")
box(ax, C2, R2, "LightGCN (768-D)", "2 layers, item vectors", "graph")

box(ax, C1, R3, "User history", "last 20 items,\nwritten as semantic IDs", "data")
box(ax, C2, R3, "T5 encoder-decoder", "4 + 4 layers, one trained\nper run, same settings", "trained")
box(ax, C3, R3, "Beam search", "top 30 codes,\nreal items only", "fixed")
box(ax, C4, R3, "Lookup table", "codes to products,\nthen mask seen items", "fixed")
box(ax, C5, R3, "Top 10", "Recall and NDCG\n@5 and @10", "data")

box(ax, C3, R4, "LightGCN (64-D)", "already trained,\nuser and item vectors", "rerank")
box(ax, C4, R4, "Re-rank", "T5 score + α × LightGCN\nscore, TIGER frozen", "rerank")

mid1, mid2 = R1 + H / 2, R2 + H / 2
arrow(ax, [(C1 + W, mid1), (C2, mid1)])
arrow(ax, [(C2 + W, mid1), (C3, mid1)], color=STYLES["graph"][1])
label(ax, (C2 + W + C3) / 2, mid1 + 0.2, "Run 2")
arrow(ax, [(C3 + W, mid1), (C4, mid1)], color=STYLES["graph"][1])
arrow(ax, [(C4 + W, mid1), (C5, mid1)])
top = R1 + H + 0.3
arrow(ax, [(C2 + W / 2, R1 + H), (C2 + W / 2, top), (C4 + W / 2, top), (C4 + W / 2, R1 + H)])
label(ax, C3 + W / 2, top, "Run 1: content only")
arrow(ax, [(C1 + W, mid2), (C2, mid2)], color=STYLES["graph"][1])
arrow(ax, [(C2 + W, mid2), (C3 + 0.75, mid2), (C3 + 0.75, R1)], color=STYLES["graph"][1])

channel = R3 + H + 0.42
ids_x = C5 + W / 2
arrow(ax, [(ids_x, R1), (ids_x, channel), (C1 + W / 2, channel), (C1 + W / 2, R3 + H)])
arrow(ax, [(C4 + W / 2, channel), (C4 + W / 2, R3 + H)])
mid3 = R3 + H / 2
for start, end in ((C1, C2), (C2, C3), (C3, C4), (C4, C5)):
    arrow(ax, [(start + W, mid3), (end, mid3)])

mid4 = R4 + H / 2
green = STYLES["rerank"][1]
arrow(ax, [(C3 + W, mid4), (C4, mid4)], color=green)
arrow(ax, [(C4 + W / 2, R3), (C4 + W / 2, R4 + H)], color=green)
label(ax, C4 + W / 2, (R3 + R4 + H) / 2, "Design A: candidates")
arrow(ax, [(C4 + W, mid4), (ids_x, mid4), (ids_x, R3)], color=green)

ax.text(0.4, 1.72, "Comparisons", fontsize=11.5, fontweight="bold", color=INK, va="center")
ax.text(0.4, 1.17, "Design B: Run 2 vs. Run 1\nsame T5 and RQ-VAE settings", fontsize=10, color=MUTED, va="center", linespacing=1.3)
ax.text(0.4, 0.5, "Design A: re-ranked vs. Run 1\nsame frozen TIGER", fontsize=10, color=MUTED, va="center", linespacing=1.3)
ax.legend(handles=[Patch(facecolor=STYLES["trained"][0], edgecolor=STYLES["trained"][1], label="Trained for each run"),
                   Patch(facecolor=STYLES["graph"][0], edgecolor=STYLES["graph"][1], label="Graph path, Run 2"),
                   Patch(facecolor=STYLES["rerank"][0], edgecolor=STYLES["rerank"][1], label="Re-ranking, Design A"),
                   Patch(facecolor=STYLES["fixed"][0], edgecolor=STYLES["fixed"][1], label="Fixed, no training"),
                   Patch(facecolor=STYLES["data"][0], edgecolor=STYLES["data"][1], label="Data")],
          loc="lower left", bbox_to_anchor=(3.35 / 16, 0.045), frameon=False, fontsize=9.5)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print(OUTPUT)
