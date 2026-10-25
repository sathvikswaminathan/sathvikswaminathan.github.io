"""tile-size-tradeoff.png

Three panels: the register cost (identical on both GPUs, 65,536 regs/SM and a
255-reg/thread limit), then what that cost buys on the T4 and on the H100.

On the T4 the issue-slot ceiling is 100% for every tile here — the LDS
instructions fit in the free slots between FFMAs — so the binding constraint is
shared-memory bandwidth (4 FMAs per load). On the H100 every LDS costs an FFMA
issue slot, so the ceiling climbs with the tile, but it flattens out.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import (
    BG,
    BLUE,
    GREY,
    GREY_L,
    INK,
    INK_SOFT,
    ORANGE,
    RED,
    RED_E,
    RED_F,
    TEAL,
    TEAL_D,
    despine,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/tile-size-tradeoff.png"

TILES = ["4×4", "8×8", "8×16", "16×16"]
ACC = [16, 64, 128, 256]
OPS = [16, 32, 48, 64]
TOTAL = [a + o for a, o in zip(ACC, OPS)]

FFMA = [16, 64, 128, 256]
LDS = [8, 16, 24, 32]
FMAS_PER_LOAD = [2.0, 4.0, 5.33, 8.0]
ETA_T4 = [100, 100, 100, 100]
ETA_H100 = [67, 80, 84, 89]

x = list(range(4))

fig = plt.figure(figsize=(19.8, 7.9), dpi=150)

fig.text(
    0.5,
    0.965,
    "The register cost of a bigger thread tile is the same on both GPUs — what it buys is not",
    ha="center",
    va="center",
    fontsize=24,
    fontweight="bold",
)

# ----------------------------------------------------------------- panel 1
ax = fig.add_axes([0.045, 0.125, 0.265, 0.74])
despine(ax)
ax.set_title("The register cost grows with the tile area", fontsize=17, fontweight="bold", pad=16)

for i in range(4):
    over = TOTAL[i] > 255
    if over:
        ax.bar(i, TOTAL[i], width=0.58, color=RED_F, edgecolor=RED_E, linewidth=1.2, zorder=2)
    else:
        ax.bar(i, ACC[i], width=0.58, color=BLUE, edgecolor="none", zorder=2)
        ax.bar(i, OPS[i], bottom=ACC[i], width=0.58, color=TEAL, edgecolor="none", zorder=2)
    ax.text(
        i,
        TOTAL[i] + 11,
        str(TOTAL[i]),
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=RED if over else INK,
    )

ax.axhline(255, color=RED, ls=(0, (6, 4)), lw=2.2, zorder=3)
ax.text(
    -0.42,
    263,
    "255 registers / thread  —  hardware limit  (T4 and H100)",
    fontsize=12.5,
    fontweight="bold",
    color=RED,
    va="bottom",
)
ax.annotate(
    "spills to local\nmemory  —  extra\nSTL / LDL slots",
    xy=(3.0, 248),
    xytext=(2.03, 150),
    fontsize=12,
    color=RED,
    linespacing=1.45,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.4, connectionstyle="arc3,rad=-0.25"),
)

handles = [
    Rectangle((0, 0), 1, 1, facecolor=BLUE),
    Rectangle((0, 0), 1, 1, facecolor=TEAL),
]
ax.legend(
    handles,
    ["accumulators  (m·n)", "2 operand sets  (2(m+n), prefetch)"],
    loc="upper left",
    fontsize=12,
    handlelength=1.4,
    handleheight=1.0,
    labelcolor=INK_SOFT,
)

ax.set_xticks(x)
ax.set_xticklabels(TILES, fontsize=14)
ax.set_ylim(0, 370)
ax.set_xlim(-0.6, 3.6)
ax.set_ylabel("registers per thread", fontsize=13)
ax.set_xlabel("thread tile size", fontsize=13, labelpad=10)

# ----------------------------------------------------------------- panel 2
ax = fig.add_axes([0.385, 0.125, 0.265, 0.74])
despine(ax)
ax.set_title("Tesla T4  —  issue slots are never the problem", fontsize=17, fontweight="bold", pad=16)

ax.plot(x, ETA_T4, color=TEAL_D, lw=2.6, marker="o", ms=11, mfc=BG, mew=2.4, zorder=4)
ax.text(
    0.55,
    103.0,
    "η = 100% for every tile",
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color=TEAL_D,
)
ax.plot([3], [100], marker="X", ms=17, color=RED, zorder=5)
ax.text(
    2.86,
    97.0,
    "never reached —\nthe tile does not fit",
    ha="right",
    va="top",
    fontsize=11.5,
    color=RED,
    linespacing=1.4,
)
ax.text(
    1.5,
    115.5,
    "the LDS for step k+1 slip into the free slots between the FFMAs,\nso the issue-slot ceiling does not depend on the tile size",
    fontsize=11.5,
    color=GREY,
    ha="center",
    va="top",
    linespacing=1.45,
    style="italic",
)

ax.set_ylim(58, 118)
ax.set_xlim(-0.6, 3.6)
ax.set_xticks(x)
ax.set_xticklabels(TILES, fontsize=14)
ax.set_ylabel("issue-slot efficiency ceiling  η  (%)", fontsize=13, color=TEAL_D)
ax.set_xlabel("thread tile size", fontsize=13, labelpad=10)
ax.tick_params(axis="y", colors=TEAL_D)

ax2 = ax.twinx()
despine(ax2, keep=("right", "bottom"))
ax2.plot(x, FMAS_PER_LOAD, color=ORANGE, lw=2.2, marker="s", ms=8, ls=(0, (5, 2.5)), zorder=4)
ax2.axhline(4, color=ORANGE, ls=(0, (3, 3)), lw=1.6, alpha=0.55)
ax2.set_ylim(0, 16)
ax2.set_ylabel("FMAs per shared-memory load", fontsize=13, color=ORANGE, rotation=270, labelpad=22)
ax2.tick_params(axis="y", colors=ORANGE)
ax2.spines["right"].set_color(GREY_L)

ax2.text(
    -0.45,
    4.35,
    "4 FMAs per load required",
    fontsize=11.5,
    color=ORANGE,
    fontweight="bold",
    va="bottom",
)
ax2.annotate(
    "2 FMAs per load —\nthe cores wait on\nshared memory half\nthe time",
    xy=(0.05, 1.95),
    xytext=(0.38, 0.5),
    fontsize=11.5,
    color=RED,
    linespacing=1.4,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3, connectionstyle="arc3,rad=0.25"),
)

# ----------------------------------------------------------------- panel 3
ax = fig.add_axes([0.725, 0.125, 0.265, 0.74])
despine(ax)
ax.set_title("H100 SXM  —  …but the efficiency it buys flattens out", fontsize=17, fontweight="bold", pad=16)

ax.plot(x, ETA_H100, color=ORANGE, lw=2.6, marker="o", ms=11, mfc=BG, mew=2.4, zorder=4)
for i, v in enumerate(ETA_H100):
    ax.text(
        i - (0.14 if i == 0 else 0),
        v + (0 if i == 0 else 2.6),
        f"{v}%",
        ha="right" if i == 0 else "center",
        va="center" if i == 0 else "bottom",
        fontsize=15,
        fontweight="bold",
        color=RED if TOTAL[i] > 255 else INK,
    )

deltas = [
    (0.62, 71.5, "+13 pts\nfor +64 regs", GREY),
    (1.52, 78.0, "+4 pts\nfor +80 regs", TEAL_D),
    (2.52, 82.5, "+5 pts\nfor +144 regs", RED),
]
for xx, yy, txt, col in deltas:
    ax.text(xx, yy, txt, ha="center", va="top", fontsize=11.5, color=col, style="italic", linespacing=1.4)

ax.plot([3], [89], marker="X", ms=17, color=RED, zorder=5)
ax.text(
    2.9,
    93.5,
    "never reached —\nthe tile does not fit",
    ha="right",
    va="bottom",
    fontsize=11.5,
    color=RED,
    linespacing=1.4,
)
ax.text(
    1.5,
    115.5,
    "every LDS takes an issue slot that an FFMA could have used,\nso a bigger tile does raise the ceiling — with diminishing returns",
    fontsize=11.5,
    color=GREY,
    ha="center",
    va="top",
    linespacing=1.45,
    style="italic",
)

ax.set_ylim(58, 118)
ax.set_xlim(-0.6, 3.6)
ax.set_xticks(x)
ax.set_xticklabels(TILES, fontsize=14)
ax.set_ylabel("issue-slot efficiency ceiling  η  (%)", fontsize=13)
ax.set_xlabel("thread tile size", fontsize=13, labelpad=10)

fig.text(
    0.5,
    0.028,
    "Both GPUs rule out 16×16 for the same reason — 320 registers overshoots the 255-per-thread limit. "
    "Between 8×8 and 8×16 the T4 gains shared-memory headroom but no issue slots, the H100 gains 4 points of η; "
    "either way the bill is 80 more registers per thread.",
    ha="center",
    va="center",
    fontsize=12.5,
    color=GREY,
    style="italic",
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
