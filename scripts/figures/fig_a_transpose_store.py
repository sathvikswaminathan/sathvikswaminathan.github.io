"""a-transpose-store-conflict.png

The cost the transpose pushes onto the store side. The 32 threads of a warp
cover 16 rows m of the A slice, two threads per row (kg = 0 and kg = 4). On the
first of the four transposed STS.32 they write As[0][m] and As[4][m], addresses
m and 512 + m — 512 floats apart, a multiple of 32, so both land in bank m.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import (
    BG,
    BLUE,
    BLUE_D,
    BLUE_F,
    GREY,
    GREY_L,
    INK,
    INK_SOFT,
    ORANGE,
    RED,
    RED_F,
    TEAL_D,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/a-transpose-store-conflict.png"

UNIT = 0.23  # inches per data unit, so cells come out square in both panels
ORANGE_F = "#f6ddc2"
ORANGE_D = "#9a5a12"

YLIM = (-6.2, 14.2)
A_XLIM = (-5.2, 11.0)
B_XLIM = (-9.6, 36.5)

Y_TITLE, Y_SUB = 13.9, 12.6

TOP_BAND, BOT_BAND = 1.25, 0.42
LEFT_PAD, MID_PAD, RIGHT_PAD = 0.40, 1.05, 0.30

A_W = (A_XLIM[1] - A_XLIM[0]) * UNIT
B_W = (B_XLIM[1] - B_XLIM[0]) * UNIT
PANEL_H = (YLIM[1] - YLIM[0]) * UNIT

FIG_W = LEFT_PAD + A_W + MID_PAD + B_W + RIGHT_PAD
FIG_H = PANEL_H + TOP_BAND + BOT_BAND

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)

fig.text(0.5, 1 - 0.42 / FIG_H, "The transpose moves the conflict to the store side",
         ha="center", va="center", fontsize=25, fontweight="bold")
fig.text(
    0.5, 1 - 0.85 / FIG_H,
    "Two threads hold the two halves of the same row m of the A slice, and the transpose sends both into column m of As — "
    "four rows apart, which is exactly one bank apart from itself.",
    ha="center", va="center", fontsize=13.5, color=GREY,
)

# ======================================================= left: who holds what
axA = fig.add_axes([LEFT_PAD / FIG_W, BOT_BAND / FIG_H, A_W / FIG_W, PANEL_H / FIG_H])
axA.set_xlim(*A_XLIM)
axA.set_ylim(*YLIM)
axA.axis("off")

axA.text(4.0, Y_TITLE, "the A slice in global memory", fontsize=15.5, fontweight="bold",
         color=INK, ha="center", va="top")
axA.text(4.0, Y_SUB, "128 rows × BK = 8, row-major", fontsize=11, color=GREY,
         ha="center", va="top")

axA.text(2.0, 11.0, "kg = 0", fontsize=11, color=BLUE_D, ha="center", va="bottom",
         fontweight="bold")
axA.text(6.0, 11.0, "kg = 4", fontsize=11, color=ORANGE_D, ha="center", va="bottom",
         fontweight="bold")
for c in range(8):
    axA.text(c + 0.5, 10.2, str(c), fontsize=8.5, color=INK_SOFT, ha="center", va="bottom")
axA.text(-0.55, 10.2, "k", fontsize=11, color=INK_SOFT, ha="right", va="bottom")

NSHOW, STEP, TOP = 8, 1.1, 9.85
for r in range(NSHOW):
    y = TOP - STEP * (r + 1)
    for c in range(8):
        axA.add_patch(Rectangle((c, y), 1, 1,
                                facecolor=BLUE_F if c < 4 else ORANGE_F,
                                edgecolor=GREY_L, lw=0.5, zorder=2))
    axA.add_patch(Rectangle((0, y), 4, 1, facecolor="none", edgecolor=BLUE, lw=1.8, zorder=4))
    axA.add_patch(Rectangle((4, y), 4, 1, facecolor="none", edgecolor=ORANGE, lw=1.8, zorder=4))
    axA.text(-0.55, y + 0.5, f"m = {r}", fontsize=10, color=INK_SOFT, ha="right", va="center")
    if r < 2:
        axA.text(2.0, y + 0.5, f"t{2 * r}", fontsize=10, color=BLUE_D, ha="center",
                 va="center", fontweight="bold", zorder=5)
        axA.text(6.0, y + 0.5, f"t{2 * r + 1}", fontsize=10, color=ORANGE_D, ha="center",
                 va="center", fontweight="bold", zorder=5)

y_last = TOP - STEP * NSHOW
for i in range(3):  # vertical ellipsis (Lato has no U+22EE)
    axA.plot([4.0], [y_last - 0.45 - 0.34 * i], marker="o", ms=3.0, color=GREY_L)
axA.text(4.0, y_last - 1.85, "… up to m = 15:  one warp covers 16 rows",
         fontsize=10.5, color=GREY, ha="center", va="top")

axA.text(
    3.0, -2.9,
    "each thread LDG.128s one half-row —\ncoalesced, exactly as before — then\nspreads it over 4 rows of As with 4 × STS.32",
    fontsize=11.5, color=INK_SOFT, ha="center", va="top", linespacing=1.5,
)

# ============================================== right: where the stores land
axB = fig.add_axes([(LEFT_PAD + A_W + MID_PAD) / FIG_W, BOT_BAND / FIG_H,
                    B_W / FIG_W, PANEL_H / FIG_H])
axB.set_xlim(*B_XLIM)
axB.set_ylim(*YLIM)
axB.axis("off")

axB.text(16, Y_TITLE, "where the first STS.32 (j = 0) lands in As", fontsize=15.5,
         fontweight="bold", color=INK, ha="center", va="top")
axB.text(16, Y_SUB, "As[k][m] at k · 128 + m — columns are shared-memory banks (address mod 32)",
         fontsize=11, color=GREY, ha="center", va="top")
for c in range(32):
    axB.text(c + 0.5, 10.6, str(c), fontsize=7.6, color=INK_SOFT, ha="center", va="bottom")

STRIPS = [
    (8.5, "As[0][·]", "addr 0–31", BLUE_F, BLUE, BLUE_D, "even threads  →  As[0][m] = m"),
    (4.6, "As[4][·]", "addr 512–543", ORANGE_F, ORANGE, ORANGE_D, "odd threads  →  As[4][m] = 512 + m"),
]

for y, name, addr, fill, edge, ink, caption in STRIPS:
    for c in range(32):
        axB.add_patch(Rectangle((c, y), 1, 1,
                                facecolor=fill if c < 16 else "#ffffff",
                                edgecolor=GREY_L, lw=0.5, zorder=2))
    axB.add_patch(Rectangle((0, y), 32, 1, facecolor="none", edgecolor=INK_SOFT, lw=1.3, zorder=3))
    axB.add_patch(Rectangle((0, y), 16, 1, facecolor="none", edgecolor=edge, lw=2.2, zorder=4))
    axB.text(-0.7, y + 0.62, name, fontsize=11, color=INK_SOFT, ha="right", va="center")
    axB.text(-0.7, y + 0.10, addr, fontsize=9.5, color=GREY, ha="right", va="center")
    axB.text(8.0, y + 1.22, caption, fontsize=11.5, color=ink, ha="center", va="bottom",
             fontweight="bold", zorder=6, bbox=dict(facecolor=BG, edgecolor="none", pad=2.5))

for c in range(16):
    axB.plot([c + 0.5, c + 0.5], [5.6, 8.5], color=RED, lw=1.0, ls=(0, (2, 2)), zorder=5)

axB.text(25.2, 7.05, "512 floats apart  =  4 rows of As,\nand 512 mod 32 = 0  →  the same bank",
         fontsize=11.5, color=RED, ha="center", va="center", fontweight="bold", linespacing=1.5)

# ---- bank occupancy
BASE, BARH = 0.6, 1.0
axB.plot([0, 32], [BASE, BASE], color=INK_SOFT, lw=1.4, zorder=3)
for c in range(16):
    axB.add_patch(Rectangle((c + 0.12, BASE), 0.76, 2 * BARH,
                            facecolor=RED_F, edgecolor=RED, lw=1.0, zorder=4))
    axB.text(c + 0.5, BASE + 2 * BARH + 0.14, "2", fontsize=8.5, color=RED,
             ha="center", va="bottom", fontweight="bold")
axB.plot([0, 32], [BASE + BARH, BASE + BARH], color=TEAL_D, lw=1.8, ls=(0, (6, 4)), zorder=5)
axB.text(32.5, BASE + BARH, "1 = conflict-free", fontsize=10.5, color=TEAL_D,
         ha="left", va="center")
axB.text(-0.7, BASE + BARH, "distinct addresses\nper bank", fontsize=10, color=INK_SOFT,
         ha="right", va="center", linespacing=1.4)
axB.text(24, BASE + 0.45, "banks 16–31 idle", fontsize=11, color=GREY, ha="center", va="center",
         zorder=6, bbox=dict(facecolor=BG, edgecolor="none", pad=2.5))

axB.text(16, -2.9, "the warp's 32 stores crowd into 16 banks  →  2 wavefronts per transposed STS",
         fontsize=13.5, fontweight="bold", color=RED, ha="center", va="center",
         bbox=dict(facecolor=RED_F, edgecolor="none", pad=8.0))

# arrow between the panels
mid_y = (BOT_BAND + PANEL_H * (6.5 - YLIM[0]) / (YLIM[1] - YLIM[0])) / FIG_H
fig.patches.append(
    FancyArrowPatch(((LEFT_PAD + A_W + 0.20) / FIG_W, mid_y),
                    ((LEFT_PAD + A_W + MID_PAD - 0.20) / FIG_W, mid_y),
                    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=24,
                    color=INK_SOFT, lw=2.0)
)
fig.text((LEFT_PAD + A_W + MID_PAD / 2) / FIG_W, mid_y + 0.018, "4 × STS.32",
         fontsize=11, color=INK_SOFT, ha="center", va="bottom", fontweight="bold")

fig.text(
    0.5, 0.016,
    "Each value of A is written into shared memory once per k tile but read at every k step, so trading 4 × STS.32 for 16 × LDS.128 is still "
    "the right trade — but the 2-way store conflict is worth removing, which padding or swizzling does.",
    ha="center", va="center", fontsize=12, color=GREY, style="italic",
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT, f"({FIG_W:.1f} x {FIG_H:.1f} in)")
