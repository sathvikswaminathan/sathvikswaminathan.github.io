"""k-tiling-slices.png

Why a CTA walks the k dimension in slices at all, and what one iteration of
that loop costs.

Left:   the geometry. A 128x128 tile of C needs a 128x4096 slab of A and a
        4096x128 slab of B, 2 MB each; the CTA stages BK = 8 columns/rows of
        them at a time, 8 KB, and loops 512 times.
Top right:    those footprints against the shared memory an SM actually has.
Bottom right: one k tile of the loop, and the hole the two barriers leave.
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
    GREEN_F,
    GREY,
    GREY_L,
    H100_ACCENT,
    IDLE,
    IDLE_E,
    INK,
    INK_SOFT,
    ORANGE,
    RED,
    RED_M,
    TEAL_D,
    TEAL_M,
    despine,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/k-tiling-slices.png"
ORANGE_D = "#9a5a12"
ORANGE_F = "#f6ddc2"

FIG_W, FIG_H = 18.0, 9.7
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)

fig.text(0.5, 1 - 0.42 / FIG_H, "A CTA walks the k dimension in slices",
         ha="center", va="center", fontsize=27, fontweight="bold")
fig.text(
    0.5, 1 - 0.85 / FIG_H,
    "The slabs a 128 × 128 tile of C needs are 2 MB each, so they never fit in shared memory. "
    "At BK = 8 the CTA stages 8 KB of them at a time and loops 512 times.",
    ha="center", va="center", fontsize=14.5, color=GREY,
)

# =============================================================== the geometry
G_X, G_Y, G_W = 0.015, 0.055, 0.395
XLIM, YLIM = (-2, 92), (-3, 94)
G_H = (G_W * FIG_W) * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0]) / FIG_H

axG = fig.add_axes([G_X, G_Y, G_W, G_H])
axG.set_xlim(*XLIM)
axG.set_ylim(*YLIM)
axG.axis("off")

AX0, AX1, AY0, AY1 = 4, 36, 8, 40        # A
CX0, CX1, CY0, CY1 = 44, 76, 8, 40       # C
BX0, BX1, BY0, BY1 = 44, 76, 50, 82      # B

SLAB, SLICE = 3.2, 2.0

for x0, x1, y0, y1, name, lx, ha in [
    (AX0, AX1, AY0, AY1, "A   4096 × 4096", AX0 + 1.0, "left"),
    (CX0, CX1, CY0, CY1, "C   4096 × 4096", CX0 + 1.0, "left"),
    (BX0, BX1, BY0, BY1, "B   4096 × 4096", BX1 - 1.0, "right"),
]:
    axG.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#ffffff",
                            edgecolor=GREY, lw=1.4, zorder=2))
    axG.text(lx, y0 - 1.6, name, fontsize=11.5, color=GREY, ha=ha, va="top")

# --- slabs and the slice inside each
axG.add_patch(Rectangle((AX0, AY1 - SLAB), AX1 - AX0, SLAB,
                        facecolor=BLUE_F, edgecolor=BLUE, lw=1.6, zorder=3))
axG.add_patch(Rectangle((BX0, BY0), SLAB, BY1 - BY0,
                        facecolor=ORANGE_F, edgecolor=ORANGE, lw=1.6, zorder=3))
axG.add_patch(Rectangle((CX0, CY1 - SLAB), SLAB, SLAB,
                        facecolor=GREEN_F, edgecolor=TEAL_D, lw=2.2, zorder=4))

for i in range(1, 9):
    x = AX0 + i * SLICE
    axG.plot([x, x], [AY1 - SLAB, AY1], color=BLUE, lw=0.6, alpha=0.5, zorder=4)
    y = BY1 - i * SLICE
    axG.plot([BX0, BX0 + SLAB], [y, y], color=ORANGE, lw=0.6, alpha=0.5, zorder=4)

axG.add_patch(Rectangle((AX0, AY1 - SLAB), SLICE, SLAB,
                        facecolor=BLUE, alpha=0.6, edgecolor=BLUE_D, lw=1.8, zorder=5))
axG.add_patch(Rectangle((BX0, BY1 - SLICE), SLAB, SLICE,
                        facecolor=ORANGE, alpha=0.6, edgecolor=ORANGE_D, lw=1.8, zorder=5))

# --- k axes
axG.annotate("", xy=(AX1 - 0.4, AY1 + 2.2), xytext=(AX0 + 0.4, AY1 + 2.2),
             arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.5))
axG.text((AX0 + AX1) / 2, AY1 + 3.0, "k   →   K / BK = 512 slices", fontsize=11.5,
         color=INK_SOFT, ha="center", va="bottom")
axG.annotate("", xy=(BX0 - 2.4, BY0 + 0.4), xytext=(BX0 - 2.4, BY1 - 0.4),
             arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.5))
axG.text(BX0 - 3.4, (BY0 + BY1) / 2, "k", fontsize=11.5, color=INK_SOFT,
         ha="right", va="center")

# --- callouts
axG.annotate(
    "slab of A\n128 rows × 4096  =  2 MB",
    xy=(AX0 + 22, AY1 - SLAB - 0.3), xytext=(AX0 + 16, AY1 - SLAB - 5.0),
    fontsize=11.5, color=BLUE_D, ha="center", va="top", linespacing=1.4,
    arrowprops=dict(arrowstyle="->", color=BLUE_D, lw=1.4),
)
axG.annotate(
    "slice of A\n128 × BK  =  4 KB",
    xy=(AX0 + SLICE / 2, AY1 + 0.3), xytext=(AX0 - 1.5, AY1 + 7.0),
    fontsize=11.5, color=BLUE_D, ha="center", va="bottom", linespacing=1.4,
    arrowprops=dict(arrowstyle="->", color=BLUE_D, lw=1.4),
)
axG.annotate(
    "slab of B\n4096 × 128 cols  =  2 MB",
    xy=(BX0 + SLAB + 0.3, BY0 + 10), xytext=(BX0 + SLAB + 3.5, BY0 + 7),
    fontsize=11.5, color=ORANGE_D, ha="left", va="center", linespacing=1.4,
    arrowprops=dict(arrowstyle="->", color=ORANGE_D, lw=1.4),
)
axG.annotate(
    "slice of B\nBK × 128  =  4 KB",
    xy=(BX0 + SLAB + 0.3, BY1 - SLICE / 2), xytext=(BX0 + SLAB + 3.5, BY1 - 2.5),
    fontsize=11.5, color=ORANGE_D, ha="left", va="center", linespacing=1.4,
    arrowprops=dict(arrowstyle="->", color=ORANGE_D, lw=1.4),
)
axG.annotate(
    "this CTA's 128 × 128 tile of C\n256 threads, 64 accumulators each",
    xy=(CX0 + SLAB + 0.3, CY1 - SLAB / 2), xytext=(CX0 + 7.0, CY1 - 9.0),
    fontsize=11.5, color=TEAL_D, ha="left", va="center", linespacing=1.4,
    arrowprops=dict(arrowstyle="->", color=TEAL_D, lw=1.4),
)

axG.add_patch(FancyArrowPatch((AX1 + 0.5, AY1 - SLAB / 2), (CX0 - 0.7, CY1 - SLAB / 2),
                              arrowstyle="-|>", mutation_scale=15, color=BLUE, lw=1.5,
                              ls=(0, (4, 3)), zorder=6))
axG.add_patch(FancyArrowPatch((BX0 + SLAB / 2, BY0 - 0.5), (CX0 + SLAB / 2, CY1 + 0.7),
                              arrowstyle="-|>", mutation_scale=15, color=ORANGE, lw=1.5,
                              ls=(0, (4, 3)), zorder=6))

axG.text(44, 92, "one slice pair = 8 KB,  and it feeds 8 rank-1 updates",
         fontsize=13, color=INK, ha="center", va="center", fontweight="bold",
         bbox=dict(facecolor=GREEN_F, edgecolor="none", pad=7.0))

# ============================================================ the footprints
axB = fig.add_axes([0.620, 0.520, 0.340, 0.260])
despine(axB, keep=("bottom",))
axB.set_xscale("log")

BARS = [
    ("one slice pair,  BK = 8", 8, TEAL_D, "8 KB"),
    ("double-buffered,  BK = 8", 16, TEAL_M, "16 KB"),
    ("T4:  shared memory per CTA", 32, ORANGE, "32 KB"),
    ("T4:  shared memory per SM", 64, ORANGE, "64 KB"),
    ("H100:  shared memory per SM", 228, H100_ACCENT, "228 KB"),
    ("both slabs,  the whole k range", 4096, RED_M, "4 MB"),
]

for i, (lab, kb, col, txt) in enumerate(BARS):
    y = len(BARS) - 1 - i
    axB.barh(y, kb, height=0.56, left=1.0, color=col, edgecolor="none", zorder=3)
    axB.text(kb * 1.2, y, txt, fontsize=12, color=col, ha="left", va="center",
             fontweight="bold", zorder=4)
    axB.text(0.86, y, lab, fontsize=12, color=INK_SOFT, ha="right", va="center")

axB.set_ylim(-0.8, len(BARS) - 0.2)
axB.set_xlim(1, 26000)
axB.set_yticks([])
axB.set_xticks([1, 8, 64, 512, 4096])
axB.set_xticklabels(["1 KB", "8 KB", "64 KB", "512 KB", "4 MB"], fontsize=11.5)
axB.tick_params(axis="x", length=4, color=GREY_L)
axB.text(-0.47, 1.30, "what has to fit in shared memory", transform=axB.transAxes,
         fontsize=17, fontweight="bold", color=INK, ha="left", va="bottom")
axB.text(-0.47, 1.13, "the slabs miss by 64× on the T4;  a slice pair clears it with room to spare",
         transform=axB.transAxes, fontsize=11.5, color=GREY, ha="left", va="bottom")

# =============================================================== the k tile
axT = fig.add_axes([0.470, 0.050, 0.505, 0.330])
axT.set_xlim(-18, 112)
axT.set_ylim(-6.6, 11.6)
axT.axis("off")

W_LOAD, W_COMP, GAP = 13.0, 28.0, 3.0
PER = W_LOAD + W_COMP + GAP
YB, YH = 3.2, 1.8

axT.text(-18, 11.3, "one iteration of the k loop, without double buffering",
         fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
axT.text(-18, 9.9, "128 × 128 × 8 = 131,072 FMAs per k tile ≈ 4,096 T4 cycles with two CTAs on the SM",
         fontsize=11.5, color=GREY, ha="left", va="top")

for t in range(2):
    x = t * PER
    axT.add_patch(Rectangle((x, YB), W_LOAD, YH, facecolor=ORANGE_F,
                            edgecolor=ORANGE, lw=1.5, zorder=3))
    axT.text(x + W_LOAD / 2, YB + YH / 2, f"LDG → STS\nslice {t}", fontsize=9.8,
             color=ORANGE_D, ha="center", va="center", linespacing=1.35, zorder=4)
    axT.add_patch(Rectangle((x + W_LOAD, YB), W_COMP, YH, facecolor=GREEN_F,
                            edgecolor=TEAL_D, lw=1.5, zorder=3))
    axT.text(x + W_LOAD + W_COMP / 2, YB + YH / 2, f"8 rank-1 updates on slice {t}",
             fontsize=10.2, color=TEAL_D, ha="center", va="center", zorder=4)
    axT.add_patch(Rectangle((x + W_LOAD + W_COMP, YB), GAP, YH, facecolor=IDLE,
                            edgecolor=IDLE_E, lw=1.0, zorder=3))
    for bx in (x + W_LOAD, x + PER):
        axT.plot([bx, bx], [YB - 0.5, YB + YH + 0.7], color=RED, lw=1.6,
                 ls=(0, (3, 2.5)), zorder=5)
        axT.text(bx, YB + YH + 0.9, "barrier", fontsize=9, color=RED,
                 ha="center", va="bottom")
    axT.text(x + PER / 2, YB + YH + 2.8, f"k tile  t = {t}", fontsize=11,
             color=INK_SOFT, ha="center", va="bottom")

axT.text(2 * PER + 2.5, YB + YH / 2, "…     × 512", fontsize=13, color=INK_SOFT,
         ha="left", va="center", fontweight="bold")
axT.text(-18, YB + YH / 2, "SM", fontsize=12.5, color=INK_SOFT, ha="left", va="center")

axT.annotate(
    "Two barriers per k tile: the first waits until the slice has fully arrived, the second until every warp\n"
    "has finished reading it. So the loads for slice t+1 cannot start until the compute on slice t is done,\n"
    "and the SM has nothing to issue while the next slice travels from global memory.",
    xy=(PER + W_LOAD / 2, YB - 0.3), xytext=(46.0, -1.4),
    fontsize=11.5, color=ORANGE_D, ha="center", va="top", linespacing=1.5,
    arrowprops=dict(arrowstyle="->", color=ORANGE_D, lw=1.5),
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
