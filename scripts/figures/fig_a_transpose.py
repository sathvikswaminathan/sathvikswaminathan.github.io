"""a-transpose-banks.png

Where one warp's A operands land in the 32 shared-memory banks, before and
after transposing the A slice on its way into shared memory.

Left:  As[m][k], the order the slice arrives in. One LDS.32 for a[0] at k = 0
       reads addresses 0, 64, 128, 192 — all multiples of 32, so all bank 0.
Right: As[k][m]. The same 32 values now cover all 32 banks exactly once, and a
       lane's 8 values are contiguous, so 2 LDS.128 replace 8 LDS.32.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from _style import (
    BG,
    BLUE_D,
    BLUE_F,
    GREY,
    GREY_L,
    INK,
    INK_SOFT,
    RED,
    RED_F,
    TEAL_D,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/a-transpose-banks.png"

NBANK = 32
NROW = 8

XLIM = (-9.0, 33.0)
YLIM = (-9.6, 15.4)

# vertical bands, in data units
Y_TITLE = 15.0      # va top
Y_SUB = 13.3        # va top
Y_BANKHDR = 11.3    # va bottom
Y_BANKNUM = 10.3    # va bottom
Y_GRP_LM = 9.15     # va bottom  (right panel)
Y_GRP_M = 8.25      # va bottom  (right panel)
Y_ANNOT = -1.2      # va top
Y_VERDICT = -6.7    # centre
Y_NOTE = -8.7       # centre

XSPAN, YSPAN = XLIM[1] - XLIM[0], YLIM[1] - YLIM[0]
FIG_W = 18.4
PANEL_W_FRAC = 0.450
PANEL_H_IN = (PANEL_W_FRAC * FIG_W) * YSPAN / XSPAN  # square cells
TOP_BAND_IN, BOT_BAND_IN = 1.15, 0.32
FIG_H = PANEL_H_IN + TOP_BAND_IN + BOT_BAND_IN
PANEL_Y = BOT_BAND_IN / FIG_H
PANEL_H_FRAC = PANEL_H_IN / FIG_H


def bank_grid(ax, fills, thick_rows, row_labels, row_label_head):
    """Draw an NBANK x NROW grid of floats; fills maps (row, col) -> facecolor."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")

    for r in range(NROW):
        y = NROW - 1 - r
        for c in range(NBANK):
            ax.add_patch(
                Rectangle(
                    (c, y), 1, 1,
                    facecolor=fills.get((r, c), "#ffffff"),
                    edgecolor=GREY_L, lw=0.5, zorder=2,
                )
            )

    for r in thick_rows:  # boundaries between rows of As
        ax.plot([0, NBANK], [NROW - r, NROW - r], color=INK_SOFT, lw=1.8, zorder=4)
    ax.plot([0, NBANK], [0, 0], color=INK_SOFT, lw=1.8, zorder=4)
    for x in (0, NBANK):
        ax.plot([x, x], [0, NROW], color=INK_SOFT, lw=1.8, zorder=4)

    for c in range(NBANK):
        ax.text(c + 0.5, Y_BANKNUM, str(c), fontsize=7.6, color=INK_SOFT, ha="center", va="bottom")
    ax.text(NBANK / 2, Y_BANKHDR, "shared-memory bank   (address mod 32)",
            fontsize=12.5, color=INK_SOFT, ha="center", va="bottom")

    ax.text(-0.8, Y_BANKNUM, row_label_head, fontsize=10.5, color=GREY, ha="right", va="bottom")
    for r, lab in enumerate(row_labels):
        if lab:
            ax.text(-0.8, NROW - 0.5 - r, lab, fontsize=10.5, color=INK_SOFT, ha="right", va="center")


def heading(ax, title, sub):
    ax.text(NBANK / 2, Y_TITLE, title, fontsize=17.5, fontweight="bold",
            color=INK, ha="center", va="top")
    ax.text(NBANK / 2, Y_SUB, sub, fontsize=11.5, color=GREY, ha="center", va="top")


fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)

fig.text(0.5, 1 - 0.40 / FIG_H, "Transposing A on its way into shared memory",
         ha="center", va="center", fontsize=27, fontweight="bold")
fig.text(
    0.5, 1 - 0.82 / FIG_H,
    "The outer product needs a column of A at every k step. Stored in the order it arrives, that column sits entirely in one bank; "
    "transposed, it spreads across all 32.",
    ha="center", va="center", fontsize=14.5, color=GREY,
)

# ============================================================ left: row-major
axL = fig.add_axes([0.030, PANEL_Y, PANEL_W_FRAC, PANEL_H_FRAC])

fills = {}
for i in range(8):  # one lane's a[0..7]: addresses 0, 8, ..., 56
    a = 8 * i
    fills[(a // 32, a % 32)] = BLUE_F
for lm in range(4):  # the 4 distinct addresses of one LDS.32 for a[0], k = 0
    a = 64 * lm
    fills[(a // 32, a % 32)] = RED_F

bank_grid(axL, fills, thick_rows=range(1, NROW),
          row_labels=[str(32 * r) for r in range(NROW)], row_label_head="addr")
heading(axL, "stored as it arrives:   As[m][k]  at  m · 8 + k",
        "BK = 8, so four rows of As fit in one pass over the banks")

for lm in range(4):
    a = 64 * lm
    axL.add_patch(Rectangle((a % 32, NROW - 1 - a // 32), 1, 1,
                            facecolor="none", edgecolor=RED, lw=2.3, zorder=6))
axL.add_patch(Rectangle((-0.22, -0.22), 1.44, NROW + 0.44,
                        facecolor="none", edgecolor=RED, lw=2.0, ls=(0, (4, 3)), zorder=5))
axL.annotate(
    "one LDS.32 for a[0] at k = 0 reads four distinct addresses —\n"
    "0, 64, 128 and 192 — and every one is a multiple of 32,\n"
    "so all four are bank 0.",
    xy=(0.5, -0.45), xytext=(3.4, Y_ANNOT),
    fontsize=12.5, color=RED, ha="left", va="top", linespacing=1.45,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.8),
)

axL.add_patch(Rectangle((-0.18, NROW - 2.18), 32.36, 2.36,
                        facecolor="none", edgecolor=BLUE_D, lw=1.7, ls=(0, (3, 2.5)), zorder=5))
axL.annotate(
    "a lane's own a[0..7] are 8 floats apart → 8 separate LDS.32",
    xy=(27.0, 7.55), xytext=(7.6, Y_GRP_LM),
    fontsize=12, color=BLUE_D, ha="left", va="bottom",
    arrowprops=dict(arrowstyle="->", color=BLUE_D, lw=1.6),
)

axL.text(NBANK / 2, Y_VERDICT, "8 × LDS.32 per k step,  each serialized into 4 wavefronts",
         fontsize=14.5, fontweight="bold", color=RED, ha="center", va="center",
         bbox=dict(facecolor=RED_F, edgecolor="none", pad=9.0))

# =========================================================== right: transposed
axR = fig.add_axes([0.520, PANEL_Y, PANEL_W_FRAC, PANEL_H_FRAC])

GRP = [TEAL_D, "#2e7ebb", "#7b5ea7", "#e8872a"]
GRP_L = ["#bfe0d8", "#cfe2f1", "#d8cfe8", "#f6ddc2"]

fills = {(0, 8 * lm + j): GRP_L[lm] for lm in range(4) for j in range(8)}

bank_grid(axR, fills, thick_rows=(4,),
          row_labels=["As[0][·]", "", "", "", "As[1][·]", "", "", ""], row_label_head="row of As")
heading(axR, "transposed on the store:   As[k][m]  at  k · 128 + m",
        "a row of As is now BM = 128 floats, four passes over the banks")

y0 = NROW - 1
for lm in range(4):
    x0 = 8 * lm
    for j in range(4):  # first LDS.128 darker than the second
        axR.add_patch(Rectangle((x0 + j, y0), 1, 1, facecolor=GRP[lm], alpha=0.40,
                                edgecolor="none", zorder=3))
    axR.add_patch(Rectangle((x0, y0), 8, 1, facecolor="none", edgecolor=GRP[lm], lw=2.4, zorder=6))
    axR.text(x0 + 4, Y_GRP_LM, f"lane group  lm = {lm}", fontsize=10.2, color=GRP[lm],
             ha="center", va="bottom", fontweight="bold")
    axR.text(x0 + 4, Y_GRP_M, f"m = {8 * lm}–{8 * lm + 7}", fontsize=9.8, color=GRP[lm],
             ha="center", va="bottom")

axR.annotate(
    "the warp's 32 distinct A values for k = 0 are contiguous and\n"
    "cover all 32 banks exactly once, and each lane group's\n"
    "8 floats come in as 2 × LDS.128.",
    xy=(3.4, Y_ANNOT), xytext=(3.4, Y_ANNOT),
    fontsize=12.5, color=TEAL_D, ha="left", va="top", linespacing=1.45,
)

axR.text(NBANK / 2, Y_VERDICT, "2 × LDS.128 per k step,  conflict-free",
         fontsize=14.5, fontweight="bold", color=TEAL_D, ha="center", va="center",
         bbox=dict(facecolor="#dcefe9", edgecolor="none", pad=9.0))
axR.text(NBANK / 2, Y_NOTE, "shaded = the first LDS.128  (a[0..3]);    lighter = the second  (a[4..7])",
         fontsize=11, color=GREY, ha="center", va="center")

# arrow between the panels
mid_y = PANEL_Y + PANEL_H_FRAC * (NROW / 2 - YLIM[0]) / YSPAN
fig.patches.append(
    FancyArrowPatch((0.4845, mid_y), (0.5125, mid_y), transform=fig.transFigure,
                    arrowstyle="-|>", mutation_scale=26, color=INK_SOFT, lw=2.0)
)
fig.text(0.4985, mid_y + 0.018, "transpose", fontsize=12, color=INK_SOFT,
         ha="center", va="bottom", fontweight="bold")

fig.text(
    0.5, 0.012,
    "Lane (lm, ln) of the 4 × 8 lane grid owns rows 8·lm … 8·lm+7 of the warp tile, so the 8 lanes sharing an lm ask for the same A address — "
    "that is a broadcast, not a conflict. What changes between the two layouts is where the 4 distinct addresses land.",
    ha="center", va="center", fontsize=12.5, color=GREY, style="italic",
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT, f"({FIG_W:.1f} x {FIG_H:.1f} in)")
