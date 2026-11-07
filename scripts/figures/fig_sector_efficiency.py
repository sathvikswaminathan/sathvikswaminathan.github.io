"""sector-efficiency.png

The 32-byte sector is the smallest unit the memory system moves, which is what
sets the lower bound on BK.

Left:  the two extremes for a warp loading one float per lane — consecutive
       addresses (4 sectors, 100%) against a 128-byte stride (32 sectors, 12.5%).
Right: the shape of a slice. A row of a B slice is 128 contiguous floats and
       always fills its sectors; a row of an A slice is only BK floats long, so
       BK = 4 wastes half of every sector and BK = 8 fills it exactly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import (
    BG,
    BLUE_F,
    GREEN_F,
    GREY,
    GREY_L,
    IDLE,
    INK,
    INK_SOFT,
    RED,
    RED_F,
    TEAL_D,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/sector-efficiency.png"

SECTOR = 8  # floats per 32-byte sector
CELL_H = 1.0

# ------------------------------------------------------------------ layout (inches)
FIG_W, FIG_H = 18.0, 7.35

L_X, L_W = 0.55, 7.56
R_X, R_W = 9.30, 7.92

L_XLIM, L_YLIM = (-1.5, 36.3), (-3.4, 4.2)          # 37.8 x 7.6
R_XLIM = (-9.5, 38.5)                                # 48.0
R_YLIM_1, R_YLIM_2 = (-2.4, 4.0), (-3.7, 4.0)        # 1-row / 2-row panels

L_UNIT = L_W / (L_XLIM[1] - L_XLIM[0])
R_UNIT = R_W / (R_XLIM[1] - R_XLIM[0])


def rect(x_in, y_in, w_in, h_in):
    return [x_in / FIG_W, y_in / FIG_H, w_in / FIG_W, h_in / FIG_H]


def strip(ax, x0, y, n, used, fill, edge, cell_w=1.0):
    for i in range(n):
        u = used(i)
        ax.add_patch(Rectangle((x0 + i * cell_w, y), cell_w, CELL_H,
                               facecolor=fill if u else IDLE,
                               edgecolor=GREY_L, lw=0.5, zorder=2))
        if not u:
            ax.add_patch(Rectangle((x0 + i * cell_w, y), cell_w, CELL_H,
                                   facecolor="none", edgecolor=RED, lw=0.0,
                                   hatch="///", alpha=0.45, zorder=3))
    for s in range(n // SECTOR):
        ax.add_patch(Rectangle((x0 + s * SECTOR * cell_w, y), SECTOR * cell_w, CELL_H,
                               facecolor="none", edgecolor=edge, lw=2.0, zorder=4))


def badge(ax, x, y, txt, col):
    ax.text(x, y, txt, fontsize=15.5, fontweight="bold", color=col,
            ha="center", va="center",
            bbox=dict(facecolor=GREEN_F if col == TEAL_D else RED_F,
                      edgecolor="none", pad=6.0))


fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)

fig.text(0.5, 1 - 0.42 / FIG_H, "Sector efficiency sets the floor on BK",
         ha="center", va="center", fontsize=27, fontweight="bold")
fig.text(
    0.5, 1 - 0.83 / FIG_H,
    "Global memory moves 32-byte sectors, never single floats. A load that touches a sector pays for all 32 bytes of it, "
    "whether or not it uses them.",
    ha="center", va="center", fontsize=14.5, color=GREY,
)

HEAD_Y = 5.70
fig.text(L_X / FIG_W, HEAD_Y / FIG_H, "one warp, one float per lane",
         fontsize=18, fontweight="bold", color=INK, ha="left", va="center")
fig.text(R_X / FIG_W, HEAD_Y / FIG_H, "the shape of a slice",
         fontsize=18, fontweight="bold", color=INK, ha="left", va="center")


# ========================================================== left: the extremes
def extreme(y_in, h_in, title, note, moved, pct, col, used_fn, tail, foot):
    ax = fig.add_axes(rect(L_X, y_in, L_W, h_in))
    ax.set_xlim(*L_XLIM)
    ax.set_ylim(*L_YLIM)
    ax.axis("off")

    ax.text(-1.5, 3.9, title, fontsize=14.5, fontweight="bold", color=col,
            ha="left", va="top")
    ax.text(-1.5, 2.7, note, fontsize=11.5, color=INK_SOFT, ha="left", va="top")

    strip(ax, 0, 0, 32, used_fn, GREEN_F if col == TEAL_D else BLUE_F, col)
    for s in range(4):
        ax.text(s * SECTOR + SECTOR / 2, -0.4, f"sector {s}", fontsize=9.5,
                color=INK_SOFT, ha="center", va="top")
    if tail:
        ax.text(33.0, CELL_H / 2, tail, fontsize=11, color=INK_SOFT, ha="left", va="center")

    ax.text(0, -1.9, moved, fontsize=12.5, color=INK_SOFT, ha="left", va="center")
    badge(ax, 29.5, -1.9, pct, col)
    if foot:
        ax.text(0, -3.1, foot, fontsize=11.5, color=col, ha="left", va="center", style="italic")


extreme(3.75, 1.52,
        "consecutive addresses  —  coalesced",
        "lanes 0–31 read floats 0–31",
        "4 sectors = 128 bytes moved,   128 bytes used", "100%", TEAL_D,
        lambda i: True, None,
        "every byte that moves is used — this is what a coalesced load looks like")

extreme(1.53, 1.52,
        "a 128-byte stride  —  uncoalesced",
        "each lane reads a float 32 floats from its neighbour's",
        "32 sectors = 1,024 bytes moved,   128 bytes used", "12.5%", RED,
        lambda i: i % SECTOR == 0, "…  32 sectors in all",
        "one useful float per sector, and the other seven are thrown away")


# ========================================================= right: slice shape
def slice_panel(y_in, h_in, ylim, title, sub, rows, pct, badge_x, col, foot):
    ax = fig.add_axes(rect(R_X, y_in, R_W, h_in))
    ax.set_xlim(*R_XLIM)
    ax.set_ylim(*ylim)
    ax.axis("off")

    ax.text(-9.5, 3.8, title, fontsize=14.5, fontweight="bold", color=col,
            ha="left", va="top")
    ax.text(-9.5, 2.6, sub, fontsize=11.5, color=INK_SOFT, ha="left", va="top")

    for i, (lab, n, used_fn) in enumerate(rows):
        y = -i * 1.3
        strip(ax, 0, y, n, used_fn, BLUE_F, col)
        ax.text(-0.7, y + CELL_H / 2, lab, fontsize=10.5, color=INK_SOFT,
                ha="right", va="center")

    badge(ax, badge_x, 0.5 - (len(rows) - 1) * 0.65, pct, col)
    ax.text(-9.5, -1.0 - (len(rows) - 1) * 1.3, foot, fontsize=11.5,
            color=INK_SOFT, ha="left", va="top")


slice_panel(4.25, 1.02, R_YLIM_1,
            "a row of a B slice  —  any BK",
            "B is row-major, so the 128 floats of a row are contiguous",
            [("row of Bs", 32, lambda i: True)],
            "100%", 35.0, TEAL_D,
            "512 bytes, 16 full sectors — BK never enters into it   (4 of the 16 drawn)")

slice_panel(2.37, 1.24, R_YLIM_2,
            "a row of an A slice  —  BK = 4",
            "only 4 columns of each row, and the next row starts 4,096 floats away",
            [("row m", 8, lambda i: i < 4), ("row m+1", 8, lambda i: i < 4)],
            "50%", 12.5, RED,
            "16 bytes wanted out of every 32-byte sector moved")

slice_panel(0.49, 1.24, R_YLIM_2,
            "a row of an A slice  —  BK = 8",
            "8 columns is exactly one sector, so nothing is wasted",
            [("row m", 8, lambda i: True), ("row m+1", 8, lambda i: True)],
            "100%", 12.5, TEAL_D,
            "two neighbouring threads cover one A row with one LDG.128 each   →   BK ≥ 8")

fig.text(
    0.5, 0.030,
    "Hatched cells are bytes the memory system moves but the k tile does not want. The waste is not permanent — a later k tile reads those "
    "columns — but it costs a second trip through the memory pipeline for data that was already on the chip.",
    ha="center", va="center", fontsize=12.5, color=GREY, style="italic",
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
