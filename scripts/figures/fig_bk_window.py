"""bk-window.png

The window for BK on each GPU.

Top:    shared memory per CTA is 2 KB x BK with double buffering, against each
        CTA's half of the SM's shared memory. The floor at BK = 8 comes from
        sector efficiency on the A slice.
Bottom: on the T4 the register file closes the window before shared memory
        does, because every thread stages BK floats of the next slice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _style import (
    BG,
    BLUE,
    BLUE_F,
    GREEN_F,
    GREY,
    GREY_L,
    H100_ACCENT,
    IDLE,
    INK,
    INK_SOFT,
    ORANGE,
    RED,
    RED_F,
    TEAL_D,
    TEAL_M,
    despine,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/bk-window.png"
ORANGE_D = "#9a5a12"

FIG_W, FIG_H = 17.6, 10.0
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)

fig.text(0.5, 1 - 0.42 / FIG_H, "How deep a slice can be",
         ha="center", va="center", fontsize=27, fontweight="bold")
fig.text(
    0.5, 1 - 0.80 / FIG_H,
    "Sector efficiency sets the floor at BK = 8. The ceiling is whichever budget runs out first — shared memory on the H100, "
    "the register file on the T4.",
    ha="center", va="center", fontsize=14.5, color=GREY,
)

BK_MAX, SMEM_MAX = 68, 140

PANELS = [
    dict(
        name="Tesla T4",
        accent=ORANGE,
        budget=32,
        budget_lab="32 KB per CTA   (64 KB per SM, split between two CTAs)",
        smem_cap=16,
        hard_cap=8,
        chosen=8,
        oob_xy=(37, 112),
        pick_xy=(27, 82),
        reg_block=(8, 16),
        marks=[4, 8, 16, 32],
        note=("Shared memory alone would allow BK = 16: its two slices take exactly 32 KB,\n"
              "the CTA's entire share, with nothing left over. The register file closes the\n"
              "window first — see below."),
    ),
    dict(
        name="H100 SXM",
        accent=H100_ACCENT,
        budget=113,
        budget_lab="≈ 113 KB per CTA   (226 KB per SM after Hopper's 1 KB reservation)",
        smem_cap=56,
        hard_cap=32,
        chosen=32,
        oob_xy=(20, 136),
        pick_xy=(47, 32),
        reg_block=None,
        marks=[8, 16, 32, 64],
        note=("A cp.async kernel stages its slices without touching registers, so shared\n"
              "memory is the only limit. BK ≤ 56 fits; the largest power of two is 32, which\n"
              "cuts the k tiles from 512 to 128 and pays for its barriers a quarter as often."),
    ),
]

for i, p in enumerate(PANELS):
    ax = fig.add_axes([0.055 + i * 0.490, 0.455, 0.405, 0.345])
    despine(ax)

    # out-of-bounds shading
    ax.add_patch(Rectangle((0, 0), 8, SMEM_MAX, facecolor=RED_F, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((8, p["budget"]), BK_MAX - 8, SMEM_MAX - p["budget"],
                           facecolor=RED_F, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((8, 0), p["smem_cap"] - 8, p["budget"],
                           facecolor=GREEN_F, edgecolor="none", zorder=0))

    bk = np.linspace(0, BK_MAX, 300)
    ax.plot(bk, 2 * bk, color=BLUE, lw=3.0, zorder=4)
    ax.text(52.5, 2 * 52.5 - 16, "2 KB × BK", fontsize=14.5, fontweight="bold",
            color=BLUE, ha="left", va="center")

    ax.axhline(p["budget"], color=p["accent"], ls=(0, (7, 4)), lw=2.4, zorder=3)
    ax.text(19, p["budget"] + 2.5, p["budget_lab"], fontsize=11,
            fontweight="bold", color=p["accent"], ha="left", va="bottom", zorder=6,
            bbox=dict(facecolor=BG, edgecolor="none", pad=2.0))

    for x, lab, col in [(8, "BK = 8", RED), (p["smem_cap"], f"BK = {p['smem_cap']}", RED)]:
        ax.plot([x, x], [0, SMEM_MAX], color=col, ls=(0, (4, 3)), lw=1.8, zorder=3)
    ax.text(7.0, SMEM_MAX - 4, "BK = 8", fontsize=13, fontweight="bold", color=RED,
            ha="right", va="top")
    ax.text(p["smem_cap"] + 1.5, SMEM_MAX - 4, f"BK = {p['smem_cap']}", fontsize=13,
            fontweight="bold", color=RED, ha="left", va="top")

    ax.text(4.0, SMEM_MAX * 0.52, "sector\nefficiency", fontsize=11, color=RED,
            ha="center", va="top", linespacing=1.4, fontweight="bold")
    ax.text(*p["oob_xy"], "will not fit in the CTA's\nshare of shared memory",
            fontsize=11, color=RED, ha="left", va="top", linespacing=1.4)

    if p["reg_block"]:
        lo, hi = p["reg_block"]
        ax.add_patch(Rectangle((lo, 0), hi - lo, p["budget"], facecolor="none",
                               edgecolor=GREY, lw=0.0, hatch="\\\\", alpha=0.55, zorder=1))
        ax.add_patch(Rectangle((lo, 0), hi - lo, p["budget"], facecolor="none",
                               edgecolor=GREY_L, lw=1.2, zorder=2))
        ax.text((lo + hi) / 2 + 0.5, 5.5, "registers\nsay no", fontsize=9.5,
                color=INK_SOFT, ha="center", va="center", linespacing=1.35)

    for m in p["marks"]:
        ok = 8 <= m <= p["hard_cap"]
        ax.plot([m], [2 * m], marker="o", ms=9,
                mfc=GREEN_F if ok else "#ffffff",
                mec=TEAL_D if ok else GREY, mew=2.0, zorder=6)

    ax.plot([p["chosen"]], [2 * p["chosen"]], marker="o", ms=16, mfc=GREEN_F,
            mec=TEAL_D, mew=2.6, zorder=7)
    ax.annotate(
        f"BK = {p['chosen']}\n{2 * p['chosen']} KB per CTA\n{4096 // p['chosen']} k tiles",
        xy=(p["chosen"] + 0.8, 2 * p["chosen"] + 1.5),
        xytext=p["pick_xy"],
        fontsize=12, fontweight="bold", color=TEAL_D, ha="center", linespacing=1.4,
        arrowprops=dict(arrowstyle="->", color=TEAL_D, lw=1.6),
    )

    ax.set_xlim(0, BK_MAX)
    ax.set_ylim(0, SMEM_MAX)
    ax.set_xticks([0, 8, 16, 32, p["smem_cap"], 64] if p["smem_cap"] not in (16, 64)
                  else [0, 8, 16, 32, 64])
    ax.set_xticklabels([str(int(t)) for t in ax.get_xticks()], fontsize=12)
    ax.set_xlabel("BK   (k steps staged in shared memory at a time)", fontsize=13, labelpad=8)
    ax.set_ylabel("shared memory per CTA  (KB, double-buffered)", fontsize=12.5)
    ax.text(0, 1.215, p["name"], transform=ax.transAxes, fontsize=19,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0, 1.020, p["note"], transform=ax.transAxes, fontsize=11.5, color=GREY,
            ha="left", va="bottom", linespacing=1.45)

# ================================================= the T4 register ledger
axR = fig.add_axes([0.055, 0.075, 0.895, 0.250])
axR.axis("off")
axR.set_xlim(-46, 185)
axR.set_ylim(-1.5, 3.3)

SEGS = [
    ("64 accumulators", 64, GREEN_F, TEAL_D),
    ("32 operands  (2 × (8 + 8), double-buffered)", 32, BLUE_F, BLUE),
    ("BK staging registers", None, "#f6ddc2", ORANGE),
    ("≈ 24 for addresses, counters, indices", 24, IDLE, GREY),
]

axR.text(-46, 3.1, "why the T4 cannot go deeper:  the register ledger of one thread",
         fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
axR.text(-46, 2.42,
         "Turing has no cp.async, so every thread holds its share of the next slice — BK floats — in registers for the whole k tile.",
         fontsize=12, color=GREY, ha="left", va="top")

for row, (bk_val, y) in enumerate([(8, 1.25), (16, 0.25)]):
    x = 0.0
    total = 0
    for lab, n, fill, edge in SEGS:
        n = bk_val if n is None else n
        axR.add_patch(Rectangle((x, y), n, 0.62, facecolor=fill, edgecolor=edge,
                                lw=1.4, zorder=3))
        if n >= 20 or row == 0:
            axR.text(x + n / 2, y + 0.31, str(n), fontsize=10.5, color=edge,
                     ha="center", va="center", fontweight="bold", zorder=4)
        x += n
        total += n

    ok = total <= 128
    axR.text(-2.0, y + 0.31, f"BK = {bk_val}", fontsize=13, color=INK_SOFT,
             ha="right", va="center", fontweight="bold")
    axR.text(x + 2.5, y + 0.31,
             f"{total} registers per thread" + ("" if ok else "   ✗"),
             fontsize=13, fontweight="bold", color=TEAL_D if ok else RED,
             ha="left", va="center", fontfamily="DejaVu Sans" if not ok else None)
    if not ok:
        axR.text(x + 2.5, y - 0.12,
                 "past each CTA's half of the register file — the second CTA has to go",
                 fontsize=11, color=RED, ha="left", va="center")
    else:
        axR.text(x + 2.5, y - 0.12,
                 "exactly 256 threads × 128 = 32,768 registers, half the file — two CTAs fit",
                 fontsize=11, color=TEAL_D, ha="left", va="center")

axR.plot([128, 128], [-0.1, 2.15], color=RED, lw=2.0, ls=(0, (5, 4)), zorder=5)
axR.text(128, 2.22, "128 registers per thread", fontsize=12, fontweight="bold",
         color=RED, ha="center", va="bottom")

# legend
lx = -46
for lab, n, fill, edge in SEGS:
    axR.add_patch(Rectangle((lx, -1.15), 4.5, 0.42, facecolor=fill, edgecolor=edge,
                            lw=1.2, zorder=3))
    axR.text(lx + 6.0, -0.94, lab, fontsize=11, color=INK_SOFT, ha="left", va="center")
    lx += 6.0 + len(lab) * 1.30

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
