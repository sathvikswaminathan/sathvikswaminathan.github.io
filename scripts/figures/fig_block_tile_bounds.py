"""block-tile-bounds.png

Top row: the feasible window for a square block tile, one panel per GPU. The
floor comes from DRAM bandwidth (25 FLOPs/byte on the T4, 20 on the H100), the
ceiling from the register file (65,536 registers per SM on both).

Bottom row: the ladder of candidate CTA tiles, scored against both targets.
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
    GREEN_F,
    GREY,
    GREY_L,
    H100_ACCENT,
    INK,
    INK_SOFT,
    ORANGE,
    RED,
    RED_F,
    RED_M,
    TEAL_D,
    TEAL_M,
    despine,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/block-tile-bounds.png"

B_MAX = 290
B_CEIL = 181

WINDOWS = [
    dict(
        name="Tesla T4",
        accent=ORANGE,
        target=25,
        floor=100,
        mem="GDDR6",
        calc="8.1 TFLOP/s  ÷  0.32 TB/s",
        note="the window is only 81 wide",
    ),
    dict(
        name="H100 SXM",
        accent=H100_ACCENT,
        target=20,
        floor=80,
        mem="HBM3",
        calc="67 TFLOP/s  ÷  3.35 TB/s",
        note="the window is 101 wide",
    ),
]

# (tile, exact AI, label shown, bar colour, verdict) — the exact AI is what the
# vs-target ratios are computed from, so they match the table in the post.
LADDER = [
    ("32 × 32", 8.0, "8.0", RED_M, "DRAM-bound on both"),
    ("64 × 64", 16.0, "16.0", RED_M, "DRAM-bound without L2 help on both"),
    ("64 × 128", 64 * 128 / (2 * (64 + 128)), "21.3", ORANGE, "DRAM-bound on the T4; no margin on the H100"),
    ("96 × 128", 96 * 128 / (2 * (96 + 128)), "27.4", TEAL_M, "clears both; awkward warp decomposition"),
    ("128 × 128", 32.0, "32.0", TEAL_D, "clears both, with headroom"),
    ("256 × 128", 256 * 128 / (2 * (256 + 128)), "42.7", TEAL_M, "more margin, at a price (only 1 CTA per SM)"),
    ("256 × 256", 64.0, "64.0", "#e5e5e5", "cannot launch (exceeds the register file)"),
]

fig = plt.figure(figsize=(17.6, 13.4), dpi=150)

fig.text(
    0.5,
    0.973,
    "Two budgets squeeze the block tile from both sides",
    ha="center",
    va="center",
    fontsize=27,
    fontweight="bold",
)
fig.text(
    0.5,
    0.945,
    "DRAM bandwidth sets a floor on B and the register file sets a ceiling — the floor moves with the GPU, the ceiling does not, "
    "and 128 is the only power of two left inside either window",
    ha="center",
    va="center",
    fontsize=14.5,
    color=GREY,
)

# ================================================================ windows
for i, w in enumerate(WINDOWS):
    ax = fig.add_axes([0.055 + i * 0.495, 0.455, 0.405, 0.405])
    despine(ax)

    ax.add_patch(Rectangle((0, 0), w["floor"], 80, facecolor=RED_F, edgecolor="none", zorder=0))
    ax.add_patch(
        Rectangle((w["floor"], 0), B_CEIL - w["floor"], 80, facecolor=GREEN_F, edgecolor="none", zorder=0)
    )
    ax.add_patch(
        Rectangle((B_CEIL, 0), B_MAX - B_CEIL, 80, facecolor=RED_F, edgecolor="none", zorder=0)
    )

    b = np.linspace(0, B_MAX, 400)
    ax.plot(b, b / 4, color=BLUE, lw=3.0, zorder=4)
    ax.text(236, 56, "AI = B / 4", fontsize=15.5, fontweight="bold", color=BLUE, ha="center")

    ax.axhline(w["target"], color=w["accent"], ls=(0, (7, 4)), lw=2.4, zorder=3)
    ax.text(
        2,
        w["target"] + 4.6,
        f"{w['target']} FLOPs per byte",
        fontsize=13,
        fontweight="bold",
        color=w["accent"],
        va="bottom",
    )
    ax.text(2, w["target"] + 1.5, w["calc"], fontsize=11.5, color=w["accent"], va="bottom")

    for xb, lab, side in [(w["floor"], f"B = {w['floor']}", -1), (B_CEIL, "B ≈ 181", 1)]:
        ax.plot([xb, xb], [0, 80], color=RED, ls=(0, (4, 3)), lw=1.8, zorder=3)
        ax.text(
            xb + side * 3.5,
            76.5,
            lab,
            fontsize=14,
            fontweight="bold",
            color=RED,
            ha="right" if side < 0 else "left",
            va="bottom",
        )

    ax.text(
        w["floor"] / 2,
        64,
        f"DRAM-bound\n\nAI < {w['target']}, so the\ncores wait on\nglobal memory",
        fontsize=11.5,
        color=RED,
        ha="center",
        va="top",
        linespacing=1.35,
    )
    ax.text(
        (B_CEIL + B_MAX) / 2,
        17,
        "will not fit\n\n2B² > 65,536 registers,\nthe CTA cannot be resident",
        fontsize=11.5,
        color=RED,
        ha="center",
        va="top",
        linespacing=1.35,
    )

    ax.plot([128], [32], marker="o", ms=16, mfc=GREEN_F, mec=TEAL_D, mew=2.6, zorder=6)
    ax.annotate(
        "B = 128\nAI = 32 FLOPs per byte",
        xy=(128, 33.8),
        xytext=(143, 60),
        fontsize=12.5,
        fontweight="bold",
        color=TEAL_D,
        ha="center",
        linespacing=1.35,
        arrowprops=dict(arrowstyle="->", color=TEAL_D, lw=1.6),
    )
    ax.text(
        (w["floor"] + B_CEIL) / 2,
        4.0,
        "feasible",
        fontsize=15,
        fontweight="bold",
        color=TEAL_D,
        ha="center",
    )

    ax.set_xlim(0, B_MAX)
    ax.set_ylim(0, 80)
    ax.set_xticks([0, 64, w["floor"], 128, B_CEIL, 256])
    ax.set_xticklabels(["0", "64", str(w["floor"]), "128", "181", "256"], fontsize=12)
    ax.set_xlabel("square block tile side  B   (CTA computes B × B of C)", fontsize=13, labelpad=8)
    ax.set_ylabel("arithmetic intensity  (FLOPs per byte of DRAM traffic)", fontsize=12.5)
    ax.set_title(
        f"{w['name']}  —  {w['mem']},  target {w['target']} FLOPs/byte",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )

# ================================================================= ladder
ax = fig.add_axes([0.055, 0.065, 0.845, 0.295])
ax.axis("off")
ax.set_xlim(0, 128)
ax.set_ylim(-1.9, len(LADDER) + 1.9)

X_BAR = 14.0
SCALE = 0.72  # AI units -> axis units for the bar region

ax.text(
    0,
    len(LADDER) + 1.3,
    "the ladder of CTA tiles",
    fontsize=18,
    fontweight="bold",
    ha="left",
    va="bottom",
)

for col_x, head, col in [(68.0, "vs 25\n(T4)", ORANGE), (79.0, "vs 20\n(H100)", H100_ACCENT)]:
    ax.text(
        col_x, len(LADDER) - 0.30, head, fontsize=12.5, fontweight="bold", color=col, ha="center",
        va="bottom", linespacing=1.3,
    )
ax.text(
    90.0, len(LADDER) - 0.30, "verdict", fontsize=12.5, fontweight="bold", color=GREY, ha="left", va="bottom"
)

for j, (name, ai, ai_lab, colr, verdict) in enumerate(LADDER):
    y = len(LADDER) - 1 - j
    launches = ai < 64.0
    ax.text(
        X_BAR - 1.5,
        y,
        name,
        fontsize=14.5,
        fontweight="bold",
        ha="right",
        va="center",
        color=INK if launches else GREY,
    )
    ax.add_patch(
        Rectangle(
            (X_BAR, y - 0.30),
            ai * SCALE,
            0.60,
            facecolor=colr,
            edgecolor="none",
        )
    )
    ax.text(
        X_BAR + ai * SCALE + 1.2,
        y,
        ai_lab,
        fontsize=13.5,
        ha="left",
        va="center",
        fontweight="bold" if name == "128 × 128" else "normal",
        color=INK if launches else GREY,
        zorder=7,
        bbox=dict(facecolor=BG, edgecolor="none", pad=1.6),
    )

    if launches:
        for col_x, tgt, col in [(68.0, 25, ORANGE), (79.0, 20, H100_ACCENT)]:
            ratio = ai / tgt
            ok = ratio >= 1.0
            ax.text(
                col_x - 2.6,
                y,
                "✓" if ok else "✗",
                fontsize=13,
                ha="center",
                va="center",
                color=col if ok else RED,
                fontfamily="DejaVu Sans",
            )
            ax.text(
                col_x + 0.6,
                y,
                f"{ratio:.2f}×",
                fontsize=13,
                ha="center",
                va="center",
                color=col if ok else RED,
                fontweight="bold" if ok else "normal",
            )
    else:
        for col_x in (68.0, 79.0):
            ax.text(col_x, y, "—", fontsize=13, ha="center", va="center", color=GREY)

    ax.text(90.0, y, verdict, fontsize=12.5, ha="left", va="center", color=GREY)

# target lines over the bar region only
for tgt, col, lab, ly in [
    (20, H100_ACCENT, "20  H100", len(LADDER) + 0.62),
    (25, ORANGE, "25  T4", len(LADDER) - 0.18),
]:
    xx = X_BAR + tgt * SCALE
    ax.plot([xx, xx], [-0.55, ly - 0.08], color=col, ls=(0, (6, 4)), lw=2.2, zorder=5)
    ax.text(xx, ly, lab, fontsize=12.5, fontweight="bold", color=col, ha="center", va="bottom")

ax.annotate(
    "",
    xy=(X_BAR + 64 * SCALE, -0.75),
    xytext=(X_BAR, -0.75),
    arrowprops=dict(arrowstyle="-", color=GREY_L, lw=1.4),
)
for tick in (0, 20, 40, 60):
    xx = X_BAR + tick * SCALE
    ax.plot([xx, xx], [-0.75, -0.88], color=GREY_L, lw=1.4)
    ax.text(xx, -1.02, str(tick), fontsize=11.5, color=INK_SOFT, ha="center", va="top")
ax.text(
    X_BAR + 32 * SCALE,
    -1.52,
    "arithmetic intensity  (FLOPs per byte)",
    fontsize=13,
    color=INK_SOFT,
    ha="center",
    va="top",
)

fig.text(
    0.5,
    0.013,
    "Every byte is assumed to come from DRAM. In practice CTAs that share rows of A or columns of B get some of their data from L2 — "
    "but the T4 has 4 MB of it against the H100's 50 MB, so there is far less to lean on.",
    ha="center",
    va="center",
    fontsize=12.5,
    color=GREY,
    style="italic",
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
