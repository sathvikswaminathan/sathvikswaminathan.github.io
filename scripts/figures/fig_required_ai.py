"""required-arithmetic-intensity.png

2x2: rows are the two GPUs, columns are "no reuse" vs "4 FMAs per load".
The point of the figure is that the T4 has half the cores AND half the
shared-memory pipe, so both GPUs land on the same required AI of 4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _style import (
    BG,
    BLUE,
    BLUE_F,
    GREY,
    GREY_L,
    IDLE,
    IDLE_E,
    INK,
    INK_SOFT,
    TEAL,
    TEAL_D,
    ORANGE,
    blank,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/required-arithmetic-intensity.png"

GPUS = [
    dict(
        key="t4",
        header="Tesla T4  (Turing)",
        sub="64 FP32 CUDA cores per SM  ·  ~64 bytes/cycle out of shared memory",
        lanes=16,
        cores=64,
        fp32=16,
        bytes_=64,
        banks_live=16,
        pipe_note="32 banks × 4 bytes of bank width, but the Turing pipe sustains only ~64 bytes per cycle",
    ),
    dict(
        key="h100",
        header="H100 SXM  (Hopper)",
        sub="128 FP32 CUDA cores per SM  ·  ~128 bytes/cycle out of shared memory",
        lanes=32,
        cores=128,
        fp32=32,
        bytes_=128,
        banks_live=32,
        pipe_note="32 banks × 4 bytes  =  128 bytes per cycle, the full bank width",
    ),
]

N_BANKS = 32


def draw_panel(ax, gpu, reuse):
    blank(ax, xlim=(0, 100), ylim=(-26, 104))

    lanes, cores, fp32 = gpu["lanes"], gpu["cores"], gpu["fp32"]
    busy = cores if reuse else fp32 // 2

    # ---- panel heading -------------------------------------------------
    ax.text(
        50,
        100,
        "Every loaded value feeds 4 FMAs" if reuse else "Every FMA loads its own operands",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        50,
        92,
        "reuse out of the register file" if reuse else "no reuse — this is the inner product, AI = 0.5",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
    )

    # ---- shared-memory bank strip --------------------------------------
    x0, x1 = 4.0, 96.0
    cw = (x1 - x0) / N_BANKS
    for i in range(N_BANKS):
        live = i < gpu["banks_live"]
        ax.add_patch(
            Rectangle(
                (x0 + i * cw + cw * 0.10, 76),
                cw * 0.80,
                7.5,
                facecolor=BLUE if live else BLUE_F,
                edgecolor=BLUE if live else GREY_L,
                linewidth=0.8,
            )
        )
    ax.add_patch(
        Rectangle((x0, 76), x1 - x0, 7.5, facecolor="none", edgecolor=GREY_L, linewidth=1.0)
    )
    ax.text(50, 87.5, gpu["pipe_note"], ha="center", va="center", fontsize=11.5, color=GREY)

    if gpu["banks_live"] < N_BANKS:
        mid = x0 + gpu["banks_live"] * cw
        ax.plot([mid, mid], [74.5, 85.0], color=GREY, lw=1.0, ls=(0, (3, 3)))
        ax.text(
            (mid + x1) / 2,
            71.5,
            "throughput-limited",
            ha="center",
            va="center",
            fontsize=10.5,
            color=GREY,
            style="italic",
        )

    ax.text(
        x0 + gpu["banks_live"] * cw / 2,
        71.5,
        f"{fp32} FP32 values delivered per cycle",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=BLUE,
    )

    # ---- arrows into the cores -----------------------------------------
    n_arrows = 8
    gx0, gx1 = 11.0, 96.0
    for i in range(n_arrows):
        ax.annotate(
            "",
            xy=(gx0 + (i + 0.5) * (gx1 - gx0) / n_arrows, 59.5),
            xytext=(gx0 + (i + 0.5) * (gx1 - gx0) / n_arrows, 66.5),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, ls=(0, (2.2, 1.6))),
        )

    # ---- the SM's lanes, 4 SMSPs ---------------------------------------
    top, bot = 58.0, 16.0
    rh = (top - bot) / 4
    lw_ = (gx1 - gx0) / lanes
    for s in range(4):
        ry = top - (s + 1) * rh
        ax.text(gx0 - 2.0, ry + rh / 2, f"SMSP {s}", ha="right", va="center", fontsize=12, color=INK_SOFT)
        for l in range(lanes):
            idx = s * lanes + l
            on = idx < busy
            ax.add_patch(
                Rectangle(
                    (gx0 + l * lw_ + lw_ * 0.12, ry + rh * 0.12),
                    lw_ * 0.76,
                    rh * 0.76,
                    facecolor=TEAL if on else IDLE,
                    edgecolor=TEAL_D if on else IDLE_E,
                    linewidth=0.9 if on else 0.6,
                )
            )
        ax.add_patch(
            Rectangle((gx0, ry), gx1 - gx0, rh, facecolor="none", edgecolor=GREY_L, linewidth=0.9)
        )

    ax.text(
        (gx0 + gx1) / 2,
        10.5,
        f"{cores} FP32 CUDA cores  —  4 SMSPs × {lanes} lanes",
        ha="center",
        va="center",
        fontsize=13,
        color=INK_SOFT,
    )

    # ---- the arithmetic -------------------------------------------------
    ax.plot([16, 84], [1.5, 1.5], color="#e5e5e5", lw=1.6)

    if reuse:
        ax.text(
            50,
            -7.5,
            f"{fp32} loads  →  {cores} FMAs",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
        ax.text(
            50,
            -15.5,
            "every lane busy  —  100% of peak",
            ha="center",
            va="center",
            fontsize=13,
            color=GREY,
        )
        ax.text(
            50,
            -23,
            f"required arithmetic intensity   {cores} FMAs / {fp32} loads  =  4 FMAs per load",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=TEAL_D,
        )
    else:
        ax.text(
            50,
            -7.5,
            f"{fp32} loads  →  {busy} FMAs",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
        ax.text(
            50,
            -15.5,
            f"{cores - busy} of the {cores} lanes sit idle  —  12.5% of peak",
            ha="center",
            va="center",
            fontsize=13,
            color=GREY,
        )
        ax.text(
            50,
            -23,
            f"filling the SM this way would need {cores} × 2 = {cores * 2} FP32 per cycle,\n"
            f"8× more than the pipe can deliver",
            ha="center",
            va="center",
            fontsize=13,
            color=INK,
            linespacing=1.35,
        )


fig = plt.figure(figsize=(17.4, 15.4), dpi=150)

fig.text(
    0.5,
    0.979,
    "Half the cores, half the pipe — the T4 and the H100 need the same arithmetic intensity",
    ha="center",
    va="center",
    fontsize=26,
    fontweight="bold",
)
fig.text(
    0.5,
    0.956,
    "how many values shared memory can hand over per cycle decides how often each one has to be reused",
    ha="center",
    va="center",
    fontsize=14.5,
    color=GREY,
)

ROWS = [0.535, 0.085]
BANNERS = [0.916, 0.466]
COLS = [0.035, 0.515]
PW, PH = 0.450, 0.355

for r, gpu in enumerate(GPUS):
    y, by = ROWS[r], BANNERS[r]
    fig.patches.append(
        Rectangle(
            (0.028, by - 0.016),
            0.944,
            0.0009,
            transform=fig.transFigure,
            facecolor="#e5e5e5",
            edgecolor="none",
        )
    )
    fig.text(
        0.035, by, gpu["header"], ha="left", va="center", fontsize=19, fontweight="bold", color=INK
    )
    fig.text(0.972, by, gpu["sub"], ha="right", va="center", fontsize=13, color=GREY)
    for c, reuse in enumerate([False, True]):
        ax = fig.add_axes([COLS[c], y, PW, PH])
        draw_panel(ax, gpu, reuse)

handles = [
    Rectangle((0, 0), 1, 1, facecolor=TEAL, edgecolor=TEAL_D, linewidth=0.9),
    Rectangle((0, 0), 1, 1, facecolor=IDLE, edgecolor=IDLE_E, linewidth=0.9),
    Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=BLUE, linewidth=0.9),
]
fig.legend(
    handles,
    ["lane issuing an FMA", "lane idle, waiting on data", "shared-memory bank delivering a value"],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.012),
    ncol=3,
    fontsize=13,
    handlelength=1.5,
    handleheight=1.0,
    columnspacing=3.2,
    labelcolor=INK_SOFT,
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
