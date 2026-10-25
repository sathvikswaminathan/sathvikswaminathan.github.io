"""one-cta-vs-two.png

The schematic itself is the same on both GPUs — 128×128 and 256×128 both put 16
warps on an SM, the difference is whether they share a barrier. What differs
between the T4 and the H100 is how much slack the DRAM has, which is the only
reason to give up the second CTA. That comparison is the bottom panel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import (
    BG,
    GREY,
    GREY_L,
    H100_ACCENT,
    IDLE,
    IDLE_E,
    INK,
    INK_SOFT,
    ORANGE,
    RED,
    RED_F,
    TEAL,
    TEAL_D,
    despine,
    use_style,
)

use_style()

OUT = Path(__file__).resolve().parents[2] / "src/content/posts/one-cta-vs-two.png"

KIND = {
    "ffma": dict(fc=TEAL, ec=TEAL_D, tc=INK, fw="bold"),
    "sync": dict(fc=RED_F, ec=RED, tc=RED, fw="normal"),
    "idle": dict(fc="#e4e4e3", ec=IDLE_E, tc=GREY, fw="normal"),
    "fill": dict(fc=TEAL, ec=TEAL_D, tc=TEAL_D, fw="bold"),
}


def bar(ax, y, h, segs, x0=0.0):
    x = x0
    for w, kind, label in segs:
        k = KIND[kind]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=k["fc"], edgecolor=k["ec"], linewidth=1.3))
        if label:
            rot = 90 if w < 3.2 else 0
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=11 if rot else 13,
                color=k["tc"],
                fontweight=k["fw"],
                rotation=rot,
            )
        x += w
    return x


fig = plt.figure(figsize=(17.6, 14.4), dpi=150)

fig.text(
    0.5,
    0.977,
    "Both configurations put 16 warps on the SM — only one of them keeps issuing during a barrier",
    ha="center",
    va="center",
    fontsize=25,
    fontweight="bold",
)
fig.text(
    0.5,
    0.952,
    "schematic timeline of a single SM; the shaded spans are cycles where the warp scheduler has an FFMA to issue",
    ha="center",
    va="center",
    fontsize=14,
    color=GREY,
)

LAB_X = 21.0  # left edge of the timeline region, in axis units out of 100

# ============================================================ two small CTAs
fig.text(0.035, 0.912, "two 128 × 128 CTAs on the SM", fontsize=19, fontweight="bold", ha="left")
fig.text(
    0.035,
    0.890,
    "8 warps each  ·  32 KB of registers each  ·  barriers are independent",
    fontsize=13,
    color=GREY,
    ha="left",
)

ax = fig.add_axes([0.035, 0.635, 0.93, 0.238])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

rows = [
    (
        74,
        "CTA 0",
        "warps 0–7",
        [(10, "idle", "load"), (17, "ffma", "FFMA"), (3, "sync", "sync"),
         (16, "ffma", "FFMA"), (3, "sync", "sync"), (16, "ffma", "FFMA"),
         (3, "sync", "sync"), (11, "ffma", "FFMA")],
    ),
    (
        46,
        "CTA 1",
        "warps 8–15",
        [(16, "idle", "load"), (16, "ffma", "FFMA"), (3, "sync", "sync"),
         (16, "ffma", "FFMA"), (3, "sync", "sync"), (15, "ffma", "FFMA"),
         (3, "sync", "sync"), (7, "ffma", "FFMA")],
    ),
]
for y, name, sub, segs in rows:
    ax.text(LAB_X - 2.5, y + 9, name, ha="right", va="center", fontsize=15, fontweight="bold")
    ax.text(LAB_X - 2.5, y - 1, sub, ha="right", va="center", fontsize=11.5, color=GREY)
    bar(ax, y, 18, [(w * 0.79, k, l) for w, k, l in segs], x0=LAB_X)

ax.text(LAB_X - 2.5, 23, "SM issue slots", ha="right", va="center", fontsize=15, fontweight="bold")
bar(
    ax,
    14,
    18,
    [(10 * 0.79, "idle", "idle"), (69 * 0.79, "fill", "the SM always has a warp with an FFMA ready")],
    x0=LAB_X,
)
ax.annotate("", xy=(100, 8), xytext=(LAB_X, 8), arrowprops=dict(arrowstyle="-", color=GREY_L, lw=1.3))
ax.text(100, 3, "time", ha="right", va="center", fontsize=12, color=GREY, style="italic")
ax.text(
    99,
    40,
    "only the\nprologue is lost",
    ha="right",
    va="center",
    fontsize=13,
    fontweight="bold",
    color=TEAL_D,
    linespacing=1.4,
)

handles = [
    Rectangle((0, 0), 1, 1, facecolor=TEAL, edgecolor=TEAL_D, linewidth=1.2),
    Rectangle((0, 0), 1, 1, facecolor=RED_F, edgecolor=RED, linewidth=1.2),
    Rectangle((0, 0), 1, 1, facecolor="#e4e4e3", edgecolor=IDLE_E, linewidth=1.2),
]
fig.legend(
    handles,
    ["issuing FFMAs", "waiting at __syncthreads", "prologue / epilogue, no compute to overlap"],
    loc="upper center",
    bbox_to_anchor=(0.52, 0.632),
    ncol=3,
    fontsize=13,
    handlelength=1.6,
    handleheight=1.1,
    columnspacing=5.0,
    labelcolor=INK_SOFT,
)

# ============================================================== one big CTA
fig.text(0.035, 0.566, "one 256 × 128 CTA on the SM", fontsize=19, fontweight="bold", ha="left")
fig.text(
    0.035,
    0.544,
    "16 warps  ·  64 KB of registers  ·  every warp waits at the same barrier",
    fontsize=13,
    color=GREY,
    ha="left",
)

ax = fig.add_axes([0.035, 0.350, 0.93, 0.172])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

segs_big = [
    (11, "idle", "load"), (17, "ffma", "FFMA"), (4, "sync", "sync"),
    (17, "ffma", "FFMA"), (4, "sync", "sync"), (17, "ffma", "FFMA"),
    (4, "sync", "sync"), (6, "ffma", "FFMA"), (9, "idle", "store C"),
]
ax.text(LAB_X - 2.5, 72, "CTA 0", ha="right", va="center", fontsize=15, fontweight="bold")
ax.text(LAB_X - 2.5, 62, "warps 0–15", ha="right", va="center", fontsize=11.5, color=GREY)
bar(ax, 58, 26, [(w * 0.79, k, l) for w, k, l in segs_big], x0=LAB_X)

ax.text(LAB_X - 2.5, 22, "SM issue slots", ha="right", va="center", fontsize=15, fontweight="bold")
segs_sm = [
    (11, "idle", "idle"), (17, "fill", ""), (4, "idle", "idle"),
    (17, "fill", ""), (4, "idle", "idle"), (17, "fill", ""),
    (4, "idle", "idle"), (6, "fill", ""), (9, "idle", "idle"),
]
bar(ax, 10, 24, [(w * 0.79, k, l) for w, k, l in segs_sm], x0=LAB_X)

# drop lines from each barrier down to the idle slot it creates
acc = LAB_X
for w, k, _ in segs_big:
    if k == "sync":
        xc = acc + w * 0.79 / 2
        ax.plot([xc, xc], [56, 36], color=RED, ls=(0, (2, 3)), lw=1.4)
    acc += w * 0.79

ax.annotate("", xy=(100, 2), xytext=(LAB_X, 2), arrowprops=dict(arrowstyle="-", color=GREY_L, lw=1.3))
ax.text(100, -3, "time", ha="right", va="top", fontsize=12, color=GREY, style="italic")
ax.text(
    99,
    45,
    "every barrier\nstalls the whole SM",
    ha="right",
    va="center",
    fontsize=13,
    fontweight="bold",
    color=RED,
    linespacing=1.4,
)

# ====================================================== when is it worth it
fig.text(
    0.035,
    0.288,
    "So when is the bigger tile worth giving up the second CTA?",
    fontsize=19,
    fontweight="bold",
    ha="left",
)
fig.text(
    0.035,
    0.266,
    "only when the FLOPs-per-byte target no longer holds — and that depends on how much slack the DRAM has",
    fontsize=13,
    color=GREY,
    ha="left",
)

ax = fig.add_axes([0.055, 0.075, 0.36, 0.165])
despine(ax)

CONFIGS = ["128 × 128\n4.29 GB of DRAM traffic", "256 × 128\n3.22 GB of DRAM traffic"]
T4_PCT = [253 / 320 * 100, 190 / 320 * 100]
H100_PCT = [2.09 / 3.35 * 100, 1.57 / 3.35 * 100]
T4_LAB = ["254 GB/s", "191 GB/s"]
H100_LAB = ["2.1 TB/s", "1.6 TB/s"]

for i in range(2):
    ax.bar(i - 0.19, T4_PCT[i], width=0.34, color=ORANGE, zorder=2)
    ax.bar(i + 0.19, H100_PCT[i], width=0.34, color=H100_ACCENT, zorder=2)
    ax.text(i - 0.19, T4_PCT[i] + 2.5, T4_LAB[i], ha="center", va="bottom", fontsize=12, color=ORANGE, fontweight="bold")
    ax.text(i + 0.19, H100_PCT[i] + 2.5, H100_LAB[i], ha="center", va="bottom", fontsize=12, color=H100_ACCENT, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(CONFIGS, fontsize=12)
ax.set_ylim(0, 132)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=11.5)
ax.set_ylabel("DRAM bandwidth needed\n(% of the documented peak)", fontsize=12, linespacing=1.4)
ax.axhline(100, color=GREY_L, lw=1.4, ls=(0, (5, 4)))
ax.spines["left"].set_bounds(0, 100)
ax.legend(
    [Rectangle((0, 0), 1, 1, facecolor=ORANGE), Rectangle((0, 0), 1, 1, facecolor=H100_ACCENT)],
    ["T4  (320 GB/s GDDR6)", "H100  (3.35 TB/s HBM3)"],
    loc="upper right",
    fontsize=11.5,
    handlelength=1.3,
    handleheight=1.0,
    labelcolor=INK_SOFT,
    frameon=True,
    facecolor=BG,
    edgecolor="none",
)

fig.text(
    0.455,
    0.195,
    "At peak compute the 4096³ GEMM takes about 17 ms on the T4 and about 2 ms on the H100.",
    fontsize=13.5,
    color=INK,
    ha="left",
    va="top",
)
fig.text(
    0.455,
    0.163,
    "On the H100 there is plenty of margin at 128 × 128 unless other kernels are eating into\n"
    "its share of HBM, so the second CTA is the better deal and the bigger tile buys nothing.",
    fontsize=13,
    color=GREY,
    ha="left",
    va="top",
    linespacing=1.55,
)
fig.text(
    0.455,
    0.100,
    "On the T4, 128 × 128 asks for about 80% of the documented GDDR6 peak, which is close to\n"
    "what the memory actually sustains. That is the one case where 256 × 128 starts to pay —\n"
    "though clock throttling lowers the compute rate, and therefore the bandwidth needed, with it.",
    fontsize=13,
    color=ORANGE,
    ha="left",
    va="top",
    linespacing=1.55,
)

fig.text(
    0.5,
    0.022,
    "The bigger tile is the better deal on paper — 42.7 FLOPs per byte against 32.0, and 3.22 GB of DRAM traffic against 4.29 GB. "
    "It is the worse deal in practice unless the kernel is actually memory-bound.",
    ha="center",
    va="center",
    fontsize=13,
    color=GREY,
    style="italic",
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.30, facecolor=BG)
print("wrote", OUT)
