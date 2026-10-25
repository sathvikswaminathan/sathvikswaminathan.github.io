"""Shared palette + rcParams for the tiling-hierarchy post figures.

Colours are sampled from the original H100 diagrams so the regenerated
T4 / H100 figures sit in the same visual system as the untouched ones
(inner-vs-outer-product, thread-tile-8x8, warp-shape, warps-per-smsp).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#fdfdfd"
INK = "#282728"
INK_SOFT = "#555555"
GREY = "#8a8a8a"
GREY_L = "#b0b0b0"

BLUE = "#2e7ebb"
BLUE_D = "#1f5f8f"
BLUE_F = "#cfe2f1"

TEAL = "#a8d5ca"
TEAL_M = "#5ca895"
TEAL_D = "#0b5a4d"
GREEN_F = "#dcefe9"

ORANGE = "#e8872a"
ORANGE_L = "#eca158"

RED = "#c0392b"
RED_M = "#cd6459"
RED_F = "#f2dad6"
RED_E = "#e3b0aa"

IDLE = "#f0f0f0"
IDLE_E = "#dcdcdb"

# Per-GPU accents: the T4 is the post's primary target, the H100 the foil.
T4_ACCENT = "#2e7ebb"
H100_ACCENT = "#7b5ea7"


def use_style():
    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 13,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.edgecolor": GREY_L,
            "axes.labelcolor": INK_SOFT,
            "text.color": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "legend.frameon": False,
        }
    )


def blank(ax, xlim=(0, 100), ylim=(0, 100)):
    """Turn an axes into a bare drawing surface."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_color(GREY_L)
