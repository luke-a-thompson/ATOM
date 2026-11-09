"""Shared figure style for the ATOM paper.

One look across every figure: Times-compatible text at print size, recessive
grid and axes, thin marks, a fixed colour per model (never per rank), and a
marker per model so identity never relies on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Column widths of the ICLR template, in inches.
TEXT_WIDTH = 5.5
HALF_WIDTH = 2.65

# Ink
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# Categorical slots, fixed order (validated: CVD dE >= 9.2 all-pairs, light surface).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

# Colour and marker follow the model, whatever else is on the plot.
MODEL_STYLE: dict[str, tuple[str, str]] = {
    "ATOM": (BLUE, "o"),
    "EGNO": (ORANGE, "s"),
    "EGNN": (AQUA, "^"),
    "EGNN-R": (AQUA, "^"),
    "ATOM NoPE": (AQUA, "D"),
}


def model_style(label: str) -> tuple[str, str]:
    return MODEL_STYLE.get(label, (INK_MUTED, "o"))


def apply_style(font_size: float = 9.0) -> None:
    """Set rcParams for print-size figures that sit inside the paper's column."""
    family = "Times New Roman"
    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
    except FileNotFoundError:
        family = "STIXGeneral"
    mpl.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def direct_label(ax: Axes, x: float, y: float, text: str, color: str, dx: float = 4, dy: float = 0) -> None:
    """Label a series at its last point, in ink, next to a coloured mark."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha="left", color=INK, fontsize=mpl.rcParams["legend.fontsize"])


def save(fig: plt.Figure, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")
