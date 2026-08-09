"""Shared Nature-scale plotting grammar for the manuscript."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


MM_PER_INCH = 25.4

COLORS = {
    "rep1": "#7C102A",
    "rep15": "#2166AC",
    "decoder": "#685994",
    "model": "#2C6E5A",
    "peach": "#FDDBC7",
    "sage": "#CCEBC5",
    "charcoal": "#2D3748",
    "ash": "#999999",
    "linen": "#F1EFEA",
    "mist": "#E7E7E5",
    "lavender": "#E6E0F0",
    "terracotta": "#BD6B6B",
    "teal": "#457B7A",
    "white": "#FFFFFF",
}

POSTERIOR_CMAP = LinearSegmentedColormap.from_list(
    "posterior_modern_teal_plum",
    [
        "#1E5555",
        "#4F817F",
        "#B8D0CB",
        "#F5F1EB",
        "#DCC6D5",
        "#9B6888",
        "#59334F",
    ],
    N=256,
)

PATTERN_CMAP = LinearSegmentedColormap.from_list(
    "pattern_linen_deep_violet",
    ["#F5F2EC", "#D4CCE2", "#9B88BC", COLORS["decoder"], "#33284E"],
    N=256,
)


def mm(value: float) -> float:
    """Convert millimetres to inches."""

    return value / MM_PER_INCH


@contextmanager
def manuscript_style():
    """Apply the final-size visual system without mutating global defaults."""

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "font.weight": "normal",
        "text.color": COLORS["charcoal"],
        "axes.labelcolor": COLORS["charcoal"],
        "axes.edgecolor": COLORS["charcoal"],
        "axes.linewidth": 0.6,
        "axes.titlesize": 8.2,
        "axes.titleweight": "semibold",
        "axes.titlepad": 4.0,
        "axes.labelsize": 7.2,
        "axes.labelpad": 2.5,
        "xtick.color": COLORS["charcoal"],
        "ytick.color": COLORS["charcoal"],
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": 6.7,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 0.9,
        "lines.linewidth": 1.35,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.linewidth": 0.55,
        "figure.facecolor": COLORS["white"],
        "axes.facecolor": COLORS["white"],
        "savefig.facecolor": COLORS["white"],
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
    with mpl.rc_context(rc):
        yield


def clean_axis(ax: mpl.axes.Axes, *, left: bool = True, bottom: bool = True) -> None:
    """Keep only meaningful structure and place ticks outside the data."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(direction="out", pad=1.8)
    ax.grid(False)


def panel_label(
    ax: mpl.axes.Axes,
    label: str,
    *,
    x: float = -0.15,
    y: float = 1.12,
) -> None:
    """Place a stable Nature-style panel letter in axes coordinates."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["charcoal"],
        clip_on=False,
    )


def stimulus_window(
    ax: mpl.axes.Axes,
    start_ms: float = 0,
    stop_ms: float = 180,
    *,
    color: str | None = None,
    alpha: float = 0.55,
    zorder: float = -20,
) -> None:
    """Add the shared prespecified response window."""

    ax.axvspan(
        start_ms,
        stop_ms,
        facecolor=color or COLORS["peach"],
        alpha=alpha,
        edgecolor="none",
        zorder=zorder,
    )


def micro_key(
    ax: mpl.axes.Axes,
    items: list[tuple[str, str, str]],
    *,
    y: float = 1.035,
) -> None:
    """Compact in-panel key: (label, colour, line style)."""

    x = 0.02
    for label, color, linestyle in items:
        ax.plot(
            [x, x + 0.075],
            [y, y],
            transform=ax.transAxes,
            color=color,
            lw=1.45,
            ls=linestyle,
            clip_on=False,
        )
        ax.text(
            x + 0.088,
            y,
            label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=6.5,
            color=COLORS["charcoal"],
            clip_on=False,
        )
        x += 0.29


def export_figure(
    fig: mpl.figure.Figure,
    output_stem: Path,
    *,
    fixed_bounds: bool = False,
) -> dict[str, Path]:
    """Write the manuscript vector and high-resolution review formats."""

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": output_stem.with_suffix(".pdf"),
        "svg": output_stem.with_suffix(".svg"),
        "png": output_stem.with_suffix(".png"),
    }
    bounds = {} if fixed_bounds else {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(paths["pdf"], **bounds)
    fig.savefig(paths["svg"], **bounds)
    fig.savefig(paths["png"], dpi=600, **bounds)
    return paths
