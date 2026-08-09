"""Build the standalone Figure-1 illustration introducing Layer 2.

The panel is deliberately schematic but mathematically exact.  It presents
the forward computation as one horizontal visual sentence and the local
learning rule as a compact return loop beneath the coincidence map and mask
bank.  No simulated values or empirical observations are implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from final_figures.style import COLORS, PATTERN_CMAP, export_figure, manuscript_style, mm


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_1_layer2_panel"
DATA_DIR = HERE / "data"

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 76.0

STRUCTURE = COLORS["charcoal"]
MUTED = "#7E8791"
HAIRLINE = "#D8DADC"
STAGE_COLORS = (
    COLORS["rep15"],
    COLORS["teal"],
    COLORS["terracotta"],
    COLORS["decoder"],
    COLORS["decoder"],
)


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = STRUCTURE,
    lw: float = 0.75,
    rad: float = 0.0,
    head: float = 2.0,
    zorder: float = 5,
    linestyle: str | tuple = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=f"-|>,head_length={head},head_width={head * 0.56}",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def _stage_header(ax) -> None:
    positions = (17.0, 51.0, 86.0, 122.0, 162.0)
    labels = (
        "Layer-1 rates",
        "multiscale traces",
        "directional map",
        "learned masks",
        "unit activity",
    )

    for left, right in zip(positions[:-1], positions[1:]):
        _arrow(
            ax,
            (left + 2.2, 60.5),
            (right - 2.4, 60.5),
            color=HAIRLINE,
            lw=0.75,
            head=1.8,
            zorder=1,
        )
    for index, (x, label, color) in enumerate(
        zip(positions, labels, STAGE_COLORS), start=1
    ):
        ax.add_patch(
            Circle((x, 60.5), 1.65, facecolor=color, edgecolor="none", zorder=3)
        )
        ax.text(
            x,
            60.5,
            str(index),
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=COLORS["white"],
            zorder=4,
        )
        ax.text(
            x,
            64.0,
            label,
            ha="center",
            va="bottom",
            fontsize=6.15,
            fontweight="semibold",
            color=STRUCTURE,
        )


def _rate_vector(ax) -> None:
    x0, x1 = 8.5, 26.0
    levels = np.array([0.18, 0.42, 1.00, 0.58, 0.10])
    ys = np.linspace(31.0, 49.0, levels.size)
    for index, (level, y) in enumerate(zip(levels, ys)):
        ax.plot([x0, x1], [y, y], color=COLORS["mist"], lw=0.55, zorder=0)
        color = COLORS["rep15"] if index == 2 else "#8FB3D7"
        ax.add_patch(
            Rectangle(
                (x0, y - 0.82),
                (x1 - x0) * level,
                1.64,
                facecolor=color,
                edgecolor="none",
                alpha=1.0 if index == 2 else 0.72,
                zorder=3,
            )
        )
    ax.plot([x0, x0], [29.7, 50.3], color=STRUCTURE, lw=0.55, zorder=2)
    ax.text(17.0, 25.3, "E(t)", ha="center", va="center", fontsize=6.1,
            fontweight="semibold", color=STAGE_COLORS[0])
    ax.text(17.0, 22.4, "read-only from layer 1", ha="center", va="center",
            fontsize=5.1, color=MUTED)


def _filterbank(ax) -> None:
    x = np.linspace(39.0, 65.0, 180)
    onset = 40.0
    time = np.maximum(x - onset, 0.0)
    taus = np.array([2.6, 4.3, 7.0, 11.2, 17.5])
    colors = ("#B8ACD1", "#A090C2", "#8771B0", "#685994", "#4C3B70")
    ax.plot([38.5, 65.5], [30.0, 30.0], color=HAIRLINE, lw=0.5, zorder=0)
    ax.plot([onset, onset], [30.0, 50.5], color=COLORS["teal"], lw=0.65,
            ls=(0, (2, 2)), zorder=1)
    for tau, color in zip(taus, colors):
        values = 30.0 + 18.2 * np.exp(-time / tau)
        values[x < onset] = 30.0
        ax.plot(x, values, color=color, lw=1.05, zorder=3)
    ax.text(38.7, 50.7, "token", ha="left", va="bottom", fontsize=5.0,
            color=COLORS["teal"], fontweight="semibold")
    ax.text(51.5, 25.3, r"$\tau_r\,\mathrm{d}s_r/\mathrm{d}t=-s_r+E$", ha="center", va="center",
            fontsize=5.6, color=STAGE_COLORS[1])
    ax.text(51.5, 22.4, r"$\tau=30\ldots500\ \mathrm{ms}$", ha="center", va="center",
            fontsize=5.1, color=MUTED)


def _matrix_cell(ax, x: float, y: float, w: float, h: float, value: float,
                 *, alpha: float = 1.0, edge: str = COLORS["white"],
                 lw: float = 0.34) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=PATTERN_CMAP(float(np.clip(value, 0.0, 1.0))),
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            zorder=3,
        )
    )


def _directional_map(ax) -> None:
    x0, y0, width, height = 72.0, 30.0, 28.0, 20.0
    rows, cols = 6, 12
    cw, ch = width / cols, height / rows
    current = 3
    values = np.zeros((rows, cols), dtype=float)
    values[current] = np.array(
        [0.08, 0.30, 0.76, 0.00, 0.56, 0.22, 0.16, 0.48, 0.92, 0.00, 0.66, 0.32]
    )
    for row in range(rows):
        for col in range(cols):
            value = values[row, col] if row == current else 0.015
            _matrix_cell(
                ax,
                x0 + col * cw,
                y0 + (rows - 1 - row) * ch,
                cw,
                ch,
                value,
                alpha=1.0,
            )
    ax.add_patch(
        Rectangle(
            (x0, y0 + (rows - 1 - current) * ch),
            width,
            ch,
            facecolor="none",
            edgecolor=COLORS["rep15"],
            linewidth=0.85,
            zorder=5,
        )
    )
    ax.plot([x0 + width / 2] * 2, [y0, y0 + height], color=COLORS["white"],
            lw=0.75, zorder=5)
    ax.text(x0 - 1.1, y0 + (rows - current - 0.5) * ch, "now", ha="right",
            va="center", fontsize=5.0, color=COLORS["rep15"],
            fontweight="semibold")
    ax.text(86.0, 25.3, r"$D=E\,s_{\mathrm{flat}}^{\mathsf{T}}$", ha="center", va="center",
            fontsize=5.8, color=STAGE_COLORS[2], fontweight="semibold")
    ax.text(86.0, 22.4, "same-channel pairs removed", ha="center", va="center",
            fontsize=5.1, color=MUTED)


def _mask_card(ax, x0: float, y0: float, *, alpha: float, front: bool) -> None:
    rows, cols = 5, 8
    width, height = 23.0, 17.0
    cw, ch = width / cols, height / rows
    values = np.full((rows, cols), 0.03)
    if front:
        values[4, 2] = 0.90
        values[3, 4] = 0.68
        values[2, 6] = 0.48
        values[4, 1] = 0.24
        values[3, 3] = 0.18
    else:
        values[1, 1] = 0.36
        values[3, 5] = 0.26
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 0.7, y0 - 0.7),
            width + 1.4,
            height + 1.4,
            boxstyle="round,pad=0.15,rounding_size=0.8",
            facecolor=COLORS["white"],
            edgecolor=STAGE_COLORS[3] if front else HAIRLINE,
            linewidth=0.75 if front else 0.5,
            alpha=alpha,
            zorder=1,
        )
    )
    for row in range(rows):
        for col in range(cols):
            _matrix_cell(
                ax,
                x0 + col * cw,
                y0 + (rows - 1 - row) * ch,
                cw,
                ch,
                values[row, col],
                alpha=alpha,
            )


def _mask_bank(ax) -> None:
    _mask_card(ax, 109.2, 28.8, alpha=0.48, front=False)
    _mask_card(ax, 111.5, 31.0, alpha=0.68, front=False)
    _mask_card(ax, 114.0, 33.0, alpha=1.00, front=True)
    ax.text(125.5, 49.0, r"$M_k$", ha="center", va="center", fontsize=6.0,
            color=STAGE_COLORS[3], fontweight="semibold", zorder=7)
    ax.text(122.0, 25.3, "K non-negative templates", ha="center", va="center",
            fontsize=5.5, color=STAGE_COLORS[3])
    ax.text(122.0, 22.4, "one mask per unit", ha="center", va="center",
            fontsize=5.1, color=MUTED)


def _unit_activity(ax) -> None:
    x_circle, x_bar = 154.0, 158.0
    activities = np.array([0.10, 0.24, 1.00, 0.38, 0.14])
    ys = np.linspace(31.0, 49.0, activities.size)
    for index, (activity, y) in enumerate(zip(activities, ys)):
        winner = index == 2
        ax.add_patch(
            Circle(
                (x_circle, y),
                1.45,
                facecolor=STAGE_COLORS[4] if winner else COLORS["white"],
                edgecolor=STAGE_COLORS[4],
                linewidth=0.9,
                alpha=1.0 if winner else 0.55,
                zorder=4,
            )
        )
        ax.plot(
            [x_bar, x_bar + 12.0 * activity],
            [y, y],
            color=STAGE_COLORS[4],
            lw=2.1 if winner else 1.15,
            alpha=1.0 if winner else 0.42,
            zorder=3,
        )
    ax.text(162.0, 25.3, r"$y_k=\mathrm{ReLU}\!\left(\langle M_k,D\rangle\right)$", ha="center", va="center",
            fontsize=5.8, color=STAGE_COLORS[4], fontweight="semibold")
    ax.text(162.0, 22.4, "all unit responses", ha="center", va="center",
            fontsize=5.1, color=MUTED)


def _forward_arrows(ax) -> None:
    pairs = (
        ((28.5, 40.0), (36.5, 40.0), None),
        ((67.0, 40.0), (70.0, 40.0), None),
        ((102.0, 40.0), (107.0, 40.0), "match"),
        ((139.5, 40.0), (150.5, 40.0), "overlap"),
    )
    for start, end, label in pairs:
        _arrow(ax, start, end, color=STRUCTURE, lw=0.75, head=1.9, zorder=6)
        if label:
            ax.text(
                0.5 * (start[0] + end[0]),
                start[1] + 1.7,
                label,
                ha="center",
                va="bottom",
                fontsize=5.0,
                color=MUTED,
            )


def _learning_loop(ax) -> None:
    x0, y0, width, height = 44.0, 3.0, 105.0, 14.0
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.35,rounding_size=1.4",
            facecolor="#FAF5F3",
            edgecolor="#E9D8D3",
            linewidth=0.55,
            zorder=0,
        )
    )
    ax.text(
        x0 + 2.0,
        y0 + height - 1.6,
        "LOCAL LEARNING",
        ha="left",
        va="top",
        fontsize=5.2,
        color=COLORS["terracotta"],
        fontweight="bold",
    )

    centers = (56.5, 80.0, 106.0, 135.0)
    labels = ("gate", "normalize", "compete", "adapt + forget")
    equations = (
        r"$\Vert D\Vert>g\,p_{\mathrm{peak}}$",
        r"$\widehat{D}=D/\Vert D\Vert$",
        r"$k^*=\arg\max_k\,\cos(M_k,\widehat{D})$",
        r"$M_{k^*}\leftarrow M_{k^*}+\eta(\widehat{D}-M_{k^*})$" + "\n"
        + r"$M_k\leftarrow(1-\lambda)M_k\quad\forall k$",
    )
    for index, (x, label, equation) in enumerate(
        zip(centers, labels, equations), start=1
    ):
        ax.add_patch(
            Circle(
                (x - 6.0, y0 + 7.0),
                1.18,
                facecolor=COLORS["white"],
                edgecolor=COLORS["terracotta"],
                linewidth=0.7,
                zorder=3,
            )
        )
        ax.text(x - 6.0, y0 + 7.0, str(index), ha="center", va="center",
                fontsize=5.0, color=COLORS["terracotta"], fontweight="bold")
        ax.text(x - 3.9, y0 + 8.8, label, ha="left", va="center",
                fontsize=5.15, color=STRUCTURE, fontweight="semibold")
        ax.text(x - 3.9, y0 + 5.4, equation, ha="left", va="center",
                fontsize=5.0, color=STRUCTURE, linespacing=1.28)
    for left, right in zip(centers[:-1], centers[1:]):
        _arrow(
            ax,
            (left + 6.4, y0 + 7.0),
            (right - 8.2, y0 + 7.0),
            color="#D7B8B0",
            lw=0.65,
            head=1.55,
            zorder=2,
        )

    _arrow(
        ax,
        (86.0, 29.0),
        (86.0, y0 + height + 0.7),
        color=COLORS["terracotta"],
        lw=0.75,
        head=1.8,
        zorder=2,
    )
    _arrow(
        ax,
        (137.0, y0 + height + 0.7),
        (128.5, 29.7),
        color=COLORS["terracotta"],
        lw=0.75,
        rad=-0.20,
        head=1.8,
        zorder=2,
    )


def build_figure() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with manuscript_style():
        fig = plt.figure(
            figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)),
            facecolor=COLORS["white"],
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, FIGURE_WIDTH_MM)
        ax.set_ylim(0, FIGURE_HEIGHT_MM)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ax.text(7.0, 71.8, "Layer 2", ha="left", va="center", fontsize=8.8,
                fontweight="bold", color=COLORS["decoder"])
        ax.plot([24.5, 24.5], [68.7, 74.2], color=HAIRLINE, lw=0.65)
        ax.text(
            28.0,
            71.8,
            "Multiscale templates turn temporal context into selective units",
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="semibold",
            color=STRUCTURE,
        )

        _stage_header(ax)
        _rate_vector(ax)
        _filterbank(ax)
        _directional_map(ax)
        _mask_bank(ax)
        _unit_activity(ax)
        _forward_arrows(ax)
        _learning_loop(ax)

        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    source_paths = (
        Path(__file__).resolve(),
        HERE.parents[1] / "layer2_multirate" / "layer2.py",
        HERE.parents[1] / "layer2_multirate" / "config.py",
    )
    provenance = {
        "asset": "Figure 1 standalone Layer-2 introduction panel",
        "status": "schematic; equations reproduce Layer2MR exactly",
        "final_size_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "algorithm": {
            "input": "E only; Layer 1 is read-only",
            "trace": "s += dt * (-s + E[:, None]) / tau[None, :]",
            "coincidence": "D = outer(E, s_flat); same-channel pairs zero",
            "readout": "y = relu(M_flat @ D_flat)",
            "learning": (
                "drive gate; normalize D; cosine best-match competition; "
                "winner instar update clipped non-negative; all-mask decay"
            ),
        },
        "sources": {
            str(path.relative_to(HERE.parents[1])): _sha256(path)
            for path in source_paths
        },
        "outputs": {
            kind: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for kind, path in paths.items()
        },
    }
    provenance_path = DATA_DIR / "figure_1_layer2_panel_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    paths["provenance"] = provenance_path
    return paths


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    _parse_args(arguments)
    paths = build_figure()
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
