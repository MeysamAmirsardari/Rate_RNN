"""Build the submission-ready empirical/model roving Figure 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation

from final_figures.figure_2.channel_erp import build_channel_erp
from final_figures.figure_2.ecog_data import build_ecog_cache
from final_figures.figure_2.inference import load_or_build_inference
from final_figures.figure_2.model_data import load_or_build_model_data
from final_figures.figure_2.posterior_buildup import (
    load_or_build_posterior_buildup,
)
from final_figures.style import (
    COLORS,
    PATTERN_CMAP,
    POSTERIOR_CMAP,
    clean_axis,
    export_figure,
    manuscript_style,
    mm,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_2_roving"

POSITIONS = (1, 2, 3)
CONDITION_ORDER = (
    "intact",
    "no_depression",
    "plasticity_frozen",
    "uniform_inhibition",
)
CONDITION_LABELS = {
    "intact": "Intact",
    "no_depression": "No depression",
    "plasticity_frozen": "Plasticity frozen",
    "uniform_inhibition": "Uniform inhibition",
}
CONDITION_SHORT = {
    "intact": "Intact",
    "no_depression": "No\ndepression",
    "plasticity_frozen": "Plasticity\nfrozen",
    "uniform_inhibition": "Uniform\ninhibition",
}
CONDITION_COLORS = {
    "intact": COLORS["model"],
    "no_depression": "#7E8792",
    "plasticity_frozen": COLORS["rep15"],
    "uniform_inhibition": COLORS["terracotta"],
}
POSITION_COLORS = {
    1: COLORS["decoder"],
    2: COLORS["teal"],
    3: COLORS["terracotta"],
}

def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_index(model: dict[str, np.ndarray], name: str) -> int:
    conditions = [str(value) for value in model["conditions"].tolist()]
    try:
        return conditions.index(name)
    except ValueError as exc:
        raise KeyError(f"Condition {name!r} is absent; found {conditions}") from exc


def _tone_labels(position: int) -> tuple[str, str, str]:
    if position == 1:
        return ("X", "B", "C")
    if position == 2:
        return ("A", "X", "C")
    if position == 3:
        return ("A", "B", "X")
    raise ValueError(f"Unknown variable-tone position: {position}")


def _sequence_context(
    ax,
    position: int,
    *,
    show_labels: bool = True,
    label_y: float = 0.09,
) -> None:
    """Show the three 180-ms tones on the shared absolute sequence clock."""

    labels = _tone_labels(position)
    for tone_index, start in enumerate((0, 180, 360)):
        stop = start + 180
        deviant = tone_index + 1 == position
        ax.axvspan(
            start,
            stop,
            color=COLORS["lavender"] if deviant else COLORS["mist"],
            alpha=0.72 if deviant else 0.58,
            linewidth=0,
            zorder=-20,
        )
        if show_labels:
            ax.text(
                (start + stop) / 2,
                label_y,
                labels[tone_index],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=5.7,
                fontweight="semibold" if deviant else "normal",
                color=COLORS["decoder"] if deviant else "#7A7F87",
                clip_on=True,
                zorder=7,
            )
    for boundary in (180, 360):
        ax.axvline(boundary, color=COLORS["ash"], lw=0.48, alpha=0.75, zorder=0)
    ax.axvline(
        (position - 1) * 180,
        color=COLORS["charcoal"],
        lw=0.58,
        ls=(0, (2.0, 2.0)),
        alpha=0.68,
        zorder=1,
    )


def _position_tag(ax, position: int) -> None:
    """Quiet in-panel position label that remains legible over traces."""

    ax.text(
        0.014,
        0.84,
        f"POSITION {position}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.3,
        fontweight="bold",
        color=COLORS["charcoal"],
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.14",
            "facecolor": COLORS["white"],
            "edgecolor": "none",
            "alpha": 0.82,
        },
        clip_on=True,
        zorder=12,
    )


#: Panel B reserves a strip above the traces for the significance rails, so a
#: rail never overlaps the evoked responses or the position label.
HEADROOM = 1.15
RAIL_HEIGHT = 1.08


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).ravel()
    padded = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(stop - 1)) for start, stop in edges.reshape(-1, 2)]


def _significance_rail(
    ax,
    time: np.ndarray,
    mask: np.ndarray,
    y: float,
    *,
    color: str,
    linewidth: float = 2.1,
) -> None:
    """Draw corrected significance as compact contiguous bars."""

    time = np.asarray(time, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()
    if time.size != mask.size:
        raise ValueError(
            f"Significance axis and mask differ: {time.size} != {mask.size}"
        )
    step = float(np.median(np.diff(time))) if time.size > 1 else 1.0
    for start, stop in _contiguous_segments(mask):
        left = time[start] - 0.25 * step
        right = time[stop] + 0.25 * step
        ax.plot(
            [left, right],
            [y, y],
            color=COLORS["white"],
            lw=linewidth + 1.5,
            solid_capstyle="butt",
            zorder=9,
            clip_on=True,
        )
        ax.plot(
            [left, right],
            [y, y],
            color=color,
            lw=linewidth,
            solid_capstyle="butt",
            zorder=10,
            clip_on=True,
        )


def _outlined_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linewidth: float,
    zorder: float = 4,
) -> None:
    """Editorial trace with a subtle light keyline over colored windows."""

    ax.plot(
        x,
        y,
        color=COLORS["white"],
        lw=linewidth + 1.15,
        alpha=0.82,
        zorder=zorder - 0.2,
    )
    ax.plot(
        x,
        y,
        color=color,
        lw=linewidth,
        zorder=zorder,
    )


def _panel_heading(
    ax,
    label: str,
    title: str,
    *,
    title_gap_pt: float = 14.0,
    label_gap_pt: float = 8.0,
) -> None:
    """Align a panel letter and title using one physical, axes-independent gap."""

    heading_transform = ax.transAxes + ScaledTranslation(
        0.0,
        title_gap_pt / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0,
        1,
        title,
        transform=heading_transform,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="semibold",
        color=COLORS["charcoal"],
        clip_on=False,
    )
    label_transform = heading_transform + ScaledTranslation(
        -label_gap_pt / 72.0,
        0.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text(
        0,
        1,
        label,
        transform=label_transform,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["charcoal"],
        clip_on=False,
    )


def _draw_panel_a(fig, spec) -> None:
    ax = fig.add_subplot(spec)
    ax.axis("off")
    _panel_heading(ax, "A", "Roving paradigm", title_gap_pt=-4.0)


def _draw_panel_b(
    fig,
    spec,
    channel_erp: dict[str, np.ndarray],
) -> list:
    heading_axis = fig.add_subplot(spec)
    heading_axis.axis("off")
    nested = spec.subgridspec(
        3,
        2,
        width_ratios=(0.18, 1.0),
        hspace=0.11,
        wspace=0.06,
    )
    label_axes = [fig.add_subplot(nested[index, 0]) for index in range(3)]
    axes = [fig.add_subplot(nested[index, 1]) for index in range(3)]
    envelopes = []
    for position in POSITIONS:
        for repetition in (1, 15):
            mean = np.asarray(
                channel_erp[f"pos{position}_rep{repetition}_mean"], dtype=float
            )
            sem = np.asarray(
                channel_erp[f"pos{position}_rep{repetition}_sem"], dtype=float
            )
            envelopes.extend((mean - sem, mean + sem))
    values = np.concatenate(envelopes)
    data_low = float(np.nanmin(values))
    data_high = float(np.nanmax(values))
    span = max(data_high - data_low, 1e-6)
    y_low = data_low - 0.10 * span
    y_high = data_high + 0.24 * span
    rail_y = data_high + 0.16 * span

    for index, (label_ax, ax, position) in enumerate(
        zip(label_axes, axes, POSITIONS)
    ):
        time = np.asarray(channel_erp["time_ms"], dtype=float)
        first = np.asarray(channel_erp[f"pos{position}_rep1_mean"], dtype=float)
        first_sem = np.asarray(
            channel_erp[f"pos{position}_rep1_sem"], dtype=float
        )
        last = np.asarray(channel_erp[f"pos{position}_rep15_mean"], dtype=float)
        last_sem = np.asarray(
            channel_erp[f"pos{position}_rep15_sem"], dtype=float
        )
        channel = int(channel_erp[f"pos{position}_channel_matlab"])
        ax.axvspan(
            0,
            180,
            color=COLORS["peach"],
            alpha=0.52,
            linewidth=0,
            zorder=-20,
        )
        ax.axvline(180, color=COLORS["ash"], lw=0.48, alpha=0.75, zorder=0)
        ax.text(
            90,
            0.06,
            "deviant tone",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=COLORS["ash"],
            clip_on=True,
        )
        ax.fill_between(
            time,
            first - first_sem,
            first + first_sem,
            color=COLORS["rep1"],
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        ax.fill_between(
            time,
            last - last_sem,
            last + last_sem,
            color=COLORS["rep15"],
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        ax.plot(time, first, color=COLORS["rep1"], lw=1.18, zorder=4)
        ax.plot(time, last, color=COLORS["rep15"], lw=1.18, zorder=4)

        # A quiet full-width track makes the inferential rail explicit. Only
        # corrected intervals receive a coloured overlay; an empty track is a
        # deliberate visual statement that no cluster survived correction.
        ax.plot(
            [time[0], time[-1]],
            [rail_y, rail_y],
            color="#C8CDD3",
            lw=0.68,
            solid_capstyle="butt",
            zorder=7,
            clip_on=True,
        )

        significant = np.asarray(
            channel_erp[f"pos{position}_significant"], dtype=bool
        )
        _significance_rail(
            ax,
            time,
            significant,
            rail_y,
            color=COLORS["teal"],
            linewidth=2.35,
        )

        ax.set_xlim(0, 360)
        ax.set_ylim(y_low, y_high)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax.set_xticks([0, 180, 360])
        label_ax.axis("off")
        label_ax.text(
            0.68,
            0.50,
            f"position {position}\ncontact {channel}",
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.7,
            fontweight="semibold",
            color=COLORS["charcoal"],
            linespacing=1.18,
        )
        if index < 2:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel("Time from variable-tone onset (ms)")
        clean_axis(ax, bottom=index == 2)

    heading_axis.text(
        -0.095,
        0.48,
        "ERP (a.u.)",
        transform=heading_axis.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.2,
        color=COLORS["charcoal"],
        clip_on=False,
    )
    _panel_heading(
        heading_axis,
        "B",
        "ECoG response",
    )
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=COLORS["rep1"], lw=1.35, label="Rep 1"),
            Line2D([0], [0], color=COLORS["rep15"], lw=1.35, label="Rep 15"),
            Line2D(
                [0],
                [0],
                color=COLORS["teal"],
                lw=1.9,
                solid_capstyle="butt",
                label="cluster-corrected P < 0.05",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.28),
        ncol=3,
        borderaxespad=0,
        handlelength=1.25,
        columnspacing=0.8,
        fontsize=5.9,
    )
    return axes


def _decoder_limits(inference: dict[str, np.ndarray]) -> tuple[float, float]:
    lows = np.concatenate(
        [inference[f"decoder_pos{p}_ci_low"] for p in POSITIONS]
    )
    highs = np.concatenate(
        [inference[f"decoder_pos{p}_ci_high"] for p in POSITIONS]
    )
    lower = max(0.30, np.floor(min(float(np.nanmin(lows)), 0.5) * 20) / 20)
    upper = min(0.90, np.ceil(float(np.nanmax(highs)) * 20) / 20)
    if upper - lower < 0.25:
        upper = min(0.90, lower + 0.25)
    return float(lower), float(upper)


def _draw_panel_c(
    fig,
    spec,
    inference: dict[str, np.ndarray],
) -> list:
    nested = spec.subgridspec(3, 1, hspace=0.11)
    axes = [fig.add_subplot(nested[index, 0]) for index in range(3)]
    time = np.asarray(inference["decoder_time_ms"], dtype=float)
    y_low, y_high = _decoder_limits(inference)
    rail_y = y_high - 0.045 * (y_high - y_low)

    for index, (ax, position) in enumerate(zip(axes, POSITIONS)):
        mean = inference[f"decoder_pos{position}_accuracy"]
        low = inference[f"decoder_pos{position}_ci_low"]
        high = inference[f"decoder_pos{position}_ci_high"]
        significant = inference[f"decoder_pos{position}_significant"]
        _sequence_context(ax, position, label_y=0.08)
        ax.fill_between(
            time,
            low,
            high,
            color=COLORS["decoder"],
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )
        _outlined_line(
            ax,
            time,
            mean,
            color=COLORS["decoder"],
            linewidth=1.30,
        )
        ax.axhline(
            0.5,
            color=COLORS["ash"],
            lw=0.55,
            ls=(0, (2.0, 2.0)),
            zorder=1,
        )
        _significance_rail(
            ax,
            time,
            significant,
            rail_y,
            color=COLORS["decoder"],
            linewidth=1.55,
        )
        ax.set_xlim(0, 600)
        ax.set_ylim(y_low, y_high)
        middle_tick = round((0.5 + y_high) / 2, 2)
        if index == 0:
            ax.set_yticks([middle_tick, y_high])
        elif index == 1:
            ax.set_yticks([0.5, middle_tick])
        else:
            ax.set_yticks([0.5, middle_tick])
        ax.set_xticks([0, 180, 360, 540, 600])
        _position_tag(ax, position)
        if index < 2:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel("Sequence time (ms)")
        clean_axis(ax, bottom=index == 2)

    axes[1].set_ylabel("Cross-validated balanced accuracy", labelpad=4.6)
    _panel_heading(
        axes[0],
        "C",
        "Rep 1 versus Rep 15 decoding",
    )
    return axes


def _posterior_limit(ecog: dict[str, np.ndarray]) -> float:
    maps = np.stack(
        [ecog[f"pos{position}_posterior_sequence"] for position in POSITIONS]
    )
    limit = float(np.nanpercentile(np.abs(maps - 0.5), 98))
    return float(np.clip(np.ceil(limit * 100) / 100, 0.10, 0.14))


def _draw_panel_d(
    fig,
    spec,
    ecog: dict[str, np.ndarray],
    inference: dict[str, np.ndarray],
) -> list:
    nested = spec.subgridspec(
        1,
        4,
        width_ratios=(1, 1, 1, 0.045),
        wspace=0.19,
    )
    axes = [fig.add_subplot(nested[0, index]) for index in range(3)]
    contour_time = np.asarray(inference["posterior_time_ms"], dtype=float)
    repetitions = np.asarray(ecog["repetitions"], dtype=float)
    limit = _posterior_limit(ecog)
    norm = TwoSlopeNorm(vmin=0.5 - limit, vcenter=0.5, vmax=0.5 + limit)
    image = None

    for index, (ax, position) in enumerate(zip(axes, POSITIONS)):
        posterior = ecog[f"pos{position}_posterior_sequence"]
        image = ax.imshow(
            posterior,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=(0, 600, 15.5, 0.5),
            cmap=POSTERIOR_CMAP,
            norm=norm,
            rasterized=True,
        )
        mask = np.asarray(
            inference[f"posterior_pos{position}_mask"], dtype=bool
        )
        if mask.shape == (contour_time.size, repetitions.size):
            mask = mask.T
        if mask.shape != (repetitions.size, contour_time.size):
            raise ValueError(
                f"Position {position} posterior mask has shape {mask.shape}; "
                f"expected {(repetitions.size, contour_time.size)}"
            )
        if np.any(mask) and np.any(~mask):
            ax.contour(
                contour_time,
                repetitions,
                mask.astype(float),
                levels=[0.5],
                colors=COLORS["white"],
                linewidths=1.45,
                zorder=5,
            )
            ax.contour(
                contour_time,
                repetitions,
                mask.astype(float),
                levels=[0.5],
                colors=COLORS["charcoal"],
                linewidths=0.55,
                zorder=6,
            )
        for boundary in (180, 360):
            ax.axvline(
                boundary,
                color=COLORS["white"],
                lw=0.45,
                alpha=0.82,
                zorder=4,
            )
        ax.set_xlim(0, 600)
        ax.set_ylim(15.5, 0.5)
        ax.set_xticks([0, 180, 360, 540, 600])
        ax.set_yticks([1, 5, 10, 15])
        if index == 1:
            ax.set_xlabel("Sequence time (ms)")
        ax.set_title(f"position {position}", loc="left", fontsize=7.0, pad=2.2)
        if index == 0:
            ax.set_ylabel("Repetition")
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        clean_axis(ax, left=index == 0)

    colorbar_axis = fig.add_subplot(nested[0, 3])
    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_ticks([0.5 - limit, 0.5, 0.5 + limit])
    colorbar.set_ticklabels(
        [f"{0.5 - limit:.2f}", "0.50", f"{0.5 + limit:.2f}"]
    )
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.set_title(
        "Rep-1\nposterior",
        fontsize=5.7,
        pad=2.0,
        color=COLORS["charcoal"],
    )
    colorbar.ax.tick_params(labelsize=5.7, length=1.5, width=0.45, pad=1.0)
    colorbar.outline.set_linewidth(0.45)

    _panel_heading(
        axes[0],
        "D",
        "Posterior geometry across trials",
        title_gap_pt=18.0,
    )
    return axes


def _draw_panel_e(fig, spec, ecog: dict[str, np.ndarray]) -> list:
    nested = spec.subgridspec(
        2,
        3,
        height_ratios=(1, 0.06),
        hspace=0.30,
        wspace=0.15,
    )
    axes = [fig.add_subplot(nested[0, index]) for index in range(3)]
    grid_map = np.asarray(ecog["grid_map"], dtype=int)
    image = None

    for index, (ax, position) in enumerate(zip(axes, POSITIONS)):
        pattern = ecog[f"pos{position}_topography"]
        grid = np.empty(grid_map.shape, dtype=float)
        for row in range(grid_map.shape[0]):
            for column in range(grid_map.shape[1]):
                grid[row, column] = pattern[grid_map[row, column] - 1]
        image = ax.imshow(
            grid,
            cmap=PATTERN_CMAP,
            norm=Normalize(0, 1),
            interpolation="nearest",
            aspect="equal",
            origin="upper",
            rasterized=True,
        )
        ax.set_title(f"position {position}", fontsize=6.5, pad=1.9)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color=COLORS["white"], linewidth=0.48)
        ax.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.axhline(3.5, color=COLORS["white"], lw=2.35, zorder=5)
        ax.axhline(3.5, color=COLORS["charcoal"], lw=0.70, zorder=6)
        marked_channel = int(ecog[f"pos{position}_erp_selected_channel"])
        if marked_channel != int(np.argmax(pattern)) + 1:
            raise AssertionError(
                f"position {position}: ERP contact is not the displayed map maximum"
            )
        marked_rows, marked_columns = np.where(grid_map == marked_channel)
        if marked_rows.size != 1:
            raise AssertionError(
                f"ERP channel {marked_channel} has {marked_rows.size} grid locations"
            )
        ax.scatter(
            float(marked_columns[0]),
            float(marked_rows[0]),
            s=13.5,
            facecolor=COLORS["white"],
            edgecolor=COLORS["charcoal"],
            linewidth=0.48,
            zorder=9,
            clip_on=True,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index == 0:
            ax.text(
                -0.12,
                0.76,
                "A1",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=5.7,
                fontweight="semibold",
                color=COLORS["charcoal"],
                clip_on=False,
            )
            ax.text(
                -0.12,
                0.24,
                "PEG",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=5.7,
                fontweight="semibold",
                color=COLORS["charcoal"],
                clip_on=False,
            )

    colorbar_axis = fig.add_subplot(nested[1, :])
    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["low", "high"])
    colorbar.set_label("Relative |activation pattern|", fontsize=5.9, labelpad=-1.4)
    colorbar.ax.tick_params(labelsize=5.6, length=1.4, width=0.45, pad=0.7)
    colorbar.outline.set_linewidth(0.45)
    _panel_heading(
        axes[0],
        "E",
        "Distributed ECoG pattern",
        title_gap_pt=18.0,
    )
    return axes


def _rnn_trace_limits(inference: dict[str, np.ndarray]) -> tuple[float, float]:
    high = 0.0
    for position in POSITIONS:
        mean = inference[f"rnn_trace_intact_pos{position}_mean"]
        sem = inference[f"rnn_trace_intact_pos{position}_sem"]
        high = max(high, float(np.nanmax(mean + sem)))
    # Deliberate headroom keeps the response crest away from titles and bounds.
    return 0.0, float(np.ceil((1.14 * high) / 2.5) * 2.5)


def _draw_panel_f(
    fig,
    spec,
    inference: dict[str, np.ndarray],
) -> list:
    nested = spec.subgridspec(3, 1, hspace=0.11)
    axes = [fig.add_subplot(nested[index, 0]) for index in range(3)]
    time = np.asarray(inference["rnn_trace_time_ms"], dtype=float)
    y_low, y_high = _rnn_trace_limits(inference)

    for index, (ax, position) in enumerate(zip(axes, POSITIONS)):
        mean = np.asarray(
            inference[f"rnn_trace_intact_pos{position}_mean"], dtype=float
        )
        sem = np.asarray(
            inference[f"rnn_trace_intact_pos{position}_sem"], dtype=float
        )
        if mean.shape[0] != 2 and mean.shape[1] == 2:
            mean = mean.T
            sem = sem.T
        _sequence_context(ax, position, label_y=0.08)
        for endpoint, color in enumerate((COLORS["rep1"], COLORS["rep15"])):
            ax.fill_between(
                time,
                mean[endpoint] - sem[endpoint],
                mean[endpoint] + sem[endpoint],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            _outlined_line(
                ax,
                time,
                mean[endpoint],
                color=color,
                linewidth=1.25,
            )
        ax.set_xlim(0, 600)
        ax.set_ylim(y_low, y_high)
        if index == 0:
            ax.set_yticks([y_high / 2, y_high])
        elif index == 1:
            ax.set_yticks([y_high / 2])
        else:
            ax.set_yticks([0, y_high / 2])
        ax.set_xticks([0, 180, 360, 540, 600])
        _position_tag(ax, position)
        if index < 2:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel("Sequence time (ms)")
        clean_axis(ax, bottom=index == 2)

    axes[1].set_ylabel("Population E rate (a.u.)", labelpad=4.5)
    _panel_heading(
        axes[0],
        "F",
        "Rate-RNN response",
        title_gap_pt=18.0,
    )
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=COLORS["rep1"], lw=1.35, label="Rep 1"),
            Line2D([0], [0], color=COLORS["rep15"], lw=1.35, label="Rep 15"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.30),
        ncol=2,
        borderaxespad=0,
        handlelength=1.25,
        columnspacing=0.8,
        fontsize=5.9,
    )
    return axes


def _significance_marked_errorbar(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    significant: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    """Use marker fill, rather than extra rails, for corrected significance."""

    ax.plot(x, mean, color=color, lw=1.08, zorder=3)
    ax.errorbar(
        x,
        mean,
        yerr=sem,
        fmt="none",
        ecolor=color,
        elinewidth=0.62,
        capsize=1.35,
        capthick=0.55,
        zorder=2.8,
    )
    facecolors = [
        color if is_significant else COLORS["white"]
        for is_significant in np.asarray(significant, dtype=bool)
    ]
    ax.scatter(
        x,
        mean,
        s=9.5,
        marker=marker,
        facecolors=facecolors,
        edgecolors=color,
        linewidths=0.65,
        label=label,
        zorder=4,
    )


def _draw_panel_g(
    fig,
    spec,
    inference: dict[str, np.ndarray],
    posterior_buildup: dict[str, np.ndarray],
) -> list:
    nested = spec.subgridspec(1, 2, wspace=0.32, width_ratios=(0.95, 1.05))
    ax_ecog = fig.add_subplot(nested[0, 0])
    ax_model = fig.add_subplot(nested[0, 1])
    repetitions = np.asarray(posterior_buildup["repetitions"], dtype=float)

    ecog_lows: list[float] = []
    ecog_highs: list[float] = []
    for position in POSITIONS:
        mean = posterior_buildup[f"posterior_buildup_pos{position}_mean"]
        sem = posterior_buildup[f"posterior_buildup_pos{position}_sem"]
        ecog_lows.append(float(np.nanmin(mean - sem)))
        ecog_highs.append(float(np.nanmax(mean + sem)))
    ecog_low = min(0.44, np.floor(min(ecog_lows) * 50) / 50 - 0.01)
    ecog_high = max(0.56, np.ceil(max(ecog_highs) * 50) / 50 + 0.01)

    for index, position in enumerate(POSITIONS):
        color = POSITION_COLORS[position]
        mean = posterior_buildup[f"posterior_buildup_pos{position}_mean"]
        sem = posterior_buildup[f"posterior_buildup_pos{position}_sem"]
        significant = posterior_buildup[
            f"posterior_buildup_pos{position}_significant"
        ]
        plot_repetitions = repetitions + (-0.07, 0.0, 0.07)[index]
        _significance_marked_errorbar(
            ax_ecog,
            plot_repetitions,
            mean,
            sem,
            significant,
            color=color,
            marker="o",
            label=f"Position {position}",
        )
    ax_ecog.axhline(
        0.5,
        color=COLORS["ash"],
        lw=0.55,
        ls=(0, (2, 2)),
        zorder=0,
    )
    ax_ecog.set_xlim(0.6, 15.4)
    ax_ecog.set_ylim(ecog_low, ecog_high)
    ax_ecog.set_xticks([1, 5, 10, 15])
    ax_ecog.set_xlabel("Repetition")
    ax_ecog.set_ylabel("Rep-1 posterior probability")
    ax_ecog.set_title("ECoG posterior", loc="left", fontsize=7.1, pad=2.5)
    ax_ecog.legend(
        loc="upper right",
        ncol=1,
        borderaxespad=0.2,
        fontsize=5.6,
        labelspacing=0.20,
        handlelength=1.05,
    )
    clean_axis(ax_ecog)

    model_lows: list[float] = []
    model_highs: list[float] = []
    for condition in CONDITION_ORDER:
        mean = inference[f"rnn_change_{condition}_mean"]
        sem = inference[f"rnn_change_{condition}_sem"]
        model_lows.append(float(np.nanmin(mean - sem)))
        model_highs.append(float(np.nanmax(mean + sem)))
    model_low = min(-34.0, np.floor(min(model_lows) / 2) * 2 - 1)
    model_high = max(6.0, np.ceil(max(model_highs) / 2) * 2 + 2)
    for index, condition in enumerate(CONDITION_ORDER):
        color = CONDITION_COLORS[condition]
        mean = inference[f"rnn_change_{condition}_mean"]
        sem = inference[f"rnn_change_{condition}_sem"]
        significant = inference[f"rnn_change_{condition}_significant"]
        marker = ("o", "s", "^", "D")[index]
        plot_repetitions = repetitions + (-0.12, -0.04, 0.04, 0.12)[
            index
        ]
        _significance_marked_errorbar(
            ax_model,
            plot_repetitions,
            mean,
            sem,
            significant,
            color=color,
            marker=marker,
            label=CONDITION_LABELS[condition],
        )
    ax_model.axhline(0, color=COLORS["ash"], lw=0.55, zorder=0)
    ax_model.set_xlim(0.6, 15.4)
    ax_model.set_ylim(model_low, model_high)
    ax_model.set_xticks([1, 5, 10, 15])
    ax_model.set_xlabel("Repetition")
    ax_model.set_title("Rate RNN response", loc="left", fontsize=7.1, pad=2.5)
    ax_model.legend(
        loc="lower left",
        ncol=1,
        borderaxespad=0.2,
        fontsize=5.45,
        labelspacing=0.18,
        handlelength=1.05,
    )
    clean_axis(ax_model)

    _panel_heading(
        ax_ecog,
        "G",
        "Repetition-dependent posterior and model response",
        title_gap_pt=18.0,
    )
    return [ax_ecog, ax_model]


def _draw_panel_h(
    fig,
    spec,
    inference: dict[str, np.ndarray],
) -> None:
    ax = fig.add_subplot(spec)
    values = np.asarray(inference["rnn_suppression_values"], dtype=float)
    means = np.asarray(inference["rnn_suppression_mean"], dtype=float)
    lows = np.asarray(inference["rnn_suppression_ci_low"], dtype=float)
    highs = np.asarray(inference["rnn_suppression_ci_high"], dtype=float)
    significant_vs_intact = np.asarray(
        inference["rnn_suppression_vs_intact_significant"], dtype=bool
    )
    rng = np.random.default_rng(314159)

    for index, condition in enumerate(CONDITION_ORDER):
        color = CONDITION_COLORS[condition]
        jitter = rng.uniform(-0.13, 0.13, size=values.shape[1])
        ax.scatter(
            np.full(values.shape[1], index) + jitter,
            values[index],
            s=8,
            color=color,
            alpha=0.28,
            linewidth=0,
            zorder=2,
        )
        ax.errorbar(
            index,
            means[index],
            yerr=np.array(
                [
                    [means[index] - lows[index]],
                    [highs[index] - means[index]],
                ]
            ),
            fmt="D",
            ms=4.0,
            color=color,
            markeredgecolor=COLORS["white"],
            markeredgewidth=0.55,
            elinewidth=1.25,
            capsize=2.1,
            capthick=0.75,
            zorder=5,
        )

    y_low = min(-0.04, float(np.nanmin(lows)) - 0.02)
    y_high = max(0.34, float(np.nanmax(highs)) + 0.04)
    for index, is_significant in enumerate(significant_vs_intact):
        if is_significant:
            ax.text(
                index,
                y_high - 0.018,
                "●",
                ha="center",
                va="top",
                fontsize=4.9,
                color=CONDITION_COLORS[CONDITION_ORDER[index]],
            )
    ax.axhline(0, color=COLORS["ash"], lw=0.60, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(-0.48, len(CONDITION_ORDER) - 0.52)
    ax.set_ylim(y_low, y_high)
    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels([CONDITION_SHORT[name] for name in CONDITION_ORDER])
    ax.set_ylabel("Suppression index")
    _panel_heading(
        ax,
        "H",
        "Causal perturbations",
        title_gap_pt=18.0,
    )
    clean_axis(ax)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "model_perturbation_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "n_order_seeds",
                "mean_suppression_index",
                "ci95_low",
                "ci95_high",
                "jointly_corrected_significant_vs_intact",
            ]
        )
        for index, condition in enumerate(CONDITION_ORDER):
            writer.writerow(
                [
                    condition,
                    values.shape[1],
                    means[index],
                    lows[index],
                    highs[index],
                    bool(significant_vs_intact[index]),
                ]
            )


def build_figure(
    *,
    force_ecog: bool = False,
    force_model: bool = False,
    force_inference: bool = False,
) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ecog = build_ecog_cache(force=force_ecog, data_dir=DATA_DIR)
    model = load_or_build_model_data(force=force_model, data_dir=DATA_DIR)
    inference = load_or_build_inference(
        ecog,
        model,
        force=force_inference,
        data_dir=DATA_DIR,
    )
    posterior_buildup = load_or_build_posterior_buildup(
        force=force_inference,
        data_dir=DATA_DIR,
    )
    channel_erp = build_channel_erp(
        ecog,
        force=force_inference,
        data_dir=DATA_DIR,
    )

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(245)))
        grid = fig.add_gridspec(
            5,
            12,
            left=0.057,
            right=0.982,
            bottom=0.043,
            top=0.978,
            wspace=0.78,
            hspace=0.56,
            height_ratios=(0.40, 1.35, 1.05, 1.35, 1.18),
        )
        _draw_panel_a(fig, grid[0, :])
        _draw_panel_b(fig, grid[1, 0:5], channel_erp)
        _draw_panel_c(fig, grid[1, 6:12], inference)
        _draw_panel_d(fig, grid[2, :], ecog, inference)
        _draw_panel_e(fig, grid[3, 0:5], ecog)
        _draw_panel_f(fig, grid[3, 6:12], inference)
        _draw_panel_g(
            fig,
            grid[4, 0:8],
            inference,
            posterior_buildup,
        )
        _draw_panel_h(fig, grid[4, 8:12], inference)
        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    provenance = {
        "figure": "Figure 2 — Roving novelty responses",
        "final_size_mm": [183, 245],
        "panel_a": "entire first row reserved for the paradigm illustration",
        "panel_sources": {
            "B": (
                "true Rep-1/Rep-15 deviant-aligned channel ERP at the "
                "independently selected Panel-E maximum, with held-out-block "
                "SEM and joint cluster-mass inference"
            ),
            "C": (
                "leakage-safe grouped endpoint decoding with paired block "
                "uncertainty and full-refit cluster permutation inference"
            ),
            "D": (
                "three leakage-safe out-of-fold posterior maps with jointly "
                "corrected cross-fitted block-level contours"
            ),
            "E": (
                "discovery-half prespecified-window Haufe activation patterns "
                "on the verified A1/PEG grid; white dots mark the map maxima "
                "used for held-out ERP estimation"
            ),
            "F": (
                "intact rate-RNN sequence-clock responses with paired "
                "order-seed SEM"
            ),
            "G": (
                "blockwise ECoG posterior buildup and model percent response "
                "change with SEM and jointly corrected within-source inference"
            ),
            "H": (
                "position-averaged paired-seed suppression indices with "
                "95% intervals and jointly corrected exact paired contrasts "
                "against intact"
            ),
        },
        "inference_scope": (
            "ECoG tests are within-animal, block-level tests for Zaatar and "
            "do not support animal-population inference. Model tests use the "
            "eight paired order seeds after averaging designed positions."
        ),
        "panel_e_erp_contact_markers_matlab": {
            str(position): int(ecog[f"pos{position}_erp_selected_channel"])
            for position in POSITIONS
        },
        "inputs": {},
        "outputs": {},
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }
    for name in (
        "ecog_figure2_data.npz",
        "ecog_provenance.json",
        "model_figure2_data.npz",
        "model_provenance.json",
        "figure_2_inference.npz",
        "figure_2_inference_provenance.json",
        "posterior_buildup_inference.npz",
        "posterior_buildup_provenance.json",
        "channel_erp_inference.npz",
        "channel_erp_provenance.json",
    ):
        path = DATA_DIR / name
        if path.exists():
            provenance["inputs"][name] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
    for name, path in paths.items():
        provenance["outputs"][name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    provenance_path = DATA_DIR / "figure_2_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    paths["provenance"] = provenance_path
    return paths


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-ecog",
        action="store_true",
        help="Repeat endpoint extraction from the original .mat files.",
    )
    parser.add_argument(
        "--force-model",
        action="store_true",
        help="Repeat all simulated order-seed sessions.",
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Repeat the corrected permutation and bootstrap analyses.",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = _parse_args(arguments)
    paths = build_figure(
        force_ecog=args.force_ecog,
        force_model=args.force_model,
        force_inference=args.force_inference,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
