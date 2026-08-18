"""Build the submission-ready stochastic figure-ground Figure 4.

Every panel is model output.  The ferret physiology this figure is compared
against is Lu, Dutta, Mohammed, Elhilali & Shamma (2025) *iScience* 28:111991
and is cited rather than reproduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from final_figures.figure_4.inference import build_inference
from final_figures.figure_4.sfg_data import (
    BIN_MS,
    CLOUD_BINS,
    FIGURE_BINS,
    FIGURE_SIZES,
    N_BINS,
    N_REPS,
    N_SEEDS,
    REFERENCE_PRESET,
    difference,
    load_all,
    mean_sem,
    modulation,
)
from final_figures.style import (
    COLORS,
    PATTERN_CMAP,
    clean_axis,
    export_figure,
    manuscript_style,
    mm,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_4_sfg"

#: Figure and ground keep one meaning throughout: the bound set is the model
#: colour, the competing background is neutral slate.
COL_FIG = COLORS["model"]
COL_GND = "#7E8792"
COL_CROSS = COLORS["ash"]
COL_SIG = COLORS["decoder"]

#: Figure size is ordinal, so it gets a lightness ramp of one hue.
SIZE_RAMP = ("#A8C6BC", "#5E9A87", COLORS["model"], "#173F34")

#: The exemplar shown wherever a single figure size is drawn.
EXEMPLAR_SIZE = 10

TRACE_INDEX = {"E": 0, "tm": 1, "rec": 2, "inh": 3, "net": 4}


def _course(data, preset: str, n_fig: int, group: str) -> np.ndarray:
    """Per-seed excitatory-rate modulation, ``(n_seeds, N_BINS)`` in per cent."""

    return modulation(data, preset, n_fig,
                      f"course_{group}")[:, :, TRACE_INDEX["E"]]


def _window(course: np.ndarray, bins) -> np.ndarray:
    """Collapse a set of time bins per seed, ignoring bins with no pips.

    The figure is frozen within a session, so a 500 ms bin that contains no
    chord onset contains none in any presentation of that session and is NaN
    for that seed.  Seeds draw different chord onsets, so the across-seed mean
    is still defined everywhere.
    """

    return np.nanmean(course[:, bins], axis=1)


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Typography helpers.  Offsets are in points, so a heading sits the same
# distance above a 25 mm panel and a 35 mm one.
# ---------------------------------------------------------------------------
def _heading(ax, title: str, subtitle: str, *, dy: float = 21.0) -> None:
    ax.annotate(title, (0, 1), xycoords="axes fraction",
                textcoords="offset points", xytext=(0, dy),
                ha="left", va="baseline", fontsize=8.2, fontweight="semibold",
                color=COLORS["charcoal"], annotation_clip=False)
    # Methodological detail belongs in the caption, not in a second grey
    # typographic layer inside the figure.


def _letter(ax, letter: str, *, dx: float = -27.0, dy: float = 21.0) -> None:
    ax.annotate(letter, (0, 1), xycoords="axes fraction",
                textcoords="offset points", xytext=(dx, dy),
                ha="left", va="baseline", fontsize=10.5, fontweight="bold",
                color=COLORS["charcoal"], annotation_clip=False)


def _subtitle(ax, text: str, *, dy: float = 4.0) -> None:
    ax.annotate(text, (0, 1), xycoords="axes fraction",
                textcoords="offset points", xytext=(0, dy),
                ha="left", va="baseline", fontsize=6.8, fontweight="semibold",
                color=COLORS["charcoal"], annotation_clip=False)


def _panel_anchor(fig, spec):
    """Invisible full-cell axis for panel titles and letters.

    Titles tied to square heatmap axes drift vertically when Matplotlib adjusts
    their aspect.  Anchoring typography to the parent grid cell keeps every
    title and panel letter on the same row baseline.
    """

    anchor = fig.add_subplot(spec)
    anchor.set_axis_off()
    anchor.patch.set_visible(False)
    anchor.set_zorder(20)
    return anchor


def _figure_band(ax) -> None:
    ax.axvspan(5.0, 10.0, facecolor=COLORS["sage"], alpha=0.42,
               edgecolor="none", zorder=-20)


def _epoch_labels(ax, y: float) -> None:
    for centre, text, colour, weight in (
            (2.5, "pre", COLORS["ash"], "normal"),
            (7.5, "figure", COL_FIG, "semibold"),
            (12.5, "post", COLORS["ash"], "normal")):
        ax.text(centre, y, text, ha="center", va="bottom", fontsize=6.2,
                color=colour, fontweight=weight, clip_on=False)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(stop - 1)) for start, stop in edges.reshape(-1, 2)]


def _stars(probability: float) -> str:
    """Map a corrected probability to the journal-standard star code."""

    if probability < 0.001:
        return "***"
    if probability < 0.01:
        return "**"
    if probability < 0.05:
        return "*"
    return ""


def _significance_bracket(
    ax,
    left: float,
    right: float,
    base: float,
    height: float,
    stars: str,
) -> None:
    """Draw a compact comparison bracket with corrected-probability stars."""

    if not stars:
        return
    ax.plot(
        [left, left, right, right],
        [base, base + height, base + height, base],
        color=COL_SIG,
        lw=0.85,
        solid_capstyle="round",
        solid_joinstyle="round",
        clip_on=False,
        zorder=9,
    )
    ax.annotate(
        stars,
        ((left + right) / 2.0, base + height),
        xytext=(0, 1.2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        color=COL_SIG,
        annotation_clip=False,
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Panel A - the stimulus
# ---------------------------------------------------------------------------
def _panel_stimulus(fig, spec, data) -> None:
    inner = spec.subgridspec(1, 2, width_ratios=[1.0, 0.17], wspace=0.13)
    ax = fig.add_subplot(inner[0, 0])

    stim = np.asarray(data["stim"], dtype=float)
    dt = float(np.asarray(data["dt"]).ravel()[0])
    figure_index = np.asarray(data["figure_index"], dtype=int)
    n_channels, n_samples = stim.shape

    is_figure = np.zeros(n_channels, dtype=bool)
    is_figure[figure_index] = True

    _figure_band(ax)
    for channel in range(n_channels):
        active = stim[channel] > 0
        if not active.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
        colour = COL_FIG if is_figure[channel] else COL_GND
        for start, stop in zip(edges[::2], edges[1::2]):
            ax.add_patch(plt.Rectangle(
                (start * dt, channel - 0.40), (stop - start) * dt, 0.80,
                facecolor=colour, edgecolor="none",
                alpha=1.0 if is_figure[channel] else 0.60))

    ax.set_xlim(0, n_samples * dt)
    ax.set_ylim(n_channels - 0.5, -0.5)
    ax.set_yticks([0, 12, 24, 36])
    ax.set_xticks([0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    clean_axis(ax)
    _epoch_labels(ax, -1.4)

    _heading(ax,
             "Stimulus design and drive balance",
             "one presentation - 37 channels over three octaves - 5 s cloud, "
             "5 s figure, 5 s cloud, chords at about 4 per s - figure channels "
             "in green, ground cloud in grey", dy=25.0)
    _letter(ax, "A", dx=-30.0, dy=25.0)

    # Rate matching, shown rather than asserted: the two sets get the same
    # number of pips per channel, so they differ only in when those pips fall.
    ax2 = fig.add_subplot(inner[0, 1])
    onsets = np.diff((stim > 0).astype(np.int8), axis=1, prepend=np.int8(0)) == 1
    counts = onsets.sum(axis=1).astype(float)
    groups = [counts[is_figure], counts[~is_figure]]
    violins = ax2.violinplot(
        groups,
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, colour, alpha in zip(
        violins["bodies"], (COL_FIG, COL_GND), (0.20, 0.16)
    ):
        body.set_facecolor(colour)
        body.set_edgecolor("none")
        body.set_alpha(alpha)

    generator = np.random.default_rng(4_004)
    means = []
    for position, values, colour in zip((0, 1), groups, (COL_FIG, COL_GND)):
        jitter = generator.uniform(-0.17, 0.17, size=values.size)
        ax2.scatter(
            position + jitter,
            values,
            s=7.5,
            facecolor=colour,
            edgecolor=COLORS["white"],
            linewidth=0.28,
            alpha=0.78,
            zorder=4,
        )
        value = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(values.size))
        means.append(value)
        ax2.errorbar(
            position,
            value,
            yerr=sem,
            fmt="D",
            mfc=COLORS["white"],
            mec=colour,
            mew=0.85,
            ms=3.2,
            ecolor=colour,
            lw=0.85,
            capsize=1.8,
            capthick=0.75,
            zorder=6,
        )
    bracket_y = max(float(np.max(groups[0])), float(np.max(groups[1]))) + 0.5
    ax2.plot([0, 0, 1, 1], [bracket_y - 0.25, bracket_y, bracket_y,
                             bracket_y - 0.25],
             color=COLORS["charcoal"], lw=0.55, clip_on=False)
    ax2.text(0.5, bracket_y + 0.15,
             f"delta = {means[0] - means[1]:.1f} pips",
             ha="center", va="bottom", fontsize=5.3,
             color=COLORS["charcoal"], clip_on=False)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["fig", "gnd"])
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(49, bracket_y + 1.3)
    ax2.set_yticks([50, 55, 60])
    ax2.set_ylabel("Pips per channel", labelpad=1.5)
    clean_axis(ax2)
    _subtitle(ax2, "Drive match")


# ---------------------------------------------------------------------------
# Panel B - learned connectivity
# ---------------------------------------------------------------------------
def _panel_structure(fig, spec, data) -> None:
    panel_anchor = _panel_anchor(fig, spec)
    inner = spec.subgridspec(1, 2, width_ratios=[0.74, 1.0], wspace=0.72)

    figure_index = np.asarray(data["figure_index"], dtype=int)
    ground_index = np.asarray(data["ground_index"], dtype=int)
    order = np.r_[figure_index, ground_index]
    weights = np.asarray(data["W_final"], dtype=float)[np.ix_(order, order)]
    n_fig = len(figure_index)

    ax = fig.add_subplot(inner[0, 0])
    top = float(np.max(weights))
    image = ax.imshow(weights, cmap=PATTERN_CMAP, aspect="equal",
                      norm=Normalize(vmin=0.0, vmax=top),
                      interpolation="nearest")
    ax.set_anchor("N")
    ax.axhline(n_fig - 0.5, color=COL_FIG, lw=0.8)
    ax.axvline(n_fig - 0.5, color=COL_FIG, lw=0.8)
    centres = [n_fig / 2 - 0.5, (n_fig + len(order)) / 2 - 0.5]
    ax.set_xticks(centres)
    ax.set_xticklabels(["fig", "ground"])
    ax.set_yticks(centres)
    ax.set_yticklabels(["fig", "ground"], rotation=90, va="center")
    ax.set_xlabel("Presynaptic", labelpad=1.5)
    ax.set_ylabel("Postsynaptic", labelpad=1.5)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    _subtitle(ax, "Learned weights")

    bar = ax.inset_axes([1.05, 0.0, 0.055, 1.0])
    colourbar = fig.colorbar(image, cax=bar)
    colourbar.set_ticks([0, top])
    colourbar.set_ticklabels(["0", "max"])
    colourbar.outline.set_visible(False)
    bar.tick_params(length=1.4, pad=1.2)

    _heading(panel_anchor, "Figure-specific recurrent learning",
             f"exemplar session - {EXEMPLAR_SIZE}-tone figure - "
             "complete recurrent matrix", dy=25.0)
    _letter(panel_anchor, "B", dx=-30.0, dy=25.0)

    trace = fig.add_subplot(inner[0, 1])
    w_t = np.asarray(data["w_t"], dtype=float)
    trajectory = np.asarray(data["trajectory"], dtype=float)
    for column, colour, label, width, style in (
            (0, COL_FIG, "figure → figure", 1.5, "-"),
            (1, COL_GND, "ground → ground", 1.2, "-"),
            (2, COL_CROSS, "cross", 1.0, (0, (3, 2)))):
        trace.plot(w_t, trajectory[:, column], color=colour, lw=width,
                   ls=style, label=label)
    trace.set_xlabel("Session time (s)")
    trace.set_ylabel("Mean weight", labelpad=1.5)
    trace.set_xlim(0, w_t.max())
    trace.set_ylim(bottom=0)
    trace.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.06))
    clean_axis(trace)
    _subtitle(trace, "Growth over the session")


# ---------------------------------------------------------------------------
# Panel C - response dynamics
# ---------------------------------------------------------------------------
def _panel_dynamics(fig, spec, data, inference) -> None:
    ax = fig.add_subplot(spec)
    time_s = (np.arange(N_BINS) + 0.5) * BIN_MS / 1000.0

    _figure_band(ax)
    top = 0.0
    for group, colour, label in (("fig", COL_FIG, "figure channels"),
                                 ("gnd", COL_GND, "ground channels")):
        value, sem = mean_sem(_course(data, REFERENCE_PRESET,
                                      EXEMPLAR_SIZE, group))
        ax.fill_between(time_s, value - sem, value + sem, color=colour,
                        alpha=0.22, linewidth=0)
        ax.plot(time_s, value, color=colour, lw=1.5, label=label)
        top = max(top, float((value + sem).max()))

    ax.set_xlim(0, 15)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(0, top * 1.40)
    ax.set_xlabel("Time in presentation (s)")
    ax.set_ylabel("Response modulation (%)", labelpad=1.5)
    clean_axis(ax)
    _epoch_labels(ax, top * 1.07)
    cluster_mask = np.asarray(inference["c_significant"], dtype=bool)
    cluster_p = np.asarray(inference["c_p_corrected"], dtype=float)
    for start, stop in _segments(cluster_mask):
        left = max(0.0, time_s[start] - BIN_MS / 2000.0)
        right = min(15.0, time_s[stop] + BIN_MS / 2000.0)
        _significance_bracket(
            ax,
            left,
            right,
            top * 1.27,
            top * 0.035,
            _stars(float(np.nanmax(cluster_p[start:stop + 1]))),
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80))

    _heading(ax, "Time course of figure enhancement",
             f"{EXEMPLAR_SIZE}-tone figure - tone-triggered, 500 ms bins - "
             f"mean +/- SEM, {N_SEEDS} seeds", dy=25.0)
    _letter(ax, "C", dx=-30.0, dy=25.0)


# ---------------------------------------------------------------------------
# Panel D - figure size
# ---------------------------------------------------------------------------
def _panel_size(fig, spec, data, inference) -> None:
    panel_anchor = _panel_anchor(fig, spec)
    inner = spec.subgridspec(
        2,
        2,
        height_ratios=[0.30, 1.0],
        width_ratios=[1.0, 1.05],
        hspace=0.05,
        wspace=0.72,
    )
    sizes = np.array(FIGURE_SIZES, dtype=float)

    structure_key = fig.add_subplot(inner[0, 0])
    structure_key.set_axis_off()
    response_key = fig.add_subplot(inner[0, 1])
    response_key.set_axis_off()

    ax = fig.add_subplot(inner[1, 0])
    for preset, colour, label, style, marker in (
            (REFERENCE_PRESET, COL_FIG, "intact", "-", "o"),
            ("uniform", COLORS["terracotta"], "uniform inhibition",
             (0, (2.5, 2)), "s")):
        value, sem = mean_sem(np.stack([
            np.asarray(data[f"{preset}|plastic|{n}|assembly_drive"], dtype=float)
            for n in FIGURE_SIZES], axis=1))
        ax.errorbar(sizes, value, yerr=sem, color=colour, lw=1.4, ls=style,
                    marker=marker, ms=2.9, capsize=1.5, capthick=0.6,
                    label=label)
    ax.plot(sizes, np.zeros_like(sizes), color=COL_GND, lw=1.2, marker="o",
            ms=2.5, label="no plasticity", zorder=1)
    ax.set_xticks(sizes)
    ax.set_xlabel("Figure size (coherent tones)")
    ax.set_ylabel("Assembly drive (a.u.)", labelpad=1.5)
    ax.set_ylim(-0.008, 0.128)
    clean_axis(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    condition_order = [
        legend_labels.index(name)
        for name in ("intact", "uniform inhibition", "no plasticity")
    ]
    structure_key.text(
        0.0, 1.0, "Structure",
        ha="left", va="top", fontsize=6.8, fontweight="semibold",
        color=COLORS["charcoal"],
    )
    structure_key.legend(
        [handles[index] for index in condition_order],
        ["intact", "uniform inh.", "no plasticity"],
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.05),
        ncol=3,
        frameon=False,
        borderpad=0.0,
        fontsize=5.25,
        labelspacing=0.0,
        handlelength=0.90,
        handletextpad=0.30,
        columnspacing=0.42,
    )

    _heading(panel_anchor, "Dependence on figure size",
             f"mean +/- SEM over {N_SEEDS} session seeds", dy=25.0)
    _letter(panel_anchor, "D", dx=-30.0, dy=25.0)

    response = fig.add_subplot(inner[1, 1])
    response.axhline(0.0, color=COLORS["charcoal"], lw=0.6)
    for group, colour, label in (
        ("fig", COL_FIG, "figure channels"),
        ("gnd", COL_GND, "ground channels"),
    ):
        per_size = []
        for n_fig in FIGURE_SIZES:
            course = _course(data, REFERENCE_PRESET, n_fig, group)
            per_size.append(_window(course, FIGURE_BINS) -
                            _window(course, CLOUD_BINS))
        value, sem = mean_sem(np.stack(per_size, axis=1))
        response.errorbar(sizes, value, yerr=sem, color=colour, lw=1.5,
                          marker="o", ms=2.9, capsize=1.5, capthick=0.6,
                          label=label)
    response.set_xticks(sizes)
    response.set_xlabel("Figure size (coherent tones)")
    response.set_ylabel("Figure-epoch change (%)", labelpad=1.5)
    response.set_ylim(-1.05, 4.75)
    clean_axis(response)
    response_key.text(
        0.0, 1.0, "Response",
        ha="left", va="top", fontsize=6.8, fontweight="semibold",
        color=COLORS["charcoal"],
    )
    response_key.legend(
        *response.get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.05),
        ncol=2,
        frameon=False,
        borderpad=0.0,
        fontsize=5.4,
        labelspacing=0.0,
        handlelength=1.0,
        handletextpad=0.34,
        columnspacing=0.55,
    )


# ---------------------------------------------------------------------------
# Panel E - mechanism
# ---------------------------------------------------------------------------
def _panel_mechanism(fig, spec, data, inference) -> None:
    ax = fig.add_subplot(spec)
    order = ("tm", "rec", "inh", "net")
    labels = ("Thalamic", "Recurrent", "Inhibition", "Net drive")
    x = np.arange(len(order), dtype=float)
    width = 0.33

    ax.axhline(0.0, color=COLORS["charcoal"], lw=0.6)
    e_p_corrected = np.asarray(inference["e_p_corrected"], dtype=float)
    for group_index, (offset, group, colour, label) in enumerate(
        ((-width / 2, "fig", COL_FIG, "figure"),
         (width / 2, "gnd", COL_GND, "ground"))
    ):
        values, sems = [], []
        for name in order:
            value, sem = mean_sem(difference(data, REFERENCE_PRESET,
                                             EXEMPLAR_SIZE,
                                             f"{name}_{group}_figure"))
            values.append(float(value)); sems.append(float(sem))
        ax.bar(x + offset, values, width, facecolor=colour, edgecolor="none",
               alpha=1.0 if group == "fig" else 0.60, label=label)
        ax.errorbar(x + offset, values, yerr=sems, fmt="none",
                    ecolor=COLORS["charcoal"], lw=0.7, capsize=1.5,
                    capthick=0.7)
        for current_index, (name, position, value, sem) in enumerate(
            zip(order, x + offset, values, sems)
        ):
            if name == "tm":
                continue
            significance_index = group_index * 3 + current_index - 1
            stars = _stars(float(e_p_corrected[significance_index]))
            if stars:
                ax.text(
                    position,
                    value + sem + 0.011,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color=COL_SIG,
                    zorder=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Change from frozen control (a.u.)", labelpad=1.5)
    ax.set_ylim(-0.005, 0.475)
    clean_axis(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.02))
    ax.annotate("0 by\nconstruction", (0, 0), textcoords="offset points",
                xytext=(0, 6), ha="center", va="bottom", fontsize=5.3,
                color=COLORS["ash"], linespacing=1.35)

    _heading(ax, "Synaptic contributions to figure enhancement",
             f"{EXEMPLAR_SIZE}-tone figure - tone-triggered currents, figure "
             f"epoch - mean +/- SEM, {N_SEEDS} seeds", dy=25.0)
    _letter(ax, "E", dx=-30.0, dy=25.0)


# ---------------------------------------------------------------------------
# Panel F - buildup
# ---------------------------------------------------------------------------
def _panel_buildup(fig, spec, data, inference) -> None:
    ax = fig.add_subplot(spec)
    presentations = np.arange(1, N_REPS + 1)

    slope_p_corrected = np.asarray(
        inference["f_slope_p_corrected"], dtype=float
    )
    for size_index, (n_fig, colour) in enumerate(zip(FIGURE_SIZES, SIZE_RAMP)):
        value, sem = mean_sem(modulation(data, REFERENCE_PRESET, n_fig,
                                         "buildup_fig"))
        ax.fill_between(presentations, value - sem, value + sem, color=colour,
                        alpha=0.20, linewidth=0)
        ax.plot(presentations, value, color=colour, lw=1.4, label=f"{n_fig}")
        stars = _stars(float(slope_p_corrected[size_index]))
        if stars:
            ax.annotate(
                stars,
                (presentations[-1], value[-1] + sem[-1]),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=COL_SIG,
                annotation_clip=False,
                zorder=8,
            )
    ax.set_xlim(1, N_REPS + 0.65)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Presentation")
    ax.set_ylabel("Figure-epoch modulation (%)", labelpad=1.5)
    clean_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    size_legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.06),
        ncol=4,
        title="Figure size (coherent tones)",
        handlelength=1.0,
        columnspacing=0.75,
    )
    size_legend.get_title().set_fontsize(6.3)
    size_legend._legend_box.align = "left"
    ax.add_artist(size_legend)

    _heading(ax, "Figure enhancement across presentations",
             "figure channels - the ferret buildup is within one figure, "
             "this model's across them", dy=25.0)
    _letter(ax, "F", dx=-30.0, dy=25.0)


# ---------------------------------------------------------------------------
# Panel G - persistence
# ---------------------------------------------------------------------------
def _panel_persistence(fig, spec, data, inference) -> None:
    ax = fig.add_subplot(spec)
    positions = np.arange(len(FIGURE_SIZES), dtype=float)
    width = 0.33

    values_by_group: dict[str, np.ndarray] = {}
    sems_by_group: dict[str, np.ndarray] = {}
    for offset, group, colour, label in ((-width / 2, "fig", COL_FIG, "figure"),
                                         (width / 2, "gnd", COL_GND, "ground")):
        values, sems = [], []
        for n_fig in FIGURE_SIZES:
            course = _course(data, REFERENCE_PRESET, n_fig, group)
            value, sem = mean_sem(_window(course, CLOUD_BINS))
            values.append(float(value)); sems.append(float(sem))
        ax.bar(positions + offset, values, width, facecolor=colour,
               edgecolor="none", alpha=1.0 if group == "fig" else 0.60,
               label=label)
        ax.errorbar(positions + offset, values, yerr=sems, fmt="none",
                    ecolor=COLORS["charcoal"], lw=0.7, capsize=1.5,
                    capthick=0.7)
        values_by_group[group] = np.asarray(values, dtype=float)
        sems_by_group[group] = np.asarray(sems, dtype=float)

    bracket_tops: list[float] = []
    g_p_corrected = np.asarray(inference["g_p_corrected"], dtype=float)
    for size_index, position in enumerate(positions):
        bracket_y = max(
            values_by_group["fig"][size_index]
            + sems_by_group["fig"][size_index],
            values_by_group["gnd"][size_index]
            + sems_by_group["gnd"][size_index],
        ) + 0.08
        bracket_tops.append(bracket_y + 0.08)
        left, right = position - width / 2, position + width / 2
        _significance_bracket(
            ax,
            left,
            right,
            bracket_y,
            0.045,
            _stars(float(g_p_corrected[size_index])),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in FIGURE_SIZES])
    ax.set_xlabel("Figure size (coherent tones)")
    ax.set_ylabel("Cloud-epoch modulation (%)", labelpad=1.5)
    ax.set_ylim(0, max(bracket_tops) + 0.17)
    clean_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.025), ncol=2)

    _heading(ax, "Modulation during cloud epochs",
             "pre- and post-figure epochs pooled - no chord is present",
             dy=25.0)
    _letter(ax, "G", dx=-30.0, dy=25.0)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_figure(data, inference) -> plt.Figure:
    fig = plt.figure(figsize=(mm(183), mm(208)))
    grid = fig.add_gridspec(
        4, 2,
        left=0.078, right=0.972, top=0.925, bottom=0.055,
        height_ratios=[1.10, 0.92, 0.92, 0.92],
        hspace=0.80, wspace=0.34)

    _panel_stimulus(fig, grid[0, :], data)
    _panel_structure(fig, grid[1, 0], data)
    _panel_dynamics(fig, grid[1, 1], data, inference)
    _panel_size(fig, grid[2, 0], data, inference)
    _panel_mechanism(fig, grid[2, 1], data, inference)
    _panel_buildup(fig, grid[3, 0], data, inference)
    _panel_persistence(fig, grid[3, 1], data, inference)
    return fig


def _write_summary(data) -> Path:
    """Every number quoted in the caption, one row per condition."""

    path = DATA_DIR / "figure_4_summary.csv"
    rows: list[dict[str, object]] = []
    for preset in (REFERENCE_PRESET, "uniform"):
        for n_fig in FIGURE_SIZES:
            row: dict[str, object] = {"preset": preset, "n_fig": n_fig}
            drive, drive_sem = mean_sem(np.asarray(
                data[f"{preset}|plastic|{n_fig}|assembly_drive"], dtype=float))
            row["assembly_drive"] = float(drive)
            row["assembly_drive_sem"] = float(drive_sem)
            for group in ("fig", "gnd"):
                course = _course(data, preset, n_fig, group)
                figure_epoch = _window(course, FIGURE_BINS)
                cloud = _window(course, CLOUD_BINS)
                for name, series in (("figure_mod_pct", figure_epoch),
                                     ("cloud_mod_pct", cloud),
                                     ("figure_minus_cloud_pct",
                                      figure_epoch - cloud)):
                    value, sem = mean_sem(series)
                    row[f"{group}_{name}"] = float(value)
                    row[f"{group}_{name}_sem"] = float(sem)
                for current in ("tm", "rec", "inh", "net"):
                    value, sem = mean_sem(difference(
                        data, preset, n_fig, f"{current}_{group}_figure"))
                    row[f"{group}_d{current}"] = float(value)
                    row[f"{group}_d{current}_sem"] = float(sem)
            rows.append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_drive_check(data) -> Path:
    """Export every channel count shown in Panel A's drive-match inset."""

    path = DATA_DIR / "figure_4_drive_check.csv"
    stimulus = np.asarray(data["stim"], dtype=float)
    figure_index = np.asarray(data["figure_index"], dtype=int)
    is_figure = np.zeros(stimulus.shape[0], dtype=bool)
    is_figure[figure_index] = True
    onsets = np.diff(
        (stimulus > 0).astype(np.int8),
        axis=1,
        prepend=np.int8(0),
    ) == 1
    counts = onsets.sum(axis=1)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_matlab", "group", "pips_per_channel"])
        for channel, (figure, count) in enumerate(
            zip(is_figure, counts), start=1
        ):
            writer.writerow([channel, "figure" if figure else "ground",
                             int(count)])
    return path


def _report(data) -> None:
    print("\n  figure-epoch change against own cloud epochs, per cent:")
    for n_fig in FIGURE_SIZES:
        parts = []
        for group in ("fig", "gnd"):
            course = _course(data, REFERENCE_PRESET, n_fig, group)
            value, sem = mean_sem(_window(course, FIGURE_BINS) -
                                  _window(course, CLOUD_BINS))
            parts.append(f"{group} {value:+.2f} +/- {sem:.2f}")
        print(f"    size {n_fig:>2d}:  " + "   ".join(parts))

    print("\n  cloud-epoch modulation, per cent (the persistent trace):")
    for n_fig in FIGURE_SIZES:
        parts = []
        for group in ("fig", "gnd"):
            course = _course(data, REFERENCE_PRESET, n_fig, group)
            value, sem = mean_sem(_window(course, CLOUD_BINS))
            parts.append(f"{group} {value:+.2f} +/- {sem:.2f}")
        print(f"    size {n_fig:>2d}:  " + "   ".join(parts))

    worst = max(abs(float(mean_sem(difference(
        data, preset, n_fig, f"tm_{group}_figure"))[0]))
        for preset in (REFERENCE_PRESET, "uniform")
        for n_fig in FIGURE_SIZES for group in ("fig", "gnd"))
    print(f"\n  drive-matching check, max |delta thalamic| = {worst:.2e} "
          "(must be 0)")

    selective = np.array([mean_sem(np.asarray(
        data[f"selective|plastic|{n}|assembly_drive"]))[0]
        for n in FIGURE_SIZES])
    uniform = np.array([mean_sem(np.asarray(
        data[f"uniform|plastic|{n}|assembly_drive"]))[0]
        for n in FIGURE_SIZES])
    print(f"  uniform vs selective assembly drive: max relative difference "
          f"{np.abs(uniform / selective - 1).max() * 100:.2f}%")


def main(arguments: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-data", action="store_true",
                        help="Repeat every simulated SFG session.")
    options = parser.parse_args(arguments)

    data = load_all(force=options.force_data)
    inference = build_inference(data, data_dir=DATA_DIR)
    with manuscript_style():
        figure = build_figure(data, inference)
        paths = export_figure(figure, OUTPUT_STEM, fixed_bounds=True)
    plt.close(figure)

    summary = _write_summary(data)
    drive_check = _write_drive_check(data)
    _report(data)
    provenance = DATA_DIR / "figure_4_provenance.json"
    provenance.write_text(json.dumps({
        "figure": "Figure 4 - stochastic figure-ground",
        "physiology_reference": (
            "Lu, Dutta, Mohammed, Elhilali & Shamma (2025) iScience 28:111991. "
            "Cited, not reproduced: every panel here is model output."
        ),
        "outputs": {kind: {"path": str(path.relative_to(HERE)),
                           "sha256": _sha256(path)}
                    for kind, path in paths.items()},
        "summary_csv": str(summary.relative_to(HERE)),
        "drive_check_csv": str(drive_check.relative_to(HERE)),
        "inference": {
            "npz": "data/figure_4_inference.npz",
            "csv": "data/figure_4_inference.csv",
            "provenance": "data/figure_4_inference_provenance.json",
        },
    }, indent=2, sort_keys=True) + "\n")

    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
