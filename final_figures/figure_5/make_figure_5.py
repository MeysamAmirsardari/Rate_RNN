"""Build the submission-ready Saffran composition Figure 5."""

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

from final_figures.figure_5.saffran_data import (
    DETECTOR_CRITERION,
    N_CHANNELS,
    WORD_NAMES,
    WORDS,
    build_saffran_data,
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
OUTPUT_STEM = OUTPUT_DIR / "figure_5_composition"

READOUT_ORDER = ("single_rate", "multi_rate")
READOUT_LABELS = {
    "single_rate": "Single timescale",
    "multi_rate": "Multiscale",
}
READOUT_SHORT = {
    "single_rate": "Single\ntimescale",
    "multi_rate": "Multiscale",
}
READOUT_COLORS = {
    "single_rate": "#7E8792",
    "multi_rate": COLORS["decoder"],
}
READOUT_MARKERS = {"single_rate": "s", "multi_rate": "o"}

#: Word identity is categorical and arbitrary, so the four hues are chosen for
#: maximum separation rather than meaning.  They carry no condition semantics.
WORD_COLORS = (
    COLORS["decoder"],
    COLORS["teal"],
    COLORS["terracotta"],
    COLORS["rep15"],
)

def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _key(exposure: str, mode: str, readout: str, field: str) -> str:
    return f"{exposure}|{mode}|{readout}|{field}"


def _get(data, exposure, mode, readout, field) -> np.ndarray:
    return np.asarray(data[_key(exposure, mode, readout, field)], dtype=float)


def _panel_anchor(fig, spec):
    """Return an invisible full-cell axis for stable panel typography."""

    anchor = fig.add_subplot(spec)
    anchor.set_axis_off()
    anchor.patch.set_visible(False)
    anchor.set_zorder(20)
    return anchor


def _panel_heading(
    ax,
    letter: str,
    title: str,
    *,
    dy: float = 18.0,
    dx: float = -23.0,
) -> None:
    """Place panel letter and title on one print-stable baseline."""

    ax.annotate(
        title,
        (0, 1),
        xycoords="axes fraction",
        textcoords="offset points",
        xytext=(0, dy),
        ha="left",
        va="baseline",
        fontsize=8.2,
        fontweight="semibold",
        color=COLORS["charcoal"],
        annotation_clip=False,
    )
    ax.annotate(
        letter,
        (0, 1),
        xycoords="axes fraction",
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left",
        va="baseline",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["charcoal"],
        annotation_clip=False,
    )


def _true_transition_matrix() -> np.ndarray:
    """P(next token | current token) implied by the paradigm definition."""

    matrix = np.zeros((N_CHANNELS, N_CHANNELS))
    finals = [w[-1] for w in WORDS]
    firsts = [w[0] for w in WORDS]
    for word in WORDS:
        for current, following in zip(word[:-1], word[1:]):
            matrix[following, current] = 1.0
    for index, final in enumerate(finals):
        for other, first in enumerate(firsts):
            if other != index:
                matrix[first, final] = 1.0 / 3.0
    return matrix


def _seed_mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    sem = float(np.nanstd(values, ddof=1) / np.sqrt(values.size))
    return mean, sem


# ---------------------------------------------------------------------------
# Panel A — the stream
# ---------------------------------------------------------------------------
def _panel_stream(fig, spec, data) -> None:
    panel_anchor = _panel_anchor(fig, spec)
    ax = fig.add_subplot(spec)
    stim = np.asarray(data["excerpt_stim"], dtype=float)
    dt = float(np.asarray(data["excerpt_dt"]).ravel()[0])
    order = np.asarray(data["excerpt_order"], dtype=int)
    onsets = np.asarray(data["excerpt_onsets"], dtype=float)
    n_tone = float(np.asarray(data["excerpt_n_tone"]).ravel()[0])

    time_ms = np.arange(stim.shape[1]) * dt * 1e3
    channel_word = np.full(N_CHANNELS, -1, dtype=int)
    for index, word in enumerate(WORDS):
        for token in word:
            channel_word[token] = index

    for channel in range(N_CHANNELS):
        active = stim[channel] > 0
        if not active.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
        for start, stop in edges.reshape(-1, 2):
            ax.add_patch(
                plt.Rectangle(
                    (time_ms[start], channel - 0.34),
                    time_ms[min(stop, time_ms.size - 1)] - time_ms[start],
                    0.68,
                    facecolor=WORD_COLORS[channel_word[channel]],
                    edgecolor="none",
                    zorder=3,
                )
            )

    stops = np.asarray(data["excerpt_word_stops"], dtype=float)
    for onset, stop, word_index in zip(onsets, stops, order):
        start, finish = onset * dt * 1e3, stop * dt * 1e3
        ax.axvline(start, color=COLORS["ash"], lw=0.45, ls=(0, (2.4, 2.0)),
                   alpha=0.85, zorder=1)
        ax.text(
            (start + finish) / 2, 1.03,
            WORD_NAMES[word_index],
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=5.4, fontweight="semibold",
            color=WORD_COLORS[word_index], clip_on=False,
        )

    ax.set_xlim(0, time_ms[-1])
    ax.set_ylim(N_CHANNELS - 0.4, -0.6)
    ax.set_yticks([0, 3, 6, 9, 11])
    ax.set_xticks(np.arange(0, time_ms[-1] + 1, 400))
    ax.set_xticklabels([])
    ax.set_ylabel("Token channel", labelpad=2.0)
    clean_axis(ax, bottom=False)
    ax.tick_params(axis="x", length=0)
    _panel_heading(
        panel_anchor,
        "A",
        "An isochronous stream contains only statistical boundaries",
    )


# ---------------------------------------------------------------------------
# Panel B — what the units do while the stream plays
# ---------------------------------------------------------------------------
def _panel_tape(fig, spec, data) -> None:
    """Whole-word unit activity under the same stream excerpt as panel A."""

    panel_anchor = _panel_anchor(fig, spec)
    ax = fig.add_subplot(spec)
    dt = float(np.asarray(data["excerpt_dt"]).ravel()[0])
    activity = np.asarray(data["excerpt_activity"], dtype=float)
    onsets = np.asarray(data["excerpt_onsets"], dtype=float) * dt * 1e3
    stops = np.asarray(data["excerpt_word_stops"], dtype=float) * dt * 1e3
    order = np.asarray(data["excerpt_order"], dtype=int)
    unit_word = np.asarray(data["exemplar_span_word"], dtype=int)
    unit_index = np.asarray(data["exemplar_span_units"], dtype=int)
    time_ms = np.arange(activity.shape[1]) * dt * 1e3

    # Tint every word occurrence in its own colour, so a trace peaking inside a
    # band of the same colour is the whole claim, seen rather than annotated.
    for onset, stop, word in zip(onsets, stops, order):
        ax.axvspan(onset, stop, color=WORD_COLORS[word], alpha=0.13,
                   linewidth=0, zorder=0)

    step = 1.25
    n_units = activity.shape[0]
    for row in range(n_units):
        trace = activity[row]
        peak = float(trace.max())
        trace = trace / peak if peak > 0 else trace
        base = (n_units - 1 - row) * step
        colour = WORD_COLORS[unit_word[row]]
        ax.fill_between(time_ms, base, base + trace, color=colour, alpha=0.55,
                        linewidth=0, zorder=3)
        ax.plot(time_ms, base + trace, color=colour, lw=0.8, zorder=4)
        ax.text(-0.006, base + 0.45, f"unit {unit_index[row]}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=5.5, color=colour, fontweight="semibold")

    ax.set_xlim(0, time_ms[-1])
    ax.set_ylim(-0.18, (n_units - 1) * step + 1.25)
    ax.set_xticks(np.arange(0, time_ms[-1] + 1, 400))
    ax.set_yticks([])
    ax.set_xlabel("Stream time (ms)")
    clean_axis(ax, left=False)
    _panel_heading(
        panel_anchor,
        "B",
        "Whole-word units activate at their learned items",
    )


# ---------------------------------------------------------------------------
# Panel D — pairwise transition structure
# ---------------------------------------------------------------------------
def _panel_transitions(fig, spec, data) -> list:
    panel_anchor = _panel_anchor(fig, spec)
    nested = spec.subgridspec(1, 3, width_ratios=(1, 1, 0.055), wspace=0.24)
    ax_true = fig.add_subplot(nested[0, 0])
    ax_learned = fig.add_subplot(nested[0, 1])

    truth = _true_transition_matrix()
    learned = np.asarray(data["exemplar_layer1_weights"], dtype=float).copy()
    # Self-connections are not transitions between tokens and are absent from
    # the designed matrix by construction.  They are strong in model0, so
    # leaving them in would set the colour scale and bury the structure this
    # panel exists to show.  They are masked in both maps and declared.
    np.fill_diagonal(learned, np.nan)
    off_diagonal_max = np.nanmax(learned)
    if off_diagonal_max > 0:
        learned = learned / off_diagonal_max
    truth = truth.copy()
    np.fill_diagonal(truth, np.nan)

    image = None
    for ax, matrix, title in (
        (ax_true, truth, "Designed statistics"),
        (ax_learned, learned, "Learned layer-1 weights"),
    ):
        cmap = PATTERN_CMAP.copy()
        cmap.set_bad(COLORS["white"])
        image = ax.imshow(
            np.ma.masked_invalid(matrix), cmap=cmap, norm=Normalize(0, 1),
            interpolation="nearest", aspect="equal", origin="upper",
            rasterized=True,
        )
        for boundary in (2.5, 5.5, 8.5):
            ax.axhline(boundary, color=COLORS["white"], lw=0.6, alpha=0.9)
            ax.axvline(boundary, color=COLORS["white"], lw=0.6, alpha=0.9)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_yticks([1, 4, 7, 10])
        ax.set_xticklabels(WORD_NAMES, fontsize=5.6)
        ax.set_yticklabels(WORD_NAMES, fontsize=5.6)
        ax.set_xlabel("Preceding token", labelpad=1.6)
        ax.set_title(title, loc="left", fontsize=6.6, pad=2.2)
        ax.tick_params(length=1.4, pad=1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_true.set_ylabel("Following token", labelpad=1.6)
    ax_learned.set_yticklabels([])

    colorbar_axis = fig.add_subplot(nested[0, 2])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["0", "max"])
    colorbar.ax.tick_params(labelsize=5.6, length=1.4, width=0.45, pad=1.0)
    colorbar.outline.set_linewidth(0.45)

    _panel_heading(panel_anchor, "E", "Layer 1 learns pairwise transitions")
    return [ax_true, ax_learned]


# ---------------------------------------------------------------------------
# Panel C — the single-timescale limitation
# ---------------------------------------------------------------------------
#: The three conditions every measure in panel E is evaluated on.  Sharing one
#: x-axis across the three sub-panels is the point: the reader learns it once,
#: and then sees which measures move and which do not.
CONTRASTS = (
    ("structured", "full", "single_rate", "Single\ntimescale", "#7E8792", "s"),
    ("structured", "full", "multi_rate", "Multiscale", COLORS["decoder"], "o"),
    ("scrambled", "full", "multi_rate", "Multiscale\nscrambled",
     COLORS["terracotta"], "^"),
)


def _panel_metrics(fig, spec, data) -> list:
    """Behaviour, function and structure on one shared set of conditions."""

    panel_anchor = _panel_anchor(fig, spec)
    nested = spec.subgridspec(1, 3, wspace=0.42)
    axes = [fig.add_subplot(nested[0, index]) for index in range(3)]
    rng = np.random.default_rng(20260731)

    measures = (
        ("auc", "Word vs part-word AUC", "Familiarity",
         "does it look familiar?", (0.34, 1.04), [0.5, 0.75, 1.0], 0.5),
        ("n_word_detectors", "Items with a selective unit", "Whole-item selectivity",
         f"any unit tuned to the item? (held-out AUC ≥ {DETECTOR_CRITERION:.2f})",
         (-0.35, 4.7), [0, 2, 4], None),
        ("n_words_covered", "Words held in order", "Ordered composition",
         "is the word represented as a word?", (-0.35, 4.7), [0, 2, 4], None),
    )

    for ax, (field, ylabel, title, question, ylim, yticks, rule) in zip(
        axes, measures
    ):
        if rule is not None:
            ax.axhline(rule, color=COLORS["ash"], lw=0.55, ls=(0, (2, 2)),
                       zorder=0)
            ax.text(2.52, rule + 0.012, "chance", fontsize=5.1,
                    color=COLORS["ash"], ha="right", va="bottom")
        for index, (exposure, mode, readout, _label, colour, marker) in enumerate(
            CONTRASTS
        ):
            values = _get(data, exposure, mode, readout, field)
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(np.full(values.size, index) + jitter, values, s=7.0,
                       color=colour, alpha=0.30, linewidth=0, zorder=2)
            mean, sem = _seed_mean_sem(values)
            ax.errorbar(index, mean, yerr=sem, fmt=marker, ms=4.2, color=colour,
                        markeredgecolor=COLORS["white"], markeredgewidth=0.55,
                        elinewidth=1.2, capsize=2.0, capthick=0.7, zorder=5)
        ax.set_xlim(-0.58, 2.58)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([c[3] for c in CONTRASTS], fontsize=5.5)
        ax.set_ylabel(ylabel, labelpad=2.0)
        ax.set_title(title, loc="left", fontsize=6.8, pad=2.4)
        clean_axis(ax)

    _panel_heading(
        panel_anchor,
        "D",
        "Only ordered structure distinguishes the readouts",
    )
    return axes


def _panel_filterbank(fig, spec, data) -> list:
    panel_anchor = _panel_anchor(fig, spec)
    nested = spec.subgridspec(1, 2, wspace=0.42, width_ratios=(1.12, 0.88))
    ax_bank = fig.add_subplot(nested[0, 0])
    ax_code = fig.add_subplot(nested[0, 1])

    time_ms = np.asarray(data["filterbank_time_ms"], dtype=float)
    normalized = np.asarray(data["filterbank_normalized"], dtype=float)
    rates_s = np.asarray(data["filterbank_rates_s"], dtype=float)
    shades = plt.get_cmap("BuPu")(np.linspace(0.32, 0.94, rates_s.size))

    ax_bank.axvspan(0, 50, color=COLORS["lavender"], alpha=0.75, lw=0, zorder=-20)
    for index, rate in enumerate(rates_s):
        ax_bank.plot(time_ms, normalized[:, index], color=shades[index],
                     lw=1.05, zorder=3)
    ax_bank.text(58, 1.10, "token", ha="left", va="top", fontsize=5.5,
                 color=COLORS["decoder"], fontweight="semibold")
    ax_bank.set_xlim(0, 1000)
    ax_bank.set_ylim(0, 1.14)
    ax_bank.set_xticks([0, 250, 500, 750, 1000])
    ax_bank.set_yticks([0, 0.5, 1.0])
    ax_bank.set_xlabel("Time since token onset (ms)")
    ax_bank.set_ylabel("Normalized trace", labelpad=2.0)
    ax_bank.set_title("Bank of six timescales", loc="left", fontsize=6.6, pad=2.4)
    ax_bank.text(
        0.985, 0.86, f"τ {rates_s[0] * 1e3:.0f}–{rates_s[-1] * 1e3:.0f} ms",
        transform=ax_bank.transAxes, ha="right", va="top", fontsize=5.6,
        color=COLORS["ash"],
    )
    clean_axis(ax_bank)

    probe_ms = (80.0, 160.0, 320.0)
    probe_colors = plt.get_cmap("BuPu")(np.linspace(0.40, 0.92, len(probe_ms)))
    for index, elapsed in enumerate(probe_ms):
        sample = int(np.argmin(np.abs(time_ms - elapsed)))
        profile = normalized[sample]
        profile = profile / profile.max() if profile.max() > 0 else profile
        ax_code.plot(rates_s * 1e3, profile, color=probe_colors[index], lw=1.05,
                     marker="o", ms=2.6, mew=0, zorder=3,
                     label=f"{elapsed:.0f} ms ago")
    ax_code.set_xscale("log")
    ax_code.set_xlim(rates_s[0] * 1e3 * 0.82, rates_s[-1] * 1e3 * 1.2)
    ax_code.set_ylim(0, 1.12)
    ax_code.set_xticks([30, 100, 300])
    ax_code.set_xticklabels(["30", "100", "300"])
    ax_code.set_yticks([0, 0.5, 1.0])
    ax_code.set_xlabel("Filter τ (ms)")
    ax_code.set_ylabel("Relative activation", labelpad=2.0)
    ax_code.set_title("Elapsed time is legible", loc="left", fontsize=6.6, pad=2.4)
    ax_code.legend(loc="lower left", fontsize=5.4, labelspacing=0.18,
                   handlelength=1.0, borderaxespad=0.15)
    clean_axis(ax_code)

    _panel_heading(
        panel_anchor,
        "F",
        "Multiscale traces add an elapsed-time code",
    )
    return [ax_bank, ax_code]


# ---------------------------------------------------------------------------
# Panel E — learned whole-word templates
# ---------------------------------------------------------------------------
def _panel_kernels(fig, spec, data) -> list:
    """Show the informative mask row as context token by filter timescale.

    The complete mask is ``channel now × (filter × context channel)``.  Each
    committed exemplar is driven from one current-token row, so plotting all 14
    rows mainly displays empty space.  Here that measured driving row is
    reshaped without reduction into ``context token × filter``.  This preserves
    every weight used by the unit while making ordered multiscale structure
    readable at final print size.
    """

    panel_anchor = _panel_anchor(fig, spec)
    if "exemplar_span_masks" not in data:
        ax = fig.add_subplot(spec)
        ax.axis("off")
        _panel_heading(panel_anchor, "C", "Whole-word templates in the learned masks")
        ax.text(0.5, 0.5, "no spanning unit in the exemplar seed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.0, color=COLORS["ash"])
        return [ax]

    masks = np.asarray(data["exemplar_span_masks"], dtype=float)
    now_channels = np.asarray(data["exemplar_span_now"], dtype=int)
    word_indices = np.asarray(data["exemplar_span_word"], dtype=int)
    taus = np.asarray(data["exemplar_taus"], dtype=float)
    n_units, n_now, n_context = masks.shape
    n_rates = taus.size
    if n_context != N_CHANNELS * n_rates:
        raise ValueError(
            f"Expected {N_CHANNELS * n_rates} context features, got {n_context}"
        )

    nested = spec.subgridspec(
        1,
        n_units + 1,
        width_ratios=(*([1.0] * n_units), 0.045),
        wspace=0.28,
    )
    axes = [fig.add_subplot(nested[0, column]) for column in range(n_units)]
    image = None

    for row, ax in enumerate(axes):
        mask = masks[row]
        current = int(now_channels[row])
        active_row = mask[current].reshape(n_rates, N_CHANNELS).T
        maximum = float(np.nanmax(active_row))
        normalized = active_row / maximum if maximum > 0 else active_row
        image = ax.imshow(
            normalized,
            cmap=PATTERN_CMAP,
            norm=Normalize(0, 1),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            rasterized=True,
        )
        word = WORDS[word_indices[row]]
        colour = WORD_COLORS[word_indices[row]]

        ax.set_title(
            f"{WORD_NAMES[word_indices[row]]}  "
            f"{' → '.join(str(token) for token in word)}\n"
            f"current token {current}",
            loc="left",
            fontsize=6.1,
            color=colour,
            pad=2.4,
        )
        ax.set_xticks(np.arange(n_rates))
        ax.set_xticklabels([f"{tau * 1e3:.0f}" for tau in taus], fontsize=4.9)
        ax.set_xlabel("filter τ (ms)", fontsize=5.4, labelpad=1.1)
        ax.set_yticks(list(word[:-1]))
        ax.set_yticklabels([str(token) for token in word[:-1]], fontsize=5.2)
        ax.tick_params(length=1.3, pad=1.0)
        if row == 0:
            ax.set_ylabel("Context token", fontsize=5.8, labelpad=1.8)
        else:
            ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar_axis = fig.add_subplot(nested[0, -1])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["low", "high"])
    colorbar.ax.tick_params(labelsize=5.2, length=1.3, width=0.45, pad=1.0)
    colorbar.outline.set_linewidth(0.45)

    _panel_heading(
        panel_anchor,
        "C",
        "Whole-word templates in the learned masks",
    )
    return axes


#: Layer-1 modes get a lightness ramp of one hue.  Reusing the condition hues
#: here would give the same colour two different meanings inside one figure.
MODE_ORDER = ("full", "frozen", "raw")
MODE_SHORT = {"full": "Plastic", "frozen": "Frozen", "raw": "Absent"}
MODE_RAMP = {
    "full": COLORS["decoder"],
    "frozen": "#9586AD",
    "raw": "#CBC4D5",
}


def _panel_controls(fig, spec, data) -> list:
    """Does composition need layer 1's learned representation?"""

    panel_anchor = _panel_anchor(fig, spec)
    nested = spec.subgridspec(1, 2, wspace=0.52)
    ax_span = fig.add_subplot(nested[0, 0])
    ax_auc = fig.add_subplot(nested[0, 1])
    rng = np.random.default_rng(4242)

    for ax, field, ylabel, ylim, yticks in (
        (ax_span, "n_words_covered", "Words held in order", (-0.35, 4.7), [0, 2, 4]),
        (ax_auc, "auc", "Word vs part-word AUC", (0.62, 1.03), [0.7, 0.85, 1.0]),
    ):
        for index, mode in enumerate(MODE_ORDER):
            values = _get(data, "structured", mode, "multi_rate", field)
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(np.full(values.size, index) + jitter, values, s=7.0,
                       color=MODE_RAMP[mode], alpha=0.30, linewidth=0, zorder=2)
            mean, sem = _seed_mean_sem(values)
            ax.errorbar(index, mean, yerr=sem, fmt="o", ms=4.2,
                        color=MODE_RAMP[mode], markeredgecolor=COLORS["white"],
                        markeredgewidth=0.55, elinewidth=1.2, capsize=2.0,
                        capthick=0.7, zorder=5)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([MODE_SHORT[m] for m in MODE_ORDER], fontsize=5.6)
        ax.set_xlabel("Layer 1", labelpad=1.6)
        ax.set_ylabel(ylabel, labelpad=2.0)
        clean_axis(ax)
    ax_span.set_title("Composition", loc="left", fontsize=6.8, pad=2.4)
    ax_auc.set_title("Behaviour", loc="left", fontsize=6.8, pad=2.4)

    _panel_heading(
        panel_anchor,
        "G",
        "Composition survives layer-1 controls",
    )
    return [ax_span, ax_auc]


def _panel_synthesis(fig, spec, data) -> list:
    """Closing state-space: familiarity and composition are distinct axes."""

    panel_anchor = _panel_anchor(fig, spec)
    ax = fig.add_subplot(spec)

    # Four conceptual states make the interpretation explicit.  The neutral
    # fills are deliberately quiet so the three measured conditions remain the
    # only categorical colours in the panel.
    ax.axvspan(-0.55, 2.0, color="#F2F3F4", alpha=0.90, lw=0, zorder=-3)
    ax.axvspan(2.0, 4.55, color=COLORS["lavender"], alpha=0.34, lw=0, zorder=-3)
    ax.axhspan(0.30, 0.50, color=COLORS["peach"], alpha=0.35, lw=0, zorder=-2)
    ax.axhline(0.50, color=COLORS["ash"], lw=0.65, ls=(0, (2, 2)), zorder=1)
    ax.axvline(2.0, color=COLORS["ash"], lw=0.55, ls=(0, (2, 2)), zorder=1)

    ax.text(0.85, 0.61, "FAMILIAR\nONLY", ha="center", va="center",
            fontsize=5.15, linespacing=1.25, fontweight="semibold",
            color=COLORS["ash"])
    ax.text(3.25, 0.61, "FAMILIAR\n+ COMPOSED", ha="center", va="center",
            fontsize=5.15, linespacing=1.25, fontweight="semibold",
            color=COLORS["decoder"])
    ax.text(1.55, 0.345, "NEITHER", ha="center", va="bottom",
            fontsize=5.15, fontweight="semibold",
            color=COLORS["terracotta"])
    ax.text(4.44, 0.492, "chance", fontsize=5.1, color=COLORS["ash"],
            ha="right", va="top")

    means: dict[str, tuple[float, float]] = {}
    label_specs = {
        "Single\ntimescale": (0.16, -0.030, "left", "top", "single timescale"),
        "Multiscale": (-0.16, -0.030, "right", "top", "multiscale"),
        "Multiscale\nscrambled": (0.16, -0.035, "left", "top", "scrambled"),
    }
    for exposure, mode, readout, label, colour, marker in CONTRASTS:
        auc_values = _get(data, exposure, mode, readout, "auc")
        span_values = _get(data, exposure, mode, readout, "n_words_covered")
        ax.scatter(
            span_values,
            auc_values,
            s=8.0,
            marker=marker,
            color=colour,
            alpha=0.23,
            linewidth=0,
            zorder=3,
        )
        auc_mean, auc_sem = _seed_mean_sem(auc_values)
        span_mean, span_sem = _seed_mean_sem(span_values)
        ax.errorbar(
            span_mean,
            auc_mean,
            xerr=span_sem,
            yerr=auc_sem,
            fmt=marker,
            ms=5.4,
            color=colour,
            markeredgecolor=COLORS["white"],
            markeredgewidth=0.65,
            elinewidth=1.15,
            capsize=2.0,
            capthick=0.7,
            zorder=7,
        )
        dx, dy, ha, va, display = label_specs[label]
        ax.text(
            span_mean + dx,
            auc_mean + dy,
            display,
            ha=ha,
            va=va,
            fontsize=5.25,
            color=colour,
            fontweight="semibold",
            zorder=8,
        )
        means[label] = (span_mean, auc_mean)

    # The structured comparison is the causal visual: changing only the
    # readout timescales moves the representation from familiar-only to
    # familiar-and-composed, with little change in behavioural familiarity.
    single_x, single_y = means["Single\ntimescale"]
    multi_x, multi_y = means["Multiscale"]
    ax.annotate(
        "",
        xy=(multi_x - 0.18, multi_y + 0.030),
        xytext=(single_x + 0.18, single_y + 0.030),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.0,
            "color": COLORS["decoder"],
            "shrinkA": 0,
            "shrinkB": 0,
            "connectionstyle": "arc3,rad=-0.05",
        },
        annotation_clip=False,
        zorder=5,
    )
    ax.text(
        (single_x + multi_x) / 2,
        max(single_y, multi_y) + 0.075,
        "add multiscale context",
        ha="center",
        va="bottom",
        fontsize=5.35,
        color=COLORS["decoder"],
        fontweight="semibold",
    )

    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(0.30, 1.04)
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xlabel("Ordered composition (words represented)")
    ax.set_ylabel("Familiarity (AUC)", labelpad=2.0)
    clean_axis(ax)
    _panel_heading(
        panel_anchor,
        "H",
        "Multiscale context adds ordered composition",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "figure_5_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "exposure", "layer1_mode", "readout", "n_seeds", "auc_mean",
            "auc_sem", "words_in_order_mean", "words_in_order_sem",
            "detectors_mean",
        ])
        for exposure, mode, readout, _label, _colour, _marker in CONTRASTS:
            auc_values = _get(data, exposure, mode, readout, "auc")
            span_values = _get(data, exposure, mode, readout, "n_words_covered")
            det = _get(data, exposure, mode, readout, "n_word_detectors")
            auc_mean, auc_sem = _seed_mean_sem(auc_values)
            span_mean, span_sem = _seed_mean_sem(span_values)
            writer.writerow([exposure, mode, readout, auc_values.size, auc_mean,
                             auc_sem, span_mean, span_sem, det.mean()])
    return [ax]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_figure(*, force_data: bool = False) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_saffran_data(force=force_data, data_dir=DATA_DIR)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(245)))
        # A and B are one stacked unit on a shared stream clock, so they get
        # their own grid with a small internal gap.
        stack = fig.add_gridspec(
            2, 1, left=0.088, right=0.972, top=0.922, bottom=0.730,
            hspace=0.46, height_ratios=(0.62, 1.00),
        )
        _panel_stream(fig, stack[0, 0], data)
        _panel_tape(fig, stack[1, 0], data)

        # C and D need the full width: C shows four context-by-timescale
        # templates, while D compares the same three conditions on three
        # distinct scientific questions.
        grid = fig.add_gridspec(
            4, 12, left=0.088, right=0.972, top=0.670, bottom=0.040,
            wspace=0.92, hspace=0.92,
            height_ratios=(1.25, 1.00, 0.95, 0.95),
        )
        _panel_kernels(fig, grid[0, :], data)
        _panel_metrics(fig, grid[1, :], data)
        _panel_transitions(fig, grid[2, 0:6], data)
        _panel_filterbank(fig, grid[2, 7:12], data)
        _panel_controls(fig, grid[3, 0:6], data)
        _panel_synthesis(fig, grid[3, 7:12], data)
        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    provenance = {
        "figure": "Figure 5 - Saffran statistical learning and composition",
        "final_size_mm": [183, 245],
        "panel_sources": {
            "A": "exemplar structured exposure stream excerpt",
            "B": "late-exposure activity of one whole-word unit per item",
            "C": (
                "measured driving row of four whole-word masks, reshaped "
                "without reduction as context token by filter timescale"
            ),
            "D": (
                "per-seed familiarity, held-out unit selectivity, and "
                "ordered-composition metrics for matched conditions"
            ),
            "E": "designed transition matrix and learned layer-1 recurrent map",
            "F": "filterbank impulse response and elapsed-time code",
            "G": "layer-1 plastic, frozen, and absent controls",
            "H": (
                "familiarity-composition state space with the matched "
                "single-to-multiscale readout transition, per exposure seed"
            ),
        },
        "inference_scope": (
            "The replication unit is the exposure order seed (n = 8). Layer 2 "
            "masks start identically in every seed, so these are independent "
            "exposure sessions rather than independent networks."
        ),
        "inputs": {},
        "outputs": {},
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }
    for name in (
        "saffran_figure5_data.npz",
        "saffran_provenance.json",
        "saffran_figure5_runs.csv",
        "saffran_exemplar.npz",
        "saffran_exemplar_provenance.json",
    ):
        path = DATA_DIR / name
        if path.exists():
            provenance["inputs"][name] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
    for kind, path in paths.items():
        provenance["outputs"][kind] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    provenance_path = DATA_DIR / "figure_5_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    paths["provenance"] = provenance_path
    return paths


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Repeat every simulated exposure session.",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = _parse_args(arguments)
    paths = build_figure(force_data=args.force_data)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
