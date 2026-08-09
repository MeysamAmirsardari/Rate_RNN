"""Build the publication-scale probability-reversal sequence Figure 4.

The biological panels deliberately remain descriptive until the lossless
trial/block MATLAB export is available. Model uncertainty and inference use
paired seeds; no trial is treated as an independent replicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import ScaledTranslation

from ECoG.ab_ba.ab_ba_channel_erp import run as build_ab_ba_channel_erp

from final_figures.figure_4_ab_ba.ecog_data import (
    LegacyComparison,
    export_panel_data,
    load_reference,
    smooth_erp_for_display,
)
from final_figures.figure_4_ab_ba.model_data import (
    AB_BA_OVERRIDES,
    CONDITIONS,
    DISPLAY_PRE_MS,
    INTRA_GAP_MS,
    INTER_GAP_MS,
    P_REGULAR,
    ROLES,
    SEQUENCES,
    TARGET_ONSET_MS,
    TIMING_LONG,
    TIMING_SHORT,
    TONE_DURATION_MS,
    build as build_model_data,
)
from final_figures.style import (
    COLORS,
    clean_axis,
    export_figure,
    manuscript_style,
    mm,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_4_ab_ba"
ECOG_PANEL_PATH = DATA_DIR / "ecog_exp1_legacy_panel_data.csv.gz"
DISPLAYED_PANELS = ("A", "B", "C", "D", "E", "F")

TIME_LIMITS = (0.0, 600.0)
TIME_TICKS = (0.0, 180.0, 360.0, 600.0)
ROLE_COLORS = {"predicted": COLORS["rep15"], "unexpected": COLORS["rep1"]}
ROLE_LABELS = {"predicted": "regular", "unexpected": "rare"}
CONDITION_COLORS = {
    "intact": COLORS["decoder"],
    "no_depression": COLORS["terracotta"],
    "no_recurrent_learning": "#7E8792",
    "uniform_inhibition": COLORS["teal"],
}
CHARCOAL = COLORS["charcoal"]
MID_GREY = "#7E8792"
WINDOW_GREY = "#E5E7E9"
WINDOW_PEACH = "#F4D6C4"
TOKEN_COLORS = {"A": COLORS["decoder"], "B": COLORS["teal"]}
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "weight_linen_violet_oxblood",
    [COLORS["linen"], "#D6D0E1", COLORS["decoder"], COLORS["rep1"]],
    N=256,
)


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel_heading(
    ax,
    label: str,
    title: str,
    *,
    title_gap_pt: float = 14.0,
    label_gap_pt: float = 8.0,
) -> None:
    """Use Figure 2's physical panel-header geometry throughout."""

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
        color=CHARCOAL,
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
        color=CHARCOAL,
        clip_on=False,
    )


def _mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    sem = values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])
    return mean, sem


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _significance_rail(
    ax,
    time: np.ndarray,
    mask: np.ndarray,
    *,
    y_axes: float = 0.95,
    color: str = CHARCOAL,
) -> None:
    """Draw restrained cluster-FWER rails in the data frame."""

    transform = ax.get_xaxis_transform()
    for start, stop in _segments(mask):
        ax.plot(
            [time[start], time[stop]],
            [y_axes, y_axes],
            transform=transform,
            color=color,
            lw=0.68,
            solid_capstyle="butt",
            clip_on=True,
            zorder=12,
        )


def _tone_context(
    ax,
    *,
    labels: tuple[str, str] | None = None,
    first_stop: float = 180.0,
    second_start: float = 180.0,
    second_stop: float = 360.0,
) -> None:
    ax.axvspan(0, first_stop, color=WINDOW_GREY, alpha=0.56, lw=0, zorder=-20)
    ax.axvspan(
        second_start,
        second_stop,
        color=WINDOW_PEACH,
        alpha=0.55,
        lw=0,
        zorder=-20,
    )
    ax.axvline(second_start, color="#9BA2AA", lw=0.48, ls=(0, (2, 2)), zorder=0)
    if labels is not None:
        trans = ax.get_xaxis_transform()
        centres = (first_stop / 2.0, (second_start + second_stop) / 2.0)
        for centre, label in zip(centres, labels):
            ax.text(
                centre,
                0.055,
                label,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=5.4,
                fontweight="semibold",
                color="#707984",
                clip_on=True,
            )


def _format_time_axis(ax, *, xlabel: bool = True) -> None:
    ax.set_xlim(*TIME_LIMITS)
    ax.set_xticks(TIME_TICKS)
    if xlabel:
        ax.set_xlabel("sequence time (ms)")
    else:
        ax.tick_params(labelbottom=False)
    clean_axis(ax)


def _sequence_icon(
    ax,
    x: float,
    y: float,
    sequence: str,
    *,
    role: str,
    scale: float = 1.0,
) -> None:
    role_color = ROLE_COLORS[role]
    box_w = 0.046 * scale
    box_h = 0.16 * scale
    gap = 0.010 * scale
    for index, token in enumerate(sequence):
        x0 = x + index * (box_w + gap)
        patch = FancyBboxPatch(
            (x0, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="#FBFAF8",
            edgecolor=role_color,
            linewidth=1.0 if role == "predicted" else 1.25,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x0 + box_w / 2,
            y,
            token,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.8 * scale,
            fontweight="bold",
            color=TOKEN_COLORS[token],
        )
    ax.plot(
        [x + box_w, x + box_w + gap],
        [y, y],
        transform=ax.transAxes,
        color=role_color,
        lw=0.75,
        clip_on=False,
    )


def panel_a(fig, spec) -> plt.Axes:
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    _panel_heading(ax, "A", "", title_gap_pt=-4.0)
    # Deliberately blank: the final paradigm illustration will be assembled
    # independently. Keeping only the panel tag preserves the page geometry.
    return ax


def _ecog_facet_title(comparison: LegacyComparison) -> str:
    if comparison.sequence_code == "AB":
        return "AB  |  5.3 to 9.4 kHz"
    return "BA  |  9.4 to 5.3 kHz"


AB_BA_INFERENCE = (
    Path(__file__).resolve().parents[2]
    / "ECoG" / "ab_ba" / "results" / "ab_ba_inference" / "ab_ba_inference.npz"
)
AB_BA_CHANNEL_ERP = (
    Path(__file__).resolve().parents[2]
    / "ECoG"
    / "ab_ba"
    / "results"
    / "ab_ba_channel_erp"
    / "ab_ba_channel_erp.npz"
)
AB_BA_CHANNEL_ERP_PROVENANCE = AB_BA_CHANNEL_ERP.with_name("provenance.json")
#: Physical sequence -> facet label. Each appears as the 15% deviant in one
#: recording and the 85% standard in the other, so the contrast is acoustically
#: identity-controlled.
SEQUENCE_FACETS = (
    ("5300_9400", "AB"),
    ("9400_5300", "BA"),
)


def panel_b(fig, spec, comparisons: tuple[LegacyComparison, ...]) -> list[plt.Axes]:
    """Held-fixed AB and BA ERPs with independent contact selection.

    Odd acquisition blocks select a contact in the prespecified second-item
    window; even blocks estimate and test it. The modest 2-ms display
    smoothing is already applied in the committed analysis artifact, while
    its cluster test uses the unsmoothed held-out block means.
    """

    with np.load(AB_BA_CHANNEL_ERP, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    time_ms = np.asarray(data["time_ms"], dtype=float)
    heading_axis = fig.add_subplot(spec)
    heading_axis.axis("off")
    inner = spec.subgridspec(
        2,
        2,
        width_ratios=(0.17, 1.0),
        hspace=0.13,
        wspace=0.05,
    )
    label_axes = [fig.add_subplot(inner[index, 0]) for index in range(2)]
    axes = [fig.add_subplot(inner[index, 1]) for index in range(2)]

    bounds = []
    for key, _ in SEQUENCE_FACETS:
        for role in ("regular", "rare"):
            mean = np.asarray(data[f"{key}_{role}_mean"], dtype=float)
            sem = np.asarray(data[f"{key}_{role}_sem"], dtype=float)
            bounds.extend((mean - sem, mean + sem))
    all_bounds = np.concatenate(bounds)
    lo, hi = float(all_bounds.min()), float(all_bounds.max())
    margin = 0.12 * max(hi - lo, 0.1)

    for index, (label_ax, ax, (key, sequence)) in enumerate(
        zip(label_axes, axes, SEQUENCE_FACETS)
    ):
        labels = (
            ("5.3 kHz", "9.4 kHz")
            if key == "5300_9400"
            else ("9.4 kHz", "5.3 kHz")
        )
        _tone_context(
            ax,
            labels=labels,
            first_stop=180.0,
            second_start=180.0,
            second_stop=360.0,
        )
        ax.axhline(0, color="#A9AFB5", lw=0.46, zorder=0)
        for role, colour in (
            ("regular", ROLE_COLORS["predicted"]),
            ("rare", ROLE_COLORS["unexpected"]),
        ):
            mean = np.asarray(data[f"{key}_{role}_mean"], dtype=float)
            sem = np.asarray(data[f"{key}_{role}_sem"], dtype=float)
            ax.fill_between(
                time_ms,
                mean - sem,
                mean + sem,
                color=colour,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            ax.plot(time_ms, mean, color=colour, lw=1.08, label=role, zorder=3)
        significant = np.asarray(data[f"{key}_significant"], dtype=bool)
        _significance_rail(
            ax,
            time_ms,
            significant,
            y_axes=0.94,
            color=CHARCOAL,
        )
        channel = int(data[f"{key}_channel_matlab"])
        label_ax.axis("off")
        label_ax.text(
            0.22,
            0.50,
            f"{sequence}\ncontact {channel}",
            transform=label_ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="semibold",
            linespacing=1.28,
            color=CHARCOAL,
        )
        ax.set_xlim(0.0, 600.0)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xticks((0.0, 180.0, 360.0, 600.0))
        clean_axis(ax)
        if index == 0:
            ax.tick_params(labelbottom=False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.0, 1.27),
                ncol=2,
                handlelength=1.35,
                columnspacing=0.7,
                borderaxespad=0,
            )
        else:
            ax.set_xlabel("Sequence time (ms)")
    heading_axis.text(
        -0.095,
        0.48,
        "ECoG response (a.u.)",
        transform=heading_axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.2,
        color=CHARCOAL,
        clip_on=False,
    )
    _panel_heading(heading_axis, "B", "ECoG response")
    return axes


def _panel_b_legacy(fig, spec, comparisons: tuple[LegacyComparison, ...]) -> list[plt.Axes]:
    inner = spec.subgridspec(1, 2, wspace=0.18)
    axes = [fig.add_subplot(inner[0, index]) for index in range(2)]
    smoothed = []
    for comparison in comparisons:
        smoothed.extend(
            (
                smooth_erp_for_display(comparison.time_ms, comparison.predicted),
                smooth_erp_for_display(comparison.time_ms, comparison.unexpected),
            )
        )
    crop_values = np.concatenate(
        [values[(comparison.time_ms >= 0) & (comparison.time_ms <= 600)]
         for comparison, values in zip(np.repeat(comparisons, 2), smoothed)]
    )
    lo, hi = float(crop_values.min()), float(crop_values.max())
    margin = 0.10 * max(hi - lo, 0.1)

    for index, (ax, comparison) in enumerate(zip(axes, comparisons)):
        regular = smooth_erp_for_display(comparison.time_ms, comparison.predicted)
        rare = smooth_erp_for_display(comparison.time_ms, comparison.unexpected)
        tone_labels = (
            ("5.3 kHz", "9.4 kHz")
            if comparison.sequence_code == "AB"
            else ("9.4 kHz", "5.3 kHz")
        )
        _tone_context(ax, labels=tone_labels)
        ax.axhline(0, color="#A9AFB5", lw=0.48, zorder=0)
        ax.plot(
            comparison.time_ms,
            regular,
            color=ROLE_COLORS["predicted"],
            lw=1.15,
            label="regular",
        )
        ax.plot(
            comparison.time_ms,
            rare,
            color=ROLE_COLORS["unexpected"],
            lw=1.15,
            label="rare",
        )
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_title(_ecog_facet_title(comparison), loc="left", fontsize=6.6, pad=3)
        _format_time_axis(ax)
        if index == 0:
            ax.set_ylabel("normalized response (a.u.)")
            ax.legend(loc="upper right", ncol=2, handlelength=1.35, columnspacing=0.7)
        else:
            ax.tick_params(labelleft=False)

    _panel_tag(axes[0], "B", x=-0.13, y=1.34)
    _heading(axes[0], "Same-sequence ECoG response", y=1.25)
    return axes


MATLAB_REPLICATION = (
    Path(__file__).resolve().parents[2] / "ECoG" / "ab_ba" / "results"
)
#: Facet -> replication directory. Each is a faithful rerun of
#: ``scripts_AB_BA.m`` on one of its two class assignments; the source saved a
#: figure for each, named after the deviant sequence.
DECODER_FACETS = (
    ("AB", "matlab_replication_AB", "AB  |  5.3 to 9.4 kHz"),
    ("BA", "matlab_replication_BA", "BA  |  9.4 to 5.3 kHz"),
)


def panel_c(fig, spec, comparisons: tuple[LegacyComparison, ...]) -> list[plt.Axes]:
    """Time-resolved Rep-1 versus Rep-15 decoding, as the source computes it.

    This is a replication of ``scripts_AB_BA.m``: 1 kHz, a 0-1360 ms window
    (``seqDur + 1000``), ridge logistic regression at ``Lambda = 1e-2``, plain
    five-fold accuracy and a 20-sample moving mean. The source's leaks are
    reproduced rather than corrected -- all trials are z-scored before
    cross-validation, and the two classes come from different recordings -- so
    the curve is descriptive and carries no interval.
    """

    inner = spec.subgridspec(1, 2, wspace=0.18)
    axes = [fig.add_subplot(inner[0, index]) for index in range(2)]

    for index, (ax, (_key, folder, title)) in enumerate(zip(axes, DECODER_FACETS)):
        data = np.load(MATLAB_REPLICATION / folder / "matlab_replication.npz")
        accuracy = np.asarray(data["smoothed"], dtype=float)
        time_ms = np.arange(1, accuracy.size + 1, dtype=float)
        note_ms = float(data["note_ms"])
        peak = int(data["peak_index"])

        _tone_context(ax, first_stop=note_ms, second_start=note_ms,
                      second_stop=2 * note_ms)
        ax.axhline(0.5, color="#8D949C", lw=0.65, ls=(0, (2, 2)), zorder=0)
        ax.plot(time_ms, accuracy, color=COLORS["decoder"], lw=1.15)
        ax.plot(time_ms[peak], accuracy[peak], marker="o", ms=2.6,
                color=COLORS["decoder"], zorder=5)
        ax.annotate(f"{accuracy[peak]:.2f} at {int(time_ms[peak])} ms",
                    (time_ms[peak], accuracy[peak]),
                    textcoords="offset points", xytext=(4, 4), fontsize=5.4,
                    color=COLORS["decoder"])

        ax.set_xlim(0, time_ms[-1])
        ax.set_xticks((0.0, 360.0, 800.0, 1200.0))
        ax.set_ylim(0.38, 0.86)
        ax.set_yticks((0.4, 0.5, 0.6, 0.7, 0.8))
        ax.set_xlabel("sequence time (ms)")
        ax.set_title(title, loc="left", fontsize=6.6, pad=3)
        clean_axis(ax)
        if index == 0:
            ax.set_ylabel("decoding accuracy")
            ax.text(1340, 0.505, "chance", ha="right", va="bottom",
                    fontsize=5.3, color="#747C85")
        else:
            ax.tick_params(labelleft=False)

    axes[0].text(
        0.0, 1.06,
        "replication of scripts_AB_BA.m - 5-fold accuracy, 20-sample moving "
        "mean - descriptive, no interval",
        transform=axes[0].transAxes, ha="left", va="bottom", fontsize=5.4,
        color=COLORS["ash"], clip_on=False,
    )
    _panel_heading(axes[0], "C", "Rep 1 versus Rep 15 decoding")
    return axes


def _panel_c_legacy(fig, spec, comparisons: tuple[LegacyComparison, ...]) -> list[plt.Axes]:
    inner = spec.subgridspec(1, 2, wspace=0.18)
    axes = [fig.add_subplot(inner[0, index]) for index in range(2)]
    for index, (ax, comparison) in enumerate(zip(axes, comparisons)):
        tone_labels = (
            ("5.3 kHz", "9.4 kHz")
            if comparison.sequence_code == "AB"
            else ("9.4 kHz", "5.3 kHz")
        )
        _tone_context(ax, labels=tone_labels)
        ax.axhline(0.5, color="#8D949C", lw=0.65, ls=(0, (2, 2)), zorder=0)
        ax.plot(
            comparison.decoder_time_ms,
            comparison.decoder_accuracy,
            color=COLORS["decoder"],
            lw=1.15,
        )
        ax.set_ylim(0.38, 0.82)
        ax.set_yticks((0.4, 0.5, 0.6, 0.7, 0.8))
        ax.set_title(_ecog_facet_title(comparison), loc="left", fontsize=6.6, pad=3)
        _format_time_axis(ax)
        if index == 0:
            ax.set_ylabel("balanced accuracy")
            ax.text(
                586,
                0.505,
                "chance",
                ha="right",
                va="bottom",
                fontsize=5.3,
                color="#747C85",
            )
        else:
            ax.tick_params(labelleft=False)

    _panel_tag(axes[0], "C", x=-0.17, y=1.34)
    _heading(axes[0], "Same-sequence context decoding", y=1.25)
    return axes


def panel_d(fig, spec, model: dict[str, np.ndarray]) -> list[plt.Axes]:
    heading_axis = fig.add_subplot(spec)
    heading_axis.axis("off")
    inner = spec.subgridspec(
        1,
        4,
        width_ratios=(1.30, 1.30, 0.74, 0.74),
        wspace=0.34,
    )
    trace_axes = [fig.add_subplot(inner[0, index]) for index in range(2)]
    matrix_axes = [fig.add_subplot(inner[0, index]) for index in (2, 3)]
    trajectory = np.asarray(model["weight_trajectory"], dtype=float)
    checkpoints = np.asarray(model["weight_checkpoints"], dtype=float)

    preferred_index = ((1, 0), (0, 1))
    reverse_index = ((0, 1), (1, 0))
    ymax = 0.0
    for context_index, ax in enumerate(trace_axes):
        pref = trajectory[context_index, :, :, preferred_index[context_index][0], preferred_index[context_index][1]]
        reverse = trajectory[context_index, :, :, reverse_index[context_index][0], reverse_index[context_index][1]]
        self_weight = 0.5 * (
            trajectory[context_index, :, :, 0, 0]
            + trajectory[context_index, :, :, 1, 1]
        )
        for values, color, label, linestyle in (
            (pref, COLORS["decoder"], "frequent link", "-"),
            (reverse, COLORS["teal"], "rare link", "-"),
            (self_weight, MID_GREY, "self-link", (0, (3, 2))),
        ):
            mean, sem = _mean_sem(values, axis=0)
            ax.fill_between(
                checkpoints,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.13,
                lw=0,
            )
            ax.plot(checkpoints, mean, color=color, lw=1.18, ls=linestyle, label=label)
            ymax = max(ymax, float(np.max(mean + sem)))
        ax.set_title(
            "AB-rich context" if context_index == 0 else "BA-rich context",
            loc="left",
            fontsize=6.6,
            pad=3,
        )
        ax.set_xlim(checkpoints[0], checkpoints[-1])
        ax.set_xticks((0, 200, 400))
        ax.set_xlabel("Training sequence")
        clean_axis(ax)
        if context_index == 0:
            ax.set_ylabel("recurrent weight")
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(1.31, 1.015),
                ncol=3,
                handlelength=1.35,
                columnspacing=0.75,
                fontsize=5.6,
            )
        else:
            ax.tick_params(labelleft=False)

    shared_ymax = max(0.02, ymax * 1.12)
    for ax in trace_axes:
        ax.set_ylim(-0.006, shared_ymax)

    terminal = trajectory[:, :, -1].mean(axis=1)
    # Both terminal matrices use one fixed, interpretable range.  The values
    # are written into every cell, so a second colour-scale axis would add
    # visual weight without adding information.
    vmax = max(0.20, float(np.max(terminal)) * 1.03)
    for context_index, ax in enumerate(matrix_axes):
        ax.imshow(
            terminal[context_index],
            vmin=0,
            vmax=vmax,
            cmap=WEIGHT_CMAP,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks((0, 1), labels=("A", "B"))
        ax.set_yticks((0, 1), labels=("A", "B"))
        ax.set_xlabel("pre", labelpad=1)
        if context_index == 0:
            ax.set_ylabel("post", labelpad=1)
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(
            "AB-rich final weights" if context_index == 0 else "BA-rich final weights",
            fontsize=6.2,
            pad=3,
        )
        for row in range(2):
            for column in range(2):
                value = terminal[context_index, row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color="white" if value > 0.52 * vmax else CHARCOAL,
                )
        ax.tick_params(length=0, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#C6C9CC")

    _panel_heading(heading_axis, "D", "Learning the frequent transition")
    return trace_axes + matrix_axes


#: Sequence colours for the mechanism panel: the bound set keeps the model
#: colour, the reversed order gets the readout hue.
SEQ_COLORS = {"AB": COLORS["model"], "BA": COLORS["decoder"]}
CURRENT_COLORS = {"recurrent": COLORS["model"], "inhibitory": COLORS["terracotta"]}


def _target_context(ax, *, labels: bool = True) -> None:
    """Shade the two tones. Time zero is the onset of the second tone.

    The first tone runs from ``-TARGET_ONSET_MS`` for ``TONE_DURATION_MS``, so
    the silent intra-pair gap is the unshaded strip just before zero. That gap
    is where the prediction lives, which is why it is left visible rather than
    being folded into either tone.
    """

    first_start = -TARGET_ONSET_MS
    ax.axvspan(first_start, first_start + TONE_DURATION_MS, color=WINDOW_GREY,
               alpha=0.56, lw=0, zorder=-20)
    ax.axvspan(0, TONE_DURATION_MS, color=WINDOW_PEACH, alpha=0.55, lw=0,
               zorder=-20)
    ax.axvline(0, color="#9BA2AA", lw=0.48, ls=(0, (2, 2)), zorder=0)
    if labels:
        ax.text(
            first_start + TONE_DURATION_MS / 2,
            0.055,
            "tone 1",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#747C85",
            clip_on=True,
        )
        ax.text(
            TONE_DURATION_MS / 2,
            0.055,
            "tone 2",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#747C85",
            clip_on=True,
        )


def _mechanism_axis(ax, time_ms: np.ndarray, *, ylabel: str,
                    xlabel: bool = False, headroom: float = 0.30) -> None:
    ax.set_xlim(-TARGET_ONSET_MS - 40.0, 320.0)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + headroom * (high - low))
    ax.set_xticks((-TARGET_ONSET_MS, 0.0, 100.0, 200.0, 300.0))
    ax.set_ylabel(ylabel, labelpad=1.5)
    if xlabel:
        ax.set_xlabel("time from second-tone onset (ms)")
    else:
        ax.tick_params(labelbottom=False)
    clean_axis(ax)


def panel_mechanism(fig, spec, model: dict[str, np.ndarray]) -> list[plt.Axes]:
    """Why the target is suppressed, channel by channel.

    Left and middle columns hold the acoustics fixed per channel and vary the
    order: channel A leads in AB and lags in BA. The informative feature is the
    small rise in channel B *during the first tone* of an AB sequence - the
    learned A->B link pre-activating the target before it sounds - and the
    inhibitory current that this pre-activation recruits.

    The right column is the identity-controlled contrast: the same physical
    sequences, regular against rare, averaged over all channels, and their
    difference.
    """

    time_ms = np.asarray(model["time_ms"], dtype=float)
    sequences = [str(s) for s in np.asarray(model["sequences"])]
    channel_response = np.asarray(model["channel_response"], dtype=float)
    channel_recurrent = np.asarray(model["channel_recurrent"], dtype=float)
    channel_inhibitory = np.asarray(model["channel_inhibitory"], dtype=float)
    population = np.asarray(model["population_response"], dtype=float)[0]
    surprise = np.asarray(model["population_surprise_response"], dtype=float)
    significant = np.asarray(model["population_time_significant"], dtype=bool)

    heading_axis = fig.add_subplot(spec)
    heading_axis.axis("off")
    inner = spec.subgridspec(2, 3, hspace=0.30, wspace=0.42)
    axes: list[plt.Axes] = []

    # --- columns 1-2: one channel each, both sequence orders ----------------
    for column, channel in enumerate((0, 1)):
        name = "A" if channel == 0 else "B"

        rate = fig.add_subplot(inner[0, column])
        for index, sequence in enumerate(sequences):
            # Average the two roles: this column is about order, not surprise.
            traces = channel_response[index, :, :, channel].mean(axis=0)
            mean, sem = _mean_sem(traces)
            rate.fill_between(time_ms, mean - sem, mean + sem,
                              color=SEQ_COLORS[sequence], alpha=0.20, lw=0)
            rate.plot(time_ms, mean, color=SEQ_COLORS[sequence], lw=1.15,
                      ls="-" if sequence == "AB" else (0, (3, 2)),
                      label=f"{sequence} sequence")
        _target_context(rate, labels=True)
        _mechanism_axis(rate, time_ms, ylabel=f"$E_{name}$ (a.u.)",
                        headroom=0.58)
        rate.set_title(f"Population {name} activity", loc="left", fontsize=6.6,
                       pad=3)
        if column == 0:
            rate.legend(
                loc="upper right",
                bbox_to_anchor=(1.02, 1.04),
                handlelength=1.2,
                ncol=2,
                columnspacing=0.8,
            )
        axes.append(rate)

        current = fig.add_subplot(inner[1, column])
        current.axhline(0.0, color=MID_GREY, lw=0.5, zorder=0)
        for index, sequence in enumerate(sequences):
            style = "-" if sequence == "AB" else (0, (3, 2))
            for key, source, sign in (("recurrent", channel_recurrent, 1.0),
                                      ("inhibitory", channel_inhibitory, -1.0)):
                mean, _ = _mean_sem(source[index, :, :, channel].mean(axis=0))
                current.plot(time_ms, sign * mean, color=CURRENT_COLORS[key],
                             lw=1.05, ls=style,
                             label=(f"{'recurrent exc.' if sign > 0 else 'inhibition'}"
                                    if sequence == "AB" else None))
        _target_context(current, labels=False)
        _mechanism_axis(current, time_ms, ylabel="current (a.u.)", xlabel=True)
        current.set_title(f"Population {name} synaptic drive", loc="left",
                          fontsize=6.6, pad=3)
        if column == 0:
            current.legend(loc="upper right", bbox_to_anchor=(1.02, 1.04),
                           handlelength=1.3, ncol=2, columnspacing=0.8)
        axes.append(current)

    # --- column 3: the identity-controlled contrast -------------------------
    rate = fig.add_subplot(inner[0, 2])
    for role_index, (role, colour) in enumerate(
            (("regular", ROLE_COLORS["predicted"]),
             ("rare", ROLE_COLORS["unexpected"]))):
        traces = population[:, role_index].mean(axis=0)
        mean, sem = _mean_sem(traces)
        rate.fill_between(time_ms, mean - sem, mean + sem, color=colour,
                          alpha=0.20, lw=0)
        rate.plot(time_ms, mean, color=colour, lw=1.15, label=role)
    _target_context(rate, labels=True)
    _mechanism_axis(rate, time_ms, ylabel=r"$\langle E \rangle$ (a.u.)",
                    headroom=0.58)
    rate.set_title("Mean excitatory activity", loc="left", fontsize=6.6, pad=3)
    rate.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), handlelength=1.3)
    axes.append(rate)

    difference = fig.add_subplot(inner[1, 2])
    mean, sem = _mean_sem(surprise.mean(axis=0))
    difference.axhline(0.0, color=MID_GREY, lw=0.5, zorder=0)
    difference.fill_between(time_ms, 0, mean, where=mean >= 0,
                            color=ROLE_COLORS["unexpected"], alpha=0.28, lw=0,
                            interpolate=True)
    difference.fill_between(time_ms, 0, mean, where=mean <= 0,
                            color=ROLE_COLORS["predicted"], alpha=0.28, lw=0,
                            interpolate=True)
    difference.fill_between(time_ms, mean - sem, mean + sem,
                            color=COLORS["decoder"], alpha=0.22, lw=0)
    difference.plot(time_ms, mean, color=COLORS["decoder"], lw=1.25)
    _significance_rail(difference, time_ms, significant.all(axis=0),
                       y_axes=0.95, color=CHARCOAL)
    _target_context(difference, labels=False)
    _mechanism_axis(difference, time_ms, ylabel=r"$\Delta \langle E \rangle$",
                    xlabel=True)
    difference.set_title("Context effect (rare - regular)", loc="left", fontsize=6.6,
                         pad=3)
    axes.append(difference)

    _panel_heading(heading_axis, "E", "Sequence-evoked network dynamics")
    return axes


def _panel_e_target(fig, spec, model: dict[str, np.ndarray]) -> list[plt.Axes]:
    inner = spec.subgridspec(
        2,
        2,
        height_ratios=(1.10, 0.90),
        wspace=0.14,
        hspace=0.18,
    )
    axes = np.asarray(
        [[fig.add_subplot(inner[row, column]) for column in range(2)] for row in range(2)]
    )
    time = np.asarray(model["time_ms"], dtype=float) + DISPLAY_PRE_MS
    # Target-channel activity is the channel-resolved analogue of the source
    # diagnostic plot, while preserving the manuscript's held-fixed contrast:
    # B is compared across contexts for physical AB and A for physical BA.
    response = np.asarray(model["response"], dtype=float)[0]
    difference = np.asarray(model["surprise_response"], dtype=float)
    significance = np.asarray(model["time_significant"], dtype=bool)
    role_names = [str(item) for item in model["roles"]]

    raw_mean = response.mean(axis=2)
    raw_sem = response.std(axis=2, ddof=1) / np.sqrt(response.shape[2])
    raw_top = float(np.max(raw_mean + raw_sem)) * 1.09
    diff_mean = difference.mean(axis=1)
    diff_sem = difference.std(axis=1, ddof=1) / np.sqrt(difference.shape[1])
    diff_limit = 1.08 * float(
        np.max(np.abs(np.r_[diff_mean - diff_sem, diff_mean + diff_sem]))
    )

    for sequence_index, sequence in enumerate(SEQUENCES):
        tokens = tuple(sequence)
        upper = axes[0, sequence_index]
        lower = axes[1, sequence_index]
        _tone_context(upper, labels=tokens)
        _tone_context(lower)
        for role_index, role in enumerate(role_names):
            color = ROLE_COLORS[role]
            mean = raw_mean[sequence_index, role_index]
            sem = raw_sem[sequence_index, role_index]
            upper.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.13, lw=0)
            upper.plot(time, mean, color=color, lw=1.22, label=ROLE_LABELS[role])
        upper.set_ylim(-0.08, raw_top)
        upper.set_title(
            f"{sequence} held fixed  |  channel {'B' if sequence == 'AB' else 'A'}",
            loc="left",
            fontsize=6.7,
            pad=3,
        )
        _format_time_axis(upper, xlabel=False)
        upper.spines["bottom"].set_visible(False)
        upper.tick_params(bottom=False)
        if sequence_index == 0:
            upper.set_ylabel("target E rate")
            upper.legend(loc="upper right", ncol=2, handlelength=1.35, columnspacing=0.7)
        else:
            upper.tick_params(labelleft=False)

        mean = diff_mean[sequence_index]
        sem = diff_sem[sequence_index]
        lower.axhline(0, color="#8D949B", lw=0.55, zorder=0)
        lower.fill_between(time, 0, mean, where=mean >= 0, color=COLORS["terracotta"], alpha=0.26, lw=0)
        lower.fill_between(time, 0, mean, where=mean < 0, color=COLORS["rep15"], alpha=0.18, lw=0)
        lower.fill_between(time, mean - sem, mean + sem, color=COLORS["decoder"], alpha=0.13, lw=0)
        lower.plot(time, mean, color=COLORS["decoder"], lw=1.18)
        _significance_rail(
            lower,
            time,
            significance[sequence_index],
            y_axes=0.95,
            color=COLORS["decoder"],
        )
        lower.set_ylim(-diff_limit, diff_limit)
        _format_time_axis(lower)
        if sequence_index == 0:
            lower.set_ylabel("rare - regular")
        else:
            lower.tick_params(labelleft=False)

    _panel_tag(axes[0, 0], "E", x=-0.105, y=1.35)
    _heading(axes[0, 0], "Context reshapes target-population responses", y=1.26)
    return list(axes.ravel())


def panel_f(fig, spec, model: dict[str, np.ndarray]) -> list[plt.Axes]:
    heading_axis = fig.add_subplot(spec)
    heading_axis.axis("off")
    perturbation_ax = fig.add_subplot(spec)
    differences = np.asarray(model["condition_effect"], dtype=float)
    means = np.asarray(model["condition_effect_mean"], dtype=float)
    lows = np.asarray(model["condition_effect_ci_low"], dtype=float)
    highs = np.asarray(model["condition_effect_ci_high"], dtype=float)
    probabilities = np.asarray(model["lesion_vs_intact_p_fwer"], dtype=float)
    labels = ("Intact", "No\ndepression", "Plasticity\nfrozen", "Uniform\ninhibition")
    x_positions = np.arange(len(labels), dtype=float)
    rng = np.random.default_rng(12)
    for seed_index in range(differences.shape[1]):
        perturbation_ax.plot(
            x_positions,
            differences[:, seed_index],
            color="#C7CBD0",
            lw=0.48,
            alpha=0.34,
            zorder=1,
        )
    for row, (condition, x) in enumerate(zip(CONDITIONS, x_positions)):
        color = CONDITION_COLORS[condition]
        jitter = rng.uniform(-0.10, 0.10, size=differences.shape[1])
        perturbation_ax.scatter(
            x + jitter,
            differences[row],
            s=7,
            color=color,
            alpha=0.33,
            edgecolors="none",
            zorder=2,
        )
        perturbation_ax.plot([x, x], [lows[row], highs[row]], color=color, lw=1.15, zorder=4)
        perturbation_ax.plot([x - 0.055, x + 0.055], [lows[row], lows[row]], color=color, lw=0.75)
        perturbation_ax.plot([x - 0.055, x + 0.055], [highs[row], highs[row]], color=color, lw=0.75)
        perturbation_ax.scatter(
            x,
            means[row],
            marker="D",
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    perturbation_ax.axhline(0, color="#8B929A", lw=0.65, ls=(0, (2, 2)), zorder=0)
    ylow = float(min(np.min(differences), np.min(lows)))
    yhigh = float(max(np.max(differences), np.max(highs)))
    span = max(yhigh - ylow, 0.05)
    upper = yhigh + 0.30 * span
    perturbation_ax.set_ylim(ylow - 0.13 * span, upper)
    for row, x in enumerate(x_positions[1:]):
        y = max(highs[row + 1], means[row + 1]) + 0.055 * span
        p_value = probabilities[row]
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        perturbation_ax.text(
            x,
            y,
            stars,
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="semibold",
            color=CHARCOAL,
        )
    perturbation_ax.set_xlim(-0.72, 3.72)
    perturbation_ax.set_xticks(x_positions, labels=labels)
    perturbation_ax.set_ylabel("Rare - regular target response")
    perturbation_ax.tick_params(axis="x", length=0)
    clean_axis(perturbation_ax)

    _panel_heading(heading_axis, "F", "Causal perturbations")
    return [perturbation_ax]


def build_figure(*, force_data: bool = False) -> tuple[plt.Figure, dict[str, Path]]:
    build_ab_ba_channel_erp(force=force_data)
    all_comparisons = load_reference()
    comparisons = tuple(all_comparisons[:2])
    if len(comparisons) != 2 or {item.family for item in comparisons} != {"Continuous tones"}:
        raise AssertionError("Figure 4 requires only the two Experiment-1 zero-gap comparisons")
    model = build_model_data(force=force_data)
    export_panel_data(comparisons, ECOG_PANEL_PATH)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(225)), constrained_layout=False)
        outer = fig.add_gridspec(
            5,
            12,
            left=0.057,
            right=0.982,
            top=0.978,
            bottom=0.045,
            hspace=0.56,
            wspace=0.78,
            height_ratios=(0.40, 1.42, 0.98, 1.58, 1.13),
        )

        panel_a(fig, outer[0, :])
        panel_b(fig, outer[1, 0:5], comparisons)
        panel_c(fig, outer[1, 6:12], comparisons)
        panel_d(fig, outer[2, :], model)
        panel_mechanism(fig, outer[3, :], model)
        panel_f(fig, outer[4, :], model)

        outputs = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)

    metadata = {
        "figure": "Figure 4",
        "title": "Probability-reversal sequence oddball",
        "displayed_panels": list(DISPLAYED_PANELS),
        "configuration": {
            "ab_ba_overrides": dict(AB_BA_OVERRIDES),
            "p_regular": P_REGULAR,
            "p_rare": 1.0 - P_REGULAR,
            "timing_long_seconds": dict(TIMING_LONG),
            "timing_short_seconds": dict(TIMING_SHORT),
            "tone_duration_ms": TONE_DURATION_MS,
            "target_onset_ms": TARGET_ONSET_MS,
            "inter_sequence_gap_ms": INTER_GAP_MS,
        },
        "ecog": {
            "experiment": "Experiment 1 only: continuous 180-ms tones, zero intra-pair gap",
            "comparisons": [item.key for item in comparisons],
            "status": "raw Open Ephys reanalysis with recovered Baphy playback labels",
            "planned_probability_each_run": {"regular": 0.85, "rare": 0.15},
            "data_contract": "six allM2 metadata tags precede neural samples; tags are never treated as time",
            "panel_b": (
                "odd acquisition blocks select one contact per physical sequence; "
                "even blocks estimate mean and SEM; exact two-sided block-label "
                "cluster mass controls FWER jointly over AB, BA and 0-600 ms"
            ),
            "display_smoothing": (
                "symmetric zero-phase Gaussian sigma 2 ms, display only; "
                "inference uses unsmoothed held-out block means"
            ),
            "scope": (
                "two recordings from one animal provisionally treated as one session "
                "at the experimenter's instruction; inference remains conditional on "
                "that assumption and does not support animal-population claims"
            ),
        },
        "model": {
            "replication_unit": "paired simulation seed; n=12",
            "uncertainty": "mean +/- SEM for time courses; 95% t intervals for scalar estimates",
            "time_inference": (
                "exact paired sign flips; cluster mass; family-wise correction jointly over "
                "physical sequence identities and the full 0-600-ms clock"
            ),
            "scalar_inference": "exact paired sign flips with max-|t| correction within each planned family",
            "leakage_guard": (
                "separate context training; identical balanced held-out streams; plasticity off at test; "
                "trials averaged within seed before inference"
            ),
        },
        "data_files": {
            "ecog_panel": str(ECOG_PANEL_PATH),
            "ecog_channel_erp": str(AB_BA_CHANNEL_ERP),
            "ecog_channel_erp_provenance": str(AB_BA_CHANNEL_ERP_PROVENANCE),
            "model_npz": str(DATA_DIR / "model_figure4_ab_ba.npz"),
            "model_provenance": str(DATA_DIR / "model_figure4_provenance.json"),
            "model_inference": str(DATA_DIR / "figure_4_inference.csv"),
        },
        "outputs": {kind: {"path": str(path), "sha256": _sha256(path)} for kind, path in outputs.items()},
    }
    metadata_path = OUTPUT_DIR / "figure_4_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    outputs["metadata"] = metadata_path
    return fig, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="recompute paired model seeds even when the provenance-keyed cache is current",
    )
    args = parser.parse_args()
    fig, outputs = build_figure(force_data=args.force_data)
    plt.close(fig)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
