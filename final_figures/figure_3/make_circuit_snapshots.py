"""Circuit-state snapshots of the AB/BA predictive cascade.

Node fill is a cell's rate at that instant; line width is a plastic weight.
Both are read straight out of the simulator -- nothing is idealised, and no
value in this figure was chosen by hand.

Two design decisions worth stating, because both could mislead if left implicit:

**Only the plastic weights are mapped to width.**  The fixed E->I and I->E
matrices are identical in every snapshot and in both conditions, so mapping them
to width would spend the reader's most salient visual channel on a constant.
They are drawn at a constant width and quoted numerically in the key.  Width
therefore means one thing only: *what the network learned*.

**Excitatory and inhibitory rates get separate colour scales.**  They differ by
roughly an order of magnitude (E peaks near 10, I near 1.8), so a shared scale
would render every interneuron black and hide the effect the figure exists to
show.  Two colourbars are drawn and the panel says they are not comparable.

Run with::

    python -m final_figures.figure_3.make_circuit_snapshots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import ScaledTranslation

from final_figures.figure_3.circuit_data import (
    CONDITIONS,
    GAP_MS,
    SNAPSHOTS,
    TONE_MS,
    build,
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
OUTPUT_STEM = OUTPUT_DIR / "ab_ba_circuit_snapshots"

#: Excitatory and inhibitory cells get separate, clearly-labelled ramps.
EXC_CMAP = LinearSegmentedColormap.from_list(
    "exc_colorbrewer_ord",
    ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"],
    N=256,
)
INH_CMAP = LinearSegmentedColormap.from_list(
    "inh_colorbrewer_blues",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    N=256,
)
#: Established ColorBrewer RdBu ramp for the frequent-minus-rare activity row.
#: Excitatory and inhibitory differences retain separate symmetric norms
#: because their magnitudes differ by roughly an order of magnitude.
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "signed_colorbrewer_rdbu",
    [
        "#2166AC",
        "#67A9CF",
        "#D1E5F0",
        "#F7F7F7",
        "#FDDBC7",
        "#EF8A62",
        "#B2182B",
    ],
    N=256,
)

# Conventional, colorblind-safe circuit semantics (Okabe-Ito family):
# every excitatory projection is warm and every inhibitory projection is cool.
COL_PLASTIC = "#D55E00"             # plastic recurrent E->E
COL_EI = "#E69F00"                  # fixed E->I
COL_IE = "#0072B2"                  # fixed I->E
COL_EXC_TRACE = COL_PLASTIC
COL_INH_TRACE = COL_IE
CONTEXT_COLORS = ("#0072B2", "#D55E00")
EXC_OUTLINE = "#A33A16"
INH_OUTLINE = "#005A8D"
DIFF_POS_EDGE = "#B2182B"
DIFF_NEG_EDGE = "#D6604D"
WINDOW_GREY = "#E5E7E9"
WINDOW_PEACH = "#F4D6C4"

#: Node positions, identical in every snapshot so the reader learns them once.
POS = {"A_e": (0.24, 0.72), "B_e": (0.76, 0.72),
       "A_i": (0.24, 0.18), "B_i": (0.76, 0.18)}
R_E, R_I = 0.115, 0.082

LW_MIN, LW_MAX = 0.35, 3.6
FIXED_LW = 0.75


def _panel_anchor(fig, spec):
    """Invisible full-cell axis that fixes title and letter alignment."""

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
    title_gap_pt: float = 14.0,
    label_gap_pt: float = 8.0,
) -> None:
    """Use the physical panel-header geometry shared by Figures 2 and 4."""

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
        letter,
        transform=label_transform,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["charcoal"],
        clip_on=False,
    )


def _plastic_width(weight: float, reference: float) -> float:
    """Line width proportional to a plastic weight, zero staying visibly zero."""

    if reference <= 0:
        return LW_MIN
    return LW_MIN + (LW_MAX - LW_MIN) * float(np.clip(weight / reference, 0, 1))


def _arrow(ax, start, end, *, width, colour, rad, shrink_a, shrink_b,
           alpha=1.0, zorder=3, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=5.2 + 1.6 * width,
        lw=width, color=colour, alpha=alpha, zorder=zorder,
        shrinkA=shrink_a, shrinkB=shrink_b, capstyle="round",
        linestyle=(0, (2.2, 1.6)) if dashed else "solid"))


def _self_loop(ax, centre, radius, *, width, colour, zorder=3):
    x, y = centre
    loop = np.linspace(0, 2 * np.pi, 90)
    rx, ry = radius * 0.62, radius * 0.62
    cx, cy = x, y + radius + ry * 0.72
    ax.plot(cx + rx * np.cos(loop), cy + ry * np.sin(loop), color=colour,
            lw=width, zorder=zorder, solid_capstyle="round")


def _draw_circuit(ax, excitatory, inhibitory, weights, *, exc_norm, inh_norm,
                  w_reference, title, exc_cmap=EXC_CMAP, inh_cmap=INH_CMAP,
                  exc_format="{:.2f}", inh_format="{:.2f}",
                  signed_weights=False, fixed_alpha=1.0,
                  show_weight_labels=True, show_population_labels=False) -> None:
    """One snapshot: four cells, their rates, and the learned weights.

    The same routine draws the difference row, so every node sits at the same
    coordinate in every panel of the figure and the three rows are directly
    comparable by eye. ``signed_weights`` switches the plastic arcs to
    magnitude-and-sign, and ``fixed_alpha`` fades the unchanging fixed
    connections without moving them.
    """

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    a_e, b_e = POS["A_e"], POS["B_e"]
    a_i, b_i = POS["A_i"], POS["B_i"]

    # --- fixed inhibitory loop, constant width -------------------------------
    for src, dst, rad in ((a_e, a_i, 0.0), (b_e, b_i, 0.0)):
        _arrow(ax, src, dst, width=FIXED_LW, colour=COL_EI, rad=rad,
               shrink_a=R_E * 100 * 0.62, shrink_b=R_I * 100 * 0.62,
               alpha=0.85 * fixed_alpha, zorder=2)
    for src, dst in ((a_e, b_i), (b_e, a_i)):
        _arrow(ax, src, dst, width=FIXED_LW * 0.55, colour=COL_EI, rad=0.30,
               shrink_a=R_E * 100 * 0.62, shrink_b=R_I * 100 * 0.62,
               alpha=0.45 * fixed_alpha, zorder=2)
    for src, dst in ((a_i, a_e), (b_i, b_e)):
        _arrow(ax, src, dst, width=FIXED_LW * 1.5, colour=COL_IE, rad=0.42,
               shrink_a=R_I * 100 * 0.62, shrink_b=R_E * 100 * 0.62,
               alpha=0.9 * fixed_alpha, zorder=2)
    for src, dst in ((a_i, b_e), (b_i, a_e)):
        _arrow(ax, src, dst, width=FIXED_LW * 0.55, colour=COL_IE, rad=-0.30,
               shrink_a=R_I * 100 * 0.62, shrink_b=R_E * 100 * 0.62,
               alpha=0.4 * fixed_alpha, zorder=2)

    # --- plastic recurrent excitation, width = learned weight ----------------
    # A->B is drawn above the excitatory pair and B->A below it, so the thick
    # arc sits on a different side in each condition -- readable at a glance
    # where a 2 pt arrowhead is not.  Labels sit at fixed heights on each side.
    for src, dst, weight, rad, label_y, valign, name in (
        (a_e, b_e, weights[1, 0], -0.42, 0.925, "bottom", "A→B"),
        (b_e, a_e, weights[0, 1], -0.26, 0.545, "top", "B→A"),
    ):
        # A recurrent E->E projection remains warm even when the plotted
        # quantity is a negative difference. Sign is carried by shade, dash
        # and the printed value; blue is reserved for inhibitory projections.
        colour = (
            DIFF_POS_EDGE if weight >= 0 else DIFF_NEG_EDGE
        ) if signed_weights else COL_PLASTIC
        _arrow(ax, src, dst,
               width=_plastic_width(abs(weight) if signed_weights else weight,
                                    w_reference),
               colour=colour, rad=rad,
               shrink_a=R_E * 100 * 0.62, shrink_b=R_E * 100 * 0.62,
               zorder=5, dashed=signed_weights and weight < 0)
        if show_weight_labels:
            text = (f"\u0394{name}  {weight:+.3f}" if signed_weights
                    else f"{name}  {weight:.3f}")
            ax.text(0.5, label_y, text, ha="center", va=valign,
                    fontsize=4.7, color=colour,
                    fontweight=("semibold" if signed_weights
                                else ("bold" if weight > 0 else "normal")),
                    zorder=8)
    for centre, self_weight in ((a_e, weights[0, 0]), (b_e, weights[1, 1])):
        _self_loop(ax, centre,
                   R_E,
                   width=_plastic_width(
                       abs(self_weight) if signed_weights else self_weight,
                       w_reference),
                   colour=COL_PLASTIC,
                   zorder=4)

    # --- cells ---------------------------------------------------------------
    for key, value, radius, cmap, norm, fmt, label in (
        ("A_e", excitatory[0], R_E, exc_cmap, exc_norm, exc_format, "A"),
        ("B_e", excitatory[1], R_E, exc_cmap, exc_norm, exc_format, "B"),
        ("A_i", inhibitory[0], R_I, inh_cmap, inh_norm, inh_format, "A"),
        ("B_i", inhibitory[1], R_I, inh_cmap, inh_norm, inh_format, "B"),
    ):
        fill = cmap(norm(value))
        is_exc = key.endswith("_e")
        outline = EXC_OUTLINE if is_exc else INH_OUTLINE
        if is_exc:
            node = Circle(
                POS[key],
                radius,
                facecolor=fill,
                edgecolor=outline,
                lw=1.05,
                zorder=6,
            )
        else:
            x, y = POS[key]
            node = FancyBboxPatch(
                (x - radius, y - radius),
                2 * radius,
                2 * radius,
                boxstyle=f"round,pad=0,rounding_size={radius * 0.34}",
                facecolor=fill,
                edgecolor=outline,
                lw=1.05,
                zorder=6,
            )
        ax.add_patch(node)
        luminance = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        ax.text(*POS[key], label, ha="center", va="center",
                fontsize=5.8 if radius == R_E else 5.0, fontweight="bold",
                color=COLORS["white"] if luminance < 0.55 else COLORS["charcoal"],
                zorder=7)
        ax.text(POS[key][0], POS[key][1] - radius - 0.055, fmt.format(value),
                ha="center", va="top", fontsize=5.0, color=COLORS["charcoal"],
                zorder=7)

    if show_population_labels:
        ax.text(0.055, POS["A_e"][1], "E", ha="center", va="center",
                fontsize=5.2, fontweight="semibold", color=COLORS["charcoal"])
        ax.text(0.055, POS["A_i"][1], "I", ha="center", va="center",
                fontsize=5.2, fontweight="semibold", color=COLORS["charcoal"])

    ax.set_title(title, loc="center", fontsize=6.6, pad=3.0,
                 fontweight="semibold")


def _draw_difference_circuit(
    ax,
    delta_e,
    delta_i,
    delta_w,
    *,
    exc_norm,
    inh_norm,
    w_reference,
    title,
    show_weight_labels=True,
    show_population_labels=False,
) -> None:
    """Frequent minus rare, drawn as the same circuit as rows B and C.

    The fixed E->I and I->E matrices are identical in the two contexts, so
    their difference is exactly zero. They are still drawn, faintly, because
    removing them leaves the interneurons floating unconnected and the reader
    has to relearn the diagram; fading them says "unchanged" without breaking
    the correspondence.
    """

    _draw_circuit(
        ax, delta_e, delta_i, delta_w,
        exc_norm=exc_norm, inh_norm=inh_norm, w_reference=w_reference,
        title=title, exc_cmap=DIFF_CMAP, inh_cmap=DIFF_CMAP,
        exc_format="{:+.2f}", inh_format="{:+.3f}",
        signed_weights=True, fixed_alpha=0.24,
        show_weight_labels=show_weight_labels,
        show_population_labels=show_population_labels,
    )


def _panel_timecourse(ax, data, name, label, *, show_ylabel, show_legend) -> None:
    excitatory = np.asarray(data[f"{name}|E"], dtype=float)
    inhibitory = np.asarray(data[f"{name}|I"], dtype=float)
    time_ms = np.arange(excitatory.shape[1])

    ax.axvspan(0, TONE_MS, color=WINDOW_GREY, alpha=0.72, lw=0, zorder=0)
    ax.axvspan(TONE_MS + GAP_MS, 2 * TONE_MS + GAP_MS, color=WINDOW_PEACH,
               alpha=0.68, lw=0, zorder=0)
    ax.text(TONE_MS / 2, 10.4, "A", ha="center", va="top", fontsize=5.6,
            color=COLORS["ash"], fontweight="bold")
    ax.text(1.5 * TONE_MS + GAP_MS, 10.4, "B", ha="center", va="top",
            fontsize=5.6, color=COLORS["charcoal"], fontweight="bold")

    for row, colour, style, name_ in ((0, COL_EXC_TRACE, "-", "E$_A$"),
                                      (1, COL_EXC_TRACE, (0, (2.4, 1.6)), "E$_B$")):
        ax.plot(time_ms, excitatory[row], color=colour, lw=1.15, ls=style,
                zorder=4, label=name_)
    for row, colour, style, name_ in ((0, COL_INH_TRACE, "-", "I$_A$"),
                                      (1, COL_INH_TRACE, (0, (2.4, 1.6)), "I$_B$")):
        ax.plot(time_ms, inhibitory[row], color=colour, lw=1.0, ls=style,
                zorder=3, label=name_)

    for moment, _title, _sub in SNAPSHOTS:
        ax.axvline(moment, color=COLORS["charcoal"], lw=0.5, ls=(0, (1.6, 1.6)),
                   alpha=0.65, zorder=2)
        ax.text(moment, 10.95, f"{moment}", ha="center", va="top",
                fontsize=4.8, color=COLORS["charcoal"])

    ax.set_xlim(-8, 175)
    ax.set_ylim(0, 11.2)
    ax.set_xticks([0, 50, 80, 130])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("Time from sequence onset (ms)")
    if show_ylabel:
        ax.set_ylabel("Rate (a.u.)", labelpad=2.0)
    ax.set_title(label, loc="left", fontsize=6.6, pad=3.0,
                 fontweight="semibold")
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), fontsize=5.4,
                  ncol=2, labelspacing=0.18, columnspacing=0.7,
                  handlelength=1.15, borderaxespad=0.2)
    clean_axis(ax)


def _panel_cascade(ax, values, ylabel, title, note, *, zoom=False) -> None:
    labels = ("freq.", "rare")
    x = np.asarray((0.0, 1.0))
    ax.bar(
        x,
        values,
        width=0.58,
        color=CONTEXT_COLORS,
        edgecolor=COLORS["white"],
        linewidth=0.50,
        zorder=2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=5.2)
    ax.set_xlim(-0.62, 1.62)
    if zoom:
        # The axis break is declared graphically so the small difference is
        # legible without implying a zero-based scale.
        low, high = min(values), max(values)
        span = high - low
        ax.set_ylim(low - 1.6 * span, high + 1.5 * span)
        # Explicit axis-break mark: the zoom is declared graphically rather
        # than hidden in a small subtitle below the panel.
        ax.plot(
            (-0.028, 0.028),
            (-0.018, 0.032),
            transform=ax.transAxes,
            color=COLORS["charcoal"],
            lw=0.65,
            clip_on=False,
        )
    else:
        top = max(values) if max(values) > 0 else 1.0
        ax.set_ylim(0, top * 1.30)
    y0, y1 = ax.get_ylim()
    offset = 0.025 * (y1 - y0)
    for index, (value, colour) in enumerate(zip(values, CONTEXT_COLORS)):
        label_y = max(value + offset, y0 + 0.045 * (y1 - y0))
        ax.text(
            index,
            label_y,
            f"{value:.3f}" if value < 1 else f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=colour,
            fontweight="semibold",
            zorder=4,
        )
    ax.set_ylabel(ylabel, labelpad=2.0)
    ax.set_title(title, loc="left", fontsize=6.25, pad=2.8,
                 fontweight="semibold")
    clean_axis(ax)


def _panel_difference(ax, data, names) -> None:
    """The effect as a time course, with the prespecified target window as the
    subject.

    The tone-A lobe and the tone-B lobe are not commensurable: the first is a
    pre-activation against a baseline of ~0, the second a suppression against a
    baseline of ~10.  Plotting both filled on one signed axis makes the larger,
    irrelevant lobe the visual subject, and the strategy document explicitly
    forbids letting an earlier sequence-onset difference stand in for the target
    effect.  The tone-A lobe is therefore drawn as an unfilled context outline
    and the target window is the only thing filled.
    """

    rare = np.asarray(data[f"{names[1]}|E"], dtype=float)[1]
    frequent = np.asarray(data[f"{names[0]}|E"], dtype=float)[1]
    difference = rare - frequent
    time_ms = np.arange(difference.size)
    target = np.zeros_like(time_ms, dtype=bool)
    target[TONE_MS + GAP_MS:2 * TONE_MS + GAP_MS] = True

    ax.axvspan(TONE_MS + GAP_MS, 2 * TONE_MS + GAP_MS, color=WINDOW_PEACH,
               alpha=0.68, lw=0, zorder=0)
    ax.axhline(0, color=COLORS["ash"], lw=0.6, zorder=1)
    ax.plot(time_ms[~target], np.where(difference, difference, np.nan)[~target],
            color=COLORS["ash"], lw=0.8, zorder=2)
    ax.fill_between(time_ms, 0, np.where(target, difference, np.nan),
                    color=CONTEXT_COLORS[1], alpha=0.24, lw=0, zorder=3)
    ax.plot(time_ms, np.where(target, difference, np.nan), color=CONTEXT_COLORS[1],
            lw=1.4, zorder=4)

    peak = int(np.argmax(np.where(target, difference, -np.inf)))
    ax.annotate(f"+{difference[peak]:.2f}", xy=(peak, difference[peak]),
                xytext=(peak + 26, difference[peak] * 1.05), fontsize=5.2,
                color=CONTEXT_COLORS[1], va="center", ha="left",
                fontweight="semibold",
                arrowprops=dict(arrowstyle="-", color=CONTEXT_COLORS[1], lw=0.5))

    ax.set_xlim(-8, 175)
    ax.set_xticks([0, 80, 160])
    ax.set_xlabel("Time (ms)", labelpad=1.6)
    ax.set_ylabel("E$_B$ rare - frequent", labelpad=2.0)
    ax.set_title("5. Target response", loc="left", fontsize=6.25, pad=2.8,
                 fontweight="semibold")
    clean_axis(ax)


def _panel_effect(ax, data) -> None:
    """Effect size across independent seed pairs -- the reliability, not the size."""

    values = np.asarray(data["suppression_pct"], dtype=float)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    rng = np.random.default_rng(11)

    ax.axhline(0, color=COLORS["ash"], lw=0.6, ls=(0, (2, 2)), zorder=1)
    seed_x = rng.uniform(-0.16, 0.16, values.size)
    effect_color = "#4D4D4D"
    ax.scatter(seed_x, values, s=11, color="#8C8C8C", alpha=0.48,
               linewidth=0, zorder=3)
    ax.errorbar(1.10, mean, yerr=sem, fmt="D", ms=4.4, color=effect_color,
                markeredgecolor=COLORS["white"], markeredgewidth=0.6,
                elinewidth=1.3, capsize=2.4, capthick=0.8, zorder=5)
    ax.text(1.34, mean, f"{mean:.2f} ± {sem:.2f}%", ha="left", va="center",
            fontsize=5.1, fontweight="semibold", color=effect_color)
    ax.set_xlim(-0.42, 2.05)
    ax.set_ylim(min(0, values.min() * 1.15), values.max() * 1.28)
    ax.set_xticks((0.0, 1.10), labels=("seeds", "mean"))
    ax.set_ylabel("Suppression (%)", labelpad=2.0)
    ax.set_title("6. Reliability", loc="left", fontsize=6.25, pad=2.8,
                 fontweight="semibold")
    clean_axis(ax)
    ax.tick_params(axis="x", length=0)


def build_figure(*, force_data: bool = False) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build(force=force_data, data_dir=DATA_DIR)

    names = [c[0] for c in CONDITIONS]
    labels = (
        "AB frequent · A→B learned",
        "AB rare · A→B unlearned",
    )
    weights = {n: np.asarray(data[f"{n}|W"], dtype=float) for n in names}
    excitatory = {n: np.asarray(data[f"{n}|E"], dtype=float) for n in names}
    inhibitory = {n: np.asarray(data[f"{n}|I"], dtype=float) for n in names}

    exc_max = max(float(excitatory[n].max()) for n in names)
    inh_max = max(float(inhibitory[n].max()) for n in names)
    w_reference = max(float(weights[n].max()) for n in names)
    exc_norm = Normalize(0.0, exc_max)
    inh_norm = Normalize(0.0, inh_max)

    snapshot_indices = np.asarray([moment for moment, _title, _sub in SNAPSHOTS],
                                  dtype=int)
    delta_e = excitatory[names[0]] - excitatory[names[1]]
    delta_i = inhibitory[names[0]] - inhibitory[names[1]]
    delta_w = weights[names[0]] - weights[names[1]]
    delta_e_max = float(np.max(np.abs(delta_e[:, snapshot_indices])))
    delta_i_max = float(np.max(np.abs(delta_i[:, snapshot_indices])))
    delta_e_norm = TwoSlopeNorm(vmin=-delta_e_max, vcenter=0.0,
                                vmax=delta_e_max)
    delta_i_norm = TwoSlopeNorm(vmin=-delta_i_max, vcenter=0.0,
                                vmax=delta_i_max)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(245)))
        grid = fig.add_gridspec(
            5, 12, left=0.057, right=0.935, top=0.958, bottom=0.045,
            wspace=0.78, hspace=0.60,
            height_ratios=(0.82, 1.00, 1.00, 1.00, 1.00))

        # ---- A: time courses -------------------------------------------------
        top = grid[0, :].subgridspec(1, 2, wspace=0.24)
        axes_tc = []
        for index, (name, label) in enumerate(zip(names, labels)):
            ax = fig.add_subplot(top[0, index])
            _panel_timecourse(
                ax,
                data,
                name,
                label,
                show_ylabel=index == 0,
                show_legend=False,
            )
            axes_tc.append(ax)
        a_anchor = _panel_anchor(fig, grid[0, :])
        _panel_heading(a_anchor, "A", "Sequence-evoked population activity")
        a_anchor.legend(
            handles=[
                Line2D([0], [0], color=COL_EXC_TRACE, lw=1.35, label="E$_A$"),
                Line2D([0], [0], color=COL_EXC_TRACE, lw=1.35,
                       ls=(0, (2.4, 1.6)), label="E$_B$"),
                Line2D([0], [0], color=COL_INH_TRACE, lw=1.20, label="I$_A$"),
                Line2D([0], [0], color=COL_INH_TRACE, lw=1.20,
                       ls=(0, (2.4, 1.6)), label="I$_B$"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.15),
            ncol=4,
            handlelength=1.25,
            columnspacing=0.75,
            fontsize=5.7,
            borderaxespad=0,
        )

        # ---- B, C: circuit snapshots ----------------------------------------
        circuit_rows: dict[str, list] = {}
        row_titles = (
            "Circuit states after AB-frequent learning",
            "Circuit states after AB-rare learning",
        )
        for row, (name, label, letter, row_title) in enumerate(
            zip(names, labels, ("B", "C"), row_titles), start=1
        ):
            nested = grid[row, :].subgridspec(1, 4, wspace=0.16)
            circuit_rows[name] = []
            for column, (moment, title, subtitle) in enumerate(SNAPSHOTS):
                ax = fig.add_subplot(nested[0, column])
                circuit_rows[name].append(ax)
                _draw_circuit(
                    ax,
                    excitatory[name][:, moment], inhibitory[name][:, moment],
                    weights[name],
                    exc_norm=exc_norm, inh_norm=inh_norm,
                    w_reference=w_reference,
                    title=f"{moment} ms · {title}",
                    show_weight_labels=column == 0,
                    show_population_labels=column == 0,
                )
            row_anchor = _panel_anchor(fig, grid[row, :])
            _panel_heading(row_anchor, letter, row_title)
            if row == 1:
                row_anchor.legend(
                    handles=[
                        Line2D([0], [0], color=COL_EI, lw=0.75,
                               label="E→I fixed"),
                        Line2D([0], [0], color=COL_PLASTIC, lw=2.4,
                               label="E→E plastic"),
                        Line2D([0], [0], color=COL_IE, lw=1.15,
                               label="I→E fixed"),
                    ],
                    loc="upper right",
                    bbox_to_anchor=(1.0, 1.22),
                    ncol=3,
                    handlelength=1.25,
                    columnspacing=0.8,
                    fontsize=5.7,
                    borderaxespad=0,
                )

        # ---- D: signed frequent-minus-rare circuit snapshots ----------------
        nested_difference = grid[3, :].subgridspec(1, 4, wspace=0.16)
        axes_difference = []
        for column, (moment, title, _subtitle) in enumerate(SNAPSHOTS):
            ax = fig.add_subplot(nested_difference[0, column])
            axes_difference.append(ax)
            _draw_difference_circuit(
                ax,
                delta_e[:, moment],
                delta_i[:, moment],
                delta_w,
                exc_norm=delta_e_norm,
                inh_norm=delta_i_norm,
                w_reference=w_reference,
                title=f"{moment} ms · {title}",
                show_weight_labels=column == 0,
                show_population_labels=column == 0,
            )
        d_anchor = _panel_anchor(fig, grid[3, :])
        _panel_heading(
            d_anchor, "D", "Identity-controlled circuit difference (frequent - rare)"
        )
        d_anchor.legend(
            handles=[
                Line2D([0], [0], marker="o", ms=5.2, lw=0,
                       markerfacecolor="white", markeredgecolor=EXC_OUTLINE,
                       label="excitatory population"),
                Line2D([0], [0], marker="s", ms=4.8, lw=0,
                       markerfacecolor="white", markeredgecolor=INH_OUTLINE,
                       label="inhibitory population"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.27),
            ncol=2,
            handletextpad=0.35,
            columnspacing=0.8,
            fontsize=5.6,
            borderaxespad=0,
        )

        # ---- E: the cascade, quantified -------------------------------------
        tone_a = slice(0, TONE_MS)
        gap = slice(TONE_MS, TONE_MS + GAP_MS)
        tone_b = slice(TONE_MS + GAP_MS, 2 * TONE_MS + GAP_MS)
        cascade = (
            ("W[B←A]", [float(weights[n][1, 0]) for n in names],
             "Learned weight", "1. The link", "plastic E→E, A→B"),
            ("preact", [float(excitatory[n][1, tone_a].max()) for n in names],
             "Peak E$_B$ (a.u.)", "2. Pre-activation", "during tone A"),
            ("inh", [float(inhibitory[n][1, gap].mean()) for n in names],
             "Mean I$_B$ (a.u.)", "3. Standing inhibition", "in the 30 ms gap"),
            ("target", [float(excitatory[n][1, tone_b].max()) for n in names],
             "Peak E$_B$ (a.u.)", "4. Target response\n(zoomed axis)",
             "during tone B · axis truncated"),
        )
        nested = grid[4, :].subgridspec(
            1,
            6,
            width_ratios=(0.86, 0.86, 0.86, 0.92, 1.20, 1.08),
            wspace=0.86,
        )
        axes_cascade = []
        for column, (_key, values, ylabel, title, note) in enumerate(cascade):
            ax = fig.add_subplot(nested[0, column])
            display_title = title.split("\n")[0]
            _panel_cascade(ax, values, ylabel, display_title, note,
                           zoom=(column == 3))
            axes_cascade.append(ax)
        _panel_difference(fig.add_subplot(nested[0, 4]), data, names)
        _panel_effect(fig.add_subplot(nested[0, 5]), data)
        e_anchor = _panel_anchor(fig, grid[4, :])
        _panel_heading(e_anchor, "E", "Predictive cascade")
        e_anchor.legend(
            handles=[
                Line2D([0], [0], color=CONTEXT_COLORS[0], lw=4.2,
                       label="AB frequent"),
                Line2D([0], [0], color=CONTEXT_COLORS[1], lw=4.2,
                       label="AB rare"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.25),
            ncol=2,
            handlelength=1.0,
            columnspacing=0.8,
            fontsize=5.7,
            borderaxespad=0,
        )

        # ---- keys ------------------------------------------------------------
        absolute_top = circuit_rows[names[0]][0].get_position().y1
        absolute_bottom = circuit_rows[names[1]][0].get_position().y0
        absolute_span = absolute_top - absolute_bottom

        cax_e = fig.add_axes([
            0.953,
            absolute_bottom + absolute_span * 0.56,
            0.011,
            absolute_span * 0.32,
        ])
        bar_e = fig.colorbar(plt.cm.ScalarMappable(norm=exc_norm, cmap=EXC_CMAP),
                             cax=cax_e)
        bar_e.set_label("E rate", fontsize=5.4, labelpad=2.0)
        bar_e.set_ticks([0, exc_max])
        bar_e.set_ticklabels(["0", f"{exc_max:.0f}"])
        bar_e.ax.tick_params(labelsize=5.0, length=1.4, width=0.45, pad=1.0)
        bar_e.outline.set_linewidth(0.45)

        cax_i = fig.add_axes([
            0.953,
            absolute_bottom + absolute_span * 0.10,
            0.011,
            absolute_span * 0.32,
        ])
        bar_i = fig.colorbar(plt.cm.ScalarMappable(norm=inh_norm, cmap=INH_CMAP),
                             cax=cax_i)
        bar_i.set_label("I rate", fontsize=5.4, labelpad=2.0)
        bar_i.set_ticks([0, inh_max])
        bar_i.set_ticklabels(["0", f"{inh_max:.1f}"])
        bar_i.ax.tick_params(labelsize=5.0, length=1.4, width=0.45, pad=1.0)
        bar_i.outline.set_linewidth(0.45)
        difference_top = axes_difference[0].get_position().y1
        difference_bottom = axes_difference[0].get_position().y0
        difference_span = difference_top - difference_bottom
        cax_de = fig.add_axes([
            0.953,
            difference_bottom + difference_span * 0.54,
            0.011,
            difference_span * 0.40,
        ])
        bar_de = fig.colorbar(
            plt.cm.ScalarMappable(norm=delta_e_norm, cmap=DIFF_CMAP),
            cax=cax_de,
        )
        bar_de.set_label("ΔE", fontsize=5.2, labelpad=1.6)
        bar_de.set_ticks([-delta_e_max, 0, delta_e_max])
        bar_de.set_ticklabels([
            f"-{delta_e_max:.1f}", "0", f"+{delta_e_max:.1f}",
        ])
        bar_de.ax.tick_params(labelsize=4.8, length=1.2, width=0.45, pad=0.8)
        bar_de.outline.set_linewidth(0.45)

        cax_di = fig.add_axes([
            0.953,
            difference_bottom + difference_span * 0.05,
            0.011,
            difference_span * 0.40,
        ])
        bar_di = fig.colorbar(
            plt.cm.ScalarMappable(norm=delta_i_norm, cmap=DIFF_CMAP),
            cax=cax_di,
        )
        bar_di.set_label("ΔI", fontsize=5.2, labelpad=1.6)
        bar_di.set_ticks([-delta_i_max, 0, delta_i_max])
        bar_di.set_ticklabels([
            f"-{delta_i_max:.2f}", "0", f"+{delta_i_max:.2f}",
        ])
        bar_di.ax.tick_params(labelsize=4.8, length=1.2, width=0.45, pad=0.8)
        bar_di.outline.set_linewidth(0.45)
        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    with (DATA_DIR / "cascade_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "quantity", "ab_frequent", "ab_rare"])
        for key_, values, ylabel, title, _note in cascade:
            writer.writerow([title, ylabel, values[0], values[1]])
    with (DATA_DIR / "snapshot_differences.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_ms", "delta_E_A", "delta_E_B", "delta_I_A", "delta_I_B",
            "delta_W_A_to_B", "delta_W_B_to_A",
        ])
        for moment, _title, _subtitle in SNAPSHOTS:
            writer.writerow([
                moment,
                delta_e[0, moment],
                delta_e[1, moment],
                delta_i[0, moment],
                delta_i[1, moment],
                delta_w[1, 0],
                delta_w[0, 1],
            ])
    return paths


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-data", action="store_true",
                        help="Repeat both simulated runs.")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    paths = build_figure(force_data=_parse_args(arguments).force_data)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
