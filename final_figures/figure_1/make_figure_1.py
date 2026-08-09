"""Build the conceptual opener, Figure 1.

Figure 1 states the claim, defines its terms, draws the circuit, shows how to
read the model's traces, maps the Results, and sets out the matrix the rest of
the paper fills in.  Panels A, B, C and E are schematics; **panel D is measured
output of ``model0``** and panel F is the prediction matrix.

The colour key established here is binding for Figures 2-6:

    depression                     terracotta
    plastic recurrent excitation   model (green)
    inhibition                     teal
    downstream multiscale readout  decoder (violet)

Every schematic panel draws in true millimetres with an equal aspect ratio, so
a circle is a circle, a 3 mm gap is 3 mm on the printed page, and the geometry
can be reasoned about at final size rather than in arbitrary axis units.
Headings and panel letters are offset in millimetres too, which is what keeps
their spacing identical across panels of very different heights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from final_figures.figure_1.panel_d_data import build_panel_d_data
from final_figures.style import (
    COLORS,
    clean_axis,
    export_figure,
    manuscript_style,
    mm,
    panel_label,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_1_architecture"

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 247.0

# ---------------------------------------------------------------------------
# The key this figure establishes.  One colour per mechanism, held for the
# rest of the manuscript.  Excitatory and inhibitory cells are additionally
# distinguished by fill and by terminal shape (arrowhead versus bar), so the
# circuit survives greyscale reproduction and colour-blind readers.
# ---------------------------------------------------------------------------
MECH = {
    "depression": COLORS["terracotta"],
    "plasticity": COLORS["model"],
    "inhibition": COLORS["teal"],
    "readout": COLORS["decoder"],
}
MECH_LABEL = {
    "depression": "Synaptic\ndepression",
    "plasticity": "Recurrent\nplasticity",
    "inhibition": "Inhibitory\nstructure",
    "readout": "Multiscale\nreadout",
}
#: Tints of the mechanism hues, used only as region washes behind the circuit.
MECH_WASH = {
    "depression": "#F7ECEC",
    "plasticity": "#E8F0EC",
    "inhibition": "#E7EFEF",
    "readout": "#EDEAF2",
}

STRUCTURE = COLORS["charcoal"]
FAINT = COLORS["ash"]

#: Type sizes.  Nothing in this figure is below 5 pt.
FS_STAGE = 6.6
FS_ANNOT = 5.6
FS_SMALL = 5.2
FS_TICK = 5.8


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Drawing surface and typography
# ---------------------------------------------------------------------------
def _axes_size_mm(fig, ax) -> tuple[float, float]:
    box = ax.get_position()
    return (box.width * fig.get_figwidth() * 25.4,
            box.height * fig.get_figheight() * 25.4)


def _mm_axes(fig, spec):
    """An axis-free drawing surface whose data coordinates are millimetres.

    The data limits are set to the axes' own printed size and the aspect is
    locked, so every radius, gap and stroke below is a real distance on the
    submitted page rather than a fraction of a box whose shape may change.
    """

    ax = fig.add_subplot(spec)
    width, height = _axes_size_mm(fig, ax)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax, width, height


def _heading(ax, height_mm: float, title: str, subtitle: str, *,
             title_mm: float = 5.2, subtitle_mm: float = 1.9,
             letter: str | None = None, letter_mm: float = 9.4,
             letter_x_mm: float = -6.4, width_mm: float = 100.0) -> None:
    """Claim line, grey gloss and panel letter, all offset in millimetres.

    Offsets are given in millimetres and converted to axes fractions here, so
    a 30 mm panel and a 6 mm sub-panel end up with visually identical heading
    spacing instead of spacing proportional to their own height.
    """

    ax.text(0, 1.0 + title_mm / height_mm, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.2, fontweight="semibold",
            color=COLORS["charcoal"], clip_on=False)
    ax.text(0, 1.0 + subtitle_mm / height_mm, subtitle, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.9, color=COLORS["ash"],
            linespacing=1.4, clip_on=False)
    if letter is not None:
        panel_label(ax, letter, x=letter_x_mm / width_mm,
                    y=1.0 + letter_mm / height_mm)


def _token(ax, x, y, *, w=4.6, h=2.4, color=STRUCTURE, alpha=1.0, zorder=3):
    """One tone token: the atomic element of every stimulus in the paper."""

    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color,
                           alpha=alpha, edgecolor="none", zorder=zorder))


def _arrow(ax, start, end, *, color=STRUCTURE, lw=0.7, rad=0.0, style="-",
           head=2.0, zorder=4, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=f"-|>,head_length={head},head_width={head * 0.55}",
        connectionstyle=f"arc3,rad={rad}", color=color, linewidth=lw,
        linestyle=style, shrinkA=0, shrinkB=0, zorder=zorder, alpha=alpha,
        joinstyle="round", capstyle="round"))


def _line(ax, start, end, *, color=STRUCTURE, lw=0.7, rad=0.0, style="-",
          zorder=4, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, linestyle=style, shrinkA=0, shrinkB=0,
        zorder=zorder, alpha=alpha, capstyle="round"))


def _bar_terminal(ax, end, direction, *, color, lw=0.9, half=0.9, zorder=5):
    """The flat terminal that marks an inhibitory synapse."""

    dx, dy = direction
    norm = float(np.hypot(dx, dy)) or 1.0
    px, py = -dy / norm * half, dx / norm * half
    ax.plot([end[0] - px, end[0] + px], [end[1] - py, end[1] + py],
            color=color, lw=lw, solid_capstyle="butt", zorder=zorder)


# ---------------------------------------------------------------------------
# Panel A — the problem
# ---------------------------------------------------------------------------
def _panel_problem(fig, spec) -> None:
    """One scene, two organisations, one operation.

    The claim is carried by the colour: both link families are drawn in the
    plasticity green because the figure asserts they are the same operation.
    The families are told apart by geometry and by direction — coincidence is
    undirected, continuation is not — rather than by hue.
    """

    ax, width, height = _mm_axes(fig, spec)

    x0, dx = 16.0, 8.2
    y0, dy = 6.2, 3.4
    n_time, n_chan = 8, 6

    def px(slot):
        return x0 + slot * dx

    def py(chan):
        return y0 + chan * dy

    for chan in range(n_chan):
        ax.plot([px(0) - 3.6, px(n_time - 1) + 3.6], [py(chan)] * 2,
                color=COLORS["mist"], lw=0.4, zorder=0)

    concurrent_slots = (0, 3, 6)
    concurrent_channels = (3, 5)
    sequential = ((1, 0, 2, 1), (4, 0, 5, 1))
    clutter = ((2, 2), (5, 4), (7, 2))

    for slot, chan in clutter:
        _token(ax, px(slot), py(chan), h=2.1, color=FAINT, alpha=0.40)
    for slot in concurrent_slots:
        for chan in concurrent_channels:
            _token(ax, px(slot), py(chan), h=2.1)
    for slot_a, chan_a, slot_b, chan_b in sequential:
        _token(ax, px(slot_a), py(chan_a), h=2.1)
        _token(ax, px(slot_b), py(chan_b), h=2.1)

    # Coincidence: undirected, vertical, zero lag.
    for slot in concurrent_slots:
        low, high = (py(c) for c in concurrent_channels)
        _line(ax, (px(slot), low + 1.2), (px(slot), high - 1.2),
              color=MECH["plasticity"], lw=1.5, zorder=5)

    # Continuation: directed, across time, non-zero lag.
    for slot_a, chan_a, slot_b, chan_b in sequential:
        _arrow(ax, (px(slot_a) + 2.6, py(chan_a) + 0.7),
               (px(slot_b) - 2.6, py(chan_b) - 0.7),
               color=MECH["plasticity"], lw=1.1, rad=-0.34, head=2.1, zorder=5)

    # A two-item key rather than labels pinned to the glyphs: at this size any
    # in-lattice callout either sits on a token or on the time axis.  The key
    # also makes the panel's claim explicit — one colour, two geometries.
    key_y = py(n_chan - 1) + 3.4
    _line(ax, (x0 - 1.0, key_y - 1.3), (x0 - 1.0, key_y + 1.3),
          color=MECH["plasticity"], lw=1.5)
    ax.text(x0 + 1.4, key_y, "lag 0 · concurrent", fontsize=FS_SMALL,
            color=MECH["plasticity"], ha="left", va="center",
            fontweight="semibold")
    _arrow(ax, (px(3.4), key_y - 1.0), (px(3.4) + 4.6, key_y - 1.0),
           color=MECH["plasticity"], lw=1.1, rad=-0.5, head=2.1)
    ax.text(px(3.4) + 6.4, key_y, "lag > 0 · sequential", fontsize=FS_SMALL,
            color=MECH["plasticity"], ha="left", va="center",
            fontweight="semibold")

    ax.text(5.6, (py(0) + py(n_chan - 1)) / 2, "Frequency", rotation=90,
            ha="center", va="center", fontsize=FS_ANNOT, color=STRUCTURE)
    _arrow(ax, (9.4, py(0) - 1.2), (9.4, py(n_chan - 1) + 1.2), color=FAINT,
           lw=0.5, head=1.7)
    ax.text((px(0) + px(n_time - 1)) / 2, 1.4, "Time", ha="center", va="center",
            fontsize=FS_ANNOT, color=STRUCTURE)
    _arrow(ax, (px(0) - 3.2, 3.6), (px(n_time - 1) + 3.2, 3.6), color=FAINT,
           lw=0.5, head=1.7)

    _heading(
        ax, height,
        "Grouping and prediction are one question at two lags",
        "elements that always co-occur are heard as one sound;\n"
        "elements that always follow are heard as a continuation",
        title_mm=6.6, subtitle_mm=1.9, letter="A", letter_mm=10.6,
        width_mm=width)


# ---------------------------------------------------------------------------
# Panel B — the definition
# ---------------------------------------------------------------------------
def _panel_definition(fig, spec) -> None:
    """Three beats, and no more: coincidence, link, prediction."""

    ax, width, height = _mm_axes(fig, spec)

    green = MECH["plasticity"]
    centres = (width * 0.165, width * 0.505, width * 0.845)
    mid = height * 0.655

    # --- beat 1: coincidence makes a token -------------------------------
    cx = centres[0]
    for offset in (2.2, -2.2):
        _token(ax, cx - 8.2, mid + offset, w=4.4, h=1.9)
    _line(ax, (cx - 4.6, mid + 3.1), (cx - 4.6, mid - 3.1), color=green, lw=1.4)
    _arrow(ax, (cx - 2.6, mid), (cx + 1.4, mid), color=FAINT, lw=0.6, head=1.8)
    _token(ax, cx + 5.6, mid, w=5.4, h=4.4, color=green)
    ax.text(cx + 5.6, mid, "1", ha="center", va="center", fontsize=FS_SMALL,
            color=COLORS["white"], fontweight="semibold", zorder=6)

    # --- beat 2: repetition makes a link ---------------------------------
    cx = centres[1]
    for depth, alpha in ((2, 0.26), (1, 0.48)):
        shift = depth * 1.5
        _token(ax, cx - 6.0 + shift, mid + shift, w=5.4, h=4.4, color=green,
               alpha=alpha, zorder=2)
        _token(ax, cx + 5.0 + shift, mid + shift, w=5.4, h=4.4, color=green,
               alpha=alpha, zorder=2)
    _token(ax, cx - 6.0, mid, w=5.4, h=4.4, color=green)
    _token(ax, cx + 5.0, mid, w=5.4, h=4.4, color=green)
    for label, offset in (("1", -6.0), ("2", 5.0)):
        ax.text(cx + offset, mid, label, ha="center", va="center",
                fontsize=FS_SMALL, color=COLORS["white"], fontweight="semibold",
                zorder=6)
    _arrow(ax, (cx - 3.0, mid - 2.4), (cx + 2.0, mid - 2.4), color=green,
           lw=1.3, rad=-0.42, head=2.1, zorder=6)
    ax.text(cx - 0.5, mid - 6.2, "× many", ha="center", va="center",
            fontsize=FS_SMALL, color=FAINT)

    # --- beat 3: the link is a prediction ---------------------------------
    cx = centres[2]
    _token(ax, cx - 5.6, mid, w=5.4, h=4.4, color=green)
    ax.text(cx - 5.6, mid, "1", ha="center", va="center", fontsize=FS_SMALL,
            color=COLORS["white"], fontweight="semibold", zorder=6)
    ax.add_patch(Rectangle((cx + 5.6 - 2.7, mid - 2.2), 5.4, 4.4,
                           facecolor=COLORS["white"], edgecolor=green,
                           linewidth=0.8, linestyle=(0, (1.5, 1.2)), zorder=3))
    ax.text(cx + 5.6, mid, "2", ha="center", va="center", fontsize=FS_SMALL,
            color=green, fontweight="semibold", zorder=6)
    _arrow(ax, (cx - 2.6, mid), (cx + 2.5, mid), color=green, lw=1.3, head=2.1)
    ax.text(cx + 5.6, mid - 4.4, "expected", ha="center", va="top",
            fontsize=FS_SMALL, color=green, style="italic")

    captions = (
        ("Coincidence", "elements that co-occur\nbecome one token"),
        ("Repetition", "tokens that recur in order\nbecome a link"),
        ("Prediction", "a link is a prediction\nof what comes next"),
    )
    for cx, (title, gloss) in zip(centres, captions):
        ax.text(cx, height * 0.245, title, ha="center", va="center",
                fontsize=FS_STAGE, fontweight="semibold", color=STRUCTURE)
        ax.text(cx, height * 0.075, gloss, ha="center", va="center",
                fontsize=FS_SMALL, color=FAINT, linespacing=1.35)
    for left, right in zip(centres[:-1], centres[1:]):
        _arrow(ax, ((left + right) / 2 - 2.2, mid),
               ((left + right) / 2 + 2.2, mid), color=STRUCTURE, lw=0.8,
               head=2.2)

    _heading(
        ax, height,
        "An auditory object is a learned predictive dependency",
        "the definition this paper works from,\nin three steps and no more",
        title_mm=6.6, subtitle_mm=1.9, letter="B", letter_mm=10.6,
        width_mm=width)


# ---------------------------------------------------------------------------
# Panel C — the circuit
# ---------------------------------------------------------------------------
def _panel_circuit(fig, spec) -> None:
    """The architecture, with plasticity drawn as a state rather than a label.

    Four stages are named for what they compute.  The three that live in the
    layer-1 rate model are ``novelty``, ``dependency`` and ``balance``; the
    fourth, ``composition``, is the downstream readout.  It is drawn reading
    the coincidence map, which is what it does — the manuscript does not
    demonstrate that it depends on layer 1's learned weights, so it is not
    drawn as a hierarchy resting on them.
    """

    ax, width, height = _mm_axes(fig, spec)

    # Four channels rather than five: the architecture claim is per-channel
    # excitation, per-channel inhibition and plastic lateral coupling, and the
    # drawing is generic over N, which differs by paradigm (2 for AB/BA, 5 for
    # roving, 14 for Saffran, 37 for figure-ground).  Four leaves the lateral
    # arcs enough vertical room to be read as trained versus untrained.
    channels = 4
    y_step = 6.7
    ys = [35.4 - index * y_step for index in range(channels)]
    x_gate, x_exc, x_inh = 22.0, 50.0, 86.0
    cell_r = 1.9

    stages = (
        ("depression", "Novelty", "is this new?", 10.0, 32.0),
        ("plasticity", "Dependency", "what goes with what, what next?",
         32.0, 66.0),
        ("inhibition", "Balance", "suppress or enhance?", 66.0, 100.0),
        ("readout", "Composition", "is this a multi-element object?",
         100.0, width - 0.5),
    )
    band_low, band_high = 10.6, 38.8
    for key, name, question, left, right in stages:
        ax.add_patch(Rectangle((left, band_low), right - left,
                               band_high - band_low, facecolor=MECH_WASH[key],
                               edgecolor="none", zorder=0))
        ax.text((left + right) / 2, band_high + 4.9, name, ha="center",
                va="bottom", fontsize=FS_STAGE, fontweight="semibold",
                color=MECH[key])
        ax.text((left + right) / 2, band_high + 1.6, question, ha="center",
                va="bottom", fontsize=FS_SMALL, color=MECH[key], style="italic")

    # --- thalamocortical input, gated by short-term depression ------------
    ax.text(3.4, (ys[0] + ys[-1]) / 2, "Tonotopic axis", rotation=90,
            ha="center", va="center", fontsize=FS_ANNOT, color=STRUCTURE)
    _arrow(ax, (6.6, ys[-1] - 2.4), (6.6, ys[0] + 2.4), color=FAINT, lw=0.5,
           head=1.7)

    for y in ys:
        _line(ax, (10.6, y), (x_gate - 2.6, y), color=STRUCTURE, lw=0.6,
              zorder=2)
        _arrow(ax, (x_gate + 2.6, y), (x_exc - cell_r - 0.6, y),
               color=STRUCTURE, lw=0.6, head=1.8, zorder=2)
        # The depressing synapse: successive responses to the same input get
        # smaller.  Drawn, not merely labelled.
        ax.add_patch(Rectangle((x_gate - 2.6, y - 1.75), 5.2, 3.5,
                               facecolor=MECH["depression"], edgecolor="none",
                               zorder=3))
        for step, bar_x in enumerate((-1.5, -0.1, 1.3)):
            bar_h = 2.4 * (0.66 ** step)
            ax.plot([x_gate + bar_x] * 2, [y - 1.2, y - 1.2 + bar_h],
                    color=COLORS["white"], lw=0.6, solid_capstyle="butt",
                    zorder=4)

    # --- excitatory cells and the plastic lateral connections -------------
    for y in ys:
        ax.add_patch(Circle((x_exc, y), cell_r, facecolor=MECH["plasticity"],
                            edgecolor="none", zorder=6))
        ax.text(x_exc, y, "E", ha="center", va="center", fontsize=FS_SMALL,
                color=COLORS["white"], fontweight="semibold", zorder=7)

    # One hub channel, so the reader sees learned and unlearned links side by
    # side.  Trained links are solid, untrained dotted: plasticity is a state
    # of the circuit here, not a caption on it.
    # Endpoints sit clear of both somata (2.6 mm from centre against a 1.9 mm
    # radius), so an arc reads as a connection between two cells rather than
    # as a stroke passing behind them.
    hub = 1
    for target, trained in ((0, True), (2, True), (3, False)):
        rad = -1.05 if abs(target - hub) == 1 else -0.80
        sign = 1.0 if ys[target] > ys[hub] else -1.0
        _arrow(ax, (x_exc - 2.3, ys[hub] + sign * 1.3),
               (x_exc - 2.3, ys[target] - sign * 1.3),
               color=MECH["plasticity"], lw=1.3 if trained else 0.9,
               rad=rad, head=2.1 if trained else 1.8,
               style="-" if trained else (0, (1.2, 1.1)),
               alpha=1.0 if trained else 0.9, zorder=5)

    # --- the fixed excitatory-inhibitory loop -----------------------------
    for y in ys:
        ax.add_patch(Circle((x_inh, y), cell_r, facecolor=COLORS["white"],
                            edgecolor=MECH["inhibition"], linewidth=1.0,
                            zorder=6))
        ax.text(x_inh, y, "I", ha="center", va="center", fontsize=FS_SMALL,
                color=MECH["inhibition"], fontweight="semibold", zorder=7)
        _arrow(ax, (x_exc + 2.4, y + 1.0), (x_inh - 2.5, y + 1.0),
               color=MECH["inhibition"], lw=0.65, head=1.8, zorder=4)
        _line(ax, (x_inh - 2.4, y - 1.0), (x_exc + 3.0, y - 1.0),
              color=MECH["inhibition"], lw=0.65, zorder=4)
        _bar_terminal(ax, (x_exc + 2.8, y - 1.0), (-1.0, 0.0),
                      color=MECH["inhibition"], lw=1.0, half=1.1)

    # Weak lateral inhibition, shown once rather than as a full matrix: each I
    # inhibits mostly its own channel, but the lateral coupling is non-zero.
    # Drawn as short branches off one I->E line, so it reads as subordinate to
    # the self connection instead of competing with it.
    branch_x = x_exc + 9.0
    for target in (0, 2):
        _line(ax, (branch_x, ys[1] - 1.0), (x_exc + 4.4, ys[target] - 1.0),
              color=MECH["inhibition"], lw=0.45, rad=0.0, zorder=3, alpha=0.7)
        _bar_terminal(ax, (x_exc + 4.2, ys[target] - 1.0), (-1.0, 0.0),
                      color=MECH["inhibition"], lw=0.8, half=0.8, zorder=3)
    # A junction dot, so the pair reads as a branch off this channel's own
    # I->E line rather than as two lines meeting nothing.
    ax.plot([branch_x], [ys[1] - 1.0], marker="o", ms=1.7,
            color=MECH["inhibition"], mec="none", zorder=5)
    ax.text(branch_x + 1.4, ys[1] - 3.0, "weak lateral", ha="left", va="top",
            fontsize=FS_SMALL, color=MECH["inhibition"], alpha=0.9)

    # --- the downstream multiscale readout --------------------------------
    x_map, x_bank, x_units = 113.0, 137.0, 160.0
    y_mid = (ys[0] + ys[-1]) / 2
    rng = np.random.default_rng(11)
    grid = rng.random((5, 5)) ** 2.1
    cell = 1.7
    for row in range(5):
        for column in range(5):
            ax.add_patch(Rectangle(
                (x_map - 2.5 * cell + column * cell,
                 y_mid + 2.5 * cell - (row + 1) * cell), cell, cell,
                facecolor=MECH["readout"],
                alpha=0.14 + 0.72 * grid[row, column],
                edgecolor=COLORS["white"], linewidth=0.25, zorder=4))
    _arrow(ax, (x_map + 2.5 * cell + 1.0, y_mid), (x_bank - 6.2, y_mid),
           color=MECH["readout"], lw=0.7, head=1.9)
    taus = (0.85, 1.9, 4.2, 9.0)
    shades = plt.get_cmap("BuPu")(np.linspace(0.42, 0.92, len(taus)))
    curve_x = np.linspace(0, 9.6, 60)
    for index, tau in enumerate(taus):
        base = y_mid + 3.9 - index * 2.9
        ax.plot(x_bank - 5.0 + curve_x, base + 2.0 * np.exp(-curve_x / tau),
                color=shades[index], lw=0.85, zorder=5)
        ax.plot([x_bank - 5.0, x_bank + 4.8], [base] * 2, color=FAINT, lw=0.3,
                zorder=3)

    _arrow(ax, (x_bank + 6.0, y_mid), (x_units - 3.2, y_mid),
           color=MECH["readout"], lw=0.7, head=1.9)
    for offset in (3.8, 0.0, -3.8):
        ax.add_patch(Circle((x_units, y_mid + offset), 1.6,
                            facecolor=MECH["readout"], edgecolor="none",
                            zorder=6))

    # All three readout captions sit on one baseline beneath their elements,
    # so nothing in this stage has a label above it competing with the stage
    # heading.
    caption_y = y_mid - 6.6
    for x, caption in ((x_map, "coincidence map\npost-rate × pre-trace"),
                       (x_bank, "bank of timescales"),
                       (x_units, "multi-element units")):
        ax.text(x, caption_y, caption, ha="center", va="top",
                fontsize=FS_SMALL, color=MECH["readout"], linespacing=1.35)

    # --- legend, two rows, and the plastic-versus-fixed statement ---------
    legend_columns = (9.0, 45.0, 81.0)
    legend_rows = (8.6, 3.8)
    items = (
        ("cell_filled", MECH["plasticity"], "excitatory cell"),
        ("solid", MECH["plasticity"], "plastic E→E, trained"),
        ("gate", MECH["depression"], "depressing synapse"),
        ("cell_open", MECH["inhibition"], "inhibitory cell"),
        ("dotted", MECH["plasticity"], "plastic E→E, untrained"),
        ("fixed", MECH["inhibition"], "fixed connection"),
    )
    for index, (kind, colour, label) in enumerate(items):
        x = legend_columns[index % 3]
        y = legend_rows[index // 3]
        if kind.startswith("cell"):
            ax.add_patch(Circle(
                (x + 1.5, y), 1.5,
                facecolor=colour if kind.endswith("filled") else COLORS["white"],
                edgecolor="none" if kind.endswith("filled") else colour,
                linewidth=0.9, zorder=5))
        elif kind == "gate":
            ax.add_patch(Rectangle((x - 0.4, y - 1.5), 4.0, 3.0,
                                   facecolor=colour, edgecolor="none",
                                   zorder=5))
        elif kind == "fixed":
            _line(ax, (x - 0.4, y), (x + 3.4, y), color=colour, lw=0.75)
            _bar_terminal(ax, (x + 3.3, y), (1.0, 0.0), color=colour, lw=1.0,
                          half=1.15)
        else:
            _arrow(ax, (x - 0.4, y), (x + 3.8, y), color=colour,
                   lw=1.25 if kind == "solid" else 0.85,
                   style="-" if kind == "solid" else (0, (1.1, 1.1)), head=1.9)
        ax.text(x + 5.6, y, label, ha="left", va="center", fontsize=FS_SMALL,
                color=STRUCTURE)

    ax.text(width - 0.5, 6.2,
            "Only E→E connections are plastic.\nE→I and I→E are fixed.",
            ha="right", va="center", fontsize=FS_SMALL, color=STRUCTURE,
            linespacing=1.4, fontweight="semibold")

    _heading(
        ax, height,
        "Three memories in one cortical circuit, and a stage that reads it",
        "depression is unlearned and channel-specific · recurrent excitation "
        "is learned and cross-channel · inhibition is fixed and slow",
        title_mm=6.6, subtitle_mm=3.0, letter="C", letter_mm=10.6,
        width_mm=width)


# ---------------------------------------------------------------------------
# Panel D — the reading key, measured
# ---------------------------------------------------------------------------
#: Channel A is the lighter tint of each row's mechanism colour, channel B the
#: full-strength one, because B is the channel the prediction lands on.
D_ROWS = (
    ("stim", "Input\npulses", (FAINT, STRUCTURE)),
    ("tm_in", "Thalamic drive\nwith depression", ("#DFAFAF", MECH["depression"])),
    ("E", "Excitatory\nrate", ("#8FB6A9", MECH["plasticity"])),
    ("rec_E", "Recurrent\nprediction", ("#8FB6A9", MECH["plasticity"])),
    ("inh_to_E", "Inhibitory\ncurrent", ("#9EC1C0", MECH["inhibition"])),
)


def _panel_reading_key(fig, spec, data) -> list:
    """Real ``model0`` traces on one clock, and the timescale gap that matters.

    Every trace is simulator output.  The two time constants printed here are
    measured back out of these traces as the 1/e fall from each signal's own
    peak, so the annotation is a property of the plotted data rather than a
    restatement of the configuration.
    """

    nested = spec.subgridspec(5, 1, hspace=0.34)
    axes = [fig.add_subplot(nested[row, 0]) for row in range(5)]

    time_ms = np.asarray(data["time_ms"], dtype=float)
    tone_a = np.asarray(data["tone_a_ms"], dtype=float)
    tone_b = np.asarray(data["tone_b_ms"], dtype=float)
    scalar = {key: float(np.asarray(data[key]).ravel()[0]) for key in (
        "tau_E_measured_ms", "tau_I_measured_ms", "tau_E_config_ms",
        "tau_I_config_ms", "inh_at_tone_b_onset", "inh_peak",
        "exc_at_tone_b_onset", "W_BA", "W_AB", "n_sequences_averaged")}
    ch_a, ch_b = 0, 1

    for index, (ax, (key, label, (light, full))) in enumerate(
            zip(axes, D_ROWS)):
        trace = np.asarray(data[key], dtype=float)
        last = index == len(D_ROWS) - 1

        for lo, hi, shade in ((tone_a[0], tone_a[1], COLORS["sage"]),
                              (tone_b[0], tone_b[1], COLORS["peach"])):
            ax.axvspan(lo, hi, color=shade, alpha=0.85, lw=0, zorder=0)

        if key == "stim":
            for channel, colour in ((ch_a, light), (ch_b, full)):
                ax.fill_between(time_ms, 0, trace[channel], color=colour,
                                alpha=0.95, lw=0, zorder=3, step="mid")
            ax.set_ylim(0, 1.5)
            ax.set_yticks([])
        else:
            for channel, colour in ((ch_a, light), (ch_b, full)):
                ax.plot(time_ms, trace[channel], color=colour, lw=1.2,
                        zorder=4 if channel == ch_b else 3)
            peak = float(trace.max())
            ax.set_ylim(-0.03 * peak, peak * 1.42)
            tick = float(f"{peak / 2:.1g}")
            ax.set_yticks([0, tick])

        ax.set_xlim(0, time_ms[-1])
        ax.set_ylabel(label, fontsize=FS_ANNOT, labelpad=3.0, linespacing=1.3,
                      rotation=0, ha="right", va="center")
        clean_axis(ax, bottom=last)
        ax.tick_params(axis="y", labelsize=FS_TICK, length=1.5, pad=1.2)
        if last:
            ax.set_xlabel("Time from sequence onset (ms)", fontsize=6.8,
                          labelpad=1.4)
            ax.set_xticks([0, 50, 80, 130, 200, 300])
            ax.tick_params(axis="x", labelsize=FS_TICK, length=1.8, pad=1.4)
        else:
            ax.set_xticks([])
            ax.tick_params(axis="x", length=0)

    for lo, hi, name in ((tone_a[0], tone_a[1], "tone A"),
                         (tone_b[0], tone_b[1], "tone B")):
        axes[0].text((lo + hi) / 2, 1.06, name,
                     transform=axes[0].get_xaxis_transform(), ha="center",
                     va="bottom", fontsize=FS_SMALL, fontweight="semibold",
                     color=STRUCTURE)
    axes[0].text(0.995, 0.52, "channel A pale · channel B full",
                 transform=axes[0].transAxes, ha="right", va="center",
                 fontsize=FS_SMALL, color=FAINT)

    # --- one right-aligned quantity per row, and two short pointers -------
    # Written without mathtext on purpose: a mathtext subscript renders at 0.7
    # of the base size, which would put 3.6 pt type on the page, and it falls
    # back to DejaVu for the italic tau, breaking the Arial-only embedding.
    axes[2].text(0.995, 0.94,
                 f"excitation: 1/e in {scalar['tau_E_measured_ms']:.0f} ms "
                 f"(excitatory τ = {scalar['tau_E_config_ms']:.0f} ms)",
                 transform=axes[2].transAxes, ha="right", va="top",
                 fontsize=FS_SMALL, color=MECH["plasticity"])
    axes[3].text(0.995, 0.94,
                 f"W[B←A] = {scalar['W_BA']:.2f}    W[A←B] = "
                 f"{scalar['W_AB']:.2f}",
                 transform=axes[3].transAxes, ha="right", va="top",
                 fontsize=FS_SMALL, color=MECH["plasticity"])
    axes[4].text(0.995, 0.94,
                 f"inhibition: 1/e in {scalar['tau_I_measured_ms']:.0f} ms "
                 f"(inhibitory τ = {scalar['tau_I_config_ms']:.0f} ms)",
                 transform=axes[4].transAxes, ha="right", va="top",
                 fontsize=FS_SMALL, color=MECH["inhibition"])

    recurrent = np.asarray(data["rec_E"], dtype=float)[ch_b]
    axes[3].text(tone_a[1] + 3.0, recurrent.max() * 1.02,
                 "B is driven before B sounds", ha="left", va="center",
                 fontsize=FS_SMALL, color=MECH["plasticity"])

    # The interval the panel exists for: excitation has gone, inhibition has
    # not, and the next tone lands inside that window.  The number itself is
    # stated in the subtitle, where the claim belongs; the panel carries only
    # the marker that shows where it is read.
    for ax in (axes[2], axes[4]):
        ax.axvspan(tone_a[1], tone_b[0], facecolor=COLORS["mist"], alpha=0.95,
                   lw=0, zorder=1)
    carried = scalar["inh_at_tone_b_onset"] / scalar["inh_peak"]
    axes[4].plot([tone_b[0]], [scalar["inh_at_tone_b_onset"]], marker="o",
                 ms=3.0, color=MECH["inhibition"], mec=COLORS["white"],
                 mew=0.5, zorder=6)
    axes[4].text(tone_b[0] - 2.5, scalar["inh_at_tone_b_onset"] * 1.55,
                 f"{carried:.0%} of peak", ha="right", va="bottom",
                 fontsize=FS_SMALL, color=MECH["inhibition"])
    axes[2].text(tone_a[1] + 12.0, 0.88, "excitation gone", ha="left",
                 va="top", transform=axes[2].get_xaxis_transform(),
                 fontsize=FS_SMALL, color=FAINT)

    width, height = _axes_size_mm(fig, axes[0])
    _heading(
        axes[0], height,
        "Inhibition outlives the excitation that caused it",
        f"measured model0 output · mean of {scalar['n_sequences_averaged']:.0f} "
        "AB sequences once the link is learned · no smoothing or rescaling\n"
        f"at tone-B onset the inhibitory current is still {carried:.0%} of its "
        "peak while the excitation that drove it has returned to zero",
        title_mm=8.4, subtitle_mm=3.0, letter="D", letter_mm=12.8,
        letter_x_mm=-22.0, width_mm=width)
    return axes


# ---------------------------------------------------------------------------
# Panel E — the four paradigms
# ---------------------------------------------------------------------------
PARADIGMS = (
    ("Roving", "novelty",
     "one token repeats, then changes",
     "does novelty need learned structure?"),
    ("AB/BA", "dependency",
     "two tones in a fixed order",
     "is the order itself learned?"),
    ("SFG", "selection",
     "coherent tones inside a cloud",
     "which grouping is expressed?"),
    ("Saffran", "composition",
     "continuous stream, statistical words",
     "are links composed into objects?"),
)


def _panel_paradigms(fig, spec) -> None:
    """The reader's map of the Results, in the order the paper takes them."""

    ax, width, height = _mm_axes(fig, spec)

    column = width / 4.0
    thumb_h = 9.0
    thumb_y = height - 0.5 - thumb_h
    rng = np.random.default_rng(7)

    for index, (name, stage, stimulus, question) in enumerate(PARADIGMS):
        left = index * column
        cx = left + column / 2
        tx0 = left + 5.0
        span = column - 12.0

        def slot(fraction, low=tx0, size=span):
            return low + fraction * size

        if name == "Roving":
            for step in range(6):
                _token(ax, slot(step * 0.135), thumb_y + thumb_h * 0.30,
                       w=span * 0.10, h=2.1, color=COLORS["rep15"])
            _token(ax, slot(0.81), thumb_y + thumb_h * 0.76, w=span * 0.10,
                   h=2.1, color=COLORS["rep1"])
            ax.text(slot(0.90), thumb_y + thumb_h * 0.76, "new", ha="left",
                    va="center", fontsize=FS_SMALL, color=COLORS["rep1"])
        elif name == "AB/BA":
            for pair in range(3):
                base = pair * 0.30
                _token(ax, slot(base + 0.04), thumb_y + thumb_h * 0.26,
                       w=span * 0.095, h=2.1, color=COLORS["rep15"])
                _token(ax, slot(base + 0.17), thumb_y + thumb_h * 0.70,
                       w=span * 0.095, h=2.1, color=COLORS["rep15"])
            _arrow(ax, (slot(0.085), thumb_y + thumb_h * 0.36),
                   (slot(0.135), thumb_y + thumb_h * 0.60),
                   color=MECH["plasticity"], lw=0.8, rad=-0.4, head=1.7)
            ax.text(slot(1.0), thumb_y + thumb_h * 0.48, "A→B", ha="right",
                    va="center", fontsize=FS_SMALL, color=MECH["plasticity"])
        elif name == "SFG":
            for step in range(11):
                for _ in range(2):
                    _token(ax, slot(step * 0.093),
                           thumb_y + rng.uniform(1.2, thumb_h - 1.2),
                           w=span * 0.055, h=1.4, color=FAINT, alpha=0.5)
            for step in (2, 4, 6, 8):
                for level in (0.30, 0.56, 0.82):
                    _token(ax, slot(step * 0.093), thumb_y + thumb_h * level,
                           w=span * 0.055, h=1.4, color=COLORS["model"])
        else:
            palette = (COLORS["decoder"], COLORS["teal"], COLORS["terracotta"])
            heights = (0.24, 0.50, 0.76)
            position = 0.0
            for word in (0, 1, 2, 0, 2, 1):
                for token in range(3):
                    _token(ax, slot(position), thumb_y + thumb_h *
                           heights[(token + word) % 3], w=span * 0.042, h=1.4,
                           color=palette[word])
                    position += 0.056

        ax.text(cx, thumb_y - 1.9, name, ha="center", va="top",
                fontsize=FS_STAGE, fontweight="semibold", color=STRUCTURE)
        ax.text(cx, thumb_y - 4.7, stimulus, ha="center", va="top",
                fontsize=FS_SMALL, color=STRUCTURE)
        ax.text(cx, thumb_y - 7.3, question, ha="center", va="top",
                fontsize=FS_SMALL, color=FAINT, style="italic")

    arrow_y = 4.6
    _arrow(ax, (column * 0.5, arrow_y), (width - column * 0.42, arrow_y),
           color=FAINT, lw=0.6, head=2.4, zorder=2)
    for index, (_name, stage, _stimulus, _question) in enumerate(PARADIGMS):
        cx = index * column + column / 2
        ax.plot([cx], [arrow_y], marker="o", ms=3.0, color=COLORS["white"],
                mec=FAINT, mew=0.6, zorder=3)
        ax.text(cx, arrow_y - 1.6, stage, ha="center", va="top",
                fontsize=FS_SMALL, color=STRUCTURE, fontweight="semibold")

    _heading(
        ax, height,
        "Four paradigms, each starting where the last one stops",
        "depression cannot represent order · a learned link cannot choose "
        "between organisations · pairwise links are not multi-element objects",
        title_mm=6.4, subtitle_mm=2.0, letter="E", letter_mm=10.4,
        width_mm=width)


# ---------------------------------------------------------------------------
# Panel F — the prediction matrix
# ---------------------------------------------------------------------------
#: ``required`` / ``not_needed`` / ``untested`` / ``not_engaged``, and whether
#: the manuscript already measures the cell.  The measured values are quoted in
#: README.md.  Three of them come out against the intuitive prediction —
#: inhibitory structure in roving and in SFG, and layer-1 plasticity in the
#: Saffran composition measure — and the matrix shows them as measured rather
#: than promising an effect the Results then withdraw.
MATRIX = {
    "Roving": {
        "depression": ("required", True),
        "plasticity": ("not_needed", True),
        "inhibition": ("not_needed", True),
        "readout": ("not_engaged", False),
    },
    "AB/BA": {
        "depression": ("not_needed", False),
        "plasticity": ("required", False),
        "inhibition": ("required", False),
        "readout": ("not_engaged", False),
    },
    "SFG": {
        "depression": ("untested", False),
        "plasticity": ("required", True),
        "inhibition": ("not_needed", True),
        "readout": ("not_engaged", False),
    },
    "Saffran": {
        "depression": ("untested", False),
        "plasticity": ("not_needed", True),
        "inhibition": ("untested", False),
        "readout": ("required", True),
    },
}
MECH_ORDER = ("depression", "plasticity", "inhibition", "readout")


def _state_glyph(ax, x, y, state, colour, *, radius=1.9) -> None:
    """One prediction, as a visual state rather than a tick or a cross."""

    if state == "required":
        ax.add_patch(Circle((x, y), radius, facecolor=colour, edgecolor="none",
                            zorder=4))
    elif state == "not_needed":
        ax.add_patch(Circle((x, y), radius, facecolor=COLORS["white"],
                            edgecolor=colour, linewidth=1.0, zorder=4))
    elif state == "untested":
        ax.add_patch(Circle((x, y), radius, facecolor=COLORS["mist"],
                            edgecolor=FAINT, linewidth=0.5, hatch="////",
                            zorder=4))
    else:
        ax.plot([x - radius * 0.7, x + radius * 0.7], [y, y], color=FAINT,
                lw=0.7, alpha=0.8, zorder=4)


def _panel_matrix(fig, spec) -> list:
    """Predictions entering the Results, with the untested cells visible."""

    nested = spec.subgridspec(1, 2, width_ratios=(1.0, 0.46), wspace=0.02)
    ax, width, height = _mm_axes(fig, nested[0, 0])
    ax_key, key_width, key_height = _mm_axes(fig, nested[0, 1])

    label_column = 20.0
    column_span = (width - label_column - 2.0) / len(MECH_ORDER)
    xs = [label_column + column_span * (index + 0.5)
          for index in range(len(MECH_ORDER))]
    row_top = height - 7.6
    row_step = (row_top - 1.0) / len(PARADIGMS)
    ys = [row_top - row_step * (index + 0.5) for index in range(len(PARADIGMS))]

    for index, key in enumerate(MECH_ORDER):
        ax.text(xs[index], row_top + 1.4, MECH_LABEL[key], ha="center",
                va="bottom", fontsize=FS_SMALL, color=MECH[key],
                fontweight="semibold", linespacing=1.3)

    for row, (name, _stage, _stimulus, _question) in enumerate(PARADIGMS):
        y = ys[row]
        if row % 2 == 0:
            ax.add_patch(Rectangle((0.0, y - row_step / 2), width - 2.0,
                                   row_step, facecolor=COLORS["linen"],
                                   alpha=0.6, edgecolor="none", zorder=0))
        ax.text(label_column - 3.4, y, name, ha="right", va="center",
                fontsize=FS_ANNOT, color=STRUCTURE, fontweight="semibold")
        for column, key in enumerate(MECH_ORDER):
            state, measured = MATRIX[name][key]
            _state_glyph(ax, xs[column], y, state, MECH[key])
            if measured:
                ax.add_patch(Rectangle((xs[column] + 1.7, y - 2.6), 1.2, 1.2,
                                       facecolor=STRUCTURE, edgecolor="none",
                                       zorder=6))

    entries = (
        ("required", "predicted required"),
        ("not_needed", "predicted not needed"),
        ("untested", "not yet tested"),
        ("not_engaged", "not engaged by this paradigm"),
    )
    y_cursor = key_height - 3.0
    step = 4.4
    for state, label in entries:
        _state_glyph(ax_key, 3.0, y_cursor, state, STRUCTURE, radius=1.7)
        ax_key.text(6.6, y_cursor, label, ha="left", va="center",
                    fontsize=FS_SMALL, color=STRUCTURE)
        y_cursor -= step
    ax_key.add_patch(Rectangle((2.4, y_cursor - 0.6), 1.2, 1.2,
                               facecolor=STRUCTURE, edgecolor="none"))
    ax_key.text(6.6, y_cursor, "already measured in this manuscript",
                ha="left", va="center", fontsize=FS_SMALL, color=STRUCTURE)

    _heading(
        ax, height,
        "Each paradigm is predicted to need a different subset",
        "predictions entering the Results, not results · the synthesis figure "
        "revisits this matrix with measured effect sizes",
        title_mm=6.4, subtitle_mm=2.0, letter="F", letter_mm=10.4,
        width_mm=width)
    return [ax, ax_key]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _band(top_mm: float, height_mm: float) -> tuple[float, float]:
    """Convert a millimetre band measured from the page top into fractions."""

    return (1.0 - top_mm / FIGURE_HEIGHT_MM,
            1.0 - (top_mm + height_mm) / FIGURE_HEIGHT_MM)


def build_figure(*, force_data: bool = False) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_panel_d_data(force=force_data, data_dir=DATA_DIR)

    left, right = 0.062, 0.985
    with manuscript_style():
        # Hatch strokes default to 1 pt, which is heavy inside a 3.8 mm glyph.
        plt.rcParams["hatch.linewidth"] = 0.45
        fig = plt.figure(figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)))

        # Bands are placed in millimetres from the page top.  Each gap holds a
        # panel's heading block, whose height is known because headings are
        # themselves offset in millimetres.
        top, bottom = _band(15.0, 29.0)
        head = fig.add_gridspec(1, 2, left=left, right=right, top=top,
                                bottom=bottom, wspace=0.14)
        _panel_problem(fig, head[0, 0])
        _panel_definition(fig, head[0, 1])

        top, bottom = _band(61.0, 45.0)
        circuit = fig.add_gridspec(1, 1, left=left, right=right, top=top,
                                   bottom=bottom)
        _panel_circuit(fig, circuit[0, 0])

        top, bottom = _band(120.0, 40.0)
        key = fig.add_gridspec(1, 1, left=0.118, right=right, top=top,
                               bottom=bottom)
        _panel_reading_key(fig, key[0, 0], data)

        top, bottom = _band(176.0, 26.0)
        paradigms = fig.add_gridspec(1, 1, left=left, right=right, top=top,
                                     bottom=bottom)
        _panel_paradigms(fig, paradigms[0, 0])

        top, bottom = _band(213.0, 29.0)
        matrix = fig.add_gridspec(1, 1, left=left, right=right, top=top,
                                  bottom=bottom)
        _panel_matrix(fig, matrix[0, 0])

        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    provenance = {
        "figure": "Figure 1 - the claim, the circuit, and the matrix",
        "final_size_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "claim": (
            "An auditory object is a learned predictive dependency among "
            "sound elements; grouping, prediction and deviance detection are "
            "manifestations of one circuit."
        ),
        "colour_key": dict(MECH),
        "panel_sources": {
            "A": "schematic - one scene, coincidence and continuation links",
            "B": "schematic - the definition in three steps",
            "C": "schematic - the model0 architecture plus the layer-2 readout",
            "D": (
                "MEASURED - model0 on the AB/BA paradigm, mean of the late AB "
                "sequences; time constants read back out of the traces"
            ),
            "E": "schematic - the four paradigms and the progression",
            "F": "prediction matrix; already-measured cells marked",
        },
        "matrix": {
            paradigm: {mech: {"state": state, "measured": measured}
                       for mech, (state, measured) in row.items()}
            for paradigm, row in MATRIX.items()
        },
        "inputs": {},
        "outputs": {},
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }
    for name in ("panel_d_traces.npz", "panel_d_provenance.json"):
        path = DATA_DIR / name
        if path.exists():
            provenance["inputs"][name] = {"path": str(path.resolve()),
                                          "sha256": _sha256(path)}
    for kind, path in paths.items():
        provenance["outputs"][kind] = {"path": str(path.resolve()),
                                       "sha256": _sha256(path)}
    provenance_path = DATA_DIR / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    paths["provenance"] = provenance_path
    return paths


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-data", action="store_true",
                        help="Repeat the panel-D simulation.")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    paths = build_figure(force_data=_parse_args(arguments).force_data)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
