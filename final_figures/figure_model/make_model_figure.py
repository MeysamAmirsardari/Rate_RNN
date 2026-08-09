"""Build the two-layer model definition figure.

Layer 1 is a rate model of A1 carrying three memories; layer 2 is a multiscale
readout that composes what layer 1 emits into multi-element objects.  The
figure exists to make the *interface* between them explicit, because that
interface is load-bearing for the manuscript's claims:

    layer 2 reads the rate vector E, and never layer 1's weight matrix W.

Every array in panel b is the real object the implementation forms — the
filterbank state, the coincidence map and a learned mask are computed by
``Layer2MR`` itself in ``model_data.py``, not drawn.  Panel a is a schematic of
the circuit, with its measured time constants annotated.

The visual grammar, colour key and export settings are those of
``final_figures.style``, so this figure sits beside Figures 1-5 unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from final_figures.figure_model.model_data import (
    N_CHANNELS,
    WORD_NAMES,
    WORDS,
    build_model_data,
)
from final_figures.style import (
    COLORS,
    PATTERN_CMAP,
    export_figure,
    manuscript_style,
    mm,
    panel_label,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_STEM = OUTPUT_DIR / "figure_model_architecture"

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 138.0

#: The key Figure 1 establishes, reused unchanged.
MECH = {
    "depression": COLORS["terracotta"],
    "plasticity": COLORS["model"],
    "inhibition": COLORS["teal"],
    "readout": COLORS["decoder"],
}
MECH_WASH = {
    "depression": "#F7ECEC",
    "plasticity": "#E8F0EC",
    "inhibition": "#E7EFEF",
    "readout": "#EDEAF2",
}

STRUCTURE = COLORS["charcoal"]
FAINT = COLORS["ash"]

FS_STAGE = 6.6
FS_ANNOT = 5.6
FS_SMALL = 5.2


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Drawing surface and typography, shared with Figure 1
# ---------------------------------------------------------------------------
def _mm_axes(fig, spec):
    ax = fig.add_subplot(spec)
    box = ax.get_position()
    width = box.width * fig.get_figwidth() * 25.4
    height = box.height * fig.get_figheight() * 25.4
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax, width, height


def _heading(ax, height_mm: float, title: str, subtitle: str, *,
             title_mm: float = 5.2, subtitle_mm: float = 1.9,
             letter: str | None = None, letter_mm: float = 9.4,
             letter_x_mm: float = -6.4, width_mm: float = 100.0) -> None:
    ax.text(0, 1.0 + title_mm / height_mm, title, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.2, fontweight="semibold",
            color=COLORS["charcoal"], clip_on=False)
    ax.text(0, 1.0 + subtitle_mm / height_mm, subtitle, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.9, color=COLORS["ash"],
            linespacing=1.4, clip_on=False)
    if letter is not None:
        panel_label(ax, letter, x=letter_x_mm / width_mm,
                    y=1.0 + letter_mm / height_mm)


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
    dx, dy = direction
    norm = float(np.hypot(dx, dy)) or 1.0
    px, py = -dy / norm * half, dx / norm * half
    ax.plot([end[0] - px, end[0] + px], [end[1] - py, end[1] + py],
            color=color, lw=lw, solid_capstyle="butt", zorder=zorder)


def _matrix(ax, values, extent, *, vmax=None, cmap=None):
    """Place a real array into millimetre coordinates."""

    values = np.asarray(values, dtype=float)
    peak = float(vmax if vmax is not None else np.nanmax(values))
    peak = peak if peak > 0 else 1.0
    return ax.imshow(values, cmap=cmap or PATTERN_CMAP,
                     norm=Normalize(0, peak), extent=extent,
                     interpolation="nearest", origin="upper", zorder=3,
                     rasterized=True)


# ---------------------------------------------------------------------------
# Panel a — layer 1
# ---------------------------------------------------------------------------
def _panel_layer1(fig, spec, data) -> None:
    """The cortical circuit and the three memories it carries."""

    ax, width, height = _mm_axes(fig, spec)
    tau_e = float(np.asarray(data["tau_E_ms"]).ravel()[0])
    tau_i = float(np.asarray(data["tau_I_ms"]).ravel()[0])
    tau_std = np.asarray(data["tau_std_s"], dtype=float)

    channels = 4
    y_step = 6.6
    ys = [37.0 - index * y_step for index in range(channels)]
    x_gate, x_exc, x_inh = 23.0, 50.0, 82.0
    cell_r = 1.95

    ax.add_patch(Rectangle((9.0, 8.4), 84.0, 33.6, facecolor="#FAFAF8",
                           edgecolor=COLORS["mist"], linewidth=0.5, zorder=0))

    ax.text(3.6, (ys[0] + ys[-1]) / 2, "Tonotopic axis", rotation=90,
            ha="center", va="center", fontsize=FS_ANNOT, color=STRUCTURE)
    _arrow(ax, (6.8, ys[-1] - 2.6), (6.8, ys[0] + 2.6), color=FAINT, lw=0.5,
           head=1.7)

    for y in ys:
        _line(ax, (10.4, y), (x_gate - 2.7, y), color=STRUCTURE, lw=0.6,
              zorder=2)
        _arrow(ax, (x_gate + 2.7, y), (x_exc - cell_r - 0.7, y),
               color=STRUCTURE, lw=0.6, head=1.8, zorder=2)
        ax.add_patch(Rectangle((x_gate - 2.7, y - 1.8), 5.4, 3.6,
                               facecolor=MECH["depression"], edgecolor="none",
                               zorder=3))
        for step, bar_x in enumerate((-1.55, -0.1, 1.35)):
            bar_h = 2.5 * (0.66 ** step)
            ax.plot([x_gate + bar_x] * 2, [y - 1.25, y - 1.25 + bar_h],
                    color=COLORS["white"], lw=0.62, solid_capstyle="butt",
                    zorder=4)

    for y in ys:
        ax.add_patch(Circle((x_exc, y), cell_r, facecolor=MECH["plasticity"],
                            edgecolor="none", zorder=6))
        ax.text(x_exc, y, "E", ha="center", va="center", fontsize=FS_SMALL,
                color=COLORS["white"], fontweight="semibold", zorder=7)
        ax.add_patch(Circle((x_inh, y), cell_r, facecolor=COLORS["white"],
                            edgecolor=MECH["inhibition"], linewidth=1.0,
                            zorder=6))
        ax.text(x_inh, y, "I", ha="center", va="center", fontsize=FS_SMALL,
                color=MECH["inhibition"], fontweight="semibold", zorder=7)
        _arrow(ax, (x_exc + 2.5, y + 1.05), (x_inh - 2.6, y + 1.05),
               color=MECH["inhibition"], lw=0.65, head=1.8, zorder=4)
        _line(ax, (x_inh - 2.5, y - 1.05), (x_exc + 3.1, y - 1.05),
              color=MECH["inhibition"], lw=0.65, zorder=4)
        _bar_terminal(ax, (x_exc + 2.9, y - 1.05), (-1.0, 0.0),
                      color=MECH["inhibition"], lw=1.0, half=1.15)

    hub = 1
    for target, trained in ((0, True), (2, True), (3, False)):
        rad = -1.05 if abs(target - hub) == 1 else -0.80
        sign = 1.0 if ys[target] > ys[hub] else -1.0
        _arrow(ax, (x_exc - 2.35, ys[hub] + sign * 1.35),
               (x_exc - 2.35, ys[target] - sign * 1.35),
               color=MECH["plasticity"], lw=1.3 if trained else 0.9, rad=rad,
               head=2.1 if trained else 1.8,
               style="-" if trained else (0, (1.2, 1.1)),
               alpha=1.0 if trained else 0.9, zorder=5)

    labels = (
        (x_gate, MECH["depression"], "short-term depression",
         f"unlearned · per channel\nτ {tau_std[0] * 1e3:.0f} ms, "
         f"{tau_std[1] * 1e3:.0f} ms, {tau_std[2]:.0f} s"),
        (x_exc - 1.0, MECH["plasticity"], "plastic lateral E→E",
         "learned · cross-channel\nrate-STDP on W"),
        (x_inh, MECH["inhibition"], "fixed E→I and I→E",
         f"unlearned · slow\nτE {tau_e:.0f} ms vs τI {tau_i:.0f} ms"),
    )
    for x, colour, title, gloss in labels:
        ax.text(x, 6.4, title, ha="center", va="top", fontsize=FS_SMALL,
                color=colour, fontweight="semibold")
        ax.text(x, 3.6, gloss, ha="center", va="top", fontsize=FS_SMALL,
                color=FAINT, linespacing=1.35)

    equations = (
        f"τE dE/dt = −E + tm + W E − M_IE I,   E := relu(E)",
        f"τI dI/dt = −I + M_EI E",
        "tm = A U x s,   x = max(0, 1 − Σ w_k D_k)",
        "ΔW = ηLTP (Wmax − W) E trᵀ − ηLTD tr Eᵀ",
    )
    ax.text(108.0, 40.0, "State equations", ha="left", va="top",
            fontsize=FS_SMALL, color=STRUCTURE, fontweight="semibold")
    for index, line in enumerate(equations):
        ax.text(108.0, 35.6 - index * 3.4, line, ha="left", va="top",
                fontsize=FS_SMALL, color=STRUCTURE)
    ax.text(108.0, 20.4,
            "M_EI and M_IE are fixed: strong on the diagonal,\n"
            "weak off it. Only W is plastic.",
            ha="left", va="top", fontsize=FS_SMALL, color=FAINT,
            linespacing=1.4)
    ax.text(108.0, 12.0,
            "Three memories: an unlearned per-channel one,\n"
            "a learned cross-channel one, and a slow\n"
            "inhibitory state that decides how they show.",
            ha="left", va="top", fontsize=FS_SMALL, color=STRUCTURE,
            linespacing=1.4)

    _heading(
        ax, height,
        "Layer 1 — a cortical circuit carrying three memories",
        "one excitatory and one inhibitory cell per tonotopic channel · "
        "trained lateral connections solid, untrained dotted",
        title_mm=6.6, subtitle_mm=2.6, letter="a", letter_mm=10.8,
        width_mm=width)


# ---------------------------------------------------------------------------
# The interface — the whole point of drawing two layers
# ---------------------------------------------------------------------------
def _panel_interface(fig, spec, data) -> None:
    ax, width, height = _mm_axes(fig, spec)
    mid = height * 0.62

    ax.add_patch(Rectangle((0.0, mid - 3.4), width, 6.8,
                           facecolor=COLORS["linen"], edgecolor="none",
                           zorder=0))
    _arrow(ax, (50.0, mid + 3.8), (50.0, mid - 4.0), color=STRUCTURE, lw=1.1,
           head=2.6, zorder=4)
    ax.text(53.0, mid, "E", ha="left", va="center", fontsize=FS_STAGE,
            fontweight="semibold", color=STRUCTURE)
    ax.text(57.5, mid, "the rate vector,\none value per channel", ha="left",
            va="center", fontsize=FS_SMALL, color=STRUCTURE, linespacing=1.4)
    ax.text(width - 1.0, mid,
            "Layer 2 reads E only — never W.\n"
            "That is why composition survives with layer 1 frozen or removed.",
            ha="right", va="center", fontsize=FS_SMALL,
            color=MECH["readout"], fontweight="semibold", linespacing=1.4)


# ---------------------------------------------------------------------------
# Panel b — layer 2, drawn from the real arrays
# ---------------------------------------------------------------------------
def _panel_layer2(fig, spec, data) -> None:
    """Filterbank, coincidence map, learned mask and output, all measured."""

    ax, width, height = _mm_axes(fig, spec)

    rates = np.asarray(data["E"], dtype=float)
    s_state = np.asarray(data["s"], dtype=float)
    coincidence = np.asarray(data["D"], dtype=float)
    mask = np.asarray(data["M"], dtype=float)
    taus = np.asarray(data["taus"], dtype=float)
    channel = int(np.asarray(data["snapshot_channel"]).ravel()[0])
    word_index = int(np.asarray(data["snapshot_word"]).ravel()[0])
    unit = int(np.asarray(data["mask_unit"]).ravel()[0])
    n_rates = taus.size
    word = WORDS[word_index]

    top, bottom = height - 3.0, height - 14.0

    # ---- stage 1: the rate vector ---------------------------------------
    x_e0, x_e1 = 4.0, 11.0
    bar_h = (top - bottom) / N_CHANNELS
    peak = float(rates.max()) or 1.0
    for index in range(N_CHANNELS):
        y = top - (index + 0.5) * bar_h
        ax.add_patch(Rectangle(
            (x_e0, y - bar_h * 0.36), (x_e1 - x_e0) * rates[index] / peak,
            bar_h * 0.72, facecolor=MECH["plasticity"], edgecolor="none",
            zorder=3))
    ax.plot([x_e0, x_e0], [bottom, top], color=FAINT, lw=0.5, zorder=2)
    ax.text((x_e0 + x_e1) / 2, top + 1.4, "E", ha="center", va="bottom",
            fontsize=FS_STAGE, fontweight="semibold", color=STRUCTURE)
    ax.text(x_e1 + 0.8, top - (channel + 0.5) * bar_h, f"ch {channel}",
            ha="left", va="center", fontsize=FS_SMALL,
            color=MECH["plasticity"])
    ax.text((x_e0 + x_e1) / 2, bottom - 1.6, "rate now", ha="center",
            va="top", fontsize=FS_SMALL, color=FAINT)

    # ---- stage 2: the filterbank ----------------------------------------
    x_s0, x_s1 = 20.0, 27.0
    _arrow(ax, (x_e1 + 3.4, (top + bottom) / 2), (x_s0 - 1.6, (top + bottom) / 2),
           color=STRUCTURE, lw=0.7, head=1.9)
    _matrix(ax, s_state, (x_s0, x_s1, bottom, top))
    ax.text((x_s0 + x_s1) / 2, top + 1.2, "s", ha="center", va="bottom",
            fontsize=FS_STAGE, fontweight="semibold", color=STRUCTURE)
    ax.text((x_s0 + x_s1) / 2, bottom - 1.4,
            f"filterbank\n{n_rates} rates, τ {taus[0] * 1e3:.0f}–"
            f"{taus[-1] * 1e3:.0f} ms", ha="center", va="top",
            fontsize=FS_SMALL, color=FAINT, linespacing=1.35)

    # ---- stage 3: the coincidence map ------------------------------------
    # Written "outer" rather than with a tensor-product sign: Arial has no
    # CIRCLED TIMES, and substituting a fallback font for one glyph would
    # break the Arial-only embedding the rest of the series keeps.
    x_d0, x_d1 = 36.0, 168.0
    _arrow(ax, (x_s1 + 1.6, (top + bottom) / 2), (x_d0 - 1.6, (top + bottom) / 2),
           color=STRUCTURE, lw=0.7, head=1.9)
    _matrix(ax, coincidence, (x_d0, x_d1, bottom, top))
    block = (x_d1 - x_d0) / n_rates
    for index in range(1, n_rates):
        ax.plot([x_d0 + index * block] * 2, [bottom, top],
                color=COLORS["white"], lw=0.5, zorder=5)
    for index in range(n_rates):
        ax.text(x_d0 + (index + 0.5) * block, top + 0.7,
                f"{taus[index] * 1e3:.0f}", ha="center", va="bottom",
                fontsize=FS_SMALL, color=FAINT)
    ax.text(x_d1, top + 2.6, "filter τ (ms)", ha="right", va="bottom",
            fontsize=FS_SMALL, color=FAINT)
    ax.text(x_d1, bottom - 1.4,
            "D = outer(E, s_flat)      14 × 84, same-channel entries zero",
            ha="right", va="top", fontsize=FS_SMALL, color=STRUCTURE,
            fontweight="semibold")

    # Only the channel firing now has a non-zero row, which is the point of
    # showing D at full size: it is almost entirely empty, and the one live
    # row is the whole of what the readout sees.
    row_y = top - (channel + 0.5) * bar_h
    for index in range(n_rates):
        ax.add_patch(Rectangle(
            (x_d0 + index * block, row_y - bar_h / 2), block, bar_h,
            fill=False, edgecolor=MECH["plasticity"], linewidth=0.5,
            zorder=6))

    # ---- stage 4: the zoom, and the learned template ---------------------
    # D has exactly one non-zero row, so the informative object is that row
    # reshaped into context token by filter timescale.  The learned mask is
    # shown beside it in the same layout, which is the whole claim: the mask
    # is a template for that pattern.
    zoom_top, zoom_bottom = 28.0, 7.5
    cell_w = 2.0
    zoom_w = cell_w * n_rates
    cell_h = (zoom_top - zoom_bottom) / N_CHANNELS
    x_zoom_d, x_zoom_m = 56.0, 88.0

    for x0, values, title, colour in (
        (x_zoom_d, coincidence[channel].reshape(n_rates, N_CHANNELS).T,
         f"D, row {channel}", STRUCTURE),
        (x_zoom_m, mask[channel].reshape(n_rates, N_CHANNELS).T,
         f"learned mask, unit {unit}", MECH["readout"]),
    ):
        _matrix(ax, values, (x0, x0 + zoom_w, zoom_bottom, zoom_top))
        ax.text(x0 + zoom_w / 2, zoom_top + 1.1, title, ha="center",
                va="bottom", fontsize=FS_SMALL, fontweight="semibold",
                color=colour)
        for token in word[:-1]:
            best = int(np.argmax(values[token]))
            y = zoom_top - (token + 0.5) * cell_h
            for edge, lw, z in ((COLORS["white"], 1.3, 6),
                                (MECH["readout"], 0.6, 7)):
                ax.add_patch(Rectangle(
                    (x0 + best * cell_w, y - cell_h / 2), cell_w, cell_h,
                    fill=False, edgecolor=edge, linewidth=lw, zorder=z))
        for index in range(n_rates):
            ax.text(x0 + (index + 0.5) * cell_w, zoom_bottom - 0.7,
                    f"{taus[index] * 1e3:.0f}", ha="right", va="center",
                    fontsize=FS_SMALL, color=FAINT, rotation=90)
        for token in word:
            y = zoom_top - (token + 0.5) * cell_h
            ax.text(x0 - 0.8, y, str(token), ha="right", va="center",
                    fontsize=FS_SMALL,
                    color=MECH["readout"] if token in word[:-1] else
                    MECH["plasticity"], fontweight="semibold")
    ax.text(x_zoom_d - 4.2, (zoom_top + zoom_bottom) / 2, "context token",
            rotation=90, ha="center", va="center", fontsize=FS_SMALL,
            color=FAINT)
    ax.text((x_zoom_d + x_zoom_m + zoom_w) / 2, zoom_bottom - 5.6,
            "filter τ (ms)", ha="center", va="top", fontsize=FS_SMALL,
            color=FAINT)

    _arrow(ax, (x_zoom_d + zoom_w + 3.0, (zoom_top + zoom_bottom) / 2),
           (x_zoom_m - 3.0, (zoom_top + zoom_bottom) / 2),
           color=FAINT, lw=0.6, head=1.9)
    ax.text((x_zoom_d + zoom_w + x_zoom_m) / 2,
            (zoom_top + zoom_bottom) / 2 + 1.1, "matches", ha="center",
            va="bottom", fontsize=FS_SMALL, color=FAINT)
    _line(ax, (x_d0 + 2.0, row_y - bar_h / 2),
          (x_zoom_d + zoom_w / 2, zoom_top + 3.4),
          color=FAINT, lw=0.45, rad=0.16, style=(0, (1.6, 1.4)), zorder=2)

    # ---- stage 5: the ordered template, stated -------------------------
    x_note = 112.0
    ax.text(x_note, zoom_top + 1.1,
            f"{WORD_NAMES[word_index]} = "
            f"{'→'.join(str(token) for token in word)}, held in order",
            ha="left", va="bottom", fontsize=FS_SMALL, fontweight="semibold",
            color=MECH["readout"])
    mask_rows = mask[channel].reshape(n_rates, N_CHANNELS).T
    entries = [(token, f"τ {taus[int(np.argmax(mask_rows[token]))] * 1e3:.0f} ms",
                MECH["readout"]) for token in word[:-1]]
    entries.append((word[-1], "firing now", MECH["plasticity"]))
    for order_index, (token, value, colour) in enumerate(entries):
        y = zoom_top - 2.6 - order_index * 3.4
        ax.text(x_note, y, f"token {token}", ha="left", va="top",
                fontsize=FS_SMALL, color=STRUCTURE)
        ax.text(x_note + 12.0, y, value, ha="left", va="top",
                fontsize=FS_SMALL, color=colour, fontweight="semibold")
    ax.text(x_note, zoom_top - 2.6 - 4 * 3.4 - 1.8,
            "Older tokens sit on strictly slower filters.\n"
            "That ordering is what makes the mask a word\n"
            "rather than a set of tokens.",
            ha="left", va="top", fontsize=FS_SMALL, color=FAINT,
            linespacing=1.45)

    equations = (
        "τr dsr/dt = −sr + E,   r = 1…6",
        "D = outer(E, s_flat),   Dii = 0",
        "y = relu(M · D)",
        "ΔMk = η (D/‖D‖ − Mk), best match only",
    )
    ax.text(4.0, zoom_top + 1.1, "Readout equations", ha="left", va="bottom",
            fontsize=FS_SMALL, fontweight="semibold", color=STRUCTURE)
    for index, line in enumerate(equations):
        ax.text(4.0, zoom_top - 2.6 - index * 3.4, line, ha="left", va="top",
                fontsize=FS_SMALL, color=STRUCTURE)
    ax.text(4.0, zoom_top - 2.6 - 4 * 3.4 - 1.8,
            "24 units, one fixed mask each.\nHow many commit is a result.",
            ha="left", va="top", fontsize=FS_SMALL, color=FAINT,
            linespacing=1.45)

    _heading(
        ax, height,
        "Layer 2 — a multiscale readout that composes relations",
        "measured arrays, not drawings · one instant of a real Saffran stream, "
        "the final token of a four-token word · colour scaled within each map",
        title_mm=6.6, subtitle_mm=2.6, letter="b", letter_mm=10.8,
        width_mm=width)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _band(top_mm: float, height_mm: float) -> tuple[float, float]:
    return (1.0 - top_mm / FIGURE_HEIGHT_MM,
            1.0 - (top_mm + height_mm) / FIGURE_HEIGHT_MM)


def build_figure(*, force_data: bool = False) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_model_data(force=force_data, data_dir=DATA_DIR)

    left, right = 0.062, 0.985
    with manuscript_style():
        fig = plt.figure(figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)))

        top, bottom = _band(14.0, 46.0)
        first = fig.add_gridspec(1, 1, left=left, right=right, top=top,
                                 bottom=bottom)
        _panel_layer1(fig, first[0, 0], data)

        top, bottom = _band(61.0, 9.0)
        middle = fig.add_gridspec(1, 1, left=left, right=right, top=top,
                                  bottom=bottom)
        _panel_interface(fig, middle[0, 0], data)

        top, bottom = _band(82.0, 52.0)
        last = fig.add_gridspec(1, 1, left=left, right=right, top=top,
                                bottom=bottom)
        _panel_layer2(fig, last[0, 0], data)

        paths = export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
        plt.close(fig)

    provenance = {
        "figure": "Two-layer model definition",
        "final_size_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "claim": (
            "Layer 1 carries three memories; layer 2 composes relations among "
            "what layer 1 emits. The interface is the rate vector E alone."
        ),
        "colour_key": dict(MECH),
        "panel_sources": {
            "a": "schematic of the model0 circuit; time constants annotated",
            "interface": "structural statement: Layer2MR.step reads E only",
            "b": (
                "MEASURED - filterbank state, coincidence map and a learned "
                "mask, computed by Layer2MR on the cached Saffran excerpt"
            ),
        },
        "inputs": {},
        "outputs": {},
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }
    for name in ("model_figure_data.npz", "model_provenance.json"):
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
                        help="Recompute the layer-1 and layer-2 arrays.")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    paths = build_figure(force_data=_parse_args(arguments).force_data)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
