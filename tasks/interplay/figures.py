"""
tasks.interplay.figures
=======================

Figures for the interplay task, in the manuscript's visual system.

Kept apart from ``interplay.py`` so the measurement code has no matplotlib
in it and the figures have no simulation in them.

Conventions, and the reasons for them
-------------------------------------
Titles name the quantity and stop.  A panel called "Segregation index" can
be read by someone who disagrees with the interpretation; a panel called
"Predictability segregates the stream" cannot, and a figure whose titles
argue with the reader is a figure that has stopped being evidence.  The
interpretation belongs in the caption and the text.

Chance and floor levels are thin rules in the plot, not parentheses in the
title.  They are properties of the measure, so they belong on the axis
where the reader can see how far the data sit from them.

Every seed is drawn.  With six runs there is no reason to show a mean and
hide the spread behind an error bar: the dots are the data, and the bar is
just their mean.  Anything that survives only in the mean should be
visible as a failure to survive in the dots.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from final_figures.style import (COLORS, clean_axis, export_figure, mm,
                                 manuscript_style)

OUT_DIR = Path(__file__).resolve().parent

#: Structured keeps the model green used for model results throughout the
#: paper; scrambled is the neutral ash already used for controls.
C_STRUCT = COLORS["model"]
C_SCRAM = COLORS["ash"]
C_BG = COLORS["terracotta"]
C_INK = COLORS["charcoal"]


# ---------------------------------------------------------------------
#  Primitives
# ---------------------------------------------------------------------
def _dots(ax, series: Sequence[np.ndarray], colors: Sequence[str],
          labels: Sequence[str], *, width: float = 0.42,
          jitter: float = 0.055, ylabel: str = "",
          groups: Sequence[tuple] = ()) -> None:
    """One column per condition: a mean rule, and every run as a dot.

    ``groups`` is ((label, first_col, last_col), ...) and draws a spanning
    caption beneath.  Without it a four-column panel reads as
    "struct. scram. struct. scram." and the reader cannot tell which pair
    carried a background.
    """
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(series, colors)):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        m = vals.mean()
        ax.hlines(m, i - width, i + width, color=col, lw=1.6, zorder=3)
        x = i + rng.uniform(-jitter, jitter, vals.size)
        ax.scatter(x, vals, s=11, facecolors="none", edgecolors=col,
                   linewidths=0.7, zorder=4, clip_on=False)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, len(labels) - 0.35)
    if ylabel:
        ax.set_ylabel(ylabel)
    clean_axis(ax)
    for text, lo, hi in groups:
        mid = (lo + hi) / 2.0
        ax.annotate("", xy=(lo - 0.34, -0.155), xytext=(hi + 0.34, -0.155),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color=COLORS["ash"]),
                    annotation_clip=False)
        ax.annotate(text, xy=(mid, -0.185), xycoords=("data",
                                                      "axes fraction"),
                    ha="center", va="top", fontsize=6.0,
                    color=COLORS["ash"], annotation_clip=False)


def _reference(ax, y: float, text: str) -> None:
    """A thin rule for a chance or floor level, labelled inside the axes.

    Inside, because a label hung off the right edge either collides with
    the next panel or silently disappears when the figure is resized.
    """
    ax.axhline(y, color=C_INK, lw=0.55, ls=(0, (3, 2)), zorder=1)
    ax.annotate(text, xy=(0.015, y), xycoords=("axes fraction", "data"),
                xytext=(0, 2.2), textcoords="offset points",
                fontsize=5.8, color=COLORS["ash"], va="bottom", ha="left")


def _letter(fig, ax, s: str) -> None:
    """Panel letter placed in FIGURE coordinates.

    panel_label() works in axes coordinates, so on a narrow panel its
    -0.15 offset is a small absolute distance and the letter lands on top
    of the centred title.  Anchoring to the axes bounding box instead
    keeps the offset constant whatever the panel's width.
    """
    bb = ax.get_position()
    fig.text(bb.x0 - 0.026, bb.y1 + 0.085, s, fontsize=10.0,
             fontweight="bold", color=C_INK, va="top", ha="left")


# ---------------------------------------------------------------------
#  Figure 1 -- the result
# ---------------------------------------------------------------------
def results_figure(table: Dict[str, List[Dict[str, float]]],
                   levels: Sequence[float],
                   curves: Dict[str, Dict[str, List[np.ndarray]]],
                   stem: str = "interplay_results") -> Dict[str, Path]:
    def col(label: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in table[label]], dtype=float)

    bg = ["structured/bg", "scrambled/bg"]
    both = ["structured/bg", "scrambled/bg",
            "structured/clean", "scrambled/clean"]
    cs2 = [C_STRUCT, C_SCRAM]
    cs4 = [C_STRUCT, C_SCRAM, C_STRUCT, C_SCRAM]
    lb2 = ["structured", "scrambled"]
    lb4 = ["struct.", "scram.", "struct.", "scram."]

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(104)))
        gs = fig.add_gridspec(2, 3, left=0.078, right=0.972, top=0.885,
                              bottom=0.10, wspace=0.50, hspace=0.68)

        # A -- segregation
        ax = fig.add_subplot(gs[0, 0])
        _dots(ax, [col(l, "seg_index") for l in bg], cs2, lb2,
              ylabel="Segregation index")
        _reference(ax, 0.0, "0")
        _letter(fig, ax, "a")
        ax.set_title("Segregation")

        # B -- learned transition weights
        ax = fig.add_subplot(gs[0, 1])
        _dots(ax, [col("structured/bg", "W_within"),
                   col("structured/bg", "W_boundary"),
                   col("structured/bg", "W_GG")],
              [C_STRUCT, COLORS["teal"], C_BG],
              ["within", "boundary", "backgr."],
              ylabel="Mean weight")
        _letter(fig, ax, "b")
        ax.set_title("Learned transitions")

        # C -- transition accuracy
        ax = fig.add_subplot(gs[0, 2])
        _dots(ax, [col(l, "trans_acc") for l in both], cs4, lb4,
              groups=(("with background", 0, 1), ("no background", 2, 3)),
              ylabel="Transition accuracy")
        _reference(ax, 0.40, "chance")
        ax.set_ylim(0, 1.05)
        _letter(fig, ax, "c")
        ax.set_title("Transition accuracy")

        # D -- word vs part-word
        ax = fig.add_subplot(gs[1, 0])
        _dots(ax, [col(l, "wp_auc") for l in both], cs4, lb4,
              groups=(("with background", 0, 1), ("no background", 2, 3)),
              ylabel="Word vs part-word (AUC)")
        floor = float(np.nanmean(col("structured/bg", "wp_auc_floor")))
        _reference(ax, floor, "no learning")
        ax.set_ylim(0, 1.05)
        _letter(fig, ax, "d")
        ax.set_title("Word discrimination")

        # E -- weight selectivity ratio
        ax = fig.add_subplot(gs[1, 1])
        _dots(ax, [col(l, "within_over_boundary") for l in bg], cs2, lb2,
              ylabel="Within / boundary weight")
        _reference(ax, 1.0, "no preference")
        _letter(fig, ax, "e")
        ax.set_title("Transition selectivity")

        # F -- robustness
        ax = fig.add_subplot(gs[1, 2])
        lv = np.asarray(levels, dtype=float)
        for name, col_ in (("structured", C_STRUCT), ("scrambled", C_SCRAM)):
            arr = np.array(curves[name]["seg_index"], dtype=float)  # (lv, seed)
            m = np.nanmean(arr, axis=1)
            ax.plot(lv, m, color=col_, lw=1.3, zorder=3, label=name)
            for j in range(arr.shape[1]):
                ax.plot(lv, arr[:, j], color=col_, lw=0.4, alpha=0.35,
                        zorder=2)
        _reference(ax, 0.0, "0")
        ax.set_xscale("log")
        ax.set_xticks(lv)
        ax.set_xticklabels([f"{v:g}" for v in lv])
        ax.set_xlabel("Background level (× drive-matched)")
        ax.set_ylabel("Segregation index")
        ax.legend(loc="upper right")
        clean_axis(ax)
        _letter(fig, ax, "f")
        ax.set_title("Robustness")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 2 -- mechanism
# ---------------------------------------------------------------------
def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else \
           "*" if p < 0.05 else "n.s."


def _bracket(ax, x0: float, x1: float, y: float, h: float, text: str,
             color: str = C_INK) -> None:
    """A square bracket with end ticks, so it reads as spanning x0..x1."""
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=0.5,
            clip_on=False, zorder=6, solid_joinstyle="miter")
    ax.annotate(text, xy=((x0 + x1) / 2, y + h), xytext=(0, 0.8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.3, color=color,
                fontweight="bold" if text != "n.s." else "normal",
                clip_on=False, zorder=6)


#: Position inside the word is ordered, so it gets an ordered ramp rather
#: than three unrelated hues.
POS_RAMP = ("#B9D6CB", "#6AA491", COLORS["model"])


def mechanism_figure(exemplar_p: Dict, groups: Dict[str, np.ndarray],
                     pos: np.ndarray, pos_bg: np.ndarray,
                     dec: Dict[str, np.ndarray],
                     rates: Dict[str, np.ndarray],
                     poscur: Dict[str, np.ndarray] | None = None,
                     tests: Dict[str, tuple] | None = None,
                     stem: str = "interplay_mechanism") -> Dict[str, Path]:
    """``pos`` is (seed, 3); ``poscur[k]`` is (seed, 3) per current."""
    cfg = exemplar_p["cfg"]
    nf = cfg.n_figure
    W = exemplar_p["W_final"]

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(66)))
        gs = fig.add_gridspec(1, 4, left=0.052, right=0.980, top=0.785,
                              bottom=0.225, wspace=0.50,
                              width_ratios=(0.92, 0.95, 0.95, 1.25))

        # A -- weight matrix
        ax = fig.add_subplot(gs[0, 0])
        # box_aspect makes the AXES square, so an aspect="auto" image
        # fills it exactly.  Leaving the image on aspect="equal" instead
        # would shrink the drawn area inside a wider box, and the title
        # and panel letter would then sit off the image.
        im = ax.imshow(W, cmap="Purples", interpolation="nearest",
                       vmin=0.0, aspect="auto")
        ax.set_box_aspect(1.0)
        ax.axhline(nf - 0.5, color=C_INK, lw=0.6)
        ax.axvline(nf - 0.5, color=C_INK, lw=0.6)
        ax.set_xlabel("Presynaptic")
        ax.set_ylabel("Postsynaptic")
        ax.set_xticks([0, nf, cfg.n_channels - 1])
        ax.set_yticks([0, nf, cfg.n_channels - 1])
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.ax.tick_params(labelsize=5.8, length=1.4, width=0.45)
        cb.outline.set_linewidth(0.45)
        _letter(fig, ax, "a")
        ax.set_title("Learned weights")

        # B -- growth
        ax = fig.add_subplot(gs[0, 1])
        for key, lab, c in (("within", "within-word", C_STRUCT),
                            ("boundary", "boundary", COLORS["teal"]),
                            ("ground", "background", C_BG),
                            ("cross", "cross", COLORS["ash"])):
            ax.plot(groups["t"], groups[key], color=c, lw=1.2, label=lab)
        ax.set_xlabel("Session time (s)")
        ax.set_ylabel("Mean weight")
        ax.legend(loc="lower right")
        clean_axis(ax)
        _letter(fig, ax, "b")
        ax.set_title("Weight growth")

        # C -- modulation by position in the word
        ax = fig.add_subplot(gs[0, 2])
        _dots(ax, [pos[:, i] for i in range(3)],
              [C_STRUCT] * 3, ["1", "2", "3"],
              ylabel="Response modulation (%)")
        _reference(ax, float(np.nanmean(pos_bg)), "background")
        ax.set_xlabel("Token position in word")
        _letter(fig, ax, "c")
        ax.set_title("Enhancement")

        # D -- how the enhancement is built, resolved by position
        ax = fig.add_subplot(gs[0, 3])
        series = (("rec_E", "Recurrent"), ("inh_to_E", "Inhibition"),
                  ("net", "Net"))
        xs = np.arange(len(series))
        w = 0.26
        rng = np.random.default_rng(1)
        hi = 0.0
        for gi, (key, _lab) in enumerate(series):
            arr = np.asarray(poscur[key], dtype=float)          # (seed, 3)
            for j in range(3):
                xc = xs[gi] + (j - 1) * w
                ax.bar(xc, arr[:, j].mean(), w * 0.92,
                       color=POS_RAMP[j], edgecolor="none", zorder=2)
                ax.scatter(xc + rng.uniform(-0.055, 0.055, arr.shape[0]),
                           arr[:, j], s=5.5, facecolors="none",
                           edgecolors=C_INK, linewidths=0.4, alpha=0.8,
                           zorder=4, clip_on=False)
            hi = max(hi, float(arr.max()))

        span = hi if hi > 0 else 1.0
        for gi, (key, _lab) in enumerate(series):
            arr = np.asarray(poscur[key], dtype=float)
            top = float(arr.max())
            d, pval = tests[key] if tests else (np.nan, np.nan)
            _bracket(ax, xs[gi] - 1.35 * w, xs[gi] + 1.35 * w,
                     top + 0.05 * span, 0.030 * span,
                     f"{_stars(pval)} \u0394{d:+.3f}")

        ax.axhline(0.0, color=COLORS["ash"], lw=0.5)
        ax.set_ylim(0, hi + 0.38 * span)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _k, lab in series])
        ax.set_ylabel("Change from frozen (a.u.)")

        # Thalamic drive is identically zero -- the frozen run sees the same
        # stimulus -- so it is stated rather than given a group of flat bars.
        tm = float(np.nanmax(np.abs(poscur["tm_in"])))
        ax.annotate(f"thalamic \u2261 0  (max |\u0394| = {tm:.0e})",
                    xy=(0.5, -0.20), xycoords="axes fraction",
                    fontsize=5.8, color=COLORS["ash"], va="top",
                    ha="center", annotation_clip=False)

        handles = [plt.Rectangle((0, 0), 1, 1, color=POS_RAMP[j])
                   for j in range(3)]
        ax.legend(handles, ["1", "2", "3"], title="Token position",
                  loc="upper center", bbox_to_anchor=(0.53, 1.02),
                  ncol=3, handlelength=0.8, handleheight=0.8,
                  columnspacing=0.6, fontsize=6.0, title_fontsize=6.0,
                  borderpad=0.15, labelspacing=0.25)
        clean_axis(ax)
        _letter(fig, ax, "d")
        ax.set_title("Mechanism")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 3 -- stimulus
# ---------------------------------------------------------------------
def stimulus_figure(pack: Dict, cfg, n_words_show: int = 7,
                    stem: str = "interplay_stimulus") -> Dict[str, Path]:
    show = n_words_show * 3 * cfg.slot
    stim = pack["stim"][:, :show]
    on = pack["stim"][cfg.n_figure:, :] > 0

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(66)))
        gs = fig.add_gridspec(2, 2, left=0.070, right=0.975, top=0.855,
                              bottom=0.12, wspace=0.32, hspace=0.72,
                              height_ratios=(1.7, 1.0),
                              width_ratios=(2.3, 1.0))

        ax = fig.add_subplot(gs[0, :])
        ax.imshow(stim, aspect="auto", origin="lower", cmap="Greys",
                  interpolation="nearest",
                  extent=[0, show, -0.5, cfg.n_channels - 0.5], vmin=0)
        ax.axhline(cfg.n_figure - 0.5, color=COLORS["model"], lw=0.8)
        for k in range(n_words_show + 1):
            ax.axvline(k * 3 * cfg.slot, color=COLORS["model"], lw=0.5,
                       ls=(0, (2, 2)), alpha=0.75)
        ax.set_ylabel("Channel")
        ax.set_xlabel("Time (ms)")
        ax.set_yticks([0, cfg.n_figure, cfg.n_channels - 1])
        ax.text(show * 0.008, cfg.n_figure - 1.0, "figure",
                fontsize=6.4, color=COLORS["model"], va="top")
        ax.text(show * 0.008, cfg.n_channels - 0.7, "background",
                fontsize=6.4, color=C_BG, va="top")
        clean_axis(ax)
        _letter(fig, ax, "a")
        ax.set_title("Stimulus, dashed lines at word boundaries")

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(on.sum(axis=0)[:show], color=C_BG, lw=0.9)
        ax.set_ylim(0, 2.2)
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Background\ntones on")
        clean_axis(ax)
        _letter(fig, ax, "b")
        ax.set_title("Background occupancy")

        ax = fig.add_subplot(gs[1, 1])
        tot = on.sum(axis=1) / on.shape[1] * 100
        ax.bar(range(len(tot)), tot, color=C_BG, width=0.72,
               edgecolor="none")
        _reference(ax, 100 / cfg.n_background, "uniform")
        ax.set_xlabel("Background channel")
        ax.set_ylabel("Time on (%)")
        clean_axis(ax)
        _letter(fig, ax, "c")
        ax.set_title("Channel balance")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths
