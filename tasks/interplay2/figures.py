"""
tasks.interplay2.figures
========================

Figures for the interplay2 task, in the manuscript's visual system.

Kept apart from ``interplay2.py`` so the measurement code has no matplotlib in
it and the figures have no simulation in them.

Three figures, in the order a reader needs them:

    interplay2_stimulus     what is in the stream, and the constraints on it
    interplay2_tape         every unit's activity along the input
    interplay2_allocation   how the population divided itself, across seeds
    interplay2_layer1       the recurrent map layer 1 learns from the same stream

Titles name the quantity and stop; chance and null levels are drawn as rules
inside the axes rather than asserted in a title; every seed is a dot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from final_figures.style import (COLORS, clean_axis, export_figure, mm,
                                 manuscript_style)

OUT_DIR = Path(__file__).resolve().parent

#: One colour per token, one for the cloud.  The tokens keep the two colours
#: the manuscript already uses for its two contrasting conditions.
C_AB = COLORS["rep1"]
C_CD = COLORS["rep15"]
C_CLOUD = COLORS["ash"]
C_INK = COLORS["charcoal"]
C_MODEL = COLORS["model"]
TOKEN_COLORS = (C_AB, C_CD)


# ---------------------------------------------------------------------
#  Primitives
# ---------------------------------------------------------------------
def _letter(fig, ax, s: str) -> None:
    """Panel letter at a fixed offset in millimetres from the axes box.

    Axes-fraction offsets are a constant *fraction*, so on a narrow panel the
    letter lands on the title and on a wide one it drifts away.  Millimetres
    do not do that.
    """
    bb = ax.get_position()
    dx = mm(2.6) / fig.get_figwidth()
    dy = mm(2.4) / fig.get_figheight()
    fig.text(bb.x0 - dx, bb.y1 + dy, s, fontsize=10.0, fontweight="bold",
             color=C_INK, va="top", ha="left")


def _dots(ax, series: Sequence[np.ndarray], colors: Sequence[str],
          labels: Sequence[str], *, width: float = 0.40,
          jitter: float = 0.06, ylabel: str = "",
          groups: Sequence[tuple] = (), empty_note: str = "") -> None:
    """A mean rule per column, every seed as a dot.

    ``groups`` is ((label, first_col, last_col), ...) and draws a spanning
    caption beneath, so a six-column panel does not read as an undifferentiated
    row of labels.  ``empty_note`` is written into any column that has no
    finite values, because a silently blank column is indistinguishable from a
    bug and here it means something specific: no unit took a token at all.
    """
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(series, colors)):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            if empty_note:
                ax.annotate(empty_note, xy=(i, 0.5), xycoords=("data",
                                                               "axes fraction"),
                            ha="center", va="center", fontsize=5.8,
                            color=COLORS["ash"], rotation=90)
            continue
        ax.hlines(v.mean(), i - width, i + width, color=col, lw=1.6, zorder=3)
        ax.scatter(i + rng.uniform(-jitter, jitter, v.size), v, s=11,
                   facecolors="none", edgecolors=col, linewidths=0.7,
                   zorder=4, clip_on=False)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, len(labels) - 0.35)
    if ylabel:
        ax.set_ylabel(ylabel)
    clean_axis(ax)
    for text, lo, hi in groups:
        ax.annotate("", xy=(lo - 0.32, -0.20), xytext=(hi + 0.32, -0.20),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color=COLORS["ash"]),
                    annotation_clip=False)
        ax.annotate(text, xy=((lo + hi) / 2.0, -0.235),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=6.0, color=COLORS["ash"], annotation_clip=False)


def _reference(ax, y: float, text: str) -> None:
    """A thin rule for a chance or null level, labelled inside the axes."""
    ax.axhline(y, color=C_INK, lw=0.55, ls=(0, (3, 2)), zorder=1)
    ax.annotate(text, xy=(0.015, y), xycoords=("axes fraction", "data"),
                xytext=(0, 2.2), textcoords="offset points", fontsize=5.8,
                color=COLORS["ash"], va="bottom", ha="left")


def _raster(ax, stim: np.ndarray, cfg, dt: float, *, lw: float = 2.2,
            gap: float = 2.6, step: float = 2.2, cloud_scale: float = 0.45) -> None:
    """Channel-by-time raster: token channels in colour, cloud in grey.

    The four token channels are lifted clear of the cloud and spaced further
    apart than the cloud rows are.  Twenty equally spaced rows in a 25 mm
    panel puts A, B, C and D closer together than their own labels, and the
    reader then cannot tell which row is which -- which is the one thing this
    panel exists to show.
    """
    n = stim.shape[0]
    n_tok = cfg.n_token_channels
    n_bg = n - n_tok

    top_of_cloud = (n_bg - 1) * cloud_scale

    def y_of(ch: int) -> float:
        if ch < n_tok:
            return top_of_cloud + gap + (n_tok - 1 - ch) * step
        return (n - 1 - ch) * cloud_scale

    for ch in range(n):
        on = stim[ch] > 0
        if not on.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.view(np.int8),
                                                       [0]])))
        col = (C_AB if ch < 2 else C_CD) if ch < n_tok else C_CLOUD
        for a, b in zip(edges[0::2], edges[1::2]):
            ax.plot([a * dt, b * dt], [y_of(ch)] * 2, color=col, lw=lw,
                    solid_capstyle="butt", zorder=3)

    ax.set_ylim(-1.0, y_of(0) + 1.0)
    ax.set_yticks([y_of(i) for i in range(n_tok)] + [top_of_cloud / 2.0])
    ax.set_yticklabels(["A", "B", "C", "D", "cloud"])
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, left=False)


# ---------------------------------------------------------------------
#  Figure 1 -- the stimulus
# ---------------------------------------------------------------------
def stimulus_figure(ex: Dict, stem: str = "interplay2_stimulus"):
    cfg, dt = ex["cfg"], ex["dt"]
    stim = ex["stim"]
    show = min(stim.shape[1], 3 * cfg.block_samples)
    ts = np.arange(show) * dt

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(78)))
        gs = fig.add_gridspec(2, 3, left=0.075, right=0.975, top=0.88,
                              bottom=0.11, wspace=0.42, hspace=0.75,
                              height_ratios=[1.25, 1.0])

        # a -- the stream
        ax = fig.add_subplot(gs[0, :])
        _raster(ax, stim[:, :show], cfg, dt)
        for name, col in (("B", C_AB), ("D", C_CD)):
            for o in ex["onsets"][name]:
                if o < show:
                    ax.axvline(o * dt, color=col, lw=0.5, alpha=0.35,
                               zorder=1)
        ax.set_xlim(0, ts[-1])
        ax.set_xlabel("Time (s)")
        ax.set_title("Three blocks of the stream")
        _letter(fig, ax, "a")

        # b -- per-channel on-time, over the WHOLE stream
        ax = fig.add_subplot(gs[1, 0])
        on = ex["checks"]["on_time"]
        cols = [C_AB, C_AB, C_CD, C_CD] + [C_CLOUD] * (len(on) - 4)
        ax.bar(np.arange(len(on)), on, color=cols, width=0.78, lw=0)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Total on-time (s)")
        ax.set_ylim(0, on.max() * 1.25)
        ax.set_title("Drive per channel")
        clean_axis(ax)
        _letter(fig, ax, "b")

        # c -- simultaneity
        ax = fig.add_subplot(gs[1, 1])
        tot = ex["checks"]["n_simul"]
        bgc = ex["checks"]["n_simul_bg"]
        vals = np.arange(0, 5)
        ax.bar(vals - 0.19, [np.mean(tot == v) for v in vals], width=0.36,
               color=C_MODEL, lw=0, label="all channels")
        ax.bar(vals + 0.19, [np.mean(bgc == v) for v in vals], width=0.36,
               color=C_CLOUD, lw=0, label="cloud only")
        ax.set_xlabel("Tones on at once")
        ax.set_ylabel("Fraction of time")
        ax.set_xticks(vals)
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2)
        ax.set_title("Simultaneity")
        clean_axis(ax)
        _letter(fig, ax, "c")

        # d -- the lag that defines a token
        ax = fig.add_subplot(gs[1, 2])
        # Dodged bars, not overlaid histograms: in `paired` the two lags are
        # both exactly one slot and one distribution would sit invisibly
        # behind the other.
        # Bins centred ON the slot multiples, so a bar sits at the lag it
        # counts rather than half a slot to the right of it.
        centres = np.arange(-cfg.block_slots, cfg.block_slots + 1) * cfg.slot
        edges = np.append(centres - cfg.slot / 2, centres[-1] + cfg.slot / 2)
        for k, (name, col) in enumerate((("B after A", C_AB),
                                         ("D after C", C_CD))):
            h, _ = np.histogram(ex["checks"]["lags"][name], bins=edges)
            ax.bar(centres + (k - 0.5) * cfg.slot * 0.42, h,
                   width=cfg.slot * 0.40, color=col, lw=0, label=name)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Tokens")
        ax.legend(loc="upper left", handlelength=1.0, borderpad=0.2)
        ax.set_title("Within-token lag")
        clean_axis(ax)
        _letter(fig, ax, "d")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 1b -- the cloud, as a listener would be shown it
# ---------------------------------------------------------------------
def cloud_figure(ex: Dict, stem: str = "interplay2_cloud",
                 n_blocks: int = 6, seed: int = 7):
    """The stimulus drawn the way a tone cloud is normally drawn.

    One dash per tone, channels on the vertical axis, time on the
    horizontal, nothing else on the page.  Two things differ from the raster
    in ``stimulus_figure``, and both are deliberate:

    * **The channel order is shuffled.**  A, B, C and D are the first four
      indices in the code, so an ordered plot puts all four tokens in a band
      at the top and the eye groups them for reasons the model never sees.
      Shuffling scatters them through the cloud, which is how they are
      scattered in frequency for a listener.  The permutation is seeded, so
      the picture is reproducible.
    * **No axes.**  This panel is the stimulus, not a measurement; a reader
      should be able to look for the repeating pair and fail to find it,
      which is the point.  A scale bar carries the one quantity that matters.
    """
    cfg, dt = ex["cfg"], ex["dt"]
    stim = ex["stim"]
    show = min(stim.shape[1], n_blocks * cfg.block_samples)
    n = stim.shape[0]

    # A seeded permutation of the rows, so the tokens sit at unremarkable
    # heights instead of in a block at the top.  Among candidate
    # permutations, take the one that pushes the four token channels
    # furthest apart: adjacent rows would let the reader group them by
    # proximity, which is a cue the model does not have and the listener
    # does not have either.
    rng = np.random.default_rng(seed)
    n_tok = cfg.n_token_channels
    best, order = -1.0, None
    for _ in range(400):
        cand = rng.permutation(n)
        rows = np.sort([int(np.flatnonzero(cand == c)[0])
                        for c in range(n_tok)])
        spread = float(np.diff(rows).min())
        if spread > best:
            best, order = spread, cand
    row_of = {int(ch): int(r) for r, ch in enumerate(order)}

    with manuscript_style():
        fig = plt.figure(figsize=(mm(88), mm(100)))
        ax = fig.add_axes([0.02, 0.075, 0.96, 0.895])

        for ch in range(n):
            on = stim[ch, :show] > 0
            if not on.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(
                [[0], on.view(np.int8), [0]])))
            col = (C_AB if ch < 2 else C_CD) if ch < 4 else C_INK
            y = row_of[ch]
            for a, b in zip(edges[0::2], edges[1::2]):
                ax.plot([a * dt, b * dt], [y, y], color=col, lw=3.0,
                        solid_capstyle="butt")

        ax.set_xlim(-0.02 * show * dt, show * dt * 1.02)
        ax.set_ylim(-1.2, n + 0.4)
        ax.set_axis_off()

        # Scale bar, in place of an axis.
        bar = 0.5
        ax.plot([0.0, bar], [-0.9, -0.9], color=C_INK, lw=1.2,
                solid_capstyle="butt")
        ax.annotate(f"{bar:.1f} s", xy=(bar / 2, -1.15), ha="center",
                    va="top", fontsize=6.5, color=C_INK)

        for label, col, x in (("A \u2192 B", C_AB, 0.0),
                              ("C \u2192 D", C_CD, 0.22),
                              ("cloud", C_INK, 0.44)):
            ax.annotate(label, xy=(x, 1.005), xycoords="axes fraction",
                        ha="left", va="bottom", fontsize=7.0, color=col,
                        fontweight="bold")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 2 -- the tape: unit activity along the input
# ---------------------------------------------------------------------
def tape_figure(ex: Dict, stem: str = "interplay2_tape", n_blocks: int = 3):
    """Every unit's output on the same clock as the stimulus that drove it.

    Nothing here is averaged or epoched.  The window comes from the end of the
    test stream, so what is drawn is settled behaviour rather than the
    transient while the masks are still forming.
    """
    cfg, dt = ex["cfg"], ex["dt"]
    rows = ex["rows"]
    n_units = len(rows)
    show = min(ex["y"].shape[1], n_blocks * cfg.block_samples)
    ts = np.arange(show) * dt

    # Which unit, if any, owns each token -- used only for colouring.
    owner = {}
    for r in rows:
        if r["top"] == (1, 0):
            owner[r["unit"]] = ("AB", C_AB)
        elif r["top"] == (3, 2):
            owner[r["unit"]] = ("CD", C_CD)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(30 + 15 * n_units)))
        gs = fig.add_gridspec(2 + n_units, 1, left=0.105, right=0.985,
                              top=0.935, bottom=0.075, hspace=0.42,
                              height_ratios=[1.9, 1.0] + [1.0] * n_units)

        def mark(ax) -> None:
            """Token events, drawn behind everything, on every row."""
            for second, col in (("B", C_AB), ("D", C_CD)):
                for o in ex["onsets"][second]:
                    if o < show:
                        ax.axvspan(o * dt, (o + cfg.tone_dur) * dt,
                                   color=col, alpha=0.13, lw=0, zorder=0)

        # -- the stimulus --
        ax = fig.add_subplot(gs[0])
        _raster(ax, ex["stim"][:, :show], cfg, dt, lw=1.9)
        mark(ax)
        ax.set_xlim(0, ts[-1])
        ax.tick_params(labelbottom=False)
        ax.set_title("Stimulus, layer 1 and the four layer-2 units on one clock")
        _letter(fig, ax, "a")

        # -- layer 1, the four token channels --
        ax = fig.add_subplot(gs[1])
        for ch, col, nm in ((0, C_AB, "A"), (1, C_AB, "B"),
                            (2, C_CD, "C"), (3, C_CD, "D")):
            ax.plot(ts, ex["E"][ch, :show], color=col, lw=0.8,
                    ls="-" if ch % 2 == 0 else (0, (2.2, 1.2)), label=nm)
        mark(ax)
        ax.set_xlim(0, ts[-1])
        ax.set_ylim(0, max(ex["E"][:4, :show].max(), 1e-9) * 1.75)
        ax.set_ylabel("Layer 1\nrate", labelpad=2)
        ax.tick_params(labelbottom=False)
        ax.legend(loc="upper right", ncol=4, handlelength=1.0, borderpad=0.2,
                  columnspacing=0.9)
        clean_axis(ax)
        _letter(fig, ax, "b")

        # -- one row per unit --
        ymax = max(ex["y"][:, :show].max(), 1e-9) * 1.15
        for i, r in enumerate(rows):
            ax = fig.add_subplot(gs[2 + i])
            tok, col = owner.get(r["unit"], (None, C_CLOUD))
            ax.fill_between(ts, 0, ex["y"][r["unit"], :show], color=col,
                            lw=0, alpha=0.85, zorder=3)
            mark(ax)
            ax.set_xlim(0, ts[-1])
            ax.set_ylim(0, ymax)
            ax.set_ylabel(f"unit {r['unit']}", rotation=0, ha="right",
                          va="center", labelpad=6, color=col)
            ax.annotate(r["top_name"], xy=(0.995, 0.86),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.2, color=col)
            if i < n_units - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time (s)")
            clean_axis(ax)
            if i == 0:
                _letter(fig, ax, "c")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 3 -- how the population divided itself
# ---------------------------------------------------------------------
def _mask_panel(ax, M: np.ndarray, cfg, title: str, col: str) -> None:
    """One unit's mask, as the coincidence map it is."""
    n = M.shape[0]
    # A committed mask is one large entry on a low pedestal, so a linear
    # scale renders everything except that entry as white and the panel says
    # nothing about what the rest of the mask is doing.  The power norm keeps
    # the winning entry obviously dominant while leaving the pedestal legible.
    im = ax.imshow(M, cmap="Greys", norm=PowerNorm(0.45, vmin=0.0,
                                                   vmax=max(M.max(), 1e-12)),
                   interpolation="nearest", aspect="equal")
    for (i, j), c in (((1, 0), C_AB), ((3, 2), C_CD)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=c, lw=1.0, zorder=5))
    ax.set_xticks([0, 3, n - 1])
    ax.set_xticklabels(["A", "D", f"{n - 1}"], fontsize=5.8)
    ax.set_yticks([0, 3, n - 1])
    ax.set_yticklabels(["A", "D", f"{n - 1}"], fontsize=5.8)
    ax.set_xlabel("recent", labelpad=1.5)
    ax.set_ylabel("now", labelpad=1.5)
    ax.set_title(title, color=col, pad=3)
    ax.tick_params(length=1.4)
    return im


def allocation_figure(store: Dict, stem: str = "interplay2_allocation"):
    conds = [c for c in ("paired", "shuffled", "sync")
             if c in store["conditions"]]
    ex = store["example"]
    cfg = ex["cfg"]

    def get(cond: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in store["conditions"][cond]],
                        dtype=float)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(112)))
        gs = fig.add_gridspec(2, 4, left=0.068, right=0.982, top=0.90,
                              bottom=0.125, wspace=0.52, hspace=0.60)

        # a-d -- the four masks of the example run
        rows = sorted(ex["rows"], key=lambda r: -max(r["enrich_AB"],
                                                     r["enrich_CD"]))
        for i, r in enumerate(rows):
            ax = fig.add_subplot(gs[0, i])
            col = (C_AB if r["top"] == (1, 0)
                   else C_CD if r["top"] == (3, 2) else C_INK)
            _mask_panel(ax, ex["masks"][r["unit"]], cfg,
                        f"unit {r['unit']}: {r['top_name']}", col)
            _letter(fig, ax, "abcd"[i])

        # e -- the pair that exists against the pair that does not
        #
        # Which of AB and CD a unit took is already visible in a-d, so the
        # useful contrast here is between the two orderings the stimulus
        # contains and the two it does not.  In `sync` the four channels are
        # one object and the cross entries are not spurious -- they are the
        # object -- which is what separates a merge from two real tokens.
        ax = fig.add_subplot(gs[1, 0])
        series, colors, labels = [], [], []
        for cond in conds:
            runs = store["conditions"][cond]
            for keys, col, nm in ((("enrich_AB", "enrich_CD"), C_MODEL,
                                   "within"),
                                  (("enrich_DA", "enrich_BC"), C_CLOUD,
                                   "cross")):
                series.append(np.array(
                    [np.mean([max(u[k] for u in r["rows"]) for k in keys])
                     for r in runs]))
                colors.append(col)
                labels.append(nm)
        _dots(ax, series, colors, labels, ylabel="Best mask entry (x flat)",
              groups=tuple((c, 2 * k, 2 * k + 1)
                           for k, c in enumerate(conds)))
        _reference(ax, 1.0, "flat mask")
        ax.set_yscale("log")
        ax.set_ylim(0.2, 60)
        ax.tick_params(axis="x", labelsize=6.0)
        for lab in ax.get_xticklabels():
            lab.set_rotation(38)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
        ax.set_title("Ordered pair")
        _letter(fig, ax, "e")

        # f -- how many runs gave each verdict
        ax = fig.add_subplot(gs[1, 1])
        order = ["two units", "one unit, both tokens", "one token only",
                 "neither token"]
        short = ["two", "merged", "one", "none"]
        width = 0.8 / len(conds)
        for k, cond in enumerate(conds):
            v = [r["verdict"] for r in store["conditions"][cond]]
            frac = [np.mean([x == o for x in v]) for o in order]
            ax.bar(np.arange(len(order)) + (k - (len(conds) - 1) / 2) * width,
                   frac, width=width * 0.92, lw=0,
                   color=[C_MODEL, COLORS["teal"], C_CLOUD][k], label=cond)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short)
        ax.set_ylabel("Fraction of runs")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2,
                  fontsize=5.6)
        ax.set_title("Allocation")
        clean_axis(ax)
        _letter(fig, ax, "f")

        # g -- decoding AB vs CD, against its own shuffled null
        ax = fig.add_subplot(gs[1, 2])
        series = [get(c, "decode_acc") for c in conds]
        _dots(ax, series, [C_MODEL] * len(conds), conds,
              ylabel="AB vs CD accuracy")
        null = np.mean([get(c, "decode_null").mean() for c in conds])
        _reference(ax, null, "shuffled labels")
        ax.set_ylim(0.4, 1.02)
        ax.set_title("Read-out")
        _letter(fig, ax, "g")

        # h -- what each owning unit takes to be the predecessor
        #
        # A committed mask is close to separable -- a "B is firing now" row
        # times an "A fired recently" column -- and that is not a defect: a
        # rank-one mask IS the product of the two factors a coincidence
        # subunit multiplies.  What matters is whether the second factor
        # actually picks out A, so this panel plots the row itself: given
        # that B is firing, how the unit weights every possible predecessor.
        ax = fig.add_subplot(gs[1, 3])
        n = ex["masks"].shape[1]
        prof, marks = [], []
        for (now, prev, col, nm) in ((1, 0, C_AB, "before B"),
                                     (3, 2, C_CD, "before D")):
            owner = [r["unit"] for r in ex["rows"] if r["top"] == (now, prev)]
            if not owner:
                continue
            row = ex["masks"][owner[0], now].copy()
            row[now] = 0.0                     # the diagonal is excluded
            prof.append((row / max(row.sum(), 1e-12), col, nm, prev))
        for k, (row, col, nm, prev) in enumerate(prof):
            ax.bar(np.arange(n) + (k - (len(prof) - 1) / 2) * 0.42, row,
                   width=0.40, color=col, lw=0, label=nm)
            marks.append((prev, col))
        _reference(ax, 1.0 / (n - 1), "flat")
        for prev, col in marks:
            ax.annotate(["A", "B", "C", "D"][prev], xy=(prev, 0.0),
                        xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=6.4, color=col,
                        fontweight="bold", annotation_clip=False)
        ax.set_xlabel("Predecessor channel", labelpad=7)
        ax.set_ylabel("Share of the unit's row")
        ax.set_xticks([0, 5, 10, 15, n - 1])
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2,
                  fontsize=5.8)
        ax.set_title("Predecessor profile")
        clean_axis(ax)
        _letter(fig, ax, "h")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 4 -- layer 1's connectivity
# ---------------------------------------------------------------------
def _weight_panel(ax, W: np.ndarray, title: str, vmax: float,
                  *, mark: bool = True, cbar_label: str = "") -> None:
    """One connectivity matrix, drawn as the map it is."""
    n = W.shape[0]
    im = ax.imshow(W, cmap="Greys", norm=PowerNorm(0.5, vmin=0.0, vmax=vmax),
                   interpolation="nearest", aspect="equal")
    if mark:
        # Solid = the ordering the stimulus contains, dashed = its reverse.
        # Drawing both is the point: the rule is directional, so the reverse
        # entry is not merely smaller, it is driven to zero.
        for (i, j), c in (((1, 0), C_AB), ((3, 2), C_CD)):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=c, lw=1.0, zorder=5))
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                                   edgecolor=c, lw=0.8, ls=(0, (1.6, 1.2)),
                                   zorder=5))
    ax.set_xticks([0, 3, n - 1])
    ax.set_xticklabels(["A", "D", f"{n - 1}"], fontsize=5.8)
    ax.set_yticks([0, 3, n - 1])
    ax.set_yticklabels(["A", "D", f"{n - 1}"], fontsize=5.8)
    ax.set_xlabel("presynaptic", labelpad=1.5)
    ax.set_ylabel("postsynaptic", labelpad=1.5)
    ax.set_title(title, pad=3)
    ax.tick_params(length=1.4)
    return im


def layer1_figure(store: Dict, stem: str = "interplay2_layer1"):
    """The recurrent map layer 1 learns, and the architecture it sits in.

    Layer 2 is not the only place the token shows up: layer 1's own
    trace-based rate-STDP on the recurrent E->E weights is driven by the same
    contingency, and it finds it.  This figure is that map.  It matters for
    two reasons -- it says the contingency is visible in the cortex itself
    and not only in the readout, and it says what layer 2 is reading when the
    `raw` control loses the effect.

    ``W[i, j]`` is the weight from channel j onto channel i, the same
    convention as a layer-2 mask: row is "now", column is "recently".
    ``W_init_scale`` is zero in this configuration, so every weight drawn
    here was learned from the stream; none of it is initialisation.
    """
    conds = [c for c in ("paired", "shuffled", "sync")
             if c in store["conditions"]]
    ex = store["example"]
    n_tok = ex["cfg"].n_token_channels

    def stack(cond: str) -> np.ndarray:
        Ws = [r["W1"] for r in store["conditions"][cond]
              if r.get("W1") is not None]
        return np.stack(Ws) if Ws else np.empty((0, 0, 0))

    mats = {c: stack(c) for c in conds}
    have = [c for c in conds if mats[c].size]
    if not have:
        return {}

    # One colour scale across all three, or "shuffled looks empty" would be a
    # property of its own normalisation rather than of the weights.
    vmax = max(float(mats[c].mean(axis=0).max()) for c in have)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(112)))
        gs = fig.add_gridspec(2, 3, left=0.068, right=0.958, top=0.90,
                              bottom=0.115, wspace=0.46, hspace=0.62)

        # a-c -- the learned map, one per condition, seed-averaged
        im = None
        for i, cond in enumerate(have[:3]):
            ax = fig.add_subplot(gs[0, i])
            im = _weight_panel(ax, mats[cond].mean(axis=0),
                               f"W after {cond}", vmax)
            _letter(fig, ax, "abc"[i])
        cax = fig.add_axes([0.966, 0.60, 0.008, 0.26])
        fig.colorbar(im, cax=cax)
        cax.set_ylabel("weight", labelpad=3, fontsize=6.0)
        cax.tick_params(labelsize=5.4, length=1.4)
        fig.text(0.068, 0.925, "solid: A\u2192B and C\u2192D    "
                 "dashed: the reverse", fontsize=6.0, color=COLORS["ash"],
                 va="bottom", ha="left")

        # d -- presynaptic profile of the two token rows
        ax = fig.add_subplot(gs[1, 0])
        Wp = mats[have[0]].mean(axis=0)
        n = Wp.shape[0]
        for k, (now, prev, col, nm) in enumerate(
                ((1, 0, C_AB, "onto B"), (3, 2, C_CD, "onto D"))):
            row = Wp[now].copy()
            # The self weight IS plastic in this configuration, but it is not
            # a transition -- it is sustained activity in one channel -- so it
            # does not belong in a plot of what precedes B.
            row[now] = 0.0
            ax.bar(np.arange(n) + (k - 0.5) * 0.42, row, width=0.40,
                   color=col, lw=0, label=nm)
            ax.annotate(["A", "B", "C", "D"][prev], xy=(prev, 0.0),
                        xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=6.4, color=col,
                        fontweight="bold", annotation_clip=False)
        ax.set_xlabel("Presynaptic channel", labelpad=7)
        ax.set_ylabel("Recurrent weight")
        ax.set_xticks([0, 5, 10, 15, n - 1])
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2,
                  fontsize=5.8)
        ax.set_title("Input to the token rows")
        clean_axis(ax)
        _letter(fig, ax, "d")

        # e -- the token weight itself, one dot per seed, per condition
        #
        # The comparison that carries the claim is ACROSS conditions, not
        # token-against-cloud within one.  The two tones of a token are
        # constrained never to overlap, while two cloud channels on opposite
        # clocks routinely do, and near-simultaneous pairs collect
        # potentiation a strictly ordered pair cannot.  So the cloud level is
        # drawn as a reference rule rather than as a matched column: it says
        # where an unstructured pair sits, not what the token would have been.
        ax = fig.add_subplot(gs[1, 1])
        series = []
        clouds = []
        for cond in have:
            Ws = mats[cond]
            series.append(np.array([(w[1, 0] + w[3, 2]) / 2.0 for w in Ws]))
            clouds.append(np.mean([w[n_tok:, n_tok:][
                ~np.eye(w.shape[0] - n_tok, dtype=bool)].mean() for w in Ws]))
        _dots(ax, series, [C_MODEL] * len(have), have,
              ylabel="Recurrent weight, token pair")
        _reference(ax, float(np.mean(clouds)), "cloud pair (not drive-matched)")
        ax.set_ylim(bottom=0.0)
        ax.set_title("Token pair weight")
        _letter(fig, ax, "e")

        # f -- the fixed architecture the learning sits inside
        ax = fig.add_subplot(gs[1, 2])
        M_IE = ex.get("M_IE")
        if M_IE is not None:
            _weight_panel(ax, M_IE, "I to E (fixed)", float(M_IE.max()),
                          mark=False)
            ax.annotate("selective: strong on the diagonal,\n"
                        "weak and uniform off it",
                        xy=(0.5, -0.30), xycoords="axes fraction",
                        ha="center", va="top", fontsize=5.8,
                        color=COLORS["ash"], annotation_clip=False)
        _letter(fig, ax, "f")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
def make_figures(store: Dict) -> Dict[str, Dict[str, Path]]:
    out = {}
    ex = store["example"]
    out["stimulus"] = stimulus_figure(ex)
    out["cloud"] = cloud_figure(ex)
    out["tape"] = tape_figure(ex)
    out["allocation"] = allocation_figure(store)
    layer1 = layer1_figure(store)
    if layer1:
        out["layer1"] = layer1
    for name, paths in out.items():
        print(f"  wrote {paths['png']}")
    return out
