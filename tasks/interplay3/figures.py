"""
tasks.interplay3.figures
========================

Figures for interplay3, in the manuscript's visual system.

    interplay3_cloud        the stimulus as a tone cloud, channel order shuffled
    interplay3_stimulus     the same stimulus with its constraints measured
    interplay3_tape         unit activity along the input
    interplay3_allocation   how the population divided itself, across seeds
    interplay3_layer1       the recurrent map layer 1 learns from the same stream

Titles name the quantity and stop; chance and null levels are drawn as rules
inside the axes rather than asserted in a title; every seed is a dot.  Maps
carry no marker boxes -- the axis labels say where the word channels are.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from final_figures.style import (COLORS, clean_axis, export_figure, mm,
                                 manuscript_style)

if __package__:
    from .config import WORD_LEN, WORD_NAMES, WORDS
else:
    from tasks.interplay3.config import WORD_LEN, WORD_NAMES, WORDS  # type: ignore

OUT_DIR = Path(__file__).resolve().parent

#: One colour per word, plus the cloud.
WORD_COLORS = (COLORS["rep1"], COLORS["rep15"], COLORS["model"])
C_CLOUD = COLORS["ash"]
C_INK = COLORS["charcoal"]
C_MODEL = COLORS["model"]


def _word_of(ch: int) -> int | None:
    for w, word in enumerate(WORDS):
        if ch in word:
            return w
    return None


def _colour_of(ch: int) -> str:
    w = _word_of(ch)
    return WORD_COLORS[w] if w is not None else C_CLOUD


# ---------------------------------------------------------------------
#  Primitives
# ---------------------------------------------------------------------
def _letter(fig, ax, s: str) -> None:
    bb = ax.get_position()
    dx = mm(2.6) / fig.get_figwidth()
    dy = mm(2.4) / fig.get_figheight()
    fig.text(bb.x0 - dx, bb.y1 + dy, s, fontsize=10.0, fontweight="bold",
             color=C_INK, va="top", ha="left")


def _dots(ax, series: Sequence[np.ndarray], colors: Sequence[str],
          labels: Sequence[str], *, width: float = 0.40,
          jitter: float = 0.06, ylabel: str = "",
          groups: Sequence[tuple] = (), empty_note: str = "") -> None:
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(series, colors)):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            if empty_note:
                ax.annotate(empty_note, xy=(i, 0.5),
                            xycoords=("data", "axes fraction"), ha="center",
                            va="center", fontsize=5.8, color=COLORS["ash"],
                            rotation=90)
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
    ax.axhline(y, color=C_INK, lw=0.55, ls=(0, (3, 2)), zorder=1)
    ax.annotate(text, xy=(0.015, y), xycoords=("axes fraction", "data"),
                xytext=(0, 2.2), textcoords="offset points", fontsize=5.8,
                color=COLORS["ash"], va="bottom", ha="left")


def _raster(ax, stim: np.ndarray, cfg, dt: float, *, lw: float = 1.9,
            gap: float = 2.6, step: float = 1.25,
            cloud_scale: float = 0.22) -> None:
    """Channel-by-time raster: word channels in colour, cloud in grey.

    The twelve word channels are lifted clear of the cloud and spaced further
    apart than the thirty-eight cloud rows are, because the panel exists to
    show which row is which and fifty equally spaced rows in a 25 mm panel
    cannot do that.
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
        edges = np.flatnonzero(np.diff(np.concatenate(
            [[0], on.view(np.int8), [0]])))
        col = _colour_of(ch)
        for a, b in zip(edges[0::2], edges[1::2]):
            ax.plot([a * dt, b * dt], [y_of(ch)] * 2, color=col, lw=lw,
                    solid_capstyle="butt", zorder=3)

    ax.set_ylim(-1.0, y_of(0) + 1.0)
    ticks = [y_of(w * WORD_LEN + WORD_LEN // 2 - 1) for w in range(len(WORDS))]
    ax.set_yticks(ticks + [top_of_cloud / 2.0])
    ax.set_yticklabels(list(WORD_NAMES) + ["cloud"])
    for lab, w in zip(ax.get_yticklabels(), range(len(WORDS))):
        lab.set_color(WORD_COLORS[w])
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, left=False)


# ---------------------------------------------------------------------
#  Figure 1 -- the cloud
# ---------------------------------------------------------------------
def cloud_figure(ex: Dict, stem: str = "interplay3_cloud",
                 n_blocks: int = 2, seed: int = 5):
    """The stimulus drawn the way a tone cloud is normally drawn.

    Channel order is shuffled, and among candidate permutations the one that
    pushes the twelve word channels furthest apart is taken: adjacent rows
    would let the reader group a word by proximity, a cue neither the model
    nor a listener has.
    """
    cfg, dt = ex["cfg"], ex["dt"]
    stim = ex["stim"]
    show = min(stim.shape[1], n_blocks * cfg.block_samples)
    n = stim.shape[0]
    n_tok = cfg.n_token_channels

    rng = np.random.default_rng(seed)
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
        fig = plt.figure(figsize=(mm(88), mm(112)))
        ax = fig.add_axes([0.02, 0.075, 0.96, 0.895])

        for ch in range(n):
            on = stim[ch, :show] > 0
            if not on.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(
                [[0], on.view(np.int8), [0]])))
            col = _colour_of(ch) if ch < n_tok else C_INK
            y = row_of[ch]
            for a, b in zip(edges[0::2], edges[1::2]):
                ax.plot([a * dt, b * dt], [y, y], color=col, lw=2.4,
                        solid_capstyle="butt")

        ax.set_xlim(-0.02 * show * dt, show * dt * 1.02)
        ax.set_ylim(-1.2, n + 0.4)
        ax.set_axis_off()

        bar = 0.5
        ax.plot([0.0, bar], [-0.9, -0.9], color=C_INK, lw=1.2,
                solid_capstyle="butt")
        ax.annotate(f"{bar:.1f} s", xy=(bar / 2, -1.15), ha="center",
                    va="top", fontsize=6.5, color=C_INK)

        x = 0.0
        for w, name in enumerate(WORD_NAMES):
            ax.annotate(name, xy=(x, 1.005), xycoords="axes fraction",
                        ha="left", va="bottom", fontsize=7.0,
                        color=WORD_COLORS[w], fontweight="bold")
            x += 0.13
        ax.annotate("cloud", xy=(x + 0.04, 1.005), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=7.0, color=C_INK,
                    fontweight="bold")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 2 -- the stimulus, measured
# ---------------------------------------------------------------------
def stimulus_figure(ex: Dict, stem: str = "interplay3_stimulus"):
    cfg, dt = ex["cfg"], ex["dt"]
    stim = ex["stim"]
    show = min(stim.shape[1], 2 * cfg.block_samples)
    ts = np.arange(show) * dt

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(84)))
        gs = fig.add_gridspec(2, 3, left=0.075, right=0.975, top=0.885,
                              bottom=0.11, wspace=0.42, hspace=0.72,
                              height_ratios=[1.45, 1.0])

        ax = fig.add_subplot(gs[0, :])
        _raster(ax, stim[:, :show], cfg, dt)
        ax.set_xlim(0, ts[-1])
        ax.set_xlabel("Time (s)")
        ax.set_title("Two blocks of the stream")
        _letter(fig, ax, "a")

        ax = fig.add_subplot(gs[1, 0])
        on = ex["checks"]["on_time"]
        ax.bar(np.arange(len(on)), on,
               color=[_colour_of(c) for c in range(len(on))], width=0.9, lw=0)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Total on-time (s)")
        ax.set_ylim(0, on.max() * 1.25)
        ax.set_title("Drive per channel")
        clean_axis(ax)
        _letter(fig, ax, "b")

        ax = fig.add_subplot(gs[1, 1])
        vals = np.arange(0, 7)
        ax.bar(vals - 0.19, [np.mean(ex["checks"]["n_simul"] == v)
                             for v in vals], width=0.36, color=C_MODEL, lw=0,
               label="all channels")
        ax.bar(vals + 0.19, [np.mean(ex["checks"]["n_simul_bg"] == v)
                             for v in vals], width=0.36, color=C_CLOUD, lw=0,
               label="cloud only")
        ax.set_xlabel("Tones on at once")
        ax.set_ylabel("Fraction of time")
        ax.set_xticks(vals)
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2,
                  fontsize=5.8)
        ax.set_title("Simultaneity")
        clean_axis(ax)
        _letter(fig, ax, "c")

        # d -- the lag between consecutive tones of a word
        ax = fig.add_subplot(gs[1, 2])
        centres = np.arange(-cfg.block_slots, cfg.block_slots + 1) * cfg.slot
        edges = np.append(centres - cfg.slot / 2, centres[-1] + cfg.slot / 2)
        for w, name in enumerate(WORD_NAMES):
            lag = np.concatenate([ex["checks"]["lags"][(w, p)]
                                  for p in range(WORD_LEN - 1)])
            h, _ = np.histogram(lag, bins=edges)
            ax.bar(centres + (w - 1) * cfg.slot * 0.30, h,
                   width=cfg.slot * 0.28, color=WORD_COLORS[w], lw=0,
                   label=name)
        ax.set_xlabel("Lag to the next tone (ms)")
        ax.set_ylabel("Tones")
        ax.set_xlim(-6 * cfg.slot, 6 * cfg.slot)
        ax.legend(loc="upper left", handlelength=1.0, borderpad=0.2,
                  fontsize=5.8)
        ax.set_title("Within-word lag")
        clean_axis(ax)
        _letter(fig, ax, "d")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 3 -- the tape
# ---------------------------------------------------------------------
def tape_figure(ex: Dict, stem: str = "interplay3_tape", n_show: int = 6):
    """The deepest units' activity on the same clock as the stimulus."""
    cfg, dt = ex["cfg"], ex["dt"]
    rows = [r for r in ex["rows"]][:n_show]
    n_rows = len(rows)
    show = ex["y"].shape[1]
    ts = np.arange(show) * dt

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(38 + 13 * n_rows)))
        gs = fig.add_gridspec(1 + n_rows, 1, left=0.115, right=0.985,
                              top=0.935, bottom=0.075, hspace=0.42,
                              height_ratios=[2.4] + [1.0] * n_rows)

        def mark(ax) -> None:
            """Each word's completion: the window its last tone opens."""
            for w, word in enumerate(WORDS):
                for o in ex["onsets"][word[-1]]:
                    ax.axvspan(o * dt, (o + cfg.tone_dur) * dt,
                               color=WORD_COLORS[w], alpha=0.16, lw=0,
                               zorder=0)

        ax = fig.add_subplot(gs[0])
        _raster(ax, ex["stim"], cfg, dt)
        mark(ax)
        ax.set_xlim(0, ts[-1])
        ax.tick_params(labelbottom=False)
        ax.set_title("Stimulus and the deepest layer-2 units on one clock")
        _letter(fig, ax, "a")

        ymax = max(ex["y"].max(), 1e-9) * 1.15
        for i, r in enumerate(rows):
            ax = fig.add_subplot(gs[1 + i])
            col = (WORD_COLORS[r["word"]] if r["spans"] and r["word"] is not None
                   else C_CLOUD)
            ax.fill_between(ts, 0, ex["y"][r["unit"]], color=col, lw=0,
                            alpha=0.85, zorder=3)
            mark(ax)
            ax.set_xlim(0, ts[-1])
            ax.set_ylim(0, ymax)
            ax.set_ylabel(f"unit {r['unit']}", rotation=0, ha="right",
                          va="center", labelpad=6, color=col)
            tag = (f"{WORD_NAMES[r['word']]}, depth {r['depth']}"
                   if r["word"] is not None else f"depth {r['depth']}")
            ax.annotate(f"{r['now_name']} now  |  {tag}", xy=(0.995, 0.86),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.2, color=col)
            if i < n_rows - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time (s)")
            clean_axis(ax)
            if i == 0:
                _letter(fig, ax, "b")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 4 -- allocation
# ---------------------------------------------------------------------
def _mask_rate_panel(ax, l2_mask: np.ndarray, tau: np.ndarray, n_ch: int,
                     row: int, word: int, title: str, col: str) -> None:
    """One spanning unit's mask row, as weight against rate.

    The image version of this -- fifty predecessor channels by six rates --
    is unreadable: the immediately preceding channel is several times
    anything else and flattens the rest of the colour scale, which is exactly
    the part that carries the claim.  What matters is not the whole row but
    four curves out of it, one per position in the word, because the
    mechanism is a *staircase*: the immediately preceding tone should peak at
    a fast rate, the one before it at a slower one, and the first tone of the
    word at the slowest.  Anything else is a unit that has learned a pair and
    stopped.
    """
    R = tau.size
    grid = l2_mask[row].reshape(R, n_ch).T          # (predecessor, rate)
    peak = max(grid.max(), 1e-12)
    ms = tau * 1e3

    cloud = grid[len(WORDS) * WORD_LEN:] / peak
    ax.fill_between(ms, cloud.min(axis=0), cloud.max(axis=0),
                    color=COLORS["mist"], lw=0, zorder=1)
    ax.plot(ms, cloud.mean(axis=0), color=C_CLOUD, lw=0.9, ls=(0, (2.5, 1.5)),
            zorder=2, label="cloud")

    shades = ["#d9d2e9", "#9a8cc4", "#5b4f8f"]      # 3 back, 2 back, 1 back
    for back in (3, 2, 1):
        p = WORDS[word][WORD_LEN - 1 - back]
        if p == row:
            continue
        prof = grid[p] / peak
        c = shades[3 - back]
        ax.plot(ms, prof, color=c, lw=1.4, zorder=4,
                label=f"{chr(ord('A') + p)}  ({back} back)")
        j = int(np.argmax(prof))
        ax.scatter([ms[j]], [prof[j]], s=16, color=c, zorder=5,
                   clip_on=False)

    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([f"{m:.0f}" for m in ms], fontsize=5.6)
    ax.minorticks_off()
    ax.set_xlabel("Filter rate (ms)")
    ax.set_ylabel("Mask weight (of row peak)")
    ax.set_ylim(0, 1.08)
    ax.set_title(title, color=col, pad=3)
    ax.legend(loc="upper right", handlelength=1.1, borderpad=0.2,
              fontsize=5.4, labelspacing=0.25)
    clean_axis(ax)


def allocation_figure(store: Dict, stem: str = "interplay3_allocation"):
    conds = [c for c in ("paired", "shuffled", "sync")
             if c in store["conditions"]]
    ex = store["example"]
    n_ch = ex["cfg"].n_channels

    def get(cond: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in store["conditions"][cond]],
                        dtype=float)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(112)))
        gs = fig.add_gridspec(2, 3, left=0.072, right=0.978, top=0.90,
                              bottom=0.125, wspace=0.44, hspace=0.62)

        # a-c -- one spanning unit per word, if there is one
        spanning = [r for r in ex["rows"] if r["spans"]]
        by_word = {}
        for r in spanning:
            by_word.setdefault(r["word"], r)
        for w in range(len(WORDS)):
            ax = fig.add_subplot(gs[0, w])
            r = by_word.get(w)
            if r is None:
                ax.set_axis_off()
                ax.annotate(f"{WORD_NAMES[w]}: no spanning unit",
                            xy=(0.5, 0.5), xycoords="axes fraction",
                            ha="center", va="center", fontsize=6.5,
                            color=COLORS["ash"])
            else:
                _mask_rate_panel(ax, ex["masks"][r["unit"]], ex["tau"], n_ch,
                                 r["now"], w,
                                 f"{WORD_NAMES[w]}: unit {r['unit']}, "
                                 f"{r['now_name']} now", WORD_COLORS[w])
            _letter(fig, ax, "abc"[w])

        # d -- how many words got a spanning unit
        ax = fig.add_subplot(gs[1, 0])
        _dots(ax, [get(c, "n_spanning") * 0 + np.array(
            [len(r["words_covered"]) for r in store["conditions"][c]])
            for c in conds], [C_MODEL] * len(conds), conds,
            ylabel="Words with a spanning unit")
        ax.set_ylim(-0.15, len(WORDS) + 0.15)
        ax.set_yticks(range(len(WORDS) + 1))
        ax.set_title("Coverage")
        _letter(fig, ax, "d")

        # e -- the depth distribution of committed units
        ax = fig.add_subplot(gs[1, 1])
        width = 0.8 / len(conds)
        depths = range(WORD_LEN)
        for k, cond in enumerate(conds):
            runs = store["conditions"][cond]
            mean = [np.mean([r["depth_hist"].get(d, 0) for r in runs])
                    for d in depths]
            ax.bar(np.arange(len(list(depths)))
                   + (k - (len(conds) - 1) / 2) * width, mean,
                   width=width * 0.92, lw=0,
                   color=[C_MODEL, COLORS["teal"], C_CLOUD][k], label=cond)
        ax.set_xticks(range(WORD_LEN))
        ax.set_xlabel("Span depth")
        ax.set_ylabel("Committed units")
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2,
                  fontsize=5.6)
        ax.set_title("How deep the units reach")
        clean_axis(ax)
        _letter(fig, ax, "e")

        # f -- three-way word decoding against its own shuffled null
        ax = fig.add_subplot(gs[1, 2])
        _dots(ax, [get(c, "decode_acc") for c in conds],
              [C_MODEL] * len(conds), conds, ylabel="Word decoding accuracy")
        null = float(np.mean([get(c, "decode_null").mean() for c in conds]))
        _reference(ax, null, "shuffled labels")
        ax.set_ylim(0.2, 1.02)
        ax.set_title("Read-out")
        _letter(fig, ax, "f")

        paths = export_figure(fig, OUT_DIR / stem)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------
#  Figure 5 -- layer 1's connectivity
# ---------------------------------------------------------------------
def _weight_panel(ax, W: np.ndarray, title: str, vmax: float):
    n = W.shape[0]
    im = ax.imshow(W, cmap="Greys", norm=PowerNorm(0.5, vmin=0.0, vmax=vmax),
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([0, 3, 7, 11, n - 1])
    ax.set_xticklabels(["A", "D", "H", "L", f"{n - 1}"], fontsize=5.6)
    ax.set_yticks([0, 3, 7, 11, n - 1])
    ax.set_yticklabels(["A", "D", "H", "L", f"{n - 1}"], fontsize=5.6)
    ax.set_xlabel("presynaptic", labelpad=1.5)
    ax.set_ylabel("postsynaptic", labelpad=1.5)
    ax.set_title(title, pad=3)
    ax.tick_params(length=1.4)
    return im


def layer1_figure(store: Dict, stem: str = "interplay3_layer1"):
    """The recurrent map layer 1 learns from the same stream.

    ``W[i, j]`` is the weight from j onto i.  ``W_init_scale`` is zero, so
    every weight drawn here was learned.  A word should appear as a chain of
    three entries just below the diagonal of its own block.
    """
    conds = [c for c in ("paired", "shuffled", "sync")
             if c in store["conditions"]]

    def stack(cond: str) -> np.ndarray:
        Ws = [r["W1"] for r in store["conditions"][cond]
              if r.get("W1") is not None]
        return np.stack(Ws) if Ws else np.empty((0, 0, 0))

    mats = {c: stack(c) for c in conds}
    have = [c for c in conds if mats[c].size]
    if not have:
        return {}

    n_tok = store["example"]["cfg"].n_token_channels
    vmax = max(float(mats[c].mean(axis=0).max()) for c in have)

    with manuscript_style():
        fig = plt.figure(figsize=(mm(183), mm(112)))
        gs = fig.add_gridspec(2, 3, left=0.068, right=0.958, top=0.90,
                              bottom=0.125, wspace=0.46, hspace=0.62)

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

        # d -- the word block, enlarged
        ax = fig.add_subplot(gs[1, 0])
        _weight_panel(ax, mats[have[0]].mean(axis=0)[:n_tok, :n_tok],
                      f"word channels only ({have[0]})", vmax)
        ax.set_xticks(range(0, n_tok, 2))
        ax.set_xticklabels([chr(ord("A") + c) for c in range(0, n_tok, 2)],
                           fontsize=5.6)
        ax.set_yticks(range(0, n_tok, 2))
        ax.set_yticklabels([chr(ord("A") + c) for c in range(0, n_tok, 2)],
                           fontsize=5.6)
        _letter(fig, ax, "d")

        # e -- within-word steps against the cloud
        ax = fig.add_subplot(gs[1, 1])
        series = []
        clouds = []
        for cond in have:
            Ws = mats[cond]
            step = np.array([
                np.mean([w[word[p + 1], word[p]]
                         for word in WORDS for p in range(WORD_LEN - 1)])
                for w in Ws])
            series.append(step)
            off = ~np.eye(Ws.shape[1] - n_tok, dtype=bool)
            clouds.append(np.mean([w[n_tok:, n_tok:][off].mean() for w in Ws]))
        _dots(ax, series, [C_MODEL] * len(have), have,
              ylabel="Recurrent weight, within-word step")
        _reference(ax, float(np.mean(clouds)),
                   "cloud pair (not drive-matched)")
        ax.set_ylim(bottom=0.0)
        ax.set_title("Within-word steps")
        _letter(fig, ax, "e")

        # f -- the fixed architecture
        ax = fig.add_subplot(gs[1, 2])
        M_IE = store["example"].get("M_IE")
        if M_IE is not None:
            _weight_panel(ax, M_IE, "I to E (fixed)", float(M_IE.max()))
            ax.annotate("selective: strong on the diagonal,\n"
                        "weak and uniform off it",
                        xy=(0.5, -0.32), xycoords="axes fraction",
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
    out["cloud"] = cloud_figure(ex)
    out["stimulus"] = stimulus_figure(ex)
    out["tape"] = tape_figure(ex)
    out["allocation"] = allocation_figure(store)
    layer1 = layer1_figure(store)
    if layer1:
        out["layer1"] = layer1
    for name, paths in out.items():
        print(f"  wrote {paths['png']}")
    return out
