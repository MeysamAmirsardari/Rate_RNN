"""
layer2_syllable.saffran.tape
============================

Raw views of the Saffran run: the input on the top row, every layer 2 unit on a
row of its own underneath, on one shared clock.

Nothing is averaged, epoched or fitted.  The window is taken from the **end** of
the exposure stream, after roughly five hundred words have gone past, so what is
drawn is the settled behaviour rather than the transient at the start.  The time
axis shows real elapsed time in the stream.

Colour code
-----------
Top row      each token is filled with the colour of the word it belongs to;
             dashed lines are word boundaries.  The model sees neither: the
             stream is isochronous and every token is identical.
Unit rows    green if the unit learned a transition that lies inside a word,
             red if it learned one that straddles a boundary.

Run
---
    python -m layer2_syllable.saffran.tape
"""
from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model0 import selective_inh
from layer2_syllable.config import L2Config
from layer2_syllable.run_ab_ba import INK, LAYER1, MUTED, _caption, _tidy, layer1_rates
from layer2_syllable.stimulus import build_stream, chunk_windows
from layer2_syllable.saffran.run_saffran import (GAP, N_CH, TONE_DUR, WORDS,
                                                 WORD_NAME, analyse, expose,
                                                 part_words)

# pale colours for the four words, saturated ones for the two unit classes
WORD_COL = ["#6b9bc9", "#e8a04f", "#7fb894", "#b39ac9"]
C_WITHIN, C_BOUND = "#1f6b4a", "#a83f28"


def _tokens_in(st, k0, k1):
    """(onset_sample, channel, word_index) for chunks k0 up to k1."""
    out = []
    for k in range(k0, k1):
        w = int(st["labels"][k])
        for j, o in enumerate(st["tone_onsets"][k]):
            out.append((int(o), int(WORDS[w][j]), w))
    return sorted(out)


def _unit_rows(fig, gs, comm, cls_of, y, t0, t1, ts):
    """One row per committed unit, no text inside the plot area."""
    ymax = max(y[comm, t0:t1].max(), 1e-9) * 1.18
    axes = []
    for r, u in enumerate(comm):
        ax = _tidy(fig.add_subplot(gs[1 + r, 0]))
        col = C_WITHIN if cls_of[u] == "within" else C_BOUND
        ax.fill_between(ts, 0, y[u, t0:t1], color=col, lw=0, alpha=0.92,
                        zorder=3)
        ax.set_xlim(ts[0], ts[-1]); ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.set_ylabel(f"unit {u}", fontsize=10, rotation=0, ha="right",
                      va="center", labelpad=10, color=col)
        axes.append(ax)

    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    return axes


# ---------------------------------------------------------------------
def fig_tape(l2, an, st, y, a1cfg, fname, n_show=11):
    """A window from the end of the exposure stream."""
    dt = a1cfg.dt
    cls_of = {r["unit"]: r["cls"] for r in an["rows"]}
    comm = [r["unit"] for r in an["rows"]]

    n_chunks = len(st["starts"])
    k0 = max(0, n_chunks - n_show)
    t0 = int(st["starts"][k0])
    t1 = min(int(st["starts"][-1] + st["lengths"][-1]), y.shape[1])
    ts = np.arange(t0, t1) * dt
    n_tone = st["n_tone"]

    rows = 1 + len(comm)
    fig = plt.figure(figsize=(16.5, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(rows + 1, 1,
                          height_ratios=[3.0] + [0.72] * len(comm) + [0.34])
    fig.suptitle(f"Late in the stream, after about {k0} words have gone past",
                 fontsize=14.5, fontweight="bold")

    def boundaries(ax, label=False):
        for k in range(k0, n_chunks):
            x = st["starts"][k] * dt
            if x > ts[-1]:
                break
            ax.axvline(x, color="0.35", lw=1.1, ls="--", alpha=0.75, zorder=2)
            if label:
                ax.text(x + 0.012, N_CH + 0.15, WORD_NAME[st["labels"][k]],
                        fontsize=10.5, fontweight="bold",
                        color=WORD_COL[st["labels"][k]], va="bottom", zorder=5)

    # ---- top row: the input, each token coloured by its word ----
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    for o, ch, w in _tokens_in(st, k0, n_chunks):
        if o * dt > ts[-1]:
            continue
        ax.add_patch(plt.Rectangle((o * dt, ch - 0.40), n_tone * dt, 0.80,
                                   facecolor=WORD_COL[w], edgecolor="none",
                                   zorder=3))
    boundaries(ax, label=True)
    ax.set_xlim(ts[0], ts[-1]); ax.set_ylim(-0.9, N_CH + 0.9)
    ax.set_yticks(range(N_CH))
    ax.set_yticklabels([str(c) for c in range(N_CH)], fontsize=8)
    ax.set_ylabel("input\nchannel", fontsize=10)
    ax.tick_params(labelbottom=False)

    axes = _unit_rows(fig, gs, comm, cls_of, y, t0, t1, ts)
    for a in axes:
        boundaries(a)
    axes[-1].set_xlabel("time in the stream (s)", fontsize=10)

    within = [u for u in comm if cls_of[u] == "within"]
    bound = [u for u in comm if cls_of[u] != "within"]
    _caption(fig, gs[rows, 0],
             "Top row: each token is coloured by the word it belongs to; dashed lines are the word boundaries. "
             "The model sees neither, since the stream is isochronous and every token is identical.\n"
             f"Green units learned a transition inside a word ({', '.join(str(u) for u in within)}). "
             f"Red units learned one that straddles a boundary ({', '.join(str(u) for u in bound)}).")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}   window {ts[0]:.1f} to {ts[-1]:.1f} s")


# ---------------------------------------------------------------------
def fig_tape_test(l2, an, a1cfg, fname, n_show=3):
    """The isolated test items, drawn the same way."""
    dt = a1cfg.dt
    cls_of = {r["unit"]: r["cls"] for r in an["rows"]}
    comm = [r["unit"] for r in an["rows"]]

    pws = part_words(WORDS)
    items = list(WORDS[:n_show]) + [p for p, _ in pws[:2 * n_show:2]][:n_show]
    kind = ["word"] * n_show + ["part word"] * n_show
    origin = {ch: wi for wi, w in enumerate(WORDS) for ch in w}

    st = build_stream(items, [1.0] * len(items), len(items), N_CH, a1cfg.dt,
                      tone_dur=TONE_DUR, intra_gap=GAP, inter_gap=0.320,
                      seed=5, order=np.arange(len(items)))
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=0)
    l2.reset_state()
    te = l2.run(E, a1cfg.dt, learn=False)
    y = te["y"]
    t1 = min(int(st["starts"][-1] + st["lengths"][-1]), y.shape[1])
    ts = np.arange(0, t1) * dt
    n_tone = st["n_tone"]

    rows = 1 + len(comm)
    fig = plt.figure(figsize=(16.5, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(rows + 1, 1,
                          height_ratios=[3.0] + [0.72] * len(comm) + [0.34])
    fig.suptitle("The test items: three words, then three part words",
                 fontsize=14.5, fontweight="bold")

    def marks(ax, label=False):
        for k in range(len(st["starts"])):
            x = st["starts"][k] * dt
            if x > ts[-1]:
                break
            ax.axvline(x, color="0.35", lw=1.1, ls="--", alpha=0.7, zorder=2)
            if label:
                ax.text(x + 0.012, N_CH + 0.15, kind[k], fontsize=10.5,
                        fontweight="bold",
                        color=INK if kind[k] == "word" else MUTED,
                        va="bottom", zorder=5)

    ax = _tidy(fig.add_subplot(gs[0, 0]))
    for k, ons in enumerate(st["tone_onsets"]):
        for j, o in enumerate(ons):
            ch = items[k][j]
            ax.add_patch(plt.Rectangle((o * dt, ch - 0.40), n_tone * dt, 0.80,
                                       facecolor=WORD_COL[origin[ch]],
                                       edgecolor="none", zorder=3))
    marks(ax, label=True)
    ax.set_xlim(0, ts[-1]); ax.set_ylim(-0.9, N_CH + 0.9)
    ax.set_yticks(range(N_CH))
    ax.set_yticklabels([str(c) for c in range(N_CH)], fontsize=8)
    ax.set_ylabel("input\nchannel", fontsize=10)
    ax.tick_params(labelbottom=False)

    axes = _unit_rows(fig, gs, comm, cls_of, y, 0, t1, ts)
    for a in axes:
        marks(a)
    axes[-1].set_xlabel("time (s)", fontsize=10)

    wins = chunk_windows(st, pad_s=0.05)
    tot = [float(y[comm, a:b].max(axis=1).sum()) for a, b in wins]
    _caption(fig, gs[rows, 0],
             "Colours carry over from the exposure stream, so a part word is visibly built from two different words. "
             "Weights are frozen.\n"
             "Summed response, left to right:   " +
             "   ".join(f"{kind[k]} {tot[k]:.0f}" for k in range(len(items))) +
             ".   A word contains two frequent transitions, a part word only one.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CH, **LAYER1)
    l2cfg = dc.replace(L2Config(), n_units=8, tau_decay=0.150)

    print("[ saffran tape ] exposing, then drawing the end of the stream raw")
    l2, st, tr, _ = expose(l2cfg, a1cfg, seed=0, record_every=0)
    an = analyse(l2)
    print(f"  {an['n_units']} units committed, "
          f"{an['n_within']} of them within word")
    fig_tape(l2, an, st, tr["y"], a1cfg, str(out / "saffran_tape.png"))
    fig_tape_test(l2, an, a1cfg, str(out / "saffran_tape_test.png"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
