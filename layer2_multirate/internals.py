"""
layer2_multirate.internals
==========================

Everything inside the multi rate layer, drawn.  No summary statistics, no
averaging, and no step of the chain left implicit.

    mr_tape.png         the input and every committed unit on one clock
    mr_dissection.png   one word, followed all the way through:
                        stimulus -> E -> filterbank -> D -> mask -> response
    mr_all_masks.png     every learned mask, with what each one encodes

Run
---
    python -m layer2_multirate.internals
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model0 import selective_inh
from layer2_multirate.config import MRConfig
from layer2_multirate.layer2 import Layer2MR
from layer2_multirate.run_saffran import expose
from layer2_multirate.sweep import span_depth
from layer2_syllable.run_ab_ba import (GRID, INK, LAYER1, MUTED, _caption,
                                       _tidy, layer1_rates)
from layer2_syllable.stimulus import build_stream
from layer2_syllable.saffran.run_saffran import (GAP, N_CH, TONE_DUR, WORDS,
                                                 WORD_NAME, word_order)

WORD_COL = ["#6b9bc9", "#e8a04f", "#7fb894", "#b39ac9"]
D_COL = {2: "#1f6b4a", 1: "#d98f2b", 0: "#9aa0a6"}      # colour by span depth
D_NAME = {2: "holds a whole word", 1: "holds one transition", 0: "neither"}


def record(l2, E, dt):
    """Run the layer and keep every internal signal."""
    T = E.shape[1]
    S = np.zeros((l2.N, l2.R, T))
    Y = np.zeros((l2.cfg.n_units, T))
    l2.reset_state()
    for t in range(T):
        y, _ = l2.step(E[:, t], dt, learn=False)
        S[:, :, t] = l2.s
        Y[:, t] = y
    return S, Y


def readout(l2, a1cfg, n_words=9, seed=41):
    rng = np.random.default_rng(seed)
    order = word_order(len(WORDS), n_words, rng)
    st = build_stream(WORDS, [0.25] * 4, n_words, N_CH, a1cfg.dt,
                      tone_dur=TONE_DUR, intra_gap=GAP, inter_gap=GAP,
                      seed=seed, order=order)
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=0)
    S, Y = record(l2, E, a1cfg.dt)
    return st, E, S, Y


def unit_table(l2):
    """Committed units with their depth and what they fire on, deepest first."""
    rows = []
    for u in np.flatnonzero(l2.committed):
        d, w = span_depth(l2, u, WORDS)
        M = l2.M[u]
        i = int(np.argmax(M.sum(axis=(1, 2))))
        row = M[i]
        j1, m1 = np.unravel_index(int(np.argmax(row)), row.shape)
        rows.append(dict(unit=int(u), depth=d, now=i, p1=int(j1),
                         tau1=float(l2.tau[m1]),
                         word=(WORDS.index(w) if w is not None else None)))
    return sorted(rows, key=lambda r: (-r["depth"], r["now"]))


# ---------------------------------------------------------------------
def fig_tape(l2, st, E, Y, a1cfg, fname):
    dt = a1cfg.dt
    tab = unit_table(l2)
    T = min(int(st["starts"][-1] + st["lengths"][-1]), Y.shape[1])
    ts = np.arange(T) * dt
    n_tone = st["n_tone"]

    rows = 1 + len(tab)
    fig = plt.figure(figsize=(17, 1.05 + 0.52 * len(tab) + 2.6),
                     constrained_layout=True)
    gs = fig.add_gridspec(rows + 1, 1,
                          height_ratios=[2.6] + [0.52] * len(tab) + [0.42])
    fig.suptitle("Every committed unit, alongside the input, on one clock",
                 fontsize=14.5, fontweight="bold")

    def bounds(ax, label=False):
        for k in range(len(st["starts"])):
            x = st["starts"][k] * dt
            if x > ts[-1]:
                break
            ax.axvline(x, color="0.4", lw=1.0, ls="--", alpha=0.7, zorder=2)
            if label:
                ax.text(x + 0.01, N_CH + 0.2, WORD_NAME[st["labels"][k]],
                        fontsize=10, fontweight="bold",
                        color=WORD_COL[st["labels"][k]], va="bottom")

    ax = _tidy(fig.add_subplot(gs[0, 0]))
    for k, ons in enumerate(st["tone_onsets"]):
        w = int(st["labels"][k])
        for j, o in enumerate(ons):
            if o * dt > ts[-1]:
                continue
            ax.add_patch(plt.Rectangle((o * dt, WORDS[w][j] - 0.4),
                                       n_tone * dt, 0.8,
                                       facecolor=WORD_COL[w], edgecolor="none"))
    bounds(ax, label=True)
    ax.set_xlim(0, ts[-1]); ax.set_ylim(-0.9, N_CH + 1.0)
    ax.set_yticks(range(N_CH)); ax.set_yticklabels(range(N_CH), fontsize=7.5)
    ax.set_ylabel("input\nchannel", fontsize=10)
    ax.tick_params(labelbottom=False)

    ymax = max(Y[[r["unit"] for r in tab], :T].max(), 1e-9) * 1.15
    for r, row in enumerate(tab):
        ax = _tidy(fig.add_subplot(gs[1 + r, 0]))
        col = D_COL[min(row["depth"], 2)]
        ax.fill_between(ts, 0, Y[row["unit"], :T], color=col, lw=0, alpha=0.92)
        bounds(ax)
        ax.set_xlim(0, ts[-1]); ax.set_ylim(0, ymax); ax.set_yticks([])
        ax.set_ylabel(f"u{row['unit']}", fontsize=9, rotation=0, ha="right",
                      va="center", labelpad=8, color=col)
        if r < len(tab) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s)", fontsize=10)

    n2 = sum(r["depth"] >= 2 for r in tab)
    n1 = sum(r["depth"] == 1 for r in tab)
    _caption(fig, gs[rows, 0],
             "Top: the input, each token coloured by its word; dashed lines are word boundaries, which the model never sees.\n"
             f"Units are ordered by how much they represent and coloured the same way: dark green holds a whole word ({n2} units), "
             f"orange holds a single transition ({n1}), grey neither.\n"
             "The green units fire once per occurrence of their own word, at the moment it completes. Everything below is the raw response with the weights frozen.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def fig_dissection(l2, a1cfg, fname, word_idx=0):
    """One word, followed through every stage of the computation."""
    dt = a1cfg.dt
    word = WORDS[word_idx]
    st = build_stream([word], [1.0], 1, N_CH, dt, tone_dur=TONE_DUR,
                      intra_gap=GAP, inter_gap=0.350, seed=0, order=[0])
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=0)
    S, Y = record(l2, E, dt)
    ons = st["tone_onsets"][0]

    # the unit that holds this word, and the moment it peaks
    tab = [r for r in unit_table(l2) if r["word"] == word_idx and r["depth"] >= 2]
    u = tab[0]["unit"] if tab else int(np.argmax(Y.max(axis=1)))
    t0 = int(np.argmax(Y[u]))
    i_now = int(np.argmax(l2.M[u].sum(axis=(1, 2))))
    T = min(int(0.34 / dt), E.shape[1])
    ts = np.arange(T) * dt

    fig = plt.figure(figsize=(17, 11.4), constrained_layout=True)
    gs = fig.add_gridspec(5, 4, height_ratios=[0.95, 1.0, 1.25, 0.95, 0.42])
    fig.suptitle(f"One word, followed all the way through.  "
                 f"Unit {u} encodes {WORD_NAME[word_idx]} = "
                 f"{word[0]} then {word[1]} then {word[2]}",
                 fontsize=14.5, fontweight="bold")

    def mark(ax):
        ax.axvline(t0 * dt, color=INK, lw=1.5, ls="--", zorder=6)

    # ---- row 0: stimulus and layer 1 ----
    ax = _tidy(fig.add_subplot(gs[0, :2]))
    for k, ch in enumerate(word):
        ax.add_patch(plt.Rectangle((ons[k] * dt, ch - 0.4), st["n_tone"] * dt,
                                   0.8, facecolor=WORD_COL[word_idx],
                                   edgecolor="none"))
        ax.text((ons[k] + st["n_tone"] / 2) * dt, ch + 0.6, f"ch {ch}",
                ha="center", fontsize=9.5, fontweight="bold", color=INK)
    mark(ax)
    ax.set_xlim(0, ts[-1]); ax.set_ylim(min(word) - 1.1, max(word) + 1.4)
    ax.set_yticks(word); ax.set_ylabel("channel", fontsize=10)
    ax.tick_params(labelbottom=False)
    ax.set_title("1.  The stimulus", fontsize=11)

    ax = _tidy(fig.add_subplot(gs[0, 2:]))
    for k, ch in enumerate(word):
        ax.plot(ts, E[ch, :T], lw=1.8, label=f"ch {ch}")
    mark(ax)
    ax.set_xlim(0, ts[-1]); ax.set_ylabel("rate", fontsize=10)
    ax.legend(fontsize=8.5, ncol=3, loc="upper left")
    ax.tick_params(labelbottom=False)
    ax.set_title("2.  Layer 1 output E, the only thing layer 2 receives",
                 fontsize=11)

    # ---- row 1: the filterbank, one panel per token ----
    for k, ch in enumerate(word):
        ax = _tidy(fig.add_subplot(gs[1, k]))
        for m in range(l2.R):
            ax.plot(ts, S[ch, m, :T], lw=1.5,
                    color=plt.cm.viridis(m / max(l2.R - 1, 1)),
                    label=f"{l2.tau[m]*1e3:.0f}")
        mark(ax)
        ax.set_xlim(0, ts[-1])
        ax.set_title(f"3.  Filterbank on ch {ch}", fontsize=10.5)
        ax.set_xlabel("time (s)", fontsize=9)
        if k == 0:
            ax.set_ylabel("trace", fontsize=10)
            ax.legend(fontsize=7, ncol=2, title="tau (ms)", title_fontsize=7,
                      loc="upper left")

    ax = _tidy(fig.add_subplot(gs[1, 3]))
    for k, ch in enumerate(word[:2]):
        prof = S[ch, :, t0]
        ax.plot(np.arange(l2.R), prof / max(prof.max(), 1e-9), "o-", lw=2, ms=6,
                label=f"ch {ch}, {int((t0-ons[k])*dt*1e3)} ms old")
    ax.set_xticks(range(l2.R))
    ax.set_xticklabels([f"{t*1e3:.0f}" for t in l2.tau], fontsize=7.5, rotation=45)
    ax.set_xlabel("tau (ms)", fontsize=9); ax.set_ylabel("normalised", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("4.  Age read off the profile", fontsize=10.5)

    # ---- row 2: the four matrices at the marked instant ----
    panels = [
        ("5.  Filterbank state s", S[:, :, t0], "Greens"),
        (f"6.  D sliced at ch {i_now}\n(s gated by who fires now)",
         E[i_now, t0] * S[:, :, t0], "Purples"),
        (f"7.  Mask of unit {u}, same slice", l2.M[u][i_now], "RdPu"),
        ("8.  Their product, cell by cell",
         l2.M[u][i_now] * E[i_now, t0] * S[:, :, t0], "Oranges"),
    ]
    for c, (title, Mx, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[2, c])
        im = ax.imshow(Mx, aspect="auto", cmap=cmap, vmin=0)
        ax.set_xticks(range(l2.R))
        ax.set_xticklabels([f"{t*1e3:.0f}" for t in l2.tau], fontsize=7,
                           rotation=45)
        ax.set_yticks(range(N_CH)); ax.set_yticklabels(range(N_CH), fontsize=6.5)
        ax.set_xlabel("rate tau (ms)", fontsize=9)
        if c == 0:
            ax.set_ylabel("channel", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for k, ch in enumerate(word[:2]):
            ax.axhline(ch, color=INK, lw=0.5, alpha=0.35)

    # ---- row 3: the responses ----
    ax = _tidy(fig.add_subplot(gs[3, :]))
    for r in unit_table(l2):
        col = D_COL[min(r["depth"], 2)]
        ax.plot(ts, Y[r["unit"], :T], lw=2.2 if r["unit"] == u else 0.9,
                color=col, alpha=1.0 if r["unit"] == u else 0.45)
    mark(ax)
    ax.set_xlim(0, ts[-1]); ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("response", fontsize=10)
    ax.set_title(f"9.  Every unit's response.  Unit {u} is the bold trace; "
                 f"its peak is the dashed line, and its value is the sum of panel 8",
                 fontsize=11)

    tot = float((l2.M[u][i_now] * E[i_now, t0] * S[:, :, t0]).sum())
    _caption(fig, gs[4, :],
             f"Read left to right. The three tokens arrive (1) and layer 1 turns them into rates (2). Each channel drives a bank of filters (3); because the filters differ in speed, "
             f"a channel's profile across them says how long ago it fired (4).\n"
             f"At the marked instant the state of the whole bank is panel 5. Multiplying by who is firing right now gives D (6), which is the same picture gated by the current channel. "
             f"Panel 7 is the unit's stored mask, and panel 8 is\ntheir cell by cell product, whose entries sum to {tot:.1f}, exactly the height of the bold trace in panel 9. "
             f"Two cells dominate that sum, at two different rates: that is the word.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}   (unit {u}, peak at {t0*dt*1e3:.0f} ms)")


# ---------------------------------------------------------------------
def fig_all_masks(l2, fname):
    tab = unit_table(l2)
    n = len(tab)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(16.5, 2.5 * nrow + 1.5), constrained_layout=True)
    gs = fig.add_gridspec(nrow + 1, ncol,
                          height_ratios=[1.0] * nrow + [0.30 / max(nrow, 1) * 3])
    fig.suptitle("Every committed unit's mask, sliced at the channel it fires on",
                 fontsize=14.5, fontweight="bold")
    vmax = max(l2.M[[r["unit"] for r in tab]].max(), 1e-9)
    for k, r in enumerate(tab):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ax.imshow(l2.M[r["unit"]][r["now"]], aspect="auto", cmap="RdPu",
                  vmin=0, vmax=vmax)
        ax.set_xticks(range(l2.R))
        ax.set_xticklabels([f"{t*1e3:.0f}" for t in l2.tau], fontsize=6,
                           rotation=45)
        ax.set_yticks(range(0, N_CH, 2))
        ax.set_yticklabels(range(0, N_CH, 2), fontsize=6)
        col = D_COL[min(r["depth"], 2)]
        lab = (f"u{r['unit']}  ch {r['now']}  depth {r['depth']}"
               + (f"  {WORD_NAME[r['word']]}" if r["depth"] >= 2 and
                  r["word"] is not None else ""))
        ax.set_title(lab, fontsize=8.5, color=col, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.6)
    _caption(fig, gs[nrow, :],
             "Predecessor channel down the side, rate along the bottom, in every panel. "
             "Dark green units hold a whole word, orange a single transition, grey neither.\n"
             "A word unit shows two bright cells stepping down and to the right: the immediate predecessor at a fast rate, the one before it at a slower rate.")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}   ({n} units)")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CH, **LAYER1)
    cfg = MRConfig()
    print("[ layer2_multirate.internals ] training, then opening it up")
    l2, _, _ = expose(cfg, a1cfg, seed=0)
    tab = unit_table(l2)
    print(f"  {len(tab)} committed; depths: "
          f"{ {d: sum(r['depth'] == d for r in tab) for d in sorted({r['depth'] for r in tab})} }")
    st, E, S, Y = readout(l2, a1cfg)
    fig_tape(l2, st, E, Y, a1cfg, str(out / "mr_tape.png"))
    fig_dissection(l2, a1cfg, str(out / "mr_dissection.png"))
    fig_all_masks(l2, str(out / "mr_all_masks.png"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
