"""
layer2_multirate.sweep
======================

Scaling tests for the multi rate layer.

The measure used throughout is **span depth**: how many tokens back a unit's
mask actually represents, in the right order, at strictly slower rates.

    depth 1   the immediate predecessor only          (a transition)
    depth 2   two tokens back                         (a trigram)
    depth L-1 every earlier token of an L token word  (the whole word)

A unit counts a predecessor only if its mask holds a substantial entry for that
channel (at least ``THRESH`` of the row's peak) **and** at a strictly slower
rate than the predecessor before it.  The second condition is what distinguishes
genuine age ordering from a unit that merely has weight scattered around.

Five sweeps:

    length     words of 2 to 6 tokens; how deep can it go
    rates      how many filters are needed
    span       how slow the slowest filter must be
    tempo      the whole stream sped up and slowed down, filterbank fixed
    capacity   units available against units used

Run
---
    python -m layer2_multirate.sweep
"""
from __future__ import annotations

import dataclasses as dc
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
from layer2_syllable.run_ab_ba import (GRID, INK, LAYER1, MUTED, _caption,
                                       _tidy, layer1_rates)
from layer2_syllable.stimulus import build_stream
from layer2_syllable.saffran.run_saffran import word_order

THRESH = 0.25          # fraction of a mask row's peak needed to count a predecessor
N_STREAM = 300         # words of exposure per run (converged by ~150)
TONE, GAP = 0.050, 0.030

C_OK, C_MID, C_BAD = "#1f6b4a", "#d98f2b", "#a83f28"
C_ACC = "#6b5b95"


def make_words(n_words=4, length=3):
    """n_words disjoint words of the given length, over n_words*length channels."""
    return [tuple(range(w * length, (w + 1) * length)) for w in range(n_words)]


# ---------------------------------------------------------------------
def run_one(words, cfg, a1cfg, tone=TONE, gap=GAP, n_stream=N_STREAM, seed=0):
    n_ch = sum(len(w) for w in words)
    rng = np.random.default_rng(1000 + seed)
    order = word_order(len(words), n_stream, rng)
    st = build_stream(words, [1.0 / len(words)] * len(words), n_stream, n_ch,
                      a1cfg.dt, tone_dur=tone, intra_gap=gap, inter_gap=gap,
                      seed=seed, order=order)
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=seed)
    l2 = Layer2MR(n_ch, cfg)
    l2.run(E, a1cfg.dt, learn=True)
    return l2


def span_depth(l2, unit, words, thresh=THRESH):
    """How many tokens back this unit represents, in order, at slower rates."""
    M = l2.M[unit]
    i = int(np.argmax(M.sum(axis=(1, 2))))            # channel it fires on
    row = M[i]                                         # (N, R)
    peak = row.max()
    if peak <= 0:
        return 0, None
    # which word, and where in it
    for w in words:
        if i in w:
            pos = w.index(i)
            break
    else:
        return 0, None

    depth, prev_tau = 0, -1.0
    for d in range(1, pos + 1):
        j = w[pos - d]
        m = int(np.argmax(row[j]))
        if row[j, m] >= thresh * peak and l2.tau[m] > prev_tau:
            depth += 1
            prev_tau = l2.tau[m]
        else:
            break
    return depth, w


def summarise(l2, words):
    """Depth statistics over the committed units."""
    comm = np.flatnonzero(l2.committed)
    depths, full_words = [], set()
    L = len(words[0])
    for u in comm:
        d, w = span_depth(l2, u, words)
        depths.append(d)
        if w is not None and d >= L - 1 and L > 1:
            full_words.add(words.index(w))
    depths = np.array(depths) if len(depths) else np.array([0])
    return dict(n_committed=len(comm), max_depth=int(depths.max()),
                mean_depth=float(depths.mean()),
                n_full=len(full_words), coverage=len(full_words) / len(words),
                depths=depths)


def repeat(words, cfg, a1cfg, seeds=(0, 1, 2), **kw):
    out = [summarise(run_one(words, dc.replace(cfg, seed=s), a1cfg, seed=s, **kw),
                     words) for s in seeds]
    keys = ["n_committed", "max_depth", "mean_depth", "n_full", "coverage"]
    return {k: (float(np.mean([o[k] for o in out])),
                float(np.std([o[k] for o in out]))) for k in keys}


# ---------------------------------------------------------------------
def main(argv=None):
    out_dir = Path(__file__).resolve().parent
    base = MRConfig()
    print("[ layer2_multirate.sweep ]  span depth = how many tokens back a unit "
          "represents, in order, at slower rates\n")

    res = {}

    # ---- A. word length ----
    print("A. word length (4 words, filterbank fixed at 30-500 ms, 24 units)")
    lengths = [2, 3, 4, 5, 6]
    A = []
    for L in lengths:
        words = make_words(4, L)
        a1 = selective_inh(N=4 * L, **LAYER1)
        r = repeat(words, base, a1)
        A.append(r)
        print(f"   L={L}  max depth {r['max_depth'][0]:.1f} of {L-1}   "
              f"mean {r['mean_depth'][0]:.2f}   full words {r['n_full'][0]:.1f}/4   "
              f"committed {r['n_committed'][0]:.0f}")
    res["length"] = (lengths, A)

    # ---- B. number of rates ----
    print("\nB. number of rates (4 token words, span fixed 30-500 ms)")
    Rs = [1, 2, 3, 4, 6, 8, 12]
    words4 = make_words(4, 4); a1_4 = selective_inh(N=16, **LAYER1)
    B = []
    for R in Rs:
        cfg = dc.replace(base, rates=tuple(np.round(np.geomspace(0.030, 0.500, R), 4)))
        r = repeat(words4, cfg, a1_4)
        B.append(r)
        print(f"   R={R:2d}  max depth {r['max_depth'][0]:.1f} of 3   "
              f"mean {r['mean_depth'][0]:.2f}   full words {r['n_full'][0]:.1f}/4")
    res["rates"] = (Rs, B)

    # ---- C. slowest rate ----
    print("\nC. slowest rate (4 token words span 240 ms from first to last onset)")
    his = [0.080, 0.150, 0.250, 0.400, 0.600, 0.900]
    C = []
    for hi in his:
        cfg = dc.replace(base, rates=tuple(np.round(np.geomspace(0.030, hi, 6), 4)))
        r = repeat(words4, cfg, a1_4)
        C.append(r)
        print(f"   slowest {hi*1e3:4.0f} ms  max depth {r['max_depth'][0]:.1f} of 3   "
              f"mean {r['mean_depth'][0]:.2f}   full words {r['n_full'][0]:.1f}/4")
    res["span"] = (his, C)

    # ---- D. tempo ----
    print("\nD. tempo (3 token words, filterbank NOT rescaled)")
    # deliberately taken far past the design point in both directions, to find
    # where the fixed bank actually breaks rather than confirming it works
    ks = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
    words3 = make_words(4, 3); a1_3 = selective_inh(N=12, **LAYER1)
    D = []
    for k in ks:
        r = repeat(words3, base, a1_3, tone=TONE * k, gap=GAP * k)
        D.append(r)
        print(f"   tempo x{k:<4.2f} token every {80*k:5.0f} ms   "
              f"max depth {r['max_depth'][0]:.1f} of 2   "
              f"full words {r['n_full'][0]:.1f}/4")
    res["tempo"] = (ks, D)

    # ---- E. capacity ----
    print("\nE. capacity (3 token words)")
    Ks = [6, 8, 12, 16, 24, 32, 48]
    E = []
    for K in Ks:
        r = repeat(words3, dc.replace(base, n_units=K), a1_3)
        E.append(r)
        print(f"   {K:2d} units  committed {r['n_committed'][0]:4.1f}   "
              f"full words {r['n_full'][0]:.1f}/4   "
              f"mean depth {r['mean_depth'][0]:.2f}")
    res["capacity"] = (Ks, E)

    np.save(out_dir / "sweep_results.npy", res, allow_pickle=True)
    fig_scaling(res, str(out_dir / "mr_scaling.png"))
    fig_examples(base, str(out_dir / "mr_depth_examples.png"))
    print("\nDone.")
    return 0


# ---------------------------------------------------------------------
def _errline(ax, x, rows, key, color, label=None, marker="o"):
    m = np.array([r[key][0] for r in rows]); e = np.array([r[key][1] for r in rows])
    ax.plot(x, m, marker + "-", color=color, lw=2, ms=6, label=label)
    ax.fill_between(x, m - e, m + e, color=color, alpha=0.18, lw=0)


def fig_scaling(res, fname):
    fig = plt.figure(figsize=(16, 9.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.52])
    fig.suptitle("How the multi rate layer scales.  Depth = how many tokens back "
                 "a unit represents, in order, at slower rates",
                 fontsize=14, fontweight="bold")

    # A. word length
    L, A = res["length"]
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    ax.plot(L, [l - 1 for l in L], "--", color=MUTED, lw=1.5,
            label="a whole word")
    _errline(ax, L, A, "max_depth", C_OK, "deepest unit found")
    _errline(ax, L, A, "mean_depth", C_ACC, "average over units", marker="s")
    ax.set_xlabel("tokens per word"); ax.set_ylabel("span depth")
    ax.set_xticks(L); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("a.  Longer words", fontsize=11)

    ax = _tidy(fig.add_subplot(gs[0, 1]))
    _errline(ax, L, A, "coverage", C_OK)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("tokens per word"); ax.set_ylabel("fraction of words with a full unit")
    ax.set_xticks(L)
    ax.set_title("b.  Complete words covered", fontsize=11)

    # B. number of rates
    R, Bv = res["rates"]
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.axhline(3, color=MUTED, ls="--", lw=1.4)
    ax.text(R[-1], 3.05, "a whole 4 token word ", fontsize=8.4, color=MUTED,
            ha="right", va="bottom")
    _errline(ax, R, Bv, "max_depth", C_OK, "deepest unit")
    _errline(ax, R, Bv, "mean_depth", C_ACC, "average", marker="s")
    ax.set_xscale("log"); ax.set_xticks(R)
    ax.set_xticklabels([str(r) for r in R])
    ax.set_xlabel("number of rates in the bank"); ax.set_ylabel("span depth")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("c.  How many filters are needed", fontsize=11)

    # C. slowest rate
    hi, Cv = res["span"]
    ax = _tidy(fig.add_subplot(gs[1, 0]))
    _errline(ax, np.array(hi) * 1e3, Cv, "max_depth", C_OK, "deepest unit")
    ax.axvline(240, color=C_BAD, ls="--", lw=1.5)
    ax.text(250, 0.15, "the word itself\nlasts 240 ms", fontsize=8.6,
            color=C_BAD, va="bottom")
    ax.axhline(3, color=MUTED, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("slowest rate in the bank (ms)"); ax.set_ylabel("span depth")
    ax.set_title("d.  The bank must outlast the word", fontsize=11)

    # D. tempo
    k, Dv = res["tempo"]
    ax = _tidy(fig.add_subplot(gs[1, 1]))
    _errline(ax, np.array(k) * 80, Dv, "max_depth", C_MID, "deepest unit")
    _errline(ax, np.array(k) * 80, Dv, "coverage", C_OK, "words fully covered",
             marker="s")
    ax.axvline(80, color=INK, ls=":", lw=1.3)
    ax.text(0.5, 0.06, "the tempo the bank\nwas designed for",
            transform=ax.transAxes, fontsize=8.4, color=INK, ha="center")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 2.35)
    ax.set_xlabel("one token every ... ms"); ax.set_ylabel("depth / coverage")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("e.  Changing the tempo, bank left fixed", fontsize=11)

    # E. capacity
    K, Ev = res["capacity"]
    ax = _tidy(fig.add_subplot(gs[1, 2]))
    _errline(ax, K, Ev, "n_committed", C_ACC, "units committed")
    ax.plot(K, K, ":", color=MUTED, lw=1.2, label="everything available")
    _errline(ax, K, Ev, "n_full", C_OK, "units holding a whole word", marker="s")
    ax.axhline(4, color=MUTED, ls="--", lw=1.2)
    ax.text(K[-1], 4.2, "4 words in the stream ", fontsize=8.4, color=MUTED,
            ha="right", va="bottom")
    ax.set_xlabel("units available"); ax.set_ylabel("units")
    ax.legend(fontsize=8.4, loc="upper left")
    ax.set_title("f.  Capacity", fontsize=11)

    _caption(fig, gs[2, :],
             "a, b.  Depth tracks word length exactly up to four tokens, with all four words covered. At five it reaches depth four but only sometimes, and at six it breaks. "
             "The ceiling is resolution, not range: the\nfirst two tokens of a six token word are 400 and 320 ms old and their profiles across this bank are no longer distinct enough to separate.\n"
             "c.  One rate can never exceed depth one, which is exactly the single rate result reproduced. Note the dip at three rates: what matters is whether the bank happens to place "
             "filters near the token spacing, so\nmore filters is not automatically better, and placement is not a free parameter to ignore.\n"
             "d.  There is a window, not a threshold. Too fast and the older tokens are gone before the last one arrives; too slow and the filters become too alike to tell ages apart. "
             "Success begins right around the word's\nown 240 ms duration.\n"
             "e.  I expected a bank fixed in absolute time to be tempo specific. It is not: full depth and full coverage hold from 20 to 320 ms per token, a sixteen fold range, because a bank "
             "spanning 30 to 500 ms contains usable\nrates for any of those tempi and the competition simply selects them. It breaks only at 480 ms per token, where the word itself outlasts the slowest filter.\n"
             "f.  The count is self limiting: with 48 units available only about 15 commit, which is roughly the number of distinct transitions the stream contains. "
             "Units holding a whole word saturate at four, the number of words.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


def fig_examples(base, fname):
    """One learned mask per word length, so the staircase can be seen growing."""
    fig = plt.figure(figsize=(16, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.26])
    fig.suptitle("The same mechanism at four word lengths: the staircase gets one step longer each time",
                 fontsize=14, fontweight="bold")

    for c, L in enumerate([2, 3, 4, 5]):
        words = make_words(4, L)
        a1 = selective_inh(N=4 * L, **LAYER1)
        l2 = run_one(words, base, a1, seed=0)
        # pick the deepest unit
        best, bd, bw = None, -1, None
        for u in np.flatnonzero(l2.committed):
            d, w = span_depth(l2, u, words)
            if d > bd:
                best, bd, bw = u, d, w
        ax = fig.add_subplot(gs[0, c])
        M = l2.M[best]
        i = int(np.argmax(M.sum(axis=(1, 2))))
        row = M[i]
        im = ax.imshow(row, aspect="auto", cmap="RdPu", vmin=0)
        ax.set_xticks(range(l2.R))
        ax.set_xticklabels([f"{t*1e3:.0f}" for t in l2.tau], fontsize=7.5,
                           rotation=45)
        ax.set_yticks(range(4 * L))
        ax.set_yticklabels(range(4 * L), fontsize=6.5)
        ax.set_xlabel("rate tau (ms)", fontsize=9)
        if c == 0:
            ax.set_ylabel("predecessor channel", fontsize=9)
        # outline the staircase actually used
        if bw is not None:
            pos = bw.index(i)
            prev = -1.0
            for d in range(1, min(bd, pos) + 1):
                j = bw[pos - d]
                m = int(np.argmax(row[j]))
                ax.add_patch(plt.Rectangle((m - 0.5, j - 0.5), 1, 1, fill=False,
                                           edgecolor=C_OK, lw=2.0))
                prev = l2.tau[m]
        ax.set_title(f"{L} token words   depth {bd} of {L-1}\n"
                     f"unit {best} fires on ch {i}", fontsize=10,
                     color=C_OK if bd >= L - 1 else C_MID, fontweight="bold")

    _caption(fig, gs[1, :],
             "Each panel is the deepest unit found for that word length, sliced at the channel it fires on. Outlined cells are the predecessors it actually uses. "
             "Reading right and up, each step back in the word\nsits at a slower rate, which is the age ordering. The staircase is the word, and it lengthens with the word "
             "until the bank runs out of range.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


if __name__ == "__main__":
    raise SystemExit(main())
