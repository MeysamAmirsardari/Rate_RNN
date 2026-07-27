"""
layer2_multirate.figures
========================

Two figures for the multi rate layer:

mr_mechanism.png   how a bank of rates encodes how long ago a token fired, and
                   why that lets one mask hold a whole word.
mr_learned.png     what the units actually learned, and the comparison with the
                   single rate version on the same paradigm.

Run
---
    python -m layer2_multirate.figures
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
from layer2_multirate.run_saffran import analyse, expose, test_items
from layer2_syllable.run_ab_ba import (GRID, INK, LAYER1, MUTED, _caption,
                                       _tidy, layer1_rates)
from layer2_syllable.stimulus import build_stream
from layer2_syllable.saffran.run_saffran import GAP, N_CH, TONE_DUR, WORDS, WORD_NAME

C_TOK = ["#c1553b", "#3d6fa8", "#2f7d5d"]     # 1st, 2nd, 3rd token of a word
C_ACC = "#6b5b95"
C_OK = "#1f6b4a"


# ---------------------------------------------------------------------
def fig_mechanism(cfg, a1cfg, fname):
    """How the filterbank tells 'just now' from 'a moment ago'."""
    dt = a1cfg.dt
    word = WORDS[0]                                     # channels 0, 1, 2
    st = build_stream([word], [1.0], 1, N_CH, dt, tone_dur=TONE_DUR,
                      intra_gap=GAP, inter_gap=0.400, seed=0, order=[0])
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=0)
    l2 = Layer2MR(N_CH, cfg)
    T = E.shape[1]
    S = np.zeros((N_CH, l2.R, T))
    for t in range(T):
        l2.step(E[:, t], dt, learn=False)
        S[:, :, t] = l2.s
    ts = np.arange(T) * dt
    ons = st["tone_onsets"][0]
    t_read = ons[2] + int(round(0.035 / dt))            # just after the 3rd tone starts

    fig = plt.figure(figsize=(15.5, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.15, 0.30])
    fig.suptitle("A bank of rates encodes how long ago each token fired",
                 fontsize=14, fontweight="bold")

    # (a) the word
    ax = _tidy(fig.add_subplot(gs[0, :]))
    for k, ch in enumerate(word):
        ax.add_patch(plt.Rectangle((ons[k] * dt, ch - 0.4),
                                   st["n_tone"] * dt, 0.8,
                                   facecolor=C_TOK[k], edgecolor="none"))
        ax.text((ons[k] + st["n_tone"] / 2) * dt, ch + 0.75, f"ch {ch}",
                ha="center", fontsize=10, fontweight="bold", color=C_TOK[k])
    ax.axvline(t_read * dt, color=INK, lw=1.4, ls="--")
    ax.text(t_read * dt + 0.008, 2.4, "read here", fontsize=10,
            fontweight="bold", color=INK)
    ax.set_xlim(0, 0.35); ax.set_ylim(-0.8, 3.2)
    ax.set_yticks([0, 1, 2]); ax.set_ylabel("channel", fontsize=10)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title("a.  One word: three tokens, 80 ms apart", fontsize=11)

    # (b) traces at a fast and a slow rate
    for col, m in enumerate([0, len(cfg.rates) // 2]):
        ax = _tidy(fig.add_subplot(gs[1, col]))
        for k, ch in enumerate(word):
            ax.plot(ts, S[ch, m, :], color=C_TOK[k], lw=2, label=f"ch {ch}")
        ax.axvline(t_read * dt, color=INK, lw=1.4, ls="--")
        ax.set_xlim(0, 0.35)
        ax.set_xlabel("time (s)", fontsize=10)
        ax.set_ylabel("trace", fontsize=10)
        ax.set_title(f"{'bc'[col]}.  Rate tau = {cfg.rates[m]*1e3:.0f} ms",
                     fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")

    # (d) the age signature: trace profile across rates at the read moment
    ax = _tidy(fig.add_subplot(gs[1, 2]))
    x = np.arange(l2.R)
    for k, ch in enumerate(word[:2]):                   # the two predecessors
        prof = S[ch, :, t_read]
        ax.plot(x, prof / prof.max(), "o-", color=C_TOK[k], lw=2.2, ms=7,
                label=f"ch {ch}, fired {(t_read-ons[k])*dt*1e3:.0f} ms ago")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t*1e3:.0f}" for t in cfg.rates], fontsize=9)
    ax.set_xlabel("rate tau (ms)", fontsize=10)
    ax.set_ylabel("trace, normalised", fontsize=10)
    ax.set_title("d.  The age signature", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")

    _caption(fig, gs[2, :],
             "At the moment the third token fires, both earlier tokens are still present, so a single rate cannot tell them apart. "
             "Across rates they differ: panel d shows the older token's trace is\n"
             "weighted toward the slow end and the recent one toward the fast end. That profile is what tells the layer how long ago "
             "something happened, and it is why one mask can now hold a whole word.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def fig_learned(l2, an, te, fname):
    """The masks of the word spanning units, and the comparison."""
    span = [r for r in an["rows"] if r["spans"]]
    fig = plt.figure(figsize=(15.5, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 0.32])
    fig.suptitle("What the units learned: four masks, each holding a whole word",
                 fontsize=14, fontweight="bold")

    vmax = max(l2.M[[r["unit"] for r in span]].max(), 1e-9) if span else 1.0
    for c, r in enumerate(span[:4]):
        ax = fig.add_subplot(gs[0, c])
        row = l2.M[r["unit"]][r["now"]]                # (N, R)
        im = ax.imshow(row, aspect="auto", cmap="RdPu", vmin=0, vmax=vmax)
        ax.set_xticks(range(l2.R))
        ax.set_xticklabels([f"{t*1e3:.0f}" for t in l2.tau], fontsize=8,
                           rotation=45)
        ax.set_yticks(range(N_CH))
        ax.set_yticklabels(range(N_CH), fontsize=7)
        ax.set_xlabel("rate tau (ms)", fontsize=9)
        if c == 0:
            ax.set_ylabel("predecessor channel", fontsize=9)
        w = WORDS[r["word"]]
        ax.set_title(f"unit {r['unit']}:  fires on ch {r['now']}\n"
                     f"{w[0]} then {w[1]} then {w[2]}   ({WORD_NAME[r['word']]})",
                     fontsize=10, color=C_OK, fontweight="bold")
        # mark the two entries it uses
        for (jj, tt, lab) in ((r["p1"], r["tau1"], "recent"),
                              (r["p2"], r["tau2"], "older")):
            mi = int(np.argmin(np.abs(l2.tau - tt)))
            ax.add_patch(plt.Rectangle((mi - 0.5, jj - 0.5), 1, 1, fill=False,
                                       edgecolor=C_OK, lw=2.0))
            ax.text(mi + 0.7, jj, lab, fontsize=8, color=C_OK,
                    va="center", fontweight="bold")

    # (e) single rate against multi rate
    ax = _tidy(fig.add_subplot(gs[1, 0]))
    ax.bar([0, 1], [0, an["n_span"]], color=[GRID, C_OK], width=0.5)
    for x, v in ((0, 0), (1, an["n_span"])):
        ax.text(x, v + 0.12, str(v), ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["single rate\n(8 units)", "filterbank\n(24 units)"],
                       fontsize=9)
    ax.set_ylim(0, max(an["n_span"], 1) + 1.0); ax.set_yticks([])
    ax.set_title("e.  Units spanning a whole word", fontsize=10.5)

    ax = _tidy(fig.add_subplot(gs[1, 1]))
    ax.bar([0, 1], [0.86, te["auc"]], color=[GRID, C_ACC], width=0.5)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    for x, v in ((0, 0.86), (1, te["auc"])):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=12,
                fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["single rate", "filterbank"],
                                              fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_ylabel("ROC", fontsize=9)
    ax.set_title("f.  Word vs part word", fontsize=10.5)

    # (g) which rate each spanning unit chose for each predecessor
    ax = _tidy(fig.add_subplot(gs[1, 2:]))
    xs = np.arange(len(span))
    w = 0.35
    ax.bar(xs - w / 2, [r["tau1"] * 1e3 for r in span], w, color="#3d6fa8",
           label="immediate predecessor")
    ax.bar(xs + w / 2, [r["tau2"] * 1e3 for r in span], w, color="#c1553b",
           label="the token before that")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"unit {r['unit']}\n{WORD_NAME[r['word']]}" for r in span],
                       fontsize=8.5)
    ax.set_ylabel("rate tau chosen (ms)", fontsize=9)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("g.  Every spanning unit put the older token on a slower rate",
                 fontsize=10.5)

    _caption(fig, gs[2, :],
             "Each panel above is one unit's mask, sliced at the channel it fires on: predecessor channel down the side, rate along the bottom. "
             "A single rate version could only ever light up one column,\n"
             "so it learned transitions and nothing represented a word. Here each mask lights up two cells at two different rates, which is the "
             "word itself: 'this token now, that one recently, that other one a while ago'.\n"
             "Nothing instructed the units to use two rates, or to put the older token on the slower one. That is what the competition settled on.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CH, **LAYER1)
    cfg = MRConfig()
    print("[ layer2_multirate.figures ]")
    fig_mechanism(cfg, a1cfg, str(out / "mr_mechanism.png"))
    l2, st, tr = expose(cfg, a1cfg, seed=0)
    an = analyse(l2)
    te = test_items(l2, a1cfg, seed=0)
    print(f"  {an['n_units']} committed, {an['n_span']} span a word, "
          f"ROC {te['auc']:.2f}")
    fig_learned(l2, an, te, str(out / "mr_learned.png"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
