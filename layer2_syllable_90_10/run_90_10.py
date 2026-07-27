"""
layer2_syllable_90_10.run_90_10
===============================

The same two layer model as ``layer2_syllable``, but the stream is now
**ninety percent AB and ten percent BA** instead of fifty fifty.  The model
code is imported unchanged; only the stream statistics differ.

Why this run exists
-------------------
On the fifty fifty stream the controls showed that layer 1 contributes nothing
that layer 2 needs: the raw stimulus, layer 1 with its recurrent weights held
at zero, and the full layer 1 all gave the same answer.  The reason is that
with both orders equally frequent the Hebbian rule in layer 1 sees A before B
exactly as often as B before A, so its recurrent weights come out symmetric and
carry no order information at all.

Making one order rare breaks that symmetry, so this is the condition in which
layer 1's learned prediction can finally matter.  It also asks a second and
more interesting question.  Layer 2 prunes units that rarely win, so a chunk
seen on only ten percent of trials may not earn a unit of its own.  If it does
not, the rare chunk is represented not as an object with its own detector, but
only as the **failure of the standard's detector to fire**.  That distinction
is directly testable against the decodability measured in the recordings.

Metric
------
Plain accuracy is meaningless here: always answering "AB" scores ninety
percent.  Everything below is **balanced accuracy**, the mean of the two per
class recalls, whose chance level is fifty percent whatever the class
proportions.  Where only one unit survives, the winner take all readout is
uninformative by construction, so the area under the ROC of the surviving
unit's response is reported as well, which asks whether the rare chunk is
legible in the magnitude of the standard's detector.

Run
---
    python -m layer2_syllable_90_10.run_90_10
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
from layer2_syllable.config import L2Config
from layer2_syllable.run_ab_ba import (C_AB, C_ACC, C_BA, GRID, LAYER1, MUTED,
                                       N_CHANNELS, _caption, _ccol,
                                       _read_mask, _tidy, auc, selectivity,
                                       train_and_test)
from layer2_syllable.visuals import fig_tape

P_STD = 0.90                 # probability of the standard chunk, AB
N_TRAIN, N_TEST = 500, 400   # more test chunks so the rare class is still sampled


# ---------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------
def rare_stats(res):
    """Balanced accuracy, and how legible the rare chunk is in each unit."""
    l2, labels = res["l2"], res["test"]["labels"]
    comm = np.flatnonzero(l2.committed)
    out = dict(n_units=int(l2.n_components), bal=float(res["acc"]), comm=comm)
    if comm.size:
        # Canonical direction: a unit that is SUPPRESSED by the rare chunk
        # separates it just as well as one that is driven by it, and raw areas
        # of 0.0 and 1.0 are equally perfect.  Folding to [0.5, 1] keeps the
        # axis readable as "how separable", never as "which way round".
        raw = [auc(res["resp"][labels == 0, u], res["resp"][labels == 1, u])
               for u in comm]
        aucs = [max(a, 1.0 - a) for a in raw]
        best = int(np.argmax(aucs))
        out.update(aucs=aucs, raw_aucs=raw, best_auc=float(aucs[best]),
                   best_unit=int(comm[best]))
    else:
        out.update(aucs=[], best_auc=0.5, best_unit=-1)
    return out


def asym(W):
    """Order asymmetry of the layer 1 recurrent weights, in [-1, 1]."""
    if W is None:
        return 0.0
    a, b = float(W[1, 0]), float(W[0, 1])
    return (a - b) / (a + b) if (a + b) > 0 else 0.0


# ---------------------------------------------------------------------
#  Figure: the vocabulary that forms
# ---------------------------------------------------------------------
def fig_vocabulary(res, res_50, fname):
    l2, labels = res["l2"], res["test"]["labels"]
    K = l2.cfg.n_units
    comm = np.flatnonzero(l2.committed)
    order = {u: i for i, u in enumerate(comm)}
    st = rare_stats(res)

    fig = plt.figure(figsize=(15.5, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.10, 0.70, 0.70, 0.26])
    fig.suptitle(f"What vocabulary forms when one order is rare "
                 f"({P_STD:.0%} AB, {1 - P_STD:.0%} BA)",
                 fontsize=13.5, fontweight="bold")

    # (a) responses by chunk type
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    rng = np.random.default_rng(0)
    for i, u in enumerate(comm):
        for lab, col in ((0, C_AB), (1, C_BA)):
            v = res["resp"][labels == lab, u]
            x = i * 2 + (0.0 if lab == 0 else 0.8)
            ax.scatter(x + rng.normal(0, 0.09, v.size), v, s=10, color=col,
                       alpha=0.45, lw=0)
            if v.size:
                ax.hlines(v.mean(), x - 0.3, x + 0.3, color=col, lw=2.6)
    ax.set_xticks([i * 2 + 0.4 for i in range(len(comm))])
    ax.set_xticklabels([f"unit {u}" for u in comm])
    ax.set_ylabel("peak response in chunk")
    ax.set_ylim(bottom=-0.5)
    ax.set_title("a.  Response by chunk type", fontsize=10.5)
    h = [plt.Line2D([], [], color=C_AB, lw=3, label="AB, the standard"),
         plt.Line2D([], [], color=C_BA, lw=3, label="BA, the rare chunk")]
    ax.legend(handles=h, fontsize=8.4, loc="upper center")

    # (b) how many units survive, against the balanced case
    ax = _tidy(fig.add_subplot(gs[0, 1]))
    vals = [2, res_50["l2"].n_components, l2.n_components]
    ax.bar([0, 1, 2], vals, color=[GRID, "#a3aebd", C_ACC], width=0.55)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.06, str(v), ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["in the\nstream", "kept at\n50/50",
                        f"kept at\n{P_STD:.0%}/{1 - P_STD:.0%}"], fontsize=8.6)
    ax.set_ylim(0, max(vals) + 1.0); ax.set_yticks([])
    ax.set_title("b.  Does the rare chunk\nearn a unit", fontsize=10.5)

    # (c) two readouts
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.bar([0, 1], [st["bal"], st["best_auc"]], color=[C_ACC, C_AB], width=0.5)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(-0.45, 0.52, "chance", fontsize=8.4, color=MUTED, va="bottom")
    for x, v in ((0, st["bal"]), (1, st["best_auc"])):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["named\n(balanced acc.)", "detected\n(ROC area)"],
                       fontsize=8.6)
    ax.set_ylim(0, 1.2)
    ax.set_title("c.  Naming it versus\ndetecting it", fontsize=10.5)

    # (d) selectivity
    ax = _tidy(fig.add_subplot(gs[0, 3]))
    si = [selectivity(res["resp"], labels, u) for u in comm]
    ax.bar(np.arange(len(comm)), si,
           color=[C_AB if v > 0 else C_BA for v in si], width=0.5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xticks(np.arange(len(comm)))
    ax.set_xticklabels([f"unit {u}" for u in comm])
    ax.set_ylabel("prefers BA        prefers AB")
    for i, v in enumerate(si):
        ax.text(i, v + (0.08 if v >= 0 else -0.19), f"{v:+.2f}", ha="center",
                fontsize=9.5, fontweight="bold")
    ax.set_title("d.  Selectivity index", fontsize=10.5)

    # (e) all eight masks, two plain rows
    vmax = max(l2.M.max(), 1e-9)
    per_row = K // 2
    for u in range(K):
        r, c = u // per_row, u % per_row
        ax = fig.add_subplot(gs[1 + r, c])
        # aspect auto: a fixed 1:1 aspect lets constrained_layout collapse the
        # whole grid around these small images
        ax.imshow(l2.M[u], cmap="RdPu", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["A", "B"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["A", "B"], fontsize=8)
        if l2.committed[u]:
            ax.set_title(f"unit {u}: {_read_mask(l2.M[u])}", fontsize=9.5,
                         color=_ccol(order[u]), fontweight="bold")
            for sp in ax.spines.values():
                sp.set_color(_ccol(order[u])); sp.set_linewidth(1.8)
        else:
            ax.set_title(f"unit {u}: silent", fontsize=9, color=MUTED)
        if c == 0:
            ax.set_ylabel("firing now", fontsize=8)
        if r == 1:
            ax.set_xlabel("fired recently", fontsize=8)

    _caption(fig, gs[3, :],
             "e.  The learned masks. Weight at (row B, column A) means the unit answers when B follows A, which is the chunk AB.\n"
             "When the rare chunk fails to earn a unit, panel c is the one to read. Balanced accuracy asks whether a winner take all\n"
             "readout can name the chunk; the area under the ROC asks the weaker question of whether the rare chunk is still legible\n"
             "in how strongly the standard's own detector responds.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
#  Figure: probability sweep and the layer 1 test
# ---------------------------------------------------------------------
def fig_controls(ps, sweep, modes, mode_bal, mode_n, mode_auc, W_by_p, fname):
    fig = plt.figure(figsize=(15.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.30])
    fig.suptitle("How the vocabulary depends on how rare the chunk is, "
                 "and whether layer 1 matters now",
                 fontsize=13.5, fontweight="bold")
    px = np.array(ps) * 100

    # (a) units committed against standard probability
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    ax.plot(px, [s["n_units"] for s in sweep], "s-", color=C_ACC, lw=2, ms=6)
    ax.axhline(2, color=MUTED, ls=":", lw=1.2)
    ax.set_ylim(0.7, 2.45)
    ax.text(px[0], 2.09, "chunk types in the stream", fontsize=8.2,
            color=MUTED, ha="left", va="bottom")
    ax.set_xlabel("probability of the standard (%)")
    ax.set_ylabel("units committed")
    ax.set_yticks([1, 2])
    ax.set_title("a.  Size of the learned vocabulary", fontsize=10.5)

    # (b) the two readouts against standard probability
    ax = _tidy(fig.add_subplot(gs[0, 1]))
    ax.plot(px, [s["best_auc"] for s in sweep], "^--", color=C_AB, lw=2.4,
            ms=7, label="detected (area under ROC)")
    ax.plot(px, [s["bal"] for s in sweep], "o-", color=C_ACC, lw=2.4, ms=7,
            label="named (balanced accuracy)")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(px[0], 0.52, "chance", fontsize=8.2, color=MUTED, va="bottom")
    ax.set_xlabel("probability of the standard (%)")
    ax.set_ylabel("score")
    ax.set_ylim(0.35, 1.12)
    ax.legend(fontsize=8.4, loc="lower left")
    ax.set_title("b.  Detecting it versus naming it", fontsize=10.5)

    # (c) layer 1 asymmetry against standard probability
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.plot(px, [asym(W) for W in W_by_p], "o-", color="#3d6fa8", lw=2, ms=6)
    ax.axhline(0, color=MUTED, ls=":", lw=1.2)
    ax.set_xlabel("probability of the standard (%)")
    ax.set_ylabel("order asymmetry of layer 1 W")
    ax.set_title("c.  Layer 1 is only asymmetric off balance", fontsize=10.5)

    # (d) the layer 1 ablation, now at ninety ten
    names = {"raw": "stimulus only\n(no cortex)",
             "frozen": "layer 1, no\nrecurrent learning",
             "full": "layer 1, full"}
    ax = _tidy(fig.add_subplot(gs[0, 3]))
    x = np.arange(len(modes)); w = 0.38
    ax.bar(x - w / 2, mode_bal, w, color="#a3aebd", label="named (balanced acc.)")
    ax.bar(x + w / 2, mode_auc, w, color=C_AB, label="detected (ROC area)")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    for xi, v in zip(x - w / 2, mode_bal):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.6)
    for xi, v in zip(x + w / 2, mode_auc):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.6)
    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in modes], fontsize=8.4)
    ax.set_ylim(0, 1.25); ax.set_ylabel("score")
    ax.legend(fontsize=8.2, loc="upper left")
    ax.set_title(f"d.  What layer 1 contributes at "
                 f"{P_STD:.0%}/{1 - P_STD:.0%}", fontsize=10.5)

    _caption(fig, gs[1, :],
             "c.  At fifty fifty the Hebbian rule in layer 1 sees each order equally often, so its recurrent weights come out "
             "symmetric and the asymmetry sits at zero. Making one order rare is what breaks that symmetry,\n"
             "which is the only regime in which layer 1's learned prediction can contribute anything to layer 2. "
             "Panel d is therefore the test that the fifty fifty run could not perform.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CHANNELS, **LAYER1)
    l2cfg = L2Config()
    kw = dict(n_train=N_TRAIN, n_test=N_TEST)

    print(f"[ layer2_syllable_90_10 ] AB at {P_STD:.0%}, BA at {1 - P_STD:.0%}")
    print(f"  layer 1: model0 selective inhibition, N={a1cfg.N}, untouched")
    print(f"  layer 2: {l2cfg.n_units} units available\n")

    res = train_and_test(l2cfg, a1cfg, mode="full", seed=0,
                         weights=(P_STD, 1 - P_STD), **kw)
    res_50 = train_and_test(l2cfg, a1cfg, mode="full", seed=0,
                            weights=(0.5, 0.5), record_every=0, **kw)
    st = rare_stats(res)
    print(f"  committed units : {st['n_units']} of {l2cfg.n_units}   "
          f"(at 50/50 the same model commits {res_50['l2'].n_components})")
    for u in st["comm"]:
        print(f"    unit {u}: reads as '{_read_mask(res['l2'].M[u])}', "
              f"selectivity "
              f"{selectivity(res['resp'], res['test']['labels'], u):+.2f}")
    print(f"  balanced accuracy : {st['bal']:.1%}  (chance 50%)")
    print(f"  area under ROC    : {st['best_auc']:.2f} on unit {st['best_unit']}")
    print(f"  layer 1 W asymmetry: {asym(res['W1']):+.3f}  "
          f"(at 50/50 it is {asym(res_50['W1']):+.3f})\n")

    fig_tape(res, a1cfg, str(out / "l2_90_10_tape.png"), n_show=18)
    fig_vocabulary(res, res_50, str(out / "l2_90_10_vocabulary.png"))

    # ---- sweep the standard probability ----
    print("  sweep, probability of the standard:")
    ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sweep, W_by_p = [], []
    for p in ps:
        r = train_and_test(l2cfg, a1cfg, mode="full", seed=0,
                           weights=(p, 1 - p), record_every=0, **kw)
        s = rare_stats(r)
        sweep.append(s); W_by_p.append(r["W1"])
        print(f"    p(std)={p:4.2f}   units {s['n_units']}   "
              f"balanced {s['bal']:.2f}   auc {s['best_auc']:.2f}   "
              f"W asym {asym(r['W1']):+.3f}")

    # ---- the layer 1 test, now in the regime where it can matter ----
    print("  control, layer 1 contribution at the unbalanced ratio:")
    modes = ["raw", "frozen", "full"]
    mode_bal, mode_n, mode_auc = [], [], []
    for m in modes:
        r = train_and_test(l2cfg, a1cfg, mode=m, seed=0,
                           weights=(P_STD, 1 - P_STD), record_every=0, **kw)
        s = rare_stats(r)
        mode_bal.append(s["bal"]); mode_n.append(s["n_units"])
        mode_auc.append(s["best_auc"])
        print(f"    {m:7s}  units {s['n_units']}   balanced {s['bal']:.2f}   "
              f"auc {s['best_auc']:.2f}")

    fig_controls(ps, sweep, modes, mode_bal, mode_n, mode_auc, W_by_p,
                 str(out / "l2_90_10_controls.png"))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
