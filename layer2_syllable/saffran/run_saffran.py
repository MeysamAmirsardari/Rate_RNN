"""
layer2_syllable.saffran.run_saffran
===================================

Saffran style statistical word segmentation, run on the two layer model.

The paradigm (Saffran, Aslin and Newport 1996)
----------------------------------------------
Four three token "words" are built from twelve distinct channels and
concatenated in random order into one continuous stream.  A word never
immediately follows itself.  The stream is **perfectly isochronous**: the gap
between two tokens inside a word and the gap across a word boundary are the
same, so nothing in the timing or the level marks a boundary.  The only thing
that distinguishes a boundary is the statistics:

    inside a word     P(next token | current) = 1.0
    across a boundary P(next token | current) = 1/3

so a within word transition is three times more frequent than any particular
boundary transition.  After exposure the model is tested, as infants are, on
isolated **words** against **part words**, which are three token sequences that
did occur in the stream but straddle a boundary.

What is actually being asked
----------------------------
Three separate questions, deliberately kept apart, because passing one does not
mean passing another:

1. Does the layer preferentially learn the within word transitions?  With
   twenty distinct transitions occurring in the stream, of which eight are
   within word, chance is 40 percent.
2. Does any single unit come to represent a whole three token word, rather than
   one transition inside it?  The prediction from the model's own structure is
   no: the coincidence map is instantaneous, a three token word has two
   informative moments, and a rank one mask cannot hold both.
3. Can the population nevertheless separate words from part words?  A word
   contains two within word transitions, a part word at most one, so counting
   transitions can pass the behavioural test without anything in the model
   representing a word.

Question 3 passing while question 2 fails would mean the model reproduces the
behaviour by a mechanism weaker than the one the behaviour is usually taken to
demonstrate, which is worth stating plainly rather than reporting as success.

Run
---
    python -m layer2_syllable.saffran.run_saffran
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
from layer2_syllable.layer2 import Layer2
from layer2_syllable.run_ab_ba import (C_AB, C_ACC, C_BA, GRID, INK, LAYER1,
                                       MUTED, _caption, _tidy, auc,
                                       layer1_rates)
from layer2_syllable.stimulus import build_stream, chunk_windows

# ---------------------------------------------------------------------
#  Paradigm
# ---------------------------------------------------------------------
WORDS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
N_CH = 12
WORD_NAME = ["W1", "W2", "W3", "W4"]

TONE_DUR = 0.050
GAP = 0.030                 # used BOTH inside a word and across a boundary
N_TRAIN_WORDS = 500
N_TEST_REPS = 12            # presentations of each isolated test item

C_WITHIN, C_BOUND = "#2f7d5d", "#c1553b"


def word_order(n_words, n_chunks, rng, allow_repeat=False):
    """Random word order in which a word never immediately repeats."""
    order, prev = [], -1
    for _ in range(n_chunks):
        opts = [i for i in range(n_words) if allow_repeat or i != prev]
        w = int(rng.choice(opts))
        order.append(w)
        prev = w
    return np.array(order)


def within_bigrams(words):
    out = set()
    for w in words:
        for a, b in zip(w[:-1], w[1:]):
            out.add((a, b))
    return out


def boundary_bigrams(words):
    out = set()
    for i, wi in enumerate(words):
        for j, wj in enumerate(words):
            if i != j:
                out.add((wi[-1], wj[0]))
    return out


def part_words(words):
    """Three token sequences that occurred but straddle a boundary."""
    out = []
    for i, wi in enumerate(words):
        for j, wj in enumerate(words):
            if i == j:
                continue
            out.append(((wi[-1], wj[0], wj[1]), "1 + 2"))   # one then two
            out.append(((wi[-2], wi[-1], wj[0]), "2 + 1"))  # two then one
    return out


# ---------------------------------------------------------------------
#  Exposure and test
# ---------------------------------------------------------------------
def expose(l2cfg, a1cfg, mode="full", seed=0, n_words=N_TRAIN_WORDS,
           scrambled=False, record_every=400):
    """Run the continuous exposure stream and return the trained layer."""
    rng = np.random.default_rng(1000 + seed)
    if scrambled:
        # same twelve tokens, but every ordered pair equally likely, so no word
        # structure exists at all
        toks = rng.integers(0, N_CH, size=n_words * 3)
        items = [(int(t),) for t in toks]
        stream = build_stream(items, [1.0] * len(items), len(items), N_CH,
                              a1cfg.dt, tone_dur=TONE_DUR, intra_gap=GAP,
                              inter_gap=GAP, seed=seed,
                              order=np.arange(len(items)))
    else:
        order = word_order(len(WORDS), n_words, rng)
        stream = build_stream(WORDS, [0.25] * 4, n_words, N_CH, a1cfg.dt,
                              tone_dur=TONE_DUR, intra_gap=GAP, inter_gap=GAP,
                              seed=seed, order=order)
    E, _ = layer1_rates(stream["stim"], a1cfg, mode=mode, seed=seed)
    l2 = Layer2(N_CH, l2cfg)
    tr = l2.run(E, a1cfg.dt, learn=True, record_every=record_every)
    return l2, stream, tr, E


def test_items(l2, a1cfg, mode="full", seed=0):
    """Present words and part words in isolation, weights frozen."""
    pws = part_words(WORDS)
    items = [w for w in WORDS] + [p for p, _ in pws]
    kind = ["word"] * len(WORDS) + ["part word"] * len(pws)
    ptype = ["word"] * len(WORDS) + [t for _, t in pws]

    rng = np.random.default_rng(77 + seed)
    order = np.concatenate([rng.permutation(len(items))
                            for _ in range(N_TEST_REPS)])
    stream = build_stream(items, [1.0] * len(items), len(order), N_CH,
                          a1cfg.dt, tone_dur=TONE_DUR, intra_gap=GAP,
                          inter_gap=0.500, seed=seed, order=order)
    E, _ = layer1_rates(stream["stim"], a1cfg, mode=mode, seed=seed)
    l2.reset_state()
    te = l2.run(E, a1cfg.dt, learn=False)

    wins = chunk_windows(stream, pad_s=0.05)
    comm = np.flatnonzero(l2.committed)
    pop = np.array([te["y"][comm, a:b].max(axis=1).sum() if comm.size else 0.0
                    for a, b in wins])
    per_unit = np.array([te["y"][:, a:b].max(axis=1) for a, b in wins])
    # labels are per distinct item, responses are per presentation, so expand
    # them through the presentation order before they are compared
    kind_pres = np.array(kind)[order]
    ptype_pres = np.array(ptype)[order]
    return dict(items=items, kind=kind_pres, ptype=ptype_pres,
                item_kind=np.array(kind), order=order, pop=pop,
                per_unit=per_unit, stream=stream, te=te)


# ---------------------------------------------------------------------
#  Analysis
# ---------------------------------------------------------------------
def analyse(l2, scrambled=False):
    """Classify what each committed unit learned."""
    wb, bb = within_bigrams(WORDS), boundary_bigrams(WORDS)
    comm = np.flatnonzero(l2.committed)
    rows = []
    for u in comm:
        M = l2.M[u]
        j, i = np.unravel_index(np.argmax(M), M.shape)   # j now, i recently
        big = (int(i), int(j))
        cls = "within" if big in wb else ("boundary" if big in bb else "other")
        tot = float(M.sum())
        conc = float(M.max() / tot) if tot > 0 else 0.0
        order = np.argsort(M.ravel())[::-1]
        j2, i2 = np.unravel_index(order[1], M.shape)
        second = float(M.ravel()[order[1]] / tot) if tot > 0 else 0.0
        # A unit representing a whole word would need its second entry to CHAIN
        # onto the first: the token that fires now in the top entry must be the
        # one that fired recently in the second.  Anything else is a single
        # token seen in a context, not a word.
        spans = bool(int(i2) == int(j) and (int(i2), int(j2)) in wb)
        # fraction of the mask in its strongest row, i.e. how much the unit is
        # committed to a single channel firing NOW
        row_conc = float(M.sum(1).max() / tot) if tot > 0 else 0.0
        rows.append(dict(unit=int(u), bigram=big, cls=cls, conc=conc,
                         second=second, spans=spans, row_conc=row_conc,
                         context=[int(c) for c in np.argsort(M[j])[::-1][:2]]))
    n_w = sum(r["cls"] == "within" for r in rows)
    n_occ = len(wb) + len(bb) if not scrambled else N_CH * (N_CH - 1)
    covered = len({r["bigram"] for r in rows if r["cls"] == "within"})
    return dict(rows=rows, n_units=len(rows), n_within=n_w,
                frac_within=(n_w / len(rows)) if rows else 0.0,
                chance=len(wb) / n_occ, covered=covered, n_within_total=len(wb),
                n_span=sum(r["spans"] for r in rows),
                mean_conc=float(np.mean([r["conc"] for r in rows])) if rows else 0.0,
                mean_second=float(np.mean([r["second"] for r in rows])) if rows else 0.0,
                mean_rowconc=float(np.mean([r["row_conc"] for r in rows])) if rows else 0.0)


def word_score(test):
    """Separation of words from part words by the summed population response."""
    w = test["pop"][test["kind"] == "word"]
    p = test["pop"][test["kind"] == "part word"]
    a = auc(w, p)
    return dict(word=w, part=p, auc=max(a, 1.0 - a),
                d=float((w.mean() - p.mean()) /
                        np.sqrt(0.5 * (w.var() + p.var()) + 1e-12)))


# ---------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------
def fig_paradigm(stream, a1cfg, fname, n_words_show=9):
    dt = a1cfg.dt
    fig = plt.figure(figsize=(15.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.0, 0.30])
    fig.suptitle("The paradigm: four words hidden in one continuous stream, "
                 "with nothing but statistics marking the boundaries",
                 fontsize=13.5, fontweight="bold")

    # (a) the four words
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    for wi, w in enumerate(WORDS):
        for k, ch in enumerate(w):
            ax.add_patch(plt.Rectangle((k, wi - 0.32), 0.86, 0.64,
                                       color=C_ACC, alpha=0.85))
            ax.text(k + 0.43, wi, str(ch), ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.2, 3.1); ax.set_ylim(-0.7, len(WORDS) - 0.3)
    ax.set_yticks(range(len(WORDS))); ax.set_yticklabels(WORD_NAME)
    ax.set_xticks([0.43, 1.43, 2.43]); ax.set_xticklabels(["1st", "2nd", "3rd"])
    ax.invert_yaxis()
    ax.set_title("a.  The four words (numbers are channels)", fontsize=10.5)

    # (b) transitional probability matrix
    ax = fig.add_subplot(gs[0, 1])
    TP = np.zeros((N_CH, N_CH))
    for w in WORDS:
        for x, y in zip(w[:-1], w[1:]):
            TP[x, y] = 1.0
    for i, wi in enumerate(WORDS):
        for j, wj in enumerate(WORDS):
            if i != j:
                TP[wi[-1], wj[0]] = 1.0 / 3.0
    im = ax.imshow(TP, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("next channel"); ax.set_ylabel("current channel")
    ax.set_title("b.  Transitional probability", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) inter onset intervals
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    on = np.array([o for ons in stream["tone_onsets"] for o in ons])
    ioi = np.diff(np.sort(on)) * dt * 1e3
    ax.hist(ioi, bins=np.arange(60, 105, 2.5), color=C_ACC)
    ax.set_xlabel("interval between token onsets (ms)")
    ax.set_ylabel("count")
    ax.set_title("c.  Timing gives nothing away", fontsize=10.5)
    ax.text(0.5, 0.72, f"every interval is\n{ioi[0]:.0f} ms",
            transform=ax.transAxes, ha="center", fontsize=10,
            fontweight="bold", color=INK)

    # (d) a stretch of the stream
    ax = _tidy(fig.add_subplot(gs[1, :]))
    t1 = int(stream["starts"][n_words_show])
    ts = np.arange(t1) * dt
    for ch in range(N_CH):
        v = stream["stim"][ch, :t1]
        ax.fill_between(ts, ch - 0.42, ch - 0.42 + 0.84 * v, step="mid",
                        color=C_ACC, lw=0, alpha=0.9)
    for k in range(n_words_show):
        x = stream["starts"][k] * dt
        ax.axvline(x, color=C_BOUND, lw=1.4, ls="--", alpha=0.8)
        ax.text(x + 0.02, N_CH - 0.2, WORD_NAME[stream["labels"][k]],
                fontsize=9.5, fontweight="bold", color=C_BOUND)
    ax.set_xlim(0, ts[-1]); ax.set_ylim(-0.8, N_CH + 0.4)
    ax.set_yticks(range(N_CH))
    ax.set_xlabel("time (s)"); ax.set_ylabel("channel")
    ax.set_title("d.  The exposure stream. The dashed lines are word boundaries "
                 "and are drawn for the reader only", fontsize=10.5)

    _caption(fig, gs[2, :],
             "The boundaries in panel d are invisible to the model: the gap inside a word and the gap across a boundary are "
             "identical, as panel c confirms, and every token is the same duration and level.\n"
             "The only thing that separates them is panel b. A transition inside a word is certain, while a transition across a "
             "boundary leads to one of three possible next words and so is three times rarer.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


def fig_learned(l2, tr, an, fname):
    K = l2.cfg.n_units
    norms, tt = tr["norm_traj"], tr["norm_t"]
    comm = np.flatnonzero(l2.committed)
    cls_of = {r["unit"]: r["cls"] for r in an["rows"]}
    big_of = {r["unit"]: r["bigram"] for r in an["rows"]}

    fig = plt.figure(figsize=(15.5, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.30])
    fig.suptitle("What the layer extracted from the stream",
                 fontsize=13.5, fontweight="bold")

    # (a) mask norms
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    for u in range(K):
        if l2.committed[u]:
            c = C_WITHIN if cls_of.get(u) == "within" else C_BOUND
            ax.plot(tt, norms[:, u], color=c, lw=1.8)
        else:
            ax.plot(tt, norms[:, u], color=MUTED, lw=0.9, alpha=0.5)
    ax.set_xlabel("time in the exposure stream (s)")
    ax.set_ylabel("mask strength")
    ax.set_title("a.  Every unit during exposure", fontsize=10.5)
    # headroom so the legend sits above the traces rather than across them
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    ax.legend(handles=[plt.Line2D([], [], color=C_WITHIN, lw=3, label="within word"),
                       plt.Line2D([], [], color=C_BOUND, lw=3, label="across a boundary"),
                       plt.Line2D([], [], color=MUTED, lw=2, label="silent")],
              fontsize=8.6, ncol=3, loc="upper center")

    # (b) what each committed unit learned
    ax = _tidy(fig.add_subplot(gs[0, 1]))
    xs = np.arange(len(an["rows"]))
    cols = [C_WITHIN if r["cls"] == "within" else C_BOUND for r in an["rows"]]
    ax.bar(xs, [1] * len(xs), color=cols, width=0.72)
    for i, r in enumerate(an["rows"]):
        ax.text(i, 0.5, f"{r['bigram'][0]}→{r['bigram'][1]}", ha="center",
                va="center", color="white", fontsize=8.6, fontweight="bold",
                rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels([r["unit"] for r in an["rows"]],
                                          fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("unit")
    ax.set_title("b.  The transition each unit learned", fontsize=10.5)

    # (c) within word fraction against chance
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.bar([0, 1], [an["chance"], an["frac_within"]],
           color=[GRID, C_WITHIN], width=0.5)
    ax.axhline(an["chance"], color=MUTED, ls=":", lw=1.2)
    for x, v in ((0, an["chance"]), (1, an["frac_within"])):
        ax.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=12,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["chance\n(transitions that occur)", "the layer"],
                       fontsize=8.8)
    ax.set_ylim(0, 1.18); ax.set_ylabel("fraction within word")
    ax.set_title("c.  Did it find the words' transitions", fontsize=10.5)

    # (d) example masks
    show = list(comm[:3])
    vmax = max(l2.M.max(), 1e-9)
    for k, u in enumerate(show):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(l2.M[u], cmap="RdPu", vmin=0, vmax=vmax, aspect="auto")
        i, j = big_of[u]
        ax.set_title(f"unit {u}: {i} then {j}  ({cls_of[u]})", fontsize=10,
                     color=C_WITHIN if cls_of[u] == "within" else C_BOUND,
                     fontweight="bold")
        ax.set_xlabel("fired recently"); ax.set_ylabel("firing now")
        ax.set_xticks(range(0, N_CH, 2)); ax.set_yticks(range(0, N_CH, 2))

    _caption(fig, gs[2, :],
             f"d.  {an['mean_rowconc']:.0%} of every mask lies in a single row, so a unit codes one token firing now together with "
             f"the set of tokens that preceded it. It is a left context detector for one token,\n"
             f"which is more than a bare transition but much less than a word. A unit representing a whole three token word would "
             f"need its second entry to chain onto its first, and {an['n_span']} of {an['n_units']} do.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


def fig_test(test, ws, an, fname):
    fig = plt.figure(figsize=(15.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.34])
    fig.suptitle("The test: isolated words against part words, weights frozen",
                 fontsize=13.5, fontweight="bold")

    # (a) distributions
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    rng = np.random.default_rng(0)
    for x, v, c, lab in ((0, ws["word"], C_WITHIN, "words"),
                         (1, ws["part"], C_BOUND, "part words")):
        ax.scatter(x + rng.normal(0, 0.07, v.size), v, s=14, color=c,
                   alpha=0.45, lw=0)
        ax.hlines(v.mean(), x - 0.25, x + 0.25, color=c, lw=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["words", "part words"])
    ax.set_ylabel("summed population response")
    ax.set_title("a.  Response to the test items", fontsize=10.5)

    # (b) the behavioural score
    ax = _tidy(fig.add_subplot(gs[0, 1]))
    ax.bar([0], [ws["auc"]], color=C_ACC, width=0.45)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(0, ws["auc"] + 0.03, f"{ws['auc']:.2f}", ha="center", fontsize=14,
            fontweight="bold")
    ax.text(-0.42, 0.52, "chance", fontsize=8.4, color=MUTED, va="bottom")
    ax.set_xticks([0]); ax.set_xticklabels(["word versus\npart word"],
                                           fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_ylabel("area under ROC")
    ax.set_title("b.  Does it pass the test", fontsize=10.5)

    # (c) by part word type
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    groups = ["word", "1 + 2", "2 + 1"]
    means = [test["pop"][test["ptype"] == g].mean() for g in groups]
    sems = [test["pop"][test["ptype"] == g].std() /
            np.sqrt(max((test["ptype"] == g).sum(), 1)) for g in groups]
    ax.bar(np.arange(3), means, yerr=sems, capsize=4,
           color=[C_WITHIN, C_BOUND, C_BOUND], width=0.55)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["word\n(2 within)", "1 + 2\n(1 within)",
                        "2 + 1\n(1 within)"], fontsize=8.8)
    ax.set_ylabel("summed population response")
    ax.set_title("c.  Graded by how many within word\ntransitions the item contains",
                 fontsize=10)

    # (d) can any single unit do it
    ax = _tidy(fig.add_subplot(gs[0, 3]))
    per = test["per_unit"]
    kinds = test["kind"]
    aucs = []
    for u in range(per.shape[1]):
        a = auc(per[kinds == "word", u], per[kinds == "part word", u])
        aucs.append(max(a, 1.0 - a))
    ax.bar(np.arange(len(aucs)), aucs, color=C_ACC, width=0.7)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.axhline(ws["auc"], color=C_WITHIN, ls="--", lw=1.6)
    ax.text(len(aucs) - 0.5, ws["auc"] + 0.02, "population ", fontsize=8.2,
            color=C_WITHIN, ha="right", va="bottom")
    ax.set_xlabel("unit"); ax.set_ylabel("area under ROC")
    ax.set_ylim(0, 1.15)
    ax.set_title("d.  Each unit on its own", fontsize=10.5)

    _caption(fig, gs[1, :],
             f"The population separates words from part words at an area under the ROC of {ws['auc']:.2f}, so the model passes the "
             "behavioural test. Panel c shows why, and it is worth being precise about it: the response simply counts how many\n"
             "high probability transitions an item contains, two for a word and one for either kind of part word. That is enough to "
             "pass, but it is not the same as having learned a word. No single unit spans a whole word (panel d, and the mask\n"
             "concentration in the previous figure), so the model reproduces the behaviour by a weaker mechanism than the "
             "behaviour is usually taken to demonstrate.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


def fig_controls(taus, tau_frac, tau_auc, tau_units, caps, cap_frac, cap_cov,
                 ctrl, fname):
    fig = plt.figure(figsize=(15.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.34])
    fig.suptitle("Controls", fontsize=13.5, fontweight="bold")
    tv = np.array(taus) * 1e3

    ax = _tidy(fig.add_subplot(gs[0, 0]))
    ax.plot(tv, tau_frac, "o-", color=C_WITHIN, lw=2, ms=6)
    ax.axhline(ctrl["chance"], color=MUTED, ls=":", lw=1.2)
    ax.text(tv[0], ctrl["chance"] + 0.02, "chance", fontsize=8.2, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("decay of the slow conductance (ms)")
    ax.set_ylabel("fraction within word")
    ax.set_ylim(0, 1.1)
    ax.set_title("a.  Finding the right transitions", fontsize=10.5)

    ax = _tidy(fig.add_subplot(gs[0, 1]))
    ax.plot(tv, tau_auc, "^-", color=C_ACC, lw=2, ms=6)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("decay of the slow conductance (ms)")
    ax.set_ylabel("word versus part word (ROC)")
    ax.set_ylim(0.35, 1.1)
    ax.set_title("b.  Passing the test", fontsize=10.5)

    # capacity: with more units than there are transitions, nothing is forced
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.plot(caps, cap_frac, "o-", color=C_WITHIN, lw=2, ms=6,
            label="fraction within word")
    ax.plot(caps, np.array(cap_cov) / 8.0, "s--", color=C_ACC, lw=2, ms=6,
            label="of the 8 word transitions,\nhow many are covered")
    ax.axhline(ctrl["chance"], color=MUTED, ls=":", lw=1.2)
    ax.text(caps[0], ctrl["chance"] + 0.02, "chance", fontsize=8.2, color=MUTED)
    ax.axvline(8, color=MUTED, ls="--", lw=1.0)
    ax.set_xlabel("units available")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7.8, loc="lower right")
    ax.set_title("c.  Capacity forces the choice", fontsize=10.5)

    ax = _tidy(fig.add_subplot(gs[0, 3]))
    ax.bar([0, 1], [ctrl["struct_frac"], ctrl["scram_frac"]],
           color=[C_WITHIN, GRID], width=0.5)
    ax.axhline(ctrl["chance"], color=MUTED, ls=":", lw=1.2)
    ax.text(-0.42, ctrl["chance"] + 0.015, "chance, structured stream",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    ax.axhline(ctrl["scram_chance"], color=MUTED, ls="--", lw=1.2)
    ax.text(-0.42, ctrl["scram_chance"] + 0.015, "chance when scrambled",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    for x, v in ((0, ctrl["struct_frac"]), (1, ctrl["scram_frac"])):
        ax.text(x, max(v, ctrl["chance"]) + 0.07, f"{v:.0%}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["the real stream", "tokens scrambled\n(no words exist)"],
                       fontsize=8.8)
    ax.set_ylim(0, 1.15); ax.set_ylabel("fraction within word")
    ax.set_title("d.  The null stream", fontsize=10.5)

    _caption(fig, gs[1, :],
             "a and b.  The conductance has to outlast a token before a transition can be seen at all, and once it lasts far longer than a word it starts bridging boundaries.\n"
             "c.  Twenty distinct transitions occur in the stream and only eight of them are within a word, so a layer given twenty units never has to choose and the preference looks weak. "
             "The dashed line marks\nmatched capacity. This is a property of the test rather than of the model, and it is why the main run uses eight units.\n"
             "d.  With the same twelve tokens in uniformly random order no words exist, every ordered pair is equally likely, and the preference for the word defining transitions vanishes as it should.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CH, **LAYER1)
    # Capacity is matched to the eight within word transitions, so the layer is
    # forced to choose which of the twenty occurring transitions to keep.  See
    # the capacity panel in the controls figure for why this matters.
    l2cfg = dc.replace(L2Config(), n_units=8, tau_decay=0.150)

    wb, bb = within_bigrams(WORDS), boundary_bigrams(WORDS)
    print("[ saffran ] four three token words in one isochronous stream")
    print(f"  channels {N_CH}, words {len(WORDS)}, "
          f"within word transitions {len(wb)}, boundary transitions {len(bb)}")
    print(f"  exposure {N_TRAIN_WORDS} words "
          f"({N_TRAIN_WORDS * 3 * (TONE_DUR + GAP):.0f} s), "
          f"layer 2 units {l2cfg.n_units}\n")

    l2, stream, tr, _ = expose(l2cfg, a1cfg, seed=0)
    an = analyse(l2)
    test = test_items(l2, a1cfg, seed=0)
    ws = word_score(test)

    print(f"  committed units : {an['n_units']} of {l2cfg.n_units}")
    for r in an["rows"]:
        print(f"    unit {r['unit']:2d}: {r['bigram'][0]:2d} then "
              f"{r['bigram'][1]:2d}   {r['cls']:8s} "
              f"(top entry holds {r['conc']:.0%} of the mask)")
    print(f"  within word     : {an['n_within']}/{an['n_units']} = "
          f"{an['frac_within']:.0%}  (chance {an['chance']:.0%}), covering "
          f"{an['covered']}/{an['n_within_total']} word transitions")
    print(f"  mask shape      : {an['mean_rowconc']:.0%} of the weight in one "
          f"row, top entry {an['mean_conc']:.0%}, second {an['mean_second']:.0%}")
    print(f"  units spanning a whole word : {an['n_span']}/{an['n_units']}")
    print(f"  word vs part word : ROC {ws['auc']:.2f}, d prime {ws['d']:.2f}\n")

    fig_paradigm(stream, a1cfg, str(out / "saffran_paradigm.png"))
    fig_learned(l2, tr, an, str(out / "saffran_learned.png"))
    fig_test(test, ws, an, str(out / "saffran_test.png"))

    # ---- controls ----
    print("  control, decay of the slow conductance:")
    taus = [0.05, 0.10, 0.15, 0.25, 0.40]
    tau_frac, tau_auc, tau_units = [], [], []
    for t in taus:
        c = dc.replace(l2cfg, tau_decay=t, tau_rise=min(0.040, t * 0.5))
        l2b, _, _, _ = expose(c, a1cfg, seed=0, record_every=0)
        ab = analyse(l2b)
        tb = test_items(l2b, a1cfg, seed=0)
        wsb = word_score(tb)
        tau_frac.append(ab["frac_within"]); tau_auc.append(wsb["auc"])
        tau_units.append(ab["n_units"])
        print(f"    decay {t*1e3:5.0f} ms  units {ab['n_units']:2d}  "
              f"within {ab['frac_within']:.0%}  roc {wsb['auc']:.2f}")

    print("  control, how many units are available:")
    caps = [4, 6, 8, 10, 12, 16, 20]
    cap_frac, cap_cov = [], []
    for K in caps:
        c = dc.replace(l2cfg, n_units=K)
        l2c, _, _, _ = expose(c, a1cfg, seed=0, record_every=0)
        ac = analyse(l2c)
        cap_frac.append(ac["frac_within"]); cap_cov.append(ac["covered"])
        print(f"    {K:2d} units  within {ac['frac_within']:3.0%}  "
              f"covers {ac['covered']}/8")

    print("  control, scrambled stream:")
    l2s, _, _, _ = expose(l2cfg, a1cfg, seed=0, scrambled=True, record_every=0)
    ans = analyse(l2s, scrambled=True)
    print(f"    units {ans['n_units']}  within {ans['frac_within']:.0%}  "
          f"(chance here {ans['chance']:.0%})")

    ctrl = dict(chance=an["chance"], struct_frac=an["frac_within"],
                scram_frac=ans["frac_within"], scram_chance=ans["chance"])
    fig_controls(taus, tau_frac, tau_auc, tau_units, caps, cap_frac, cap_cov,
                 ctrl, str(out / "saffran_controls.png"))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
