"""
layer2_syllable.run_ab_ba
=========================

Two layer experiment.  Layer 1 is ``model0`` with two tonotopic channels, A and
B, and is not modified.  Layer 2 is a population of eight units that reads
layer 1 activity and learns, without supervision, which ordered pairs occur in
the stream.

The stream contains AB and BA chunks at fifty fifty, so neither channel and
neither chunk is more frequent than the other.  Averaged over a chunk the two
chunk types are physically identical: both contain one A tone and one B tone.
The only thing that distinguishes them is the order, so a rate readout cannot
work and any selectivity that appears has to come from the temporal structure.

The script trains on one stream, freezes the weights, evaluates on a second
independent stream, and then runs two controls:

  timescale    sweep the decay of the slow conductance, including the
               degenerate case where it collapses onto E, the two factors share
               a timescale and the order information provably disappears,
  layer 1      compare layer 2 driven by the raw stimulus, by layer 1 with its
               recurrent weights held at zero, and by the full layer 1.  This
               separates what the cortical dynamics contribute from what the
               learned recurrent connectivity contributes.

Run
---
    python -m layer2_syllable.run_ab_ba
"""
from __future__ import annotations

import dataclasses as dc
import sys
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model0 import selective_inh, simulate
from layer2_syllable.config import L2Config
from layer2_syllable.layer2 import Layer2
from layer2_syllable.stimulus import build_stream, chunk_windows

# ---------------------------------------------------------------------
#  Experiment constants
# ---------------------------------------------------------------------
A, B = 0, 1
WORDS = [(A, B), (B, A)]
WORD_LABEL = ["AB", "BA"]
N_CHANNELS = 2
N_TRAIN, N_TEST = 400, 150

# Same layer 1 regime as the AB/BA oddball task, so the two studies are
# directly comparable.
LAYER1 = dict(w_IE_self=3.0, w_EI_self=0.40, W_norm=4.0)

# ---- a restrained, readable plotting style --------------------------
INK = "#22252a"
MUTED = "#8a9099"
GRID = "#dfe3e8"
C_A, C_B = "#c1553b", "#3d6fa8"          # channel A, channel B
C_AB, C_BA = "#d98f2b", "#3f8f7a"        # chunk type AB, BA
C_ACC = "#7b5ea7"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "bold", "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 110,
})


def _tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    return ax


def _caption(fig, cell, text):
    """Put explanatory text in its own reserved cell of the grid.

    Text drawn outside an axes is invisible to constrained_layout, so notes
    anchored below a panel silently collide with whatever is there.  Giving the
    caption an axes of its own means the layout engine reserves room for it.
    Line breaks are written by hand so the wrapping never depends on the
    figure size.
    """
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=8.6,
            color=MUTED, va="top", ha="left", linespacing=1.5)
    return ax


# distinct colours for the units that survive, assigned in order
COMMIT_COLORS = ["#7b5ea7", "#d98f2b", "#3f8f7a", "#c1553b", "#3d6fa8"]


def _ccol(i):
    return COMMIT_COLORS[i % len(COMMIT_COLORS)]


# ---------------------------------------------------------------------
#  Driving layer 1
# ---------------------------------------------------------------------
def layer1_rates(stim, cfg, mode="full", seed=0):
    """Return the (N, T) input that layer 2 will read.

    mode 'raw'    the stimulus itself, no cortex at all
    mode 'frozen' layer 1 with recurrent weights held at zero, so adaptation
                  and selective inhibition act but nothing is learned
    mode 'full'   layer 1 as normal, recurrent plasticity on
    """
    if mode == "raw":
        return stim.copy(), None
    if mode == "frozen":
        out = simulate(stim, cfg=cfg, W_init=np.zeros((cfg.N, cfg.N)),
                       learn=False, seed=seed)
    elif mode == "full":
        out = simulate(stim, cfg=cfg, learn=True, seed=seed)
    else:
        raise ValueError(f"unknown layer 1 mode: {mode!r}")
    return out["E"], out["W_final"]


# ---------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------
def chunk_responses(y, stream):
    """Peak response of every unit within every chunk: (n_chunks, n_units)."""
    wins = chunk_windows(stream, pad_s=0.05)
    return np.array([y[:, a:b].max(axis=1) for a, b in wins])


def selectivity(resp, labels, unit):
    """(AB minus BA) over (AB plus BA) for one unit.  Zero means no preference."""
    a = resp[labels == 0, unit].mean()
    b = resp[labels == 1, unit].mean()
    return (a - b) / (a + b) if (a + b) > 0 else 0.0


def _balanced(pred, labels):
    """Mean of the per class recalls.

    With an unbalanced stream, plain accuracy is misleading: always answering
    "AB" scores ninety percent on a ninety ten stream while carrying no
    information.  Balanced accuracy has chance at one over the number of
    classes whatever the class proportions are, so it is the metric used
    whenever the stream is unbalanced.  On a fifty fifty stream the two agree.
    """
    recalls = []
    for c in range(len(WORDS)):
        m = labels == c
        if m.any():
            recalls.append(float((pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def decode(resp, labels, committed, metric="balanced"):
    """Assign each chunk to the most active committed unit and score it.

    Unit identity is arbitrary, so every assignment of units to chunk types is
    tried and the best is reported.  Chance is one over the number of words.
    """
    idx = np.flatnonzero(committed)
    if idx.size == 0:
        return 0.0, None, np.zeros((len(WORDS), len(WORDS)))
    winner = idx[np.argmax(resp[:, idx], axis=1)]
    score = (lambda p: _balanced(p, labels)) if metric == "balanced" \
        else (lambda p: float((p == labels).mean()))
    best_acc, best_map = -1.0, None
    uniq = list(idx)
    for perm in permutations(range(len(WORDS)), min(len(uniq), len(WORDS))):
        mapping = {u: p for u, p in zip(uniq, perm)}
        pred = np.array([mapping.get(w, -1) for w in winner])
        acc = score(pred)
        if acc > best_acc:
            best_acc, best_map = acc, mapping
    conf = np.zeros((len(WORDS), len(WORDS)))
    if best_map is not None:
        pred = np.array([best_map.get(w, -1) for w in winner])
        for t, p in zip(labels, pred):
            if p >= 0:
                conf[t, p] += 1
    return max(best_acc, 0.0), best_map, conf


def auc(x0, x1):
    """Area under the ROC for separating two response samples.

    Reported when only one unit survives, because then the winner take all
    readout is uninformative by construction and the real question is whether
    the magnitude of that single unit still distinguishes the two chunks.
    """
    if len(x0) == 0 or len(x1) == 0:
        return 0.5
    allv = np.concatenate([x0, x1])
    r = np.argsort(np.argsort(allv)) + 1.0
    r0 = r[:len(x0)].sum()
    u0 = r0 - len(x0) * (len(x0) + 1) / 2.0
    return float(u0 / (len(x0) * len(x1)))


def _read_mask(M):
    """Turn a mask into a sentence a reader can check against the picture."""
    i, j = np.unravel_index(np.argmax(M), M.shape)
    now, before = "AB"[i], "AB"[j]
    return f"{now} after {before}" if i != j else f"{now} sustained"


# ---------------------------------------------------------------------
#  One full train and test cycle
# ---------------------------------------------------------------------
def train_and_test(l2cfg, a1cfg, mode="full", seed=0, record_every=200,
                   weights=(0.5, 0.5), n_train=None, n_test=None):
    n_train = N_TRAIN if n_train is None else n_train
    n_test = N_TEST if n_test is None else n_test
    train = build_stream(WORDS, weights, n_train, N_CHANNELS, a1cfg.dt,
                         seed=100 + seed)
    test = build_stream(WORDS, weights, n_test, N_CHANNELS, a1cfg.dt,
                        seed=900 + seed)
    E_train, W_train = layer1_rates(train["stim"], a1cfg, mode=mode, seed=seed)
    E_test, _ = layer1_rates(test["stim"], a1cfg, mode=mode, seed=seed)

    l2 = Layer2(N_CHANNELS, l2cfg)
    tr = l2.run(E_train, a1cfg.dt, learn=True, record_every=record_every)
    l2.reset_state()                       # clear the trace, keep weights and stats
    te = l2.run(E_test, a1cfg.dt, learn=False)

    resp = chunk_responses(te["y"], test)
    acc, mapping, conf = decode(resp, test["labels"], l2.committed)
    return dict(l2=l2, train=train, test=test, E_train=E_train, E_test=E_test,
                tr=tr, te=te, resp=resp, acc=acc, mapping=mapping, conf=conf,
                W1=W_train, weights=tuple(weights))


# ---------------------------------------------------------------------
#  Figure 1: what the layer sees and what it does
# ---------------------------------------------------------------------
def fig_mechanism(res, a1cfg, fname):
    dt = a1cfg.dt
    stream, l2 = res["test"], res["l2"]
    E, s, y = res["E_test"], res["te"]["s"], res["te"]["y"]

    n_show = 5
    t_end = stream["starts"][n_show] if n_show < len(stream["starts"]) else E.shape[1]
    ts = np.arange(t_end) * dt
    labels = stream["labels"][:n_show]

    fig = plt.figure(figsize=(14.5, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(5, 4, height_ratios=[1.0, 1.0, 1.0, 1.0, 0.34],
                          width_ratios=[1.0, 1.0, 1.0, 0.78])
    fig.suptitle("What layer 2 reads from layer 1, and how it responds",
                 fontsize=13.5, fontweight="bold")

    # (a) stimulus
    ax = _tidy(fig.add_subplot(gs[0, :3]))
    for ch, col, nm in ((A, C_A, "A"), (B, C_B, "B")):
        ax.fill_between(ts, 0, stream["stim"][ch, :t_end], step="mid",
                        color=col, alpha=0.75, lw=0, label=f"tone {nm}")
    for k in range(n_show):
        x = stream["starts"][k] * dt
        ax.axvline(x, color=GRID, lw=1)
        ax.text(x + 0.02, 1.05, WORD_LABEL[labels[k]], fontsize=10,
                fontweight="bold", color=C_AB if labels[k] == 0 else C_BA)
    ax.set_ylim(0, 1.45); ax.set_yticks([])
    ax.set_title("a.  The stream: AB and BA chunks, fifty fifty")
    ax.legend(loc="upper right", ncol=2, fontsize=8.5)

    # (b) layer 1 rates
    ax = _tidy(fig.add_subplot(gs[1, :3]))
    ax.plot(ts, E[A, :t_end], color=C_A, lw=1.6, label="E of channel A")
    ax.plot(ts, E[B, :t_end], color=C_B, lw=1.6, label="E of channel B")
    ax.set_ylabel("rate")
    ax.set_ylim(top=E[:, :t_end].max() * 1.35)
    ax.set_title("b.  Layer 1 excitatory activity (model0, unchanged)")
    ax.legend(loc="upper right", ncol=2, fontsize=8.5)

    # (c) slow trace
    ax = _tidy(fig.add_subplot(gs[2, :3]))
    ax.plot(ts, s[A, :t_end], color=C_A, lw=1.6, ls="--", label="slow trace of A")
    ax.plot(ts, s[B, :t_end], color=C_B, lw=1.6, ls="--", label="slow trace of B")
    ax.set_ylabel("conductance")
    ax.set_ylim(top=s[:, :t_end].max() * 1.4)
    ax.set_title("c.  The slow conductance in layer 2 (rise 40 ms, decay 150 ms)")
    ax.legend(loc="upper right", ncol=2, fontsize=8.5)

    # (d) unit responses
    ax = _tidy(fig.add_subplot(gs[3, :3]))
    comm = np.flatnonzero(l2.committed)
    for i, u in enumerate(comm):
        ax.plot(ts, y[u, :t_end], lw=1.9, color=_ccol(i),
                label=f"unit {u}: {_read_mask(l2.M[u])}")
    for k in range(n_show):
        ax.axvline(stream["starts"][k] * dt, color=GRID, lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("response")
    ax.set_ylim(top=max(y[comm, :t_end].max() * 1.35, 1e-9))
    ax.set_title("d.  The committed layer 2 units, tested with learning switched off")
    ax.legend(loc="upper right", ncol=len(comm), fontsize=8.5)

    # (e, f) the directional map at chunk completion
    for row, want in ((0, 0), (1, 1)):
        k = int(np.flatnonzero(stream["labels"] == want)[0])
        a, b = chunk_windows(stream)[k]
        seg, sseg = E[:, a:b], s[:, a:b]
        nrm = [np.linalg.norm(np.outer(seg[:, i], sseg[:, i]))
               for i in range(seg.shape[1])]
        j = int(np.argmax(nrm))
        D = np.outer(seg[:, j], sseg[:, j])
        np.fill_diagonal(D, 0.0)          # the layer reads distinct pairs only
        ax = fig.add_subplot(gs[row * 2:row * 2 + 2, 3])
        im = ax.imshow(D, cmap="RdPu", vmin=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["A", "B"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["A", "B"])
        ax.set_xlabel("fired recently (slow)", fontsize=9)
        ax.set_ylabel("firing now (fast)", fontsize=9)
        ax.set_title(f"{'e' if row == 0 else 'f'}.  D at the end of "
                     f"an {WORD_LABEL[want]} chunk", fontsize=10)
        for i in range(2):
            for jj in range(2):
                ax.text(jj, i, f"{D[i, jj]:.1f}", ha="center", va="center",
                        fontsize=10,
                        color="white" if D[i, jj] > D.max() * 0.55 else INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _caption(fig, gs[4, :],
             "Both chunk types contain exactly one A tone and one B tone, so "
             "anything that averages over a chunk cannot tell them apart. Only the order differs.\n"
             "Layer 2 reads panel b and nothing else. Because its conductance rises slowly (panel c), a channel's own trace is "
             "still small while that channel fires,\nwhich is what stops the units from simply re-coding single tones. "
             "Panels e and f show the resulting coincidence map at the moment a chunk completes:\n"
             "the entry that lights up is B-after-A for an AB chunk and A-after-B for a BA chunk.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
#  Figure 2: the competition, and how many units survive it
# ---------------------------------------------------------------------
def fig_learning(res, fname):
    l2, tr = res["l2"], res["tr"]
    norms, tt = tr["norm_traj"], tr["norm_t"]
    K = l2.cfg.n_units
    comm = l2.committed
    order = {u: i for i, u in enumerate(np.flatnonzero(comm))}

    fig = plt.figure(figsize=(14.5, 9.6), constrained_layout=True)
    # Two plain rows for the masks rather than a nested subgridspec: nested
    # grids with explicit spacing fight constrained_layout and collapse the
    # square image axes to nothing.
    gs = fig.add_gridspec(4, 4, height_ratios=[1.15, 0.72, 0.72, 0.20])
    fig.suptitle("Eight units compete, and the stream decides how many survive",
                 fontsize=13.5, fontweight="bold")

    # (a) mask norm trajectories
    ax = _tidy(fig.add_subplot(gs[0, :2]))
    for u in range(K):
        if comm[u]:
            ax.plot(tt, norms[:, u], color=_ccol(order[u]), lw=2.0,
                    label=f"unit {u}: {_read_mask(l2.M[u])}")
        else:
            ax.plot(tt, norms[:, u], color=MUTED, lw=1.0, alpha=0.5)
    thr = l2.cfg.commit_frac * l2.mask_norms.max()
    ax.axhline(thr, color=MUTED, ls=":", lw=1.2)
    ax.text(tt[-1], thr, "commitment level ", fontsize=8.4, color=MUTED,
            va="bottom", ha="right")
    ax.set_xlabel("time in the training stream (s)")
    ax.set_ylabel("mask strength")
    ax.set_ylim(bottom=-0.03)
    ax.set_title("a.  Synaptic weight of every unit during training")
    ax.legend(loc="center right", fontsize=8.5)

    # (b) win counts
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    cols = [_ccol(order[u]) if comm[u] else GRID for u in range(K)]
    ax.bar(np.arange(K), l2.win_counts, color=cols)
    ax.set_xlabel("unit"); ax.set_ylabel("time steps won")
    ax.set_xticks(np.arange(K))
    ax.set_title("b.  Outcome of the competition")

    # (c) discovered count
    ax = _tidy(fig.add_subplot(gs[0, 3]))
    ax.bar([0, 1], [len(WORDS), l2.n_components], color=[GRID, C_ACC], width=0.5)
    for x, v in ((0, len(WORDS)), (1, l2.n_components)):
        ax.text(x, v + 0.07, str(v), ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["chunk types\nin the stream", "units the layer\ncommitted"],
                       fontsize=9)
    ax.set_ylim(0, max(len(WORDS), l2.n_components) + 0.9)
    ax.set_yticks([])
    ax.set_title("c.  How many components")

    # (d) every final mask
    vmax = max(l2.M.max(), 1e-9)
    per_row = K // 2
    for u in range(K):
        r, c = u // per_row, u % per_row
        ax = fig.add_subplot(gs[1 + r, c])
        ax.imshow(l2.M[u], cmap="RdPu", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["A", "B"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["A", "B"], fontsize=8)
        if comm[u]:
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
             "d.  The learned masks, all eight of them. Weight at (row B, column A) means the unit answers when B follows A, which is the chunk AB. "
             "Units that keep winning grow;\nunits that never win receive only decay and fade to zero, which is what deactivates them. "
             "The number of survivors is an outcome here, not a setting: eight units were available throughout.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
#  Figure 3: selectivity on the held out stream
# ---------------------------------------------------------------------
def fig_selectivity(res, fname):
    l2, test, resp = res["l2"], res["test"], res["resp"]
    labels = test["labels"]
    comm = np.flatnonzero(l2.committed)

    fig = plt.figure(figsize=(14.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.20])
    fig.suptitle("Selectivity on a fresh stream, with the weights frozen",
                 fontsize=13.5, fontweight="bold")

    # (a) response distributions
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    rng = np.random.default_rng(0)
    for i, u in enumerate(comm):
        for lab, col in ((0, C_AB), (1, C_BA)):
            v = resp[labels == lab, u]
            x = i * 2 + (0.0 if lab == 0 else 0.8)
            ax.scatter(x + rng.normal(0, 0.09, v.size), v, s=9, color=col,
                       alpha=0.45, lw=0)
            ax.hlines(v.mean(), x - 0.28, x + 0.28, color=col, lw=2.6)
    ax.set_xticks([i * 2 + 0.4 for i in range(len(comm))])
    ax.set_xticklabels([f"unit {u}" for u in comm])
    ax.set_ylabel("peak response in chunk")
    ax.set_ylim(bottom=-0.5, top=resp[:, comm].max() * 1.28)
    ax.set_title("a.  Response by chunk type")
    h = [plt.Line2D([], [], color=C_AB, lw=3, label="AB chunks"),
         plt.Line2D([], [], color=C_BA, lw=3, label="BA chunks")]
    ax.legend(handles=h, fontsize=8.5, loc="upper center", ncol=2)

    # (b) selectivity index
    ax = _tidy(fig.add_subplot(gs[0, 1]))
    si = [selectivity(resp, labels, u) for u in comm]
    ax.bar(np.arange(len(comm)), si,
           color=[C_AB if v > 0 else C_BA for v in si], width=0.5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks(np.arange(len(comm)))
    ax.set_xticklabels([f"unit {u}" for u in comm])
    ax.set_ylabel("prefers BA        prefers AB")
    ax.set_title("b.  Selectivity index")
    for i, v in enumerate(si):
        ax.text(i, v + (0.07 if v >= 0 else -0.16), f"{v:+.2f}", ha="center",
                fontsize=10, fontweight="bold")

    # (c) confusion
    ax = fig.add_subplot(gs[0, 2])
    conf = res["conf"]
    cn = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    im = ax.imshow(cn, cmap="Purples", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cn[i, j]:.0%}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cn[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(WORD_LABEL)
    ax.set_yticks([0, 1]); ax.set_yticklabels(WORD_LABEL)
    ax.set_xlabel("unit that won"); ax.set_ylabel("chunk actually played")
    ax.set_title("c.  Which unit wins each chunk")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) accuracy against chance
    ax = _tidy(fig.add_subplot(gs[0, 3]))
    ax.bar([0, 1], [0.5, res["acc"]], color=[GRID, C_ACC], width=0.5)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["chance", "layer 2"])
    ax.set_ylim(0, 1.2); ax.set_ylabel("fraction of chunks correct")
    ax.text(1, res["acc"] + 0.04, f"{res['acc']:.0%}", ha="center",
            fontsize=13, fontweight="bold")
    ax.set_title("d.  Held out accuracy")

    _caption(fig, gs[1, :],
             "A selectivity index of plus one means the unit answers only to AB, minus one only to BA. "
             "The two chunk types contain exactly the same two tones and differ only in their order,\n"
             "so the stimulus itself carries no rate cue at all. Layer 1 adaptation does introduce a "
             "weak one, because the first tone of a chunk is less adapted than the second; see the\n"
             "separability panel in the skeptic figure, where the same units still succeed on the raw "
             "stimulus, for which every rate cue is exactly zero.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
#  Figure 4: controls
# ---------------------------------------------------------------------
def fig_controls(tau_vals, tau_acc, tau_n, modes, mode_acc, mode_n, fname):
    fig = plt.figure(figsize=(14.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.26])
    fig.suptitle("Controls: what the result actually depends on",
                 fontsize=13.5, fontweight="bold")
    tv = np.array(tau_vals) * 1e3

    # (a) timescale sweep, accuracy
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    ax.plot(tv, tau_acc, "o-", color=C_ACC, lw=2, ms=6)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(tv[-1], 0.52, "chance ", fontsize=8.4, color=MUTED,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("decay of the slow conductance (ms)")
    ax.set_ylabel("held out accuracy")
    ax.set_ylim(0.35, 1.12)
    ax.set_title("a.  The memory has to span a chunk")

    # (b) how many units commit across the sweep
    ax = _tidy(fig.add_subplot(gs[0, 1]))
    ax.plot(tv, tau_n, "s-", color=C_AB, lw=2, ms=6)
    ax.axhline(len(WORDS), color=MUTED, ls=":", lw=1.2)
    ax.text(tv[-1], len(WORDS) + 0.06, "chunk types in the stream ",
            fontsize=8.4, color=MUTED, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("decay of the slow conductance (ms)")
    ax.set_ylabel("units committed")
    ax.set_ylim(min(tau_n) - 0.5, max(tau_n) + 0.9)
    ax.set_title("b.  How many units the layer keeps")

    # (c) contribution of layer 1
    names = {"raw": "stimulus only\n(no cortex)",
             "frozen": "layer 1, no\nrecurrent learning",
             "full": "layer 1, full"}
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.bar(np.arange(len(modes)), mode_acc,
           color=["#c8ced6", "#a3aebd", C_ACC][:len(modes)], width=0.5)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(-0.45, 0.52, "chance", fontsize=8.4, color=MUTED,
            ha="left", va="bottom")
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([names[m] for m in modes], fontsize=8.4)
    ax.set_ylim(0, 1.2); ax.set_ylabel("held out accuracy")
    for i, v in enumerate(mode_acc):
        ax.text(i, v + 0.04, f"{v:.0%}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_title("c.  What layer 1 contributes")

    _caption(fig, gs[1, :],
             "a and b.  When the conductance becomes as fast as E the two factors share a timescale, the coincidence map turns symmetric and the order information "
             "disappears exactly, so the layer drops to chance.\nWhen it lasts far longer than a chunk the trace bridges the silence between chunks and spare units start "
             "duplicating. Only the middle range recovers both the order and the correct number of units.\n"
             "c.  At fifty fifty the recurrent weights of layer 1 become symmetric and carry no order information, so this comparison asks whether layer 1 contributes at all.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CHANNELS, **LAYER1)
    l2cfg = L2Config()

    print("[ layer2_syllable ] two layer AB versus BA, fifty fifty")
    print(f"  layer 1: model0 selective inhibition, N={a1cfg.N}, untouched")
    print(f"  layer 2: {l2cfg.n_units} units, conductance rise "
          f"{l2cfg.tau_rise*1e3:.0f} ms, decay {l2cfg.tau_decay*1e3:.0f} ms, "
          f"distinct pairs only = {l2cfg.no_self_pairs}")
    print(f"  stream : {N_TRAIN} training chunks, {N_TEST} held out chunks\n")

    res = train_and_test(l2cfg, a1cfg, mode="full", seed=0)
    l2 = res["l2"]
    print(f"  committed units : {l2.n_components} of {l2cfg.n_units}")
    for u in np.flatnonzero(l2.committed):
        print(f"    unit {u}: reads as '{_read_mask(l2.M[u])}', "
              f"selectivity {selectivity(res['resp'], res['test']['labels'], u):+.2f}")
    print(f"  held out accuracy : {res['acc']:.1%}  (chance {1/len(WORDS):.0%})\n")

    fig_mechanism(res, a1cfg, str(out / "l2_mechanism.png"))
    fig_learning(res, str(out / "l2_learning.png"))
    fig_selectivity(res, str(out / "l2_selectivity.png"))

    # ---- control 1: the trace timescale ----
    print("  control, decay of the slow conductance:")
    tau_vals = [0.005, 0.020, 0.050, 0.100, 0.150, 0.300, 0.800]
    tau_acc, tau_n = [], []
    for tv in tau_vals:
        c = dc.replace(l2cfg, tau_decay=tv,
                       tau_rise=min(l2cfg.tau_rise, tv * 0.5))
        r = train_and_test(c, a1cfg, mode="full", seed=0, record_every=0)
        tau_acc.append(r["acc"]); tau_n.append(r["l2"].n_components)
        print(f"    decay {tv*1e3:6.0f} ms   accuracy {r['acc']:6.1%}   "
              f"committed {r['l2'].n_components}")

    # ---- control 2: what layer 1 contributes ----
    print("  control, layer 1 contribution:")
    modes = ["raw", "frozen", "full"]
    mode_acc, mode_n = [], []
    for m in modes:
        r = train_and_test(l2cfg, a1cfg, mode=m, seed=0, record_every=0)
        mode_acc.append(r["acc"]); mode_n.append(r["l2"].n_components)
        print(f"    {m:7s}  accuracy {r['acc']:6.1%}   "
              f"committed {r['l2'].n_components}")

    fig_controls(tau_vals, tau_acc, tau_n, modes, mode_acc, mode_n,
                 str(out / "l2_controls.png"))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
