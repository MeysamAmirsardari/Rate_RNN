"""
layer2_syllable.visuals
=======================

Two further figures, aimed at a reader who does not want to take the summary
statistics on trust.

l2_tape.png      A long continuous stretch of the stream with every signal
                 stacked in its own row: the tones, layer 1 activity, the slow
                 conductance in layer 2, and then all eight layer 2 units, one
                 per row.  The background of every row is tinted by chunk type,
                 so the correspondence between what was played and which unit
                 answered can simply be read off, chunk by chunk, including the
                 six units that do nothing.

l2_skeptic.png   The checks a sceptical reader would ask for: responses aligned
                 to chunk onset, every single trial rather than an average, how
                 much information a rate code could possibly carry compared
                 with what the units carry, the trajectory of the coincidence
                 map itself, and stability across seeds.

Run
---
    python -m layer2_syllable.visuals
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
from layer2_syllable.run_ab_ba import (A, B, C_A, C_AB, C_ACC, C_B, C_BA, GRID,
                                       INK, LAYER1, MUTED, N_CHANNELS,
                                       WORD_LABEL, WORDS, _caption, _ccol,
                                       _read_mask, _tidy, chunk_windows,
                                       train_and_test)


def _stripe(ax, stream, dt, k0, k1, alpha=0.10):
    """Tint the background of an axes by chunk type."""
    for k in range(k0, k1):
        a = stream["starts"][k] * dt
        b = a + stream["lengths"][k] * dt
        ax.axvspan(a, b, color=C_AB if stream["labels"][k] == 0 else C_BA,
                   alpha=alpha, lw=0, zorder=0)


# ---------------------------------------------------------------------
#  Figure 5: everything stacked in rows against the same clock
# ---------------------------------------------------------------------
def fig_tape(res, a1cfg, fname, n_show=14):
    dt = a1cfg.dt
    stream, l2 = res["test"], res["l2"]
    E, s, y = res["E_test"], res["te"]["s"], res["te"]["y"]
    K = l2.cfg.n_units
    comm = l2.committed
    order = {u: i for i, u in enumerate(np.flatnonzero(comm))}

    t0 = 0
    t1 = int(stream["starts"][n_show])
    ts = np.arange(t0, t1) * dt

    rows = 3 + K
    heights = [0.85, 0.95, 0.85] + [0.55] * K
    fig = plt.figure(figsize=(16, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(rows + 1, 1, height_ratios=heights + [0.30])
    fig.suptitle("One continuous stretch of the stream, every signal in its own row",
                 fontsize=14, fontweight="bold")

    def newrow(i, ylab, last=False):
        ax = _tidy(fig.add_subplot(gs[i, 0]))
        _stripe(ax, stream, dt, 0, n_show)
        for k in range(n_show):
            ax.axvline(stream["starts"][k] * dt, color=GRID, lw=0.9, zorder=1)
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylabel(ylab, fontsize=9, rotation=0, ha="right", va="center",
                      labelpad=10)
        if not last:
            ax.tick_params(labelbottom=False)
        return ax

    # tones
    ax = newrow(0, "tones")
    for ch, col, nm in ((A, C_A, "A"), (B, C_B, "B")):
        ax.fill_between(ts, 0, stream["stim"][ch, t0:t1], step="mid",
                        color=col, alpha=0.85, lw=0, label=f"tone {nm}", zorder=3)
    ax.set_ylim(0, 1.7); ax.set_yticks([])
    # legend lifted into the title margin so it cannot sit on a chunk label
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2,
               fontsize=9, bbox_to_anchor=(0.995, 0.985))
    for k in range(n_show):
        x = (stream["starts"][k] + stream["lengths"][k] * 0.16) * dt
        ax.text(x, 1.30, WORD_LABEL[stream["labels"][k]], fontsize=10,
                fontweight="bold", ha="center",
                color=C_AB if stream["labels"][k] == 0 else C_BA, zorder=4)

    # layer 1
    ax = newrow(1, "layer 1\nrate E")
    ax.plot(ts, E[A, t0:t1], color=C_A, lw=1.5, zorder=3)
    ax.plot(ts, E[B, t0:t1], color=C_B, lw=1.5, zorder=3)
    ax.set_ylim(0, E[:, t0:t1].max() * 1.2)

    # slow conductance
    ax = newrow(2, "layer 2\nconductance")
    ax.plot(ts, s[A, t0:t1], color=C_A, lw=1.4, ls="--", zorder=3)
    ax.plot(ts, s[B, t0:t1], color=C_B, lw=1.4, ls="--", zorder=3)
    ax.set_ylim(0, s[:, t0:t1].max() * 1.25)

    # the eight units
    ymax = max(y[:, t0:t1].max(), 1e-9) * 1.25
    for u in range(K):
        ax = newrow(3 + u, f"unit {u}", last=(u == K - 1))
        if comm[u]:
            col = _ccol(order[u])
            ax.fill_between(ts, 0, y[u, t0:t1], color=col, alpha=0.85, lw=0,
                            zorder=3)
            ax.text(0.004, 0.72, _read_mask(l2.M[u]), transform=ax.transAxes,
                    fontsize=8.8, fontweight="bold", color=col, zorder=4)
        else:
            ax.plot(ts, y[u, t0:t1], color=MUTED, lw=1.0, zorder=3)
            ax.text(0.004, 0.72, "silent", transform=ax.transAxes,
                    fontsize=8.4, color=MUTED, zorder=4)
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        if u == K - 1:
            ax.set_xlabel("time (s)")

    _caption(fig, gs[rows, 0],
             "Background tint gives the chunk that was played: orange for AB, green for BA. "
             "The two committed units answer at the completion of their own chunk type and stay flat "
             "for the other,\nchunk after chunk, with the weights frozen. The remaining six units are "
             "shown on the same scale and are doing nothing at all, which is the point: the layer was "
             "given eight and used two.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
#  Figure 6: the checks a sceptic asks for
# ---------------------------------------------------------------------
def _dprime(x, y):
    return abs(x.mean() - y.mean()) / np.sqrt(0.5 * (x.var() + y.var()) + 1e-12)


def _epoch_units(res, n_pre=0.02, n_post=0.30):
    """Unit responses aligned to chunk onset: (n_chunks, n_units, n_t)."""
    dt = res["test"]["dt"]
    y = res["te"]["y"]
    pre, post = int(n_pre / dt), int(n_post / dt)
    out = []
    for st in res["test"]["starts"]:
        a, b = st - pre, st + post
        if a < 0 or b > y.shape[1]:
            out.append(None); continue
        out.append(y[:, a:b])
    keep = [i for i, v in enumerate(out) if v is not None]
    return np.array([out[i] for i in keep]), np.array(keep), pre, dt


def fig_skeptic(res, res_raw, seed_acc, seed_n, a1cfg, fname):
    l2, test = res["l2"], res["test"]
    labels = test["labels"]
    comm = np.flatnonzero(l2.committed)
    ep, keep, pre, dt = _epoch_units(res)
    lab = labels[keep]
    tt = (np.arange(ep.shape[2]) - pre) * dt

    fig = plt.figure(figsize=(17.5, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.30])
    fig.suptitle("The checks a sceptical reader would ask for",
                 fontsize=14, fontweight="bold")

    # (a, b) chunk triggered average, one panel per committed unit
    for i, u in enumerate(comm[:2]):
        ax = _tidy(fig.add_subplot(gs[0, i]))
        for want, col in ((0, C_AB), (1, C_BA)):
            v = ep[lab == want, u, :]
            m, e = v.mean(0), v.std(0) / np.sqrt(max(len(v), 1))
            ax.plot(tt, m, color=col, lw=2, label=f"{WORD_LABEL[want]} chunks")
            ax.fill_between(tt, m - e, m + e, color=col, alpha=0.25, lw=0)
        ax.axvline(0, color=MUTED, ls=":", lw=1)
        ax.set_xlabel("time from chunk onset (s)")
        ax.set_ylabel("response")
        ax.set_ylim(top=ep[:, u, :].max() * 1.32)
        ax.set_title(f"{'ab'[i]}.  Unit {u}, mean and s.e.m.", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper right")

    # (c) every single trial, sorted by chunk type
    srt = np.argsort(lab, kind="stable")
    for i, u in enumerate(comm[:2]):
        ax = fig.add_subplot(gs[0, 2 + i])
        im = ax.imshow(ep[srt, u, :], aspect="auto", cmap="magma",
                       extent=[tt[0], tt[-1], len(srt), 0], vmin=0)
        nAB = int((lab == 0).sum())
        ax.axhline(nAB, color="white", lw=1.4)
        ax.text(tt[-1] * 0.98, nAB * 0.5, "AB", color="white", fontsize=10,
                fontweight="bold", ha="right", va="center")
        ax.text(tt[-1] * 0.98, nAB + (len(srt) - nAB) * 0.5, "BA",
                color="white", fontsize=10, fontweight="bold",
                ha="right", va="center")
        ax.set_xlabel("time from chunk onset (s)")
        ax.set_ylabel("chunk (sorted by type)")
        ax.set_title(f"{'cd'[i]}.  Unit {u}, every chunk", fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (e) how much a rate code could carry, versus the units
    ax = _tidy(fig.add_subplot(gs[1, 0]))
    names = ["total E", "E of A", "E of B", "layer 2\nunit"]
    vals_full, vals_raw = [], []
    for r, store in ((res, vals_full), (res_raw, vals_raw)):
        E = r["E_test"]
        wins = chunk_windows(r["test"], pad_s=0.05)
        lb = r["test"]["labels"]
        eA = np.array([E[0, a:b].sum() for a, b in wins])
        eB = np.array([E[1, a:b].sum() for a, b in wins])
        uu = r["resp"][:, np.flatnonzero(r["l2"].committed)[0]]
        for arr in (eA + eB, eA, eB, uu):
            store.append(_dprime(arr[lb == 0], arr[lb == 1]))
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, vals_full, w, color="#a3aebd", label="through layer 1")
    ax.bar(x + w / 2, vals_raw, w, color=C_ACC, label="raw stimulus")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(0, 400)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylabel("separability, d prime")
    ax.set_title("e.  What a rate code could carry", fontsize=10.5)
    ax.legend(fontsize=8.4, loc="upper left")
    for xi, v in zip(x - w / 2, vals_full):
        ax.text(xi, v * 1.25 + 0.06, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, vals_raw):
        ax.text(xi, v * 1.25 + 0.06, f"{v:.2f}", ha="center", fontsize=8)

    # (f) the coincidence map itself, as a trajectory
    ax = _tidy(fig.add_subplot(gs[1, 1]))
    E, s = res["E_test"], res["te"]["s"]
    wins = chunk_windows(test, pad_s=0.02)
    n_traj = 12
    for want, col in ((0, C_AB), (1, C_BA)):
        idx = np.flatnonzero(labels == want)[:n_traj]
        for j, k in enumerate(idx):
            a, b = wins[k]
            ax.plot(E[B, a:b] * s[A, a:b], E[A, a:b] * s[B, a:b],
                    color=col, lw=1.2, alpha=0.55,
                    label=f"{WORD_LABEL[want]} chunks" if j == 0 else None)
    lim = max(np.nanmax(E[B] * s[A]), np.nanmax(E[A] * s[B])) * 1.12
    ax.set_xlim(-lim * 0.04, lim); ax.set_ylim(-lim * 0.04, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("D[B after A]"); ax.set_ylabel("D[A after B]")
    ax.set_title("f.  The coincidence map in a chunk", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")

    # (g) stability across seeds
    ax = _tidy(fig.add_subplot(gs[1, 2]))
    xs = np.arange(len(seed_acc))
    ax.bar(xs, seed_acc, color=C_ACC, width=0.55)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(len(xs) - 0.4, 0.52, "chance", fontsize=8.2, color=MUTED,
            ha="right", va="bottom")
    for i, (a_, n_) in enumerate(zip(seed_acc, seed_n)):
        ax.text(i, a_ + 0.035, f"{a_:.0%}", ha="center", fontsize=8.6,
                fontweight="bold")
        ax.text(i, 0.10, f"{n_}", ha="center", fontsize=10, color="white",
                fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in xs], fontsize=9)
    ax.set_xlabel("seed  (number inside the bar = units committed)", fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_ylabel("held out accuracy")
    ax.set_title("g.  Independent seeds", fontsize=10.5)

    # (h) the two units against each other, one point per chunk
    ax = _tidy(fig.add_subplot(gs[1, 3]))
    if len(comm) >= 2:
        for want, col in ((0, C_AB), (1, C_BA)):
            ax.scatter(res["resp"][labels == want, comm[0]],
                       res["resp"][labels == want, comm[1]],
                       s=18, color=col, alpha=0.55, lw=0,
                       label=f"{WORD_LABEL[want]} chunks")
        lim = res["resp"][:, comm[:2]].max() * 1.1
        ax.plot([0, lim], [0, lim], color=MUTED, ls=":", lw=1)
        ax.set_xlim(-lim * 0.03, lim); ax.set_ylim(-lim * 0.03, lim)
        ax.set_xlabel(f"unit {comm[0]} response")
        ax.set_ylabel(f"unit {comm[1]} response")
        ax.legend(fontsize=8.5, loc="upper center")
    ax.set_title("h.  One point per chunk, held out", fontsize=10.5)

    _caption(fig, gs[2, :],
             "c and d.  Every held out chunk is drawn rather than an average, so no trial is hidden by the mean.\n"
             "e.  A rate code is not quite empty through layer 1: adaptation makes the first tone of a chunk larger "
             "than the second, giving per channel energy a weak cue (d prime near 1). On the raw stimulus that cue is\n"
             "exactly zero, because AB and BA contain identical tones, and the very same units still separate the two "
             "perfectly. That is the cleanest statement that the layer reads order and not level.\n"
             "f.  The coincidence map leaves the origin along one axis for AB and the other for BA. "
             "h.  The two units are close to mutually exclusive on held out chunks.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    a1cfg = selective_inh(N=N_CHANNELS, **LAYER1)
    l2cfg = L2Config()

    print("[ layer2_syllable.visuals ]")
    res = train_and_test(l2cfg, a1cfg, mode="full", seed=0)
    print(f"  main run: {res['l2'].n_components} units committed, "
          f"accuracy {res['acc']:.1%}")
    fig_tape(res, a1cfg, str(out / "l2_tape.png"))

    res_raw = train_and_test(l2cfg, a1cfg, mode="raw", seed=0, record_every=0)
    seed_acc, seed_n = [], []
    for sd in range(5):
        r = train_and_test(l2cfg, a1cfg, mode="full", seed=sd, record_every=0)
        seed_acc.append(r["acc"]); seed_n.append(r["l2"].n_components)
        print(f"  seed {sd}: accuracy {r['acc']:6.1%}, "
              f"committed {r['l2'].n_components}")
    fig_skeptic(res, res_raw, seed_acc, seed_n, a1cfg, str(out / "l2_skeptic.png"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
