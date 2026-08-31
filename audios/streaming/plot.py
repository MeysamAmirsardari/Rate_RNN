"""The figures.  Log axes throughout, because the data are ratios."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .analyse import PAPER, boundary
from .config import Design

MARK = {30.0: ("D", "none"), 50.0: ("s", "k"), 70.0: ("^", "none")}
COL = {6.0: "#1f77b4", 9.0: "#2ca02c", 15.0: "#d62728"}


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure2(d: Design, cells: list[dict], path, show_paper: bool = True):
    """The replication, drawn the way Fig. 2 is drawn.

    Threshold against frequency separation, one line per sequence type, on a
    log axis from 1 to 100 ms.  The published points are behind ours in grey
    so the comparison is a look rather than an arithmetic exercise.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    xs = sorted({c["df_st"] for c in cells})

    if show_paper:
        for gap in (30.0, 50.0, 70.0, None):
            y = [PAPER.get((x, gap)) for x in xs]
            if all(v is not None for v in y):
                ax.plot(xs, y, "-", color="0.75", lw=6, solid_capstyle="round",
                        zorder=1)
        ax.plot([], [], "-", color="0.75", lw=6, label="Elhilali 2009")

    for gap in (30.0, 50.0, 70.0):
        pts = sorted([c for c in cells
                      if not c["b_only"] and c["gap_a_ms"] == gap],
                     key=lambda c: c["df_st"])
        if not pts:
            continue
        m, fc = MARK[gap]
        x = [c["df_st"] for c in pts]
        y = [c["threshold_ms"] for c in pts]
        lo = [max(1e-3, c["threshold_ms"] - c["lo"]) for c in pts]
        hi = [max(1e-3, c["hi"] - c["threshold_ms"]) for c in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=m, color="k", mfc=fc, ms=7,
                    lw=1, capsize=3, ls="-", zorder=3,
                    label=f"{gap:.0f} ms A gap"
                          + (" (synchronous)" if gap == 50.0 else ""))
    pts = sorted([c for c in cells if c["b_only"]], key=lambda c: c["df_st"])
    if pts:
        ax.errorbar([c["df_st"] for c in pts],
                    [c["threshold_ms"] for c in pts],
                    yerr=[[max(1e-3, c["threshold_ms"] - c["lo"]) for c in pts],
                          [max(1e-3, c["hi"] - c["threshold_ms"]) for c in pts]],
                    fmt="x", color="k", ms=8, lw=1, capsize=3, ls="none",
                    zorder=3, label="B tones only")

    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.set_xticks(xs)
    ax.set_xlabel("$\\Delta F$ (semitones)")
    ax.set_ylabel("threshold (ms)")
    ax.set_title("asynchrony detection", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean(ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sweep(d: Design, cells: list[dict], path):
    """The question the paper only simulated.

    Threshold against how far the B stream has been slid, from synchronous
    to alternating.  Two references are drawn because both matter: the
    listener's own synchronous threshold, which is where one stream lives,
    and their B-only threshold if it was measured, which is where two
    streams live -- a listener who has segregated cannot use the A tones at
    all, so they should land on it.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    for df in sorted({c["df_st"] for c in cells if c["pct"] is not None}):
        pts = sorted([c for c in cells
                      if c["df_st"] == df and c["pct"] is not None],
                     key=lambda c: c["pct"])
        x = [c["lag_ms"] for c in pts]
        y = [c["threshold_ms"] for c in pts]
        lo = [max(1e-3, c["threshold_ms"] - c["lo"]) for c in pts]
        hi = [max(1e-3, c["hi"] - c["threshold_ms"]) for c in pts]
        col = COL.get(df, "k")
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=col, ms=5, lw=1.4,
                    capsize=3, label=f"{df:.0f} st", zorder=3)
        b = boundary(cells, df)
        if np.isfinite(b.get("lag_ms", np.nan)):
            ax.axvline(b["lag_ms"], color=col, ls=":", lw=1)
            ax.annotate(f"{b['lag_ms']:.0f} ms", (b["lag_ms"], ax.get_ylim()[1]),
                        color=col, fontsize=8, ha="center", va="top")
    for c in cells:
        if c["b_only"]:
            ax.axhline(c["threshold_ms"], color="0.6", ls="--", lw=1)
            ax.annotate("B only: two streams", (0, c["threshold_ms"]),
                        fontsize=8, color="0.4", va="bottom")
            break
    ax.set_yscale("log")
    ax.set_xlabel(f"lag of the B stream (ms)   "
                  f"0 = synchronous, {d.alternation_ms():.0f} = alternating")
    ax.set_ylabel("threshold (ms)")
    ax.set_title("how far the B stream can slide before it comes apart",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tracks(d: Design, runs: list[dict], trials: list[dict], path):
    """Every adaptive run, level against trial.

    A run that never settled, or that sat on the floor, or that walked away
    at the end, is visible here and nowhere else in the analysis.
    """
    cells = sorted({r["cell"] for r in runs})
    n = len(cells)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow),
                           sharex=True, sharey=True, squeeze=False,
                           constrained_layout=True)
    for a in ax.ravel()[n:]:
        a.set_visible(False)
    for a, cell in zip(ax.ravel(), cells):
        for run in sorted({int(t["run"]) for t in trials
                           if t["cell"] == cell}):
            ts = [t for t in trials if t["cell"] == cell
                  and int(t["run"]) == run]
            a.plot([float(t["dt_ms"]) for t in ts], lw=0.9, alpha=0.8)
        th = [r["threshold_ms"] for r in runs if r["cell"] == cell]
        if th:
            a.axhline(float(np.exp(np.mean(np.log(th)))), color="k", lw=1.2)
        a.set_yscale("log")
        a.set_ylim(d.dt_min_ms * 0.8, d.dt_max_ms * 1.3)
        a.axhline(d.dt_max_ms, color="0.8", lw=0.8, ls=":")
        a.axhline(d.dt_min_ms, color="0.8", lw=0.8, ls=":")
        a.set_title(cell, fontsize=8, loc="left")
        _clean(a)
    for a in ax[-1]:
        a.set_xlabel("trial")
    for a in ax[:, 0]:
        a.set_ylabel("dT (ms)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stimulus(d: Design, path, *, df_st: float = 6.0):
    """What the sequences look like, drawn from the real onsets."""
    from .stimulus import onsets_ms
    if d.mode == "sweep":
        rows = [(f"{p:.0f} %", d.sweep_gap_ms, d.lag_ms(p))
                for p in d.sweep_pct]
    else:
        rows = [(f"A gap {g:.0f} ms", g, 0.0) for g in d.gap_a_ms]
    fig, ax = plt.subplots(len(rows), 1, figsize=(8.4, 1.05 * len(rows)),
                           sharex=True, sharey=True, constrained_layout=True)
    ax = np.atleast_1d(ax)
    for a, (name, gap, lag) in zip(ax, rows):
        for dt, col, off in ((0.0, "k", 0.0),):
            aa, bb = onsets_ms(d, gap_a_ms=gap, lag_ms=lag, dt_ms=dt, sign=1)
            for t in aa:
                a.add_patch(plt.Rectangle((t, 0.12), d.tone, 0.22,
                                          color="0.35"))
            for i, t in enumerate(bb):
                a.add_patch(plt.Rectangle((t, 0.62), d.tone, 0.22,
                                          color="#d62728" if i == len(bb) - 1
                                          else "0.35"))
        a.set_ylim(0, 1)
        a.set_yticks([0.23, 0.73])
        a.set_yticklabels(["A", "B"], fontsize=8)
        a.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        _clean(a)
    ax[-1].set_xlim(0, max(d.interval_ms(g) for g in
                           (d.gap_a_ms if d.mode == "replicate"
                            else (d.sweep_gap_ms,))))
    ax[-1].set_xlabel("time (ms).  the target B tone is the red one")
    fig.savefig(path, dpi=150)
    plt.close(fig)
