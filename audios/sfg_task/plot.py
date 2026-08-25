"""Figures: what the stimulus looks like, and that the two intervals match."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Design
from .stimulus import make_pool, trial

RED = "#E8121A"


def raster_ax(d: Design, pl: dict, sch: dict, ax, seconds: float | None = None):
    """One interval: every tone a dot, the figure's tones in red."""
    t = sch["slot"] * d.hop_ms / 1000.0
    f = pl["st"][sch["chan"]]
    g = sch["is_fig"]
    ax.plot(t[~g], f[~g], "s", ms=2.6, color="k", mec="none")
    ax.plot(t[g], f[g], "s", ms=2.6, color=RED, mec="none")
    ax.set_xlim(0, seconds or d.interval_s)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def raster(d: Design, path: Path, steps=None, seed: int = 7,
           variant: str = "rise") -> Path:
    """One row per step, the two intervals of the same trial side by side."""
    steps = list(d.steps_ms if steps is None else steps)
    pl = make_pool(d)
    fig, ax = plt.subplots(len(steps), 2, figsize=(11, 1.9 * len(steps)),
                           sharex=True, sharey=True, squeeze=False,
                           constrained_layout=True)
    for row, step in zip(ax, steps):
        pair = trial(d, pl, step_ms=step, seed=seed, variant=variant)
        for a, sch, name in zip(row, pair, ("figure", "no figure")):
            raster_ax(d, pl, sch, a)
            if step == steps[0]:
                a.set_title(name, fontsize=10)
        row[0].set_ylabel(f"{step:g} ms\nst re {d.f_lo:.0f} Hz", fontsize=9)
    for a in ax[-1]:
        a.set_xlabel("Time (s)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def envelopes(res: list[dict], path: Path,
              win_ms: tuple[float, float] = (-150.0, 500.0)) -> Path:
    """The element-locked envelope of both intervals, per step.  They should
    lie on top of each other: the loudness pulse is not the cue."""
    fig, ax = plt.subplots(1, len(res), figsize=(2.5 * len(res), 2.8),
                           sharey=True, constrained_layout=True, squeeze=False)
    for a, r in zip(ax[0], res):
        t = np.linspace(*win_ms, r["epoch"][0].size)
        a.plot(t, r["epoch"][0], color=RED, lw=1.2, label="figure")
        a.plot(t, r["epoch"][1], color="k", lw=1.2, ls="--", label="no figure")
        a.axvline(0, color="0.7", lw=.8)
        a.set_title(f"{r['step_ms']:g} ms", fontsize=10)
        a.set_xlabel("Time re element (ms)")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0][0].set_ylabel("Envelope (dB re mean)")
    ax[0][-1].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def psychometric(summary, fit, path: Path, chance: float) -> Path:
    """Proportion correct and d' against step, with the fit."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for v, g in summary.groupby("variant"):
        main = v == "rise"
        ax[0].errorbar(g.step_ms, g.pc, yerr=g.se, fmt="o", ms=5,
                       color="k" if main else None,
                       mfc="k" if main else "none", capsize=2, label=v)
        ax[1].plot(g.step_ms, g.dprime, "o", ms=5, color="k" if main else None,
                   mfc="k" if main else "none", label=v)
    if fit:
        x = np.linspace(0, max(summary.step_ms) * 1.05, 200)
        ax[0].plot(x, fit["curve"](x), "-", color=RED, lw=1.6)
        for a, y in ((ax[0], fit["pc_at_d1"]), (ax[1], 1.0)):
            a.axhline(y, color="0.8", lw=.8, ls=":")
        ax[0].axvline(fit["step_at_d1"], color=RED, lw=.8, ls=":")
        ax[0].set_title(f"d' = 1 at {fit['step_at_d1']:.0f} ms "
                        f"[{fit['ci'][0]:.0f}, {fit['ci'][1]:.0f}]",
                        fontsize=10)
    ax[0].axhline(chance, color="0.6", lw=.8)
    ax[0].set_ylim(0.35, 1.02)
    ax[0].set_ylabel("Proportion correct")
    ax[1].set_ylabel("d'")
    ax[1].axhline(0, color="0.6", lw=.8)
    for a in ax:
        a.set_xlabel("Step between successive figure tones (ms)")
        a.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
