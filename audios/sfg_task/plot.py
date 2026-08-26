"""Figures: what the stimulus looks like, and that the two intervals match."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Design
from .stimulus import make_pool, trial

RED = "#E8121A"


def raster_ax(d: Design, pl: dict, sch: dict, ax, seconds: float | None = None,
              bg: str = "k", ms: float = 2.6):
    """One interval: every tone a dot, the figure's tones in red."""
    t = sch["slot"] * d.hop_ms / 1000.0
    f = pl["st"][sch["chan"]]
    g = sch["is_fig"]
    ax.plot(t[~g], f[~g], "s", ms=ms, color=bg, mec="none")
    ax.plot(t[g], f[g], "s", ms=ms, color=RED, mec="none")
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


def _bare(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def colours(steps):
    u = sorted(set(steps))
    return dict(zip(u, plt.cm.viridis(np.linspace(0, .88, len(u)))))


def accuracy(summary, fit, path: Path, marks=None, chance: float = .5,
             title: str = "", note: str = "") -> Path:
    """Accuracy against delay, on its own, as large and plain as it goes.

    This is the figure the experiment is for, so it carries nothing that is
    not the answer: the seven points, what they mean, and how sure they are.
    """
    g = summary[summary.variant == "rise"].sort_values("step_ms")
    col = colours(g.step_ms)
    x = g.step_ms.values
    pad = max(2.0, .04 * (x.max() - x.min()))

    fig, ax = plt.subplots(figsize=(6.8, 4.9), constrained_layout=True)
    ax.axhspan(0, chance, color="0.945", lw=0, zorder=0)
    ax.axhline(chance, color="0.45", lw=1.0, zorder=1)
    ax.annotate("chance", (x.min() - pad, chance), xytext=(3, 4),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8.5, color="0.45")

    if fit is not None and fit["trusted"]:
        z = np.linspace(x.min(), x.max(), 300)
        ax.plot(z, fit["curve"](z), "-", color=RED, lw=1.8, zorder=2)
        if np.isfinite(fit["ci"][0]):
            ax.axvspan(max(fit["ci"][0], x.min() - pad),
                       min(fit["ci"][1], x.max() + pad),
                       color=RED, alpha=.10, lw=0, zorder=1)
        ax.axvline(fit["step_at_d1"], color=RED, lw=1.1, ls="--", zorder=2)
        ax.annotate("d' = 1 at {:.0f} ms\n[{:.0f}, {:.0f}]".format(
            fit["step_at_d1"], *fit["ci"]), (fit["step_at_d1"], 1.045),
            xytext=(5, 0), textcoords="offset points", fontsize=9,
            color=RED, va="top")
    else:
        ax.plot(x, g.pc, "-", color="0.62", lw=1.3, zorder=2)

    ax.errorbar(x, g.pc, yerr=[g.pc - g.lo, g.hi - g.pc], fmt="none",
                ecolor="0.15", elinewidth=1.4, capsize=4, capthick=1.4,
                zorder=3)
    ax.scatter(x, g.pc, s=95, c=[col[s] for s in x], edgecolors="k",
               linewidths=1.0, zorder=4)
    for x_, y_, m in (marks or []):
        ax.annotate(m, (x_, y_), ha="center", va="bottom", fontsize=9,
                    color="0.3")

    lo = min(.35, float(g.lo.min()) - .04)
    ax.set_ylim(lo, 1.09)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_xticks(x)
    ax.set_xlabel("delay between successive figure tones (ms)", fontsize=11)
    ax.set_ylabel("proportion correct", fontsize=11)
    if title:
        ax.set_title(title, fontsize=11.5, loc="left", pad=22)
    if note:
        ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", va="bottom", fontsize=8.5,
                    color="0.42")
    _bare(ax)
    ax.tick_params(labelsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def psychometric(summary, fit, path: Path, chance: float = .5,
                 marks=None) -> Path:
    """Proportion correct and d' against step, with the fit and its CI."""
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True)
    col = colours(summary.step_ms)
    for v, g in summary.groupby("variant"):
        main = v == "rise"
        err = [(g.pc - g.lo).values, (g.hi - g.pc).values]
        ax[0].errorbar(g.step_ms, g.pc, yerr=err, fmt="none", capsize=3,
                       lw=1.2, ecolor="k" if main else "0.6")
        ax[0].scatter(g.step_ms, g.pc, s=48, zorder=3, label=v,
                      c=[col[s] for s in g.step_ms] if main else "0.6",
                      edgecolors="k", linewidths=.8,
                      marker="o" if main else "s")
        ax[1].errorbar(g.step_ms, g.dprime, yerr=g.se_dprime, fmt="none",
                       capsize=3, lw=1.2, ecolor="k" if main else "0.6")
        ax[1].scatter(g.step_ms, g.dprime, s=48, zorder=3, label=v,
                      c=[col[s] for s in g.step_ms] if main else "0.6",
                      edgecolors="k", linewidths=.8,
                      marker="o" if main else "s")
    xs = np.sort(summary.step_ms.unique())
    pad = max(2.0, .04 * (xs.max() - xs.min()))
    if fit and fit["trusted"]:
        z = np.linspace(xs.min(), xs.max(), 300)
        ax[0].plot(z, fit["curve"](z), "-", color=RED, lw=1.8, zorder=1)
        ax[0].axhline(fit["pc_at_d1"], color="0.8", lw=.8, ls=":")
        ax[1].axhline(1.0, color="0.8", lw=.8, ls=":")
        if np.isfinite(fit["ci"][0]):
            ax[0].axvspan(max(fit["ci"][0], xs.min() - pad),
                          min(fit["ci"][1], xs.max() + pad),
                          color=RED, alpha=.10, lw=0)
        ax[0].axvline(fit["step_at_d1"], color=RED, lw=1.0, ls="--")
        ax[0].set_title("d' = 1 at {:.0f} ms [{:.0f}, {:.0f}] 95% CI".format(
            fit["step_at_d1"], *fit["ci"]), fontsize=10)
    elif fit:
        ax[0].set_title("no usable threshold: " + fit["why"][0],
                        fontsize=9, color=RED, wrap=True)
    for x_, y_, m in (marks or []):
        ax[0].annotate(m, (x_, y_), ha="center", va="bottom", fontsize=8.5,
                       color="0.35")
    ax[0].axhline(chance, color="0.55", lw=.9)
    ax[0].set_ylim(.3, 1.08)
    ax[0].set_ylabel("proportion correct")
    ax[1].set_ylabel("d'")
    ax[1].axhline(0, color="0.55", lw=.9)
    for a in ax:
        a.set_xlabel("delay between successive figure tones (ms)")
        a.set_xlim(xs.min() - pad, xs.max() + pad)
        a.set_xticks(xs)
        if summary.variant.nunique() > 1:
            a.legend(frameon=False, fontsize=8)
        _bare(a)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def timecourse(tc: dict, summary, path: Path, sessions=None,
               chance: float = .5) -> Path:
    """Accuracy against position in the experiment, one colour per delay,
    with each delay's overall accuracy alongside on the same axis."""
    cur = tc["curves"]
    steps = sorted(k for k in cur if k != "all")
    col = colours(steps)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True,
                           gridspec_kw=dict(width_ratios=[3.4, 1]),
                           constrained_layout=True)
    a = cur["all"]
    ax[0].fill_between(a["x"], a["pc"] - a["se"], a["pc"] + a["se"],
                       color="0.7", alpha=.45, lw=0)
    ax[0].plot(a["x"], a["pc"], color="k", lw=2.6, zorder=5, label="all")
    for s in steps:
        # no band per delay: seven of them overlapping is soup, and the
        # panel on the right carries the uncertainty properly
        ax[0].plot(cur[s]["x"], cur[s]["pc"], color=col[s], lw=1.8,
                   label=f"{s:g} ms")
    for b in ([] if sessions is None else list(sessions)):
        ax[0].axvline(b, color="0.55", lw=.9, ls="--")
    ax[0].axhline(chance, color="0.55", lw=.9)
    ax[0].set_xlabel("trial in the experiment")
    ax[0].set_ylabel("proportion correct")
    ax[0].set_xlim(a["x"][0], a["x"][-1])
    ax[0].set_ylim(.15, 1.05)
    ax[0].set_title("Gaussian smoothing {:.0f} trials wide; the grey band is "
                    "1 SE on the effective N".format(tc["bw"]), fontsize=9)
    ax[0].legend(frameon=False, fontsize=8.5, ncols=8, loc="upper center",
                 bbox_to_anchor=(.5, -.13), columnspacing=1.2,
                 handlelength=1.4)

    g = summary[summary.variant == "rise"].sort_values("step_ms")
    ax[1].plot(g.step_ms, g.pc, "-", color="0.6", lw=1.0, zorder=1)
    ax[1].errorbar(g.step_ms, g.pc, yerr=[g.pc - g.lo, g.hi - g.pc],
                   fmt="none", ecolor="k", lw=1.2, capsize=3, zorder=2)
    ax[1].scatter(g.step_ms, g.pc, s=52, c=[col[s] for s in g.step_ms],
                  edgecolors="k", linewidths=.8, zorder=3)
    ax[1].axhline(chance, color="0.55", lw=.9)
    ax[1].set_xlabel("delay (ms)")
    ax[1].set_title("whole experiment", fontsize=9)
    for a in ax:
        _bare(a)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def diagnostics(ck, cmp, d, path: Path) -> Path:
    """The checks, and response time against delay.

    Only the first row has 0.5 for a null.  The rest come in pairs, and the
    question about a pair is whether its two halves agree, so each pair is
    banded together and carries the p value for that comparison.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3),
                           gridspec_kw=dict(width_ratios=[1.7, 1]),
                           constrained_layout=True)
    y = np.arange(len(ck))[::-1]
    bias = ck.check.str.startswith("said")
    pairs = [(i, i + 1) for i in range(int(bias.sum()), len(ck) - 1, 2)]
    for n, (i, j) in enumerate(pairs):
        ax[0].axhspan(y[j] - .5, y[i] + .5, color="0.93" if n % 2 else "0.97",
                      lw=0, zorder=0)
    ax[0].errorbar(ck.value, y, xerr=[ck.value - ck.lo, ck.hi - ck.value],
                   fmt="o", ms=5, color="k", capsize=3, lw=1.2, zorder=3)
    ax[0].plot([.5, .5], [y[bias.values].min() - .5, y[0] + .5],
               color=RED, lw=1.2, ls="--", zorder=2)
    ax[0].annotate("chance", (.5, y[0] + .55), color=RED, fontsize=8,
                   ha="center", va="bottom")
    look = {r.comparison: r.p for r in cmp.itertuples()}
    for i, j in pairs:
        key = f"{ck.check[i]} vs {ck.check[j]}"
        if key in look:
            pv = look[key]
            ax[0].annotate(f"p = {pv:.2f}" + ("  *" if pv < .05 else ""),
                           (1.005, (y[i] + y[j]) / 2), xycoords=("axes fraction",
                                                                 "data"),
                           va="center", fontsize=8,
                           color=RED if pv < .05 else "0.35")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(ck.check)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(y.min() - .6, y.max() + 1.1)
    ax[0].set_xlabel("proportion, with 95% CI")
    ax[0].set_title("the top row should sit at chance; each pair below "
                    "should agree with itself", fontsize=9)

    col = colours(d.step_ms)
    for s_, g in d.groupby("step_ms"):
        q = np.percentile(g.rt, [25, 50, 75])
        ax[1].errorbar(s_, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]],
                       fmt="none", ecolor="k", capsize=3, lw=1.2)
        ax[1].plot(s_, q[1], "o", ms=8, mfc=col[s_], mec="k", mew=.8, zorder=3)
    ax[1].set_xlabel("delay between successive figure tones (ms)")
    ax[1].set_ylabel("response time (s), median and IQR")
    ax[1].set_title("harder trials should take longer", fontsize=9)
    for a_ in ax:
        _bare(a_)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def group_curves(per_subject, path: Path, chance: float = .5) -> Path:
    """One thin line per subject and the mean with its SEM, which is the
    level a claim about listeners lives at."""
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True)
    steps = sorted({s for r in per_subject for s in r["summary"].step_ms})
    M = np.full((len(per_subject), len(steps)), np.nan)
    for i, r in enumerate(per_subject):
        g = r["summary"][r["summary"].variant == "rise"]
        ax[0].plot(g.step_ms, g.pc, "-", color="0.72", lw=1.0, zorder=1)
        for j, s in enumerate(steps):
            hit = g[g.step_ms == s]
            if len(hit):
                M[i, j] = hit.pc.values[0]
    m = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(M), axis=0))
    ax[0].errorbar(steps, m, yerr=sem, fmt="o-", color="k", lw=2, ms=6,
                   capsize=3, zorder=3, label=f"mean of {len(per_subject)}")
    ax[0].axhline(chance, color="0.55", lw=.9)
    ax[0].set_ylim(.3, 1.05)
    ax[0].set_ylabel("proportion correct")
    ax[0].set_xlabel("delay between successive figure tones (ms)")
    ax[0].legend(frameon=False, fontsize=8)

    th = np.array([r["threshold"] for r in per_subject], float)
    th = th[np.isfinite(th)]
    ax[1].plot(np.random.default_rng(0).normal(0, .04, th.size), th, "o",
               ms=7, mfc="0.75", mec="k", mew=.8)
    if th.size:
        ax[1].errorbar([0], [th.mean()],
                       yerr=[th.std(ddof=1) / np.sqrt(th.size)],
                       fmt="_", ms=34, color=RED, lw=2.2, capsize=8)
    ax[1].set_xlim(-.35, .35)
    ax[1].set_xticks([])
    ax[1].set_ylabel("delay at which d' = 1 (ms)")
    ax[1].set_title("each subject, mean and SEM", fontsize=9)
    for a in ax:
        _bare(a)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
