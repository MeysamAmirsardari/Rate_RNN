"""
tasks.sfg2.sfg2
===============

Stochastic Figure-Ground (SFG) experiment driver for model0.

Paradigm (see ``stimulus.py`` for the faithful generator).  One
PRESENTATION = pre (5 s, no figure) / figure (5 s, coherent chord) /
post (5 s, no figure), bracketed by 0.5 s silences.  A SESSION repeats
the presentation ``n_reps`` times with the FIGURE FROZEN (same onsets +
channels every time) and the GROUND drawn FRESH every presentation.

Two measures, no STRFs:

  (A) WEIGHTS.  The plastic E->E matrix W, grouped into
        W_FF  figure -> figure   (coherent, should grow)
        W_GG  ground -> ground   (incoherent, should stay flat)
        W_FG  figure <-> ground  (cross)
      tracked over the session and as a function of figure size
      (4 / 6 / 8 / 10 coherent channels).

  (B) EXCITATION / INHIBITION decomposition.  model0 exposes, per E
      channel and per ms, the three currents that sum to its net input
        tm_in    thalamic feedforward excitation   (A_TC * U * x * s)
        rec_E    recurrent excitation               (W @ E)   <- the binder
        inh_to_E inhibition                          (M_IE @ I)
        net      = tm_in + rec_E - inh_to_E
      We average each over the figure-channel set vs. the ground-channel
      set, per epoch (pre / figure / post), per presentation.  The
      signature of binding is that during the FIGURE epoch the figure
      channels' RECURRENT excitation (and net drive) rises above the
      ground channels' as W_FF accumulates, while the PRE epoch
      (no figure) stays balanced -- a within-presentation control.

Mechanism (model0 terms)
------------------------
The coherent chord makes the figure channels co-fire on the same ms.
Their eligibility traces overlap, so Hebbian rate-STDP grows W_FF; the
recurrent term W @ E then feeds figure activity back onto figure
channels.  Ground channels fire incoherently, their traces rarely
coincide, W_GG is held near zero by decay, and per-channel inhibition
keeps ground net drive flat.

Figures (simple, testing-style titles; figure = green, ground = blue-gray):
  sfg2_stim.png            one presentation (raster + marginals)
  sfg2_weights_{preset}.png   W_FF/W_GG/W_FG over session + vs figure size
  sfg2_currents_{preset}.png  4 currents, figure vs ground, per presentation
  sfg2_balance_{preset}.png   recurrent gap early/late + net contrast vs size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- run-as-script / run-as-module support --------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, INH_PRESETS, simulate

try:
    from .stimulus import (build_session, N_CHANNELS, N_FIG_OPTS, DT_MS,
                           TONE_DUR_MS, PRE_MS, FIGURE_MS, POST_MS, FREQS_HZ)
except ImportError:                                   # run as a bare script
    from stimulus import (build_session, N_CHANNELS, N_FIG_OPTS, DT_MS,
                          TONE_DUR_MS, PRE_MS, FIGURE_MS, POST_MS, FREQS_HZ)


# =====================================================================
#  Run
# =====================================================================
def run_sfg2(cfg: A1Config, n_fig: int, n_reps: int,
             base_seed: int = 0, fig_seed: int = 12345,
             record_W_dt: float = 0.25, seed: int = 7) -> Dict:
    """Build an SFG session (figure size ``n_fig``) and run model0.

    Returns the ``simulate`` output dict augmented with ``stim``,
    ``sess`` and the figure / ground channel indices.
    """
    stim, sess = build_session(n_fig, n_reps, base_seed=base_seed,
                               fig_seed=fig_seed, with_silence=True)
    if cfg.N != N_CHANNELS:
        raise ValueError(f"cfg.N={cfg.N} but SFG stimulus has {N_CHANNELS} channels")
    snap_every = max(1, int(round(record_W_dt / cfg.dt)))
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(stim=stim, sess=sess,
               fig_idx=sess["fig_idx"], gnd_idx=sess["gnd_idx"])
    return out


# =====================================================================
#  Analysis
# =====================================================================
def compute_W_groups(W: np.ndarray, fig_idx: np.ndarray, gnd_idx: np.ndarray):
    """Mean off-diagonal figure->figure, ground->ground, figure<->ground W."""
    N = W.shape[0]
    eye = np.eye(N, dtype=bool)
    mask_FF = np.zeros((N, N), dtype=bool); mask_FF[np.ix_(fig_idx, fig_idx)] = True
    mask_GG = np.zeros((N, N), dtype=bool); mask_GG[np.ix_(gnd_idx, gnd_idx)] = True
    mask_FG = np.zeros((N, N), dtype=bool)
    mask_FG[np.ix_(fig_idx, gnd_idx)] = True
    mask_FG[np.ix_(gnd_idx, fig_idx)] = True
    mask_FF &= ~eye
    mask_GG &= ~eye
    return np.array([W[mask_FF].mean(), W[mask_GG].mean(), W[mask_FG].mean()])


_EPOCHS = ("pre", "figure", "post")


def epoch_currents(out: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Mean current per epoch, per presentation, for figure vs ground sets.

    Returns ``{current: {'fig': (n_reps, 3), 'gnd': (n_reps, 3)}}`` where
    the 3 columns are the (pre, figure, post) epochs and ``current`` is
    one of tm / rec / inh / net.  Means are taken over the full epoch
    window (all ms) and over the channel set.
    """
    sess = out["sess"]
    fig_idx, gnd_idx = sess["fig_idx"], sess["gnd_idx"]
    reps = sess["reps"]
    nR = len(reps)

    arrays = {
        "tm":  out["tm_in"],
        "rec": out["rec_E"],
        "inh": out["inh_to_E"],
        "net": out["tm_in"] + out["rec_E"] - out["inh_to_E"],
    }
    res = {k: {"fig": np.zeros((nR, 3)), "gnd": np.zeros((nR, 3))}
           for k in arrays}
    for r, rep in enumerate(reps):
        for e, ep in enumerate(_EPOCHS):
            lo, hi = rep["bounds"][ep]
            for k, arr in arrays.items():
                res[k]["fig"][r, e] = arr[fig_idx, lo:hi].mean()
                res[k]["gnd"][r, e] = arr[gnd_idx, lo:hi].mean()
    return res


def summarize_run(out: Dict) -> Dict:
    """Extract the small structures the plots need; drop the big histories."""
    fig_idx, gnd_idx = out["fig_idx"], out["gnd_idx"]
    traj = np.array([compute_W_groups(Wk, fig_idx, gnd_idx)
                     for Wk in out["W_traj"]])           # (S, 3) FF,GG,FG
    return dict(
        W_t=np.asarray(out["W_t"]),
        traj=traj,
        final=compute_W_groups(out["W_final"], fig_idx, gnd_idx),  # (3,)
        ec=epoch_currents(out),
        n_reps=out["sess"]["n_reps"],
        n_fig=out["sess"]["n_fig"],
    )


# =====================================================================
#  Plotting style
# =====================================================================
COL_FIG = "#2CA02C"      # figure  -- green
COL_GND = "#4D75BF"      # ground  -- blue-gray
COL_CTL = "0.60"         # control / cross


def _ax(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:  ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(fontsize=8.5, frameon=False)


def _epoch_spans(ax, sess, rep=0):
    """Shade pre / figure / post epochs of one presentation (in seconds)."""
    b = sess["reps"][rep]["bounds"]
    start = sess["reps"][rep]["start"]
    dt = DT_MS / 1000.0
    shades = {"pre": "0.93", "figure": (0.18, 0.63, 0.18, 0.10), "post": "0.93"}
    for ep in _EPOCHS:
        lo, hi = b[ep]
        ax.axvspan((lo - start) * dt, (hi - start) * dt,
                   color=shades[ep], zorder=0)


# =====================================================================
#  Figure 1 -- stimulus sanity (one presentation)
# =====================================================================
def plot_stimulus(fname: str, n_fig: int = 10):
    """Raster of one SFG presentation + the two activity marginals."""
    stim, sess = build_session(n_fig, n_reps=1, with_silence=True)
    dt = DT_MS / 1000.0
    fig_set = set(int(c) for c in sess["fig_idx"])
    T = stim.shape[1]

    fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0])
    fig.suptitle("SFG stimulus (one presentation)",
                 fontsize=13, fontweight="bold")

    # ---- raster ----
    ax = fig.add_subplot(gs[0, :])
    _epoch_spans(ax, sess)
    for ch in fig_set:                                   # highlight figure rows
        ax.axhspan(ch - 0.5, ch + 0.5, color=COL_FIG, alpha=0.10, zorder=0.5)
    for ch in range(N_CHANNELS):
        row = (stim[ch] > 0).astype(int)
        starts = np.where(np.diff(np.r_[0, row]) == 1)[0]
        col = COL_FIG if ch in fig_set else COL_GND
        for s in starts:
            ax.barh(ch, TONE_DUR_MS / 1000.0, left=s * dt, height=0.8,
                    color=col, edgecolor="none", zorder=2)
    ax.set_xlim(0, T * dt)
    ax.set_ylim(N_CHANNELS - 0.5, -0.5)
    ax.set_yticks(range(0, N_CHANNELS, 4))
    # epoch labels inside the top of each shaded region (white bbox so they
    # stay readable over the pips; no collision with the axis title)
    b = sess["reps"][0]["bounds"]; st = sess["reps"][0]["start"]
    for ep, lab in (("pre", "pre (no figure)"), ("figure", "FIGURE"),
                    ("post", "post (no figure)")):
        lo, hi = b[ep]
        ax.text(((lo + hi) / 2 - st) * dt, 0.2, lab, ha="center", va="top",
                fontsize=9, color=(COL_FIG if ep == "figure" else "0.4"),
                fontweight="bold" if ep == "figure" else "normal", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL_FIG, label="figure channels / chord"),
                       Patch(color=COL_GND, label="ground cloud")],
              fontsize=8.5, frameon=False, loc="lower right", ncol=2)
    _ax(ax, title="Raster", xlabel="time (s)", ylabel="channel")

    # ---- time marginal (per channel) ----
    ax = fig.add_subplot(gs[1, 0])
    tm = stim.mean(axis=1)
    colors = [COL_FIG if ch in fig_set else COL_GND for ch in range(N_CHANNELS)]
    ax.bar(range(N_CHANNELS), tm, color=colors, width=0.85)
    ax.axhline(tm.mean(), color="0.4", ls=":", lw=1.0)
    _ax(ax, title="Per-channel activity (time marginal)",
        xlabel="channel", ylabel="mean activity")

    # ---- population envelope over time ----
    ax = fig.add_subplot(gs[1, 1])
    bin_ms = 250
    nb = T // bin_ms
    pop = stim[:, :nb * bin_ms].mean(axis=0).reshape(nb, bin_ms).mean(axis=1)
    bc = (np.arange(nb) + 0.5) * bin_ms * dt
    _epoch_spans(ax, sess)
    ax.plot(bc, pop, color="0.20", lw=1.5)
    ax.axhline(pop.mean(), color="0.4", ls=":", lw=1.0)
    ax.set_ylim(0, pop.max() * 1.4)
    ax.set_xlim(0, T * dt)
    _ax(ax, title="Population envelope (250 ms bins)",
        xlabel="time (s)", ylabel="mean activity")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# =====================================================================
#  Figure 2 -- weights
# =====================================================================
def plot_weights(store: Dict[int, Dict], sizes: List[int], preset: str,
                 fname: str):
    """W_FF / W_GG / W_FG over the session (largest figure) + vs figure size."""
    big = max(sizes)
    s = store[big]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    fig.suptitle(f"E to E weights  [{preset}]", fontsize=12, fontweight="bold")

    # ---- left: trajectory (largest figure) ----
    ax = axes[0]
    W_t, traj = s["W_t"], s["traj"]
    ax.plot(W_t, traj[:, 0], color=COL_FIG, lw=2.6, label=r"$W_{F\to F}$ (figure)")
    ax.plot(W_t, traj[:, 1], color=COL_GND, lw=2.0, label=r"$W_{G\to G}$ (ground)")
    ax.plot(W_t, traj[:, 2], color=COL_CTL, lw=1.6, ls="--",
            label=r"$W_{F\leftrightarrow G}$ (cross)")
    _ax(ax, title=f"Over session ({big}-tone figure)",
        xlabel="time (s)", ylabel="mean W", legend=True)

    # ---- right: final weights vs figure size ----
    ax = axes[1]
    sz = np.array(sizes)
    FF = np.array([store[n]["final"][0] for n in sizes])
    GG = np.array([store[n]["final"][1] for n in sizes])
    FG = np.array([store[n]["final"][2] for n in sizes])
    ax.plot(sz, FF, "-o", color=COL_FIG, lw=2.2, ms=6, label=r"$W_{F\to F}$")
    ax.plot(sz, GG, "-o", color=COL_GND, lw=1.8, ms=5, label=r"$W_{G\to G}$")
    ax.plot(sz, FG, "--o", color=COL_CTL, lw=1.5, ms=4, label=r"$W_{F\leftrightarrow G}$")
    ax.set_xticks(sz)
    _ax(ax, title="Final weights vs figure size",
        xlabel="figure size (coherent channels)", ylabel="mean W", legend=True)

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# =====================================================================
#  Figure 3 -- excitation / inhibition currents
# =====================================================================
def plot_currents(store: Dict[int, Dict], sizes: List[int], preset: str,
                  fname: str):
    """Four currents, figure vs ground, across presentations (largest figure).

    Solid = the FIGURE epoch (figure present); dotted gray = the figure
    channels during the PRE epoch (no figure) -- a within-presentation
    control showing the elevation is figure-evoked, not channel bias.
    """
    big = max(sizes)
    ec = store[big]["ec"]
    nR = store[big]["n_reps"]
    reps = np.arange(1, nR + 1)

    panels = [("tm", "Thalamic excitation"), ("rec", "Recurrent excitation"),
              ("inh", "Inhibition"), ("net", "Net drive")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle(f"Synaptic currents: figure vs ground  ({big}-tone figure)  [{preset}]",
                 fontsize=12, fontweight="bold")

    for ax, (k, title) in zip(axes.ravel(), panels):
        fig_e = ec[k]["fig"][:, 1]          # figure channels, figure epoch
        gnd_e = ec[k]["gnd"][:, 1]          # ground channels, figure epoch
        fig_pre = ec[k]["fig"][:, 0]        # figure channels, pre epoch (control)
        if k == "net":
            ax.axhline(0, color="0.7", lw=0.7)
        ax.plot(reps, fig_e, color=COL_FIG, lw=2.2, label="figure ch (figure epoch)")
        ax.plot(reps, gnd_e, color=COL_GND, lw=2.0, label="ground ch (figure epoch)")
        ax.plot(reps, fig_pre, color=COL_CTL, lw=1.4, ls=":",
                label="figure ch (pre epoch, control)")
        _ax(ax, title=title, xlabel="presentation #",
            ylabel="mean current", legend=True)

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# =====================================================================
#  Figure 4 -- figure / ground balance
# =====================================================================
def plot_balance(store: Dict[int, Dict], sizes: List[int], preset: str,
                 fname: str):
    """Recurrent-excitation gap early vs late + net contrast vs figure size."""
    big = max(sizes)
    ec = store[big]["ec"]
    nR = store[big]["n_reps"]
    K = max(1, min(10, nR // 4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    fig.suptitle(f"Figure / ground balance  [{preset}]",
                 fontsize=12, fontweight="bold")

    # ---- left: recurrent exc, figure vs ground, early vs late (figure epoch) ----
    ax = axes[0]
    rec = ec["rec"]
    early_fig = rec["fig"][:K, 1].mean(); late_fig = rec["fig"][-K:, 1].mean()
    early_gnd = rec["gnd"][:K, 1].mean(); late_gnd = rec["gnd"][-K:, 1].mean()
    x = np.array([0, 1]); w = 0.38
    ax.bar(x - w / 2, [early_fig, early_gnd], w, color=[COL_FIG, COL_GND],
           alpha=0.45, label=f"early (first {K})")
    ax.bar(x + w / 2, [late_fig, late_gnd], w, color=[COL_FIG, COL_GND],
           label=f"late (last {K})")
    ax.set_xticks(x); ax.set_xticklabels(["figure ch", "ground ch"])
    _ax(ax, title=f"Recurrent excitation ({big}-tone, figure epoch)",
        ylabel="mean recurrent current", legend=True)

    # ---- right: late net contrast (figure - ground) vs figure size ----
    ax = axes[1]
    sz = np.array(sizes)
    contrast = np.array([
        (store[n]["ec"]["net"]["fig"][-K:, 1].mean()
         - store[n]["ec"]["net"]["gnd"][-K:, 1].mean())
        for n in sizes])
    ax.bar(sz, contrast, width=1.2, color=COL_FIG, alpha=0.85)
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xticks(sz)
    _ax(ax, title=f"Net drive contrast (last {K} presentations)",
        xlabel="figure size (coherent channels)",
        ylabel="figure - ground net drive")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# =====================================================================
#  Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="SFG (sfg2) experiment for model0")
    ap.add_argument("--reps", type=int, default=100,
                    help="presentations per session (default 100)")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(N_FIG_OPTS),
                    help="figure sizes to sweep (default 4 6 8 10)")
    ap.add_argument("--presets", nargs="+", default=list(INH_PRESETS),
                    help="inhibition presets (default selective uniform)")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--fig-seed", type=int, default=12345)
    args = ap.parse_args()

    sizes = sorted(args.sizes)

    print("[ SFG stimulus sanity ]")
    plot_stimulus("sfg2_stim.png", n_fig=max(sizes))

    for preset in args.presets:
        factory = INH_PRESETS[preset]
        cfg = factory(N=N_CHANNELS)

        print(f"\n========================================================")
        print(f"[ SFG2 -- inhibition preset '{preset}'  "
              f"(N={N_CHANNELS}, {args.reps} presentations) ]")
        print(f"  w_EI = (self {cfg.w_EI_self}, lat {cfg.w_EI_lat}); "
              f"w_IE = (self {cfg.w_IE_self}, lat {cfg.w_IE_lat})")

        store: Dict[int, Dict] = {}
        for n_fig in sizes:
            out = run_sfg2(cfg, n_fig, args.reps,
                           base_seed=args.base_seed, fig_seed=args.fig_seed)
            store[n_fig] = summarize_run(out)
            FF, GG, FG = store[n_fig]["final"]
            print(f"  n_fig={n_fig:2d}:  W_FF={FF:.4f}  W_GG={GG:.4f}  "
                  f"W_FG={FG:.4f}   FF/GG={FF/(GG+1e-9):.1f}x")
            del out                                       # free big histories

        print("[ Plotting ]")
        plot_weights(store, sizes, preset, f"sfg2_weights_{preset}.png")
        plot_currents(store, sizes, preset, f"sfg2_currents_{preset}.png")
        plot_balance(store, sizes, preset, f"sfg2_balance_{preset}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
