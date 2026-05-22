"""
sfg.py
======

Stochastic Figure-Ground (SFG) paradigm for model0.

Reference paradigm: Teki, Chait, Kumar, von Kriegstein & Griffiths
(2011) J. Neurosci.; the baphy fgFrozen figure-ground sound objects.

Stimulus
--------
A tone cloud on N tonotopic channels.

  Ground.  Random pulses are placed by GLOBAL stratified sampling: all
  random onset times are spread one per [0,T]/n sub-interval (jittered
  within), then each onset is assigned to a channel.  Onsets are random
  and NOT quantised to a grid; the population onset rate is uniform in
  time (no bursts, no silent gaps); and because onset times are
  distinct, no two ground pulses start together -- no vertical-line
  alignment.

  Figure.  A fixed subset of channels (the figure channels) additionally
  fire COHERENTLY: a pulse on every figure channel, aligned, once per
  figure period.  This regular synchronous repetition is what makes the
  figure perceptually segregate from the ground.

  Figure channels also carry ground: besides their coherent pulses they
  receive stratified random pulses too, exactly like ground channels.

Every channel receives the same total number of pulses P, so the
time-marginal (activity averaged over time, per channel) is exactly
flat.  Stratified onsets keep the freq-marginal (activity averaged over
channels, per time bin) near-uniform -- the only structured component
is the figure's own coherent repetition.

Model: model0 with the AB/BA-task configuration, N overridden to 18.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Run-as-script support.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, simulate

# ---- paradigm defaults --------------------------------------------------------
N_CHANNELS    = 18
FIG_IDX       = np.array([1, 4, 7, 10, 13, 16])   # 6 figure channels, spaced
PULSE_DUR     = 25e-3
FIG_PERIOD    = 250e-3                             # coherent figure repeat period
P_PER_CHANNEL = 28                                 # total pulses per channel
T_DEFAULT     = 5.0                                # one SFG presentation


# =====================================================================
#  Stimulus
# =====================================================================
def build_sfg_stim(
    cfg: A1Config,
    fig_idx: np.ndarray,
    T: float = T_DEFAULT,
    pulse_dur: float = PULSE_DUR,
    fig_period: float = FIG_PERIOD,
    P: int = P_PER_CHANNEL,
    tone_amp: float = 1.0,
    seed: int = 7,
) -> Tuple[np.ndarray, dict]:
    """Build the SFG tone cloud.

    Coherent figure pulses are placed on a regular grid (period
    ``fig_period``).  Every other pulse is placed by stratified sampling
    -- one pulse per [0,T]/need sub-interval, at a random position
    inside it -- so onsets are un-gridded but evenly spread.  Each
    channel ends with exactly ``P`` pulses.

    Returns (stim (N, T_steps), meta).
    """
    rng = np.random.default_rng(seed)
    N, dt = cfg.N, cfg.dt

    n_pulse = int(round(pulse_dur / dt))
    T_steps = int(round(T / dt))
    max_on  = T_steps - n_pulse

    fig_idx = np.asarray(fig_idx, dtype=int)
    gnd_idx = np.setdiff1d(np.arange(N), fig_idx)
    is_fig  = np.zeros(N, dtype=bool); is_fig[fig_idx] = True

    # coherent figure onsets: a regular grid -- the temporally coherent figure
    fig_period_steps = int(round(fig_period / dt))
    coh_onsets = np.arange(0, max_on + 1, fig_period_steps)
    n_coh = len(coh_onsets)
    if P < n_coh:
        raise ValueError(f"P={P} below the coherent-pulse count {n_coh}")

    stim = np.zeros((N, T_steps))
    onsets = [[] for _ in range(N)]                # placed onsets per channel

    # 1. coherent figure pulses (aligned grid)
    for ch in fig_idx:
        for on in coh_onsets:
            stim[ch, on:on + n_pulse] = tone_amp
        onsets[ch] = list(coh_onsets)

    # 2. random pulses -- GLOBAL stratified placement.  All random onset
    #    *times* are spread one per [0, max_on]/n_random sub-interval
    #    (jittered within), so the population onset rate is uniform in
    #    time.  Each onset is then assigned to a channel (quota-weighted,
    #    overlap-checked): per-channel counts stay balanced and
    #    per-channel onsets are random and un-gridded.  Because the
    #    onset times are distinct, no two ground pulses start together --
    #    there is no vertical-line alignment.
    quota = np.full(N, P, dtype=int)
    for ch in fig_idx:
        quota[ch] -= n_coh                          # coherent pulses placed
    n_random = int(quota.sum())

    edges = np.linspace(0, max_on, n_random + 1)
    onset_times = np.sort(np.array(
        [int(rng.uniform(edges[k], edges[k + 1])) for k in range(n_random)]))

    for t_on in onset_times:
        elig = [ch for ch in range(N) if quota[ch] > 0 and
                all(abs(t_on - e) >= n_pulse for e in onsets[ch])]
        if not elig:                                # rare: relax overlap rule
            elig = [ch for ch in range(N) if quota[ch] > 0]
            if not elig:
                break
        w = np.array([quota[ch] for ch in elig], dtype=float)
        ch = int(rng.choice(elig, p=w / w.sum()))
        stim[ch, t_on:t_on + n_pulse] = tone_amp
        onsets[ch].append(int(t_on))
        quota[ch] -= 1

    per_channel = np.array([len(o) for o in onsets])
    meta = dict(
        coh_onsets=coh_onsets, n_coh=n_coh, gnd_idx=gnd_idx,
        n_pulse_steps=n_pulse, fig_period=fig_period, P=P,
        per_channel=per_channel,
    )
    return stim, meta


def stim_marginals(stim: np.ndarray, cfg: A1Config,
                   bin_dur: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_marginal, freq_marginal, bin_centres).

    time_marginal : (N,)        per-channel activity averaged over time
    freq_marginal : (n_bins,)   per-bin activity averaged over channels
    bin_centres   : (n_bins,)   bin centre times (s)

    The freq-marginal is binned at 250 ms (the figure period) so that
    each bin contains one coherent figure repetition: the curve then
    reflects how uniformly the stimulus is distributed across the trial,
    rather than the figure's own intended periodic ripple.
    """
    dt = cfg.dt
    time_marg = stim.mean(axis=1)
    n_bin = int(round(bin_dur / dt))
    n_bins = stim.shape[1] // n_bin
    pop = stim[:, : n_bins * n_bin].mean(axis=0)        # mean over channels
    freq_marg = pop.reshape(n_bins, n_bin).mean(axis=1)
    bin_centres = (np.arange(n_bins) + 0.5) * bin_dur
    return time_marg, freq_marg, bin_centres


# =====================================================================
#  Experiment
# =====================================================================
def run_sfg(
    cfg: Optional[A1Config] = None,
    fig_idx: Optional[np.ndarray] = None,
    T: float = T_DEFAULT,
    seed: int = 7,
) -> dict:
    """Build the SFG tone cloud and run model0 on it."""
    if cfg is None:
        cfg = A1Config(N=N_CHANNELS)
    if fig_idx is None:
        fig_idx = FIG_IDX

    stim, meta = build_sfg_stim(cfg, fig_idx, T=T, seed=seed)
    snap_every = max(1, int(round(0.05 / cfg.dt)))
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(stim=stim, fig_idx=np.asarray(fig_idx), T=T, **meta)
    return out


def compute_W_groups(W: np.ndarray, fig_idx: np.ndarray, gnd_idx: np.ndarray):
    """Mean off-diagonal figure-figure, ground-ground, figure<->ground W."""
    N = W.shape[0]
    eye = np.eye(N, dtype=bool)
    mask_FF = np.zeros((N, N), dtype=bool); mask_FF[np.ix_(fig_idx, fig_idx)] = True
    mask_GG = np.zeros((N, N), dtype=bool); mask_GG[np.ix_(gnd_idx, gnd_idx)] = True
    mask_FG = np.zeros((N, N), dtype=bool)
    mask_FG[np.ix_(fig_idx, gnd_idx)] = True
    mask_FG[np.ix_(gnd_idx, fig_idx)] = True
    mask_FF &= ~eye
    mask_GG &= ~eye
    return W[mask_FF].mean(), W[mask_GG].mean(), W[mask_FG].mean()


# =====================================================================
#  Plotting
# =====================================================================
COL_FIG = "#4ECB59"
COL_GND = "#4D75BF"


def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _draw_raster(ax, res, n_show):
    """Draw the stimulus raster up to sample ``n_show``.

    A pulse is green only if it is a coherent figure pulse (a figure
    channel firing on the coherent grid); every other pulse, including
    the random pulses figure channels also carry, is blue.
    """
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]
    fig_set = {int(c) for c in res["fig_idx"]}
    coh_set = {int(o) for o in res["coh_onsets"]}
    n_pulse = res["n_pulse_steps"]

    for ch in res["fig_idx"]:
        ax.axhspan(ch - 0.5, ch + 0.5, color=COL_FIG, alpha=0.10, zorder=0)
    for ch in range(cfg.N):
        starts = np.where(np.diff(np.concatenate(
            [[0], (stim[ch, :n_show] > 0).astype(int)])) == 1)[0]
        for s in starts:
            coh_fig = (ch in fig_set) and (int(s) in coh_set)
            ax.barh(ch, n_pulse * dt, left=s * dt, height=0.72,
                    color=(COL_FIG if coh_fig else COL_GND), edgecolor="none")
    ax.set_xlim(0, n_show * dt)
    ax.set_ylim(cfg.N - 0.5, -0.5)
    ax.set_yticks(range(cfg.N))


def plot_sfg_marginals(res: dict, fname: str):
    """Minimal check of the two stimulus marginals, each a single curve."""
    cfg = res["cfg"]
    tm, fm, bc = stim_marginals(res["stim"], cfg)
    N = cfg.N

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), constrained_layout=True)
    fig.suptitle("SFG stimulus marginals", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(range(N), tm, color="0.20", lw=1.5, marker="o", ms=3)
    ax.axhline(tm.mean(), color="0.65", ls=":", lw=1.0)
    ax.set_ylim(0, tm.max() * 1.4)
    _setup_axes(ax, title=f"time marginal (per channel), CV "
                          f"{tm.std()/tm.mean()*100:.1f}%",
                xlabel="channel", ylabel="mean activity")

    ax = axes[1]
    ax.plot(bc, fm, color="0.20", lw=1.5)
    ax.axhline(fm.mean(), color="0.65", ls=":", lw=1.0)
    ax.set_ylim(0, fm.max() * 1.4)
    _setup_axes(ax, title=f"freq marginal (per 250 ms bin), CV "
                          f"{fm.std()/fm.mean()*100:.1f}%",
                xlabel="time (s)", ylabel="mean activity")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_sfg_run(res: dict, fname: str):
    """Raster, mean activity (figure vs ground), W_FF/W_GG/W_FG evolution."""
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]; E = res["E"]
    fig_idx, gnd_idx = res["fig_idx"], res["gnd_idx"]
    t = res["t"]

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else None
    W_t = res["W_t"]

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.2, 1.4])
    fig.suptitle("Stochastic Figure-Ground (model0)",
                 fontsize=13, fontweight="bold")

    ax = fig.add_subplot(gs[0])
    n_show = stim.shape[1]
    _draw_raster(ax, res, n_show)
    _setup_axes(ax, title="Stimulus raster (coherent figure pulses green, "
                          "random pulses blue)",
                xlabel="time (s)", ylabel="channel")

    ax = fig.add_subplot(gs[1])
    ax.plot(t, E[gnd_idx].mean(0), color=COL_GND, lw=1.5, label="Ground")
    ax.plot(t, E[fig_idx].mean(0), color=COL_FIG, lw=1.5, label="Figure")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Mean excitatory rate (group average)",
                xlabel="time (s)", ylabel=r"$\langle E\rangle$")

    ax = fig.add_subplot(gs[2])
    if Wt is not None and len(W_t):
        FF = np.empty(len(W_t)); GG = np.empty(len(W_t)); FG = np.empty(len(W_t))
        for k in range(len(W_t)):
            FF[k], GG[k], FG[k] = compute_W_groups(Wt[k], fig_idx, gnd_idx)
        ax.plot(W_t, FF, color=COL_FIG, lw=2.4, label=r"$W_{F\to F}$")
        ax.plot(W_t, GG, color=COL_GND, lw=2.4, label=r"$W_{G\to G}$")
        ax.plot(W_t, FG, color="0.45", lw=1.6, ls="--",
                label=r"$W_{F\leftrightarrow G}$")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    _setup_axes(ax, title=r"Recurrent E$\to$E weight evolution (group mean)",
                xlabel="time (s)", ylabel="mean W")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_sfg_W(res: dict, fname: str):
    """Final W matrix, original and figure-block-reordered views."""
    fig_idx, gnd_idx = res["fig_idx"], res["gnd_idx"]
    W = res["W_final"]
    N = W.shape[0]
    Wmax = max(W.max(), 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    perm = np.concatenate([fig_idx, gnd_idx])
    im = axes[0].imshow(W[np.ix_(perm, perm)], cmap="viridis", origin="upper",
                        vmin=0, vmax=Wmax)
    axes[0].add_patch(plt.Rectangle((-0.5, -0.5), len(fig_idx), len(fig_idx),
                                    fill=False, edgecolor=COL_FIG, lw=3))
    axes[0].set_xticks(range(N)); axes[0].set_xticklabels(perm, fontsize=7)
    axes[0].set_yticks(range(N)); axes[0].set_yticklabels(perm, fontsize=7)
    axes[0].set_xlabel("pre"); axes[0].set_ylabel("post")
    axes[0].set_title("Reordered, figure block top-left",
                      fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(W, cmap="viridis", origin="upper", vmin=0, vmax=Wmax)
    axes[1].set_xticks(range(N)); axes[1].set_yticks(range(N))
    axes[1].tick_params(labelsize=7)
    axes[1].set_xlabel("pre"); axes[1].set_ylabel("post")
    axes[1].set_title("Original channel order", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    cfg = A1Config(N=N_CHANNELS)
    fig_idx = FIG_IDX

    print(f"[ Running SFG on model0, N={N_CHANNELS}, "
          f"{len(fig_idx)} figure channels, T={T_DEFAULT:.0f}s ]")
    res = run_sfg(cfg=cfg, fig_idx=fig_idx, T=T_DEFAULT, seed=7)

    tm, fm, _ = stim_marginals(res["stim"], cfg)
    pc = res["per_channel"]
    print(f"  pulses per channel:  min={pc.min()}  max={pc.max()}  "
          f"(target {res['P']})")
    print(f"  time-marginal CV:    {tm.std()/tm.mean()*100:.1f}%")
    print(f"  freq-marginal CV:    {fm.std()/fm.mean()*100:.1f}%  "
          f"(per 250 ms bin)")

    print("[ Plotting ]")
    plot_sfg_marginals(res, "sfg_m0_marginals.png")
    plot_sfg_run(res,       "sfg_m0_run.png")
    plot_sfg_W(res,         "sfg_m0_W.png")

    W = res["W_final"]
    W_FF, W_GG, W_FG = compute_W_groups(W, fig_idx, res["gnd_idx"])
    print(f"\nFinal weight statistics:")
    print(f"  mean W_F->F       = {W_FF:.4f}")
    print(f"  mean W_G->G       = {W_GG:.4f}")
    print(f"  mean W_F<->G      = {W_FG:.4f}")
    print(f"  ratio W_FF / W_GG = {W_FF/(W_GG+1e-9):.2f}x")
    print("Done.")


if __name__ == "__main__":
    main()
