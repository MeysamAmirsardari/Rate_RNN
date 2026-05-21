"""
sfg.py
======

Stochastic Figure-Ground (SFG) paradigm for model0.

Mirrors sfg_rnn2.m but adapted to model0's tone-selective-inhibition
architecture and to two extra design constraints the user requested:

  (1) Every channel has (almost) the same total pulse count, so no
      channel dominates the all-channel mean by raw activity.
  (2) Every time-window has (almost) the same number of channels
      simultaneously active, so the stimulus doesn't have bursty
      time-periods and silent ones.

Stimulus design
---------------
- N = 12 channels.  fig_idx = [1, 4, 7, 10] (4 figure, 8 ground).
- T discretised into windows of length ``window`` (default 250 ms).
- Coherent windows (every other): the 4 figure channels fire together,
  + 2 random ground channels (constant density).
- Non-coherent windows: 6 random ground channels.
- Pulse onsets are aligned to the window start (25 ms pulse).

With ``coherent_every = 2`` and 8 ground channels: every channel
receives exactly the same number of pulses over the whole run.

Expected learning outcome
-------------------------
- Figure-figure E->E weights grow to ~W_max  (24 same-time co-firings
  per pair when T=12s, n_windows=48).
- Ground-ground weights stay smaller (random combinatorics of pair
  co-firings — same expected count, but spread across all pair indices
  rather than concentrated on the figure indices).
- Figure-ground weights stay near zero (figure and ground never fire
  in the same window).

Model & config: ``model0`` with the AB/BA-task config unchanged, except
``N`` is overridden to 12.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Allow `python tasks/sfg_model0/sfg.py` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, simulate


# =====================================================================
# Stimulus
# =====================================================================
def build_sfg_stim(
    cfg: A1Config,
    fig_idx: np.ndarray,
    T: float = 20.0,
    pulse_dur: float = 25e-3,
    window: float = 250e-3,
    K: int = 6,
    coherent_every: int = 2,
    tone_amp: float = 1.0,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the SFG stimulus matrix with equal per-channel pulse count
    and constant per-window channel density.

    Parameters
    ----------
    cfg : A1Config
    fig_idx : (n_fig,) ndarray
        Indices of the figure channels (must be in [0, cfg.N)).
    T : float
        Total simulation time (s).
    pulse_dur : float
        Width of each pulse (s).
    window : float
        Window length (s).  Pulse onset is aligned to window start.
    K : int
        Number of channels active per window (constant density).
    coherent_every : int
        Make every k-th window a "coherent" window (figure fires together).
    tone_amp : float
        Pulse amplitude.
    seed : int
        RNG seed for the ground-channel sub-selection.

    Returns
    -------
    stim : (N, T_steps) ndarray
    coherent_windows : (n_coh,) ndarray of window indices
    gnd_idx : (n_gnd,) ndarray of ground channel indices
    """
    rng = np.random.default_rng(seed)

    N = cfg.N
    dt = cfg.dt

    n_pulse_steps  = int(round(pulse_dur / dt))
    n_window_steps = int(round(window / dt))
    n_windows      = int(round(T / window))
    T_steps        = n_windows * n_window_steps

    fig_idx = np.asarray(fig_idx, dtype=int)
    gnd_idx = np.setdiff1d(np.arange(N), fig_idx)
    n_fig = len(fig_idx)
    n_gnd = len(gnd_idx)

    K_gnd_coh = K - n_fig          # ground slots in a coherent window
    K_gnd_nc  = K                   # ground slots in a non-coherent window

    if K_gnd_coh < 0 or K_gnd_coh > n_gnd or K_gnd_nc > n_gnd:
        raise ValueError(
            f"density K={K} incompatible with n_fig={n_fig}, n_gnd={n_gnd}"
        )

    stim = np.zeros((N, T_steps))
    coherent_windows = []

    for w in range(n_windows):
        t0 = w * n_window_steps
        t1 = t0 + n_pulse_steps
        is_coherent = (w % coherent_every == 0)

        if is_coherent:
            ground_chosen = rng.choice(gnd_idx, K_gnd_coh, replace=False)
            active = np.concatenate([fig_idx, ground_chosen])
            coherent_windows.append(w)
        else:
            active = rng.choice(gnd_idx, K_gnd_nc, replace=False)

        stim[active, t0:t1] = tone_amp

    return stim, np.array(coherent_windows, dtype=int), gnd_idx


def stim_summary(stim: np.ndarray, cfg: A1Config, window: float = 250e-3):
    """Verification helper: report per-channel pulses and per-window density."""
    N = stim.shape[0]
    dt = cfg.dt
    n_window_steps = int(round(window / dt))
    n_windows = stim.shape[1] // n_window_steps

    pulse_starts = np.diff(np.concatenate([np.zeros((N, 1)),
                                           (stim > 0).astype(int)], axis=1),
                           axis=1) == 1
    per_channel = pulse_starts.sum(axis=1)
    per_window  = np.zeros(n_windows, dtype=int)
    for w in range(n_windows):
        t0 = w * n_window_steps
        t1 = t0 + n_window_steps
        per_window[w] = ((stim[:, t0:t1] > 0).any(axis=1)).sum()
    return per_channel, per_window


# =====================================================================
# Experiment
# =====================================================================
def run_sfg(
    cfg: Optional[A1Config] = None,
    fig_idx: Optional[np.ndarray] = None,
    T: float = 20.0,
    seed: int = 7,
) -> dict:
    """Build SFG stimulus and run model0 on it."""
    if cfg is None:
        cfg = A1Config(N=12)
    if fig_idx is None:
        fig_idx = np.array([1, 4, 7, 10])

    stim, coh_windows, gnd_idx = build_sfg_stim(cfg, fig_idx, T=T, seed=seed)
    snap_every = max(1, int(round(0.05 / cfg.dt)))   # 50 ms
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(
        stim=stim,
        fig_idx=fig_idx,
        gnd_idx=gnd_idx,
        coh_windows=coh_windows,
        T=T,
    )
    return out


def compute_W_groups(W: np.ndarray, fig_idx: np.ndarray, gnd_idx: np.ndarray):
    """Mean off-diagonal weights for figure-figure, ground-ground, and
    figure<->ground sub-blocks of W."""
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
# Plotting
# =====================================================================
COL_FIG = "#4ECB59"
COL_GND = "#4D75BF"
COL_FG  = "#737373"


def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_sfg_run(res: dict, fname: str):
    """Main figure: raster, mean activity, W_FF/W_GG/W_FG evolution."""
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]
    E    = res["E"]
    fig_idx = res["fig_idx"]
    gnd_idx = res["gnd_idx"]
    t    = res["t"]
    pulse_dur = 25e-3
    n_pulse_steps = int(round(pulse_dur / dt))

    Wt  = np.stack(res["W_traj"]) if len(res["W_traj"]) else None
    W_t = res["W_t"]

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.2, 1.4])
    fig.suptitle("Stochastic Figure-Ground (model0)",
                 fontsize=13, fontweight="bold")

    # ---- (1) stimulus raster ----
    ax = fig.add_subplot(gs[0])
    for ch in fig_idx:
        ax.axhspan(ch - 0.5, ch + 0.5,
                   color=COL_FIG, alpha=0.10, zorder=0)
    for ch in range(cfg.N):
        starts = np.where(np.diff(np.concatenate([[0],
                                  (stim[ch] > 0).astype(int)])) == 1)[0]
        color = COL_FIG if ch in fig_idx else COL_GND
        for s in starts:
            ax.barh(ch, n_pulse_steps * dt, left=s * dt,
                    height=0.7, color=color, edgecolor="none")
    ax.set_ylim(cfg.N - 0.5, -0.5)
    ax.set_yticks(range(cfg.N))
    _setup_axes(ax,
                title="Stimulus raster — figure channels (green) co-fire every 2nd window",
                xlabel="time (s)", ylabel="channel")

    # ---- (2) mean E rate, figure vs ground ----
    ax = fig.add_subplot(gs[1])
    ax.plot(t, E[gnd_idx].mean(0), color=COL_GND, lw=1.6, label="Ground")
    ax.plot(t, E[fig_idx].mean(0), color=COL_FIG, lw=1.6, label="Figure")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Mean excitatory rate (group average)",
                xlabel="time (s)", ylabel=r"$\langle E\rangle$")

    # ---- (3) W evolution ----
    ax = fig.add_subplot(gs[2])
    if Wt is not None and len(W_t) > 0:
        W_FF = np.zeros(len(W_t))
        W_GG = np.zeros(len(W_t))
        W_FG = np.zeros(len(W_t))
        for k in range(len(W_t)):
            W_FF[k], W_GG[k], W_FG[k] = compute_W_groups(Wt[k], fig_idx, gnd_idx)
        ax.plot(W_t, W_FF, color=COL_FIG, lw=2.4, label=r"$W_{F\to F}$")
        ax.plot(W_t, W_GG, color=COL_GND, lw=2.4, label=r"$W_{G\to G}$")
        ax.plot(W_t, W_FG, color=COL_FG,  lw=1.6, ls="--",
                label=r"$W_{F\leftrightarrow G}$")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    _setup_axes(ax, title=r"Recurrent E$\to$E weight evolution (mean per group)",
                xlabel="time (s)", ylabel="mean W")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_sfg_W(res: dict, fname: str):
    """Final W matrix, two views (original + reordered)."""
    fig_idx = res["fig_idx"]
    gnd_idx = res["gnd_idx"]
    W = res["W_final"]
    N = W.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    Wmax = max(W.max(), 1e-3)

    perm = np.concatenate([fig_idx, gnd_idx])
    W_perm = W[np.ix_(perm, perm)]
    im = axes[0].imshow(W_perm, cmap="viridis", origin="upper",
                        vmin=0, vmax=Wmax)
    axes[0].add_patch(plt.Rectangle((-0.5, -0.5),
                                    len(fig_idx), len(fig_idx),
                                    fill=False, edgecolor=COL_FIG, lw=3))
    axes[0].set_xticks(range(N)); axes[0].set_xticklabels(perm)
    axes[0].set_yticks(range(N)); axes[0].set_yticklabels(perm)
    axes[0].set_xlabel("pre"); axes[0].set_ylabel("post")
    axes[0].set_title("Reordered — figure block top-left",
                      fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(W, cmap="viridis", origin="upper",
                        vmin=0, vmax=Wmax)
    for i in fig_idx:
        for j in fig_idx:
            if i != j:
                axes[1].add_patch(plt.Rectangle((j - 0.5, i - 0.5),
                                                1, 1, fill=False,
                                                edgecolor=COL_FIG, lw=1.8))
    axes[1].set_xticks(range(N)); axes[1].set_yticks(range(N))
    axes[1].set_xlabel("pre"); axes[1].set_ylabel("post")
    axes[1].set_title("Original channel order",
                      fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_sfg_currents(res: dict, fname: str):
    """Synaptic currents: feed-forward + recurrent excitation vs inhibition,
    averaged over figure and ground channels."""
    cfg = res["cfg"]
    t = res["t"]
    fig_idx = res["fig_idx"]
    gnd_idx = res["gnd_idx"]
    tm_in   = res["tm_in"]
    rec_E   = res["rec_E"]
    inh_to_E = res["inh_to_E"]

    Exc = tm_in + rec_E
    Inh = inh_to_E
    Net = Exc - Inh

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)

    axes[0].plot(t, Inh[gnd_idx].mean(0), color=COL_GND, lw=1.8, label="Ground")
    axes[0].plot(t, Inh[fig_idx].mean(0), color=COL_FIG, lw=1.8, label="Figure")
    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(axes[0], title="Inhibitory current onto E (per-channel)",
                xlabel="time (s)", ylabel="I current")

    axes[1].plot(t, Exc[gnd_idx].mean(0), color=COL_GND, lw=1.8, label="Ground")
    axes[1].plot(t, Exc[fig_idx].mean(0), color=COL_FIG, lw=1.8, label="Figure")
    axes[1].legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(axes[1], title=r"Excitatory current  (TC + W$\cdot$E)",
                xlabel="time (s)", ylabel="E current")

    axes[2].plot(t, Net[gnd_idx].mean(0), color=COL_GND, lw=1.8, label="Ground")
    axes[2].plot(t, Net[fig_idx].mean(0), color=COL_FIG, lw=1.8, label="Figure")
    axes[2].axhline(0, color="0.4", lw=0.6)
    axes[2].legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(axes[2], title="Net drive (excitation − inhibition)",
                xlabel="time (s)", ylabel="net")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
# Main
# =====================================================================
def main():
    cfg = A1Config(N=12)
    fig_idx = np.array([1, 4, 7, 10])

    print("[ Running SFG on model0, T=60s ]")
    res = run_sfg(cfg=cfg, fig_idx=fig_idx, T=60.0, seed=7)

    # stimulus diagnostics — verify the design constraints
    per_ch, per_win = stim_summary(res["stim"], cfg)
    print(f"  per-channel pulses: min={per_ch.min()}, max={per_ch.max()}, "
          f"mean={per_ch.mean():.1f}")
    print(f"  per-window channels: min={per_win.min()}, max={per_win.max()}, "
          f"mean={per_win.mean():.1f}")

    print("[ Plotting ]")
    plot_sfg_run(res,      "sfg_m0_run.png")
    plot_sfg_W(res,        "sfg_m0_W.png")
    plot_sfg_currents(res, "sfg_m0_currents.png")

    W = res["W_final"]
    W_FF, W_GG, W_FG = compute_W_groups(W, fig_idx, res["gnd_idx"])
    print(f"\nFinal weight statistics:")
    print(f"  mean W_F->F      = {W_FF:.4f}")
    print(f"  mean W_G->G      = {W_GG:.4f}")
    print(f"  mean W_F<->G     = {W_FG:.4f}")
    print(f"  ratio W_FF / W_GG = {W_FF/(W_GG+1e-9):.2f}x")
    print(f"  ratio W_FF / W_FG = {W_FF/(W_FG+1e-9):.2f}x")


if __name__ == "__main__":
    main()
