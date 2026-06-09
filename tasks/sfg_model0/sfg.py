"""
sfg.py
======

Stochastic Figure-Ground (SFG) paradigm for model0.

Reference paradigm: Teki, Chait, Kumar, von Kriegstein & Griffiths
(2011) J. Neurosci.; the baphy fgFrozen figure-ground sound objects.

Stimulus -- one presentation
----------------------------
A tone cloud on N tonotopic channels.

  Ground.  Random pulses placed by GLOBAL stratified sampling: all
  random onset times are spread one per [0,T]/n sub-interval (jittered
  within), then each onset is assigned to a channel.  Onsets are random
  and not quantised to a grid; the population onset rate is uniform in
  time; and because onset times are distinct, no two ground pulses
  start together (no vertical-line alignment).

  Figure.  A fixed subset of channels (the figure channels) additionally
  fire COHERENTLY: a pulse on every figure channel, aligned, once per
  figure period.  Figure channels also carry stratified random pulses,
  exactly like ground channels.

Every channel receives the same total number of pulses, so the
time-marginal is flat; stratified onsets keep the freq-marginal uniform.

Session -- repeated presentations
---------------------------------
One presentation is too short to shape recurrent plasticity, so a
session repeats the presentation as ``n_trials`` trials:

  - The coherent FIGURE is FROZEN -- identical figure channels and
    identical coherent grid in every trial.  It therefore repeats and
    shapes the figure-figure recurrent weights W_FF.
  - The random GROUND is drawn FRESH every trial.  Ground co-firings
    are random pairs that differ trial to trial, so they average out
    and W_GG stays flat.

After n_trials presentations, W_FF >> W_GG: the model has learned the
figure.

Model: model0 with the AB/BA-task configuration, N overridden to 28.
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

from model0 import A1Config, INH_PRESETS, simulate, shared_config

# ---- paradigm defaults --------------------------------------------------------
N_CHANNELS    = 28
FIG_IDX       = np.array([4, 9, 14, 19, 24])       # 5 figure channels, spaced
PULSE_DUR     = 25e-3
FIG_PERIOD    = 250e-3                             # coherent figure repeat period
P_PER_CHANNEL = 28                                 # pulses per channel per trial
T_TRIAL       = 5.0                                # one SFG presentation
TRIAL_GAP     = 0.5                                # silence between presentations
N_TRIALS      = 40                                 # presentations per session


# =====================================================================
#  Stimulus -- one presentation
# =====================================================================
def build_sfg_stim(
    cfg: A1Config,
    fig_idx: np.ndarray,
    T: float = T_TRIAL,
    pulse_dur: float = PULSE_DUR,
    fig_period: float = FIG_PERIOD,
    P: int = P_PER_CHANNEL,
    tone_amp: float = 1.0,
    seed: int = 7,
) -> Tuple[np.ndarray, dict]:
    """Build one SFG presentation (a tone cloud with a coherent figure).

    Coherent figure pulses sit on a regular grid (period ``fig_period``).
    Every other pulse is placed by global stratified sampling -- onset
    times spread one per sub-interval, each assigned to a channel.  Each
    channel ends with exactly ``P`` pulses.
    """
    rng = np.random.default_rng(seed)
    N, dt = cfg.N, cfg.dt

    n_pulse = int(round(pulse_dur / dt))
    T_steps = int(round(T / dt))
    max_on  = T_steps - n_pulse

    fig_idx = np.asarray(fig_idx, dtype=int)

    # coherent figure onsets: a regular grid -- the temporally coherent figure
    fig_period_steps = int(round(fig_period / dt))
    coh_onsets = np.arange(0, max_on + 1, fig_period_steps)
    n_coh = len(coh_onsets)
    if P < n_coh:
        raise ValueError(f"P={P} below the coherent-pulse count {n_coh}")

    stim = np.zeros((N, T_steps))
    onsets = [[] for _ in range(N)]

    # 1. coherent figure pulses (aligned grid)
    for ch in fig_idx:
        for on in coh_onsets:
            stim[ch, on:on + n_pulse] = tone_amp
        onsets[ch] = list(coh_onsets)

    # 2. random pulses -- global stratified placement (see module docstring).
    quota = np.full(N, P, dtype=int)
    for ch in fig_idx:
        quota[ch] -= n_coh
    n_random = int(quota.sum())

    edges = np.linspace(0, max_on, n_random + 1)
    onset_times = np.sort(np.array(
        [int(rng.uniform(edges[k], edges[k + 1])) for k in range(n_random)]))

    for t_on in onset_times:
        elig = [ch for ch in range(N) if quota[ch] > 0 and
                all(abs(t_on - e) >= n_pulse for e in onsets[ch])]
        if not elig:
            elig = [ch for ch in range(N) if quota[ch] > 0]
            if not elig:
                break
        w = np.array([quota[ch] for ch in elig], dtype=float)
        ch = int(rng.choice(elig, p=w / w.sum()))
        stim[ch, t_on:t_on + n_pulse] = tone_amp
        onsets[ch].append(int(t_on))
        quota[ch] -= 1

    meta = dict(coh_onsets=coh_onsets, n_coh=n_coh,
                n_pulse_steps=n_pulse, fig_period=fig_period, P=P)
    return stim, meta


# =====================================================================
#  Session -- repeated presentations
# =====================================================================
def build_sfg_session(
    cfg: A1Config,
    fig_idx: np.ndarray,
    n_trials: int = N_TRIALS,
    T_trial: float = T_TRIAL,
    trial_gap: float = TRIAL_GAP,
    seed: int = 7,
) -> Tuple[np.ndarray, dict]:
    """Concatenate ``n_trials`` SFG presentations.

    The coherent figure is frozen (identical every trial); the random
    ground is drawn fresh each trial (a new seed per trial).
    """
    dt = cfg.dt
    gap_steps = int(round(trial_gap / dt))
    fig_idx = np.asarray(fig_idx, dtype=int)

    blocks, trial_starts, coh_global = [], [], []
    n_pulse_steps = fig_period = None
    pos = 0
    for t in range(n_trials):
        stim_t, meta_t = build_sfg_stim(cfg, fig_idx, T=T_trial,
                                        seed=seed + 1 + t)
        trial_starts.append(pos)
        coh_global.extend(int(pos + o) for o in meta_t["coh_onsets"])
        n_pulse_steps = meta_t["n_pulse_steps"]
        fig_period = meta_t["fig_period"]
        blocks.append(stim_t)
        pos += stim_t.shape[1]
        if gap_steps and t < n_trials - 1:
            blocks.append(np.zeros((cfg.N, gap_steps)))
            pos += gap_steps

    stim = np.concatenate(blocks, axis=1)
    gnd_idx = np.setdiff1d(np.arange(cfg.N), fig_idx)
    meta = dict(
        coh_onsets=np.array(coh_global), gnd_idx=gnd_idx,
        n_pulse_steps=n_pulse_steps, fig_period=fig_period,
        trial_starts=np.array(trial_starts),
        T_trial_steps=int(round(T_trial / dt)),
        n_trials=n_trials, trial_gap=trial_gap, T_trial=T_trial,
    )
    return stim, meta


def stim_marginals(stim: np.ndarray, cfg: A1Config,
                   bin_dur: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time_marginal, freq_marginal, bin_centres) for one block.

    time_marginal : (N,)       per-channel activity averaged over time
    freq_marginal : (n_bins,)  per-bin activity averaged over channels
    bin_centres   : (n_bins,)  bin centre times (s)
    """
    dt = cfg.dt
    time_marg = stim.mean(axis=1)
    n_bin = int(round(bin_dur / dt))
    n_bins = stim.shape[1] // n_bin
    pop = stim[:, : n_bins * n_bin].mean(axis=0)
    freq_marg = pop.reshape(n_bins, n_bin).mean(axis=1)
    bin_centres = (np.arange(n_bins) + 0.5) * bin_dur
    return time_marg, freq_marg, bin_centres


# =====================================================================
#  Experiment
# =====================================================================
def run_sfg(
    cfg: Optional[A1Config] = None,
    fig_idx: Optional[np.ndarray] = None,
    n_trials: int = N_TRIALS,
    seed: int = 7,
) -> dict:
    """Build a repeated-presentation SFG session and run model0 on it."""
    if cfg is None:
        cfg = shared_config(N=N_CHANNELS)
    if fig_idx is None:
        fig_idx = FIG_IDX

    stim, meta = build_sfg_session(cfg, fig_idx, n_trials=n_trials, seed=seed)
    snap_every = max(1, int(round(0.25 / cfg.dt)))
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(stim=stim, fig_idx=np.asarray(fig_idx), **meta)
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


def _draw_raster(ax, res, lo, hi):
    """Draw the stimulus raster for samples [lo, hi).

    Green = coherent figure pulse; blue = every other pulse (including
    the random pulses figure channels also carry).
    """
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]
    fig_set = {int(c) for c in res["fig_idx"]}
    coh_set = {int(o) for o in res["coh_onsets"]}
    n_pulse = res["n_pulse_steps"]

    for ch in res["fig_idx"]:
        ax.axhspan(ch - 0.5, ch + 0.5, color=COL_FIG, alpha=0.10, zorder=0)
    seg = stim[:, lo:hi]
    for ch in range(cfg.N):
        starts = np.where(np.diff(np.concatenate(
            [[0], (seg[ch] > 0).astype(int)])) == 1)[0]
        for s in starts:
            coh_fig = (ch in fig_set) and ((lo + int(s)) in coh_set)
            ax.barh(ch, n_pulse * dt, left=(lo + s) * dt, height=0.72,
                    color=(COL_FIG if coh_fig else COL_GND), edgecolor="none")
    ax.set_xlim(lo * dt, hi * dt)
    ax.set_ylim(cfg.N - 0.5, -0.5)
    ax.set_yticks(range(0, cfg.N, 2))


def plot_sfg_marginals(res: dict, fname: str):
    """Minimal check of the two stimulus marginals (one presentation)."""
    cfg = res["cfg"]
    one_trial = res["stim"][:, : res["T_trial_steps"]]
    tm, fm, bc = stim_marginals(one_trial, cfg)
    N = cfg.N

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), constrained_layout=True)
    fig.suptitle("SFG stimulus marginals (one presentation)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(range(N), tm, color="0.20", lw=1.5, marker="o", ms=3)
    ax.axhline(tm.mean(), color="0.65", ls=":", lw=1.0)
    ax.set_ylim(0, tm.max() * 1.4)
    _setup_axes(ax, title=f"time marginal (per channel)",
                          # f"{tm.std()/tm.mean()*100:.1f}%",
                xlabel="channel", ylabel="mean activity")

    ax = axes[1]
    ax.plot(bc, fm, color="0.20", lw=1.5)
    ax.axhline(fm.mean(), color="0.65", ls=":", lw=1.0)
    ax.set_ylim(0, fm.max() * 1.4)
    _setup_axes(ax, title=f"freq marginal (per 250 ms bin)",
                          # f"{fm.std()/fm.mean()*100:.1f}%",
                xlabel="time (s)", ylabel="mean activity")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_sfg_run(res: dict, fname: str):
    """Raster (first 2 trials), mean activity, W_FF/W_GG/W_FG evolution."""
    cfg = res["cfg"]; dt = cfg.dt
    E = res["E"]
    fig_idx, gnd_idx = res["fig_idx"], res["gnd_idx"]
    t = res["t"]

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else None
    W_t = res["W_t"]

    # window: first 2 trials (incl. the gap between them)
    trial_span = res["T_trial_steps"] + int(round(res["trial_gap"] / dt))
    hi = min(2 * trial_span, res["stim"].shape[1])

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.1, 1.6])
    fig.suptitle(f"SFG, "
                 f"{res['n_trials']} presentations",
                 fontsize=13, fontweight="bold")

    ax = fig.add_subplot(gs[0])
    _draw_raster(ax, res, 0, hi)
    _setup_axes(ax, title="Stimulus raster (first 2 presentations), "
                          "",
                xlabel="time (s)", ylabel="channel")

    ax = fig.add_subplot(gs[1])
    ax.plot(t[:hi], E[gnd_idx][:, :hi].mean(0), color=COL_GND, lw=1.3,
            label="Ground")
    ax.plot(t[:hi], E[fig_idx][:, :hi].mean(0), color=COL_FIG, lw=1.3,
            label="Figure")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Mean excitatory rate",
                xlabel="time (s)", ylabel=r"$\langle E\rangle$")

    ax = fig.add_subplot(gs[2])
    if Wt is not None and len(W_t):
        FF = np.empty(len(W_t)); GG = np.empty(len(W_t)); FG = np.empty(len(W_t))
        for k in range(len(W_t)):
            FF[k], GG[k], FG[k] = compute_W_groups(Wt[k], fig_idx, gnd_idx)
        ax.plot(W_t, FF, color=COL_FIG, lw=2.6, label=r"$W_{F\to F}$ (figure)")
        ax.plot(W_t, GG, color=COL_GND, lw=2.2, label=r"$W_{G\to G}$ (ground)")
        ax.plot(W_t, FG, color="0.45", lw=1.6, ls="--",
                label=r"$W_{F\leftrightarrow G}$")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    _setup_axes(ax, title="E to E weight evolution over the session",
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
    axes[0].set_xticks(range(N)); axes[0].set_xticklabels(perm, fontsize=6)
    axes[0].set_yticks(range(N)); axes[0].set_yticklabels(perm, fontsize=6)
    axes[0].set_xlabel("pre"); axes[0].set_ylabel("post")
    axes[0].set_title("Reordered, figure block top-left",
                      fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(W, cmap="viridis", origin="upper", vmin=0, vmax=Wmax)
    axes[1].set_xticks(range(0, N, 2)); axes[1].set_yticks(range(0, N, 2))
    axes[1].tick_params(labelsize=7)
    axes[1].set_xlabel("pre"); axes[1].set_ylabel("post")
    axes[1].set_title("Original channel order", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average of width ``win`` samples."""
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def plot_sfg_drive(res: dict, fname: str,
                   smooth_ms: float = 1000.0, ds: int = 50):
    """Synaptic drive over the whole session: the inhibition, the
    excitation, and the net drive received by the figure vs the ground
    channels.  Companion to the MATLAB SFG 'synaptic inputs' figure.

    Traces are smoothed with a ``smooth_ms`` moving average: at full
    session length (tens of trials) the per-pulse detail is not
    resolvable, so the smoothed envelope is what conveys the overall
    behaviour -- the figure-vs-ground levels and how they drift as the
    figure connection is learned.  ``ds`` downsamples for display only.
    """
    cfg = res["cfg"]; dt = cfg.dt
    fig_idx, gnd_idx = res["fig_idx"], res["gnd_idx"]
    win = max(1, int(round(smooth_ms * 1e-3 / dt)))

    inh = res["inh_to_E"]
    exc = res["tm_in"] + res["rec_E"]
    net = exc - inh
    t = res["t"][::ds]

    def env(arr, idx):
        return _smooth(arr[idx].mean(0), win)[::ds]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True,
                             sharex=True)
    fig.suptitle(f"SFG synaptic drive over the session "
                 f"({smooth_ms:.0f} ms moving average)",
                 fontsize=12, fontweight="bold")

    panels = [
        (axes[0], inh, "Inhibition received", "inhibitory current", False),
        (axes[1], exc, "Excitation received", "excitatory current", False),
        (axes[2], net, "Net synaptic drive", "net drive", True),
    ]
    for ax, arr, title, ylab, zero in panels:
        if zero:
            ax.axhline(0, color="0.6", lw=0.7)
        eg = env(arr, gnd_idx)
        ef = env(arr, fig_idx)
        # shade the figure-vs-ground gap so it is visible even where the
        # two envelopes nearly coincide
        ax.fill_between(t, eg, ef, color="0.55", alpha=0.30, lw=0)
        ax.plot(t, eg, color=COL_GND, lw=1.3, label="Ground")
        ax.plot(t, ef, color=COL_FIG, lw=1.3, label="Figure")
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        _setup_axes(ax, title=title, ylabel=ylab)
    axes[2].set_xlabel("time (s)", fontsize=10)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    fig_idx = FIG_IDX

    # Loop over both inhibition-structure presets.  Filenames carry the
    # preset tag; the two runs do not overwrite.
    for inh_name in INH_PRESETS:
        cfg = shared_config(N=N_CHANNELS, inh=inh_name)

        print(f"\n========================================================")
        print(f"[ SFG -- inhibition preset '{inh_name}' "
              f"(N={N_CHANNELS}, {len(fig_idx)} figure channels, "
              f"{N_TRIALS} presentations) ]")
        print(f"  w_EI = (self {cfg.w_EI_self}, lat {cfg.w_EI_lat}); "
              f"w_IE = (self {cfg.w_IE_self}, lat {cfg.w_IE_lat})")

        res = run_sfg(cfg=cfg, fig_idx=fig_idx, n_trials=N_TRIALS, seed=7)

        tm, fm, _ = stim_marginals(res["stim"][:, : res["T_trial_steps"]], cfg)
        print(f"  time-marginal CV (one trial):  {tm.std()/tm.mean()*100:.1f}%")
        print(f"  freq-marginal CV (one trial):  {fm.std()/fm.mean()*100:.1f}%")

        print("[ Plotting ]")
        plot_sfg_marginals(res, f"sfg_m0_marginals_{inh_name}.png")
        plot_sfg_run(res,       f"sfg_m0_run_{inh_name}.png")
        plot_sfg_W(res,         f"sfg_m0_W_{inh_name}.png")
        plot_sfg_drive(res,     f"sfg_m0_drive_{inh_name}.png")

        W = res["W_final"]
        W_FF, W_GG, W_FG = compute_W_groups(W, fig_idx, res["gnd_idx"])
        print(f"\nFinal weight statistics [{inh_name}]:")
        print(f"  mean W_F->F       = {W_FF:.4f}")
        print(f"  mean W_G->G       = {W_GG:.4f}")
        print(f"  mean W_F<->G      = {W_FG:.4f}")
        print(f"  ratio W_FF / W_GG = {W_FF/(W_GG+1e-9):.2f}x")
        print(f"  ratio W_FF / W_FG = {W_FF/(W_FG+1e-9):.2f}x")
    print("\nDone.")


if __name__ == "__main__":
    main()
