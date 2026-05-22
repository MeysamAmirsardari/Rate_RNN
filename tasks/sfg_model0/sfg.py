"""
sfg.py
======

Stochastic Figure-Ground (SFG) paradigm for model0.

Adapted from sfg_rnn2.m, with a rebuilt stimulus that satisfies three
design goals:

  (1) Flat time-marginal.   Averaging the channels x time stimulus over
      *time* gives a near-uniform per-channel total -- no channel is
      over-stimulated.

  (2) Flat freq-marginal.   Averaging over *channels* gives a uniform
      per-slot count -- no bursty time windows and no silent ones.

  (3) Figure channels also carry ground.   The figure channels are not
      "figure only": besides their coherent pulses they also receive
      random pulses, exactly like the ground channels.  They present
      both figure and ground.

Stimulus construction
---------------------
Time is a grid of ``slot_dur`` slots; each slot carries up to one pulse
per channel (``pulse_dur`` wide).  Every slot has exactly ``K`` active
channels -> the freq-marginal is flat by construction.

  - Coherent slots (every ``coherent_every``-th slot): the N_fig figure
    channels fire together (aligned to the slot onset -> a coherent
    figure).  The remaining K - N_fig slots are random ground.
  - Non-coherent slots: K channels drawn at random from *all* N channels
    (figure channels included -> figure presents ground).

Per-channel totals are equalised by quota-weighted sampling: a channel
that still owes pulses is proportionally more likely to be picked, so
every channel ends near the same total -> the time-marginal is flat.

Non-coherent pulses are jittered within their slot; coherent figure
pulses stay aligned so the figure remains temporally coherent.

Model: model0 with the AB/BA-task config, N overridden to 18.
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
SLOT_DUR      = 75e-3
K_PER_SLOT    = 8                                  # active channels per slot
COHERENT_EVERY = 4                                 # figure coheres every 4 slots
T_DEFAULT     = 60.0


# =====================================================================
#  Stimulus
# =====================================================================
def build_sfg_stim(
    cfg: A1Config,
    fig_idx: np.ndarray,
    T: float = T_DEFAULT,
    pulse_dur: float = PULSE_DUR,
    slot_dur: float = SLOT_DUR,
    K: int = K_PER_SLOT,
    coherent_every: int = COHERENT_EVERY,
    tone_amp: float = 1.0,
    seed: int = 7,
) -> Tuple[np.ndarray, dict]:
    """Build the balanced SFG stimulus.

    Returns (stim (N, T_steps), meta).
    """
    rng = np.random.default_rng(seed)
    N, dt = cfg.N, cfg.dt

    n_pulse  = int(round(pulse_dur / dt))
    n_slot   = int(round(slot_dur / dt))
    n_slots  = int(round(T / slot_dur))
    T_steps  = n_slots * n_slot

    fig_idx = np.asarray(fig_idx, dtype=int)
    gnd_idx = np.setdiff1d(np.arange(N), fig_idx)
    n_fig   = len(fig_idx)
    is_fig  = np.zeros(N, dtype=bool); is_fig[fig_idx] = True

    if K < n_fig or K > N:
        raise ValueError(f"K={K} incompatible with N={N}, n_fig={n_fig}")

    coherent = np.zeros(n_slots, dtype=bool)
    coherent[::coherent_every] = True

    # ---- choose which channels are active in each slot ----
    active = np.zeros((N, n_slots), dtype=bool)
    P = int(round(K * n_slots / N))          # target pulses per channel
    quota = np.full(N, P, dtype=float)

    # coherent figure pulses (figure channels fire together)
    for s in np.flatnonzero(coherent):
        active[fig_idx, s] = True
        quota[fig_idx] -= 1.0

    # fill every slot up to K, picking channels weighted by remaining quota
    for s in rng.permutation(n_slots):
        need = K - int(active[:, s].sum())
        if need <= 0:
            continue
        elig = np.flatnonzero(~active[:, s])
        w = np.clip(quota[elig], 0.0, None) + 1e-3   # small floor for robustness
        w = w / w.sum()
        pick = rng.choice(elig, size=min(need, len(elig)), replace=False, p=w)
        active[pick, s] = True
        quota[pick] -= 1.0

    # ---- render to (N, T) with within-slot jitter ----
    stim = np.zeros((N, T_steps))
    max_jit = n_slot - n_pulse
    for s in range(n_slots):
        base = s * n_slot
        for ch in np.flatnonzero(active[:, s]):
            # coherent figure pulses stay aligned; everything else jitters
            aligned = coherent[s] and is_fig[ch]
            off = 0 if aligned else int(rng.integers(0, max_jit + 1))
            stim[ch, base + off : base + off + n_pulse] = tone_amp

    meta = dict(
        coherent=coherent, active=active, gnd_idx=gnd_idx,
        n_slots=n_slots, n_slot_steps=n_slot, n_pulse_steps=n_pulse,
        slot_dur=slot_dur, pulse_dur=pulse_dur, P_target=P,
        n_coherent=int(coherent.sum()),
    )
    return stim, meta


def stim_marginals(stim: np.ndarray, meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time_marginal, freq_marginal).

    time_marginal : (N,)  per-channel mean activity (averaged over time)
    freq_marginal : (n_slots,)  per-slot mean activity (averaged over
                                channels, then over the slot)
    """
    time_marginal = stim.mean(axis=1)
    n_slots, n_slot = meta["n_slots"], meta["n_slot_steps"]
    sl = stim[:, : n_slots * n_slot].reshape(stim.shape[0], n_slots, n_slot)
    freq_marginal = sl.mean(axis=(0, 2))     # mean over channels and within-slot
    return time_marginal, freq_marginal


# =====================================================================
#  Experiment
# =====================================================================
def run_sfg(
    cfg: Optional[A1Config] = None,
    fig_idx: Optional[np.ndarray] = None,
    T: float = T_DEFAULT,
    seed: int = 7,
) -> dict:
    """Build the balanced SFG stimulus and run model0 on it."""
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


def _draw_raster(ax, res, n_show):
    """Draw the stimulus raster up to sample ``n_show``.

    Pulses are coloured by *type*, not by channel: a pulse is green only
    if it is a coherent figure pulse (a figure channel in a coherent
    slot); every other pulse -- including the random pulses that figure
    channels also carry -- is blue.
    """
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]
    fig_set = {int(c) for c in res["fig_idx"]}
    coherent = res["coherent"]
    n_slot_steps = res["n_slot_steps"]
    n_pulse = res["n_pulse_steps"]

    for ch in res["fig_idx"]:
        ax.axhspan(ch - 0.5, ch + 0.5, color=COL_FIG, alpha=0.10, zorder=0)
    for ch in range(cfg.N):
        starts = np.where(np.diff(np.concatenate(
            [[0], (stim[ch, :n_show] > 0).astype(int)])) == 1)[0]
        for s in starts:
            slot = s // n_slot_steps
            is_coh_fig = (ch in fig_set and slot < len(coherent)
                          and coherent[slot])
            color = COL_FIG if is_coh_fig else COL_GND
            ax.barh(ch, n_pulse * dt, left=s * dt, height=0.72,
                    color=color, edgecolor="none")
    ax.set_xlim(0, n_show * dt)
    ax.set_ylim(cfg.N - 0.5, -0.5)
    ax.set_yticks(range(cfg.N))


def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_sfg_stimulus(res: dict, fname: str, t_show: float = 8.0):
    """Stimulus raster with its two 1-D marginals -- the design check.

    The freq-marginal (bottom) and time-marginal (right) should both be
    near-uniform: that is the explicit design goal.
    """
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]
    fig_idx = res["fig_idx"]
    n_pulse = res["n_pulse_steps"]
    N = cfg.N

    time_marg, freq_marg = stim_marginals(stim, res)
    n_show = int(round(t_show / dt))
    n_show = min(n_show, stim.shape[1])

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    fig.suptitle(
        "SFG stimulus — flat marginals, figure channels also carry ground",
        fontsize=12, fontweight="bold")

    # ---- raster ----
    ax_r = fig.add_subplot(gs[0, 0])
    _draw_raster(ax_r, res, n_show)
    _setup_axes(ax_r, title=f"Stimulus raster (first {t_show:.0f} s) — "
                            f"coherent figure pulses green, random pulses blue",
                ylabel="channel")

    # ---- time-marginal (right): per-channel mean activity ----
    ax_t = fig.add_subplot(gs[0, 1], sharey=ax_r)
    colors = [COL_FIG if ch in fig_idx else COL_GND for ch in range(N)]
    ax_t.barh(range(N), time_marg, height=0.72, color=colors, edgecolor="none")
    ax_t.axvline(time_marg.mean(), color="0.3", lw=1.0, ls="--")
    ax_t.set_ylim(N - 0.5, -0.5)
    _setup_axes(ax_t, title="time-marginal\n(mean over time)",
                xlabel="mean activity")
    ax_t.tick_params(labelleft=False)

    # ---- freq-marginal (bottom): per-slot mean activity ----
    ax_f = fig.add_subplot(gs[1, 0], sharex=ax_r)
    slot_dur = res["slot_dur"]
    n_slot_show = int(round(t_show / slot_dur))
    slot_t = (np.arange(n_slot_show) + 0.5) * slot_dur
    ax_f.bar(slot_t, freq_marg[:n_slot_show], width=slot_dur * 0.9,
             color="0.55", edgecolor="none")
    ax_f.axhline(freq_marg.mean(), color="0.2", lw=1.0, ls="--")
    ax_f.set_xlim(0, n_show * dt)
    _setup_axes(ax_f, title="freq-marginal (mean over channels, per slot)",
                xlabel="time (s)", ylabel="mean activity")

    # ---- corner: stats ----
    ax_s = fig.add_subplot(gs[1, 1]); ax_s.axis("off")
    tm, fm = time_marg, freq_marg
    txt = (f"time-marginal\n"
           f"  mean {tm.mean():.4f}\n"
           f"  spread {tm.min():.4f}–{tm.max():.4f}\n"
           f"  CV {tm.std()/tm.mean()*100:.1f}%\n\n"
           f"freq-marginal\n"
           f"  mean {fm.mean():.4f}\n"
           f"  spread {fm.min():.4f}–{fm.max():.4f}\n"
           f"  CV {fm.std()/fm.mean()*100:.2f}%")
    ax_s.text(0.0, 0.95, txt, fontsize=8.5, va="top", family="monospace")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_sfg_run(res: dict, fname: str):
    """Raster, mean activity (figure vs ground), W_FF/W_GG/W_FG evolution."""
    cfg = res["cfg"]; dt = cfg.dt
    stim = res["stim"]; E = res["E"]
    fig_idx, gnd_idx = res["fig_idx"], res["gnd_idx"]
    t = res["t"]
    n_pulse = res["n_pulse_steps"]

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else None
    W_t = res["W_t"]

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.2, 1.4])
    fig.suptitle("Stochastic Figure-Ground (model0)",
                 fontsize=13, fontweight="bold")

    # raster
    ax = fig.add_subplot(gs[0])
    n_show = min(int(round(8.0 / dt)), stim.shape[1])
    _draw_raster(ax, res, n_show)
    _setup_axes(ax, title="Stimulus raster (first 8 s) — coherent figure "
                          "pulses green, random pulses blue",
                xlabel="time (s)", ylabel="channel")

    # mean activity
    ax = fig.add_subplot(gs[1])
    ax.plot(t, E[gnd_idx].mean(0), color=COL_GND, lw=1.5, label="Ground")
    ax.plot(t, E[fig_idx].mean(0), color=COL_FIG, lw=1.5, label="Figure")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Mean excitatory rate (group average)",
                xlabel="time (s)", ylabel=r"$\langle E\rangle$")

    # W evolution
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
    axes[0].set_title("Reordered — figure block top-left",
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

    print(f"[ Running SFG on model0 — N={N_CHANNELS}, "
          f"{len(fig_idx)} figure channels, T={T_DEFAULT:.0f}s ]")
    res = run_sfg(cfg=cfg, fig_idx=fig_idx, T=T_DEFAULT, seed=7)

    # ---- stimulus diagnostics ----
    tm, fm = stim_marginals(res["stim"], res)
    print(f"  time-marginal (per channel):  mean={tm.mean():.4f}  "
          f"spread={tm.min():.4f}-{tm.max():.4f}  CV={tm.std()/tm.mean()*100:.1f}%")
    print(f"  freq-marginal (per slot):     mean={fm.mean():.4f}  "
          f"spread={fm.min():.4f}-{fm.max():.4f}  CV={fm.std()/fm.mean()*100:.2f}%")

    print("[ Plotting ]")
    plot_sfg_stimulus(res, "sfg_m0_stimulus.png")
    plot_sfg_run(res,      "sfg_m0_run.png")
    plot_sfg_W(res,        "sfg_m0_W.png")

    W = res["W_final"]
    W_FF, W_GG, W_FG = compute_W_groups(W, fig_idx, res["gnd_idx"])
    print(f"\nFinal weight statistics:")
    print(f"  mean W_F->F      = {W_FF:.4f}")
    print(f"  mean W_G->G      = {W_GG:.4f}")
    print(f"  mean W_F<->G     = {W_FG:.4f}")
    print(f"  ratio W_FF / W_GG = {W_FF/(W_GG+1e-9):.2f}x")
    print(f"  ratio W_FF / W_FG = {W_FF/(W_FG+1e-9):.2f}x")
    print("Done.")


if __name__ == "__main__":
    main()
