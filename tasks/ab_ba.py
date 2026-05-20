"""
ab_ba.py
========

AB-vs-BA oddball paradigm for the A1 rate model.

Stimulus design
---------------
- Two pure tones: A (channel ch_A) and B (channel ch_B), tonotopically separated.
- Sequence = AB or BA, each tone 50 ms, no intra-sequence gap.
- Inter-sequence gap = 500 ms.
- Run 1:  p(AB) = 0.90,  p(BA) = 0.10   (BA is the deviant)
- Run 2:  p(AB) = 0.10,  p(BA) = 0.90   (AB is the deviant)
- 400 trials per run.

What we expect (the model's predictions)
----------------------------------------
1. **Hebbian sequence learning.** With AB frequent, channel A consistently
   precedes channel B in time.  The post-trace asymmetry of the learning rule
   then potentiates W[B<-A] (LTP) and depresses W[A<-B] (LTD).  Run 2
   reverses this.   This is the rate-coded analogue of STDP
   (Markram et al. 1997; Bi & Poo 1998).

2. **Stimulus-specific adaptation (SSA).** Short-term depression on the TC
   synapses reduces the response of the standard while the rare deviant tone
   activates "fresh" channels — a deviant-enhanced response, as observed
   in vivo (Ulanovsky, Las & Nelken, Nat. Neurosci. 2003; Nieto-Diego &
   Malmierca, PLOS Biol. 2016).

3. **Prediction-error signature.** After training, the deviant evokes a
   larger response than the standard *for the same second-tone identity*,
   because the recurrent prediction from the first tone matches the
   standard but not the deviant (Cooke et al. 2018; Parras et al. 2017).

Outputs:  PNG figures saved alongside this script.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from model import A1Config, simulate


# =====================================================================
#  Stimulus
# =====================================================================
def _tuning(N: int, centre: int, sigma: float) -> np.ndarray:
    """Gaussian tonotopic receptive field, peak = 1 at ``centre``."""
    if sigma <= 0.0:
        v = np.zeros(N); v[centre] = 1.0
        return v
    chs = np.arange(N)
    v = np.exp(-0.5 * ((chs - centre) / sigma) ** 2)
    return v / v.max()


def build_stim(
    seq_codes: List[str],
    cfg: A1Config,
    ch_A: int,
    ch_B: int,
    tone_dur: float = 50e-3,
    intra_gap: float = 0.0,
    inter_gap: float = 500e-3,
    tone_amp: float = 1.0,
    tuning_sigma: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Assemble an (N, T) stimulus matrix from a list of 'AB' / 'BA' codes."""
    dt = cfg.dt
    n_tone  = int(round(tone_dur  / dt))
    n_intra = int(round(intra_gap / dt))
    n_inter = int(round(inter_gap / dt))
    n_seq   = 2 * n_tone + n_intra + n_inter
    T_total = n_seq * len(seq_codes)

    tA = tone_amp * _tuning(cfg.N, ch_A, tuning_sigma)
    tB = tone_amp * _tuning(cfg.N, ch_B, tuning_sigma)

    stim = np.zeros((cfg.N, T_total))
    starts = np.zeros(len(seq_codes), dtype=int)

    for k, code in enumerate(seq_codes):
        s = k * n_seq
        starts[k] = s
        if code == "AB":
            stim[:, s             : s + n_tone]                           = tA[:, None]
            stim[:, s + n_tone + n_intra : s + 2 * n_tone + n_intra]      = tB[:, None]
        elif code == "BA":
            stim[:, s             : s + n_tone]                           = tB[:, None]
            stim[:, s + n_tone + n_intra : s + 2 * n_tone + n_intra]      = tA[:, None]
        else:
            raise ValueError(f"bad seq code: {code!r}")

    return stim, starts, n_seq


def shuffled_codes(n_total: int, p_AB: float, rng: np.random.Generator) -> List[str]:
    """Random interleaving of 'AB' (proportion p_AB) and 'BA'."""
    n_AB = int(round(n_total * p_AB))
    arr = np.array(["AB"] * n_AB + ["BA"] * (n_total - n_AB))
    rng.shuffle(arr)
    return arr.tolist()


# =====================================================================
#  Experiment runner
# =====================================================================
def run_experiment(
    p_AB: float,
    n_trials: int = 400,
    seed: int = 0,
    cfg: Optional[A1Config] = None,
    ch_A: int = 2,
    ch_B: int = 5,
) -> dict:
    """Build the stimulus and run one full oddball session."""
    if cfg is None:
        cfg = A1Config(N=8)

    rng = np.random.default_rng(seed)
    codes = shuffled_codes(n_trials, p_AB, rng)
    stim, starts, n_seq = build_stim(codes, cfg, ch_A, ch_B)

    # snapshot W every ~0.5 s (cheap enough)
    snap_every = max(1, int(round(0.5 / cfg.dt)))
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(
        stim=stim,
        codes=np.array(codes),
        seq_starts=starts,
        n_seq=n_seq,
        p_AB=p_AB,
        ch_A=ch_A,
        ch_B=ch_B,
    )
    return out


def evoked_per_trial(E_hist: np.ndarray, starts: np.ndarray, n_seq: int) -> np.ndarray:
    """Cut E history into per-trial epochs of length n_seq.  Returns (Ntrials, N, n_seq)."""
    out = np.empty((len(starts), E_hist.shape[0], n_seq))
    for k, s in enumerate(starts):
        out[k] = E_hist[:, s : s + n_seq]
    return out


# =====================================================================
#  Plotting
# =====================================================================
def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_run(res: dict, suptitle: str, fname: str):
    """Eight-panel figure summarising one oddball run."""
    cfg = res["cfg"]; dt = cfg.dt
    E   = res["E"];  I = res["I"]
    u, x = res["u"], res["x"]
    stim = res["stim"]
    ch_A, ch_B = res["ch_A"], res["ch_B"]
    n_seq = res["n_seq"]

    n_show = 6                                  # first N trials in the raster panels
    show_T = n_show * n_seq
    t_show = res["t"][:show_T]

    ev = evoked_per_trial(E, res["seq_starts"], n_seq)
    codes = res["codes"]
    is_AB = codes == "AB"
    # use the second half (post-learning) for averages
    half = len(codes) // 2
    AB_late = ev[half:][is_AB[half:]]
    BA_late = ev[half:][~is_AB[half:]]
    ts = np.arange(n_seq) * dt

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else np.empty((0, cfg.N, cfg.N))
    W_t = res["W_t"]
    W_f = res["W_final"]

    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # ---- (1) stimulus raster (first n_show trials) ----
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(stim[:, :show_T], aspect="auto", origin="lower",
              cmap="Oranges", interpolation="nearest",
              extent=[0, show_T * dt, -0.5, cfg.N - 0.5])
    # mark trial starts and codes
    for k in range(n_show):
        ax.axvline(k * n_seq * dt, color="0.4", lw=0.5, ls=":")
        ax.text((k + 0.5) * n_seq * dt, cfg.N - 0.2, codes[k],
                ha="center", va="top", fontsize=9, color="0.2")
    ax.axhline(ch_A, color="tab:red",  lw=0.6, ls="--", alpha=0.5)
    ax.axhline(ch_B, color="tab:blue", lw=0.6, ls="--", alpha=0.5)
    _setup_axes(ax, title=f"Stimulus raster (first {n_show} trials)",
                xlabel="time (s)", ylabel="channel")

    # ---- (2) E activity, channels A and B ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_show, E[ch_A, :show_T], color="tab:red",  lw=1.2, label=f"E[A={ch_A}]")
    ax.plot(t_show, E[ch_B, :show_T], color="tab:blue", lw=1.2, label=f"E[B={ch_B}]")
    ax.legend(fontsize=8, frameon=False)
    _setup_axes(ax, title="E rate at A and B (first trials)",
                xlabel="time (s)", ylabel="rate")

    # ---- (3) global inhibition + STP variables on ch_A ----
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_show, I[:show_T], color="tab:blue", lw=1.2, label="global I")
    ax2 = ax.twinx()
    ax2.plot(t_show, x[ch_A, :show_T], color="tab:purple", lw=1.0,
             label=f"x[A]  (resources)")
    ax2.plot(t_show, u[ch_A, :show_T], color="tab:olive",  lw=1.0,
             label=f"u[A]  (util.)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("u, x", fontsize=10)
    ax.legend(loc="upper left",  fontsize=8, frameon=False)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)
    _setup_axes(ax, title="Inhibition & short-term depression",
                xlabel="time (s)", ylabel="I rate")

    # ---- (4) average evoked response, A channel ----
    ax = fig.add_subplot(gs[2, 0])
    if len(AB_late):
        ax.plot(ts, AB_late.mean(0)[ch_A], color="tab:green",
                lw=2, label=f"AB  (n={len(AB_late)})")
    if len(BA_late):
        ax.plot(ts, BA_late.mean(0)[ch_A], color="tab:purple", ls="--",
                lw=2, label=f"BA  (n={len(BA_late)})")
    _setup_axes(ax, title=f"Evoked E on channel A (ch {ch_A}), 2nd half",
                xlabel="time in sequence (s)", ylabel="rate")
    ax.legend(fontsize=8, frameon=False)

    # ---- (5) average evoked response, B channel ----
    ax = fig.add_subplot(gs[2, 1])
    if len(AB_late):
        ax.plot(ts, AB_late.mean(0)[ch_B], color="tab:green",
                lw=2, label=f"AB  (n={len(AB_late)})")
    if len(BA_late):
        ax.plot(ts, BA_late.mean(0)[ch_B], color="tab:purple", ls="--",
                lw=2, label=f"BA  (n={len(BA_late)})")
    _setup_axes(ax, title=f"Evoked E on channel B (ch {ch_B}), 2nd half",
                xlabel="time in sequence (s)", ylabel="rate")
    ax.legend(fontsize=8, frameon=False)

    # ---- (6) W trajectory: key directional weights ----
    ax = fig.add_subplot(gs[3, 0])
    if Wt.size:
        ax.plot(W_t, Wt[:, ch_B, ch_A], color="tab:green",
                lw=2,  label=r"$W_{B\leftarrow A}$  (AB direction)")
        ax.plot(W_t, Wt[:, ch_A, ch_B], color="tab:purple",
                lw=2,  label=r"$W_{A\leftarrow B}$  (BA direction)")
        ax.plot(W_t, Wt[:, ch_A, ch_A], color="tab:red",   lw=1, alpha=0.6,
                label=r"$W_{A\leftarrow A}$  self")
        ax.plot(W_t, Wt[:, ch_B, ch_B], color="tab:blue",  lw=1, alpha=0.6,
                label=r"$W_{B\leftarrow B}$  self")
    ax.legend(fontsize=8, frameon=False)
    _setup_axes(ax, title="Recurrent E->E weight evolution",
                xlabel="time (s)", ylabel="W")

    # ---- (7) learned W matrix ----
    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(W_f, cmap="viridis", aspect="equal", origin="upper",
                   vmin=0, vmax=max(W_f.max(), 1e-3))
    ax.scatter([ch_A], [ch_B], facecolors="none", edgecolors="lime",   s=120, lw=2,
               label=r"$W_{B\leftarrow A}$")
    ax.scatter([ch_B], [ch_A], facecolors="none", edgecolors="magenta", s=120, lw=2,
               label=r"$W_{A\leftarrow B}$")
    ax.set_xlabel("pre", fontsize=10); ax.set_ylabel("post", fontsize=10)
    ax.set_title("Final W (row=post, col=pre)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_comparison(res1: dict, res2: dict, fname: str):
    """Cross-run comparison: weights, SSA index, asymmetry."""
    cfg = res1["cfg"]; dt = cfg.dt
    ch_A, ch_B = res1["ch_A"], res1["ch_B"]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    fig.suptitle("Run 1 (90% AB)  vs  Run 2 (90% BA)",
                 fontsize=13, fontweight="bold")

    # ---- learned W, both runs ----
    Wmax = max(res1["W_final"].max(), res2["W_final"].max(), 1e-3)
    for k, (res, sub) in enumerate(((res1, gs[0, 0]), (res2, gs[0, 1]))):
        ax = fig.add_subplot(sub)
        im = ax.imshow(res["W_final"], cmap="viridis", origin="upper",
                       vmin=0, vmax=Wmax, aspect="equal")
        ax.scatter([ch_A], [ch_B], facecolors="none", edgecolors="lime",
                   s=120, lw=2)
        ax.scatter([ch_B], [ch_A], facecolors="none", edgecolors="magenta",
                   s=120, lw=2)
        ax.set_xlabel("pre"); ax.set_ylabel("post")
        ax.set_title(f"Run {k + 1}: p(AB)={res['p_AB']:.0%}",
                     fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- asymmetry index ----
    def asym(W):
        return (W[ch_B, ch_A] - W[ch_A, ch_B]) / (W[ch_B, ch_A] + W[ch_A, ch_B] + 1e-9)

    ax = fig.add_subplot(gs[0, 2])
    asyms = [asym(res1["W_final"]), asym(res2["W_final"])]
    bars  = ax.bar(["Run 1\n90% AB", "Run 2\n90% BA"], asyms,
                   color=["tab:green", "tab:purple"])
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel(r"asymmetry  $(W_{B\leftarrow A}-W_{A\leftarrow B})\,/\,$sum")
    ax.set_title("Sequence asymmetry", fontsize=11, fontweight="bold")
    for b, v in zip(bars, asyms):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    _setup_axes(ax)

    # ---- SSA: standard vs deviant evoked, on second-tone window ----
    # In each run the standard is the frequent sequence, deviant is the rare one.
    def evoked_means(res):
        ev = evoked_per_trial(res["E"], res["seq_starts"], res["n_seq"])
        codes = res["codes"]
        half  = len(codes) // 2
        # second-tone window inside the sequence:
        n_tone = int(round(50e-3 / dt))
        # tone 2 occupies samples [n_tone, 2*n_tone)
        win = slice(n_tone, 2 * n_tone)
        ev_AB = ev[half:][codes[half:] == "AB"]
        ev_BA = ev[half:][codes[half:] == "BA"]
        # peak rate of the second-tone channel during tone-2 window:
        peakA_of_BA = ev_BA[:, ch_A, win].max(axis=1) if len(ev_BA) else np.array([])
        peakB_of_AB = ev_AB[:, ch_B, win].max(axis=1) if len(ev_AB) else np.array([])
        return ev_AB, ev_BA, peakA_of_BA, peakB_of_AB

    ev1_AB, ev1_BA, peakA_BA_1, peakB_AB_1 = evoked_means(res1)
    ev2_AB, ev2_BA, peakA_BA_2, peakB_AB_2 = evoked_means(res2)

    # Run 1: standard = AB, deviant = BA.
    #   compare second-tone responses: B as standard (in AB) vs A as deviant (in BA)
    # Run 2: standard = BA, deviant = AB.  Roles swap.
    def ssa_idx(dev, std):
        d, s = np.mean(dev), np.mean(std)
        return (d - s) / (d + s + 1e-9)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(["std (B in AB)", "dev (A in BA)"],
           [peakB_AB_1.mean(), peakA_BA_1.mean()],
           color=["tab:green", "tab:purple"])
    ax.set_title(f"Run 1 2nd-tone peak\nSI = {ssa_idx(peakA_BA_1, peakB_AB_1):+.2f}",
                 fontsize=11, fontweight="bold")
    _setup_axes(ax, ylabel="peak E rate")

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(["std (A in BA)", "dev (B in AB)"],
           [peakA_BA_2.mean(), peakB_AB_2.mean()],
           color=["tab:purple", "tab:green"])
    ax.set_title(f"Run 2 2nd-tone peak\nSI = {ssa_idx(peakB_AB_2, peakA_BA_2):+.2f}",
                 fontsize=11, fontweight="bold")
    _setup_axes(ax, ylabel="peak E rate")

    # ---- overlay of mean responses, both runs ----
    ax = fig.add_subplot(gs[1, 2])
    n_seq = res1["n_seq"]; ts = np.arange(n_seq) * dt
    if len(ev1_AB):
        ax.plot(ts, ev1_AB.mean(0)[ch_B], color="tab:green", lw=2,
                label="Run1: B in AB (std)")
    if len(ev1_BA):
        ax.plot(ts, ev1_BA.mean(0)[ch_A], color="tab:purple", lw=2,
                label="Run1: A in BA (dev)")
    if len(ev2_BA):
        ax.plot(ts, ev2_BA.mean(0)[ch_A], color="tab:purple", lw=2, ls="--",
                label="Run2: A in BA (std)")
    if len(ev2_AB):
        ax.plot(ts, ev2_AB.mean(0)[ch_B], color="tab:green", lw=2, ls="--",
                label="Run2: B in AB (dev)")
    ax.legend(fontsize=7, frameon=False)
    _setup_axes(ax, title="Second-tone response (mean, 2nd half)",
                xlabel="time in sequence (s)", ylabel="E rate")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_surprise(res_std: dict, res_dev: dict, fname: str,
                  tone_dur: float = 50e-3):
    """Surprise contrast for the AB sequence.

    The physical stimulus (AB) is identical in both runs.  Only the
    *expectation* differs:  AB is the **standard** in ``res_std`` (Run 1,
    90% AB) but a rare **deviant** in ``res_dev`` (Run 2, 10% AB).
    The DEV-minus-STD trace isolates the expectation effect — the
    rate-coded analogue of the auditory mismatch response
    (Naatanen+2007 ; Parras+2017 ; Cooke+2018).
    """
    cfg = res_std["cfg"]; dt = cfg.dt
    ch_A, ch_B = res_std["ch_A"], res_std["ch_B"]
    n_seq = res_std["n_seq"]
    ts = np.arange(n_seq) * dt
    n_tone = int(round(tone_dur / dt))

    # post-learning AB epochs from each run
    ev_std = evoked_per_trial(res_std["E"], res_std["seq_starts"], n_seq)
    ev_dev = evoked_per_trial(res_dev["E"], res_dev["seq_starts"], n_seq)
    half_s = len(res_std["codes"]) // 2
    half_d = len(res_dev["codes"]) // 2
    AB_std = ev_std[half_s:][res_std["codes"][half_s:] == "AB"]
    AB_dev = ev_dev[half_d:][res_dev["codes"][half_d:] == "AB"]

    if len(AB_std) == 0 or len(AB_dev) == 0:
        raise RuntimeError("Not enough AB trials in one of the runs.")

    mean_std = AB_std.mean(0)                # (N, n_seq)
    mean_dev = AB_dev.mean(0)
    diff     = mean_dev - mean_std            # surprise = DEV - STD

    # population mean (all channels) and channel-B specific
    pop_std  = mean_std.mean(0)
    pop_dev  = mean_dev.mean(0)
    pop_diff = diff.mean(0)
    B_std    = mean_std[ch_B]
    B_dev    = mean_dev[ch_B]
    B_diff   = diff[ch_B]

    def _shade_tones(ax):
        ax.axvspan(0,            n_tone * dt, color="tab:red",  alpha=0.08,
                   label="tone A")
        ax.axvspan(n_tone * dt,  2 * n_tone * dt, color="tab:blue", alpha=0.08,
                   label="tone B")

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        f"Surprise contrast for the AB sequence  (DEV − STD)\n"
        f"STD: Run 1, p(AB)=90%   n={len(AB_std)}     "
        f"DEV: Run 2, p(AB)=10%   n={len(AB_dev)}",
        fontsize=12, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 2)

    # ---- (TL) population mean — STD vs DEV ----
    ax = fig.add_subplot(gs[0, 0])
    _shade_tones(ax)
    ax.plot(ts, pop_std, color="tab:blue", lw=2, label="STD (Run 1)")
    ax.plot(ts, pop_dev, color="tab:red",  lw=2, label="DEV (Run 2)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="All-channel mean E rate",
                xlabel="time in sequence (s)", ylabel=r"$\langle E_i\rangle_i$")

    # ---- (TR) population surprise (DEV - STD) ----
    ax = fig.add_subplot(gs[0, 1])
    _shade_tones(ax)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.plot(ts, pop_diff, color="tab:purple", lw=2)
    ax.fill_between(ts, 0, pop_diff,
                    where=(pop_diff > 0), color="tab:red",
                    alpha=0.25, label="DEV > STD")
    ax.fill_between(ts, 0, pop_diff,
                    where=(pop_diff < 0), color="tab:blue",
                    alpha=0.25, label="DEV < STD")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Surprise  (DEV − STD, all-channel mean)",
                xlabel="time in sequence (s)", ylabel=r"$\Delta\,\langle E\rangle$")

    # ---- (BL) channel B — STD vs DEV ----
    ax = fig.add_subplot(gs[1, 0])
    _shade_tones(ax)
    ax.plot(ts, B_std, color="tab:blue", lw=2, label="STD (Run 1)")
    ax.plot(ts, B_dev, color="tab:red",  lw=2, label="DEV (Run 2)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title=f"Channel B (ch {ch_B}) E rate",
                xlabel="time in sequence (s)", ylabel=r"$E_B$")

    # ---- (BR) channel B surprise ----
    ax = fig.add_subplot(gs[1, 1])
    _shade_tones(ax)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.plot(ts, B_diff, color="tab:purple", lw=2)
    ax.fill_between(ts, 0, B_diff,
                    where=(B_diff > 0), color="tab:red",
                    alpha=0.25)
    ax.fill_between(ts, 0, B_diff,
                    where=(B_diff < 0), color="tab:blue",
                    alpha=0.25)
    _setup_axes(ax, title=r"Surprise on channel B  ($E_B^{\rm DEV} - E_B^{\rm STD}$)",
                xlabel="time in sequence (s)", ylabel=r"$\Delta E_B$")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    cfg = A1Config(N=8)

    print("[ Run 1 ]  90% AB / 10% BA")
    res1 = run_experiment(p_AB=0.90, n_trials=400, seed=1, cfg=cfg,
                          ch_A=2, ch_B=5)
    print("[ Run 2 ]  10% AB / 90% BA")
    res2 = run_experiment(p_AB=0.10, n_trials=400, seed=2, cfg=cfg,
                          ch_A=2, ch_B=5)

    print("[ Plotting ]")
    plot_run(res1, "Run 1 — 90% AB (standard) / 10% BA (deviant)",
             "ab_ba_run1.png")
    plot_run(res2, "Run 2 — 10% AB (deviant)  / 90% BA (standard)",
             "ab_ba_run2.png")
    plot_comparison(res1, res2, "ab_ba_comparison.png")
    plot_surprise(res1, res2, "ab_ba_surprise.png")

    # ---- text summary ----
    def W_pair(W):
        return W[res1["ch_B"], res1["ch_A"]], W[res1["ch_A"], res1["ch_B"]]
    wba1, wab1 = W_pair(res1["W_final"])
    wba2, wab2 = W_pair(res2["W_final"])
    print("\nLearned recurrent weights (B<-A , A<-B):")
    print(f"  Run 1 (90% AB):  W[B<-A] = {wba1:.3f}   W[A<-B] = {wab1:.3f}"
          f"   asym = {(wba1 - wab1):+.3f}")
    print(f"  Run 2 (90% BA):  W[B<-A] = {wba2:.3f}   W[A<-B] = {wab2:.3f}"
          f"   asym = {(wba2 - wab2):+.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
