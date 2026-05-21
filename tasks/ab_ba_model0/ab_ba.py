"""
tasks.ab_ba_model0.ab_ba
========================

AB-vs-BA oddball paradigm for the **tone-selective inhibition** A1 model
(``model0``).

Differences from ``tasks.ab_ba``
-------------------------------
1. Imports ``A1Config`` / ``simulate`` from ``model0`` instead of ``model``.
2. ``out['I']`` is now per-channel (N, T) rather than scalar (T,), so the
   "global I" panel becomes per-channel ``I[A]`` and ``I[B]`` traces.
3. ``intra_gap = 30 ms`` (was 0).  The selective-inhibition mechanism
   requires a brief gap so that I_B can outlive E_B and still be high
   when tone B arrives.  Biologically this just acknowledges that real
   tone sequences have some inter-tone interval.
4. An extra ``plot_inhibition_timing`` figure shows the diagnostic of
   the new mechanism: I_B is elevated at the tone-B onset in STD but
   not in DEV.
5. Output filenames are prefixed ``m0_`` so they do not clobber the
   ``model/`` results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Run-as-script support: `python tasks/ab_ba_model0/ab_ba.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, simulate


# =====================================================================
#  Stimulus
# =====================================================================
def _tuning(N: int, centre: int, sigma: float) -> np.ndarray:
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
    intra_gap: float = 30e-3,         # NEW DEFAULT: 30 ms — see module docstring
    inter_gap: float = 500e-3,
    tone_amp: float = 1.0,
    tuning_sigma: float = 0.0,        # delta tuning — no tonotopic overlap
) -> Tuple[np.ndarray, np.ndarray, int]:
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
            stim[:, s             : s + n_tone]                      = tA[:, None]
            stim[:, s + n_tone + n_intra : s + 2 * n_tone + n_intra] = tB[:, None]
        elif code == "BA":
            stim[:, s             : s + n_tone]                      = tB[:, None]
            stim[:, s + n_tone + n_intra : s + 2 * n_tone + n_intra] = tA[:, None]
        else:
            raise ValueError(f"bad seq code: {code!r}")

    return stim, starts, n_seq


def shuffled_codes(n_total: int, p_AB: float, rng: np.random.Generator) -> List[str]:
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
    ch_A: int = 0,
    ch_B: int = 1,
) -> dict:
    if cfg is None:
        cfg = A1Config()

    rng = np.random.default_rng(seed)
    codes = shuffled_codes(n_trials, p_AB, rng)
    stim, starts, n_seq = build_stim(codes, cfg, ch_A, ch_B)

    snap_every = max(1, int(round(0.5 / cfg.dt)))
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(
        stim=stim, codes=np.array(codes), seq_starts=starts, n_seq=n_seq,
        p_AB=p_AB, ch_A=ch_A, ch_B=ch_B,
    )
    return out


def evoked_per_trial(arr: np.ndarray, starts: np.ndarray, n_seq: int) -> np.ndarray:
    """Cut a (N, T) or (T,) history into per-trial epochs of length n_seq."""
    if arr.ndim == 1:
        out = np.empty((len(starts), n_seq))
        for k, s in enumerate(starts):
            out[k] = arr[s : s + n_seq]
    else:
        out = np.empty((len(starts), arr.shape[0], n_seq))
        for k, s in enumerate(starts):
            out[k] = arr[:, s : s + n_seq]
    return out


# =====================================================================
#  Plotting helpers
# =====================================================================
def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _tone_windows(n_seq: int, dt: float, intra_gap: float = 30e-3, tone_dur: float = 50e-3):
    n_tone = int(round(tone_dur / dt))
    n_intra = int(round(intra_gap / dt))
    return (0, n_tone), (n_tone + n_intra, 2 * n_tone + n_intra)


def _shade_tones(ax, n_seq, dt, intra_gap=30e-3, tone_dur=50e-3):
    (a0, a1), (b0, b1) = _tone_windows(n_seq, dt, intra_gap, tone_dur)
    ax.axvspan(a0 * dt, a1 * dt, color="tab:red",  alpha=0.08, label="tone A")
    ax.axvspan(b0 * dt, b1 * dt, color="tab:blue", alpha=0.08, label="tone B")


# =====================================================================
#  Plotting
# =====================================================================
def plot_run(res: dict, suptitle: str, fname: str):
    cfg = res["cfg"]; dt = cfg.dt
    E, I = res["E"], res["I"]
    ch_A, ch_B = res["ch_A"], res["ch_B"]
    n_seq = res["n_seq"]
    stim = res["stim"]

    n_show = 6
    show_T = n_show * n_seq
    t_show = res["t"][:show_T]

    ev = evoked_per_trial(E, res["seq_starts"], n_seq)
    evI = evoked_per_trial(I, res["seq_starts"], n_seq)
    codes = res["codes"]
    is_AB = codes == "AB"
    half = len(codes) // 2
    AB_late = ev[half:][is_AB[half:]]
    BA_late = ev[half:][~is_AB[half:]]
    AB_late_I = evI[half:][is_AB[half:]]
    BA_late_I = evI[half:][~is_AB[half:]]
    ts = np.arange(n_seq) * dt

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else np.empty((0, cfg.N, cfg.N))
    W_t = res["W_t"]
    W_f = res["W_final"]

    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    gs = fig.add_gridspec(5, 2)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # (1) stimulus raster
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(stim[:, :show_T], aspect="auto", origin="lower",
              cmap="Oranges", interpolation="nearest",
              extent=[0, show_T * dt, -0.5, cfg.N - 0.5])
    for k in range(n_show):
        ax.axvline(k * n_seq * dt, color="0.4", lw=0.5, ls=":")
        ax.text((k + 0.5) * n_seq * dt, cfg.N - 0.2, codes[k],
                ha="center", va="top", fontsize=9, color="0.2")
    ax.axhline(ch_A, color="tab:red",  lw=0.6, ls="--", alpha=0.5)
    ax.axhline(ch_B, color="tab:blue", lw=0.6, ls="--", alpha=0.5)
    _setup_axes(ax, title=f"Stimulus raster (first {n_show} trials)",
                xlabel="time (s)", ylabel="channel")

    # (2) E rate on A and B
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_show, E[ch_A, :show_T], color="tab:red",  lw=1.2, label=f"E[A={ch_A}]")
    ax.plot(t_show, E[ch_B, :show_T], color="tab:blue", lw=1.2, label=f"E[B={ch_B}]")
    ax.legend(fontsize=8, frameon=False)
    _setup_axes(ax, title="Excitatory rate at A and B",
                xlabel="time (s)", ylabel="rate")

    # (3) I rate on A and B  (replaces "global I")
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_show, I[ch_A, :show_T], color="tab:red",  lw=1.2, label=f"I[A={ch_A}]")
    ax.plot(t_show, I[ch_B, :show_T], color="tab:blue", lw=1.2, label=f"I[B={ch_B}]")
    ax.legend(fontsize=8, frameon=False)
    _setup_axes(ax, title="Tone-selective inhibitory rates",
                xlabel="time (s)", ylabel="I rate")

    # (4) evoked E on A
    ax = fig.add_subplot(gs[2, 0])
    _shade_tones(ax, n_seq, dt)
    if len(AB_late):
        ax.plot(ts, AB_late.mean(0)[ch_A], color="tab:green",  lw=2, label=f"AB (n={len(AB_late)})")
    if len(BA_late):
        ax.plot(ts, BA_late.mean(0)[ch_A], color="tab:purple", ls="--", lw=2, label=f"BA (n={len(BA_late)})")
    _setup_axes(ax, title=f"Evoked E on channel A (ch {ch_A})",
                xlabel="time in sequence (s)", ylabel="rate")
    ax.legend(fontsize=8, frameon=False)

    # (5) evoked E on B
    ax = fig.add_subplot(gs[2, 1])
    _shade_tones(ax, n_seq, dt)
    if len(AB_late):
        ax.plot(ts, AB_late.mean(0)[ch_B], color="tab:green",  lw=2, label=f"AB (n={len(AB_late)})")
    if len(BA_late):
        ax.plot(ts, BA_late.mean(0)[ch_B], color="tab:purple", ls="--", lw=2, label=f"BA (n={len(BA_late)})")
    _setup_axes(ax, title=f"Evoked E on channel B (ch {ch_B})",
                xlabel="time in sequence (s)", ylabel="rate")
    ax.legend(fontsize=8, frameon=False)

    # (6) evoked I on B — diagnostic of the new mechanism
    ax = fig.add_subplot(gs[3, 0])
    _shade_tones(ax, n_seq, dt)
    if len(AB_late_I):
        ax.plot(ts, AB_late_I.mean(0)[ch_B], color="tab:green",  lw=2, label=f"AB (n={len(AB_late_I)})")
    if len(BA_late_I):
        ax.plot(ts, BA_late_I.mean(0)[ch_B], color="tab:purple", ls="--", lw=2, label=f"BA (n={len(BA_late_I)})")
    _setup_axes(ax, title=f"Evoked I on channel B (predicted-tone interneuron)",
                xlabel="time in sequence (s)", ylabel="I rate")
    ax.legend(fontsize=8, frameon=False)

    # (7) evoked I on A
    ax = fig.add_subplot(gs[3, 1])
    _shade_tones(ax, n_seq, dt)
    if len(AB_late_I):
        ax.plot(ts, AB_late_I.mean(0)[ch_A], color="tab:green",  lw=2, label=f"AB (n={len(AB_late_I)})")
    if len(BA_late_I):
        ax.plot(ts, BA_late_I.mean(0)[ch_A], color="tab:purple", ls="--", lw=2, label=f"BA (n={len(BA_late_I)})")
    _setup_axes(ax, title=f"Evoked I on channel A",
                xlabel="time in sequence (s)", ylabel="I rate")
    ax.legend(fontsize=8, frameon=False)

    # (8) W trajectory
    ax = fig.add_subplot(gs[4, 0])
    if Wt.size:
        ax.plot(W_t, Wt[:, ch_B, ch_A], color="tab:green", lw=2,
                label=r"$W_{B\leftarrow A}$  (AB direction)")
        ax.plot(W_t, Wt[:, ch_A, ch_B], color="tab:purple", lw=2,
                label=r"$W_{A\leftarrow B}$  (BA direction)")
        ax.plot(W_t, Wt[:, ch_A, ch_A], color="tab:red",  lw=1, alpha=0.6,
                label=r"$W_{A\leftarrow A}$  self")
        ax.plot(W_t, Wt[:, ch_B, ch_B], color="tab:blue", lw=1, alpha=0.6,
                label=r"$W_{B\leftarrow B}$  self")
    ax.legend(fontsize=8, frameon=False)
    _setup_axes(ax, title="Recurrent E->E weight evolution",
                xlabel="time (s)", ylabel="W")

    # (9) learned W matrix
    ax = fig.add_subplot(gs[4, 1])
    im = ax.imshow(W_f, cmap="viridis", aspect="equal", origin="upper",
                   vmin=0, vmax=max(W_f.max(), 1e-3))
    ax.scatter([ch_A], [ch_B], facecolors="none", edgecolors="lime",    s=120, lw=2,
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


def plot_surprise(res_std: dict, res_dev: dict, fname: str,
                  tone_dur: float = 50e-3):
    """STD-vs-DEV contrast for the AB sequence, channel-B focused."""
    cfg = res_std["cfg"]; dt = cfg.dt
    ch_A, ch_B = res_std["ch_A"], res_std["ch_B"]
    n_seq = res_std["n_seq"]
    ts = np.arange(n_seq) * dt

    ev_std = evoked_per_trial(res_std["E"], res_std["seq_starts"], n_seq)
    ev_dev = evoked_per_trial(res_dev["E"], res_dev["seq_starts"], n_seq)
    half_s = len(res_std["codes"]) // 2
    half_d = len(res_dev["codes"]) // 2
    AB_std = ev_std[half_s:][res_std["codes"][half_s:] == "AB"]
    AB_dev = ev_dev[half_d:][res_dev["codes"][half_d:] == "AB"]
    if len(AB_std) == 0 or len(AB_dev) == 0:
        raise RuntimeError("Not enough AB trials in one of the runs.")

    mean_std, mean_dev = AB_std.mean(0), AB_dev.mean(0)
    diff = mean_dev - mean_std
    pop_std, pop_dev, pop_diff = mean_std.mean(0), mean_dev.mean(0), diff.mean(0)
    B_std, B_dev, B_diff = mean_std[ch_B], mean_dev[ch_B], diff[ch_B]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        f"Surprise contrast — selective-inhibition model "
        f"(DEV − STD, AB sequence)\n"
        f"STD: Run 1, p(AB)=90%   n={len(AB_std)}     "
        f"DEV: Run 2, p(AB)=10%   n={len(AB_dev)}",
        fontsize=12, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, pop_std, color="tab:blue", lw=2, label="STD (Run 1)")
    ax.plot(ts, pop_dev, color="tab:red",  lw=2, label="DEV (Run 2)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="All-channel mean E rate",
                xlabel="time in sequence (s)", ylabel=r"$\langle E_i\rangle_i$")

    ax = fig.add_subplot(gs[0, 1])
    _shade_tones(ax, n_seq, dt)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.plot(ts, pop_diff, color="tab:purple", lw=2)
    ax.fill_between(ts, 0, pop_diff, where=(pop_diff > 0), color="tab:red",  alpha=0.25, label="DEV > STD")
    ax.fill_between(ts, 0, pop_diff, where=(pop_diff < 0), color="tab:blue", alpha=0.25, label="DEV < STD")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Surprise (DEV − STD, all-channel mean)",
                xlabel="time in sequence (s)", ylabel=r"$\Delta\,\langle E\rangle$")

    ax = fig.add_subplot(gs[1, 0])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, B_std, color="tab:blue", lw=2, label="STD (Run 1)")
    ax.plot(ts, B_dev, color="tab:red",  lw=2, label="DEV (Run 2)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title=f"Channel B (ch {ch_B}) E rate",
                xlabel="time in sequence (s)", ylabel=r"$E_B$")

    ax = fig.add_subplot(gs[1, 1])
    _shade_tones(ax, n_seq, dt)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.plot(ts, B_diff, color="tab:purple", lw=2)
    ax.fill_between(ts, 0, B_diff, where=(B_diff > 0), color="tab:red",  alpha=0.25)
    ax.fill_between(ts, 0, B_diff, where=(B_diff < 0), color="tab:blue", alpha=0.25)
    _setup_axes(ax, title=r"Surprise on channel B  ($E_B^{\rm DEV} - E_B^{\rm STD}$)",
                xlabel="time in sequence (s)", ylabel=r"$\Delta E_B$")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_inhibition_timing(res_std: dict, res_dev: dict, fname: str):
    """The mechanistic diagnostic: I_B at tone-B onset, STD vs DEV."""
    cfg = res_std["cfg"]; dt = cfg.dt
    ch_A, ch_B = res_std["ch_A"], res_std["ch_B"]
    n_seq = res_std["n_seq"]
    ts = np.arange(n_seq) * dt

    def mean_AB(res, key):
        ev = evoked_per_trial(res[key], res["seq_starts"], n_seq)
        half = len(res["codes"]) // 2
        sel = ev[half:][res["codes"][half:] == "AB"]
        return sel.mean(0) if len(sel) else None

    E_std, E_dev = mean_AB(res_std, "E"), mean_AB(res_dev, "E")
    I_std, I_dev = mean_AB(res_std, "I"), mean_AB(res_dev, "I")
    inh_std, inh_dev = mean_AB(res_std, "inh_to_E"), mean_AB(res_dev, "inh_to_E")

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "Selective-inhibition mechanism — predicted-tone interneuron carries the memory",
        fontsize=12, fontweight="bold",
    )
    gs = fig.add_gridspec(3, 2)

    # E[B] STD vs DEV
    ax = fig.add_subplot(gs[0, 0])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, E_std[ch_B], color="tab:blue", lw=2, label="STD (B is predicted)")
    ax.plot(ts, E_dev[ch_B], color="tab:red",  lw=2, label="DEV (B is unpredicted)")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax, title=f"E[B={ch_B}]  —  predicted vs unpredicted tone B",
                xlabel="time in sequence (s)", ylabel="rate")

    # I[B] STD vs DEV — THE KEY DIAGNOSTIC
    ax = fig.add_subplot(gs[0, 1])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, I_std[ch_B], color="tab:blue", lw=2, label="STD (B is predicted)")
    ax.plot(ts, I_dev[ch_B], color="tab:red",  lw=2, label="DEV (B is unpredicted)")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax,
                title=f"I[B={ch_B}]  —  pre-built during tone A under prediction",
                xlabel="time in sequence (s)", ylabel="I rate")

    # E[A] for context
    ax = fig.add_subplot(gs[1, 0])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, E_std[ch_A], color="tab:blue", lw=2, label="STD")
    ax.plot(ts, E_dev[ch_A], color="tab:red",  lw=2, label="DEV")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax, title=f"E[A={ch_A}]  (driver tone)",
                xlabel="time in sequence (s)", ylabel="rate")

    # I[A] for context
    ax = fig.add_subplot(gs[1, 1])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, I_std[ch_A], color="tab:blue", lw=2, label="STD")
    ax.plot(ts, I_dev[ch_A], color="tab:red",  lw=2, label="DEV")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax, title=f"I[A={ch_A}]",
                xlabel="time in sequence (s)", ylabel="I rate")

    # inh_to_E[B] — the effective subtractive current on E[B]
    ax = fig.add_subplot(gs[2, 0])
    _shade_tones(ax, n_seq, dt)
    ax.plot(ts, inh_std[ch_B], color="tab:blue", lw=2, label="STD")
    ax.plot(ts, inh_dev[ch_B], color="tab:red",  lw=2, label="DEV")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax,
                title=r"Inhibitory current onto $E_B$  ($M_{IE}\cdot I$)",
                xlabel="time in sequence (s)", ylabel="current")

    # Difference traces
    ax = fig.add_subplot(gs[2, 1])
    _shade_tones(ax, n_seq, dt)
    ax.axhline(0, color="0.4", lw=0.6)
    dI = I_std[ch_B] - I_dev[ch_B]
    dE = E_dev[ch_B] - E_std[ch_B]
    ax.plot(ts, dI, color="tab:purple", lw=2, label=r"$\Delta I_B$ (STD−DEV)")
    ax.plot(ts, dE, color="tab:orange", lw=2, label=r"$\Delta E_B$ (DEV−STD = suppression)")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax, title="Prediction-driven inhibitory load and the resulting E-suppression",
                xlabel="time in sequence (s)", ylabel="rate diff.")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    cfg = A1Config()

    print("[ Run 1 ]  90% AB / 10% BA")
    res1 = run_experiment(p_AB=0.90, n_trials=400, seed=1, cfg=cfg, ch_A=0, ch_B=1)
    print("[ Run 2 ]  10% AB / 90% BA")
    res2 = run_experiment(p_AB=0.10, n_trials=400, seed=2, cfg=cfg, ch_A=0, ch_B=1)

    print("[ Plotting ]")
    plot_run(res1, "Run 1 — 90% AB (standard) / 10% BA (deviant) — selective-inhibition model",
             "m0_ab_ba_run1.png")
    plot_run(res2, "Run 2 — 10% AB (deviant)  / 90% BA (standard) — selective-inhibition model",
             "m0_ab_ba_run2.png")
    plot_surprise(res1, res2, "m0_ab_ba_surprise.png")
    plot_inhibition_timing(res1, res2, "m0_ab_ba_inhibition.png")

    # ---- text summary ----
    ch_A, ch_B = res1["ch_A"], res1["ch_B"]
    def W_pair(W):
        return W[ch_B, ch_A], W[ch_A, ch_B]
    wba1, wab1 = W_pair(res1["W_final"])
    wba2, wab2 = W_pair(res2["W_final"])

    n_tone = int(round(50e-3 / cfg.dt))
    n_intra = int(round(30e-3 / cfg.dt))
    win_B = slice(n_tone + n_intra, 2 * n_tone + n_intra)

    def AB_mean_E(res):
        ev = evoked_per_trial(res["E"], res["seq_starts"], res["n_seq"])
        half = len(res["codes"]) // 2
        sel = ev[half:][res["codes"][half:] == "AB"]
        return sel.mean(0) if len(sel) else None

    abE_STD = AB_mean_E(res1); abE_DEV = AB_mean_E(res2)
    peak_STD = abE_STD[ch_B, win_B].max(); peak_DEV = abE_DEV[ch_B, win_B].max()
    mean_STD = abE_STD[ch_B, win_B].mean(); mean_DEV = abE_DEV[ch_B, win_B].mean()

    print("\nLearned recurrent weights (B<-A , A<-B):")
    print(f"  Run 1 (90% AB):  W[B<-A] = {wba1:.3f}   W[A<-B] = {wab1:.3f}"
          f"   asym = {(wba1 - wab1):+.3f}")
    print(f"  Run 2 (90% BA):  W[B<-A] = {wba2:.3f}   W[A<-B] = {wab2:.3f}"
          f"   asym = {(wba2 - wab2):+.3f}")
    print(f"\nTone-B response, channel B:")
    print(f"  peak  STD={peak_STD:.2f}  DEV={peak_DEV:.2f}  "
          f"-> suppression of predicted = {(peak_DEV-peak_STD)/peak_DEV*100:+.1f}%")
    print(f"  mean  STD={mean_STD:.2f}  DEV={mean_DEV:.2f}  "
          f"-> suppression of predicted = {(mean_DEV-mean_STD)/mean_DEV*100:+.1f}%")
    print("Done.")


if __name__ == "__main__":
    main()
