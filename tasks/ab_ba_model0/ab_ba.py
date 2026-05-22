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
    """Four-row summary of one oddball run.

    Rows
    ----
    1. Stimulus raster (first 6 trials).
    2. Activity   — trial-averaged firing rate E of channels A and B.
    3. Input      — trial-averaged inhibitory current onto E (M_IE @ I).
    4. Recurrent E->E weight evolution and final W.

    Rows 2-3 are averaged separately over the post-learning (2nd-half)
    AB trials (green) and BA trials (purple) — i.e. the stimulus-locked
    trial average, aligned to sequence onset.
    """
    cfg = res["cfg"]; dt = cfg.dt
    E   = res["E"]
    inh = res["inh_to_E"]
    ch_A, ch_B = res["ch_A"], res["ch_B"]
    n_seq = res["n_seq"]
    stim = res["stim"]

    n_show = 6
    show_T = n_show * n_seq

    ev_E   = evoked_per_trial(E,   res["seq_starts"], n_seq)
    ev_inh = evoked_per_trial(inh, res["seq_starts"], n_seq)
    codes = res["codes"]
    is_AB = codes == "AB"
    half = len(codes) // 2
    AB_E,   BA_E   = ev_E[half:][is_AB[half:]],   ev_E[half:][~is_AB[half:]]
    AB_inh, BA_inh = ev_inh[half:][is_AB[half:]], ev_inh[half:][~is_AB[half:]]
    ts = np.arange(n_seq) * dt

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else np.empty((0, cfg.N, cfg.N))
    W_t = res["W_t"]
    W_f = res["W_final"]

    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # ---- row 1: stimulus raster ----
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

    # ---- row 2: ACTIVITY — firing rate E, channel A and channel B ----
    for col, ch, name in [(0, ch_A, "A"), (1, ch_B, "B")]:
        ax = fig.add_subplot(gs[1, col])
        _shade_tones(ax, n_seq, dt)
        if len(AB_E):
            ax.plot(ts, AB_E.mean(0)[ch], color="tab:green", lw=2,
                    label=f"AB trial (n={len(AB_E)})")
        if len(BA_E):
            ax.plot(ts, BA_E.mean(0)[ch], color="tab:purple", ls="--", lw=2,
                    label=f"BA trial (n={len(BA_E)})")
        ax.legend(fontsize=8, frameon=False)
        _setup_axes(ax, title=f"Activity — firing rate, channel {name}",
                    xlabel="time in sequence (s)", ylabel=f"$E_{name}$")

    # ---- row 3: INPUT — inhibitory current onto E, channel A and B ----
    for col, ch, name in [(0, ch_A, "A"), (1, ch_B, "B")]:
        ax = fig.add_subplot(gs[2, col])
        _shade_tones(ax, n_seq, dt)
        if len(AB_inh):
            ax.plot(ts, AB_inh.mean(0)[ch], color="tab:green", lw=2,
                    label=f"AB trial (n={len(AB_inh)})")
        if len(BA_inh):
            ax.plot(ts, BA_inh.mean(0)[ch], color="tab:purple", ls="--", lw=2,
                    label=f"BA trial (n={len(BA_inh)})")
        ax.legend(fontsize=8, frameon=False)
        _setup_axes(ax, title=f"Input — inhibition onto channel {name}",
                    xlabel="time in sequence (s)",
                    ylabel=f"inh $\\rightarrow E_{name}$")

    # ---- row 4: weight evolution + final W ----
    ax = fig.add_subplot(gs[3, 0])
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

    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(W_f, cmap="viridis", aspect="equal", origin="upper",
                   vmin=0, vmax=max(W_f.max(), 1e-3))
    ax.set_xticks([ch_A, ch_B]); ax.set_xticklabels(["A", "B"])
    ax.set_yticks([ch_A, ch_B]); ax.set_yticklabels(["A", "B"])
    ax.set_xlabel("pre", fontsize=10); ax.set_ylabel("post", fontsize=10)
    ax.set_title("Final W", fontsize=11, fontweight="bold")
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


def plot_recurrent_masking(res: dict, fname: str):
    """Diagnostic: the learned A->B recurrent input is delivered to channel
    B in *both* the AB and the BA trial — but it produces firing only in AB.

    Mechanism.  ``rec_E[B] = W[B<-A] * E_A`` is the same in both trials
    (same learned weight, same tone-A drive).  What differs is B's own
    tone-selective interneuron ``I_B``:

      - AB trial:  tone A is FIRST, so B has not fired yet -> I_B ~ 0 ->
        the A->B current passes the relu and pre-activates B.
      - BA trial:  tone A is SECOND; B fired as the first tone and loaded
        I_B (tau_I = 80 ms), so I_B is still high when tone A arrives ->
        the SAME A->B current is shunted below threshold -> E_B ~ 0.

    This is the selective-inhibition mechanism (the same one that gives the
    MMN) and the rate-model firing-vs-current distinction, in one figure.

    Uses the post-learning (2nd-half) AB and BA epochs of Run 1.
    """
    cfg = res["cfg"]; dt = cfg.dt
    ch_A, ch_B = res["ch_A"], res["ch_B"]
    n_seq = res["n_seq"]
    ts = np.arange(n_seq) * dt
    n_tone  = int(round(50e-3 / dt))
    n_intra = int(round(30e-3 / dt))

    def mean_by_code(key, code):
        ev = evoked_per_trial(res[key], res["seq_starts"], n_seq)
        half = len(res["codes"]) // 2
        sel = ev[half:][res["codes"][half:] == code]
        return sel.mean(0)

    keys = ["E", "I", "rec_E", "inh_to_E", "tm_in"]
    AB = {k: mean_by_code(k, "AB") for k in keys}
    BA = {k: mean_by_code(k, "BA") for k in keys}
    AB["net"] = AB["tm_in"] + AB["rec_E"] - AB["inh_to_E"]
    BA["net"] = BA["tm_in"] + BA["rec_E"] - BA["inh_to_E"]

    # tone-A window per trial type (A is 1st tone in AB, 2nd in BA)
    wA = {"AB": slice(0, n_tone),
          "BA": slice(n_tone + n_intra, 2 * n_tone + n_intra)}
    xmax  = (2 * n_tone + n_intra) * dt + 0.05
    W_ba  = res["W_final"][ch_B, ch_A]

    def shade(ax, code):
        t1 = (0, n_tone)
        t2 = (n_tone + n_intra, 2 * n_tone + n_intra)
        c1 = "tab:red"  if code == "AB" else "tab:blue"
        c2 = "tab:blue" if code == "AB" else "tab:red"
        ax.axvspan(t1[0] * dt, t1[1] * dt, color=c1, alpha=0.09)
        ax.axvspan(t2[0] * dt, t2[1] * dt, color=c2, alpha=0.09)
        ax.axvline(wA[code].start * dt, color="tab:red", lw=1.0, ls=":")

    fig = plt.figure(figsize=(13, 16), constrained_layout=True)
    gs = fig.add_gridspec(6, 2)
    fig.suptitle(
        "The learned A→B input reaches channel B in BOTH trials — "
        "but is masked in BA by B's own interneuron\n"
        f"Run 1 (90% AB), post-learning,  W[B←A] = {W_ba:.2f}    "
        "(red band = tone A, blue band = tone B, dotted = tone-A onset)",
        fontsize=11.5, fontweight="bold")

    for col, (code, D, head) in enumerate([
            ("AB", AB, "AB trial (standard) — tone A 1st, B is fresh"),
            ("BA", BA, "BA trial (deviant) — tone A 2nd, B just fired")]):
        wAc = wA[code]
        tA0 = wAc.start * dt

        # --- row 1: E rates ---
        ax = fig.add_subplot(gs[0, col])
        shade(ax, code)
        ax.plot(ts, D["E"][ch_A], color="tab:red",  lw=1.8, label="$E_A$")
        ax.plot(ts, D["E"][ch_B], color="tab:blue", lw=1.8, label="$E_B$")
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        _setup_axes(ax, title=head, ylabel="rate")

        # --- row 2: rec_E[B] — the A->B recurrent current ---
        ax = fig.add_subplot(gs[1, col])
        shade(ax, code)
        ax.plot(ts, D["rec_E"][ch_B], color="tab:green", lw=2.3)
        ax.set_xlim(0, xmax)
        pk = D["rec_E"][ch_B, wAc].max()
        ax.annotate(f"peak = {pk:.2f}",
                    xy=(tA0 + n_tone * dt * 0.5, pk),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9.5, color="tab:green",
                    fontweight="bold")
        _setup_axes(ax, title=r"A→B recurrent current   "
                              r"$rec_E[B]=W[B{\leftarrow}A]\cdot E_A$",
                    ylabel="current")

        # --- row 3: I_B — B's interneuron ---
        ax = fig.add_subplot(gs[2, col])
        shade(ax, code)
        ax.plot(ts, D["I"][ch_B], color="tab:purple", lw=2.3)
        ax.set_xlim(0, xmax)
        i_at = D["I"][ch_B, wAc.start]
        ax.annotate(f"$I_B$ = {i_at:.2f}\nat tone-A onset",
                    xy=(tA0, i_at), xytext=(12, 4),
                    textcoords="offset points", fontsize=9.5,
                    color="tab:purple", fontweight="bold")
        _setup_axes(ax, title=r"B's tone-selective interneuron   $I_B$",
                    ylabel="I rate")

        # --- row 4: inh_to_E[B] vs rec_E[B] ---
        ax = fig.add_subplot(gs[3, col])
        shade(ax, code)
        ax.plot(ts, D["inh_to_E"][ch_B], color="tab:red", lw=2.3,
                label=r"inh$\rightarrow E_B$")
        ax.plot(ts, D["rec_E"][ch_B], color="tab:green", lw=1.5, ls="--",
                label="rec_E[B] (same scale)")
        ax.set_xlim(0, xmax)
        inh_at = D["inh_to_E"][ch_B, wAc.start]
        ax.annotate(f"inh = {inh_at:.1f}\nat tone-A onset",
                    xy=(tA0, inh_at), xytext=(12, 2),
                    textcoords="offset points", fontsize=9.5,
                    color="tab:red", fontweight="bold")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        _setup_axes(ax, title=r"Inhibitory current onto $E_B$   "
                              r"$inh=M_{IE}\cdot I$",
                    ylabel="current")

        # --- row 5: net current balance ---
        ax = fig.add_subplot(gs[4, col])
        shade(ax, code)
        ax.axhline(0, color="0.3", lw=0.8)
        exc = D["tm_in"][ch_B] + D["rec_E"][ch_B]
        ax.fill_between(ts, 0, exc, color="tab:green", alpha=0.35,
                        label="excitation (TC + rec)")
        ax.fill_between(ts, 0, -D["inh_to_E"][ch_B], color="tab:red",
                        alpha=0.30, label="− inhibition")
        ax.plot(ts, D["net"][ch_B], color="black", lw=2.0, label="net drive")
        ax.plot(ts, D["E"][ch_B], color="tab:blue", lw=2.3,
                label="$E_B$ (firing)")
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        _setup_axes(ax, title=r"Current balance at $E_B$:  "
                              r"net $=$ TC $+$ rec $-$ inh",
                    xlabel="time in sequence (s)", ylabel="current / rate")

    # --- row 6: snapshot summary bar chart (spans both columns) ---
    ax = fig.add_subplot(gs[5, :])
    mid = {c: (wA[c].start + wA[c].stop) // 2 for c in ("AB", "BA")}
    labels = [r"$rec_E[B]$"     "\n(A→B drive)",
              r"$inh\rightarrow E_B$" "\n(inhibition)",
              r"net$_E[B]$"      "\n(TC+rec−inh)",
              r"$E_B$"          "\n(firing rate)"]
    vals_AB = [AB["rec_E"][ch_B, mid["AB"]], AB["inh_to_E"][ch_B, mid["AB"]],
               AB["net"][ch_B, mid["AB"]],   AB["E"][ch_B, mid["AB"]]]
    vals_BA = [BA["rec_E"][ch_B, mid["BA"]], BA["inh_to_E"][ch_B, mid["BA"]],
               BA["net"][ch_B, mid["BA"]],   BA["E"][ch_B, mid["BA"]]]
    x = np.arange(len(labels)); w = 0.36
    b1 = ax.bar(x - w / 2, vals_AB, w, color="tab:green",  label="AB trial")
    b2 = ax.bar(x + w / 2, vals_BA, w, color="tab:purple", label="BA trial")
    ax.axhline(0, color="0.3", lw=0.8)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:.2f}",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -12),
                        textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _setup_axes(ax, title="Snapshot at mid-tone-A: identical A→B drive, "
                          "opposite outcome — inhibition is the only thing "
                          "that changed",
                ylabel="current / rate")

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
    plot_recurrent_masking(res1, "m0_ab_ba_recurrent_masking.png")

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
