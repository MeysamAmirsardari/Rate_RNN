"""
local_global.py
===============

Local-Global oddball paradigm for model0 (Bekinschtein et al. 2009;
Wacongne et al. 2011).

Two five-tone sequence types, built from two tones x and y carried by
two tonotopic channels:

    xxxxx   -- five identical tones
    xxxxy   -- four identical tones + a different fifth tone

Timing:  each tone 50 ms; 100 ms gap between tones within a sequence;
1 s gap between sequences.  A sequence therefore spans
5*50 + 4*100 + 1000 = 1650 ms.

Two runs, with the standard/deviant roles swapped:

    Run 1:  standard = xxxxy (80%),  deviant = xxxxx (20%)
    Run 2:  standard = xxxxx (80%),  deviant = xxxxy (20%)

Each run has two phases:
    habituation : 20 trials, standard only        (not analysed)
    test        : 400 trials, 80% std / 20% dev, interleaved

Only the test phase is analysed -- its 2nd half (post-learning).

Three contrasts
---------------
    Local effect        :  STD xxxxy  -  STD xxxxx
        Both sequences are *global standards*, so the contrast isolates
        the local deviance -- the fifth tone being y vs x.

    Global effect (x)   :  DEV xxxxx  -  STD xxxxx
        The identical physical token xxxxx, heard as a rare deviant vs
        as the frequent standard -- the pure global (rarity) effect.

    Global effect (y)   :  DEV xxxxy  -  STD xxxxy
        The global effect for the xxxxy token.  (The task spec wrote
        this as "DEV xxxxy - DEV xxxxy", which is identically zero;
        read here as the obviously intended DEV - STD contrast.)

Model: model0, configuration unchanged from the AB/BA task.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Run-as-script support: `python tasks/local_global_model0/local_global.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, INH_PRESETS, simulate

# ---- fixed paradigm constants -------------------------------------------------
N_PER_SEQ = 5                  # tones per sequence
TONE_DUR  = 50e-3              # s
INTRA_GAP = 100e-3             # s  (between tones within a sequence)
INTER_GAP = 1000e-3           # s  (between sequences)


# =====================================================================
#  Stimulus
# =====================================================================
def build_lg_stim(
    seq_codes: List[str],
    cfg: A1Config,
    ch_x: int,
    ch_y: int,
    tone_dur: float = TONE_DUR,
    intra_gap: float = INTRA_GAP,
    inter_gap: float = INTER_GAP,
    tone_amp: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Assemble an (N, T) stimulus from a list of 'xxxxx'/'xxxxy' codes.

    Each character of a code selects the channel of that tone:
    'x' -> ch_x, 'y' -> ch_y.
    """
    dt = cfg.dt
    n_tone  = int(round(tone_dur  / dt))
    n_intra = int(round(intra_gap / dt))
    n_inter = int(round(inter_gap / dt))
    n_seq   = N_PER_SEQ * n_tone + (N_PER_SEQ - 1) * n_intra + n_inter
    T_total = n_seq * len(seq_codes)

    stim = np.zeros((cfg.N, T_total))
    starts = np.zeros(len(seq_codes), dtype=int)

    for k, code in enumerate(seq_codes):
        if len(code) != N_PER_SEQ:
            raise ValueError(f"bad sequence code: {code!r}")
        s = k * n_seq
        starts[k] = s
        for i, ch_char in enumerate(code):
            t0 = s + i * (n_tone + n_intra)
            ch = ch_x if ch_char == "x" else ch_y
            stim[ch, t0 : t0 + n_tone] = tone_amp

    return stim, starts, n_seq


# =====================================================================
#  Experiment
# =====================================================================
def run_experiment(
    std_type: str,
    p_std: float = 0.80,
    n_habituation: int = 20,
    n_test: int = 400,
    seed: int = 0,
    cfg: Optional[A1Config] = None,
    ch_x: int = 0,
    ch_y: int = 1,
) -> dict:
    """Run one local-global session: habituation phase + test phase.

    ``std_type`` is the standard sequence ('xxxxx' or 'xxxxy'); the
    other type is the deviant.
    """
    if cfg is None:
        cfg = A1Config()
    if std_type not in ("xxxxx", "xxxxy"):
        raise ValueError(f"std_type must be 'xxxxx' or 'xxxxy', got {std_type!r}")
    dev_type = "xxxxx" if std_type == "xxxxy" else "xxxxy"

    rng = np.random.default_rng(seed)

    # habituation: standard only
    habit = [std_type] * n_habituation
    # test: 80% standard / 20% deviant, interleaved
    n_std = int(round(n_test * p_std))
    test = np.array([std_type] * n_std + [dev_type] * (n_test - n_std))
    rng.shuffle(test)
    codes = habit + test.tolist()

    stim, starts, n_seq = build_lg_stim(codes, cfg, ch_x, ch_y)
    snap_every = max(1, int(round(0.5 / cfg.dt)))
    out = simulate(stim, cfg=cfg, record_W_every=snap_every, seed=seed)
    out.update(
        stim=stim, codes=np.array(codes), seq_starts=starts, n_seq=n_seq,
        n_habituation=n_habituation, n_test=n_test,
        std_type=std_type, dev_type=dev_type, p_std=p_std,
        ch_x=ch_x, ch_y=ch_y,
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


def condition_mean(res: dict, code: str, key: str = "E") -> Tuple[np.ndarray, int]:
    """Trial-averaged ``key`` history for sequences matching ``code``,
    over the 2nd half of the test phase (post-learning).

    Returns (mean (N, n_seq), n_trials_used).
    """
    ev = evoked_per_trial(res[key], res["seq_starts"], res["n_seq"])
    n_hab, n_test = res["n_habituation"], res["n_test"]
    lo = n_hab + n_test // 2                       # 2nd half of test phase
    sel = ev[lo:][res["codes"][lo:] == code]
    return sel.mean(0), len(sel)


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


def _shade_tones(ax, dt, highlight_last=True):
    """Shade the 5 tone windows; the 5th (critical) tone in purple."""
    n_tone  = int(round(TONE_DUR / dt))
    n_intra = int(round(INTRA_GAP / dt))
    for i in range(N_PER_SEQ):
        t0 = i * (n_tone + n_intra) * dt
        t1 = t0 + n_tone * dt
        is_last = (i == N_PER_SEQ - 1)
        if is_last and highlight_last:
            ax.axvspan(t0, t1, color="tab:purple", alpha=0.16)
        else:
            ax.axvspan(t0, t1, color="0.55", alpha=0.10)


def _active_span(dt: float) -> float:
    """Duration (s) of the active part of a sequence (5 tones + 4 gaps)."""
    n_tone  = int(round(TONE_DUR / dt))
    n_intra = int(round(INTRA_GAP / dt))
    return (N_PER_SEQ * n_tone + (N_PER_SEQ - 1) * n_intra) * dt


def _tone5_window(dt: float, post_ms: float = 150.0) -> slice:
    """Sample window for the response to the 5th (critical) tone."""
    n_tone  = int(round(TONE_DUR / dt))
    n_intra = int(round(INTRA_GAP / dt))
    t5_on = (N_PER_SEQ - 1) * (n_tone + n_intra)
    return slice(t5_on, t5_on + int(round(post_ms * 1e-3 / dt)))


# =====================================================================
#  Plotting — per-run summary
# =====================================================================
def plot_run(res: dict, suptitle: str, fname: str):
    """Three-row per-run summary: raster, activity, weight evolution."""
    cfg = res["cfg"]; dt = cfg.dt
    E = res["E"]
    ch_x, ch_y = res["ch_x"], res["ch_y"]
    n_seq = res["n_seq"]
    stim = res["stim"]
    std_type, dev_type = res["std_type"], res["dev_type"]

    # raster: first 5 test-phase trials (skip habituation so deviants appear)
    test0 = res["n_habituation"]
    show_lo = test0 * n_seq
    n_show = 5
    show_hi = show_lo + n_show * n_seq
    show_codes = res["codes"][test0 : test0 + n_show]

    ev_E = evoked_per_trial(E, res["seq_starts"], n_seq)
    n_hab, n_test = res["n_habituation"], res["n_test"]
    lo = n_hab + n_test // 2
    std_E = ev_E[lo:][res["codes"][lo:] == std_type]
    dev_E = ev_E[lo:][res["codes"][lo:] == dev_type]
    ts = np.arange(n_seq) * dt
    xmax = _active_span(dt) + 0.20

    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else np.empty((0, cfg.N, cfg.N))
    W_t = res["W_t"]
    W_f = res["W_final"]

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # ---- row 1: stimulus raster ----
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(stim[:, show_lo:show_hi], aspect="auto", origin="lower",
              cmap="Oranges", interpolation="nearest",
              extent=[0, n_show * n_seq * dt, -0.5, cfg.N - 0.5])
    for k in range(n_show):
        ax.axvline(k * n_seq * dt, color="0.4", lw=0.5, ls=":")
        ax.text((k + 0.5) * n_seq * dt, cfg.N - 0.15, show_codes[k],
                ha="center", va="top", fontsize=9, color="0.2")
    ax.axhline(ch_x, color="tab:red",  lw=0.6, ls="--", alpha=0.5)
    ax.axhline(ch_y, color="tab:blue", lw=0.6, ls="--", alpha=0.5)
    ax.set_yticks([ch_x, ch_y]); ax.set_yticklabels(["x", "y"])
    _setup_axes(ax, title=f"Stimulus raster (first {n_show} test trials)",
                xlabel="time (s)", ylabel="channel")

    # ---- row 2: trial-averaged activity ----
    # left: all-channel mean E
    ax = fig.add_subplot(gs[1, 0])
    _shade_tones(ax, dt)
    if len(std_E):
        ax.plot(ts, std_E.mean(0).mean(0), color="tab:blue", lw=2,
                label=f"standard {std_type} (n={len(std_E)})")
    if len(dev_E):
        ax.plot(ts, dev_E.mean(0).mean(0), color="tab:red", lw=2, ls="--",
                label=f"deviant {dev_type} (n={len(dev_E)})")
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _setup_axes(ax, title="Activity — all-channel mean firing rate",
                xlabel="time in sequence (s)", ylabel=r"$\langle E\rangle$")

    # right: per-channel E for the standard
    ax = fig.add_subplot(gs[1, 1])
    _shade_tones(ax, dt)
    if len(std_E):
        m = std_E.mean(0)
        ax.plot(ts, m[ch_x], color="tab:red",  lw=2, label="$E_x$")
        ax.plot(ts, m[ch_y], color="tab:blue", lw=2, label="$E_y$")
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _setup_axes(ax, title=f"Activity — per channel, standard ({std_type})",
                xlabel="time in sequence (s)", ylabel="rate")

    # ---- row 3: weight evolution + final W ----
    ax = fig.add_subplot(gs[2, 0])
    if Wt.size:
        ax.plot(W_t, Wt[:, ch_x, ch_x], color="tab:red",   lw=2,
                label=r"$W_{x\leftarrow x}$  (x self)")
        ax.plot(W_t, Wt[:, ch_y, ch_x], color="tab:purple", lw=2,
                label=r"$W_{y\leftarrow x}$  (x$\rightarrow$y)")
        ax.plot(W_t, Wt[:, ch_x, ch_y], color="tab:green", lw=1.4, alpha=0.8,
                label=r"$W_{x\leftarrow y}$  (y$\rightarrow$x)")
        ax.plot(W_t, Wt[:, ch_y, ch_y], color="tab:blue",  lw=1.4, alpha=0.8,
                label=r"$W_{y\leftarrow y}$  (y self)")
    ax.legend(fontsize=8, frameon=False)
    _setup_axes(ax, title="Recurrent E->E weight evolution",
                xlabel="time (s)", ylabel="W")

    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(W_f, cmap="viridis", aspect="equal", origin="upper",
                   vmin=0, vmax=max(W_f.max(), 1e-3))
    ax.set_xticks([ch_x, ch_y]); ax.set_xticklabels(["x", "y"])
    ax.set_yticks([ch_x, ch_y]); ax.set_yticklabels(["x", "y"])
    ax.set_xlabel("pre", fontsize=10); ax.set_ylabel("post", fontsize=10)
    ax.set_title("Final W", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Plotting — the three contrasts
# =====================================================================
def plot_comparisons(comparisons: List[dict], cfg: A1Config, fname: str):
    """Four-row figure: the three local-global contrasts + a summary bar.

    Each ``comparisons`` entry is a dict with:
        name  : short label
        expr  : the contrast expression (string)
        A, B  : (N, n_seq) trial-averaged condition means; difference = A - B
        lblA, lblB : legend labels
    """
    dt = cfg.dt
    n_seq = comparisons[0]["A"].shape[1]
    ts = np.arange(n_seq) * dt
    xmax = _active_span(dt) + 0.20
    t5_win = _tone5_window(dt)

    fig = plt.figure(figsize=(13, 13), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    inh_tag = ("uniform inhibition"
               if (cfg.w_EI_self == cfg.w_EI_lat
                   and cfg.w_IE_self == cfg.w_IE_lat)
               else "selective inhibition")
    fig.suptitle(
        f"Local-Global paradigm — model0 [{inh_tag}]\n"
        "purple band = 5th (critical) tone; effect measured over its "
        "150 ms response window",
        fontsize=12, fontweight="bold")

    effect_sizes = []
    for row, cmp in enumerate(comparisons):
        A = cmp["A"].mean(0)            # all-channel mean
        B = cmp["B"].mean(0)
        diff = A - B
        eff = diff[t5_win].mean()
        effect_sizes.append(eff)

        # ---- left: overlaid conditions ----
        ax = fig.add_subplot(gs[row, 0])
        _shade_tones(ax, dt)
        ax.plot(ts, A, color="tab:red",  lw=2, label=cmp["lblA"])
        ax.plot(ts, B, color="tab:blue", lw=2, label=cmp["lblB"])
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        _setup_axes(ax, title=cmp["name"],
                    xlabel="time in sequence (s)",
                    ylabel=r"$\langle E\rangle$")

        # ---- right: difference ----
        ax = fig.add_subplot(gs[row, 1])
        _shade_tones(ax, dt)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.plot(ts, diff, color="tab:purple", lw=2)
        ax.fill_between(ts, 0, diff, where=(diff > 0), color="tab:red",
                        alpha=0.25)
        ax.fill_between(ts, 0, diff, where=(diff < 0), color="tab:blue",
                        alpha=0.25)
        ax.set_xlim(0, xmax)
        ax.annotate(f"tone-5 effect = {eff:+.3f}",
                    xy=(0.97, 0.92), xycoords="axes fraction",
                    ha="right", fontsize=9.5, fontweight="bold",
                    color="tab:purple")
        _setup_axes(ax, title=f"{cmp['expr']}",
                    xlabel="time in sequence (s)",
                    ylabel=r"$\Delta\langle E\rangle$")

    # ---- row 4: summary bar chart ----
    ax = fig.add_subplot(gs[3, :])
    names = [c["name"] for c in comparisons]
    colors = ["tab:green", "tab:orange", "tab:purple"]
    bars = ax.bar(names, effect_sizes, color=colors[:len(names)], width=0.55)
    ax.axhline(0, color="0.3", lw=0.8)
    for b, v in zip(bars, effect_sizes):
        ax.annotate(f"{v:+.3f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -13),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold")
    _setup_axes(ax, title="Effect size — mean Δ⟨E⟩ over the 5th-tone "
                          "response window",
                ylabel=r"$\Delta\langle E\rangle$")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    ch_x, ch_y = 0, 1

    # Loop over both inhibition-structure presets so the user can see
    # the contribution of tone-selectivity to the local-global effects.
    # Filenames carry the preset tag; the two runs do not overwrite.
    for inh_name, inh_factory in INH_PRESETS.items():
        cfg = inh_factory()

        print(f"\n========================================================")
        print(f"[ Local-global -- inhibition preset '{inh_name}' ]")
        print(f"  w_EI = (self {cfg.w_EI_self}, lat {cfg.w_EI_lat}); "
              f"w_IE = (self {cfg.w_IE_self}, lat {cfg.w_IE_lat})")

        print("[ Run 1 ]  standard = xxxxy (80%) / deviant = xxxxx (20%)")
        res1 = run_experiment("xxxxy", seed=1, cfg=cfg, ch_x=ch_x, ch_y=ch_y)
        print("[ Run 2 ]  standard = xxxxx (80%) / deviant = xxxxy (20%)")
        res2 = run_experiment("xxxxx", seed=2, cfg=cfg, ch_x=ch_x, ch_y=ch_y)

        # ---- the four condition means (2nd half of test phase) ----
        STD_xxxxy, n_sy = condition_mean(res1, "xxxxy")   # standard in Run 1
        DEV_xxxxx, n_dx = condition_mean(res1, "xxxxx")   # deviant  in Run 1
        STD_xxxxx, n_sx = condition_mean(res2, "xxxxx")   # standard in Run 2
        DEV_xxxxy, n_dy = condition_mean(res2, "xxxxy")   # deviant  in Run 2

        print("[ Plotting ]")
        plot_run(res1,
                 f"Run 1 — standard xxxxy / deviant xxxxx — local-global "
                 f"[{inh_name} inhibition]",
                 f"lg_m0_run1_{inh_name}.png")
        plot_run(res2,
                 f"Run 2 — standard xxxxx / deviant xxxxy — local-global "
                 f"[{inh_name} inhibition]",
                 f"lg_m0_run2_{inh_name}.png")

        comparisons = [
            dict(name="Local effect",
                 expr="STD xxxxy  −  STD xxxxx",
                 A=STD_xxxxy, lblA=f"STD xxxxy (n={n_sy})",
                 B=STD_xxxxx, lblB=f"STD xxxxx (n={n_sx})"),
            dict(name="Global effect (xxxxx)",
                 expr="DEV xxxxx  −  STD xxxxx",
                 A=DEV_xxxxx, lblA=f"DEV xxxxx (n={n_dx})",
                 B=STD_xxxxx, lblB=f"STD xxxxx (n={n_sx})"),
            dict(name="Global effect (xxxxy)",
                 expr="DEV xxxxy  −  STD xxxxy",
                 A=DEV_xxxxy, lblA=f"DEV xxxxy (n={n_dy})",
                 B=STD_xxxxy, lblB=f"STD xxxxy (n={n_sy})"),
        ]
        plot_comparisons(comparisons, cfg,
                         f"lg_m0_comparisons_{inh_name}.png")

        # ---- text summary ----
        t5 = _tone5_window(cfg.dt)
        print(f"\nTone-5 effect sizes [{inh_name}] "
              f"(mean Δ⟨E⟩ over the 5th-tone window):")
        for cmp in comparisons:
            d = (cmp["A"].mean(0) - cmp["B"].mean(0))[t5].mean()
            print(f"  {cmp['name']:<24s} ({cmp['expr']}):  {d:+.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
