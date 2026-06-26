"""
erp_sequences.make_erps
=======================

Event-related-potential (ERP) responses of the **tone-selective inhibition**
A1 rate model (``model0``) for four multi-tone sequence-deviance paradigms:

    ABC / ACB     order deviant -- swap the 2nd and 3rd tones
    ABC / CBA     order deviant -- full temporal reversal
    AB  / AC      feature deviant -- change the 2nd tone identity (B -> C)
    AC  / BC      feature deviant -- change the 1st tone identity (A -> B)

Every tunable parameter lives in :class:`erp_sequences.config.ERPConfig` -- the
model0 knobs, the stimulus timing, the oddball protocol, the analysis window and
the paradigm set.  This module is just the engine: build the stimulus, run the
model, epoch the excitatory ``E`` and inhibitory ``I`` rates around sequence
onset, average the post-learning standard vs deviant trials, and plot the
population-mean ERP (the rate model's analogue of an evoked field) together with
the DEV-STD deviance (its MMN-like mismatch response).

Run
---
    python -m erp_sequences.make_erps                       # ERPConfig() defaults (85/15)
    python -m erp_sequences.make_erps --p-dev 0.10          # override from the CLI
    python -m erp_sequences.make_erps --tone-dur 0.06 --w-ie-self 4.0

Or drive it programmatically with an explicit config::

    from erp_sequences.config import ERPConfig
    from erp_sequences.make_erps import run
    run(ERPConfig(p_dev=0.10, inhibition="uniform"), outdir="erp_uniform")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Run-as-script support: `python erp_sequences/make_erps.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, INH_PRESETS, simulate
from erp_sequences.config import DEFAULT_PARADIGMS, ERPConfig

# ---------------------------------------------------------------------
#  Presentation (channel labels / colours; not "parameters to tune")
# ---------------------------------------------------------------------
_CH_CYCLE = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
             "tab:brown"]
_STD_COLOR = "tab:blue"
_DEV_COLOR = "tab:red"


def _letter(ch: int) -> str:
    return chr(ord("A") + ch)


def _ch_color(ch: int) -> str:
    return _CH_CYCLE[ch % len(_CH_CYCLE)]


# ---------------------------------------------------------------------
#  Stimulus
# ---------------------------------------------------------------------
def _tuning(N: int, centre: int) -> np.ndarray:
    v = np.zeros(N)
    v[centre] = 1.0
    return v


def build_stim(seqs: Sequence[Sequence[int]], a1: A1Config, ecfg: ERPConfig):
    """Concatenate token sequences into a (N, T) stimulus.

    Every sequence in ``seqs`` has the same number of tokens, so all epochs
    share a length and align for trial averaging.  Returns the stimulus, the
    per-sequence onset samples, the sequence length ``n_seq``, the within-
    sequence tone-onset offsets (samples) and the tone length (samples)."""
    dt = a1.dt
    n_tone = int(round(ecfg.tone_dur / dt))
    n_intra = int(round(ecfg.intra_gap / dt))
    n_inter = int(round(ecfg.inter_gap / dt))
    n_tok = len(seqs[0])
    n_seq = n_tok * n_tone + (n_tok - 1) * n_intra + n_inter
    T = n_seq * len(seqs)

    tunings = {ch: ecfg.tone_amp * _tuning(a1.N, ch) for ch in range(a1.N)}
    stim = np.zeros((a1.N, T))
    starts = np.zeros(len(seqs), dtype=int)
    tone_onsets = [j * (n_tone + n_intra) for j in range(n_tok)]
    for k, seq in enumerate(seqs):
        s = k * n_seq
        starts[k] = s
        for j, ch in enumerate(seq):
            o = s + tone_onsets[j]
            stim[:, o:o + n_tone] = tunings[ch][:, None]
    return stim, starts, n_seq, tone_onsets, n_tone


def make_stream(std: Sequence[int], dev: Sequence[int], n_total: int,
                p_dev: float, rng: np.random.Generator):
    """Random oddball order of `std`/`dev` sequences (exact deviant count)."""
    n_dev = int(round(n_total * p_dev))
    is_dev = np.array([False] * (n_total - n_dev) + [True] * n_dev)
    rng.shuffle(is_dev)
    seqs = [tuple(dev) if d else tuple(std) for d in is_dev]
    return seqs, is_dev


# ---------------------------------------------------------------------
#  Run one paradigm
# ---------------------------------------------------------------------
def run_paradigm(std: Sequence[int], dev: Sequence[int], *,
                 a1: A1Config, ecfg: ERPConfig) -> dict:
    rng = np.random.default_rng(ecfg.seed)
    seqs, is_dev = make_stream(std, dev, ecfg.n_trials, ecfg.p_dev, rng)
    stim, starts, n_seq, tone_onsets, n_tone = build_stim(seqs, a1, ecfg)
    snap_every = max(1, int(round(0.5 / a1.dt)))
    out = simulate(stim, cfg=a1, record_W_every=snap_every, seed=ecfg.seed)
    out.update(seqs=seqs, seq_starts=starts, n_seq=n_seq, is_dev=is_dev,
               tone_onsets=tone_onsets, n_tone=n_tone,
               std=tuple(std), dev=tuple(dev), ecfg=ecfg)
    return out


def evoked_per_trial(arr: np.ndarray, starts: np.ndarray, n_seq: int) -> np.ndarray:
    """Cut a (N, T) history into per-trial epochs -> (n_trials, N, n_seq)."""
    out = np.empty((len(starts), arr.shape[0], n_seq))
    for k, s in enumerate(starts):
        out[k] = arr[:, s:s + n_seq]
    return out


# ---------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------
def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)


def _msem(x: np.ndarray):
    """Mean and standard error across trials (axis 0)."""
    m = x.mean(0)
    s = x.std(0) / np.sqrt(max(len(x), 1))
    return m, s


def _shade_tones(ax, tone_onsets, n_tone, dt, dev_slot=None):
    for j, o in enumerate(tone_onsets):
        hl = (dev_slot is not None and j == dev_slot)
        ax.axvspan(o * dt, (o + n_tone) * dt,
                   color="0.55" if hl else "0.80",
                   alpha=0.28 if hl else 0.18, lw=0)
        ax.text((o + n_tone / 2) * dt, 0.995, f"T{j + 1}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="0.35")


def _band(ax, t, m, s, color, label, ls="-"):
    ax.plot(t, m, color=color, lw=2.0, ls=ls, label=label)
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.16, lw=0)


def _schematic(ax, seq, a1, ecfg, tone_onsets, n_tone, xmax, title):
    dt = a1.dt
    stim1, _, _, _, _ = build_stim([seq], a1, ecfg)
    ax.imshow(stim1[:, :int(xmax / dt)], aspect="auto", origin="lower",
              cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
              extent=[0, xmax, -0.5, a1.N - 0.5])
    for o, ch in zip(tone_onsets, seq):
        ax.text((o + n_tone / 2) * dt, ch, _letter(ch), ha="center", va="center",
                fontsize=12, fontweight="bold", color=_ch_color(ch))
    ax.set_yticks(range(a1.N))
    ax.set_yticklabels([_letter(c) for c in range(a1.N)])
    ax.set_xlabel("time in sequence (s)", fontsize=9)
    ax.set_ylabel("channel", fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    _clean(ax)


def compute_deviance(res: dict) -> dict:
    """Post-learning standard/deviant population-mean E & I, their DEV-STD
    deviance traces (+SEM), and the dominant *signed* peak of each over the
    deviating-tone window.  Shared by :func:`plot_erp` and the sweeps so the
    metric is defined in exactly one place."""
    a1: A1Config = res["cfg"]
    ecfg: ERPConfig = res["ecfg"]
    dt = a1.dt
    n_seq = res["n_seq"]
    tone_onsets = res["tone_onsets"]
    n_tone = res["n_tone"]
    std, dev = res["std"], res["dev"]
    n_tok = len(std)
    # first slot whose tone identity differs -> the deviation onset
    dev_slot = next((j for j in range(n_tok) if std[j] != dev[j]), n_tok - 1)

    # post-learning epochs (2nd half by default), population mean over channels
    half = int(round(ecfg.learn_frac * ecfg.n_trials))
    is_dev = res["is_dev"][half:]
    evE = evoked_per_trial(res["E"], res["seq_starts"], n_seq)[half:].mean(1)
    evI = evoked_per_trial(res["I"], res["seq_starts"], n_seq)[half:].mean(1)
    E_std, E_dev = _msem(evE[~is_dev]), _msem(evE[is_dev])
    I_std, I_dev = _msem(evI[~is_dev]), _msem(evI[is_dev])

    ts = np.arange(n_seq) * dt
    dE = E_dev[0] - E_std[0]; dEs = np.sqrt(E_dev[1] ** 2 + E_std[1] ** 2)
    dI = I_dev[0] - I_std[0]; dIs = np.sqrt(I_dev[1] ** 2 + I_std[1] ** 2)
    win = slice(tone_onsets[dev_slot],
                tone_onsets[-1] + n_tone + int(round(ecfg.post_tone_window / dt)))
    iE = int(np.argmax(np.abs(dE[win]))); iI = int(np.argmax(np.abs(dI[win])))
    return dict(
        ts=ts, dev_slot=dev_slot,
        n_std=int((~is_dev).sum()), n_dev=int(is_dev.sum()),
        E_std=E_std, E_dev=E_dev, I_std=I_std, I_dev=I_dev,
        dE=dE, dEs=dEs, dI=dI, dIs=dIs,
        peak_dE=float(dE[win][iE]), t_dE=float(ts[win][iE]),
        peak_dI=float(dI[win][iI]), t_dI=float(ts[win][iI]))


def plot_erp(res: dict, name: str, fname: str):
    a1: A1Config = res["cfg"]
    ecfg: ERPConfig = res["ecfg"]
    dt = a1.dt
    tone_onsets = res["tone_onsets"]
    n_tone = res["n_tone"]
    std, dev = res["std"], res["dev"]
    std_lab = "".join(_letter(c) for c in std)
    dev_lab = "".join(_letter(c) for c in dev)

    m = compute_deviance(res)
    E_std_m, E_std_s = m["E_std"]; E_dev_m, E_dev_s = m["E_dev"]
    I_std_m, I_std_s = m["I_std"]; I_dev_m, I_dev_s = m["I_dev"]
    dE, dEs, dI, dIs = m["dE"], m["dEs"], m["dI"], m["dIs"]
    ts = m["ts"]; dev_slot = m["dev_slot"]; n_std = m["n_std"]; n_dev = m["n_dev"]

    tones_end = tone_onsets[-1] + n_tone
    xmax = (tones_end + int(round(ecfg.plot_pad / dt))) * dt

    fig = plt.figure(figsize=(13, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(
        f"{std_lab} vs {dev_lab}  --  sequence-deviance ERP "
        f"(tone-selective-inhibition A1 model)\n"
        f"standard {std_lab} (p={1 - ecfg.p_dev:.0%})   "
        f"deviant {dev_lab} (p={ecfg.p_dev:.0%})   "
        f"post-learning trials: n_std={n_std}, n_dev={n_dev}   "
        f"(deviating tone: T{dev_slot + 1})",
        fontsize=12.5, fontweight="bold")

    # ---- row 0: stimulus schematics ----
    _schematic(fig.add_subplot(gs[0, 0]), std, a1, ecfg, tone_onsets, n_tone,
               xmax, f"Standard sequence: {std_lab}")
    _schematic(fig.add_subplot(gs[0, 1]), dev, a1, ecfg, tone_onsets, n_tone,
               xmax, f"Deviant sequence: {dev_lab}")

    def _onset_line(ax):
        ax.axvline(tone_onsets[dev_slot] * dt, color="0.25", lw=1.1, ls=":",
                   label=f"deviation onset (T{dev_slot + 1})")

    # ---- row 1: excitatory ERP + deviance ----
    ax = fig.add_subplot(gs[1, 0])
    _shade_tones(ax, tone_onsets, n_tone, dt, dev_slot)
    _band(ax, ts, E_std_m, E_std_s, _STD_COLOR, f"standard {std_lab}")
    _band(ax, ts, E_dev_m, E_dev_s, _DEV_COLOR, f"deviant {dev_lab}")
    _onset_line(ax)
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title("Excitatory ERP  --  population mean rate "
                 r"$\langle E_i\rangle_i$", fontsize=11, fontweight="bold")
    ax.set_xlabel("time in sequence (s)", fontsize=9)
    ax.set_ylabel(r"$\langle E\rangle$", fontsize=10)
    _clean(ax)

    ax = fig.add_subplot(gs[1, 1])
    _shade_tones(ax, tone_onsets, n_tone, dt, dev_slot)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.plot(ts, dE, color="tab:purple", lw=2.0)
    ax.fill_between(ts, dE - dEs, dE + dEs, color="tab:purple", alpha=0.15, lw=0)
    ax.fill_between(ts, 0, dE, where=(dE > 0), color="tab:red", alpha=0.22,
                    label="DEV > STD (mismatch)")
    ax.fill_between(ts, 0, dE, where=(dE < 0), color="tab:blue", alpha=0.22,
                    label="DEV < STD")
    _onset_line(ax)
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title(r"Excitatory deviance  $\Delta\langle E\rangle$ (DEV $-$ STD)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("time in sequence (s)", fontsize=9)
    ax.set_ylabel(r"$\Delta\langle E\rangle$", fontsize=10)
    _clean(ax)

    # ---- row 2: inhibitory ERP + deviance ----
    ax = fig.add_subplot(gs[2, 0])
    _shade_tones(ax, tone_onsets, n_tone, dt, dev_slot)
    _band(ax, ts, I_std_m, I_std_s, _STD_COLOR, f"standard {std_lab}")
    _band(ax, ts, I_dev_m, I_dev_s, _DEV_COLOR, f"deviant {dev_lab}")
    _onset_line(ax)
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title("Inhibitory ERP  --  population mean rate "
                 r"$\langle I_i\rangle_i$", fontsize=11, fontweight="bold")
    ax.set_xlabel("time in sequence (s)", fontsize=9)
    ax.set_ylabel(r"$\langle I\rangle$", fontsize=10)
    _clean(ax)

    ax = fig.add_subplot(gs[2, 1])
    _shade_tones(ax, tone_onsets, n_tone, dt, dev_slot)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.plot(ts, dI, color="tab:purple", lw=2.0)
    ax.fill_between(ts, dI - dIs, dI + dIs, color="tab:purple", alpha=0.15, lw=0)
    ax.fill_between(ts, 0, dI, where=(dI > 0), color="tab:red", alpha=0.22,
                    label="DEV > STD")
    ax.fill_between(ts, 0, dI, where=(dI < 0), color="tab:blue", alpha=0.22,
                    label="DEV < STD (less pre-built inhibition)")
    _onset_line(ax)
    ax.set_xlim(0, xmax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title(r"Inhibitory deviance  $\Delta\langle I\rangle$ (DEV $-$ STD)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("time in sequence (s)", fontsize=9)
    ax.set_ylabel(r"$\Delta\langle I\rangle$", fontsize=10)
    _clean(ax)

    fig.savefig(fname, dpi=ecfg.dpi)
    plt.close(fig)

    print(f"  {std_lab}/{dev_lab}: "
          f"peak E-deviance = {m['peak_dE']:+.3f} at {m['t_dE']*1e3:.0f} ms; "
          f"peak I-deviance = {m['peak_dI']:+.3f} at {m['t_dI']*1e3:.0f} ms; "
          f"saved {Path(fname).name}")


# ---------------------------------------------------------------------
def run(ecfg: ERPConfig, outdir, paradigm_keys=None) -> int:
    """Run every (selected) paradigm under `ecfg` and write one figure each."""
    a1 = ecfg.a1config()
    keys = list(ecfg.paradigms) if paradigm_keys is None else list(paradigm_keys)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ erp_sequences ] model0 inhibition={ecfg.inhibition}, N={a1.N}, "
          f"{ecfg.n_trials} trials/paradigm, p_dev={ecfg.p_dev:.0%} -> {out}")
    for key in keys:
        std, dev = ecfg.paradigms[key]
        res = run_paradigm(std, dev, a1=a1, ecfg=ecfg)
        plot_erp(res, key, str(out / f"erp_{key}.png"))
    print("Done.")
    return 0


def main(argv=None) -> int:
    d = ERPConfig()        # defaults
    ap = argparse.ArgumentParser(
        description="Sequence-deviance ERPs (E and I) for the model0 A1 model. "
                    "Defaults come from erp_sequences.config.ERPConfig.")
    ap.add_argument("--paradigms", nargs="+", default=None,
                    choices=list(DEFAULT_PARADIGMS),
                    help="subset to run (default: all)")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--trials", type=int, default=d.n_trials)
    ap.add_argument("--p-dev", type=float, default=d.p_dev, dest="p_dev")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--inhibition", default=d.inhibition, choices=list(INH_PRESETS))
    ap.add_argument("--n-channels", type=int, default=d.n_channels, dest="n_channels")
    # weight knobs default to None -> the preset/MMN-regime value (see ERPConfig)
    ap.add_argument("--w-ie-self", type=float, default=None, dest="w_IE_self")
    ap.add_argument("--w-ie-lat", type=float, default=None, dest="w_IE_lat")
    ap.add_argument("--w-ei-self", type=float, default=None, dest="w_EI_self")
    ap.add_argument("--w-ei-lat", type=float, default=None, dest="w_EI_lat")
    ap.add_argument("--w-norm", type=float, default=None, dest="W_norm")
    # convenience: set both I->E self & lateral at once ("gain for all", uniform)
    ap.add_argument("--w-ie", type=float, default=None, dest="w_IE",
                    help="set w_IE_self AND w_IE_lat to this (uniform I->E gain)")
    ap.add_argument("--tone-dur", type=float, default=d.tone_dur, dest="tone_dur")
    ap.add_argument("--intra-gap", type=float, default=d.intra_gap, dest="intra_gap")
    ap.add_argument("--inter-gap", type=float, default=d.inter_gap, dest="inter_gap")
    ap.add_argument("--dpi", type=int, default=d.dpi)
    args = ap.parse_args(argv)

    # --w-ie broadcasts to both I->E self & lateral; per-entry flags override it
    w_IE_self = args.w_IE_self if args.w_IE_self is not None else args.w_IE
    w_IE_lat = args.w_IE_lat if args.w_IE_lat is not None else args.w_IE
    ecfg = ERPConfig(
        inhibition=args.inhibition, n_channels=args.n_channels,
        w_IE_self=w_IE_self, w_IE_lat=w_IE_lat,
        w_EI_self=args.w_EI_self, w_EI_lat=args.w_EI_lat, W_norm=args.W_norm,
        tone_dur=args.tone_dur, intra_gap=args.intra_gap, inter_gap=args.inter_gap,
        p_dev=args.p_dev, n_trials=args.trials, seed=args.seed, dpi=args.dpi,
    )
    return run(ecfg, args.outdir, args.paradigms)


if __name__ == "__main__":
    raise SystemExit(main())
