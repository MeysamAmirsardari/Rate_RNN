"""
erp_sequences.sweep
===================

Parameter sweeps for the model0 sequence-deviance ERPs.  For each value of a
swept parameter it runs every paradigm, extracts the dominant signed deviance
(``compute_deviance`` -- the same metric the ERP figures report), and plots a
**summary curve**: peak DEV-STD deviance (E and I) versus the parameter, one
line per paradigm (solid = feature deviants, dashed = order deviants).  Optional
multi-seed runs add error bars; ``--grid`` instead maps the tau_I x intra_gap
plane as a heatmap.

The sweeps target the model's load-bearing mechanism and dissociate its two
deviance sources (predictive selective-inhibition vs. stimulus-specific
adaptation):

    tau_i        slow inhibition (PV ~10 ms -> SST ~80 ms): the predicted-tone
                 interneuron must outlive its tone.  Order deviance should grow
                 with tau_I; SSA-driven feature deviance stays ~flat.
    intra_gap    the prediction must survive the inter-tone gap (ratio gap/tau_I).
    tone_dur     integration time available to build the predictive inhibition.
    selectivity  lateral/self ratio of I->E, selective -> uniform at fixed
                 row-sum: the MMN degrades as inhibition loses tone-tuning.
    w_ie         overall uniform inhibition gain (control; ~flat, no selectivity).
    u_std        SSA strength: turning it down isolates the pure predictive MMN.
    p_dev        oddball probability curve (canonical MMN control).
    n_trials     learning curve: the order MMN is experience-dependent.

Run
---
    python -m erp_sequences.sweep --param tau_i
    python -m erp_sequences.sweep --param selectivity --seeds 5
    python -m erp_sequences.sweep --grid                       # 2D tau_I x intra_gap
    python -m erp_sequences.sweep --param all                  # every 1D sweep (slow)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from erp_sequences.config import DEFAULT_PARADIGMS, ERPConfig
from erp_sequences.make_erps import (_clean, _letter, compute_deviance, plot_erp,
                                     run_paradigm)

_ORDER = {"abc_acb", "abc_cba"}          # order deviants (vs feature deviants)


def _plabel(key: str) -> str:
    std, dev = DEFAULT_PARADIGMS[key]
    return "/".join("".join(_letter(c) for c in seq) for seq in (std, dev))


def _selectivity_weights(r, N=3, w_self_sel=3.0, w_lat_sel=0.20):
    """I->E self/lateral for a given lateral:self ratio r, holding the selective
    row-sum (total inhibition per cell) fixed -- so only the *tuning* changes."""
    R = w_self_sel + (N - 1) * w_lat_sel
    S = R / (1.0 + (N - 1) * r)
    return dict(w_IE_self=S, w_IE_lat=r * S)


# ---------------------------------------------------------------------
#  1D sweep registry:  base = regime kwargs, vary(v) = per-value kwargs
# ---------------------------------------------------------------------
SWEEPS = {
    "tau_i": dict(
        values=[0.01, 0.02, 0.04, 0.08, 0.16], label=r"$\tau_I$ (s)", logx=True,
        base=dict(inhibition="selective", p_dev=0.10),
        vary=lambda v: dict(extra_model={"tau_I": v})),
    "intra_gap": dict(
        values=[0.0, 0.015, 0.03, 0.06, 0.12], label="intra-tone gap (s)",
        base=dict(inhibition="selective", p_dev=0.10),
        vary=lambda v: dict(intra_gap=v)),
    "tone_dur": dict(
        values=[0.02, 0.035, 0.05, 0.08, 0.12], label="tone duration (s)",
        base=dict(inhibition="selective", p_dev=0.10),
        vary=lambda v: dict(tone_dur=v)),
    "selectivity": dict(
        values=[0.0667, 0.15, 0.30, 0.50, 0.75, 1.0],
        label=r"I$\to$E lateral/self ratio  (selective $\to$ uniform)",
        base=dict(inhibition="selective", p_dev=0.10),
        vary=lambda r: _selectivity_weights(r)),
    "w_ie": dict(
        values=[0.1, 0.16, 0.3, 0.6, 1.0], label=r"uniform $w_{IE}$ (all pairs)",
        base=dict(inhibition="uniform", p_dev=0.10),
        vary=lambda v: dict(w_IE_self=v, w_IE_lat=v)),
    "u_std": dict(
        values=[0.0, 0.0005, 0.001, 0.002], label=r"$U_{std}$ (SSA strength)",
        base=dict(inhibition="selective", p_dev=0.10),
        vary=lambda v: dict(extra_model={"U_std": v})),
    "p_dev": dict(
        values=[0.5, 0.3, 0.15, 0.10, 0.05], label="deviant probability",
        base=dict(inhibition="selective"),
        vary=lambda v: dict(p_dev=v)),
    "n_trials": dict(
        values=[150, 300, 600, 1200], label="n_trials (learning)", logx=True,
        base=dict(inhibition="selective", p_dev=0.10),
        vary=lambda v: dict(n_trials=int(v))),
}


# ---------------------------------------------------------------------
def run_1d(name, *, outdir, seeds=(1,), paradigms=None, save_figs=False):
    spec = SWEEPS[name]
    vals = spec["values"]
    keys = paradigms or list(DEFAULT_PARADIGMS)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    dE = {k: np.full((len(vals), len(seeds)), np.nan) for k in keys}
    dI = {k: np.full((len(vals), len(seeds)), np.nan) for k in keys}

    for i, v in enumerate(vals):
        for j, sd in enumerate(seeds):
            base = dict(spec["base"]); base["seed"] = sd
            ecfg = ERPConfig(**base, **spec["vary"](v))
            a1 = ecfg.a1config()
            for k in keys:
                std, dev = DEFAULT_PARADIGMS[k]
                m = compute_deviance(run_paradigm(std, dev, a1=a1, ecfg=ecfg))
                dE[k][i, j], dI[k][i, j] = m["peak_dE"], m["peak_dI"]
                if save_figs:
                    sub = out / f"{name}_{v:g}"; sub.mkdir(exist_ok=True)
                    res = run_paradigm(std, dev, a1=a1, ecfg=ecfg)
                    plot_erp(res, k, str(sub / f"erp_{k}.png"))
        print(f"  {name}={v:<7g} " +
              "  ".join(f"{k}:dE={np.nanmean(dE[k][i]):+.3f}" for k in keys))
    _plot_summary(name, spec, vals, dE, dI, keys, out)
    return vals, dE, dI


def _plot_summary(name, spec, vals, dE, dI, keys, out):
    x = np.asarray(vals, float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    panels = ((axes[0], dE, r"peak $\Delta\langle E\rangle$  (DEV $-$ STD)",
               "Excitatory deviance"),
              (axes[1], dI, r"peak $\Delta\langle I\rangle$  (DEV $-$ STD)",
               "Inhibitory deviance"))
    for ax, D, ylab, ttl in panels:
        for k in keys:
            mean, std = np.nanmean(D[k], 1), np.nanstd(D[k], 1)
            ls = "--" if k in _ORDER else "-"
            (ln,) = ax.plot(x, mean, ls=ls, marker="o", lw=2, label=_plabel(k))
            if D[k].shape[1] > 1:
                ax.fill_between(x, mean - std, mean + std,
                                color=ln.get_color(), alpha=0.15, lw=0)
        ax.axhline(0, color="0.5", lw=0.7)
        if spec.get("logx"):
            ax.set_xscale("log")
        ax.set_xlabel(spec["label"], fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        _clean(ax)
    axes[0].legend(fontsize=8.5, frameon=False,
                   title="paradigm  (— feature, -- order)", title_fontsize=8.5)
    fig.suptitle(f"Sequence-deviance vs {spec['label']}   --   model0, "
                 f"{spec['base'].get('inhibition', '?')} inhibition",
                 fontsize=12.5, fontweight="bold")
    f = out / f"sweep_{name}.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  -> {f}")


# ---------------------------------------------------------------------
def run_grid(*, outdir, seeds=(1,), paradigms=None,
             tau_vals=(0.01, 0.02, 0.04, 0.08, 0.16),
             gap_vals=(0.0, 0.03, 0.06, 0.12)):
    """2D tau_I x intra_gap map of the peak excitatory deviance per paradigm."""
    keys = paradigms or list(DEFAULT_PARADIGMS)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    Z = {k: np.full((len(gap_vals), len(tau_vals)), np.nan) for k in keys}
    for gi, g in enumerate(gap_vals):
        for ti, tau in enumerate(tau_vals):
            acc = {k: [] for k in keys}
            for sd in seeds:
                ecfg = ERPConfig(inhibition="selective", p_dev=0.10, intra_gap=g,
                                 seed=sd, extra_model={"tau_I": tau})
                a1 = ecfg.a1config()
                for k in keys:
                    std, dev = DEFAULT_PARADIGMS[k]
                    acc[k].append(
                        compute_deviance(run_paradigm(std, dev, a1=a1, ecfg=ecfg))["peak_dE"])
            for k in keys:
                Z[k][gi, ti] = float(np.mean(acc[k]))
        print(f"  gap={g*1e3:.0f}ms done")
    _plot_grid(Z, tau_vals, gap_vals, keys, out)
    return Z


def _plot_grid(Z, tau_vals, gap_vals, keys, out):
    n = len(keys)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.8 * nrow),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    vmax = max(np.nanmax(np.abs(Z[k])) for k in keys) or 1.0
    for ax, k in zip(axes, keys):
        im = ax.imshow(Z[k], origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(tau_vals)))
        ax.set_xticklabels([f"{t*1e3:.0f}" for t in tau_vals])
        ax.set_yticks(range(len(gap_vals)))
        ax.set_yticklabels([f"{g*1e3:.0f}" for g in gap_vals])
        ax.set_xlabel(r"$\tau_I$ (ms)", fontsize=10)
        ax.set_ylabel("intra-tone gap (ms)", fontsize=10)
        ax.set_title(f"{_plabel(k)}   peak $\\Delta\\langle E\\rangle$",
                     fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Predictive-suppression phase map: "
                 r"deviance over $\tau_I \times$ gap (selective inhibition)",
                 fontsize=12.5, fontweight="bold")
    f = out / "sweep_grid_tauI_gap.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  -> {f}")


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Parameter sweeps for the model0 sequence-deviance ERPs.")
    ap.add_argument("--param", default="all", choices=list(SWEEPS) + ["all"])
    ap.add_argument("--grid", action="store_true",
                    help="run the 2D tau_I x intra_gap heatmap instead of a 1D sweep")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (1..N)")
    ap.add_argument("--paradigms", nargs="+", default=None,
                    choices=list(DEFAULT_PARADIGMS))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "sweeps"))
    ap.add_argument("--save-figs", action="store_true", dest="save_figs",
                    help="also save the full ERP figure for every swept value")
    args = ap.parse_args(argv)

    seeds = tuple(range(1, args.seeds + 1))
    out = Path(args.outdir)
    if args.grid:
        print(f"[ sweep ] 2D tau_I x intra_gap (selective), seeds={seeds} -> {out}")
        run_grid(outdir=out, seeds=seeds, paradigms=args.paradigms)
        return 0
    names = list(SWEEPS) if args.param == "all" else [args.param]
    for nm in names:
        print(f"[ sweep ] {nm}  seeds={seeds} -> {out}")
        run_1d(nm, outdir=out, seeds=seeds, paradigms=args.paradigms,
               save_figs=args.save_figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
