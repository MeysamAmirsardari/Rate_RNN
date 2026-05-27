"""
tasks.oddball.oddball
=====================

Streaming oddball / SSA paradigm for model0.

Four conditions (each an independent session on a fresh A1Config-sized
network, same seed, different stimulus sequence):

  1. oddball_f1dev  -- f1 deviant (10%),  f2 standard (90%)
  2. oddball_f2dev  -- f2 deviant (10%),  f1 standard (90%)
  3. deviant_alone  -- f1 at 10%, silence otherwise (same trial positions
                       as in oddball_f1dev; controls for stimulus rate)
  4. diverse_broad  -- f1 and f2 each at 10%; remaining 80% spread
                       equally across the other 10 channels (many-
                       standards control)

Outputs:
    oddball_m0_raster_{inh}.png     stimulus rasters for the 4 conditions
    oddball_m0_responses_{inh}.png  per-condition mean E(t) on f1, f2
    oddball_m0_ssa_{inh}.png        SSA indices and many-standards control

Tones map 1:1 to channels: ch1 -> 0, ch2 -> 1, ... .  A1Config.N must
therefore be >= cfg.n_channels (= 12 by default).

This file implements only the paradigm and analysis.  model0 itself is
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, INH_PRESETS, simulate

if __package__:
    from .config import OddballConfig, ALL_CONDITIONS, get_preset
else:
    from tasks.oddball.config import OddballConfig, ALL_CONDITIONS, get_preset


# =====================================================================
#  Sequence generation
# =====================================================================
def generate_oddball_sequence(
    n_trials: int, dev_prob: float,
    dev_channel: str, std_channel: str,
    rng: np.random.Generator,
) -> List[str]:
    """Bernoulli oddball: deviant at probability ``dev_prob``."""
    seq = []
    for _ in range(n_trials):
        seq.append(dev_channel if rng.random() < dev_prob else std_channel)
    return seq


def generate_deviant_alone_sequence(
    reference_seq: List[str], dev_channel: str,
) -> List[str]:
    """Replace standards with silence; keep deviant positions intact.

    Trial timing is preserved -- silence trials still occupy a full SOA.
    """
    return [dev_channel if c == dev_channel else "silence"
            for c in reference_seq]


def generate_diverse_broad_sequence(
    n_trials: int, all_channels: List[str], dev_prob: float,
    f1_name: str, f2_name: str, rng: np.random.Generator,
) -> List[str]:
    """Many-standards: f1 and f2 each at dev_prob, rest split evenly."""
    other = [c for c in all_channels if c not in (f1_name, f2_name)]
    p_other = (1.0 - 2 * dev_prob) / len(other)
    p = np.array([dev_prob if c in (f1_name, f2_name) else p_other
                  for c in all_channels])
    p /= p.sum()
    idx = rng.choice(len(all_channels), size=n_trials, p=p)
    return [all_channels[i] for i in idx]


def build_condition_sequence(
    condition: str, cfg: OddballConfig, rng: np.random.Generator,
) -> List[str]:
    """Return the per-trial channel-name list for one condition.

    For ``deviant_alone`` the deviant-trial positions are matched to the
    f1-dev oddball sequence built from the same seed.
    """
    # Always build the f1dev reference (deviant-alone needs it).
    rng_ref = np.random.default_rng(rng.bit_generator)
    rng_ref_state = rng_ref.bit_generator.state
    rng.bit_generator.state = rng_ref_state
    oddball_f1dev_seq = generate_oddball_sequence(
        cfg.n_trials, cfg.dev_prob, cfg.f1_name, cfg.f2_name, rng_ref)

    if condition == "oddball_f1dev":
        return oddball_f1dev_seq
    if condition == "oddball_f2dev":
        return generate_oddball_sequence(
            cfg.n_trials, cfg.dev_prob, cfg.f2_name, cfg.f1_name,
            np.random.default_rng(cfg.seed + 501))
    if condition == "deviant_alone":
        return generate_deviant_alone_sequence(oddball_f1dev_seq, cfg.f1_name)
    if condition == "diverse_broad":
        return generate_diverse_broad_sequence(
            cfg.n_trials, list(cfg.channel_names), cfg.dev_prob,
            cfg.f1_name, cfg.f2_name,
            np.random.default_rng(cfg.seed + 502))
    raise ValueError(f"Unknown condition {condition!r}")


# =====================================================================
#  Stimulus
# =====================================================================
def build_oddball_stim(
    channel_seq: List[str], cfg: OddballConfig, a1_cfg: A1Config,
) -> Dict:
    """Build the full-session (N, T_total) stimulus.

    Returns dict with: stim, trial_starts, trial_channel, trial_is_deviant
    (the last filled only if a dev_label is provided by the caller).
    """
    if abs(a1_cfg.dt - 1e-3) > 1e-9:
        raise ValueError(
            f"OddballConfig assumes a1_cfg.dt = 1 ms; got {a1_cfg.dt}")
    if a1_cfg.N < cfg.n_channels:
        raise ValueError(
            f"a1_cfg.N = {a1_cfg.N} < cfg.n_channels = {cfg.n_channels}")

    soa = cfg.soa
    n = len(channel_seq)
    T = n * soa
    stim = np.zeros((a1_cfg.N, T))
    starts = np.arange(n) * soa
    for ti, ch in enumerate(channel_seq):
        if ch == "silence":
            continue
        ci = cfg.channel_index(ch)
        t0 = starts[ti]
        stim[ci, t0:t0 + cfg.tone_dur] = cfg.tone_amp
    return dict(stim=stim, trial_starts=starts,
                trial_channel=np.array(channel_seq, dtype="U8"))


# =====================================================================
#  Experiment
# =====================================================================
def run_condition(
    condition: str, cfg: Optional[OddballConfig] = None,
    a1_cfg: Optional[A1Config] = None,
) -> dict:
    """Run one streaming-oddball condition."""
    if cfg is None:
        cfg = OddballConfig()
    if a1_cfg is None:
        a1_cfg = A1Config(N=cfg.n_channels)

    rng = np.random.default_rng(cfg.seed)
    channel_seq = build_condition_sequence(condition, cfg, rng)
    stim_pack = build_oddball_stim(channel_seq, cfg, a1_cfg)

    # Snapshot W every ~1 s for the W-evolution panel.
    snap_every = max(1, int(round(1.0 / a1_cfg.dt)))
    out = simulate(stim_pack["stim"], cfg=a1_cfg,
                   record_W_every=snap_every, seed=cfg.seed)

    # Labels: which trials count as "deviant" for this condition?
    if condition == "oddball_f1dev":
        dev_name = cfg.f1_name
    elif condition == "oddball_f2dev":
        dev_name = cfg.f2_name
    elif condition == "deviant_alone":
        dev_name = cfg.f1_name
    else:
        dev_name = None

    is_dev = (stim_pack["trial_channel"] == dev_name) if dev_name else \
             np.zeros(len(channel_seq), dtype=bool)

    out.update(stim_pack)
    out["condition"]        = condition
    out["dev_name"]         = dev_name
    out["trial_is_deviant"] = is_dev
    out["oddball_cfg"]      = cfg
    return out


def run_all_conditions(
    cfg: Optional[OddballConfig] = None,
    a1_cfg: Optional[A1Config] = None,
) -> Dict[str, dict]:
    if cfg is None:
        cfg = OddballConfig()
    if a1_cfg is None:
        a1_cfg = A1Config(N=cfg.n_channels)

    results = {}
    for cond in cfg.conditions:
        print(f"  [ condition {cond} ]")
        results[cond] = run_condition(cond, cfg, a1_cfg)
    return results


# =====================================================================
#  Epoch extraction
# =====================================================================
def epoch_array(arr: np.ndarray, starts: np.ndarray,
                pre: int, total: int) -> np.ndarray:
    """Cut (N, T) into (n_trials, N, total) epochs of length ``total``
    aligned to ``starts - pre``."""
    n_trials = len(starts)
    N, T = arr.shape
    out = np.zeros((n_trials, N, total))
    for k, s in enumerate(starts):
        lo = max(0, s - pre)
        hi = min(T, s - pre + total)
        out[k, :, lo - (s - pre): hi - (s - pre)] = arr[:, lo:hi]
    return out


def epoch_E(res: dict) -> np.ndarray:
    cfg: OddballConfig = res["oddball_cfg"]
    return epoch_array(res["E"], res["trial_starts"],
                       cfg.pre_stim_ms, cfg.epoch_steps)


# =====================================================================
#  SSA analysis
# =====================================================================
def compute_responses(res: dict, channel_idx: int,
                      use_second_half: bool = True) -> Dict[str, np.ndarray]:
    """Mean E response on a given channel, split by trial role."""
    cfg: OddballConfig = res["oddball_cfg"]
    epochs = epoch_E(res)                          # (n_trials, N, T)
    tch = res["trial_channel"]
    n = len(tch)
    half = n // 2 if use_second_half else 0
    win = slice(cfg.tone_onset_in_epoch,
                cfg.tone_onset_in_epoch + cfg.tone_dur)
    target_name = f"ch{channel_idx + 1}"
    sel_present = (tch == target_name)
    # In the second half only (post-adaptation):
    sel_present[:half] = False
    traces = epochs[sel_present][:, channel_idx, :]
    peaks  = traces[:, win].max(axis=1) if len(traces) else np.array([])
    means  = traces[:, win].mean(axis=1) if len(traces) else np.array([])
    return dict(traces=traces, peaks=peaks, means=means,
                n=int(sel_present.sum()))


def ssa_index(R_dev: float, R_std: float) -> float:
    """Standard SSA index: (R_dev - R_std) / (R_dev + R_std).  In [-1, 1]."""
    denom = R_dev + R_std
    return (R_dev - R_std) / denom if denom > 0 else 0.0


# =====================================================================
#  Plotting
# =====================================================================
def _setup_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


_COND_COLOUR = {
    "oddball_f1dev":  "tab:red",
    "oddball_f2dev":  "tab:blue",
    "deviant_alone":  "tab:green",
    "diverse_broad":  "goldenrod",
}


def _inh_tag(a1_cfg: A1Config) -> str:
    if (a1_cfg.w_EI_self == a1_cfg.w_EI_lat
            and a1_cfg.w_IE_self == a1_cfg.w_IE_lat):
        return "uniform inhibition"
    return "selective inhibition"


def plot_rasters(results: Dict[str, dict], fname: str,
                 n_trials_show: int = 20):
    """4-panel stimulus raster for each condition."""
    cfg: OddballConfig = next(iter(results.values()))["oddball_cfg"]
    a1_cfg = next(iter(results.values()))["cfg"]
    dt = a1_cfg.dt
    show_T = min(n_trials_show * cfg.soa, results[cfg.conditions[0]]["stim"].shape[1])

    fig, axes = plt.subplots(len(cfg.conditions), 1,
                             figsize=(11, 2.0 * len(cfg.conditions) + 0.5),
                             sharex=True, constrained_layout=True)
    if len(cfg.conditions) == 1:
        axes = [axes]
    fig.suptitle(f"Streaming oddball: stimulus rasters "
                 f"[{_inh_tag(a1_cfg)}, first {n_trials_show} trials]",
                 fontsize=12, fontweight="bold")

    for ax, cond in zip(axes, cfg.conditions):
        res = results[cond]
        stim = res["stim"][:, :show_T]
        ax.imshow(stim, aspect="auto", origin="lower",
                  cmap="Greys", interpolation="nearest",
                  extent=[0, show_T * dt, -0.5, cfg.n_channels - 0.5])
        ax.axhline(cfg.f1_channel, color="tab:red",  lw=0.6, ls=":", alpha=0.7)
        ax.axhline(cfg.f2_channel, color="tab:blue", lw=0.6, ls=":", alpha=0.7)
        ax.set_yticks([cfg.f1_channel, cfg.f2_channel])
        ax.set_yticklabels([f"{cfg.f1_name} (f1)", f"{cfg.f2_name} (f2)"])
        _setup_axes(ax, title=cond, ylabel="channel")
    axes[-1].set_xlabel("time (s)")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_responses(results: Dict[str, dict], fname: str):
    """Per-condition mean E response on f1 and f2 (second half of session)."""
    cfg: OddballConfig = next(iter(results.values()))["oddball_cfg"]
    a1_cfg = next(iter(results.values()))["cfg"]
    dt = a1_cfg.dt
    ts = (np.arange(cfg.epoch_steps) - cfg.pre_stim_ms) * dt
    tone_s = cfg.tone_dur * dt

    fig, axes = plt.subplots(2, len(cfg.conditions),
                             figsize=(3.4 * len(cfg.conditions), 6.2),
                             sharex=True, sharey="row",
                             constrained_layout=True)
    fig.suptitle(f"Streaming oddball: per-condition responses "
                 f"[{_inh_tag(a1_cfg)}]",
                 fontsize=12, fontweight="bold")

    for col, cond in enumerate(cfg.conditions):
        res = results[cond]
        col_colour = _COND_COLOUR.get(cond, "0.4")
        for row, (chan_idx, chan_label) in enumerate(
                [(cfg.f1_channel, "f1"), (cfg.f2_channel, "f2")]):
            ax = axes[row, col]
            ax.axvspan(0, tone_s, color=col_colour, alpha=0.08)
            r = compute_responses(res, chan_idx, use_second_half=True)
            if len(r["traces"]):
                m  = r["traces"].mean(0)
                se = r["traces"].std(0) / max(1.0, np.sqrt(len(r["traces"])))
                ax.fill_between(ts, m - se, m + se, color=col_colour, alpha=0.18, lw=0)
                ax.plot(ts, m, color=col_colour, lw=1.8,
                        label=f"n={r['n']}")
                ax.legend(fontsize=7, frameon=False, loc="upper right")
            else:
                ax.text(0.5, 0.5, "no trials", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.5")
            title = f"{cond}\n{chan_label} = {f'ch{chan_idx+1}'}" if row == 0 \
                    else f"{chan_label} = ch{chan_idx+1}"
            _setup_axes(ax,
                        title=title,
                        ylabel=f"E[{chan_label}]" if col == 0 else None,
                        xlabel="time from tone onset (s)" if row == 1 else None)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_ssa_summary(results: Dict[str, dict], fname: str):
    """SSA indices and many-standards control on f1 and f2."""
    cfg: OddballConfig = next(iter(results.values()))["oddball_cfg"]
    a1_cfg = next(iter(results.values()))["cfg"]

    def mean_peak(cond, ch_idx):
        if cond not in results:
            return np.nan, 0
        r = compute_responses(results[cond], ch_idx, use_second_half=True)
        return (float(r["peaks"].mean()) if len(r["peaks"]) else np.nan,
                int(r["n"]))

    # Responses
    f1_DEV,  n_f1_DEV  = mean_peak("oddball_f1dev",  cfg.f1_channel)
    f1_STD,  n_f1_STD  = mean_peak("oddball_f2dev",  cfg.f1_channel)
    f1_ALONE,n_f1_AL   = mean_peak("deviant_alone",  cfg.f1_channel)
    f1_DB,   n_f1_DB   = mean_peak("diverse_broad",  cfg.f1_channel)

    f2_DEV,  n_f2_DEV  = mean_peak("oddball_f2dev",  cfg.f2_channel)
    f2_STD,  n_f2_STD  = mean_peak("oddball_f1dev",  cfg.f2_channel)
    f2_DB,   n_f2_DB   = mean_peak("diverse_broad",  cfg.f2_channel)

    SI_f1 = ssa_index(f1_DEV, f1_STD) if not (np.isnan(f1_DEV) or np.isnan(f1_STD)) else np.nan
    SI_f2 = ssa_index(f2_DEV, f2_STD) if not (np.isnan(f2_DEV) or np.isnan(f2_STD)) else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle(f"Streaming oddball: SSA summary "
                 f"[{_inh_tag(a1_cfg)}]",
                 fontsize=12, fontweight="bold")

    # Panel 1: f1 peak response across conditions
    ax = axes[0]
    vals = [f1_DEV, f1_STD, f1_ALONE, f1_DB]
    labs = ["DEV\n(odd f1)", "STD\n(odd f2)", "ALONE\n(dev-alone)", "DB\n(diverse)"]
    colors = [_COND_COLOUR["oddball_f1dev"], _COND_COLOUR["oddball_f2dev"],
              _COND_COLOUR["deviant_alone"], _COND_COLOUR["diverse_broad"]]
    bars = ax.bar(labs, vals, color=colors, width=0.7)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
    _setup_axes(ax, title=f"f1 = {cfg.f1_name}  peak E response",
                ylabel="peak E")

    # Panel 2: f2 peak response across conditions
    ax = axes[1]
    vals = [f2_DEV, f2_STD, f2_DB]
    labs = ["DEV\n(odd f2)", "STD\n(odd f1)", "DB\n(diverse)"]
    colors = [_COND_COLOUR["oddball_f2dev"], _COND_COLOUR["oddball_f1dev"],
              _COND_COLOUR["diverse_broad"]]
    bars = ax.bar(labs, vals, color=colors, width=0.7)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold")
    _setup_axes(ax, title=f"f2 = {cfg.f2_name}  peak E response",
                ylabel="peak E")

    # Panel 3: SSA indices
    ax = axes[2]
    ax.bar(["SI(f1)", "SI(f2)"], [SI_f1, SI_f2],
           color=["tab:red", "tab:blue"], width=0.5)
    ax.axhline(0, color="0.3", lw=0.8)
    for x, v in zip([0, 1], [SI_f1, SI_f2]):
        if not np.isnan(v):
            ax.annotate(f"{v:+.3f}", xy=(x, v),
                        xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")
    _setup_axes(ax,
                title="SSA index  (R_dev − R_std)/(R_dev + R_std)",
                ylabel="SSA index")
    ax.set_ylim(-1, 1)

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    cfg = get_preset("default")

    print(f"[ Streaming oddball -- preset '{cfg.name}' ]")
    print(f"  {cfg.n_channels} channels, f1={cfg.f1_name}, f2={cfg.f2_name} "
          f"(delta_f={cfg.delta_f})")
    print(f"  {cfg.n_trials} trials/condition, SOA={cfg.soa} ms, "
          f"dev_prob={cfg.dev_prob}")
    print(f"  conditions: {cfg.conditions}")

    for inh_name, inh_factory in INH_PRESETS.items():
        a1_cfg = inh_factory(N=cfg.n_channels)
        print(f"\n========================================================")
        print(f"[ Inhibition preset '{inh_name}' ]")
        print(f"  w_EI = (self {a1_cfg.w_EI_self}, lat {a1_cfg.w_EI_lat}); "
              f"w_IE = (self {a1_cfg.w_IE_self}, lat {a1_cfg.w_IE_lat})")

        results = run_all_conditions(cfg=cfg, a1_cfg=a1_cfg)

        print("[ Plotting ]")
        plot_rasters(results,      f"oddball_m0_raster_{inh_name}.png")
        plot_responses(results,    f"oddball_m0_responses_{inh_name}.png")
        plot_ssa_summary(results,  f"oddball_m0_ssa_{inh_name}.png")

        # Summary printout
        from collections import Counter
        for cond, res in results.items():
            counts = Counter(res["trial_channel"].tolist())
            tot = sum(counts.values())
            if cond == "diverse_broad":
                # No "deviant" label here -- show f1 and f2 counts.
                n_f1 = counts.get(cfg.f1_name, 0)
                n_f2 = counts.get(cfg.f2_name, 0)
                print(f"  {cond:<16s}: f1={n_f1} f2={n_f2} "
                      f"(rest={tot - n_f1 - n_f2}) / {tot} trials")
            elif cond == "deviant_alone":
                n_dev = counts.get(res["dev_name"], 0)
                n_sil = counts.get("silence", 0)
                print(f"  {cond:<16s}: {n_dev:>3d} dev (f1) / "
                      f"{n_sil:>3d} silence / {tot} trials")
            else:
                ne = int(res["trial_is_deviant"].sum())
                ns = tot - ne
                print(f"  {cond:<16s}: {ne:>3d} dev / {ns:>3d} std / "
                      f"{tot} trials")
    print("\nDone.")


if __name__ == "__main__":
    main()
