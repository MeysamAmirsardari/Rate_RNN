"""
tasks.syllable.syllable
=======================

Roving Oddball paradigm for model0 driven by *syllable figures*.

This is the roving-oddball task (``tasks.roving``) with one change: each
syllable is a **figure** -- a spectral activation pattern spread across
``n_channels`` (default 35) tonotopic channels with possibly different
amplitudes per channel -- rather than a single channel.  Syllables share
some channels and differ in others (see ``config.SYLLABLE_FORMANTS``),
mimicking the partial spectral overlap of real speech syllables.

A session = sequence of blocks; each block = ``n_reps_per_block``
consecutive presentations of one 3-syllable word.  Block order is
constrained random (no two consecutive blocks of the same word).
Plasticity is on throughout.

``A1Config.N`` must equal ``cfg.n_channels`` (each channel has its own
E and I unit).

Outputs four figures (per inhibition preset, except the figure bank):
    syllable_m0_figures.png       the five syllable figures over channels
                                  (inhibition-independent; saved once).
    syllable_m0_overview_*.png    stimulus raster, per-syllable population
                                  E rate, recurrent weight evolution.
    syllable_m0_repsupp_*.png     per-word population response at rep 1 vs
                                  rep N (post-adaptation) + surprisal.
    syllable_m0_blockdyn_*.png    peak deviant-syllable population E rate
                                  across the 15 repetitions within a block.

This file implements only the task -- model0 itself is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Put the project root on sys.path so the package-relative import below
# works in both `python -m tasks.syllable.syllable` and
# `python tasks/syllable/syllable.py` modes.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, INH_PRESETS, simulate, shared_config

if __package__:
    from .config import SyllableConfig, get_preset
else:
    from tasks.syllable.config import SyllableConfig, get_preset


# =====================================================================
#  Block-order generation
# =====================================================================
def generate_block_order(cfg: SyllableConfig,
                         rng: np.random.Generator) -> List[str]:
    """Constrained random block order: no consecutive identical-word blocks."""
    words = list(cfg.words)
    counts = {w: cfg.n_blocks_per_word for w in words}
    order: List[str] = []
    while sum(counts.values()) > 0:
        available = [w for w in words if counts[w] > 0]
        if order:
            no_repeat = [w for w in available if w != order[-1]]
            if no_repeat:
                available = no_repeat
        w = available[int(rng.integers(len(available)))]
        order.append(w)
        counts[w] -= 1
    return order


# =====================================================================
#  Stimulus
# =====================================================================
def build_syllable_stim(
    block_order: List[str],
    cfg: SyllableConfig,
    a1_cfg: A1Config,
) -> Dict:
    """Build the full-session (N, T_total) stimulus.

    Each syllable slot is filled by broadcasting that syllable's figure
    (per-channel amplitudes) across the slot's time window.

    Returns dict with: stim, seq_starts, trial_period, seq_word,
    seq_block, seq_rep, block_order.
    """
    if abs(a1_cfg.dt - 1e-3) > 1e-9:
        raise ValueError(
            f"SyllableConfig assumes a1_cfg.dt = 1 ms; got {a1_cfg.dt}")
    if a1_cfg.N != cfg.n_channels:
        raise ValueError(
            f"a1_cfg.N = {a1_cfg.N} must equal cfg.n_channels = "
            f"{cfg.n_channels}")

    trial_period = cfg.trial_period
    n_total = cfg.n_total_seqs
    T_total = trial_period * n_total
    slot = cfg.syll_dur + cfg.syll_gap

    # Pre-compute the figure (per-channel amplitude vector) for each syllable.
    figures = {s: cfg.figure(s) for s in cfg.syllables}

    stim = np.zeros((a1_cfg.N, T_total))
    seq_starts = np.zeros(n_total, dtype=int)
    seq_word = np.empty(n_total, dtype="U3")
    seq_block = np.zeros(n_total, dtype=int)
    seq_rep = np.zeros(n_total, dtype=int)

    seq_idx = 0
    for bi, word in enumerate(block_order):
        for rep in range(cfg.n_reps_per_block):
            t0 = seq_idx * trial_period
            seq_starts[seq_idx] = t0
            seq_word[seq_idx] = word
            seq_block[seq_idx] = bi
            seq_rep[seq_idx] = rep
            for i, syll in enumerate(word):
                ts = t0 + i * slot
                stim[:, ts:ts + cfg.syll_dur] = figures[syll][:, None]
            seq_idx += 1

    return dict(
        stim=stim, seq_starts=seq_starts, trial_period=trial_period,
        seq_word=seq_word, seq_block=seq_block, seq_rep=seq_rep,
        block_order=list(block_order),
    )


# =====================================================================
#  Experiment
# =====================================================================
def run_experiment(
    cfg: Optional[SyllableConfig] = None,
    a1_cfg: Optional[A1Config] = None,
) -> dict:
    """Build the session stimulus and run model0 on it."""
    if cfg is None:
        cfg = SyllableConfig()
    if a1_cfg is None:
        a1_cfg = shared_config(N=cfg.n_channels)

    rng = np.random.default_rng(cfg.seed)
    block_order = generate_block_order(cfg, rng)
    stim_pack = build_syllable_stim(block_order, cfg, a1_cfg)

    snap_every = max(1, int(round(0.5 / a1_cfg.dt)))   # every 500 ms
    out = simulate(stim_pack["stim"], cfg=a1_cfg,
                   record_W_every=snap_every, seed=cfg.seed)

    out.update(stim_pack)
    out["syllable_cfg"] = cfg
    return out


# =====================================================================
#  Epoch extraction
# =====================================================================
def evoked_per_trial(arr: np.ndarray, seq_starts: np.ndarray,
                     epoch_pre: int, epoch_post: int) -> np.ndarray:
    """Cut an (N, T) or (T,) history into per-trial epochs.

    Each epoch spans ``[start - epoch_pre, start + epoch_post)``; samples
    that fall outside ``arr`` are zero-padded.
    """
    n_seq = len(seq_starts)
    epoch_len = epoch_pre + epoch_post
    if arr.ndim == 1:
        out = np.zeros((n_seq, epoch_len))
        T = arr.shape[0]
        for k, s in enumerate(seq_starts):
            lo, hi = max(0, s - epoch_pre), min(T, s + epoch_post)
            out[k, (lo - (s - epoch_pre)):(hi - (s - epoch_pre))] = arr[lo:hi]
    else:
        out = np.zeros((n_seq, arr.shape[0], epoch_len))
        T = arr.shape[1]
        for k, s in enumerate(seq_starts):
            lo, hi = max(0, s - epoch_pre), min(T, s + epoch_post)
            out[k, :, (lo - (s - epoch_pre)):(hi - (s - epoch_pre))] = arr[:, lo:hi]
    return out


def epoch_E(res: dict) -> np.ndarray:
    """Per-trial E epochs: shape (n_seq, N, pre+stim+post)."""
    cfg: SyllableConfig = res["syllable_cfg"]
    return evoked_per_trial(res["E"], res["seq_starts"],
                            cfg.pre_stim_ms, cfg.stim_steps + cfg.post_stim_ms)


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


_WORD_COLOURS: Dict[str, str] = {
    "ABC": "tab:red",   "ACB": "tab:red",   "CAB": "tab:red",
    "ABD": "tab:blue",  "ADB": "tab:blue",  "DAB": "tab:blue",
    "ABE": "tab:green", "AEB": "tab:green", "EAB": "tab:green",
}


def _word_colour(word: str) -> str:
    return _WORD_COLOURS.get(word, "0.4")


# Per-syllable colours for the figure-bank and population-rate plots.
_SYLL_COLOURS: Dict[str, str] = {
    "A": "tab:purple", "B": "tab:orange", "C": "tab:red",
    "D": "tab:blue",   "E": "tab:green",
}


def _syll_colour(syll: str) -> str:
    return _SYLL_COLOURS.get(syll, "0.4")


def _inh_tag(a1_cfg: A1Config) -> str:
    """Human-readable inhibition-structure tag inferred from a1_cfg."""
    if (a1_cfg.w_EI_self == a1_cfg.w_EI_lat
            and a1_cfg.w_IE_self == a1_cfg.w_IE_lat):
        return "uniform inhibition"
    return "selective inhibition"


def _cross_weight(W: np.ndarray, cfg: SyllableConfig,
                  pre_syll: str, post_syll: str) -> float:
    """Mean recurrent weight from ``pre_syll``'s channels to
    ``post_syll``'s channels (excluding self-channel pairs).

    With figure encoding a syllable spans several channels, so the
    'predictive' link pre -> post is a *block* of W entries; we summarise
    it by its mean over off-diagonal (pre != post channel) pairs.
    """
    pre = cfg.active_channels(pre_syll)
    post = cfg.active_channels(post_syll)
    if len(pre) == 0 or len(post) == 0:
        return 0.0
    block = W[np.ix_(post, pre)]
    mask = post[:, None] != pre[None, :]
    vals = block[mask]
    return float(vals.mean()) if vals.size else float(block.mean())


def plot_syllable_figures(cfg: SyllableConfig, fname: str):
    """The five syllable figures (spectral patterns) over channels.

    Inhibition-independent -- characterises the stimulus set only.
    """
    figs = {s: cfg.figure(s) for s in cfg.syllables}
    fmat = cfg.figure_matrix                       # (n_channels, n_syllables)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.suptitle("Syllable figures (spectral activation patterns)",
                 fontsize=13, fontweight="bold")

    # ----- Left: per-channel amplitude as stems (channels are discrete
    #        and non-contiguous, so markers -- not lines -- are honest).
    #        A small per-syllable x-offset separates the shared-core
    #        channels that all syllables occupy. -----
    ax = axes[0]
    n_s = cfg.n_syllables
    for k, s in enumerate(cfg.syllables):
        chans = cfg.active_channels(s)
        amps = figs[s][chans]
        off = (k - (n_s - 1) / 2) * 0.13
        ax.vlines(chans + off, 0, amps, color=_syll_colour(s),
                  lw=1.4, alpha=0.85)
        ax.plot(chans + off, amps, "o", color=_syll_colour(s), ms=4,
                label=f"/{s}/")
    ax.legend(fontsize=9, frameon=False, ncol=n_s)
    _setup_axes(ax, title="per-channel amplitude",
                xlabel="channel (log-frequency)", ylabel="amplitude")

    # ----- Right: image (syllable x channel) -----
    ax = axes[1]
    im = ax.imshow(fmat.T, aspect="auto", origin="lower", cmap="magma",
                   interpolation="nearest",
                   extent=[-0.5, cfg.n_channels - 0.5,
                           -0.5, cfg.n_syllables - 0.5])
    ax.set_yticks(range(cfg.n_syllables))
    ax.set_yticklabels([f"/{s}/" for s in cfg.syllables])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="amplitude")
    _setup_axes(ax, title="figure matrix", xlabel="channel")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_session_overview(res: dict, fname: str, n_trials_show: int = 8):
    """Stimulus raster + per-syllable population E rate + weight evolution."""
    a1_cfg = res["cfg"]
    cfg: SyllableConfig = res["syllable_cfg"]
    dt = a1_cfg.dt
    stim = res["stim"]
    E = res["E"]
    Wt = np.stack(res["W_traj"]) if len(res["W_traj"]) else None
    W_t = res["W_t"]

    show_T = min(n_trials_show * cfg.trial_period, stim.shape[1])
    t_show = res["t"][:show_T]
    starts_shown = res["seq_starts"][res["seq_starts"] < show_T]

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    fig.suptitle(f"Syllable roving, model0 [{_inh_tag(a1_cfg)}] "
                 f"({cfg.n_blocks} blocks x {cfg.n_reps_per_block} reps "
                 f"= {cfg.n_total_seqs} trials, deviant pos "
                 f"{cfg.deviant_syllable_pos})",
                 fontsize=13, fontweight="bold")

    # ----- Row 1: stimulus raster (all channels) -----
    ax = fig.add_subplot(gs[0])
    ax.imshow(stim[:, :show_T], aspect="auto", origin="lower",
              cmap="Oranges", interpolation="nearest",
              extent=[0, show_T * dt, -0.5, a1_cfg.N - 0.5])
    for k, s in enumerate(starts_shown):
        ax.axvline(s * dt, color="0.4", lw=0.5, ls=":")
        word = res["seq_word"][k]
        ax.text(s * dt + cfg.stim_steps * dt / 2, a1_cfg.N - 0.5,
                word, ha="center", va="top", fontsize=9,
                color=_word_colour(word))
    _setup_axes(ax, title=f"Stimulus raster (first {len(starts_shown)} trials)",
                xlabel="time (s)", ylabel="channel")

    # ----- Row 2: per-syllable population E rate -----
    # Sum E across each syllable's active channels (the population the
    # figure drives).  Five traces instead of 35 raw channels.
    ax = fig.add_subplot(gs[1])
    for s in cfg.syllables:
        chans = cfg.active_channels(s)
        pop = E[chans, :show_T].sum(axis=0)
        ax.plot(t_show, pop, color=_syll_colour(s), lw=1.0,
                label=f"E[/{s}/]")
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              ncol=cfg.n_syllables)
    _setup_axes(ax, title="Excitatory population rate per syllable "
                          "(sum over active channels)",
                xlabel="time (s)", ylabel="summed rate")

    # ----- Row 3: recurrent weight evolution (syllable-block means) -----
    ax = fig.add_subplot(gs[2])
    if Wt is not None:
        # Predictive transitions, summarised as the mean cross-block
        # weight between the syllables' active-channel sets:
        #   <W[B<-A]>      -- the A -> B transition (always present)
        #   <W[{C,D,E}<-B]> -- the B -> {variable} links
        ab = np.array([_cross_weight(Wt[k], cfg, "A", "B")
                       for k in range(Wt.shape[0])])
        ax.plot(W_t, ab, color="0.20", lw=2.0,
                label=r"$\langle W_{B\leftarrow A}\rangle$")
        for var in cfg.variable_syllables:
            vb = np.array([_cross_weight(Wt[k], cfg, "B", var)
                           for k in range(Wt.shape[0])])
            ax.plot(W_t, vb, color=_syll_colour(var), lw=1.6,
                    label=fr"$\langle W_{{{var}\leftarrow B}}\rangle$")
        ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    _setup_axes(ax, title="Recurrent E->E weight evolution "
                          "(mean over syllable channel blocks)",
                xlabel="time (s)", ylabel="mean W")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_repetition_suppression(res: dict, fname: str):
    """Repetition suppression across the full 3-syllable sequence.

    Row 1: per-word population response (sum of E across the channels the
           word actually drives), rep 1 vs rep N.
    Row 2: surprisal trace, defined as rep_1 - rep_N (the part of the
           novel response that fades with repetition).  This is the
           model-side analogue of the rep-1-vs-rep-15 decoding signal
           seen in ECoG of the local-global / roving task.
    """
    a1_cfg = res["cfg"]
    cfg: SyllableConfig = res["syllable_cfg"]
    dt = a1_cfg.dt

    epochs = epoch_E(res)                 # (n_seq, N, epoch_len)
    rep = res["seq_rep"]; word = res["seq_word"]
    n_reps = cfg.n_reps_per_block

    # Full epoch window: pre + 3 syllables + post.
    epoch_len = cfg.epoch_steps
    ts = (np.arange(epoch_len) - cfg.pre_stim_ms) * dt   # 0 at seq onset
    syll_onsets_s = [(o - cfg.pre_stim_ms) * dt
                     for o in cfg.syll_onsets_in_epoch]
    syll_dur_s = cfg.syll_dur * dt

    words = list(cfg.words)
    fig, axes = plt.subplots(2, len(words),
                             figsize=(4.6 * len(words), 6.4),
                             constrained_layout=True, sharex=True)
    if len(words) == 1:
        axes = axes[:, None]
    fig.suptitle(f"Syllable roving: repetition suppression and surprisal "
                 f"[{_inh_tag(a1_cfg)}, deviant pos "
                 f"{cfg.deviant_syllable_pos}]",
                 fontsize=13, fontweight="bold")

    dev_idx = cfg.deviant_syllable_index

    for col, w in enumerate(words):
        # Channels this word drives = union of its syllables' figures.
        chans = cfg.word_channels(w)
        sel_first = (word == w) & (rep == 0)
        sel_last  = (word == w) & (rep == n_reps - 1)

        # Population response = sum across the word's active channels.
        first_pop = epochs[sel_first][:, chans, :].sum(axis=1)   # (n_first, T)
        last_pop  = epochs[sel_last ][:, chans, :].sum(axis=1)

        first_mean = first_pop.mean(0) if len(first_pop) else np.zeros(epoch_len)
        last_mean  = last_pop.mean(0)  if len(last_pop)  else np.zeros(epoch_len)
        first_sem  = (first_pop.std(0) / max(1.0, np.sqrt(len(first_pop)))
                      if len(first_pop) else np.zeros(epoch_len))
        last_sem   = (last_pop.std(0)  / max(1.0, np.sqrt(len(last_pop)))
                      if len(last_pop)  else np.zeros(epoch_len))

        col_word = _word_colour(w)
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # ----- syllable-window shading -----
        for si, t0_s in enumerate(syll_onsets_s):
            syll_char = w[si]
            shade = _word_colour(w) if si == dev_idx else "0.5"
            alpha = 0.13 if si == dev_idx else 0.06
            for ax in (ax_top, ax_bot):
                ax.axvspan(t0_s, t0_s + syll_dur_s, color=shade, alpha=alpha)
            ax_top.text(t0_s + syll_dur_s / 2, 0, f"/{syll_char}/",
                        ha="center", va="bottom", fontsize=9, color="0.3")

        # ----- Row 1: rep 1 vs rep N -----
        ax_top.fill_between(ts, first_mean - first_sem, first_mean + first_sem,
                            color=col_word, alpha=0.18, lw=0)
        ax_top.fill_between(ts, last_mean  - last_sem,  last_mean  + last_sem,
                            color=col_word, alpha=0.10, lw=0)
        ax_top.plot(ts, first_mean, color=col_word, lw=2.0,
                    label=f"rep 1 (n={len(first_pop)})")
        ax_top.plot(ts, last_mean,  color=col_word, lw=2.0, ls="--",
                    label=f"rep {n_reps} (n={len(last_pop)})")
        ax_top.legend(fontsize=8, frameon=False, loc="upper right")
        _setup_axes(ax_top,
                    title=f"word {w}  (population E across {len(chans)} channels)",
                    ylabel="summed E rate")

        # ----- Row 2: surprisal = rep 1 - rep N -----
        surp = first_mean - last_mean
        ax_bot.axhline(0, color="0.7", lw=0.8)
        ax_bot.plot(ts, surp, color=col_word, lw=2.0)
        ax_bot.fill_between(ts, 0, surp, color=col_word, alpha=0.15)
        _setup_axes(ax_bot, title="surprisal  (rep 1 - rep N)",
                    xlabel="time from sequence onset (s)",
                    ylabel=r"$\Delta$E rate")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_block_dynamics(res: dict, fname: str):
    """Peak deviant-syllable population E rate across the 15 reps in a block."""
    a1_cfg = res["cfg"]
    cfg: SyllableConfig = res["syllable_cfg"]

    epochs = epoch_E(res)
    rep = res["seq_rep"]; word = res["seq_word"]
    n_reps = cfg.n_reps_per_block

    dev_idx = cfg.deviant_syllable_index
    dev_t = cfg.syll_onsets_in_epoch[dev_idx]
    win = slice(dev_t, dev_t + cfg.syll_dur)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    fig.suptitle(f"[{_inh_tag(a1_cfg)}]",
                 fontsize=11, fontweight="bold")
    for w in cfg.words:
        dev_char = w[dev_idx]
        dev_ch = cfg.active_channels(dev_char)
        means, sems = [], []
        for r in range(n_reps):
            sel = (word == w) & (rep == r)
            if not sel.any():
                means.append(np.nan); sems.append(np.nan); continue
            # Population trace = sum over the deviant syllable's channels;
            # peak within the deviant-syllable window.
            peaks = epochs[sel][:, dev_ch, win].sum(axis=1).max(axis=1)
            means.append(peaks.mean())
            sems.append(peaks.std() / max(1.0, np.sqrt(len(peaks))))
        ax.errorbar(np.arange(1, n_reps + 1), means, yerr=sems,
                    color=_word_colour(w), lw=2.0, marker="o", ms=4,
                    capsize=2.5,
                    label=f"word {w} (deviant /{dev_char}/)")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax,
                title="Repetition suppression: peak population E at the "
                      "deviant syllable",
                xlabel="repetition within block (1..15)",
                ylabel="peak summed E rate")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    cfg = get_preset("default")

    # The syllable figures are inhibition-independent -- save the figure
    # bank once and report the realised channel structure.
    print(f"[ Syllable figures ]")
    print(f"  {cfg.n_channels} channels; {cfg.n_active} active per syllable; "
          f"common core = {cfg.core_size} channels")
    print(f"  mean pairwise overlap = {100 * cfg.mean_pairwise_overlap():.1f}% "
          f"(target {100 * cfg.overlap:.0f}%)")
    for s in cfg.syllables:
        ch = cfg.active_channels(s)
        print(f"    /{s}/  channels {list(ch)}  mean amp "
              f"{cfg.figure(s)[ch].mean():.3f}")
    plot_syllable_figures(cfg, "syllable_m0_figures.png")

    # Run both inhibition-structure presets back to back.  Filenames are
    # tagged with the preset name so the figures don't overwrite each
    # other.  Multi-timescale TC depression is enabled in both -- its
    # 5 s component survives the 1 s ITI and integrates across the 15
    # reps, which is what makes rep 1 differ from rep 15 at all.
    for inh_name in INH_PRESETS:
        a1_cfg = shared_config(N=cfg.n_channels, inh=inh_name)

        print(f"\n========================================================")
        print(f"[ Syllable roving -- inhibition preset '{inh_name}' ]")
        print(f"  words = {cfg.words}, deviant_syllable_pos = "
              f"{cfg.deviant_syllable_pos}")
        print(f"  {cfg.n_channels} channels; {cfg.n_blocks} blocks x "
              f"{cfg.n_reps_per_block} reps = {cfg.n_total_seqs} trials")
        print(f"  trial period {cfg.trial_period} ms; total sim "
              f"{cfg.n_total_seqs * cfg.trial_period / 1000:.1f} s")
        print(f"  w_EI = (self {a1_cfg.w_EI_self}, lat {a1_cfg.w_EI_lat}); "
              f"w_IE = (self {a1_cfg.w_IE_self}, lat {a1_cfg.w_IE_lat})")

        res = run_experiment(cfg=cfg, a1_cfg=a1_cfg)

        print("[ Plotting ]")
        plot_session_overview(res,       f"syllable_m0_overview_{inh_name}.png")
        plot_repetition_suppression(res, f"syllable_m0_repsupp_{inh_name}.png")
        plot_block_dynamics(res,         f"syllable_m0_blockdyn_{inh_name}.png")

        W = res["W_final"]
        print(f"  Final weight summary [{inh_name}] (mean cross-block W):")
        print(f"    <W[B<-A]> = {_cross_weight(W, cfg, 'A', 'B'):.3f}")
        for var in cfg.variable_syllables:
            print(f"    <W[{var}<-B]> = {_cross_weight(W, cfg, 'B', var):.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
