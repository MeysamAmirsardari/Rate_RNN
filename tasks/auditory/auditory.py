"""
tasks.auditory.auditory
=======================

Roving Oddball paradigm for model0 driven by **real speech**.

Each syllable is the mel-spectrogram of a recorded WAV (A="boo",
B="pee", C="tah", D="bey"/bay, E="see"), sampled at one frame per model
step and normalised to a per-channel thalamo-cortical drive over
``n_channels`` (default 180) mel bands -- an artificial cochleo-thalamic
input to the cortical model.  Apart from the stimulus, the paradigm is
identical to ``tasks.roving`` / ``tasks.syllable``: blocks of one
repeated 3-syllable word, constrained-random block order, plasticity on
throughout, deviant at position 3 (ABC / ABD / ABE).

At 180 channels the model's lateral inhibition acts as a global
normaliser: the dense log-mel drive is contrast-enhanced so that mainly
the above-average mel bands fire -- a sparse cortical code for a dense
thalamic input.

Outputs
-------
Inhibition-independent (saved once):
    auditory_m0_spectrograms.png  the five syllable mel-spectrograms (dB).
    auditory_m0_input.png         the assembled thalamic drive for one
                                  example sequence of each word
                                  (the model input, in a dedicated figure).

Per inhibition preset:
    auditory_m0_overview_*.png    drive raster, per-syllable population E,
                                  recurrent weight evolution.
    auditory_m0_repsupp_*.png     per-word population response rep 1 vs
                                  rep N + surprisal.
    auditory_m0_blockdyn_*.png    peak deviant-syllable population E across
                                  the 15 repetitions within a block.

This file implements only the task -- model0 itself is unchanged.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Put the project root on sys.path so the package-relative import below
# works in both `python -m tasks.auditory.auditory` and
# `python tasks/auditory/auditory.py` modes.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, INH_PRESETS, simulate

if __package__:
    from .config import AuditoryConfig, get_preset
    from .audio import MelEncoder
else:
    from tasks.auditory.config import AuditoryConfig, get_preset
    from tasks.auditory.audio import MelEncoder


# =====================================================================
#  Block-order generation
# =====================================================================
def generate_block_order(cfg: AuditoryConfig,
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
def build_auditory_stim(
    block_order: List[str],
    cfg: AuditoryConfig,
    a1_cfg: A1Config,
    enc: MelEncoder,
) -> Dict:
    """Build the full-session (N, T_total) stimulus from the syllable drives.

    Each syllable slot is filled with that syllable's normalised mel-drive
    (n_channels x syll_dur).

    Returns dict with: stim, seq_starts, trial_period, seq_word,
    seq_block, seq_rep, block_order.
    """
    if abs(a1_cfg.dt - 1e-3) > 1e-9:
        raise ValueError(
            f"AuditoryConfig assumes a1_cfg.dt = 1 ms; got {a1_cfg.dt}")
    if a1_cfg.N != cfg.n_channels:
        raise ValueError(
            f"a1_cfg.N = {a1_cfg.N} must equal cfg.n_channels = "
            f"{cfg.n_channels}")

    trial_period = cfg.trial_period
    n_total = cfg.n_total_seqs
    T_total = trial_period * n_total
    slot = cfg.syll_dur + cfg.syll_gap

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
                stim[:, ts:ts + cfg.syll_dur] = enc.drive[syll]
            seq_idx += 1

    return dict(
        stim=stim, seq_starts=seq_starts, trial_period=trial_period,
        seq_word=seq_word, seq_block=seq_block, seq_rep=seq_rep,
        block_order=list(block_order),
    )


def build_example_sequence(word: str, cfg: AuditoryConfig,
                           enc: MelEncoder) -> np.ndarray:
    """The thalamic drive (n_channels, stim_steps) for a single ``word``."""
    slot = cfg.syll_dur + cfg.syll_gap
    seq = np.zeros((cfg.n_channels, cfg.stim_steps))
    for i, syll in enumerate(word):
        ts = i * slot
        seq[:, ts:ts + cfg.syll_dur] = enc.drive[syll]
    return seq


# =====================================================================
#  Experiment
# =====================================================================
def run_experiment(
    cfg: Optional[AuditoryConfig] = None,
    a1_cfg: Optional[A1Config] = None,
    enc: Optional[MelEncoder] = None,
) -> dict:
    """Build the speech-driven session stimulus and run model0 on it."""
    if cfg is None:
        cfg = AuditoryConfig()
    if a1_cfg is None:
        a1_cfg = A1Config(N=cfg.n_channels)
    if enc is None:
        enc = MelEncoder(cfg)

    rng = np.random.default_rng(cfg.seed)
    block_order = generate_block_order(cfg, rng)
    stim_pack = build_auditory_stim(block_order, cfg, a1_cfg, enc)

    snap_every = max(1, int(round(0.5 / a1_cfg.dt)))   # every 500 ms
    out = simulate(stim_pack["stim"], cfg=a1_cfg,
                   record_W_every=snap_every, seed=cfg.seed)

    out.update(stim_pack)
    out["auditory_cfg"] = cfg
    out["encoder"] = enc
    return out


# =====================================================================
#  Epoch extraction
# =====================================================================
def evoked_per_trial(arr: np.ndarray, seq_starts: np.ndarray,
                     epoch_pre: int, epoch_post: int) -> np.ndarray:
    """Cut an (N, T) or (T,) history into per-trial epochs (zero-padded)."""
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
    cfg: AuditoryConfig = res["auditory_cfg"]
    return evoked_per_trial(res["E"], res["seq_starts"],
                            cfg.pre_stim_ms, cfg.stim_steps + cfg.post_stim_ms)


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


_WORD_COLOURS: Dict[str, str] = {
    "ABC": "tab:red",   "ACB": "tab:red",   "CAB": "tab:red",
    "ABD": "tab:blue",  "ADB": "tab:blue",  "DAB": "tab:blue",
    "ABE": "tab:green", "AEB": "tab:green", "EAB": "tab:green",
}


def _word_colour(word: str) -> str:
    return _WORD_COLOURS.get(word, "0.4")


_SYLL_COLOURS: Dict[str, str] = {
    "A": "tab:purple", "B": "tab:orange", "C": "tab:red",
    "D": "tab:blue",   "E": "tab:green",
}


def _syll_colour(syll: str) -> str:
    return _SYLL_COLOURS.get(syll, "0.4")


def _syll_label(cfg: AuditoryConfig, syll: str) -> str:
    return f"/{syll}/ ({cfg.syllable_files[syll]})"


def _inh_tag(a1_cfg: A1Config) -> str:
    if (a1_cfg.w_EI_self == a1_cfg.w_EI_lat
            and a1_cfg.w_IE_self == a1_cfg.w_IE_lat):
        return "uniform inhibition"
    return "selective inhibition"


def _cross_weight(W: np.ndarray, enc: MelEncoder,
                  pre_syll: str, post_syll: str) -> float:
    """Mean recurrent weight from ``pre_syll``'s channels to ``post_syll``'s
    channels (off-diagonal pairs only)."""
    pre = enc.active_channels(pre_syll)
    post = enc.active_channels(post_syll)
    if len(pre) == 0 or len(post) == 0:
        return 0.0
    block = W[np.ix_(post, pre)]
    mask = post[:, None] != pre[None, :]
    vals = block[mask]
    return float(vals.mean()) if vals.size else float(block.mean())


# =====================================================================
#  Plots: stimulus / input (inhibition-independent)
# =====================================================================
def plot_spectrograms(enc: MelEncoder, cfg: AuditoryConfig, fname: str):
    """The five syllable mel-spectrograms (log-mel, dB) -- the auditory
    front-end representation that becomes the model input."""
    freqs = enc.mel_frequencies()
    syls = list(cfg.syllables)
    fig, axes = plt.subplots(1, len(syls), figsize=(3.1 * len(syls), 4.0),
                             constrained_layout=True, sharey=True)
    if len(syls) == 1:
        axes = [axes]
    fig.suptitle("Syllable mel-spectrograms (log-mel, dB) -- thalamic front end",
                 fontsize=13, fontweight="bold")

    vmin = -cfg.top_db
    im = None
    for ax, s in zip(axes, syls):
        db = enc.db[s]                                  # (n_channels, T_raw)
        T_ms = db.shape[1] * cfg.hop_ms
        im = ax.imshow(db, aspect="auto", origin="lower", cmap="magma",
                       vmin=vmin, vmax=0.0, interpolation="nearest",
                       extent=[0, T_ms, 0, cfg.n_channels])
        _setup_axes(ax, title=_syll_label(cfg, s), xlabel="time (ms)")
        # secondary y labels in kHz at a few channels
    axes[0].set_ylabel("mel channel", fontsize=10)
    # frequency ticks (kHz) on a twin axis of the last panel
    ax2 = axes[-1].twinx()
    ax2.set_ylim(0, cfg.n_channels)
    n_ticks = 6
    tick_idx = np.linspace(0, cfg.n_channels - 1, n_ticks).astype(int)
    ax2.set_yticks(tick_idx)
    ax2.set_yticklabels([f"{freqs[i]/1000:.1f}" for i in tick_idx], fontsize=8)
    ax2.set_ylabel("freq (kHz)", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.06, label="dB")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


def plot_input(enc: MelEncoder, cfg: AuditoryConfig, fname: str):
    """Dedicated figure of the assembled thalamic drive (the model input)
    for one example sequence of each word."""
    words = list(cfg.words)
    dt_ms = 1.0
    fig, axes = plt.subplots(1, len(words), figsize=(4.4 * len(words), 4.2),
                             constrained_layout=True, sharey=True)
    if len(words) == 1:
        axes = [axes]
    fig.suptitle("Model input: assembled thalamic drive per word "
                 f"(syll_amp={cfg.syll_amp}, top_db={cfg.top_db:.0f})",
                 fontsize=13, fontweight="bold")

    slot = cfg.syll_dur + cfg.syll_gap
    im = None
    for ax, w in zip(axes, words):
        seq = build_example_sequence(w, cfg, enc)       # (n_channels, stim_steps)
        T_ms = seq.shape[1] * dt_ms
        im = ax.imshow(seq, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0.0, vmax=cfg.syll_amp, interpolation="nearest",
                       extent=[0, T_ms, 0, cfg.n_channels])
        for i, syll in enumerate(w):
            t0 = i * slot
            ax.axvline(t0, color="w", lw=0.6, ls=":")
            ax.text(t0 + cfg.syll_dur / 2, cfg.n_channels * 0.97,
                    _syll_label(cfg, syll), ha="center", va="top",
                    fontsize=9, color="w")
        _setup_axes(ax, title=f"word {w}", xlabel="time (ms)")
    axes[0].set_ylabel("mel channel", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="drive")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Plots: model responses (per inhibition preset)
# =====================================================================
def plot_session_overview(res: dict, fname: str, n_trials_show: int = 6):
    """Drive raster + per-syllable population E rate + weight evolution."""
    a1_cfg = res["cfg"]
    cfg: AuditoryConfig = res["auditory_cfg"]
    enc: MelEncoder = res["encoder"]
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
    fig.suptitle(f"Auditory roving, model0 [{_inh_tag(a1_cfg)}] "
                 f"({cfg.n_channels} mel ch, {cfg.n_blocks} blocks x "
                 f"{cfg.n_reps_per_block} reps = {cfg.n_total_seqs} trials, "
                 f"deviant pos {cfg.deviant_syllable_pos})",
                 fontsize=13, fontweight="bold")

    # ----- Row 1: thalamic-drive raster -----
    ax = fig.add_subplot(gs[0])
    ax.imshow(stim[:, :show_T], aspect="auto", origin="lower",
              cmap="viridis", interpolation="nearest",
              extent=[0, show_T * dt, 0, a1_cfg.N])
    for k, s in enumerate(starts_shown):
        ax.axvline(s * dt, color="w", lw=0.5, ls=":")
        word = res["seq_word"][k]
        ax.text(s * dt + cfg.stim_steps * dt / 2, a1_cfg.N * 0.97,
                word, ha="center", va="top", fontsize=9,
                color=_word_colour(word))
    _setup_axes(ax, title=f"Thalamic drive raster (first {len(starts_shown)} trials)",
                xlabel="time (s)", ylabel="mel channel")

    # ----- Row 2: per-syllable population E rate -----
    ax = fig.add_subplot(gs[1])
    for s in cfg.syllables:
        chans = enc.active_channels(s)
        pop = E[chans, :show_T].sum(axis=0)
        ax.plot(t_show, pop, color=_syll_colour(s), lw=1.0,
                label=_syll_label(cfg, s))
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              ncol=cfg.n_syllables)
    _setup_axes(ax, title="Excitatory population rate per syllable "
                          "(sum over active mel channels)",
                xlabel="time (s)", ylabel="summed rate")

    # ----- Row 3: recurrent weight evolution -----
    ax = fig.add_subplot(gs[2])
    if Wt is not None:
        ab = np.array([_cross_weight(Wt[k], enc, "A", "B")
                       for k in range(Wt.shape[0])])
        ax.plot(W_t, ab, color="0.20", lw=2.0,
                label=r"$\langle W_{B\leftarrow A}\rangle$")
        for var in cfg.variable_syllables:
            vb = np.array([_cross_weight(Wt[k], enc, "B", var)
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

    Row 1: per-word population response (sum of E across the mel channels
           the word drives), rep 1 vs rep N.
    Row 2: surprisal = rep_1 - rep_N.
    """
    a1_cfg = res["cfg"]
    cfg: AuditoryConfig = res["auditory_cfg"]
    enc: MelEncoder = res["encoder"]
    dt = a1_cfg.dt

    epochs = epoch_E(res)                 # (n_seq, N, epoch_len)
    rep = res["seq_rep"]; word = res["seq_word"]
    n_reps = cfg.n_reps_per_block

    epoch_len = cfg.epoch_steps
    ts = (np.arange(epoch_len) - cfg.pre_stim_ms) * dt
    syll_onsets_s = [(o - cfg.pre_stim_ms) * dt
                     for o in cfg.syll_onsets_in_epoch]
    syll_dur_s = cfg.syll_dur * dt

    words = list(cfg.words)
    fig, axes = plt.subplots(2, len(words),
                             figsize=(4.6 * len(words), 6.4),
                             constrained_layout=True, sharex=True)
    if len(words) == 1:
        axes = axes[:, None]
    fig.suptitle(f"Auditory roving: repetition suppression and surprisal "
                 f"[{_inh_tag(a1_cfg)}, deviant pos "
                 f"{cfg.deviant_syllable_pos}]",
                 fontsize=13, fontweight="bold")

    dev_idx = cfg.deviant_syllable_index

    for col, w in enumerate(words):
        chans = enc.word_channels(w)
        sel_first = (word == w) & (rep == 0)
        sel_last  = (word == w) & (rep == n_reps - 1)

        first_pop = epochs[sel_first][:, chans, :].sum(axis=1)
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

        for si, t0_s in enumerate(syll_onsets_s):
            syll_char = w[si]
            shade = _word_colour(w) if si == dev_idx else "0.5"
            alpha = 0.13 if si == dev_idx else 0.06
            for ax in (ax_top, ax_bot):
                ax.axvspan(t0_s, t0_s + syll_dur_s, color=shade, alpha=alpha)
            ax_top.text(t0_s + syll_dur_s / 2, 0, f"/{syll_char}/",
                        ha="center", va="bottom", fontsize=9, color="0.3")

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
                    title=f"word {w}  (population E across {len(chans)} mel ch)",
                    ylabel="summed E rate")

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
    """Peak deviant-syllable population E rate across the reps in a block."""
    a1_cfg = res["cfg"]
    cfg: AuditoryConfig = res["auditory_cfg"]
    enc: MelEncoder = res["encoder"]

    epochs = epoch_E(res)
    rep = res["seq_rep"]; word = res["seq_word"]
    n_reps = cfg.n_reps_per_block

    dev_idx = cfg.deviant_syllable_index
    dev_t = cfg.syll_onsets_in_epoch[dev_idx]
    win = slice(dev_t, dev_t + cfg.syll_dur)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    fig.suptitle(f"[{_inh_tag(a1_cfg)}]", fontsize=11, fontweight="bold")
    for w in cfg.words:
        dev_char = w[dev_idx]
        dev_ch = enc.active_channels(dev_char)
        means, sems = [], []
        for r in range(n_reps):
            sel = (word == w) & (rep == r)
            if not sel.any():
                means.append(np.nan); sems.append(np.nan); continue
            peaks = epochs[sel][:, dev_ch, win].sum(axis=1).max(axis=1)
            means.append(peaks.mean())
            sems.append(peaks.std() / max(1.0, np.sqrt(len(peaks))))
        ax.errorbar(np.arange(1, n_reps + 1), means, yerr=sems,
                    color=_word_colour(w), lw=2.0, marker="o", ms=4,
                    capsize=2.5,
                    label=f"word {w} (deviant {_syll_label(cfg, dev_char)})")
    ax.legend(fontsize=9, frameon=False)
    _setup_axes(ax,
                title="Repetition suppression: peak population E at the "
                      "deviant syllable",
                xlabel="repetition within block (1..N)",
                ylabel="peak summed E rate")

    fig.savefig(fname, dpi=150)
    print(f"  saved {fname}")
    return fig


# =====================================================================
#  Main
# =====================================================================
def main():
    warnings.filterwarnings("ignore", message="Empty filters detected")
    cfg = get_preset("default")
    enc = MelEncoder(cfg)

    print(f"[ Auditory front end ]")
    print(f"  {cfg.n_channels} mel channels, sr={cfg.sr} Hz, n_fft={cfg.n_fft}, "
          f"hop={cfg.hop_length} samp ({cfg.hop_ms} ms), "
          f"fmin/fmax={cfg.fmin:.0f}/{cfg.fmax:.0f} Hz, top_db={cfg.top_db:.0f}")
    print(f"  empty (all-silent) mel channels: {enc.n_empty_channels()}"
          f"/{cfg.n_channels}")
    for s in cfg.syllables:
        d = enc.drive[s]
        print(f"    /{s}/={cfg.syllable_files[s]:4s} drive peak={d.max():.3f} "
              f"mean={d.mean():.3f} active_ch={len(enc.active_channels(s))}")

    # Inhibition-independent figures (saved once).
    plot_spectrograms(enc, cfg, "auditory_m0_spectrograms.png")
    plot_input(enc, cfg, "auditory_m0_input.png")

    # Run both inhibition-structure presets back to back.
    for inh_name, inh_factory in INH_PRESETS.items():
        a1_cfg = inh_factory(N=cfg.n_channels, multiscale_std=True)

        print(f"\n========================================================")
        print(f"[ Auditory roving -- inhibition preset '{inh_name}' ]")
        print(f"  words = {cfg.words}, deviant_syllable_pos = "
              f"{cfg.deviant_syllable_pos}")
        print(f"  {cfg.n_blocks} blocks x {cfg.n_reps_per_block} reps "
              f"= {cfg.n_total_seqs} trials")
        print(f"  trial period {cfg.trial_period} ms; total sim "
              f"{cfg.n_total_seqs * cfg.trial_period / 1000:.1f} s")
        print(f"  w_EI = (self {a1_cfg.w_EI_self}, lat {a1_cfg.w_EI_lat}); "
              f"w_IE = (self {a1_cfg.w_IE_self}, lat {a1_cfg.w_IE_lat})")

        res = run_experiment(cfg=cfg, a1_cfg=a1_cfg, enc=enc)

        print("[ Plotting ]")
        plot_session_overview(res,       f"auditory_m0_overview_{inh_name}.png")
        plot_repetition_suppression(res, f"auditory_m0_repsupp_{inh_name}.png")
        plot_block_dynamics(res,         f"auditory_m0_blockdyn_{inh_name}.png")

        W = res["W_final"]
        print(f"  Final weight summary [{inh_name}] (mean cross-block W; "
              f"diluted over ~100-ch active sets, so O(1e-5)):")
        print(f"    <W[B<-A]> = {_cross_weight(W, enc, 'A', 'B'):.3e}")
        for var in cfg.variable_syllables:
            print(f"    <W[{var}<-B]> = {_cross_weight(W, enc, 'B', var):.3e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
