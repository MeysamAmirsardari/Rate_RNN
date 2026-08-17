"""
tasks.double_tone.double_tone
=============================

The Wacongne, Changeux & Dehaene (2012) double-tone paradigm on model0.

    python -m tasks.double_tone.double_tone [--preset short] [--presets selective]

Runs the same stimulus twice per inhibition preset:

    plastic   learning is on in both phases -- the predictive-coding model
    frozen    W is pinned at zero throughout -- the habituation model,
              which has short-term depression and nothing else

The contrast between the two columns is the paper's Figure 8B: habituation
alone predicts the second tone of a rare AA pair to be SMALL, because it
repeats a tone heard 200 ms earlier; prediction says it should be LARGE,
because the network expected B.  Only the plastic column should separate
the four conditions.

Outputs, written next to this file rather than into the working directory
    double_tone_traces_<inh>.png    per-condition E / I / inhibition traces
    double_tone_responses_<inh>.png habituation vs predictive bar chart
    double_tone_weights_<inh>.png   W[B<-A] across the session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, simulate

if __package__:
    from .config import (PAIRS, RARE, FREQUENT, TONE_A, TONE_B,
                         DoubleToneConfig, get_preset,
                         model_config, uniform_model_config, global_model_config)
else:
    from tasks.double_tone.config import (  # type: ignore
        PAIRS, RARE, FREQUENT, TONE_A, TONE_B,
        DoubleToneConfig, get_preset, model_config, uniform_model_config,
        global_model_config)


#: Figures are written next to the code that makes them.
OUT_DIR = Path(__file__).resolve().parent


def _out(name: str) -> str:
    return str(OUT_DIR / name)


COND_COLORS = {"AB": "#22409A", "AA": "#E8442A",
               "BB": "#9B1B1B", "BA": "#2E9B47"}
COND_ORDER = ("AB", "AA", "BB", "BA")
COND_LABEL = {"AB": "freq AB", "AA": "rare AA",
              "BB": "rare BB", "BA": "rare BA"}


# =====================================================================
#  Trial order
# =====================================================================
def generate_pair_order(cfg: DoubleToneConfig,
                        rng: np.random.Generator) -> List[str]:
    """Test-phase pair order with no two consecutive rare pairs."""
    counts = dict(cfg.n_each)
    order: List[str] = []
    for _ in range(cfg.n_test):
        avail = [k for k, v in counts.items() if v > 0]
        if cfg.no_consecutive_rare and order and order[-1] in RARE:
            non_rare = [k for k in avail if k not in RARE]
            if non_rare:
                avail = non_rare
        # Sample proportionally to what is left, so the tail stays mixed.
        weights = np.array([counts[k] for k in avail], dtype=float)
        choice = avail[int(rng.choice(len(avail), p=weights / weights.sum()))]
        order.append(choice)
        counts[choice] -= 1
    return order


def build_stimulus(order: List[str], cfg: DoubleToneConfig,
                   rng: np.random.Generator) -> Dict:
    """Learning phase (all AB) followed by the test phase."""
    phases: List[str] = []
    words: List[str] = []
    onsets: List[int] = []

    # Lead-in so the first pair still has a pre-stimulus baseline to cut.
    t = cfg.pre_stim_ms
    for _ in range(cfg.n_learn):
        onsets.append(t)
        words.append(FREQUENT)
        phases.append("learn")
        t += cfg.isi_learn

    # Let the network settle before the test phase begins.
    t += cfg.iti_test_min
    for word in order:
        onsets.append(t)
        words.append(word)
        phases.append("test")
        t += int(rng.integers(cfg.iti_test_min, cfg.iti_test_max + 1))

    total = t + cfg.pair_span + cfg.post_stim_ms
    stim = np.zeros((2, total), dtype=float)
    for start, word in zip(onsets, words):
        first, second = PAIRS[word]
        for channel, offset in ((first, 0), (second, cfg.soa_within)):
            a = start + offset
            stim[channel, a:a + cfg.tone_dur] = cfg.tone_amp

    return dict(stim=stim, pair_onsets=np.asarray(onsets, dtype=int),
                pair_word=words, pair_phase=phases)


# =====================================================================
#  Simulation
# =====================================================================
def run_experiment(cfg: DoubleToneConfig, a1_cfg: A1Config,
                   learn: bool = True, stim_pack: Dict | None = None) -> Dict:
    if stim_pack is None:
        rng = np.random.default_rng(cfg.seed)
        stim_pack = build_stimulus(generate_pair_order(cfg, rng), cfg, rng)

    snap = max(1, int(round(0.25 / a1_cfg.dt)))       # every 250 ms
    W_init = np.zeros((a1_cfg.N, a1_cfg.N)) if not learn else None
    out = simulate(stim_pack["stim"], cfg=a1_cfg, W_init=W_init, learn=learn,
                   record_W_every=snap, seed=cfg.seed)
    out.update(stim_pack)
    out["dt_cfg"] = cfg
    out["learn"] = learn
    return out


# =====================================================================
#  Analysis
# =====================================================================
def _epochs(res: Dict) -> np.ndarray:
    """(n_pairs, N, epoch_steps) excitatory epochs, pre-stimulus padded."""
    cfg: DoubleToneConfig = res["dt_cfg"]
    E = res["E"]
    out = np.zeros((len(res["pair_onsets"]), E.shape[0], cfg.epoch_steps))
    for k, start in enumerate(res["pair_onsets"]):
        a = start - cfg.pre_stim_ms
        out[k] = E[:, a:a + cfg.epoch_steps]
    return out


def _epochs_of(res: Dict, key: str) -> np.ndarray:
    cfg: DoubleToneConfig = res["dt_cfg"]
    arr = res[key]
    out = np.zeros((len(res["pair_onsets"]), arr.shape[0], cfg.epoch_steps))
    for k, start in enumerate(res["pair_onsets"]):
        a = start - cfg.pre_stim_ms
        out[k] = arr[:, a:a + cfg.epoch_steps]
    return out


def analysed_pairs(res: Dict) -> np.ndarray:
    """Boolean mask of test pairs that survive the paper's exclusions."""
    cfg: DoubleToneConfig = res["dt_cfg"]
    words = res["pair_word"]
    keep = np.array([p == "test" for p in res["pair_phase"]])
    if cfg.drop_frequent_after_rare:
        for k in range(1, len(words)):
            if words[k] == FREQUENT and words[k - 1] in RARE:
                keep[k] = False
    return keep


def responses(res: Dict) -> Dict[str, np.ndarray]:
    """Per-condition [tone1, tone2] population response, mean and sem.

    The response is the peak of the summed excitatory rate in the window
    after each tone onset -- the model analogue of the paper's peak-window
    magnetometer amplitude.
    """
    cfg: DoubleToneConfig = res["dt_cfg"]
    ep = _epochs(res).sum(axis=1)                  # (n_pairs, epoch) summed E
    keep = analysed_pairs(res)
    words = np.asarray(res["pair_word"])
    o1, o2 = cfg.tone_onsets_in_epoch
    w = cfg.response_window

    out: Dict[str, np.ndarray] = {}
    for cond in COND_ORDER:
        sel = keep & (words == cond)
        # Evoked deflection: peak in the window minus the level the network
        # was already sitting at when the tone arrived.  Without this the
        # standing prediction -- which is present BEFORE the second tone --
        # is counted as part of the response to it.
        peaks = np.stack(
            [ep[sel, o1:o1 + w].max(axis=1) - ep[sel, o1],
             ep[sel, o2:o2 + w].max(axis=1) - ep[sel, o2]], axis=1)
        out[cond] = peaks
        out[cond + "_mean"] = peaks.mean(axis=0)
        out[cond + "_sem"] = peaks.std(axis=0, ddof=1) / np.sqrt(len(peaks))
        out[cond + "_n"] = np.array([len(peaks)])
    return out


def normalised(resp: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalise to the mean first-tone response, as the paper does."""
    ref = np.mean([resp[c + "_mean"][0] for c in COND_ORDER])
    return {c: resp[c + "_mean"] / ref for c in COND_ORDER} | {
        c + "_sem": resp[c + "_sem"] / ref for c in COND_ORDER}


# =====================================================================
#  Plots
# =====================================================================
def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_responses(plastic: Dict, frozen: Dict, fname: str, inh: str):
    """The paper's Figure 8B layout: habituation vs predictive coding."""
    rp, rf = responses(plastic), responses(frozen)
    np_, nf = normalised(rp), normalised(rf)

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.4), sharex=True,
                             sharey=True, constrained_layout=True)
    fig.suptitle("Double-tone paradigm (Wacongne et al. 2012), model0 "
                 f"[{inh} inhibition]\n"
                 "second tone of rare AA must beat its own depression",
                 fontsize=12, fontweight="bold")

    for row, cond in enumerate(COND_ORDER):
        for col, (data, title) in enumerate(
                ((nf, "Habituation model\n(depression only, W = 0)"),
                 (np_, "Predictive coding model\n(plastic)"))):
            ax = axes[row, col]
            ax.bar([0, 1], data[cond], yerr=data[cond + "_sem"],
                   color=COND_COLORS[cond], width=0.55,
                   error_kw=dict(lw=1.0, capsize=3))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["tone 1", "tone 2"])
            _clean(ax)
            if col == 0:
                ax.set_ylabel(COND_LABEL[cond], fontsize=11,
                              fontweight="bold", color=COND_COLORS[cond])
            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
    axes[0, 0].set_ylim(0, None)
    fig.supylabel("Population response (normalised to mean tone 1)",
                  fontsize=10)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_traces(res: Dict, fname: str, inh: str):
    """One representative late-session epoch per condition."""
    cfg: DoubleToneConfig = res["dt_cfg"]
    E = _epochs(res)
    inh_E = _epochs_of(res, "inh_to_E")
    keep = analysed_pairs(res)
    words = np.asarray(res["pair_word"])
    t = np.arange(cfg.epoch_steps) - cfg.pre_stim_ms
    o1, o2 = cfg.tone_onsets_in_epoch

    fig, axes = plt.subplots(4, 1, figsize=(7.6, 9.0), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"Double-tone: last analysed epoch per condition "
                 f"[{inh} inhibition]", fontsize=12, fontweight="bold")

    for ax, cond in zip(axes, COND_ORDER):
        idx = np.flatnonzero(keep & (words == cond))
        k = idx[-1]
        first, second = PAIRS[cond]
        for ch, name, style in ((TONE_A, "A", "-"), (TONE_B, "B", "--")):
            ax.plot(t, E[k, ch], style, lw=1.6,
                    color="#22409A" if ch == TONE_A else "#E8442A",
                    label=f"E[{name}]")
            ax.plot(t, -inh_E[k, ch], style, lw=1.0, alpha=0.55,
                    color="#666666",
                    label=f"inhibition onto {name}" if ch == TONE_A else None)
        for onset, tone in ((o1 - cfg.pre_stim_ms, cond[0]),
                            (o2 - cfg.pre_stim_ms, cond[1])):
            ax.axvspan(onset, onset + cfg.tone_dur, color="0.85", zorder=0)
            ax.text(onset + cfg.tone_dur / 2, ax.get_ylim()[1],
                    tone, ha="center", va="top", fontsize=11,
                    fontweight="bold")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel(COND_LABEL[cond], fontsize=10, fontweight="bold",
                      color=COND_COLORS[cond])
        _clean(ax)
    axes[0].legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel("time from first tone onset (ms)")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_weights(res: Dict, fname: str, inh: str):
    cfg: DoubleToneConfig = res["dt_cfg"]
    W = np.stack(res["W_traj"])
    t = np.arange(len(W)) * 0.25
    fig, ax = plt.subplots(figsize=(7.6, 3.2), constrained_layout=True)
    ax.plot(t, W[:, TONE_B, TONE_A], lw=1.8, color="#22409A",
            label=r"W[B$\leftarrow$A]  (the prediction)")
    ax.plot(t, W[:, TONE_A, TONE_B], lw=1.4, color="#E8442A",
            label=r"W[A$\leftarrow$B]")
    ax.plot(t, W[:, TONE_A, TONE_A], lw=1.0, color="0.55", ls="--",
            label=r"W[A$\leftarrow$A]")
    learn_end = cfg.n_learn * cfg.isi_learn / 1000.0
    ax.axvspan(0, learn_end, color="#FFE9A8", zorder=0)
    ax.text(learn_end / 2, ax.get_ylim()[1], "learning",
            ha="center", va="top", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("recurrent weight")
    ax.set_title(f"Association across the session [{inh} inhibition]",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    _clean(ax)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# =====================================================================
#  Entry point
# =====================================================================
def report(plastic: Dict, frozen: Dict) -> None:
    rp, rf = responses(plastic), responses(frozen)
    np_, nf = normalised(rp), normalised(rf)
    print(f"    {'cond':>8} {'n':>4} | {'habituation':>21} | "
          f"{'predictive':>21}")
    print(f"    {'':>8} {'':>4} | {'tone1':>9} {'tone2':>10} | "
          f"{'tone1':>9} {'tone2':>10}")
    for cond in COND_ORDER:
        n = int(rp[cond + "_n"][0])
        print(f"    {COND_LABEL[cond]:>8} {n:>4} | "
              f"{nf[cond][0]:>9.3f} {nf[cond][1]:>10.3f} | "
              f"{np_[cond][0]:>9.3f} {np_[cond][1]:>10.3f}")
    aa = np_["AA"][1] / np_["AB"][1]
    aa_hab = nf["AA"][1] / nf["AB"][1]
    print(f"    tone-2 ratio  rare AA / freq AB:  "
          f"habituation {aa_hab:.2f}x   predictive {aa:.2f}x")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default")
    p.add_argument("--presets", nargs="+", default=["selective", "uniform"])
    args = p.parse_args(argv)

    cfg = get_preset(args.preset)
    builders = {"selective": model_config, "uniform": uniform_model_config,
                "global": global_model_config}

    for inh in args.presets:
        a1_cfg = builders[inh]()
        rng = np.random.default_rng(cfg.seed)
        stim_pack = build_stimulus(generate_pair_order(cfg, rng), cfg, rng)

        print(f"\n========================================================")
        print(f"[ Double tone -- '{inh}' inhibition ]")
        print(f"  {cfg.n_learn} learning pairs, {cfg.n_test} test pairs "
              f"({cfg.n_each}); ITI {cfg.iti_test_min/1000:.0f}-"
              f"{cfg.iti_test_max/1000:.0f} s")
        print(f"  tau_trace={a1_cfg.tau_trace*1e3:.0f} ms  "
              f"tau_I={a1_cfg.tau_I*1e3:.0f} ms  W_max={a1_cfg.W_max}  "
              f"W_norm={a1_cfg.W_norm}  W_decay={a1_cfg.W_decay:.0e}")
        print(f"  total simulated {stim_pack['stim'].shape[1]/1000:.0f} s")

        plastic = run_experiment(cfg, a1_cfg, learn=True,
                                 stim_pack=stim_pack)
        frozen = run_experiment(cfg, a1_cfg, learn=False,
                                stim_pack=stim_pack)

        W = plastic["W_final"]
        print(f"  W[B<-A] final = {W[TONE_B, TONE_A]:.3f} "
              f"(W_max {a1_cfg.W_max})")
        report(plastic, frozen)

        plot_responses(plastic, frozen, _out(f"double_tone_responses_{inh}.png"), inh)
        plot_traces(plastic, _out(f"double_tone_traces_{inh}.png"), inh)
        plot_weights(plastic, _out(f"double_tone_weights_{inh}.png"), inh)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
