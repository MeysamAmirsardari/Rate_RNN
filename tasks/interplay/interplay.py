"""
tasks.interplay.interplay
=========================

Saffran statistical word learning inside a competing background: does
temporal predictability act as a grouping cue the way simultaneous
coherence does in figure-ground?

    python -m tasks.interplay.interplay [--preset short] [--seeds 3]

The design is a 2 x 2 -- structured / scrambled crossed with background
on / off -- because the hypothesis is an interaction, not a main effect.
Structure should buy more when there is something to segregate from.

Outputs, written next to this file rather than into the working directory
    interplay_stimulus.png     the stimulus, with both background
                               constraints shown rather than asserted
    interplay_measures.png     the 2 x 2 on every measure
    interplay_layer1.png       figure-ground-style layer-1 panels
    interplay_robustness.png   performance against background level
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model0 import A1Config, simulate

if __package__:
    from .config import (WORDS, WORD_NAMES, InterplayConfig, get_preset,
                         model_config, frozen_model_config,
                         uniform_model_config)
else:
    from tasks.interplay.config import (  # type: ignore
        WORDS, WORD_NAMES, InterplayConfig, get_preset, model_config,
        frozen_model_config, uniform_model_config)


#: Figures are written next to the code that makes them, not into whatever
#: directory the run happened to start in.
OUT_DIR = Path(__file__).resolve().parent


def _out(name: str) -> str:
    return str(OUT_DIR / name)


C_FIG, C_BG = "#2C6E5A", "#BD6B6B"
C_STRUCT, C_SCRAM = "#2166AC", "#999999"


# =====================================================================
#  Stimulus
# =====================================================================
def word_order(cfg: InterplayConfig, rng) -> np.ndarray:
    """Random word sequence; a word never immediately follows itself."""
    order = np.empty(cfg.n_words, dtype=int)
    prev = -1
    for k in range(cfg.n_words):
        choices = [w for w in range(len(WORDS))
                   if not (cfg.allow_word_repeat is False and w == prev)]
        prev = order[k] = int(rng.choice(choices))
    return order


def token_sequence(cfg: InterplayConfig, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Token channels in presentation order, plus the word index of each.

    ``scrambled`` permutes the token sequence.  That holds every token's
    frequency EXACTLY fixed -- it is the same multiset -- and destroys
    only the order, so spectrum and total drive are unchanged and the
    single thing removed is predictability.
    """
    order = word_order(cfg, rng)
    tokens = np.concatenate([np.asarray(WORDS[w]) for w in order])
    wid = np.repeat(order, 3)
    if cfg.structure == "scrambled":
        perm = rng.permutation(len(tokens))
        tokens, wid = tokens[perm], wid[perm]
    elif cfg.structure != "structured":
        raise ValueError(f"unknown structure {cfg.structure!r}")
    return tokens, wid


def build_background(cfg: InterplayConfig, T: int, rng) -> np.ndarray:
    """Background channel active at every sample; exactly one, always.

    Channel order is repeated random permutations of the background pool,
    which makes the per-channel totals equal by construction rather than
    in expectation.
    """
    chans = np.asarray(list(cfg.background_channels))
    n_tiles = int(np.ceil(T / cfg.bg_dur)) + 1
    seq: List[int] = []
    prev = -1
    while len(seq) < n_tiles:
        block = rng.permutation(chans)
        if cfg.bg_no_immediate_repeat and block[0] == prev:
            block = np.roll(block, 1)
        seq.extend(block.tolist())
        prev = seq[-1]
    who = np.repeat(np.asarray(seq[:n_tiles]), cfg.bg_dur)[:T]
    return who


def build_stimulus(cfg: InterplayConfig, rng) -> Dict:
    tokens, wid = token_sequence(cfg, rng)
    T = len(tokens) * cfg.slot
    stim = np.zeros((cfg.n_channels, T))

    onsets = np.arange(len(tokens)) * cfg.slot
    for onset, ch in zip(onsets, tokens):
        stim[ch, onset:onset + cfg.tone_dur] = cfg.fig_amp

    bg_who = None
    if cfg.background:
        bg_who = build_background(cfg, T, rng)
        stim[bg_who, np.arange(T)] = cfg.bg_amp

        # -- the two constraints, checked rather than trusted --
        on = stim[cfg.n_figure:, :] > 0
        per_sample = on.sum(axis=0)
        assert per_sample.min() == 1 and per_sample.max() == 1, (
            "background must have exactly one tone on at every sample; "
            f"got min {per_sample.min()}, max {per_sample.max()}")
        totals = on.sum(axis=1)
        assert np.ptp(totals) <= cfg.bg_dur, (
            f"background channels unbalanced: totals span {np.ptp(totals)} "
            f"samples (> one tone of {cfg.bg_dur})")

    return dict(stim=stim, tokens=tokens, word_id=wid, onsets=onsets,
                bg_who=bg_who, T=T, cfg=cfg)


def drive_per_pool(pack: Dict) -> Tuple[float, float]:
    """Total thalamic drive delivered to each pool, summed over the run."""
    cfg: InterplayConfig = pack["cfg"]
    stim = pack["stim"]
    return (float(stim[:cfg.n_figure].sum()),
            float(stim[cfg.n_figure:].sum()))


# =====================================================================
#  Transition bookkeeping
# =====================================================================
def within_transitions() -> List[Tuple[int, int]]:
    return [(w[i], w[i + 1]) for w in WORDS for i in range(len(w) - 1)]


def boundary_transitions() -> List[Tuple[int, int]]:
    return [(a[-1], b[0]) for a in WORDS for b in WORDS if a is not b]


def part_words() -> List[Tuple[int, ...]]:
    """Three-token sequences that straddle a boundary and did occur."""
    out = []
    for a in WORDS:
        for b in WORDS:
            if a is b:
                continue
            out.append((a[-1], b[0], b[1]))       # 1 + 2
            out.append((a[-2], a[-1], b[0]))      # 2 + 1
    return out


# =====================================================================
#  Simulation
# =====================================================================
def run(cfg: InterplayConfig, a1_cfg: A1Config, rng=None,
        snap_ms: int = 0) -> Dict:
    rng = rng or np.random.default_rng(cfg.seed)
    pack = build_stimulus(cfg, rng)
    snap = int(round(snap_ms / 1000.0 / a1_cfg.dt)) if snap_ms else 0
    out = simulate(pack["stim"], cfg=a1_cfg, learn=True,
                   record_W_every=snap, seed=cfg.seed)
    out.update(pack)
    out["a1_cfg"] = a1_cfg
    return out


def plastic_and_frozen(cfg: InterplayConfig, snap_ms: int = 0
                       ) -> Tuple[Dict, Dict]:
    """The same stimulus with learning able to act, and unable to.

    ``frozen`` keeps adaptation and inhibition and removes only the
    recurrent gain, so every difference between the two runs is what the
    learned transitions did.  Seeds match, so the stimulus is identical
    sample for sample.
    """
    return (run(cfg, model_config(cfg.n_channels), snap_ms=snap_ms),
            run(cfg, frozen_model_config(cfg.n_channels), snap_ms=snap_ms))


# =====================================================================
#  SFG-style layer-1 readouts
# =====================================================================
CURRENTS = ("tm_in", "rec_E", "inh_to_E", "net")


def current_decomposition(res_p: Dict, res_f: Dict) -> Dict[str, float]:
    """Plastic-minus-frozen change in each current, per pool.

    The same decomposition the figure-ground analysis uses.  Thalamic
    drive must come out at exactly zero: the multiscale depression
    variable is driven by the stimulus alone, so it cannot differ between
    two runs on the same stimulus.  A non-zero value there means the
    comparison is not seed-matched and nothing else in the panel is
    trustworthy.
    """
    cfg: InterplayConfig = res_p["cfg"]
    nf = cfg.n_figure
    out: Dict[str, float] = {}
    for name in CURRENTS:
        if name == "net":
            a = res_p["tm_in"] + res_p["rec_E"] - res_p["inh_to_E"]
            b = res_f["tm_in"] + res_f["rec_E"] - res_f["inh_to_E"]
        else:
            a, b = res_p[name], res_f[name]
        out[f"{name}_fig"] = float(a[:nf].mean() - b[:nf].mean())
        out[f"{name}_bg"] = float(a[nf:].mean() - b[nf:].mean())
    return out


def position_modulation(res_p: Dict, res_f: Dict) -> Dict[str, np.ndarray]:
    """Response modulation by a token's position inside its word.

    This is the temporal counterpart of the figure-versus-ground contrast.
    Token 1 has no predecessor inside its word, so nothing predicts it and
    it should show no enhancement.  Tokens 2 and 3 are predicted by the
    token before them, so if the learned transitions are what enhance the
    stream, the enhancement should appear only there.
    """
    cfg: InterplayConfig = res_p["cfg"]
    tokens, onsets = res_p["tokens"], res_p["onsets"]
    pos = np.arange(len(tokens)) % 3          # position inside the word
    Ep, Ef = res_p["E"], res_f["E"]

    mod = [[] for _ in range(3)]
    for k, (ch, o) in enumerate(zip(tokens, onsets)):
        a, b = o, o + cfg.tone_dur
        p, f = Ep[ch, a:b].mean(), Ef[ch, a:b].mean()
        if f > 1e-9:
            mod[pos[k]].append(100.0 * (p - f) / f)
    means = np.array([np.mean(m) if m else np.nan for m in mod])
    sems = np.array([np.std(m, ddof=1) / np.sqrt(len(m)) if len(m) > 1
                     else np.nan for m in mod])

    # Background pool over the same run, as the flat reference.
    nf = cfg.n_figure
    bp, bf = Ep[nf:].mean(), Ef[nf:].mean()
    bg = 100.0 * (bp - bf) / bf if bf > 1e-9 else np.nan
    return dict(mean=means, sem=sems, background=np.array([bg]))


def weight_groups(res: Dict) -> Dict[str, np.ndarray]:
    """Weight trajectories split into the four groups that matter."""
    cfg: InterplayConfig = res["cfg"]
    nf = cfg.n_figure
    traj = np.stack(res["W_traj"]) if len(res["W_traj"]) else None
    if traj is None:
        return {}
    win = within_transitions()
    bnd = boundary_transitions()
    gg = [(i, j) for i in cfg.background_channels
          for j in cfg.background_channels if i != j]
    cross = [(i, j) for i in range(nf) for j in cfg.background_channels]
    pick = lambda pairs: traj[:, [j for _, j in pairs], [i for i, _ in pairs]
                              ].mean(axis=1)
    return dict(t=res["W_t"], within=pick(win), boundary=pick(bnd),
                ground=pick(gg), cross=pick(cross))


def probe(items: Sequence[Sequence[int]], cfg: InterplayConfig,
          a1_cfg: A1Config, W: np.ndarray) -> np.ndarray:
    """Present isolated items with learning off; return each item's response.

    The weights learned during exposure are frozen, so this measures what
    the learned transitions do and not what the probe itself teaches.
    """
    # Each item is simulated from a FRESH state.  Presenting them in one
    # continuous run does not work: the 24 part-words reuse channels among
    # themselves while the 4 words are built from disjoint ones, so later
    # part-words land on more-adapted channels.  That artefact alone gives
    # AUC = 0.96 with W set to zero, i.e. with nothing learned at all.
    pad = 300
    out = []
    for item in items:
        L = len(item) * cfg.slot
        stim = np.zeros((cfg.n_channels, L + pad))
        for j, ch in enumerate(item):
            o = j * cfg.slot
            stim[ch, o:o + cfg.tone_dur] = cfg.fig_amp
        res = simulate(stim, cfg=a1_cfg, W_init=W.copy(), learn=False, seed=0)
        out.append(res["E"][:cfg.n_figure, :L].sum())
    return np.asarray(out)


# =====================================================================
#  Measures
# =====================================================================
def auc(x0: np.ndarray, x1: np.ndarray) -> float:
    """P(a draw from x1 exceeds a draw from x0)."""
    if len(x0) == 0 or len(x1) == 0:
        return 0.5
    allv = np.concatenate([x0, x1])
    r = np.argsort(np.argsort(allv)) + 1.0
    u1 = r[len(x0):].sum() - len(x1) * (len(x1) + 1) / 2.0
    return float(u1 / (len(x0) * len(x1)))


def pool_rates(res: Dict) -> Tuple[np.ndarray, np.ndarray]:
    cfg: InterplayConfig = res["cfg"]
    E = res["E"]
    return E[:cfg.n_figure].mean(axis=0), E[cfg.n_figure:].mean(axis=0)


def segregation(res: Dict) -> Dict[str, float]:
    """Figure-over-background contrast and its d'."""
    fig, bg = pool_rates(res)
    if not res["cfg"].background:
        return dict(fig=float(fig.mean()), bg=float("nan"),
                    index=float("nan"), dprime=float("nan"))
    f, b = float(fig.mean()), float(bg.mean())
    sd = np.sqrt((fig.var(ddof=1) + bg.var(ddof=1)) / 2.0)
    return dict(fig=f, bg=b, index=(f - b) / (f + b),
                dprime=float((f - b) / sd) if sd > 0 else float("nan"))


def weight_structure(res: Dict) -> Dict[str, float]:
    """SFG-style weight readout, split by pool and by within/boundary."""
    cfg: InterplayConfig = res["cfg"]
    W = res["W_final"]
    nf = cfg.n_figure
    off = ~np.eye(cfg.n_channels, dtype=bool)
    ff = W[:nf, :nf][off[:nf, :nf]]
    gg = W[nf:, nf:][off[nf:, nf:]]
    fg = np.concatenate([W[:nf, nf:].ravel(), W[nf:, :nf].ravel()])

    win = np.array([W[j, i] for i, j in within_transitions()])
    bnd = np.array([W[j, i] for i, j in boundary_transitions()])
    return dict(W_FF=float(ff.mean()), W_GG=float(gg.mean()),
                W_FG=float(fg.mean()),
                FF_over_GG=float(ff.mean() / gg.mean())
                if gg.mean() > 1e-6 else float("nan"),
                W_within=float(win.mean()), W_boundary=float(bnd.mean()),
                within_over_boundary=float(win.mean() / (bnd.mean() + 1e-12)))


def transition_accuracy(res: Dict) -> float:
    """Fraction of the top-8 learned transitions that are within-word.

    Twenty transitions occur in the stream, eight of them within a word,
    so chance is 8/20 = 0.40.
    """
    W = res["W_final"]
    win = set(within_transitions())
    allt = list(win) + boundary_transitions()
    vals = np.array([W[j, i] for i, j in allt])
    top = np.argsort(vals)[::-1][:len(win)]
    return float(np.mean([allt[k] in win for k in top]))


def word_vs_partword(res: Dict) -> Dict[str, float]:
    """Do whole words evoke a bigger response than part-words?

    Under A_rec > 0 a learned transition EXCITES its target, so a word
    (two within-word transitions) should out-drive a part-word (at most
    one).  AUC > 0.5 is the prediction.
    """
    cfg: InterplayConfig = res["cfg"]
    a1 = res["a1_cfg"]
    W = res["W_final"]
    pw = part_words()
    w_resp = probe(list(WORDS), cfg, a1, W)
    p_resp = probe(pw, cfg, a1, W)
    # The chance level for this AUC is NOT 0.5.  Words and part-words differ
    # in how much within-item adaptation they carry even with nothing
    # learned, so the floor is whatever the same probe gives at W = 0.  It
    # is measured rather than assumed, and reported next to the result.
    zero = np.zeros_like(W)
    floor = auc(probe(pw, cfg, a1, zero), probe(list(WORDS), cfg, a1, zero))
    return dict(auc=auc(p_resp, w_resp), auc_floor=floor,
                word=float(w_resp.mean()), part=float(p_resp.mean()),
                ratio=float(w_resp.mean() / (p_resp.mean() + 1e-12)))


def buildup(res: Dict) -> Dict[str, float]:
    """Segregation early vs late in the exposure."""
    cfg: InterplayConfig = res["cfg"]
    if not cfg.background:
        return dict(early=float("nan"), late=float("nan"),
                    gain=float("nan"))
    fig, bg = pool_rates(res)
    n = len(fig)
    k = max(1, int(cfg.early_frac * n))

    def idx(a, b):
        f, g = fig[a:b].mean(), bg[a:b].mean()
        return float((f - g) / (f + g))

    e, l = idx(0, k), idx(n - k, n)
    return dict(early=e, late=l, gain=l - e)


def measure_all(res: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update({f"seg_{k}": v for k, v in segregation(res).items()})
    out.update(weight_structure(res))
    out["trans_acc"] = transition_accuracy(res)
    out.update({f"wp_{k}": v for k, v in word_vs_partword(res).items()})
    out.update({f"build_{k}": v for k, v in buildup(res).items()})
    d_fig, d_bg = drive_per_pool(res)
    out["drive_fig"], out["drive_bg"] = d_fig, d_bg
    out["drive_ratio"] = d_bg / d_fig if d_fig else float("nan")
    return out


# =====================================================================
#  Plots
# =====================================================================
def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_stimulus(cfg: InterplayConfig, fname: str, n_words_show: int = 8):
    rng = np.random.default_rng(cfg.seed)
    pack = build_stimulus(cfg.replace(n_words=max(n_words_show, 12)), rng)
    show = n_words_show * 3 * cfg.slot
    stim = pack["stim"][:, :show]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2),
                             gridspec_kw=dict(height_ratios=(3, 1, 1)),
                             constrained_layout=True)
    fig.suptitle("Interplay stimulus: a Saffran stream inside a "
                 "one-tone-at-a-time background", fontsize=13,
                 fontweight="bold")

    ax = axes[0]
    ax.imshow(stim, aspect="auto", origin="lower", cmap="magma",
              interpolation="nearest", extent=[0, show, -0.5,
                                               cfg.n_channels - 0.5])
    ax.axhline(cfg.n_figure - 0.5, color="w", lw=1.4)
    for k in range(0, n_words_show * 3, 3):
        ax.axvline(k * cfg.slot, color="#7FD4C1", lw=0.8, ls=":")
    ax.text(show * 0.005, cfg.n_figure - 1.4, "FIGURE  (4 words x 3 tokens)",
            color=C_FIG, fontsize=9, fontweight="bold", va="top")
    ax.text(show * 0.005, cfg.n_channels - 0.8, "BACKGROUND",
            color="#F2C0C0", fontsize=9, fontweight="bold", va="top")
    ax.set_ylabel("channel")
    _clean(ax)

    ax = axes[1]
    on = pack["stim"][cfg.n_figure:, :] > 0
    ax.plot(on.sum(axis=0)[:show], color=C_BG, lw=1.2)
    ax.set_ylim(0, 2.5)
    ax.set_ylabel("bg tones\non")
    ax.set_title("exactly one background tone at every sample "
                 f"(min {on.sum(axis=0).min()}, max {on.sum(axis=0).max()})",
                 fontsize=9)
    _clean(ax)

    ax = axes[2]
    totals = on.sum(axis=1) / on.shape[1] * 100
    ax.bar(range(len(totals)), totals, color=C_BG, width=0.7)
    ax.axhline(100 / cfg.n_background, color="0.35", ls="--", lw=1.0)
    ax.set_ylabel("% of time")
    ax.set_xlabel("background channel")
    ax.set_title("every background channel on for the same total time",
                 fontsize=9)
    _clean(ax)

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_measures(table: Dict[str, List[Dict[str, float]]], fname: str):
    panels = [
        ("seg_index", "Segregation index\n(fig - bg)/(fig + bg)", True),
        ("within_over_boundary", "Learned within / boundary\ntransition weight", False),
        ("trans_acc", "Transition accuracy\n(chance = 0.40)", False),
        ("wp_auc", "Word vs part-word AUC\n(chance = 0.50)", False),
        ("FF_over_GG", "W_FF / W_GG\n(figure vs background)", True),
        ("build_gain", "Segregation buildup\n(late - early)", True),
    ]
    labels = list(table.keys())
    colors = [C_STRUCT if "structured" in l else C_SCRAM for l in labels]
    hatch = ["" if "/bg" in l else "//" for l in labels]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    fig.suptitle("Interplay: does temporal predictability segregate a stream "
                 "from a background?", fontsize=13, fontweight="bold")

    for ax, (key, title, bg_only) in zip(axes.ravel(), panels):
        means, sems, xs, cs, hs = [], [], [], [], []
        for i, lab in enumerate(labels):
            vals = np.array([r[key] for r in table[lab]], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            xs.append(i)
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals))
                        if len(vals) > 1 else 0.0)
            cs.append(colors[i]); hs.append(hatch[i])
        bars = ax.bar(xs, means, yerr=sems, color=cs, width=0.66,
                      error_kw=dict(lw=1.0, capsize=3))
        for b, h in zip(bars, hs):
            b.set_hatch(h)
        for chance, key_ in ((0.40, "trans_acc"), (0.50, "wp_auc"),
                             (1.0, "within_over_boundary"),
                             (1.0, "FF_over_GG")):
            if key == key_:
                ax.axhline(chance, color="0.3", ls="--", lw=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("/", "\n") for l in labels],
                           fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        _clean(ax)

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_layer1(res_p: Dict, res_f: Dict, fname: str):
    """Figure-ground-style layer-1 panels for the temporal case."""
    cfg: InterplayConfig = res_p["cfg"]
    nf = cfg.n_figure
    W = res_p["W_final"]
    grp = weight_groups(res_p)
    dec = current_decomposition(res_p, res_f)
    pos = position_modulation(res_p, res_f)

    fig = plt.figure(figsize=(13.2, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    fig.suptitle("Interplay, layer 1: the enhancement is recurrent, and it "
                 "falls on the tokens that were predicted",
                 fontsize=13, fontweight="bold")

    # -- A: learned weight matrix --------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(W, cmap="Purples", interpolation="nearest")
    ax.axhline(nf - 0.5, color=C_FIG, lw=1.2)
    ax.axvline(nf - 0.5, color=C_FIG, lw=1.2)
    ax.set_xlabel("presynaptic"); ax.set_ylabel("postsynaptic")
    ax.set_title("Learned weights\n(figure block top-left)", fontsize=10,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # -- B: weight growth over the session -----------------------------
    ax = fig.add_subplot(gs[0, 1])
    if grp:
        for key, lab, col, ls in (
                ("within", "within-word", C_FIG, "-"),
                ("boundary", "boundary", "#7FB3A6", "-"),
                ("ground", "background", C_BG, "-"),
                ("cross", "cross", "0.6", "--")):
            ax.plot(grp["t"], grp[key], ls, color=col, lw=1.6, label=lab)
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("session time (s)"); ax.set_ylabel("mean weight")
    ax.set_title("Growth over the session", fontsize=10, fontweight="bold")
    _clean(ax)

    # -- C: modulation by position inside the word ---------------------
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(3)
    ax.bar(x, pos["mean"], yerr=pos["sem"], color=C_FIG, width=0.6,
           error_kw=dict(lw=1.0, capsize=3))
    ax.axhline(float(pos["background"][0]), color=C_BG, ls="--", lw=1.4,
               label="background pool")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["token 1\n(unpredicted)",
                                          "token 2", "token 3"], fontsize=8)
    ax.set_ylabel("response modulation (%)")
    ax.set_title("Enhancement falls on predicted tokens", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)

    # -- D: current decomposition --------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    names = ["Thalamic", "Recurrent", "Inhibition", "Net drive"]
    xs = np.arange(len(names)); w = 0.36
    figv = [dec[f"{c}_fig"] for c in CURRENTS]
    bgv = [dec[f"{c}_bg"] for c in CURRENTS]
    ax.bar(xs - w / 2, figv, w, color=C_FIG, label="figure")
    ax.bar(xs + w / 2, bgv, w, color=C_BG, label="background")
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("change from frozen control (a.u.)")
    ax.set_title(f"Where the enhancement comes from "
                 f"(thalamic = {figv[0]:+.2e}, zero by construction)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    _clean(ax)

    # -- E: pool rates, plastic vs frozen ------------------------------
    ax = fig.add_subplot(gs[1, 2])
    labels = ["figure", "background"]
    pv = [res_p["E"][:nf].mean(), res_p["E"][nf:].mean()]
    fv = [res_f["E"][:nf].mean(), res_f["E"][nf:].mean()]
    xs = np.arange(2)
    ax.bar(xs - w / 2, fv, w, color="0.72", label="frozen")
    ax.bar(xs + w / 2, pv, w, color=C_FIG, label="plastic")
    for i, (a, b) in enumerate(zip(fv, pv)):
        ax.text(i, max(a, b) * 1.02, f"{100*(b-a)/a:+.1f}%", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("mean E rate (a.u.)")
    ax.set_title("Pool rates", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    _clean(ax)

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_robustness(levels: Sequence[float],
                    curves: Dict[str, Dict[str, List[float]]], fname: str):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9),
                             constrained_layout=True)
    fig.suptitle("Robustness to background level "
                 "(1.0 = drive-matched to the figure)", fontsize=12,
                 fontweight="bold")
    for ax, (key, title, chance) in zip(axes, (
            ("seg_index", "Segregation index", None),
            ("wp_auc", "Word vs part-word AUC", 0.5),
            ("trans_acc", "Transition accuracy", 0.4))):
        for lab, series in curves.items():
            ax.plot(levels, series[key], "o-", lw=1.6, ms=4.5,
                    color=C_STRUCT if "structured" in lab else C_SCRAM,
                    label=lab)
        if chance is not None:
            ax.axhline(chance, color="0.35", ls="--", lw=1.0)
        ax.set_xlabel("background level")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xscale("log")
        _clean(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


# =====================================================================
#  Entry point
# =====================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="short")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--inh", default="selective",
                   choices=("selective", "uniform"))
    p.add_argument("--skip-robustness", action="store_true")
    args = p.parse_args(argv)

    base = get_preset(args.preset)
    builder = {"selective": model_config,
               "uniform": uniform_model_config}[args.inh]

    print(f"[ Interplay -- {base.name}, {args.inh} inhibition, "
          f"{args.seeds} seeds ]")
    print(f"  {base.n_figure} figure + {base.n_background} background "
          f"channels; token {base.tone_dur} ms, slot {base.slot} ms, "
          f"tau_trace {model_config(base.n_channels).tau_trace*1e3:.0f} ms")
    print(f"  {base.n_words} words = {base.n_tokens} tokens "
          f"= {base.n_tokens * base.slot / 1000:.0f} s per run")

    plot_stimulus(base, _out("interplay_stimulus.png"))

    table: Dict[str, List[Dict[str, float]]] = {}
    for structure in ("structured", "scrambled"):
        for bg in (True, False):
            label = f"{structure}/{'bg' if bg else 'clean'}"
            rows = []
            for s in range(args.seeds):
                cfg = base.replace(structure=structure, background=bg,
                                   seed=base.seed + s)
                a1 = builder(cfg.n_channels)
                rows.append(measure_all(run(cfg, a1)))
            table[label] = rows
            m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
            print(f"\n  [{label}]")
            print(f"    drive fig/bg           {m['drive_fig']:.3g} / "
                  f"{m['drive_bg']:.3g}   (ratio {m['drive_ratio']:.3f})")
            print(f"    segregation index      {m['seg_index']:+.4f}"
                  f"   d' {m['seg_dprime']:+.3f}")
            print(f"    W within / boundary    {m['within_over_boundary']:.3f}"
                  f"   (within {m['W_within']:.4f}, "
                  f"boundary {m['W_boundary']:.4f})")
            print(f"    W_FF / W_GG            {m['FF_over_GG']:.3f}")
            print(f"    transition accuracy    {m['trans_acc']:.3f}"
                  f"   (chance 0.40)")
            print(f"    word vs part-word AUC  {m['wp_auc']:.3f}"
                  f"   (no-learning floor {m['wp_auc_floor']:.3f})")
            print(f"    buildup early->late    {m['build_early']:+.4f} -> "
                  f"{m['build_late']:+.4f}  (gain {m['build_gain']:+.4f})")

    plot_measures(table, _out("interplay_measures.png"))

    # SFG-style layer-1 panels, on the condition the hypothesis is about.
    l1_cfg = base.replace(structure="structured", background=True)
    rp, rf = plastic_and_frozen(l1_cfg, snap_ms=500)
    plot_layer1(rp, rf, _out("interplay_layer1.png"))
    dec = current_decomposition(rp, rf)
    pm = position_modulation(rp, rf)
    print(f"\n  === layer 1 ===")
    print(f"    thalamic change fig/bg  {dec['tm_in_fig']:+.2e} / "
          f"{dec['tm_in_bg']:+.2e}   (must be 0)")
    for c, n in zip(CURRENTS[1:], ("recurrent", "inhibition", "net drive")):
        print(f"    {n:<22} {dec[c + '_fig']:+.4f} / {dec[c + '_bg']:+.4f}")
    print(f"    modulation by position  "
          + "  ".join(f"tok{i+1}: {v:+.2f}%" for i, v in enumerate(pm["mean"]))
          + f"   background: {pm['background'][0]:+.2f}%")

    # ---- the hypothesis, stated as the interaction ----
    def mean_of(label, key):
        return float(np.nanmean([r[key] for r in table[label]]))

    print("\n  === hypothesis ===")
    seg_s = mean_of("structured/bg", "seg_index")
    seg_r = mean_of("scrambled/bg", "seg_index")
    print(f"    segregation, structured {seg_s:+.4f} vs "
          f"scrambled {seg_r:+.4f}   difference {seg_s - seg_r:+.4f}")
    for key, name in (("wp_auc", "word vs part-word AUC"),
                      ("trans_acc", "transition accuracy")):
        d_bg = mean_of("structured/bg", key) - mean_of("scrambled/bg", key)
        d_cl = mean_of("structured/clean", key) - mean_of("scrambled/clean", key)
        print(f"    {name}: structure effect with bg {d_bg:+.4f}, "
              f"clean {d_cl:+.4f}, interaction {d_bg - d_cl:+.4f}")

    if not args.skip_robustness:
        levels = [0.25, 0.5, 1.0, 2.0, 4.0]
        curves: Dict[str, Dict[str, List[float]]] = {}
        for structure in ("structured", "scrambled"):
            series = {k: [] for k in ("seg_index", "wp_auc", "trans_acc")}
            for lv in levels:
                vals = []
                for s in range(args.seeds):
                    cfg = base.replace(structure=structure, background=True,
                                       bg_level=lv, seed=base.seed + s)
                    vals.append(measure_all(run(cfg, builder(cfg.n_channels))))
                for k in series:
                    series[k].append(float(np.nanmean([v[k] for v in vals])))
            curves[structure] = series
            print(f"  robustness [{structure}] "
                  + "  ".join(f"{lv}x:{v:+.3f}"
                              for lv, v in zip(levels, series["seg_index"])))
        plot_robustness(levels, curves, _out("interplay_robustness.png"))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
