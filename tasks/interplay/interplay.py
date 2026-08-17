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
    interplay_stimulus.png    the stimulus and its two background constraints
                               constraints shown rather than asserted
    interplay_results.png     the structured-vs-scrambled comparison
    interplay_mechanism.png   weights, position modulation, currents

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


def position_currents(res_p: Dict, res_f: Dict) -> Dict[str, np.ndarray]:
    """Plastic-minus-frozen current change, resolved by position in word.

    The figure-versus-background contrast is confounded: the background is
    never silent, so its channels always carry a live trace and draw more
    recurrent current whatever is learned.  Token position inside a word is
    not confounded -- same pool, same duty cycle, same channels over the
    session -- and it is the contrast the hypothesis actually makes, since
    only tokens 2 and 3 have a predecessor to be predicted by.

    Differencing against the frozen run cancels the stimulus exactly, so
    the thalamic term is zero at every position by construction and is
    returned as the check that it is.
    """
    cfg: InterplayConfig = res_p["cfg"]
    tokens, onsets = res_p["tokens"], res_p["onsets"]
    pos = np.arange(len(tokens)) % 3
    fields = {"tm_in": ("tm_in",), "rec_E": ("rec_E",),
              "inh_to_E": ("inh_to_E",), "net": ("tm_in", "rec_E",
                                                 "inh_to_E")}

    def stack(res):
        return {k: (res["tm_in"] + res["rec_E"] - res["inh_to_E"]
                    if k == "net" else res[k]) for k in fields}

    A, B = stack(res_p), stack(res_f)
    out: Dict[str, np.ndarray] = {}
    for k in fields:
        per = [[] for _ in range(3)]
        for i, (ch, o) in enumerate(zip(tokens, onsets)):
            a, b = o, o + cfg.tone_dur
            per[pos[i]].append(A[k][ch, a:b].mean() - B[k][ch, a:b].mean())
        out[k] = np.array([np.mean(v) for v in per])
    return out


def paired_sign_flip(diff: np.ndarray) -> Tuple[float, float]:
    """Exact sign-flip permutation test on paired within-seed differences.

    Exact because the seeds are few: with n <= 20 every one of the 2^n
    sign assignments is enumerated, so the p value is the true
    permutation p and not an estimate of it.  The pairing is real -- both
    members of each difference come from the same seed and therefore the
    same stimulus -- so sign flipping is the correct exchangeability.
    """
    import itertools
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan"), float("nan")
    obs = float(d.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-15))
    return obs, p


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
#  Entry point
# =====================================================================
def _agg(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--inh", default="selective",
                   choices=("selective", "uniform"))
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--figures-only", action="store_true",
                   help="re-render from the cached results, no simulation")
    args = p.parse_args(argv)

    import pickle

    from tasks.interplay import figures

    cache = Path(__file__).resolve().parent / "results.pkl"
    if args.figures_only:
        # Style changes should not cost a 30-minute re-simulation.
        with open(cache, "rb") as fh:
            d = pickle.load(fh)
        figures.stimulus_figure(d["pack"], d["scfg"])
        figures.mechanism_figure(d["ex_p"], d["ex_groups"], d["pos_all"],
                                 d["pos_bg"], d["dec_all"], d["rates"],
                                 poscur=d.get("poscur"),
                                 tests=d.get("tests"))
        figures.results_figure(d["table"], d["levels"], d["curves"])
        print(f"re-rendered from {cache.name}")
        return 0

    base = get_preset(args.preset)
    builder = {"selective": model_config,
               "uniform": uniform_model_config}[args.inh]
    seeds = [base.seed + s for s in range(args.seeds)]

    print(f"[ Interplay -- {base.name}, {args.inh} inhibition, "
          f"{args.seeds} seeds ]")
    print(f"  {base.n_figure} figure + {base.n_background} background "
          f"channels; tone {base.tone_dur} ms in both streams, "
          f"slot {base.slot} ms")
    print(f"  {base.n_words} words = {base.n_tokens} tokens "
          f"= {base.n_tokens * base.slot / 1000:.0f} s per run")

    # ---- stimulus ----
    scfg = base.replace(structure="structured", background=True)
    figures.stimulus_figure(
        build_stimulus(scfg, np.random.default_rng(scfg.seed)), scfg)

    # ---- the 2 x 2 ----
    table: Dict[str, List[Dict[str, float]]] = {}
    for structure in ("structured", "scrambled"):
        for bg in (True, False):
            label = f"{structure}/{'bg' if bg else 'clean'}"
            rows = []
            for s in seeds:
                cfg = base.replace(structure=structure, background=bg, seed=s)
                rows.append(measure_all(run(cfg, builder(cfg.n_channels))))
            table[label] = rows
            m = {k: np.nanmean(_agg(rows, k)) for k in rows[0]}
            print(f"\n  [{label}]  n = {len(rows)} seeds")
            print(f"    drive ratio bg/fig     {m['drive_ratio']:.3f}")
            print(f"    segregation index      {m['seg_index']:+.4f}")
            print(f"    W within / boundary    {m['within_over_boundary']:.3f}")
            print(f"    transition accuracy    {m['trans_acc']:.3f}")
            print(f"    word vs part-word AUC  {m['wp_auc']:.3f}  "
                  f"(floor {m['wp_auc_floor']:.3f})")

    # ---- layer 1, every seed ----
    pos_all, dec_all = [], {f"{c}_{p}": [] for c in CURRENTS
                            for p in ("fig", "bg")}
    rates = {"fig_p": [], "fig_f": [], "bg_p": [], "bg_f": []}
    pos_bg, ex_p, ex_groups = [], None, None
    poscur = {k: [] for k in CURRENTS}
    for i, s in enumerate(seeds):
        cfg = base.replace(structure="structured", background=True, seed=s)
        rp, rf = plastic_and_frozen(cfg, snap_ms=500 if i == 0 else 0)
        pm = position_modulation(rp, rf)
        pos_all.append(pm["mean"]); pos_bg.append(pm["background"][0])
        for k, v in current_decomposition(rp, rf).items():
            dec_all[k].append(v)
        pc = position_currents(rp, rf)
        for k in CURRENTS:
            poscur[k].append(pc[k])
        nf = cfg.n_figure
        rates["fig_p"].append(rp["E"][:nf].mean())
        rates["fig_f"].append(rf["E"][:nf].mean())
        rates["bg_p"].append(rp["E"][nf:].mean())
        rates["bg_f"].append(rf["E"][nf:].mean())
        if i == 0:
            ex_p, ex_groups = rp, weight_groups(rp)
    pos_all = np.vstack(pos_all)
    dec_all = {k: np.asarray(v) for k, v in dec_all.items()}
    poscur = {k: np.vstack(v) for k, v in poscur.items()}

    # Position 1 has no predecessor inside the word, position 3 has two.
    # The paired difference is within-seed, so an exact sign-flip test is
    # the right null.
    tests = {k: paired_sign_flip(poscur[k][:, 2] - poscur[k][:, 0])
             for k in CURRENTS}

    print(f"\n  [layer 1]  n = {len(seeds)} seeds")
    print(f"    thalamic change        {dec_all['tm_in_fig'].mean():+.2e} "
          f"(must be 0)")
    floor = 2.0 / 2 ** len(seeds)
    print(f"    current change, position 1 -> 3 (exact sign-flip, "
          f"n = {len(seeds)}; smallest attainable P = {floor:.4f})")
    for k, name in zip(CURRENTS, ("thalamic", "recurrent", "inhibition",
                                  "net")):
        d, pv = tests[k]
        m = poscur[k].mean(axis=0)
        print(f"      {name:<11} {m[0]:+.4f} -> {m[2]:+.4f}   "
              f"delta {d:+.4f}   P = {pv:.4f}")
    print(f"    modulation by position "
          + "  ".join(f"tok{i+1} {pos_all[:, i].mean():+.2f}%"
                      for i in range(3))
          + f"   background {np.mean(pos_bg):+.2f}%")

    figures.mechanism_figure(ex_p, ex_groups, pos_all,
                             np.asarray(pos_bg), dec_all,
                             {k: np.asarray(v) for k, v in rates.items()},
                             poscur=poscur, tests=tests)

    # ---- robustness ----
    levels = [0.25, 0.5, 1.0, 2.0, 4.0]
    curves: Dict[str, Dict[str, List[np.ndarray]]] = {}
    if not args.skip_robustness:
        for structure in ("structured", "scrambled"):
            series = {k: [] for k in ("seg_index", "wp_auc", "trans_acc")}
            for lv in levels:
                vals = []
                for s in seeds:
                    cfg = base.replace(structure=structure, background=True,
                                       bg_level=lv, seed=s)
                    vals.append(measure_all(run(cfg, builder(cfg.n_channels))))
                for k in series:
                    series[k].append(_agg(vals, k))
            curves[structure] = series
    else:
        for structure in ("structured", "scrambled"):
            curves[structure] = {k: [np.array([np.nan] * len(seeds))
                                     for _ in levels]
                                 for k in ("seg_index", "wp_auc",
                                           "trans_acc")}

    with open(cache, "wb") as fh:
        pickle.dump(dict(table=table, levels=levels, curves=curves,
                         pos_all=pos_all, pos_bg=np.asarray(pos_bg),
                         dec_all=dec_all,
                         rates={k: np.asarray(v) for k, v in rates.items()},
                         ex_p={k: ex_p[k] for k in ("W_final", "cfg")},
                         ex_groups=ex_groups, scfg=scfg,
                         poscur=poscur, tests=tests,
                         pack=build_stimulus(
                             scfg, np.random.default_rng(scfg.seed))), fh)

    paths = figures.results_figure(table, levels, curves)

    # ---- the comparison the design exists for ----
    def mean_of(label, key):
        return float(np.nanmean(_agg(table[label], key)))

    print("\n  === structured vs scrambled, with background ===")
    for key, name in (("seg_index", "segregation index"),
                      ("trans_acc", "transition accuracy"),
                      ("wp_auc", "word vs part-word AUC"),
                      ("within_over_boundary", "within/boundary weight")):
        a, b = mean_of("structured/bg", key), mean_of("scrambled/bg", key)
        print(f"    {name:<24} {a:+.4f}  vs  {b:+.4f}   "
              f"difference {a - b:+.4f}")

    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
