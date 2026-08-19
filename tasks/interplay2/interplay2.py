"""
tasks.interplay2.interplay2
===========================

Two ordered tokens hidden in a continuous tone cloud, read out by a four-unit
layer 2.  Does each token get a unit?

    python -m tasks.interplay2.interplay2 [--preset short] [--seeds 5]

Measurement only; everything that draws is in ``figures.py``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from layer2_syllable.layer2 import Layer2
from model0 import simulate

if __package__:
    from .config import (A, B, C, CHANNEL_NAMES, CONDITIONS, D, Interplay2Config,
                         LAYER1_MODES, TOKENS, TOKEN_NAMES, get_preset,
                         layer1_config, layer2_config)
else:
    from tasks.interplay2.config import (  # type: ignore
        A, B, C, CHANNEL_NAMES, CONDITIONS, D, Interplay2Config, LAYER1_MODES,
        TOKENS, TOKEN_NAMES, get_preset, layer1_config, layer2_config)


#: Figures and caches are written next to the code that makes them.
OUT_DIR = Path(__file__).resolve().parent


def _out(name: str) -> str:
    return str(OUT_DIR / name)


# =====================================================================
#  Stimulus
# =====================================================================
def _token_slots(cfg: Interplay2Config, rng) -> Dict[str, np.ndarray]:
    """Slot index of each token channel, per block.

    Returns arrays of shape ``(n_blocks,)`` for each of A, B, C, D, plus the
    clock each pair sits on.  All three conditions give every channel exactly
    one slot per block; they differ only in how the four slots are related.
    """
    nb, ns = cfg.n_blocks, cfg.block_slots

    if cfg.condition == "paired":
        j0 = rng.integers(0, ns - 1, size=nb)        # AB on clock 0
        j1 = rng.integers(0, ns - 1, size=nb)        # CD on clock 1
        return dict(A=j0, B=j0 + 1, C=j1, D=j1 + 1, clock_CD=1)

    if cfg.condition == "shuffled":
        # Same duty, same non-simultaneity, random lag.  Two distinct slots
        # per pair, assigned to the two channels in random order, so the lag
        # from A to B is random in SIGN as well as in size.
        def pair() -> Tuple[np.ndarray, np.ndarray]:
            s = np.array([rng.choice(ns, size=2, replace=False)
                          for _ in range(nb)])
            return s[:, 0], s[:, 1]
        a, b = pair()
        c, d = pair()
        return dict(A=a, B=b, C=c, D=d, clock_CD=1)

    if cfg.condition == "sync":
        # CD moves onto clock 0 and starts with AB: A with C, B with D.
        j = rng.integers(0, ns - 1, size=nb)
        return dict(A=j, B=j + 1, C=j, D=j + 1, clock_CD=0)

    raise ValueError(f"unknown condition {cfg.condition!r}")


def build_stimulus(cfg: Interplay2Config, rng) -> Dict:
    """An ``(n_channels, T)`` tone cloud with the tokens written into it.

    Two clocks, half a slot apart.  A background voice on each, so both
    tokens have one synchronous voice and one asynchronous voice.  Every
    channel -- token or background -- gets exactly one tone per block.
    """
    ns_total = cfg.n_blocks * cfg.block_slots
    T = ns_total * cfg.slot + cfg.offset
    stim = np.zeros((cfg.n_channels, T))

    def onset(clock: int, block: int, slot: int) -> int:
        return (block * cfg.block_slots + slot) * cfg.slot + clock * cfg.offset

    def paint(ch: int, o: int) -> None:
        stim[ch, o:o + cfg.tone_dur] = cfg.amp

    # ---- tokens ----
    slots = _token_slots(cfg, rng)
    clock_CD = int(slots["clock_CD"])
    onsets: Dict[str, np.ndarray] = {}
    for name, ch, clock in (("A", A, 0), ("B", B, 0),
                            ("C", C, clock_CD), ("D", D, clock_CD)):
        o = np.array([onset(clock, blk, int(s))
                      for blk, s in enumerate(slots[name])])
        onsets[name] = o
        for oo in o:
            paint(ch, int(oo))

    # ---- background: one voice per clock ----
    #
    # Each block deals the sixteen background channels into two disjoint
    # halves, one per voice, so every channel plays exactly once per block
    # and no channel is tied to a clock.  Dealing them disjointly also
    # settles the only way two background tones could collide: the clocks
    # are half a slot apart, so voice 1's tone in slot k overlaps voice 0's
    # tones in slots k and k+1, and a channel in both voices at those slots
    # would fire one long tone instead of two.  Disjoint halves rule that
    # out inside a block; the block boundary is the one remaining case, and
    # the roll below rules it out there.
    bg = np.asarray(list(cfg.background_channels))
    per_voice = cfg.bg_per_voice
    last_slot = cfg.block_slots - 1
    prev_tail = -1                       # voice 1, last slot, previous block
    for blk in range(cfg.n_blocks):
        deal = rng.permutation(bg)
        halves = [deal[v * per_voice:(v + 1) * per_voice]
                  for v in range(cfg.n_voices)]
        if halves[0][0] == prev_tail:
            halves[0] = np.roll(halves[0], 1)
        for voice, block_chans in enumerate(halves):
            for slot, ch in enumerate(block_chans):
                paint(int(ch), onset(voice, blk, slot))
        prev_tail = int(halves[cfg.n_voices - 1][last_slot])

    _check(cfg, stim, T)
    return dict(stim=stim, onsets=onsets, slots=slots, T=T,
                clock_CD=clock_CD, cfg=cfg)


def _check(cfg: Interplay2Config, stim: np.ndarray, T: int) -> None:
    """The design constraints, verified rather than trusted."""
    on = stim > 0

    # Every channel on for exactly the same total time.  This is the
    # constraint that makes "the layer found the token" a statement about
    # order rather than about frequency.
    totals = on.sum(axis=1)
    assert np.ptp(totals) == 0, (
        f"channels unbalanced: totals span {np.ptp(totals)} samples "
        f"(min {totals.min()}, max {totals.max()})")

    # The cloud is never silent, and never carries more than one tone per
    # voice.  With a gap after every tone the count is no longer constant --
    # the two voices are half a slot apart, so their gaps do not coincide and
    # the total alternates between one and two.  What must hold is that it
    # never reaches zero, because a silent moment would let the layer-2 trace
    # reset and hand the model a segmentation cue the stimulus is meant to
    # withhold.  The two half-slot edges are excluded: the offset clock has
    # not started at t = 0 and has not finished at t = T.
    interior = slice(cfg.offset, T - cfg.offset)
    bgc = on[cfg.n_token_channels:, interior].sum(axis=0)
    assert bgc.min() >= 1, "the cloud must never fall silent"
    assert bgc.max() <= cfg.n_voices, (
        f"at most {cfg.n_voices} cloud tones at once; got {bgc.max()}")

    # The two tones of a token never overlap: a token is an order, not a
    # chord.  (In `sync` the two TOKENS overlap each other by design; the
    # tones within each still do not.)
    for (i, j) in TOKENS:
        assert not np.any(on[i] & on[j]), (
            f"token channels {i} and {j} overlap in time")


# =====================================================================
#  Running the two layers
# =====================================================================
def layer1_rates(stim: np.ndarray, a1cfg, mode: str = "full", seed: int = 0):
    """What layer 2 reads.

    ``raw``     the stimulus, no cortex at all
    ``frozen``  layer 1 with recurrent weights held at zero: adaptation and
                selective inhibition act, nothing is learned
    ``full``    layer 1 as normal
    """
    if mode == "raw":
        return stim.copy(), None
    if mode == "frozen":
        out = simulate(stim, cfg=a1cfg, W_init=np.zeros((a1cfg.N, a1cfg.N)),
                       learn=False, seed=seed)
    elif mode == "full":
        out = simulate(stim, cfg=a1cfg, learn=True, seed=seed)
    else:
        raise ValueError(f"unknown layer 1 mode {mode!r}")
    return out["E"], out["W_final"]


def train_and_test(cfg: Interplay2Config, *, mode: str = "full", seed: int = 0,
                   inh: str = "selective", record_every: int = 0) -> Dict:
    """Learn on one stream, freeze, measure on an independent one."""
    a1cfg = layer1_config(cfg.n_channels, inh=inh)
    l2cfg = layer2_config(seed=seed)

    train_stim = build_stimulus(cfg.replace(seed=seed),
                                np.random.default_rng(100 + seed))
    test_stim = build_stimulus(cfg.replace(seed=seed),
                               np.random.default_rng(900 + seed))

    E_train, W1 = layer1_rates(train_stim["stim"], a1cfg, mode=mode, seed=seed)
    E_test, _ = layer1_rates(test_stim["stim"], a1cfg, mode=mode, seed=seed)

    l2 = Layer2(cfg.n_channels, l2cfg)
    tr = l2.run(E_train, a1cfg.dt, learn=True, record_every=record_every)
    l2.reset_state()                      # clear the trace, keep the weights
    te = l2.run(E_test, a1cfg.dt, learn=False)

    return dict(cfg=cfg, a1cfg=a1cfg, l2=l2, mode=mode, seed=seed,
                train=train_stim, test=test_stim, tr=tr, te=te,
                E_train=E_train, E_test=E_test, W1=W1)


# =====================================================================
#  Reading the masks
# =====================================================================
def _pair_name(cfg: Interplay2Config, i: int, j: int) -> str:
    """``"B after A"`` for token channels, ``"bg after bg"`` otherwise."""
    def nm(ch: int) -> str:
        return CHANNEL_NAMES.get(ch, f"c{ch}")
    return f"{nm(i)} after {nm(j)}"


def mask_readout(res: Dict) -> List[Dict]:
    """One row per unit: what it settled on, and how concentrated it is.

    ``enrichment`` is the mask mass on an entry divided by the mass a flat
    mask would put there, so 1.0 means "no preference" whatever the size of
    the map.  With twenty channels and the diagonal excluded a flat mask
    spreads over 380 entries, so a unit that has genuinely committed to one
    ordered pair scores in the tens or hundreds.
    """
    cfg, l2 = res["cfg"], res["l2"]
    n = cfg.n_channels
    flat_entry = 1.0 / (n * (n - 1))
    bg0 = cfg.n_token_channels

    rows = []
    for k in range(l2.cfg.n_units):
        M = l2.M[k]
        tot = float(M.sum())
        norm = float(np.linalg.norm(M))
        if tot <= 0:
            rows.append(dict(unit=k, norm=norm, top=None, top_name="silent",
                             enrich_AB=0.0, enrich_CD=0.0, bg_mass=0.0,
                             committed=False))
            continue
        i, j = np.unravel_index(int(np.argmax(M)), M.shape)
        rows.append(dict(
            unit=k,
            norm=norm,
            top=(int(i), int(j)),
            top_name=_pair_name(cfg, int(i), int(j)),
            enrich_AB=float(M[B, A] / tot / flat_entry),
            enrich_CD=float(M[D, C] / tot / flat_entry),
            # The two orderings that pair one token's channels with the
            # other's.  They do not occur in `paired`, so they are the
            # baseline for "this entry is real"; in `sync` the four channels
            # form one object and they should rise to match the within-token
            # entries, which is how a merge is told apart from two objects
            # that merely happen to sit on two units.
            enrich_DA=float(M[D, A] / tot / flat_entry),
            enrich_BC=float(M[B, C] / tot / flat_entry),
            bg_mass=float(M[bg0:, bg0:].sum() / tot),
            committed=bool(l2.committed[k]),
        ))
    return rows


def allocation(res: Dict, enrich_thresh: float = 10.0) -> Dict:
    """Did each token get a unit of its own?

    A unit "codes" a token when its single largest mask entry IS that token's
    ordered pair and the mass there is at least ``enrich_thresh`` times flat.
    Requiring the argmax, not merely a large value, is what stops a unit that
    is mostly background from being counted twice.
    """
    rows = mask_readout(res)
    ab = [r for r in rows if r["top"] == (B, A)
          and r["enrich_AB"] >= enrich_thresh]
    cd = [r for r in rows if r["top"] == (D, C)
          and r["enrich_CD"] >= enrich_thresh]

    both = [r for r in rows
            if r["enrich_AB"] >= enrich_thresh
            and r["enrich_CD"] >= enrich_thresh]

    if ab and cd:
        verdict = "two units"
    elif both:
        verdict = "one unit, both tokens"
    elif ab or cd:
        verdict = "one token only"
    else:
        verdict = "neither token"

    return dict(rows=rows, verdict=verdict,
                unit_AB=ab[0]["unit"] if ab else None,
                unit_CD=cd[0]["unit"] if cd else None,
                n_ab=len(ab), n_cd=len(cd), n_both=len(both),
                n_committed=int(res["l2"].committed.sum()))


# =====================================================================
#  Reading the responses
# =====================================================================
def event_responses(res: Dict, pad_ms: int = 60) -> Dict[str, np.ndarray]:
    """Peak of every unit in the window that contains each token.

    The window opens at the SECOND tone's onset, because that is the only
    moment at which the ordered pair exists; it closes ``pad_ms`` after that
    tone ends, which is long enough for the layer's output to have peaked and
    short enough not to run into the next token.
    """
    cfg, y = res["cfg"], res["te"]["y"]
    T = y.shape[1]
    width = cfg.tone_dur + pad_ms
    on = res["test"]["onsets"]

    out: Dict[str, np.ndarray] = {}
    for tok, second in zip(TOKEN_NAMES, ("B", "D")):
        wins = [(int(o), min(int(o) + width, T)) for o in on[second]]
        out[tok] = np.array([y[:, a:b].max(axis=1) for a, b in wins if b > a])

    # A matched cloud baseline: same window length, drawn from moments that
    # contain neither second tone, so "responds to its token" is measured
    # against the cloud rather than against zero.
    busy = np.zeros(T, dtype=bool)
    for second in ("B", "D"):
        for o in on[second]:
            busy[int(o):min(int(o) + width, T)] = True
    rng = np.random.default_rng(0)
    free: List[Tuple[int, int]] = []
    for _ in range(4 * len(on["B"])):
        a = int(rng.integers(0, T - width))
        if not busy[a:a + width].any():
            free.append((a, a + width))
    out["cloud"] = (np.array([y[:, a:b].max(axis=1) for a, b in free])
                    if free else np.zeros((0, y.shape[0])))
    return out


def auc(x0: np.ndarray, x1: np.ndarray) -> float:
    """Area under the ROC separating two response samples."""
    if len(x0) == 0 or len(x1) == 0:
        return 0.5
    allv = np.concatenate([x0, x1])
    r = np.argsort(np.argsort(allv)) + 1.0
    u0 = r[:len(x0)].sum() - len(x0) * (len(x0) + 1) / 2.0
    return float(u0 / (len(x0) * len(x1)))


def _best_assignment(winner: np.ndarray, labels: np.ndarray,
                     units: np.ndarray) -> Tuple[float, Dict[int, int]]:
    """Balanced accuracy under the best unit-to-token assignment.

    Unit identity is arbitrary, so the units have to be assigned to tokens
    somehow.  Balanced accuracy is a sum over classes of per-class recalls,
    and each unit contributes to exactly one class, so the best assignment is
    separable: give every unit the token on which it wins the largest
    FRACTION of that token's events.  The syllable task searched permutations
    instead, which is the same thing when there are as many units as classes
    and silently drops the surplus units when there are more -- here there are
    four units and two tokens, so it would ignore half the population.
    """
    n_c = np.array([max((labels == c).sum(), 1) for c in range(len(TOKENS))])
    mapping: Dict[int, int] = {}
    recall = np.zeros(len(TOKENS))
    for u in units:
        share = np.array([(winner[labels == c] == u).sum() / n_c[c]
                          for c in range(len(TOKENS))])
        c = int(np.argmax(share))
        mapping[int(u)] = c
        recall[c] += share[c]
    return float(recall.mean()), mapping


def decode_tokens(resp: Dict[str, np.ndarray], committed: np.ndarray,
                  n_shuffle: int = 200, seed: int = 0) -> Dict:
    """Winner-take-all across committed units: AB or CD?

    Because the assignment above is fitted to the labels, chance is NOT 0.5:
    four units assigned optimally beat 0.5 on pure noise.  The null is
    therefore measured, by refitting the assignment on shuffled labels, and
    reported alongside the score.  Comparing against 0.5 would be reading a
    fitting artefact as a result.
    """
    idx = np.flatnonzero(committed)
    if idx.size == 0:
        return dict(acc=0.5, null=0.5, mapping=None)

    X = np.concatenate([resp["AB"], resp["CD"]])
    labels = np.concatenate([np.zeros(len(resp["AB"]), dtype=int),
                             np.ones(len(resp["CD"]), dtype=int)])
    winner = idx[np.argmax(X[:, idx], axis=1)]

    acc, mapping = _best_assignment(winner, labels, idx)
    rng = np.random.default_rng(seed)
    null = np.array([_best_assignment(winner, rng.permutation(labels), idx)[0]
                     for _ in range(n_shuffle)])
    return dict(acc=acc, mapping=mapping, null=float(null.mean()),
                null_hi=float(np.quantile(null, 0.95)))


def summarise(res: Dict) -> Dict:
    """Everything scalar about one run."""
    alloc = allocation(res)
    resp = event_responses(res)
    dec = decode_tokens(resp, res["l2"].committed)

    rows = alloc["rows"]
    sel, above = [], []
    for r in rows:
        a = resp["AB"][:, r["unit"]]
        c = resp["CD"][:, r["unit"]]
        g = resp["cloud"][:, r["unit"]] if len(resp["cloud"]) else np.zeros(1)
        s = (a.mean() - c.mean()) / (a.mean() + c.mean()) \
            if (a.mean() + c.mean()) > 0 else 0.0
        sel.append(float(s))
        # P(the unit's preferred token beats a matched cloud window).
        above.append(auc(np.maximum(a, c), g))

    return dict(seed=res["seed"], mode=res["mode"],
                condition=res["cfg"].condition,
                verdict=alloc["verdict"], n_committed=alloc["n_committed"],
                unit_AB=alloc["unit_AB"], unit_CD=alloc["unit_CD"],
                n_ab=alloc["n_ab"], n_cd=alloc["n_cd"], n_both=alloc["n_both"],
                rows=rows, masks=res["l2"].M.copy(), selectivity=np.array(sel),
                auc_vs_cloud=np.array(above),
                decode_acc=dec["acc"], decode_null=dec["null"],
                decode_null_hi=dec["null_hi"], resp=resp)


# =====================================================================
#  Run
# =====================================================================
def run_condition(preset: str, condition: str, seeds: Sequence[int],
                  mode: str = "full", **overrides) -> List[Dict]:
    out = []
    for s in seeds:
        cfg = get_preset(preset, condition=condition, **overrides)
        res = train_and_test(cfg, mode=mode, seed=int(s))
        summ = summarise(res)
        summ["_res"] = res
        out.append(summ)
        print(f"    seed {s}: {summ['verdict']:<24s} "
              f"committed {summ['n_committed']}/{res['l2'].cfg.n_units}  "
              f"decode {summ['decode_acc']:.3f}")
    return out


def _report(label: str, runs: List[Dict]) -> None:
    verdicts = [r["verdict"] for r in runs]
    two = sum(v == "two units" for v in verdicts)
    print(f"\n  {label}")
    print(f"    two units, one per token : {two}/{len(runs)} runs")
    for v in ("one unit, both tokens", "one token only", "neither token"):
        n = sum(x == v for x in verdicts)
        if n:
            print(f"    {v:<25s}: {n}/{len(runs)} runs")
    acc = np.array([r["decode_acc"] for r in runs])
    null = np.array([r["decode_null"] for r in runs])
    print(f"    AB vs CD decoding        : {acc.mean():.3f} "
          f"+- {acc.std():.3f}   (shuffled null {null.mean():.3f})")


def main(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--mode", default="full", choices=LAYER1_MODES)
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    p.add_argument("--layer1-sweep", action="store_true",
                   help="also run the paired condition through raw / frozen")
    p.add_argument("--figures-only", action="store_true")
    args = p.parse_args(argv)

    cache = Path(_out("results.pkl"))
    if args.figures_only:
        with open(cache, "rb") as fh:
            store = pickle.load(fh)
    else:
        seeds = list(range(args.seeds))
        store = {"conditions": {}, "layer1": {}, "preset": args.preset,
                 "seeds": seeds, "mode": args.mode}
        for cond in args.conditions:
            print(f"\n[{cond}]")
            runs = run_condition(args.preset, cond, seeds, mode=args.mode)
            store["conditions"][cond] = [
                {k: v for k, v in r.items() if k != "_res"} for r in runs]
            _report(cond, runs)
            if cond == args.conditions[0]:
                store["example"] = _example(runs[0]["_res"])

        if args.layer1_sweep:
            for m in ("raw", "frozen"):
                print(f"\n[paired, layer 1 = {m}]")
                runs = run_condition(args.preset, "paired", seeds, mode=m)
                store["layer1"][m] = [
                    {k: v for k, v in r.items() if k != "_res"} for r in runs]
                _report(f"paired / {m}", runs)

        with open(cache, "wb") as fh:
            pickle.dump(store, fh)

    from tasks.interplay2.figures import make_figures
    make_figures(store)
    return store


def _example(res: Dict, n_blocks: int = 6) -> Dict:
    """A window from the END of the test stream, for the tape figure.

    From the end, because what is worth drawing is the settled behaviour and
    not the transient while the masks are still forming.
    """
    cfg = res["cfg"]
    T = res["te"]["y"].shape[1]
    width = n_blocks * cfg.block_samples
    t0 = max(0, T - width - cfg.offset)
    t1 = t0 + width

    on = res["test"]["onsets"]
    keep = {k: v[(v >= t0) & (v < t1)] - t0 for k, v in on.items()}

    # Whole-stream summaries.  The drawing window is six blocks and its edges
    # cut tones, so the design checks have to be computed on the full stream
    # or they would show an imbalance the stimulus does not have.
    full = res["test"]["stim"] > 0
    interior = slice(cfg.offset, full.shape[1] - cfg.offset)
    checks = dict(
        on_time=full.sum(axis=1) * res["a1cfg"].dt,
        n_simul=full[:, interior].sum(axis=0),
        n_simul_bg=full[cfg.n_token_channels:, interior].sum(axis=0),
        lags={f"{b} after {a}": (on[b] - on[a]) * res["a1cfg"].dt * 1e3
              for a, b in (("A", "B"), ("C", "D"))},
    )
    return dict(checks=checks,
                stim=res["test"]["stim"][:, t0:t1],
                E=res["E_test"][:, t0:t1],
                s=res["te"]["s"][:, t0:t1],
                y=res["te"]["y"][:, t0:t1],
                onsets=keep, t0=t0, dt=res["a1cfg"].dt, cfg=cfg,
                masks=res["l2"].M.copy(),
                committed=res["l2"].committed.copy(),
                rows=mask_readout(res))


if __name__ == "__main__":
    main()
