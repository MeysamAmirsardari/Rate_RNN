"""
tasks.interplay3.interplay3
===========================

Three four-tone words hidden in a fifty-channel tone cloud, read out by the
multi-rate layer 2.  Does each word get a unit that spans the whole word?

    python -m tasks.interplay3.interplay3 [--preset short] [--seeds 3]

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

from layer2_multirate.sweep import span_depth
from model0 import simulate

if __package__:
    from .config import (CHANNEL_NAMES, CONDITIONS, Interplay3Config,
                         LAYER1_MODES, SPAN_THRESH, WORD_LEN, WORD_NAMES,
                         WORDS, get_preset, layer1_config, layer2_config)
    from .fast_layer2 import FastLayer2MR
else:
    from tasks.interplay3.config import (  # type: ignore
        CHANNEL_NAMES, CONDITIONS, Interplay3Config, LAYER1_MODES, SPAN_THRESH,
        WORD_LEN, WORD_NAMES, WORDS, get_preset, layer1_config, layer2_config)
    from tasks.interplay3.fast_layer2 import FastLayer2MR  # type: ignore


OUT_DIR = Path(__file__).resolve().parent


def _out(name: str) -> str:
    return str(OUT_DIR / name)


# =====================================================================
#  Stimulus
# =====================================================================
def _word_slots(cfg: Interplay3Config, rng) -> Dict[str, np.ndarray]:
    """Slot index of every word channel, per block, plus each word's clock.

    All three conditions give every channel exactly one slot per block; they
    differ only in how those slots are related.
    """
    nb = cfg.n_blocks

    #: (n_words, n_blocks) -- which clock each word sits on in each block.
    if cfg.condition == "sync":
        clocks = np.zeros((len(WORDS), nb), dtype=int)
    else:
        clocks = np.array([[cfg.word_clock(w, b) for b in range(nb)]
                           for w in range(len(WORDS))])

    #: How many slots this word may use in this block; clock-dependent, see
    #: ``Interplay3Config.last_slot``.
    top = np.vectorize(cfg.last_slot)(clocks)          # (n_words, nb)

    slots = np.empty((len(WORDS), nb, WORD_LEN), dtype=int)

    if cfg.condition in ("paired", "sync"):
        # Four consecutive slots.  In `sync` every word takes the SAME start
        # on clock 0, so the twelve channels become one object of four
        # simultaneous triples.
        if cfg.condition == "sync":
            start = np.broadcast_to(
                rng.integers(0, top[0] - WORD_LEN + 2, size=nb), top.shape)
        else:
            start = np.array([[rng.integers(0, top[w, b] - WORD_LEN + 2)
                               for b in range(nb)] for w in range(len(WORDS))])
        for p in range(WORD_LEN):
            slots[:, :, p] = start + p

    elif cfg.condition == "shuffled":
        # Four distinct slots per word per block, assigned to the four
        # channels in random order, so the lag between consecutive tones is
        # random in sign and size.  Duty, spectrum and simultaneity are
        # exactly as in `paired`; only the order is gone.
        for w in range(len(WORDS)):
            for b in range(nb):
                slots[w, b] = rng.choice(top[w, b] + 1, size=WORD_LEN,
                                         replace=False)

    else:
        raise ValueError(f"unknown condition {cfg.condition!r}")

    return dict(slots=slots, clocks=clocks)      # slots: (n_words, nb, L)


def build_stimulus(cfg: Interplay3Config, rng) -> Dict:
    """An ``(n_channels, T)`` cloud with the three words written into it."""
    offsets = cfg.clock_offsets
    ns_total = cfg.n_blocks * cfg.block_slots
    T = ns_total * cfg.slot + max(offsets) + cfg.tone_dur
    stim = np.zeros((cfg.n_channels, T))

    def onset(clock: int, block: int, slot: int) -> int:
        return ((block * cfg.block_slots + slot) * cfg.slot
                + offsets[clock])

    def paint(ch: int, o: int) -> None:
        stim[ch, o:o + cfg.tone_dur] = cfg.amp

    # ---- words ----
    plan = _word_slots(cfg, rng)
    slots, clocks = plan["slots"], plan["clocks"]
    onsets: Dict[int, np.ndarray] = {}
    for w, word in enumerate(WORDS):
        for p, ch in enumerate(word):
            o = np.array([onset(int(clocks[w, b]), b, int(slots[w, b, p]))
                          for b in range(cfg.n_blocks)])
            onsets[ch] = o
            for oo in o:
                paint(ch, int(oo))

    # ---- cloud ----
    #
    # Two voices, nineteen slots, thirty-eight channels: one tone per channel
    # per block, dealt as a permutation split into disjoint halves.  Disjoint
    # halves rule out a channel colliding with itself inside a block; the
    # block boundary is the one case left, because a tone on a late clock in
    # the last slot runs past the block edge and can meet the first slot of
    # the next block.  The roll below rules that out.
    bg = np.asarray(list(cfg.background_channels))
    per_voice = cfg.n_background // cfg.n_voices
    voice_clocks = cfg.voice_clocks
    prev_tail: set = set()
    for b in range(cfg.n_blocks):
        deal = rng.permutation(bg)
        halves = [deal[v * per_voice:(v + 1) * per_voice]
                  for v in range(cfg.n_voices)]
        for v, half in enumerate(halves):
            for _ in range(len(half)):
                if int(half[0]) not in prev_tail:
                    break
                half = np.roll(half, 1)
            halves[v] = half
        for v, half in enumerate(halves):
            for slot, ch in enumerate(half):
                paint(int(ch), onset(int(voice_clocks[v]), b, slot))
        prev_tail = {int(h[-1]) for h in halves}

    _check(cfg, stim, T)
    return dict(stim=stim, onsets=onsets, slots=slots, clocks=clocks,
                T=T, cfg=cfg)


def _check(cfg: Interplay3Config, stim: np.ndarray, T: int) -> None:
    """The design constraints, verified rather than trusted."""
    on = stim > 0

    # Every channel on for exactly the same total time.  This is what makes
    # "the layer found the word" a statement about order and not about rate.
    totals = on.sum(axis=1)
    assert np.ptp(totals) == 0, (
        f"channels unbalanced: totals span {np.ptp(totals)} samples "
        f"(min {totals.min()}, max {totals.max()})")

    # The cloud never falls silent.  A silent moment would let the layer-2
    # filterbank drain and hand the model a segmentation cue the design
    # withholds.  The edges are excluded: the offset clocks have not started
    # at t = 0 and have not finished at t = T.
    edge = max(cfg.clock_offsets) + cfg.slot
    bgc = on[cfg.n_token_channels:, edge:T - edge].sum(axis=0)
    assert bgc.min() >= 1, "the cloud must never fall silent"
    assert bgc.max() <= cfg.n_voices, (
        f"at most {cfg.n_voices} cloud tones at once; got {bgc.max()}")

    # No two tones of the same word ever overlap: a word is an order, not a
    # chord.  (In `sync` the three WORDS overlap each other by design.)
    for word in WORDS:
        for a in range(len(word)):
            for b in range(a + 1, len(word)):
                assert not np.any(on[word[a]] & on[word[b]]), (
                    f"channels {word[a]} and {word[b]} of the same word "
                    f"overlap in time")


# =====================================================================
#  Running the two layers
# =====================================================================
def layer1_rates(stim: np.ndarray, a1cfg, mode: str = "full", seed: int = 0):
    """What layer 2 reads: the stimulus, a frozen cortex, or a full one."""
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


def train_and_test(cfg: Interplay3Config, *, mode: str = "full", seed: int = 0,
                   inh: str = "selective", **l2_over) -> Dict:
    """Learn on one stream, freeze, measure on an independent one."""
    a1cfg = layer1_config(cfg.n_channels, inh=inh)
    l2cfg = layer2_config(seed=seed, **l2_over)

    train = build_stimulus(cfg.replace(seed=seed),
                           np.random.default_rng(100 + seed))
    test = build_stimulus(cfg.replace(seed=seed, n_blocks=cfg.n_test_blocks),
                          np.random.default_rng(900 + seed))

    E_train, W1 = layer1_rates(train["stim"], a1cfg, mode=mode, seed=seed)
    E_test, _ = layer1_rates(test["stim"], a1cfg, mode=mode, seed=seed)

    l2 = FastLayer2MR(cfg.n_channels, l2cfg)
    l2.run(E_train, a1cfg.dt, learn=True)
    l2.reset_state()                     # clear the filterbank, keep weights
    te = l2.run(E_test, a1cfg.dt, learn=False)

    return dict(cfg=cfg, a1cfg=a1cfg, l2=l2, mode=mode, seed=seed,
                train=train, test=test, te=te, E_test=E_test, W1=W1)


# =====================================================================
#  Reading the masks
# =====================================================================
def unit_table(res: Dict) -> List[Dict]:
    """One row per committed unit: what it fires on, and how deep it reaches.

    ``depth`` is ``layer2_multirate``'s own measure -- how many tokens back
    the unit represents, in order, at strictly slower rates.  A four-tone word
    has three predecessors, so ``depth == 3`` means the unit spans the whole
    word.
    """
    l2 = res["l2"]
    rows = []
    for u in np.flatnonzero(l2.committed):
        depth, word = span_depth(l2, int(u), WORDS, thresh=SPAN_THRESH)
        i = int(np.argmax(l2.M[u].sum(axis=1)))     # strongest input channel
        w_idx = next((k for k, w in enumerate(WORDS) if i in w), None)
        pos = WORDS[w_idx].index(i) if w_idx is not None else None
        rows.append(dict(unit=int(u), depth=int(depth), now=i,
                         now_name=CHANNEL_NAMES.get(i, f"c{i}"),
                         word=w_idx, pos=pos,
                         spans=bool(depth >= WORD_LEN - 1
                                    and w_idx is not None),
                         norm=float(l2.mask_norms[u])))
    return sorted(rows, key=lambda r: (-r["depth"], r["now"]))


def allocation(res: Dict) -> Dict:
    """Did each word get a unit that spans it?"""
    rows = unit_table(res)
    spanning = [r for r in rows if r["spans"]]
    covered = sorted({r["word"] for r in spanning})

    if len(covered) == len(WORDS):
        verdict = "all words spanned"
    elif covered:
        verdict = f"{len(covered)} of {len(WORDS)} words spanned"
    else:
        verdict = "no word spanned"

    return dict(rows=rows, verdict=verdict, n_committed=len(rows),
                n_spanning=len(spanning), words_covered=covered,
                depth_hist={d: sum(r["depth"] == d for r in rows)
                            for d in range(WORD_LEN)})


# =====================================================================
#  Reading the responses
# =====================================================================
def event_responses(res: Dict, pad_ms: int = 60) -> Dict[str, np.ndarray]:
    """Peak of every unit in the window that completes each word.

    The window opens at the LAST tone's onset, because that is the only
    moment at which the whole word exists, and closes ``pad_ms`` after that
    tone ends.
    """
    cfg, y = res["cfg"], res["te"]["y"]
    T = y.shape[1]
    width = cfg.tone_dur + pad_ms
    on = res["test"]["onsets"]

    out: Dict[str, np.ndarray] = {}
    busy = np.zeros(T, dtype=bool)
    for w, word in enumerate(WORDS):
        last = on[word[-1]]
        wins = [(int(o), min(int(o) + width, T)) for o in last]
        out[WORD_NAMES[w]] = np.array(
            [y[:, a:b].max(axis=1) for a, b in wins if b > a])
        for a, b in wins:
            busy[a:b] = True

    rng = np.random.default_rng(0)
    free: List[Tuple[int, int]] = []
    for _ in range(6 * len(on[WORDS[0][-1]])):
        a = int(rng.integers(0, T - width))
        if not busy[a:a + width].any():
            free.append((a, a + width))
    out["cloud"] = (np.array([y[:, a:b].max(axis=1) for a, b in free])
                    if free else np.zeros((0, y.shape[0])))
    return out


def _best_assignment(winner: np.ndarray, labels: np.ndarray,
                     units: np.ndarray) -> float:
    """Balanced accuracy under the best unit-to-word assignment.

    Separable, as in interplay2: balanced accuracy sums per-class recalls and
    each unit contributes to exactly one class, so give every unit the word on
    which it wins the largest fraction of that word's events.
    """
    n_c = np.array([max((labels == c).sum(), 1) for c in range(len(WORDS))])
    recall = np.zeros(len(WORDS))
    for u in units:
        share = np.array([(winner[labels == c] == u).sum() / n_c[c]
                          for c in range(len(WORDS))])
        recall[int(np.argmax(share))] += share.max()
    return float(recall.mean())


def decode_words(resp: Dict[str, np.ndarray], committed: np.ndarray,
                 n_shuffle: int = 200, seed: int = 0) -> Dict:
    """Winner-take-all across committed units: which word?

    The assignment is fitted to the labels, so chance is not 1/3.  The null is
    measured by refitting on shuffled labels rather than assumed.
    """
    idx = np.flatnonzero(committed)
    if idx.size == 0:
        return dict(acc=1.0 / len(WORDS), null=1.0 / len(WORDS))

    X = np.concatenate([resp[n] for n in WORD_NAMES])
    labels = np.concatenate([np.full(len(resp[n]), c, dtype=int)
                             for c, n in enumerate(WORD_NAMES)])
    winner = idx[np.argmax(X[:, idx], axis=1)]

    acc = _best_assignment(winner, labels, idx)
    rng = np.random.default_rng(seed)
    null = np.array([_best_assignment(winner, rng.permutation(labels), idx)
                     for _ in range(n_shuffle)])
    return dict(acc=acc, null=float(null.mean()),
                null_hi=float(np.quantile(null, 0.95)))


def auc(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) == 0 or len(x1) == 0:
        return 0.5
    allv = np.concatenate([x0, x1])
    r = np.argsort(np.argsort(allv)) + 1.0
    u0 = r[:len(x0)].sum() - len(x0) * (len(x0) + 1) / 2.0
    return float(u0 / (len(x0) * len(x1)))


def summarise(res: Dict) -> Dict:
    alloc = allocation(res)
    resp = event_responses(res)
    dec = decode_words(resp, res["l2"].committed)

    cloud = resp["cloud"]
    above = []
    for r in alloc["rows"]:
        best = np.max([resp[n][:, r["unit"]] for n in WORD_NAMES], axis=0)
        g = cloud[:, r["unit"]] if len(cloud) else np.zeros(1)
        above.append(auc(best, g))

    return dict(seed=res["seed"], mode=res["mode"],
                condition=res["cfg"].condition,
                verdict=alloc["verdict"], rows=alloc["rows"],
                n_committed=alloc["n_committed"],
                n_spanning=alloc["n_spanning"],
                words_covered=alloc["words_covered"],
                depth_hist=alloc["depth_hist"],
                masks=res["l2"].M_true.copy(),
                W1=None if res["W1"] is None else np.asarray(res["W1"]).copy(),
                auc_vs_cloud=np.array(above),
                decode_acc=dec["acc"], decode_null=dec["null"],
                resp={k: v for k, v in resp.items()})


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
        print(f"    seed {s}: {summ['verdict']:<26s} "
              f"committed {summ['n_committed']:2d}  "
              f"spanning {summ['n_spanning']:2d}  "
              f"words {[WORD_NAMES[w] for w in summ['words_covered']]}  "
              f"decode {summ['decode_acc']:.3f}", flush=True)
    return out


def _report(label: str, runs: List[Dict]) -> None:
    n_all = sum(len(r["words_covered"]) == len(WORDS) for r in runs)
    cov = np.array([len(r["words_covered"]) for r in runs], dtype=float)
    acc = np.array([r["decode_acc"] for r in runs])
    null = np.array([r["decode_null"] for r in runs])
    print(f"\n  {label}")
    print(f"    all three words spanned : {n_all}/{len(runs)} runs")
    print(f"    words covered           : {cov.mean():.2f} +- {cov.std():.2f}"
          f"  of {len(WORDS)}")
    print(f"    word decoding           : {acc.mean():.3f} "
          f"+- {acc.std():.3f}   (shuffled null {null.mean():.3f})",
          flush=True)


def main(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="default")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--mode", default="full", choices=LAYER1_MODES)
    p.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    p.add_argument("--layer1-sweep", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    #: A per-process cache suffix, so the three conditions can be run as three
    #: processes and merged afterwards; see ``run_parallel.py``.
    p.add_argument("--tag", default="")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    cache = Path(_out(f"results{args.tag}.pkl"))
    if args.figures_only:
        with open(cache, "rb") as fh:
            store = pickle.load(fh)
    else:
        seeds = list(range(args.seeds))
        store = {"conditions": {}, "layer1": {}, "preset": args.preset,
                 "seeds": seeds, "mode": args.mode}
        for cond in args.conditions:
            print(f"\n[{cond}]", flush=True)
            runs = run_condition(args.preset, cond, seeds, mode=args.mode)
            store["conditions"][cond] = [
                {k: v for k, v in r.items() if k != "_res"} for r in runs]
            _report(cond, runs)
            if cond == args.conditions[0]:
                store["example"] = _example(runs[0]["_res"])

        if args.layer1_sweep:
            for m in ("raw", "frozen"):
                print(f"\n[paired, layer 1 = {m}]", flush=True)
                runs = run_condition(args.preset, "paired", seeds, mode=m)
                store["layer1"][m] = [
                    {k: v for k, v in r.items() if k != "_res"} for r in runs]
                _report(f"paired / {m}", runs)

        with open(cache, "wb") as fh:
            pickle.dump(store, fh)

    if not args.no_figures:
        from tasks.interplay3.figures import make_figures
        make_figures(store)
    return store


def _example(res: Dict, n_blocks: int = 3) -> Dict:
    """A window from the END of the test stream, for the tape figure."""
    cfg = res["cfg"]
    T = res["te"]["y"].shape[1]
    width = n_blocks * cfg.block_samples
    t0 = max(0, T - width - cfg.slot)
    t1 = t0 + width

    on = res["test"]["onsets"]
    keep = {ch: v[(v >= t0) & (v < t1)] - t0 for ch, v in on.items()}

    full = res["test"]["stim"] > 0
    edge = max(cfg.clock_offsets) + cfg.slot
    checks = dict(
        on_time=full.sum(axis=1) * res["a1cfg"].dt,
        n_simul=full[:, edge:-edge].sum(axis=0),
        n_simul_bg=full[cfg.n_token_channels:, edge:-edge].sum(axis=0),
        # Lag from each tone of a word to the next, over the whole stream.
        lags={(w, p): (on[word[p + 1]] - on[word[p]]) * res["a1cfg"].dt * 1e3
              for w, word in enumerate(WORDS)
              for p in range(WORD_LEN - 1)},
    )

    from model0.model import _build_inh_matrices
    M_EI, M_IE = _build_inh_matrices(res["a1cfg"])

    return dict(checks=checks, cfg=cfg, dt=res["a1cfg"].dt,
                stim=res["test"]["stim"][:, t0:t1],
                E=res["E_test"][:, t0:t1],
                y=res["te"]["y"][:, t0:t1],
                onsets=keep, t0=t0,
                masks=res["l2"].M_true.copy(),
                tau=np.asarray(res["l2"].tau),
                rows=unit_table(res),
                W1=None if res["W1"] is None else np.asarray(res["W1"]).copy(),
                M_EI=M_EI.copy(), M_IE=M_IE.copy())


if __name__ == "__main__":
    main()
