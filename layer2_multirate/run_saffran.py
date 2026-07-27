"""
layer2_multirate.run_saffran
============================

The Saffran paradigm again, but layer 2 now reads a **filterbank of rates**.

The question this run exists to answer
--------------------------------------
With a single trace the layer learned transitions, not words: a three token
word has two informative moments and a mask could only hold one predecessor, so
ABC fragmented into "B after A" and "C after B" across two units, and zero of
eight units spanned a whole word.

A bank of rates should remove that limit.  At the moment C fires, B is recent
at fast rates and A survives only at slow ones, so one mask can hold

    C now   +   B at a fast rate   +   A at a slow rate

which is ABC in a single unit.  The four learning rules are unchanged; only the
representation is wider.

The test is therefore not accuracy, which the single rate version already
passed by counting transitions.  It is whether a unit's mask contains **two
distinct predecessors at two different rates, in the order the word has them**.

Run
---
    python -m layer2_multirate.run_saffran
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model0 import selective_inh
from layer2_multirate.config import MRConfig
from layer2_multirate.layer2 import Layer2MR
from layer2_syllable.run_ab_ba import LAYER1, layer1_rates
from layer2_syllable.stimulus import build_stream, chunk_windows
from layer2_syllable.saffran.run_saffran import (GAP, N_CH, TONE_DUR, WORDS,
                                                 WORD_NAME, auc,
                                                 boundary_bigrams, part_words,
                                                 within_bigrams, word_order)

N_TRAIN_WORDS = 500
N_TEST_REPS = 12


# ---------------------------------------------------------------------
def expose(cfg, a1cfg, seed=0, n_words=N_TRAIN_WORDS, record_every=0):
    rng = np.random.default_rng(1000 + seed)
    order = word_order(len(WORDS), n_words, rng)
    st = build_stream(WORDS, [0.25] * 4, n_words, N_CH, a1cfg.dt,
                      tone_dur=TONE_DUR, intra_gap=GAP, inter_gap=GAP,
                      seed=seed, order=order)
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=seed)
    l2 = Layer2MR(N_CH, cfg)
    tr = l2.run(E, a1cfg.dt, learn=True, record_every=record_every)
    return l2, st, tr


def word_of(ch):
    """(word index, position) of a channel, or (None, None)."""
    for wi, w in enumerate(WORDS):
        if ch in w:
            return wi, w.index(ch)
    return None, None


def analyse(l2):
    """What did each unit learn, and does any unit span a whole word?"""
    tau = l2.tau
    wb, bb = within_bigrams(WORDS), boundary_bigrams(WORDS)
    rows = []
    for u in np.flatnonzero(l2.committed):
        M = l2.M[u]                                  # (N, N, R)
        tot = float(M.sum())
        i = int(np.argmax(M.sum(axis=(1, 2))))       # the channel firing NOW
        row = M[i]                                   # (N, R) predecessors x rates

        # strongest predecessor, and the strongest of a DIFFERENT channel
        j1, m1 = np.unravel_index(int(np.argmax(row)), row.shape)
        other = row.copy(); other[j1, :] = 0.0
        j2, m2 = np.unravel_index(int(np.argmax(other)), other.shape)

        w1 = float(row[j1, m1] / tot) if tot > 0 else 0.0
        w2 = float(row[j2, m2] / tot) if tot > 0 else 0.0

        # does (j2, j1, i) appear in a word, in that order?
        spans = any(tuple(w) == (int(j2), int(j1), int(i)) for w in WORDS)
        # is the top pair a within word transition at all?
        cls = ("within" if (int(j1), int(i)) in wb
               else ("boundary" if (int(j1), int(i)) in bb else "other"))
        wi, pos = word_of(int(i))
        rows.append(dict(unit=int(u), now=int(i), p1=int(j1), p2=int(j2),
                         tau1=float(tau[m1]), tau2=float(tau[m2]),
                         w1=w1, w2=w2, cls=cls, spans=bool(spans),
                         word=wi, pos=pos,
                         slower=bool(tau[m2] > tau[m1])))
    n = len(rows)
    spanning = [r for r in rows if r["spans"]]
    return dict(rows=rows, n_units=n,
                n_within=sum(r["cls"] == "within" for r in rows),
                n_span=len(spanning),
                words_spanned=sorted({r["word"] for r in spanning
                                      if r["word"] is not None}),
                frac_within=(sum(r["cls"] == "within" for r in rows) / n) if n else 0.0)


def test_items(l2, a1cfg, seed=0):
    pws = part_words(WORDS)
    items = [w for w in WORDS] + [p for p, _ in pws]
    kind = np.array(["word"] * len(WORDS) + ["part word"] * len(pws))
    rng = np.random.default_rng(77 + seed)
    order = np.concatenate([rng.permutation(len(items))
                            for _ in range(N_TEST_REPS)])
    st = build_stream(items, [1.0] * len(items), len(order), N_CH, a1cfg.dt,
                      tone_dur=TONE_DUR, intra_gap=GAP, inter_gap=0.500,
                      seed=seed, order=order)
    E, _ = layer1_rates(st["stim"], a1cfg, mode="full", seed=seed)
    l2.reset_state()
    te = l2.run(E, a1cfg.dt, learn=False)
    wins = chunk_windows(st, pad_s=0.05)
    comm = np.flatnonzero(l2.committed)
    pop = np.array([te["y"][comm, a:b].max(axis=1).sum() for a, b in wins])
    per = np.array([te["y"][:, a:b].max(axis=1) for a, b in wins])
    k = kind[order]
    a = auc(pop[k == "word"], pop[k == "part word"])
    return dict(pop=pop, per_unit=per, kind=k, stream=st, te=te,
                auc=max(a, 1.0 - a), items=items, order=order)


# ---------------------------------------------------------------------
def main(argv=None):
    a1cfg = selective_inh(N=N_CH, **LAYER1)
    cfg = MRConfig()
    print("[ layer2_multirate ] Saffran, layer 2 with a filterbank of rates")
    print(f"  rates (s): {', '.join(f'{t:.3f}' for t in cfg.rates)}")
    print(f"  units available: {cfg.n_units}   channels: {N_CH}   "
          f"mask shape per unit: ({N_CH}, {N_CH}, {len(cfg.rates)})")
    print(f"  exposure: {N_TRAIN_WORDS} words\n")

    l2, st, tr = expose(cfg, a1cfg, seed=0)
    an = analyse(l2)
    te = test_items(l2, a1cfg, seed=0)

    print(f"  committed units: {an['n_units']} of {cfg.n_units}")
    print(f"  units whose mask spans a whole word: {an['n_span']}")
    print(f"  words covered by a spanning unit: "
          f"{[WORD_NAME[w] for w in an['words_spanned']]}\n")
    print("  unit  now  <- p1 (tau)      <- p2 (tau)      w1    w2   class     spans")
    for r in an["rows"]:
        print(f"   {r['unit']:3d}   {r['now']:2d}   "
              f"{r['p1']:2d} ({r['tau1']*1e3:4.0f} ms)   "
              f"{r['p2']:2d} ({r['tau2']*1e3:4.0f} ms)   "
              f"{r['w1']:.2f}  {r['w2']:.2f}  {r['cls']:8s}  "
              f"{'YES' if r['spans'] else '.'}")
    print(f"\n  word vs part word: ROC {te['auc']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
