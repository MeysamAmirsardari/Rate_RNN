"""
audios.cloud
============

An unpredictable tone cloud to put the targets inside.

Two kinds of uniformity, both enforced rather than hoped for
------------------------------------------------------------
**Uniform in time.**  The cloud has to be equally dense at every instant, or
its envelope becomes a cue in its own right and a target could be found by
where the background happens to thin out.  Five voices, staggered by a fifth
of the slot, with a tone lasting four fifths of the slot: each voice is on
80% of the time and their silences are evenly interleaved, so **exactly four
cloud tones sound at every instant**, with no fluctuation at all.

    slot 50 ms, voices at 0 10 20 30 40 ms, tone 40 ms  ->  4 on, always

**Uniform in frequency.**  Every cloud channel has to be used equally often,
or the rarer ones become salient and the commoner ones become a drone.
Channels are dealt from a pack that is reshuffled only when exhausted, so the
counts are equal to within a single pass, and the *order* stays random -- which
is what makes the cloud unpredictable while its statistics are flat.

**Unpredictable.**  There are no repeating pairs in the cloud: the pack is
reshuffled every pass, so no channel reliably follows any other.  That is the
whole point -- the targets are the only thing in the file with a recurring
temporal structure.

What the cloud is not allowed to do
-----------------------------------
It never plays a target frequency.  The channels are a semitone grid and the
target semitones are removed from it, so a target tone is never in doubt as to
whether it belongs to the figure; if the cloud could sound the same frequency,
"is that the target" would become a question about frequency rather than about
timing, which is the opposite of the design.

It also never plays the same channel in adjacent slots.  The voices are only
a fifth of a slot apart, so the same channel used twice nearby would run into
itself and sound as one long tone rather than two -- a duration cue the cloud
is not supposed to have.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

if __package__:
    from . import core
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audios import core  # type: ignore

TONE_MS = 40.0
SLOT_MS = 50.0
N_VOICES = 5                 # tone/slot = 4/5, so exactly 4 sound at once
F_REF = 1000.0
ST_LO, ST_HI = -12, 21       # 500 Hz to 3364 Hz, just under three octaves


def channels(exclude_st: Sequence[float] = ()) -> np.ndarray:
    """Semitone grid around the reference, minus the target semitones."""
    grid = np.arange(ST_LO, ST_HI + 1, dtype=float)
    keep = np.array([not any(abs(s - e) < 0.5 for e in exclude_st)
                     for s in grid])
    return F_REF * 2.0 ** (grid[keep] / 12.0)


def deal(n_slots: int, n_ch: int, rng: np.random.Generator) -> np.ndarray:
    """Channel index for every (slot, voice).

    Dealt from a pack so the per-channel counts stay level, and swapped
    within the pack -- never redrawn -- when the adjacent-slot rule bites, so
    the counts survive the constraint exactly.
    """
    out = np.empty((n_slots, N_VOICES), dtype=int)
    pack = rng.permutation(n_ch)
    i = 0
    prev: set = set()
    for s in range(n_slots):
        group: list[int] = []
        for _ in range(N_VOICES):
            if i >= pack.size:
                pack, i = rng.permutation(n_ch), 0
            j = i
            while j < pack.size and (pack[j] in prev or pack[j] in group):
                j += 1
            if j >= pack.size:
                j = i
            pack[i], pack[j] = pack[j], pack[i]
            group.append(int(pack[i]))
            i += 1
        out[s] = group
        prev = set(group)
    return out


def build(total_samples: int, exclude_st: Sequence[float] = (),
          seed: int = 0) -> tuple:
    """A cloud of ``total_samples``; returns the signal and its channel use."""
    freqs = channels(exclude_st)
    rng = np.random.default_rng(seed)
    n_slots = int(np.ceil(total_samples / core.samples(SLOT_MS))) + 1
    idx = deal(n_slots, freqs.size, rng)

    pips = [core.tone(f, TONE_MS) for f in freqs]
    slot = core.samples(SLOT_MS)
    step = slot // N_VOICES
    # room for the last slot's latest voice plus its whole tone
    x = np.zeros(n_slots * slot + (N_VOICES - 1) * step + pips[0].size)
    for s in range(n_slots):
        for v in range(N_VOICES):
            o = s * slot + v * step
            p = pips[idx[s, v]]
            x[o:o + p.size] += p
    return x[:total_samples], freqs, idx


def report(x: np.ndarray, freqs: np.ndarray, idx: np.ndarray) -> str:
    """Check the two uniformities on the rendered cloud, not on the plan."""
    counts = np.bincount(idx.ravel(), minlength=freqs.size)
    edge = core.samples(SLOT_MS) * 2
    env = core.envelope(x[edge:x.size - edge], win_ms=1.0)

    # how many tones sound at once: count voices whose tone covers each sample
    slot, step = core.samples(SLOT_MS), core.samples(SLOT_MS) // N_VOICES
    n_tone = core.samples(TONE_MS)
    cover = np.zeros(x.size + slot, dtype=int)
    for s in range(idx.shape[0]):
        for v in range(N_VOICES):
            o = s * slot + v * step
            cover[o:o + n_tone] += 1
    inner = cover[edge:x.size - edge]

    return (f"  cloud: {freqs.size} channels, {freqs[0]:.0f}-{freqs[-1]:.0f} Hz "
            f"(semitone grid), {idx.size} tones\n"
            f"         concurrency {inner.min()}-{inner.max()} "
            f"(constant at {N_VOICES - 1} by construction); "
            f"per-channel use {counts.min()}-{counts.max()} "
            f"(spread {counts.max() - counts.min()})")
