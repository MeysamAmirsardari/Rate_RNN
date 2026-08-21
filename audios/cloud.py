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


def channels(exclude_st: Sequence[float] = (), *, f_ref: float = F_REF,
             st_lo: float = ST_LO, st_hi: float = ST_HI,
             guard_st: float = 0.5, grid_st: float = 1.0,
             exclude_hz: Sequence[float] = ()) -> np.ndarray:
    """Semitone grid around the reference, minus everything near a target.

    ``guard_st`` is how close a cloud channel may come to a target: nothing
    within that many semitones survives.  It is a real design parameter, not
    a rounding tolerance.  Too small and the cloud sits on top of the figure
    and masks it; too large and the cloud is spectrally elsewhere, so
    segregating the figure stops being a task.
    """
    ex = list(exclude_st) + [12.0 * np.log2(f / f_ref) for f in exclude_hz]
    grid = np.arange(st_lo, st_hi + grid_st / 2.0, grid_st)
    keep = np.array([not any(abs(s - e) < guard_st for e in ex) for s in grid])
    return f_ref * 2.0 ** (grid[keep] / 12.0)


def deal(n_slots: int, n_ch: int, rng: np.random.Generator,
         n_voices: int = N_VOICES) -> np.ndarray:
    """Channel index for every (slot, voice).

    Dealt from a pack so the per-channel counts stay level, and swapped
    within the pack -- never redrawn -- when the adjacent-slot rule bites, so
    the counts survive the constraint exactly.
    """
    out = np.empty((n_slots, n_voices), dtype=int)
    pack = rng.permutation(n_ch)
    i = 0
    prev: set = set()
    for s in range(n_slots):
        group: list[int] = []
        for _ in range(n_voices):
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
          seed: int = 0, *, tone_ms: float = TONE_MS,
          slot_ms: float = SLOT_MS, n_voices: int = N_VOICES,
          complex_tones: bool = False, **chan_kw) -> tuple:
    """A cloud of ``total_samples``; returns the signal and its channel use.

    Concurrency is ``n_voices - 1`` and is exactly constant provided
    ``tone_ms`` is ``(n_voices - 1) / n_voices`` of ``slot_ms``: each voice is
    then silent for one slot-fraction and the silences interleave, so the
    count never moves.  Density is set by ``n_voices``, not by leaving gaps.
    """
    freqs = channels(exclude_st, **chan_kw)
    rng = np.random.default_rng(seed)
    slot = core.samples(slot_ms)
    step = slot // n_voices
    n_slots = int(np.ceil(total_samples / slot)) + 1
    idx = deal(n_slots, freqs.size, rng, n_voices)

    make = core.complex_tone if complex_tones else core.tone
    pips = [make(f, tone_ms) for f in freqs]
    x = np.zeros(n_slots * slot + (n_voices - 1) * step + pips[0].size)
    for s in range(n_slots):
        for v in range(n_voices):
            o = s * slot + v * step
            p = pips[idx[s, v]]
            x[o:o + p.size] += p
    return x[:total_samples], freqs, idx


def report(x: np.ndarray, freqs: np.ndarray, idx: np.ndarray, *,
           tone_ms: float = TONE_MS, slot_ms: float = SLOT_MS,
           n_voices: int = N_VOICES) -> str:
    """Check the two uniformities on the rendered cloud, not on the plan."""
    counts = np.bincount(idx.ravel(), minlength=freqs.size)
    edge = core.samples(slot_ms) * 2

    # how many tones sound at once: count voices whose tone covers each sample
    slot, step = core.samples(slot_ms), core.samples(slot_ms) // n_voices
    n_tone = core.samples(tone_ms)
    cover = np.zeros(x.size + slot, dtype=int)
    for s in range(idx.shape[0]):
        for v in range(idx.shape[1]):
            o = s * slot + v * step
            cover[o:o + n_tone] += 1
    inner = cover[edge:x.size - edge]

    return (f"  cloud: {freqs.size} channels, {freqs[0]:.0f}-{freqs[-1]:.0f} Hz "
            f"(semitone grid), {idx.size} tones\n"
            f"         concurrency {inner.min()}-{inner.max()} "
            f"(constant at {n_voices - 1} by construction); "
            f"per-channel use {counts.min()}-{counts.max()} "
            f"(spread {counts.max() - counts.min()})")


def concurrency(onsets, total: int, n_tone: int) -> np.ndarray:
    """How many tones of length ``n_tone`` sound at each sample."""
    c = np.zeros(total + n_tone, dtype=int)
    for o in onsets:
        c[int(o):int(o) + n_tone] += 1
    return c


def schedule(total: int, freqs: np.ndarray, fig, n_total: int, *,
             tone_ms: float = TONE_MS, step_ms: float = 2.5,
             guard_ms: float | None = None, seed: int = 5) -> dict:
    """A cloud that fills the figure's complement, so the total never moves.

    ``fig`` is the figure as ``(channel index, onset sample)`` pairs -- the
    channel, not just the time, because **the cloud sounds the figure's own
    frequencies too**.  It has to.  If a channel were reserved for the figure
    then every tone in it would be a figure tone, and the figure could be
    picked out by frequency alone without the timing ever being used, which is
    the one thing a timing experiment cannot allow.  Sharing the channels
    makes frequency uninformative and leaves the pattern as the only cue.

    Two consequences follow, and both are handled here rather than hoped for:

    * A cloud tone must not land on a figure tone in the same channel, so the
      figure's occupancy is marked busy before any dealing starts.
    * A figure channel starts the run already partly used, so its figure tones
      are counted toward its total and the dealer works down the **total**
      count.  Channels are picked least-used first with random tie-breaking,
      which equalises figure and cloud channels instead of leaving figure
      channels conspicuously rare or conspicuously common.

    ``guard_ms`` keeps a channel silent for that long either side of its own
    tones, so two tones in one channel can never abut into a single longer
    one; it defaults to one tone length.

    Filling is greedy and deliberately conservative: a tone is added only if
    it cannot push the total above ``n_total`` anywhere in its own length.
    That guarantees the ceiling is never crossed and costs a shallow dip just
    before each burst, where tones already sounding cannot be withdrawn.

    ``n_total`` must be at least the figure's own peak, because nothing the
    cloud does can take a tone away from the figure.  Pass the peak taken
    across **every condition of a series**, not each condition's own, or the
    density becomes a cue that co-varies with the manipulation.
    """
    n_tone = core.samples(tone_ms)
    guard = n_tone if guard_ms is None else core.samples(guard_ms)
    fig = [(int(k), int(o)) for k, o in fig]

    c_tot = concurrency([o for _, o in fig], total, n_tone)
    busy = np.zeros((freqs.size, total + n_tone + guard + 1), dtype=bool)
    counts = np.zeros(freqs.size, dtype=int)
    for k, o in fig:
        busy[k, max(0, o - guard):o + n_tone + guard] = True
        counts[k] += 1
    n_fig = counts.copy()

    rng = np.random.default_rng(seed)
    ev = []
    for o in range(0, total, core.samples(step_ms)):
        w = slice(o, o + n_tone)
        wg = slice(max(0, o - guard), o + n_tone + guard)
        while c_tot[w].max() < n_total:
            # least-used first, random among equals: the counts stay level and
            # the order stays unpredictable, which is what the cloud is for
            for k in np.lexsort((rng.random(freqs.size), counts)):
                if not busy[k, w].any():
                    break
            else:
                break
            busy[k, wg] = True
            c_tot[w] += 1
            counts[k] += 1
            ev.append((freqs[k], o))

    x = np.zeros(total + n_tone)
    pips: dict = {}
    for f, o in ev:
        if f not in pips:
            pips[f] = core.tone(f, tone_ms)
        x[o:o + n_tone] += pips[f]
    return dict(x=x[:total], freqs=freqs, events=ev, counts=counts,
                cloud_counts=counts - n_fig, fig_counts=n_fig)
