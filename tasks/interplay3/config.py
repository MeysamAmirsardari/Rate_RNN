"""
tasks.interplay3.config
=======================

Three four-tone words buried in a fifty-channel tone cloud, read out by the
**multi-rate** layer 2.  Does each word get a unit that spans the whole word?

Relation to interplay2
----------------------
``tasks.interplay2`` is the single-rate version and stays exactly as it is.
There, a token was two tones and one slow trace was enough: a mask entry
``M[B, A]`` says "B now, A recently", and that is the whole token.

A four-tone word has three transitions, and a single-rate layer fragments
across them -- it learns "B after A" and "C after B" separately and nothing
represents ABCD.  ``layer2_multirate`` exists for exactly this: a bank of
low-pass filters, because the *ratio of a channel's trace across rates encodes
how long ago it fired*, so one mask can hold

    D firing now  AND  C recent at a fast rate  AND  B at a slower one
                                                AND  A at the slowest

The learning rules are untouched; only the representation widens.  What this
task asks is whether that still works when the word is buried in a cloud that
never falls silent -- the condition ``layer2_multirate`` was never tested in,
since its Saffran stream had nothing else in it.

The stimulus
------------
Fifty channels.  Twelve of them carry three four-tone words:

    word 0   channels 0  1  2  3
    word 1   channels 4  5  6  7
    word 2   channels 8  9 10 11

There are three clocks, a third of a slot apart, and in every block the three
words sit on three different ones, so no two words can ever start together:
they are asynchronous with each other by construction.  Which clock a word
gets rotates by block, so word identity carries no phase either.  The
remaining thirty-eight channels carry the cloud.

Every one of the fifty channels is on for *exactly* the same total time -- one
50 ms tone per 19-slot block -- so a word cannot be found by rate.  Tones are
50 ms with a 10 ms gap on every channel alike, so the stream is isochronous and
the gap marks nothing.

Why nineteen slots
------------------
Not a preference: it is forced.  Exact duty matching needs the cloud to place
one tone per channel per block, so ``n_voices * block_slots`` must equal the
number of cloud channels, and thirty-eight factorises only as 2 x 19.  Two
voices of nineteen slots is the only option that keeps fifty channels total and
matches the duty exactly.  Three words of four tones then occupy twelve of the
nineteen slots, which is as dense as the words can be packed.

With two voices and three clocks, a voice cannot sit on every clock at once,
and the voices are the half that must stay put: a 50 ms tone in a 60 ms slot
leaves a hole that only an *adjacent* clock fills, so the voices are pinned to
clocks 0 and 1 and the words rotate past them instead.  Over any three blocks
each word is synchronous with a voice in exactly two, so no word is better or
worse accompanied than any other -- the property the per-clock voices bought
in interplay2, obtained by moving the other half of the stimulus.

Conditions
----------
``paired``    the test case
``shuffled``  each word's four channels land on four independently drawn
              slots instead of four consecutive ones, so duty, spectrum,
              simultaneity and onset density are untouched and only the order
              is gone
``sync``      all three words move onto clock 0 and start together, so the
              twelve channels are one object rather than three
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple

from layer2_multirate.config import MRConfig, default_rates
from model0 import A1Config, shared_config


# =====================================================================
#  Vocabulary
# =====================================================================
WORD_LEN = 4
N_WORDS = 3

#: Three four-tone words over the first twelve channels.
WORDS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(range(w * WORD_LEN, (w + 1) * WORD_LEN)) for w in range(N_WORDS))
WORD_NAMES = tuple(f"W{w + 1}" for w in range(N_WORDS))

#: Channel letters, for read-outs.  A-D is word 0, E-H word 1, I-L word 2.
CHANNEL_NAMES = {c: chr(ord("A") + c) for c in range(N_WORDS * WORD_LEN)}

CONDITIONS = ("paired", "shuffled", "sync")
LAYER1_MODES = ("raw", "frozen", "full")


# =====================================================================
#  Layer 2 -- multi-rate
# =====================================================================
#: Six log-spaced rates.  The span is what matters: the slowest must outlast a
#: whole word so the first tone is still legible when the last arrives, and the
#: fastest must separate the immediately preceding tone from the one before.
#: A word here is 4 x 60 = 240 ms, and ``layer2_multirate``'s own sweep found
#: 250-600 ms works for a 240 ms word while 80-150 ms is too short.  30-500 ms
#: -- the package default -- sits inside that, so it is kept unchanged.
RATES = default_rates(n=6, lo=0.030, hi=0.500)

#: Five units, three words.  Deliberately tight -- barely more than the
#: vocabulary -- so that "each word got a unit" is a real competition rather
#: than an inevitability, and so that the fifty channels cannot simply be
#: tiled one unit per channel.
N_UNITS = 5

#: Re-measured for this stimulus and this population size.  The package
#: default of 5e-3 was set on a stream with nothing in it but words; here
#: fifty channels are live, five cloud tones sound at every instant, and the
#: coincidence map has many comparable entries at every moment.  The right
#: value also depends on how many units there are, because ``eta`` is per
#: winning step and five units each win a fifth of the steps.
#:
#: Measured, 280 blocks, five units, layer 1 in the loop, seed 0:
#:
#:     eta                     1e-3   5e-4   2e-4   1e-4
#:     units on a word channel    1      4      5      4
#:     words spanned              0      1      2      1
#:
#: At 24 units on the sparser earlier version of this stimulus the optimum
#: was 5e-4; the population size moved it.  Re-measure it if either changes.
ETA = 2e-4
LAM = ETA / 50.0
GATE_FRAC = 0.15
COMMIT_FRAC = 0.20

#: Fraction of a mask row's peak that a predecessor must reach to count as
#: represented, in ``span_depth``.  The package value.
SPAN_THRESH = 0.25


def layer2_config(**overrides) -> MRConfig:
    """Multi-rate layer-2 population for this task."""
    cfg = MRConfig(rates=RATES, n_units=N_UNITS, eta=ETA, lam=LAM,
                   gate_frac=GATE_FRAC, commit_frac=COMMIT_FRAC)
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


# =====================================================================
#  Layer 1
# =====================================================================
def layer1_config(n_channels: int, inh: str = "selective",
                  **overrides) -> A1Config:
    """The shared ``model0`` configuration at this channel count."""
    return shared_config(n_channels, inh=inh, **overrides)


# =====================================================================
#  Paradigm
# =====================================================================
@dataclass(frozen=True)
class Interplay3Config:
    """Three four-tone words hidden in a fifty-channel cloud."""

    name: str = "default"

    # ---- Channels ----
    n_background: int = 38          # 12 word channels + 38 = 50

    # ---- Timing (ms) ----
    tone_dur: int = 50
    tone_gap: int = 10

    # ---- Block ----
    #: Slots per block, and how many times each word occurs in one.
    block_slots: int = 19
    words_per_block: int = 3

    #: Simultaneous cloud tones.  This is the density knob, and it is bounded
    #: by exact duty matching: ``n_voices * block_slots`` must equal
    #: ``n_background * words_per_block``, i.e. 6 x 19 = 38 x 3.
    n_voices: int = 6

    # ---- Exposure ----
    n_blocks: int = 280             # exposure: 280 x 1.14 s = 319 s

    #: The measurement stream is shorter than the exposure stream, because the
    #: two are asked for different things.  Exposure has to be long enough for
    #: the masks to settle; measurement only has to supply enough word
    #: instances for the response statistics, and seventy of each is already
    #: more than the read-out needs.  Running it at exposure length costs half
    #: the wall clock and buys nothing.
    n_test_blocks: int = 70

    # ---- Design ----
    condition: str = "paired"

    amp: float = 1.0
    seed: int = 0

    # ============ Derived ============
    @property
    def n_token_channels(self) -> int:
        return N_WORDS * WORD_LEN

    @property
    def n_channels(self) -> int:
        return self.n_token_channels + self.n_background

    @property
    def token_channels(self) -> range:
        return range(0, self.n_token_channels)

    @property
    def background_channels(self) -> range:
        return range(self.n_token_channels, self.n_channels)

    @property
    def slot(self) -> int:
        return self.tone_dur + self.tone_gap

    @property
    def n_clocks(self) -> int:
        """One clock per word, so no two words can start together."""
        return N_WORDS

    @property
    def clock_offsets(self) -> Tuple[int, ...]:
        """Onset offsets, evenly spread across one slot."""
        return tuple(k * self.slot // self.n_clocks
                     for k in range(self.n_clocks))

    @property
    def voice_offsets(self) -> Tuple[int, ...]:
        """Onset offsets of the cloud voices, evenly spread across one slot.

        Six voices ten milliseconds apart, with 50 ms tones in a 60 ms slot,
        means each voice is silent for one tenth of the cycle and the silences
        are staggered: **exactly five cloud tones are on at every instant**,
        with no fluctuation to be mistaken for structure.
        """
        return tuple(v * self.slot // self.n_voices
                     for v in range(self.n_voices))

    def _check_duty(self) -> None:
        """Exact duty matching relates density, block length and word rate."""
        lhs = self.n_voices * self.block_slots
        rhs = self.n_background * self.words_per_block
        if lhs != rhs:
            raise ValueError(
                f"n_voices * block_slots ({lhs}) must equal n_background * "
                f"words_per_block ({rhs}) for every channel to have the same "
                f"duty")

    def last_slot(self, clock: int) -> int:
        """Last slot on this clock whose tone still ends inside the block.

        A tone on a late clock starts up to two thirds of a slot after the
        nominal grid point, so the final slot of a block can run past the
        block edge.  For the cloud that is exactly what keeps the timeline
        tiled and is wanted.  For a word it is not: the last tone of one
        instance would overlap the first tone of the next, and two channels
        of the same word would be simultaneous -- which is the one thing a
        word must never be.  So word placement stops one slot earlier on the
        offset clocks.
        """
        usable = self.block_samples - self.tone_dur - self.clock_offsets[clock]
        return usable // self.slot

    def word_clock(self, word: int, block: int) -> int:
        """Which clock a word sits on in a given block.

        The words rotate through the clocks instead of owning one, so that
        word identity carries no phase at all: over any three blocks each
        word sits on each clock exactly once, and each word is synchronous
        with a cloud voice exactly two blocks in three.  Within a block the
        three words are still on three different clocks, so no two of them
        can ever start together -- which is the property that matters.
        """
        return (word + block) % self.n_clocks

    @property
    def block_samples(self) -> int:
        return self.block_slots * self.slot

    @property
    def duty(self) -> float:
        return self.tone_dur / self.block_samples

    def replace(self, **kw) -> "Interplay3Config":
        return dataclasses.replace(self, **kw)


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> Interplay3Config:
    return Interplay3Config(**kw)


def short(**kw) -> Interplay3Config:
    """150 blocks (171 s), for iteration only."""
    return Interplay3Config(name="short", n_blocks=150, n_test_blocks=50, **kw)


def long_(**kw) -> Interplay3Config:
    return Interplay3Config(name="long", n_blocks=1200, n_test_blocks=200, **kw)


PRESETS = {"default": default, "short": short, "long": long_}


def get_preset(name: str, **overrides) -> Interplay3Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. "
                         f"Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name](**overrides)


def condition_grid() -> Tuple[Dict[str, object], ...]:
    return tuple(dict(condition=c, label=c) for c in CONDITIONS)
