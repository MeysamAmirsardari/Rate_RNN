"""
tasks.interplay2.config
=======================

Two ordered tokens buried in a tone cloud: does layer 2 give each of them a
unit of its own?

The question
------------
``layer2_syllable`` showed that a population of coincidence units can learn an
ordered pair (B after A) when the pair is presented in isolation, separated by
half a second of silence.  Silence does most of the segmentation there: the
slow conductance decays to nothing between chunks, so at the onset of every
chunk the coincidence map is almost empty and the only structure in it is the
chunk itself.

This task removes that help.  The stream here is a **continuous tone cloud**
-- two background tones are on at every single sample, from the first to the
last, and there is no moment of silence anywhere.  Two ordered tokens are
hidden inside it:

    A then B   (B one slot after A)      on the integer clock
    C then D   (D one slot after C)      on a clock offset by half a slot

and sixteen further channels carry the cloud, with no ordered pair in them at
all.  Nothing marks a token: no gap, no amplitude change, no onset the cloud
does not also produce.  A token is defined purely by the fact that the pair
recurs while every background pair does not.

The layer has **four units and two tokens to find**, so the interesting
outcome is not "does it learn something" but *how it divides itself*:

    two units, one per token         the tokens are separate objects
    one unit for both                the layer merged them
    no unit on either                the cloud swamped the contingency

What could produce the answer for the wrong reason, and what stops it
---------------------------------------------------------------------
**Frequency.**  If A, B, C and D were on more often than a background
channel, the units would find them by rate and order would be irrelevant.
Every one of the twenty channels is therefore on for *exactly* the same total
time: one 100 ms tone per 800 ms block, a duty cycle of 12.5% for the token
channels and the background channels alike.  This is enforced by construction
and asserted at build time, not left to expectation.

**Simultaneity.**  If a token pair were also a co-active pair it could be
bound the way a figure-ground figure is bound, with no order involved.  The
two tones of a token never overlap; they abut.

**Onset density.**  If the token channels lived on a clock of their own while
the background sat on another, "token" and "predictable phase" would be
confounded.  The cloud is therefore split into two voices, one on each clock,
so AB has a background voice synchronous with it and so does CD.  Each token
is asynchronous with exactly one voice and with the other token.

**The contingency itself.**  ``shuffled`` keeps all of the above and drops one
thing: A, B, C and D still get one tone per block each, but at independently
drawn slots, so the lag from A to B is random in sign and size instead of
always +1 slot.  Duty, spectrum, simultaneity and onset density are untouched.
Anything that survives ``shuffled`` was never about prediction.

**Asynchrony between the tokens.**  ``sync`` moves CD onto the integer clock
and starts it with AB, so A and C are simultaneous and so are B and D.  The
four channels are then one object rather than two, and a layer that is really
allocating *per object* should collapse onto fewer units.  This is the
condition that says whether "two units" means anything.

Timing, and why the tone is 100 ms
----------------------------------
Tones tile the timeline with no gap, so the tone duration IS the lag inside a
token, and the slow conductance has to be large at that lag and small at every
other.  What sets how well the token stands out is the width of the
conductance kernel measured in slots -- the number of past channels still
carrying a trace when the second tone arrives.  Each of those is a competitor
for the same mask entry, and there are only sixteen background channels to
share them out between.

That ratio is scale-free: sweeping tone duration from 50 to 150 ms with the
kinetics scaled alongside gives the same contrast to three significant
figures (3.44 at ``tau_rise/tone = 0.36``, ``tau_decay/tone = 0.50``,
identically at every duration, with and without layer 1 in the loop).  The
duration is therefore free, and it is spent on making the kinetics
biological rather than on making the stimulus fast:

    tone 50 ms   ->  tau_rise 18 ms, tau_decay 25 ms   too fast to be NMDA
    tone 100 ms  ->  tau_rise 36 ms, tau_decay 50 ms   GluN2A-like

A 36 ms rise and a 50 ms decay are the kinetics of a GluN2A-dominated NMDA
current, which is what a coincidence subunit reading a slow afferent should
have.  Widening the kernel instead is not free: at ``tau_decay = 120 ms`` the
token pair is only 1.9 times a flat mask entry, because sixteen background
channels are inside the window at comparable amplitude, and the population
never differentiates.

Layer 1
-------
``model0`` at the shared configuration, unmodified, exactly as every other
paradigm uses it.  Layer 2 reads its excitatory rates.  The ``raw`` and
``frozen`` layer-1 modes are kept as controls so the contribution of the
cortical dynamics can be separated from the contribution of the readout.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple

from layer2_syllable.config import L2Config
from model0 import A1Config, shared_config


# =====================================================================
#  Channels
# =====================================================================
#: The two ordered tokens, as (first, second) channel indices.
A, B, C, D = 0, 1, 2, 3
TOKENS: Tuple[Tuple[int, int], ...] = ((A, B), (C, D))
TOKEN_NAMES = ("AB", "CD")
CHANNEL_NAMES = {A: "A", B: "B", C: "C", D: "D"}

#: Conditions.  See the module docstring for what each one removes.
CONDITIONS = ("paired", "shuffled", "sync")

#: Layer-1 modes, in increasing order of how much cortex is in the loop.
LAYER1_MODES = ("raw", "frozen", "full")


# =====================================================================
#  Layer 2
# =====================================================================
#: GluN2A-like: 36 ms rise, 50 ms decay, i.e. 0.36 and 0.50 of the tone.
#: See the module docstring -- the ratio to the tone is what matters and the
#: absolute values are then chosen to be biological.
TAU_RISE  = 36e-3
TAU_DECAY = 50e-3

#: Four units, two tokens.  The population is deliberately not larger than
#: twice the number of tokens: with eight units the question "did each token
#: get a unit" is easy, and with four it is a real competition.
N_UNITS = 4

#: The cloud never goes quiet, so the drive gate that mattered in the
#: syllable task (learn only above a fraction of peak drive) is almost always
#: open here.  It is kept, at its published value, because switching it off
#: would be a change of model rather than a change of stimulus.
GATE_FRAC = 0.15

#: An order of magnitude slower than the syllable task's 5e-3, and this is
#: the one parameter that had to move.  The instar rule makes each mask an
#: exponential moving average of the normalised coincidence map over the
#: moments that unit wins, with a memory of 1/eta winning steps.  With two
#: channels and silence between chunks that map is nearly empty and a short
#: memory is enough.  Here two background tones are on at every sample, so the
#: map has of order ten comparable entries at every moment and a short memory
#: leaves the mask tracking whichever background pair happened to be on.
#:
#: Measured, 400 blocks, layer 1 in the loop, the top mask entry of each unit:
#:
#:     eta      2e-3      1e-3      5e-4      2.5e-4
#:     result   nothing   nothing   AB + CD   AB + CD
#:
#: with the token entries reaching 10x a flat mask at 5e-4 and 15-20x at
#: 2.5e-4.  ``lam`` keeps the published ratio to ``eta`` (1/50), so the
#: pruning competition is unchanged; only the averaging window moved.
ETA = 2.5e-4
LAM = ETA / 50.0
COMMIT_FRAC = 0.20


def layer2_config(**overrides) -> L2Config:
    """Layer-2 population for this task."""
    cfg = L2Config(
        tau_rise=TAU_RISE,
        tau_decay=TAU_DECAY,
        no_self_pairs=True,
        n_units=N_UNITS,
        eta=ETA,
        lam=LAM,
        gate_frac=GATE_FRAC,
        commit_frac=COMMIT_FRAC,
        rho=0.0,
    )
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
class Interplay2Config:
    """A tone cloud with two ordered tokens hidden in it."""

    name: str = "default"

    # ---- Channels ----
    n_token_channels: int = 4      # A, B, C, D
    n_background: int = 16         # the cloud

    # ---- Timing (ms) ----
    tone_dur: int = 100            # tones abut, so this is also the slot
    block_slots: int = 8           # slots per block; one token of each per block

    # ---- Exposure ----
    n_blocks: int = 400            # 400 x 800 ms = 320 s, i.e. 400 of each token

    # ---- Design ----
    condition: str = "paired"

    # ---- Amplitudes ----
    #: One amplitude for every channel.  A separate token amplitude would be
    #: the loudness confound the duty matching exists to remove.
    amp: float = 1.0

    seed: int = 0

    # ============ Derived ============
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
        """Onset-to-onset interval.  Equal to ``tone_dur``: no gaps."""
        return self.tone_dur

    @property
    def offset(self) -> int:
        """Clock 1 lags clock 0 by half a slot.

        Half, rather than any other fraction, because it is the phase that
        is maximally far from both the preceding and the following clock-0
        onset -- the least ambiguous meaning of "asynchronous".
        """
        return self.slot // 2

    @property
    def n_voices(self) -> int:
        """Background voices: one per clock, so both tokens are treated alike."""
        return 2

    @property
    def bg_per_voice(self) -> int:
        """Background channels dealt to each voice in a block."""
        n, v = self.n_background, self.n_voices
        if n % v or self.block_slots != n // v:
            raise ValueError(
                f"{n} background channels over {v} voices must fill "
                f"{self.block_slots} slots each; got {n // v}")
        return n // v

    @property
    def block_samples(self) -> int:
        return self.block_slots * self.slot

    @property
    def duty(self) -> float:
        """On-time fraction, the same for all twenty channels."""
        return self.tone_dur / self.block_samples

    @property
    def n_tokens_each(self) -> int:
        return self.n_blocks

    def replace(self, **kw) -> "Interplay2Config":
        return dataclasses.replace(self, **kw)


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> Interplay2Config:
    return Interplay2Config(**kw)


def short(**kw) -> Interplay2Config:
    """100 blocks (80 s), for iteration only -- too short to learn."""
    return Interplay2Config(name="short", n_blocks=100, **kw)


def long_(**kw) -> Interplay2Config:
    return Interplay2Config(name="long", n_blocks=1000, **kw)


PRESETS = {"default": default, "short": short, "long": long_}


def get_preset(name: str, **overrides) -> Interplay2Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. "
                         f"Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name](**overrides)


def condition_grid() -> Tuple[Dict[str, object], ...]:
    return tuple(dict(condition=c, label=c) for c in CONDITIONS)
