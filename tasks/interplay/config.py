"""
tasks.interplay.config
======================

Statistical word learning carried out *in a competing background* -- the
Saffran paradigm crossed with stochastic figure-ground.

The question
------------
In figure-ground, a set of channels is bound into a figure because they
fire TOGETHER: coherence is simultaneous, and the recurrent weights that
grow between co-active channels amplify the figure over the ground.  In
Saffran, a set of channels belongs to the same word because they fire IN
ORDER: coherence is sequential.

If the model's recurrent plasticity is what binds a figure, then the
sequential case should work the same way, and a stream whose next token
is PREDICTABLE should segregate from a background whose next token is not
-- with no simultaneity anywhere in the stimulus.  That is the hypothesis
this task tests: **temporal predictability is a grouping cue, and the
enhancement it produces is the same enhancement figure-ground produces.**

Enhancement, not suppression
----------------------------
The same learned weight can do opposite things depending on where its
current is delivered, and this task deliberately uses the other regime
from ``tasks.double_tone``.

    A_pred  > 0   prediction delivered to I  -> the predicted channel is
                  SUPPRESSED.  This is expectation suppression, the MMN
                  regime, and it is what double_tone needs.
    A_rec   > 0   prediction delivered to E  -> the predicted channel is
                  EXCITED.  This is assembly binding, the figure-ground
                  regime, and it is what this task needs.

So ``A_rec > 0`` and ``A_pred = 0`` here.  A word is a temporal assembly:
each token excites the next, the excitation accumulates along the word,
and the whole stream rides above a background that never predicts itself.

Design
------
``n_figure`` channels (12) carry four three-token words; ``n_background``
channels (12) carry the competing stream.  The two pools are disjoint, so
"figure" and "background" are defined by the stimulus and never by the
model.

Timing is chosen so the prediction can actually reach the next token.
A token is 50 ms with a 30 ms gap, an 80 ms slot, and ``tau_trace`` is
100 ms -- so a token's trace is still at e^-0.8 = 45% of its peak when
the next token starts.  At the 30 ms default trace of ``A1Config`` it
would be at e^-2.7 = 7% and nothing would bind.  The stream is perfectly
isochronous (the gap inside a word equals the gap across a boundary), so
no timing cue marks a word boundary and only the transition statistics
can do it:

    inside a word     P(next | current) = 1.0
    across a boundary P(next | current) = 1/3

The background
--------------
Exactly one background tone is on at every time point -- never two, never
none.  Background tones tile the timeline contiguously, and their channel
order is drawn as repeated random permutations of the background pool, so

    * total background activity is CONSTANT in time (no envelope cue that
      could be mistaken for structure), and
    * every background channel is on for the same total duration (no
      channel is a better or worse competitor than any other).

Both are enforced and asserted rather than assumed.

Drive matching
--------------
The figure pool is driven 50/80 = 62.5% of the time and the background
pool 100% of the time, so equal amplitudes would give the background 1.6x
the total thalamic drive and any "figure enhancement" would be measuring
that instead.  ``bg_level = 1.0`` scales the background amplitude to
equalise total drive per pool; the robustness sweep varies it around 1.0
deliberately, and the run asserts the match at 1.0.

Conditions
----------
A 2 x 2, because the hypothesis is an INTERACTION and not a main effect:

                     background off      background on
    structured       enhancement ref     the test case
    scrambled        control             control

``scrambled`` keeps every token's frequency identical and destroys only
the order, so it holds the spectrum and the drive fixed and removes just
the predictability.  If the hypothesis is right, structure buys more when
there is a background to segregate from than when there is not.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple

from model0 import A1Config


# =====================================================================
#  Paradigm vocabulary
# =====================================================================
#: Four three-token words over the twelve figure channels (Saffran 1996).
WORDS: Tuple[Tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5),
                                      (6, 7, 8), (9, 10, 11))
WORD_NAMES = ("W1", "W2", "W3", "W4")

STRUCTURES = ("structured", "scrambled")
BACKGROUNDS = (False, True)


# =====================================================================
#  Model constants
# =====================================================================
TAU_TRACE = 100e-3     # spans the 80 ms slot; see module docstring
W_MAX     = 0.35
W_NORM    = 3.0

#: Recurrent gain.  Measured stability ceiling, not a preference: at 1.0 the
#: peak rate jumps from 15 to 122 and at 2.0 the network diverges outright.
#: The fan-in here is 24 channels rather than the 2 of ``double_tone``, so
#: the usable gain is more than an order of magnitude lower.
A_REC     = 0.6

#: Forgetting rate, tau = 17 s.  This is the selectivity knob, and it works
#: for the same reason it does in roving: a within-word transition occurs
#: three times as often as any particular boundary transition, so the two
#: separate only if the weights forget fast enough to track the ratio
#: rather than accumulate towards a common ceiling.  Measured, structured
#: within/boundary weight ratio against the scrambled control:
#:
#:     W_decay   5e-3    2e-2    6e-2    1.5e-1
#:     structured  1.22    1.31    1.54     1.94
#:     scrambled   1.12    1.12    1.13     1.17
#:
#: 6e-2 is the balance: the ratio has separated clearly while the
#: segregation index has not yet started leaking into the control.
W_DECAY   = 6e-2

W_EI_SELF = 0.20
W_EI_LAT  = 0.05
W_IE_SELF = 0.65
W_IE_LAT  = 0.05       # weak, so the two pools compete without blanketing


def model_config(n_channels: int, **overrides) -> A1Config:
    """Selective-inhibition configuration for the interplay task."""
    cfg = A1Config(
        N=n_channels,
        multiscale_std=True,
        tau_trace=TAU_TRACE,
        recurrent_from_trace=True,
        A_rec=A_REC,
        A_pred=0.0,
        A_cancel=0.0,
        plastic_self=False,          # transitions only, never self-drive
        W_max=W_MAX,
        W_max_self=W_MAX,
        W_norm=W_NORM,
        W_decay=W_DECAY,
        w_EI_self=W_EI_SELF,
        w_EI_lat=W_EI_LAT,
        w_IE_self=W_IE_SELF,
        w_IE_lat=W_IE_LAT,
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def uniform_model_config(n_channels: int, **overrides) -> A1Config:
    """Row-sum-matched blanket-inhibition control."""
    n = n_channels
    ei = (W_EI_SELF + (n - 1) * W_EI_LAT) / n
    ie = (W_IE_SELF + (n - 1) * W_IE_LAT) / n
    return model_config(n_channels, w_EI_self=ei, w_EI_lat=ei,
                        w_IE_self=ie, w_IE_lat=ie, **overrides)


def frozen_model_config(n_channels: int, **overrides) -> A1Config:
    """Plasticity-free control: adaptation and inhibition, no learning.

    This is the floor every enhancement claim is measured against.  Any
    figure-over-background advantage that survives here is not prediction.
    """
    return model_config(n_channels, A_rec=0.0, **overrides)


# =====================================================================
#  Paradigm configuration
# =====================================================================
@dataclass(frozen=True)
class InterplayConfig:
    """Saffran statistical learning inside a competing background."""

    name: str = "default"

    # ---- Channels ----
    n_figure: int = 12
    n_background: int = 12

    # ---- Timing (ms) ----
    tone_dur: int = 50
    tone_gap: int = 30             # inside AND across words: isochronous

    # ---- Exposure ----
    n_words: int = 500
    allow_word_repeat: bool = False

    # ---- Background ----
    background: bool = True
    bg_level: float = 1.0          # 1.0 = drive-matched to the figure pool
    bg_no_immediate_repeat: bool = True

    # ---- Structure ----
    structure: str = "structured"  # or "scrambled"

    # ---- Amplitudes ----
    fig_amp: float = 1.0

    # ---- Analysis ----
    early_frac: float = 0.20       # first/last fraction used for buildup
    seed: int = 3

    # ============ Derived ============
    @property
    def n_channels(self) -> int:
        return self.n_figure + self.n_background

    @property
    def figure_channels(self) -> range:
        return range(0, self.n_figure)

    @property
    def background_channels(self) -> range:
        return range(self.n_figure, self.n_channels)

    @property
    def slot(self) -> int:
        """Token onset-to-onset interval."""
        return self.tone_dur + self.tone_gap

    @property
    def bg_dur(self) -> int:
        """Background tone duration -- the SAME as a figure token.

        Tied rather than set, so the two streams can never differ in tone
        length.  The background still covers the timeline completely
        because its tones abut with no gap, while the figure leaves
        ``tone_gap`` between its own; that difference in DUTY is what the
        drive matching in ``bg_amp`` corrects for.  What must not differ
        is the length of an individual tone, since that sets how much
        adaptation each tone accrues and would otherwise confound every
        figure-versus-background comparison.
        """
        return self.tone_dur

    @property
    def figure_duty(self) -> float:
        """Fraction of time the figure pool is driven at all."""
        return self.tone_dur / self.slot

    @property
    def bg_amp(self) -> float:
        """Background amplitude that equalises total drive per pool.

        The figure pool delivers ``fig_amp * figure_duty`` per unit time
        summed over channels; the background pool is on continuously and
        so delivers ``bg_amp``.  Equating them and scaling by bg_level:
        """
        return self.bg_level * self.fig_amp * self.figure_duty

    @property
    def n_tokens(self) -> int:
        return self.n_words * 3

    def replace(self, **kw) -> "InterplayConfig":
        return dataclasses.replace(self, **kw)


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> InterplayConfig:
    return InterplayConfig(**kw)


def short(**kw) -> InterplayConfig:
    """150 words, for iteration only."""
    return InterplayConfig(name="short", n_words=150, **kw)


def long_(**kw) -> InterplayConfig:
    return InterplayConfig(name="long", n_words=1200, **kw)


PRESETS = {"default": default, "short": short, "long": long_}


def get_preset(name: str, **overrides) -> InterplayConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. "
                         f"Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name](**overrides)


#: The 2 x 2 the hypothesis lives in.
def condition_grid() -> Tuple[Dict[str, object], ...]:
    return tuple(
        dict(structure=s, background=b,
             label=f"{s}/{'bg' if b else 'clean'}")
        for s in STRUCTURES for b in BACKGROUNDS
    )
