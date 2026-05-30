"""
tasks.syllable.config
======================

Configuration for the Roving Oddball paradigm on model0 driven by
*syllables* instead of pure tones.

Relation to ``tasks.roving``
----------------------------
The protocol is identical to the roving-oddball task (blocks of one
repeated 3-syllable word, constrained-random block order, plasticity on
throughout).  The only difference is the *stimulus*: each syllable is no
longer a single tonotopic channel but a **figure** -- a spectral pattern
of activation spread across many channels with possibly different
amplitudes per channel.

Figures ("syllable spectrograms")
---------------------------------
There are ``n_channels`` (default 35) tonotopic channels spanning a
log-frequency axis in [0, 1].  Each syllable is the sum of a few
*formants*: Gaussian energy bumps centred at fractional positions along
that axis, each with its own relative amplitude.  Syllables deliberately
**share some formants and differ in others** -- exactly as real speech
syllables share spectral structure (e.g. a common vowel formant) while
differing elsewhere -- so their figures overlap partially in channel
space.  See ``SYLLABLE_FORMANTS`` for the layout.

Vocabulary
----------
Five syllables A, B, C, D, E.  As in roving, A and B are the *shared*
syllables (they recur in every word) and C, D, E are the *variable*
(deviant) syllables.  With ``deviant_syllable_pos=3`` the three word
types are ABC / ABD / ABE.  (The user described "A B C D E"; the
shared/variable split mirrors the roving tone vocabulary one-to-one.)

Timing
------
``syll_dur = 180 ms`` per syllable, ``syll_gap = 0`` inside a sequence,
``seq_gap = 1000 ms`` of silence between sequences.  A trial therefore
spans ``3*180 + 0 + 1000 = 1540 ms``.

Model parameters are *not* in this config -- they live in
``model0.A1Config`` and are passed separately to ``run_experiment``.

References
----------
- Bekinschtein, Dehaene, Rohaut, Tadel, Cohen, Naccache (2009) PNAS
  106:1672 -- local-global / roving oddball framework.
- Mesgarani, Cheung, Johnson, Chang (2014) Science 343:1006 -- spectro-
  temporal receptive fields and the spectral structure of speech
  syllables in human STG, motivating the formant-overlap figure model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


# =====================================================================
#  Word generation
# =====================================================================
SHARED_SYLLABLES = ("A", "B")
VARIABLE_SYLLABLES = ("C", "D", "E")


def _make_words(deviant_syllable_pos: int) -> Tuple[str, ...]:
    """Three word types for a given deviant-syllable position.

    The deviant position is filled by C, D, or E (one per word); the two
    non-deviant positions are filled with A and B in order.

    >>> _make_words(3)
    ('ABC', 'ABD', 'ABE')
    >>> _make_words(2)
    ('ACB', 'ADB', 'AEB')
    >>> _make_words(1)
    ('CAB', 'DAB', 'EAB')
    """
    if deviant_syllable_pos not in (1, 2, 3):
        raise ValueError(
            f"deviant_syllable_pos must be 1, 2 or 3; got "
            f"{deviant_syllable_pos}")
    shared_positions = [p for p in (1, 2, 3) if p != deviant_syllable_pos]
    words = []
    for var_syll in VARIABLE_SYLLABLES:
        slots = [""] * 3
        slots[deviant_syllable_pos - 1] = var_syll
        for s_idx, p in enumerate(shared_positions):
            slots[p - 1] = SHARED_SYLLABLES[s_idx]
        words.append("".join(slots))
    return tuple(words)


# =====================================================================
#  Syllable figures (spectral activation patterns)
# =====================================================================
# Each syllable is a list of formants ``(centre_fraction, amplitude)``.
# ``centre_fraction`` is the formant's position along the log-frequency
# axis in [0, 1] (so the layout is independent of ``n_channels``);
# ``amplitude`` is its relative strength before per-figure peak
# normalisation.
#
# The layout is hand-designed so that syllables share some formants and
# differ in others (the shared channels are noted on the right):
#
#     A: 0.18  0.42  0.70
#     B: 0.18  0.54  0.82          (0.18 shared with A)
#     C: 0.28  0.42  0.88          (0.42 shared with A)
#     D: 0.28  0.60  0.70          (0.28 shared with C; 0.70 shared with A)
#     E: 0.36  0.54  0.88          (0.54 shared with B; 0.88 shared with C)
#
# This gives every syllable both a private formant and at least one
# formant it shares with another syllable -- the partial spectral
# overlap characteristic of real speech.
SYLLABLE_FORMANTS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "A": ((0.18, 1.0), (0.42, 0.7), (0.70, 0.5)),
    "B": ((0.18, 0.9), (0.54, 0.8), (0.82, 0.5)),
    "C": ((0.28, 1.0), (0.42, 0.6), (0.88, 0.6)),
    "D": ((0.28, 0.8), (0.60, 0.9), (0.70, 0.5)),
    "E": ((0.36, 1.0), (0.54, 0.7), (0.88, 0.6)),
}


# =====================================================================
#  Configuration dataclass
# =====================================================================
@dataclass(frozen=True)
class SyllableConfig:
    """Paradigm parameters for a syllable-roving session on model0."""

    name: str = "default"

    # ---- Spectral channels ----
    n_channels: int = 35

    # ---- Timing (ms; assumes A1Config.dt == 1 ms) ----
    syll_dur: int = 180
    syll_gap: int = 0
    seq_gap: int = 1000

    # ---- Per-trial analysis padding kept inside the recorded window ----
    pre_stim_ms: int = 100
    post_stim_ms: int = 200

    # ---- Protocol ----
    n_sylls_per_seq: int = 3
    deviant_syllable_pos: int = 3
    words_override: Optional[Tuple[str, ...]] = None
    n_blocks_per_word: int = 10        # 3 words * 10 = 30 blocks
    n_reps_per_block: int = 15         # 30 blocks * 15 = 450 trials

    # ---- Syllable vocabulary ----
    syllables: Tuple[str, ...] = ("A", "B", "C", "D", "E")

    # ---- Figure (spectral pattern) parameters ----
    formant_width: float = 1.5         # Gaussian sigma, in channels
    active_thresh: float = 0.2         # fraction-of-peak cut for "active"
    syll_amp: float = 1.0              # global gain applied to every figure

    # ---- Reproducibility ----
    seed: int = 42

    # ============ Derived properties ============
    @property
    def words(self) -> Tuple[str, ...]:
        """Word types in this session.  Auto-generated from
        ``deviant_syllable_pos`` unless ``words_override`` is set."""
        if self.words_override is not None:
            return tuple(self.words_override)
        return _make_words(self.deviant_syllable_pos)

    @property
    def n_syllables(self) -> int:
        return len(self.syllables)

    @property
    def n_blocks(self) -> int:
        return len(self.words) * self.n_blocks_per_word

    @property
    def n_total_seqs(self) -> int:
        return self.n_blocks * self.n_reps_per_block

    @property
    def stim_steps(self) -> int:
        """Stimulus-only steps (3 syllables with intra-sequence gaps)."""
        n = self.n_sylls_per_seq
        return n * self.syll_dur + max(0, n - 1) * self.syll_gap

    @property
    def trial_period(self) -> int:
        """One trial = stimulus + inter-sequence silence (1540 ms default)."""
        return self.stim_steps + self.seq_gap

    @property
    def epoch_steps(self) -> int:
        """Per-trial recorded window length (pre + stim + post)."""
        return self.pre_stim_ms + self.stim_steps + self.post_stim_ms

    @property
    def syll_onsets_in_epoch(self) -> Tuple[int, ...]:
        """Sample indices of syllable onsets inside the recorded epoch
        (already offset by ``pre_stim_ms``)."""
        slot = self.syll_dur + self.syll_gap
        return tuple(self.pre_stim_ms + i * slot
                     for i in range(self.n_sylls_per_seq))

    @property
    def deviant_syllable_index(self) -> int:
        """0-based index of the deviant syllable within a sequence."""
        return self.deviant_syllable_pos - 1

    @property
    def shared_syllables(self) -> Tuple[str, ...]:
        return SHARED_SYLLABLES

    @property
    def variable_syllables(self) -> Tuple[str, ...]:
        return VARIABLE_SYLLABLES

    # ---- Figures ----
    def figure(self, syllable: str) -> np.ndarray:
        """Spectral activation pattern for ``syllable``: shape (n_channels,).

        The figure is a sum of Gaussian formant bumps, peak-normalised to
        1 and then scaled by ``syll_amp`` so every syllable delivers a
        comparable peak drive regardless of how many formants it has.
        """
        if syllable not in SYLLABLE_FORMANTS:
            raise ValueError(
                f"No formant layout for syllable {syllable!r}; "
                f"known: {tuple(SYLLABLE_FORMANTS)}")
        ch = np.arange(self.n_channels, dtype=float)
        fig = np.zeros(self.n_channels)
        for frac, amp in SYLLABLE_FORMANTS[syllable]:
            centre = frac * (self.n_channels - 1)
            fig += amp * np.exp(-0.5 * ((ch - centre) / self.formant_width) ** 2)
        peak = fig.max()
        if peak > 0:
            fig = fig / peak
        return self.syll_amp * fig

    @property
    def figure_matrix(self) -> np.ndarray:
        """All syllable figures stacked column-wise: (n_channels, n_syllables)."""
        return np.stack([self.figure(s) for s in self.syllables], axis=1)

    def active_channels(self, syllable: str) -> np.ndarray:
        """Channels driven above ``active_thresh`` * peak by ``syllable``."""
        fig = self.figure(syllable)
        peak = fig.max()
        if peak <= 0:
            return np.empty(0, dtype=int)
        return np.where(fig >= self.active_thresh * peak)[0]

    def word_channels(self, word: str) -> np.ndarray:
        """Union of active channels across all syllables in ``word``."""
        chans = set()
        for s in word:
            chans.update(self.active_channels(s).tolist())
        return np.array(sorted(chans), dtype=int)

    def replace(self, **kw) -> "SyllableConfig":
        return dataclasses.replace(self, **kw)

    def __post_init__(self):
        if self.n_channels < 2:
            raise ValueError(f"n_channels must be >= 2; got {self.n_channels}")
        # Validate the word machinery and that every syllable has a figure.
        _ = _make_words(self.deviant_syllable_pos)
        for s in self.syllables:
            if s not in SYLLABLE_FORMANTS:
                raise ValueError(
                    f"Syllable {s!r} has no entry in SYLLABLE_FORMANTS "
                    f"(known: {tuple(SYLLABLE_FORMANTS)}).")


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> SyllableConfig:
    """Canonical: 30 blocks (10 per word) x 15 reps = 450 trials, deviant pos 3."""
    return SyllableConfig(**kw)


def short(**kw) -> SyllableConfig:
    """Faster: 15 blocks (5 per word) x 15 reps = 225 trials."""
    return SyllableConfig(name="short", n_blocks_per_word=5, **kw)


def long_(**kw) -> SyllableConfig:
    """Longer: 90 blocks (30 per word) x 15 reps = 1350 trials."""
    return SyllableConfig(name="long", n_blocks_per_word=30, **kw)


def deviant_pos2(**kw) -> SyllableConfig:
    """Deviant at position 2: words ACB / ADB / AEB."""
    return SyllableConfig(name="deviant_pos2", deviant_syllable_pos=2, **kw)


def deviant_pos1(**kw) -> SyllableConfig:
    """Deviant at position 1: words CAB / DAB / EAB."""
    return SyllableConfig(name="deviant_pos1", deviant_syllable_pos=1, **kw)


PRESETS: Dict[str, Callable[..., SyllableConfig]] = {
    "default":      default,
    "short":        short,
    "long":         long_,
    "deviant_pos2": deviant_pos2,
    "deviant_pos1": deviant_pos1,
}


def get_preset(name: str, **overrides) -> SyllableConfig:
    """Look up a preset by name with optional parameter overrides."""
    if name not in PRESETS:
        avail = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available: {avail}")
    return PRESETS[name](**overrides)
