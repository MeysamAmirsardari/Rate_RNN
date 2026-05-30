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
There are ``n_channels`` (default 40) tonotopic channels.  Each syllable
activates exactly ``n_active`` (default 10) of them with heterogeneous
per-channel amplitudes.  All syllables share a common **core** of
``round(overlap * n_active)`` channels (default ``overlap = 0.6`` -> 6
shared channels = 60% pairwise overlap); the remaining channels are
*private* and disjoint across syllables.

The common-core structure is not a stylistic choice -- it is the only
way to make every pair of syllables overlap by the same high fraction.
Demanding 60-65% overlap with *all* other syllables (not just spectral
neighbours) forces a shared spectral region common to the whole set
(e.g. a common vowel formant), because a syllable's mere 10 channels
cannot otherwise reach that much overlap with four different partners at
once.  A tonotopic-shift layout, by contrast, only overlaps neighbours.

Per-channel amplitudes are drawn deterministically (from ``seed``) in
``[amp_min, amp_max]`` and then rescaled so every syllable has the
**same average amplitude over its active channels** (``syll_amp``).  See
``_figure_bank``.

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
#  Configuration dataclass
# =====================================================================
@dataclass(frozen=True)
class SyllableConfig:
    """Paradigm parameters for a syllable-roving session on model0."""

    name: str = "default"

    # ---- Spectral channels ----
    n_channels: int = 40

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
    n_active: int = 10                 # channels activated per syllable
    overlap: float = 0.6               # target pairwise overlap fraction
    amp_min: float = 0.4               # min per-channel amplitude (pre-norm)
    amp_max: float = 1.0               # max per-channel amplitude (pre-norm)
    syll_amp: float = 1.0              # mean amplitude over active channels

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
    @property
    def core_size(self) -> int:
        """Number of channels shared by every syllable (the common core)."""
        c = int(round(self.overlap * self.n_active))
        return max(0, min(c, self.n_active))

    def _figure_bank(self) -> np.ndarray:
        """Deterministically build the (n_channels, n_syllables) figure matrix.

        A common core of ``core_size`` channels is shared by every syllable;
        the remaining ``n_active - core_size`` channels of each syllable are
        private and disjoint across syllables, so every pair of syllables
        overlaps in exactly ``core_size`` channels.  Per-channel amplitudes
        are drawn in ``[amp_min, amp_max]`` and then rescaled so each
        syllable's mean amplitude over its active channels equals
        ``syll_amp`` (equal average drive across syllables).
        """
        rng = np.random.default_rng(self.seed)
        n_syll = self.n_syllables
        core_size = self.core_size
        private_size = self.n_active - core_size

        perm = rng.permutation(self.n_channels)
        core = perm[:core_size]
        pool = perm[core_size:]
        if private_size * n_syll > len(pool):
            raise ValueError(
                f"Cannot place {private_size} private channels for each of "
                f"{n_syll} syllables in {len(pool)} non-core channels; "
                f"reduce n_active/overlap or raise n_channels.")

        fmat = np.zeros((self.n_channels, n_syll))
        for j in range(n_syll):
            priv = pool[j * private_size:(j + 1) * private_size]
            chans = np.concatenate([core, priv]).astype(int)
            amps = rng.uniform(self.amp_min, self.amp_max, size=chans.size)
            amps *= self.syll_amp / amps.mean()      # equal average amplitude
            fmat[chans, j] = amps
        return fmat

    @property
    def figure_matrix(self) -> np.ndarray:
        """All syllable figures stacked column-wise: (n_channels, n_syllables)."""
        return self._figure_bank()

    def figure(self, syllable: str) -> np.ndarray:
        """Spectral activation pattern for ``syllable``: shape (n_channels,)."""
        return self._figure_bank()[:, self.syllables.index(syllable)]

    def active_channels(self, syllable: str) -> np.ndarray:
        """Channels this syllable activates (nonzero amplitude)."""
        return np.where(self.figure(syllable) > 0)[0]

    def word_channels(self, word: str) -> np.ndarray:
        """Union of active channels across all syllables in ``word``."""
        chans = set()
        for s in word:
            chans.update(self.active_channels(s).tolist())
        return np.array(sorted(chans), dtype=int)

    def overlap_matrix(self) -> np.ndarray:
        """Pairwise overlap fraction |S_i ∩ S_j| / n_active for all syllables."""
        sets = [set(self.active_channels(s).tolist()) for s in self.syllables]
        n = self.n_syllables
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = len(sets[i] & sets[j]) / self.n_active
        return M

    def mean_pairwise_overlap(self) -> float:
        """Average overlap fraction over distinct syllable pairs."""
        M = self.overlap_matrix()
        n = self.n_syllables
        iu = np.triu_indices(n, k=1)
        return float(M[iu].mean()) if len(iu[0]) else 0.0

    def replace(self, **kw) -> "SyllableConfig":
        return dataclasses.replace(self, **kw)

    def __post_init__(self):
        if self.n_channels < 2:
            raise ValueError(f"n_channels must be >= 2; got {self.n_channels}")
        if not (1 <= self.n_active <= self.n_channels):
            raise ValueError(
                f"n_active must be in [1, n_channels={self.n_channels}]; "
                f"got {self.n_active}")
        if not (0.0 <= self.overlap <= 1.0):
            raise ValueError(f"overlap must be in [0, 1]; got {self.overlap}")
        # Validate the word machinery.
        _ = _make_words(self.deviant_syllable_pos)
        # Validate the private-channel budget (raises with a clear message).
        _ = self._figure_bank()


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
