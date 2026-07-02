"""
tasks.roving.config
===================

Configuration for the Roving Oddball paradigm on model0.

Paradigm
--------
Blocks of ``n_reps_per_block`` consecutive presentations of one 3-tone
word.  Three default word types (deviant tone at position 3):

    ABC -- A is tone 1, B is tone 2, C is the deviant tone.
    ABD
    ABE

The deviant position is configurable: with ``deviant_tone_pos=2`` the
words are ACB/ADB/AEB, with ``deviant_tone_pos=1`` they are CAB/DAB/EAB.
A and B are the shared tones; C, D, E are the variable (deviant) tones.

Block order is constrained random: no two consecutive blocks present
the same word.  Plasticity remains on throughout -- the network is
always learning.

Timing
------
``tone_dur = 180 ms`` per tone, ``tone_gap = 0`` between tones inside
a sequence, ``seq_gap = 1000 ms`` of silence between sequences.  A
trial therefore spans ``3*180 + 0 + 1000 = 1540 ms``.

Model parameters are *not* in this config -- they live in
``model0.A1Config`` and are passed separately to ``run_experiment``.

References
----------
- Bekinschtein, Dehaene, Rohaut, Tadel, Cohen, Naccache (2009) PNAS
  106:1672 -- local-global / roving oddball framework.
- The same paradigm has been implemented for a SORN-based network at
  task/roving/ in a sibling project; only the paradigm logic
  (timing, word generation, block-order constraint) is reused here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


# =====================================================================
#  Word generation
# =====================================================================
SHARED_TONES = ("A", "B")
VARIABLE_TONES = ("C", "D", "E")


def _make_words(deviant_tone_pos: int) -> Tuple[str, ...]:
    """Three word types for a given deviant-tone position.

    The deviant position is filled by C, D, or E (one per word); the
    two non-deviant positions are filled with A and B in order.

    >>> _make_words(3)
    ('ABC', 'ABD', 'ABE')
    >>> _make_words(2)
    ('ACB', 'ADB', 'AEB')
    >>> _make_words(1)
    ('CAB', 'DAB', 'EAB')
    """
    if deviant_tone_pos not in (1, 2, 3):
        raise ValueError(
            f"deviant_tone_pos must be 1, 2 or 3; got {deviant_tone_pos}")
    shared_positions = [p for p in (1, 2, 3) if p != deviant_tone_pos]
    words = []
    for var_tone in VARIABLE_TONES:
        slots = [""] * 3
        slots[deviant_tone_pos - 1] = var_tone
        for s_idx, p in enumerate(shared_positions):
            slots[p - 1] = SHARED_TONES[s_idx]
        words.append("".join(slots))
    return tuple(words)


# =====================================================================
#  Configuration dataclass
# =====================================================================
@dataclass(frozen=True)
class RovingConfig:
    """Paradigm parameters for a roving-oddball session on model0."""

    name: str = "default"

    # ---- Timing (ms; assumes A1Config.dt == 1 ms) ----
    tone_dur: int = 180
    tone_gap: int = 0
    seq_gap: int = 1000

    # ---- Per-trial analysis padding kept inside the recorded window ----
    pre_stim_ms: int = 100
    post_stim_ms: int = 200

    # ---- Protocol ----
    n_tones_per_seq: int = 3
    deviant_tone_pos: int = 1
    words_override: Optional[Tuple[str, ...]] = None
    n_blocks_per_word: int = 10        # 3 words * 10 = 30 blocks
    n_reps_per_block: int = 15         # 30 blocks * 15 = 450 trials

    # ---- Channel assignment (tone character -> channel index) ----
    tones: Tuple[str, ...] = ("A", "B", "C", "D", "E")

    # ---- Stimulus amplitude ----
    tone_amp: float = 1.0

    # ---- Reproducibility ----
    seed: int = 42

    # ============ Derived properties ============
    @property
    def words(self) -> Tuple[str, ...]:
        """Word types in this session.  Auto-generated from
        ``deviant_tone_pos`` unless ``words_override`` is set."""
        if self.words_override is not None:
            return tuple(self.words_override)
        return _make_words(self.deviant_tone_pos)

    @property
    def n_blocks(self) -> int:
        return len(self.words) * self.n_blocks_per_word

    @property
    def n_total_seqs(self) -> int:
        return self.n_blocks * self.n_reps_per_block

    @property
    def stim_steps(self) -> int:
        """Stimulus-only steps (3 tones with intra-sequence gaps)."""
        n = self.n_tones_per_seq
        return n * self.tone_dur + max(0, n - 1) * self.tone_gap

    @property
    def trial_period(self) -> int:
        """One trial = stimulus + inter-sequence silence (1540 ms default)."""
        return self.stim_steps + self.seq_gap

    @property
    def epoch_steps(self) -> int:
        """Per-trial recorded window length (pre + stim + post)."""
        return self.pre_stim_ms + self.stim_steps + self.post_stim_ms

    @property
    def tone_onsets_in_epoch(self) -> Tuple[int, ...]:
        """Sample indices of tone onsets inside the recorded epoch
        (already offset by ``pre_stim_ms``)."""
        slot = self.tone_dur + self.tone_gap
        return tuple(self.pre_stim_ms + i * slot
                     for i in range(self.n_tones_per_seq))

    @property
    def deviant_tone_index(self) -> int:
        """0-based index of the deviant tone within a sequence."""
        return self.deviant_tone_pos - 1

    @property
    def shared_tones(self) -> Tuple[str, ...]:
        return SHARED_TONES

    @property
    def variable_tones(self) -> Tuple[str, ...]:
        return VARIABLE_TONES

    def tone_channel(self, tone: str) -> int:
        """Channel index for a tone character (A->0, B->1, ...)."""
        return self.tones.index(tone)

    def replace(self, **kw) -> "RovingConfig":
        return dataclasses.replace(self, **kw)


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> RovingConfig:
    """Canonical: 30 blocks (10 per word) x 15 reps = 450 trials, deviant at pos 3."""
    return RovingConfig(**kw)


def short(**kw) -> RovingConfig:
    """Faster: 15 blocks (5 per word) x 15 reps = 225 trials."""
    return RovingConfig(name="short", n_blocks_per_word=5, **kw)


def long_(**kw) -> RovingConfig:
    """Longer: 90 blocks (30 per word) x 15 reps = 1350 trials."""
    return RovingConfig(name="long", n_blocks_per_word=30, **kw)


def deviant_pos2(**kw) -> RovingConfig:
    """Deviant at position 2: words ACB / ADB / AEB."""
    return RovingConfig(name="deviant_pos2", deviant_tone_pos=2, **kw)


def deviant_pos1(**kw) -> RovingConfig:
    """Deviant at position 1: words CAB / DAB / EAB."""
    return RovingConfig(name="deviant_pos1", deviant_tone_pos=1, **kw)


PRESETS: Dict[str, Callable[..., RovingConfig]] = {
    "default":      default,
    "short":        short,
    "long":         long_,
    "deviant_pos2": deviant_pos2,
    "deviant_pos1": deviant_pos1,
}


def get_preset(name: str, **overrides) -> RovingConfig:
    """Look up a preset by name with optional parameter overrides."""
    if name not in PRESETS:
        avail = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available: {avail}")
    return PRESETS[name](**overrides)
