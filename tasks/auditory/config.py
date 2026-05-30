"""
tasks.auditory.config
======================

Configuration for the Roving Oddball paradigm on model0 driven by
**real speech** -- the mel-spectrograms of recorded syllables -- fed in
as the thalamo-cortical (TC) input to a 180-channel A1 model.

Relation to ``tasks.roving`` / ``tasks.syllable``
-------------------------------------------------
The protocol is identical to the roving-oddball task (blocks of one
repeated 3-syllable word, constrained-random block order, plasticity on
throughout).  Only the *stimulus* changes:

    roving    -- each syllable is a single tonotopic channel.
    syllable  -- each syllable is a synthetic multi-channel "figure".
    auditory  -- each syllable is the **mel-spectrogram of a real WAV**,
                 i.e. a biologically-grounded cochleo-thalamic
                 representation, on ``n_channels`` (default 180) mel
                 bands spanning a log-frequency (tonotopic) axis.

Stimulus
--------
Five recorded syllables, mapped A..E:

    A = "boo"   B = "pee"   C = "tah"   D = "bey" (bay.wav)   E = "see"

Each WAV is loaded, resampled to ``sr``, and converted to a log-mel
spectrogram with ``hop_ms = 1 ms`` per frame so that one spectrogram
frame maps 1:1 onto one model integration step (``A1Config.dt = 1 ms``).
The spectrogram is normalised against a **shared** reference across all
five syllables (so natural loudness differences are preserved) onto
[0, 1] and scaled by ``syll_amp`` to become the per-channel TC drive.
See ``tasks.auditory.audio``.

Timing
------
The recorded syllables are each 180 ms, matching ``syll_dur = 180 ms``;
``syll_gap = 0`` inside a sequence, ``seq_gap = 1000 ms`` between
sequences.  A trial spans ``3*180 + 0 + 1000 = 1540 ms``.

Model parameters are *not* in this config -- they live in
``model0.A1Config`` and are passed separately to ``run_experiment``.

References
----------
- Bekinschtein et al. (2009) PNAS 106:1672 -- roving / local-global.
- Mesgarani, Cheung, Johnson, Chang (2014) Science 343:1006 -- speech
  spectro-temporal representation in human auditory cortex.
- Chi, Ru, Shamma (2005) JASA 118:887 -- auditory spectrogram / cortical
  spectro-temporal model motivating a mel-like tonotopic input.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple


# Default location of the recorded-syllable WAV folder (sibling of this
# package: ``tasks/syllables_wav``).
_DEFAULT_WAV_DIR = Path(__file__).resolve().parent.parent / "syllables_wav"


# =====================================================================
#  Word generation
# =====================================================================
SHARED_SYLLABLES = ("A", "B")
VARIABLE_SYLLABLES = ("C", "D", "E")

# Syllable -> WAV stem.  "bey" is spelled "bay" in the recording set.
DEFAULT_SYLLABLE_FILES: Dict[str, str] = {
    "A": "boo",
    "B": "pee",
    "C": "tah",
    "D": "bay",
    "E": "see",
}


def _make_words(deviant_syllable_pos: int) -> Tuple[str, ...]:
    """Three word types for a given deviant-syllable position.

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
class AuditoryConfig:
    """Paradigm parameters for a speech-driven roving session on model0."""

    name: str = "default"

    # ---- Spectral channels (== number of mel bands) ----
    n_channels: int = 180

    # ---- Audio / mel-spectrogram parameters ----
    sr: int = 16000                    # resample target (Hz)
    n_fft: int = 512                   # STFT window (samples)
    hop_ms: float = 1.0                # frame hop (ms) -> 1 frame == 1 step
    fmin: float = 50.0                 # lowest mel edge (Hz)
    fmax: float = 8000.0               # highest mel edge (Hz)
    top_db: float = 60.0               # dynamic range below shared peak (dB)
    per_syllable_norm: bool = False    # True -> equalise loudness per syllable
    active_thresh: float = 0.35        # fraction-of-peak cut for "active" chan
    wav_dir: Optional[str] = None      # None -> DEFAULT_WAV_DIR
    syllable_files: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SYLLABLE_FILES))

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
    n_blocks_per_word: int = 6         # 3 words * 6 = 18 blocks
    n_reps_per_block: int = 15         # 18 blocks * 15 = 270 trials

    # ---- Syllable vocabulary ----
    syllables: Tuple[str, ...] = ("A", "B", "C", "D", "E")

    # ---- Stimulus amplitude ----
    syll_amp: float = 1.0

    # ---- Reproducibility ----
    seed: int = 42

    # ============ Derived properties ============
    @property
    def words(self) -> Tuple[str, ...]:
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
    def hop_length(self) -> int:
        """STFT hop in samples (one frame per ``hop_ms`` of audio)."""
        return max(1, int(round(self.sr * self.hop_ms / 1000.0)))

    @property
    def stim_steps(self) -> int:
        n = self.n_sylls_per_seq
        return n * self.syll_dur + max(0, n - 1) * self.syll_gap

    @property
    def trial_period(self) -> int:
        return self.stim_steps + self.seq_gap

    @property
    def epoch_steps(self) -> int:
        return self.pre_stim_ms + self.stim_steps + self.post_stim_ms

    @property
    def syll_onsets_in_epoch(self) -> Tuple[int, ...]:
        slot = self.syll_dur + self.syll_gap
        return tuple(self.pre_stim_ms + i * slot
                     for i in range(self.n_sylls_per_seq))

    @property
    def deviant_syllable_index(self) -> int:
        return self.deviant_syllable_pos - 1

    @property
    def shared_syllables(self) -> Tuple[str, ...]:
        return SHARED_SYLLABLES

    @property
    def variable_syllables(self) -> Tuple[str, ...]:
        return VARIABLE_SYLLABLES

    def wav_path(self, syllable: str) -> Path:
        """Absolute path to the WAV file backing ``syllable``."""
        if syllable not in self.syllable_files:
            raise ValueError(
                f"No WAV mapping for syllable {syllable!r}; "
                f"known: {tuple(self.syllable_files)}")
        base = Path(self.wav_dir) if self.wav_dir else _DEFAULT_WAV_DIR
        return base / f"{self.syllable_files[syllable]}.wav"

    def replace(self, **kw) -> "AuditoryConfig":
        return dataclasses.replace(self, **kw)

    def __post_init__(self):
        if self.n_channels < 2:
            raise ValueError(f"n_channels must be >= 2; got {self.n_channels}")
        if self.fmax <= self.fmin:
            raise ValueError(
                f"fmax ({self.fmax}) must exceed fmin ({self.fmin})")
        if self.fmax > self.sr / 2:
            raise ValueError(
                f"fmax ({self.fmax}) exceeds Nyquist (sr/2 = {self.sr / 2})")
        _ = _make_words(self.deviant_syllable_pos)
        for s in self.syllables:
            if s not in self.syllable_files:
                raise ValueError(
                    f"Syllable {s!r} has no WAV mapping (known: "
                    f"{tuple(self.syllable_files)}).")


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> AuditoryConfig:
    """Canonical: 18 blocks (6 per word) x 15 reps = 270 trials, deviant pos 3."""
    return AuditoryConfig(**kw)


def short(**kw) -> AuditoryConfig:
    """Faster: 6 blocks (2 per word) x 8 reps = 48 trials (smoke test)."""
    return AuditoryConfig(name="short", n_blocks_per_word=2,
                          n_reps_per_block=8, **kw)


def full(**kw) -> AuditoryConfig:
    """Roving-matched: 30 blocks (10 per word) x 15 reps = 450 trials."""
    return AuditoryConfig(name="full", n_blocks_per_word=10, **kw)


def deviant_pos2(**kw) -> AuditoryConfig:
    """Deviant at position 2: words ACB / ADB / AEB."""
    return AuditoryConfig(name="deviant_pos2", deviant_syllable_pos=2, **kw)


def deviant_pos1(**kw) -> AuditoryConfig:
    """Deviant at position 1: words CAB / DAB / EAB."""
    return AuditoryConfig(name="deviant_pos1", deviant_syllable_pos=1, **kw)


PRESETS: Dict[str, Callable[..., AuditoryConfig]] = {
    "default":      default,
    "short":        short,
    "full":         full,
    "deviant_pos2": deviant_pos2,
    "deviant_pos1": deviant_pos1,
}


def get_preset(name: str, **overrides) -> AuditoryConfig:
    if name not in PRESETS:
        avail = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available: {avail}")
    return PRESETS[name](**overrides)
