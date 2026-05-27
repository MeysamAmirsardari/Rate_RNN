"""
tasks.oddball.config
====================

Configuration for the streaming oddball / SSA paradigm on model0.

Paradigm
--------
12 frequency channels (ch1..ch12).  Each channel maps 1:1 to a model0
tonotopic channel (its own E and I unit).  Four conditions, each run
as an independent session on a fresh A1Config-sized network with the
same seed but a different stimulus sequence:

  1. **oddball_f1dev**  -- f1 deviant (10%),  f2 standard (90%)
  2. **oddball_f2dev**  -- f2 deviant (10%),  f1 standard (90%)
  3. **deviant_alone**  -- f1 at 10% rate, silence at 90% (same timing
                            as f1 in condition 1; controls for stimulus
                            rate)
  4. **diverse_broad**  -- f1 and f2 each at 10%, the remaining 80%
                            split equally across the other 10 channels
                            (many-standards control)

Channel selection
-----------------
``f1_base`` sets the 1-indexed channel for f1 (default 4 -> ch4).
``delta_f`` sets the frequency distance (default 1 -> f2 = ch5).

Timing
------
Each tone is ``tone_dur`` ms (default 50 ms) followed by ``tone_gap`` ms
of silence (default 100 ms).  SOA = 150 ms.

References
----------
- Ulanovsky, Las, Nelken (2003) Nat. Neurosci. 6:391 -- SSA in A1.
- Nieto-Diego & Malmierca (2016) PLoS Biol. 14:e1002397 -- SSA along
  the auditory pathway.
- Bekinschtein et al. (2009) PNAS 106:1672 -- local-global / oddball
  paradigm framework.
- The same 4-condition structure was used in a sibling SORN-based
  project (task/streaming/); only the paradigm logic is reused here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


# =====================================================================
#  Conditions
# =====================================================================
ALL_CONDITIONS: Tuple[str, ...] = (
    "oddball_f1dev",
    "oddball_f2dev",
    "deviant_alone",
    "diverse_broad",
)


# =====================================================================
#  Configuration dataclass
# =====================================================================
@dataclass(frozen=True)
class OddballConfig:
    """Paradigm parameters for a streaming-oddball session on model0."""

    name: str = "default"

    # ---- Channels ----
    n_channels: int = 12
    f1_base: int = 4                  # 1-indexed channel for f1
    delta_f: int = 1                  # f2 = ch{f1_base + delta_f}

    # ---- Timing (ms; assumes a1_cfg.dt == 1 ms) ----
    tone_dur: int = 50
    tone_gap: int = 100
    pre_stim_ms: int = 50
    post_stim_ms: int = 100

    # ---- Protocol ----
    n_trials: int = 500
    dev_prob: float = 0.10

    # ---- Conditions to run ----
    conditions: Tuple[str, ...] = ALL_CONDITIONS

    # ---- Stimulus amplitude ----
    tone_amp: float = 1.0

    # ---- Reproducibility ----
    seed: int = 42

    # ============ Derived properties ============
    @property
    def channel_names(self) -> Tuple[str, ...]:
        return tuple(f"ch{i+1}" for i in range(self.n_channels))

    @property
    def f1_name(self) -> str:
        return f"ch{self.f1_base}"

    @property
    def f2_name(self) -> str:
        return f"ch{self.f1_base + self.delta_f}"

    @property
    def f1_channel(self) -> int:
        return self.f1_base - 1

    @property
    def f2_channel(self) -> int:
        return self.f1_base + self.delta_f - 1

    @property
    def soa(self) -> int:
        """Stimulus-onset asynchrony = tone_dur + tone_gap."""
        return self.tone_dur + self.tone_gap

    @property
    def epoch_steps(self) -> int:
        """Per-trial recorded window (pre + tone + post)."""
        return self.pre_stim_ms + self.tone_dur + self.post_stim_ms

    @property
    def tone_onset_in_epoch(self) -> int:
        return self.pre_stim_ms

    def channel_index(self, name: str) -> int:
        """0-based index for 'ch{i}' (or -1 for 'silence')."""
        if name == "silence":
            return -1
        if not name.startswith("ch"):
            raise ValueError(f"Bad channel name: {name!r}")
        return int(name[2:]) - 1

    def __post_init__(self):
        f2_idx = self.f1_base + self.delta_f
        if f2_idx < 1 or f2_idx > self.n_channels:
            raise ValueError(
                f"f2 = ch{f2_idx} out of range [1, {self.n_channels}]. "
                f"Adjust f1_base or delta_f.")
        for c in self.conditions:
            if c not in ALL_CONDITIONS:
                raise ValueError(f"Unknown condition {c!r}; "
                                 f"valid: {ALL_CONDITIONS}")

    def replace(self, **kw) -> "OddballConfig":
        return dataclasses.replace(self, **kw)


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> OddballConfig:
    """Canonical: 12 channels, 500 trials, f1=ch4, delta_f=1."""
    return OddballConfig(**kw)


def short(**kw) -> OddballConfig:
    """Faster: 200 trials per condition."""
    return OddballConfig(name="short", n_trials=200, **kw)


def wide_df(**kw) -> OddballConfig:
    """Wider frequency distance: delta_f=5 -> f1=ch4, f2=ch9."""
    return OddballConfig(name="wide_df", delta_f=5, **kw)


PRESETS: Dict[str, Callable[..., OddballConfig]] = {
    "default":  default,
    "short":    short,
    "wide_df":  wide_df,
}


def get_preset(name: str, **overrides) -> OddballConfig:
    if name not in PRESETS:
        avail = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available: {avail}")
    return PRESETS[name](**overrides)
