"""
layer2_multirate.config
=======================

Parameters for the multi rate layer 2.  Layer 1 is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def default_rates(n=6, lo=0.030, hi=0.500):
    """Log spaced time constants from 30 ms to 500 ms (about 5 Hz to 2 Hz)."""
    return tuple(np.round(np.geomspace(lo, hi, n), 4))


@dataclass
class MRConfig:
    """Configuration of the multi rate layer 2 population."""

    # ---- the filterbank ----------------------------------------------
    # One leaky integrator per channel per rate.  The span matters more than
    # the count: the slowest must outlast a whole chunk so the first token is
    # still legible when the last one arrives, and the fastest must be short
    # enough to separate the immediately preceding token from the one before.
    rates: tuple = field(default_factory=default_rates)   # seconds

    # ---- population --------------------------------------------------
    # Deliberately generous.  How many stay committed is a result.
    n_units: int = 24

    # ---- plasticity (identical to the single rate version) -----------
    eta: float = 5e-3              # instar rate, per time step
    lam: float = 1e-4              # synaptic decay, per time step
    gate_frac: float = 0.15        # learn only above this fraction of peak drive
    w_init: float = 0.05
    commit_frac: float = 0.20

    seed: int = 0
