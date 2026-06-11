"""
live_demo_cortical.cortex
=========================

Cortical **temporal rate** front end -- the half of the Chi/Shamma cortical
model that actually matters for temporal coherence (and that the first cut got
wrong by using a spectral *scale* filter, which just smears along frequency).

Each channel's envelope is band-pass filtered in TIME (modulation rates ~2-32
Hz) by a causal difference-of-exponentials, then half-wave rectified.  This
emphasises onsets / the modulation rate and discards the sustained level, so
channels that turn on and off TOGETHER (a temporally coherent figure) produce
correlated, onset-locked drive, while incoherent background does not -- without
touching the frequency axis (no spectral blur, tonotopy preserved).

Stateful and causal (keeps the two leaky integrators across audio blocks),
vectorised over channels.
"""
from __future__ import annotations

import numpy as np


class CortexFrontEnd:
    """Per-channel temporal band-pass (modulation) filter on the mel drive."""

    def __init__(self, cfg, rate_low: float = 2.0, rate_high: float = 32.0,
                 mix: float = 1.0):
        self.cfg = cfg
        self.mix = float(mix)                  # 0 = raw mel, 1 = full rate-filtered
        dt = cfg.dt
        N = cfg.n_channels
        # leaky-integrator coefficients for the fast (high-rate) and slow
        # (low-rate) envelopes; their difference is a 2..32 Hz band-pass.
        self.a_fast = float(np.clip(dt * 2 * np.pi * rate_high, 0.0, 1.0))
        self.a_slow = float(np.clip(dt * 2 * np.pi * rate_low, 0.0, 1.0))
        self.s_fast = np.zeros(N)
        self.s_slow = np.zeros(N)

    def reset(self):
        self.s_fast[:] = 0.0
        self.s_slow[:] = 0.0

    def process(self, drive: np.ndarray) -> np.ndarray:
        """``(N, k)`` mel drive -> ``(N, k)`` onset/modulation-emphasised drive."""
        k = drive.shape[1]
        if k == 0:
            return drive
        out = np.empty_like(drive)
        sf, ss = self.s_fast, self.s_slow
        af, as_ = self.a_fast, self.a_slow
        for t in range(k):
            x = drive[:, t]
            sf += af * (x - sf)
            ss += as_ * (x - ss)
            out[:, t] = np.maximum(sf - ss, 0.0)   # band-pass, half-wave rectified
        self.s_fast, self.s_slow = sf, ss
        # keep the rate-filtered drive on roughly the same scale as the input
        mo = out.max()
        if mo > 1e-9:
            out *= (drive.max() + 1e-9) / mo
        if self.mix >= 1.0:
            return out
        return (1.0 - self.mix) * drive + self.mix * out
