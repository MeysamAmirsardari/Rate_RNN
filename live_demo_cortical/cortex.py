"""
live_demo_cortical.cortex
=========================

Cortical spectro-temporal **front end** (Chi, Ru & Shamma, JASA 2005;
the same stage used by the temporal-coherence model of Teki 2013 /
Krishnan 2014).

Stage 1 of the cortical model is a **multi-scale spectral wavelet analysis**:
every cochleo-mel frame is filtered along the (log-)frequency axis by a bank
of Gabor wavelets at several *scales* (cycles/octave).  Coarse scales pool
energy across many neighbouring channels, fine scales keep near-tone
resolution; the magnitude responses are combined into a single "cortically
spread" drive.  Because a single tone now activates a *band* of channels,
neighbouring channels co-activate -> the off-diagonal / diagonal-stripe
structure of cortical coincidence matrices (Teki 2013, Fig 3C) appears in the
model's learned recurrent map W.

Efficiency
----------
The whole ``(N, k)`` block is convolved along frequency **once per scale**
(``scipy.ndimage.convolve1d``) -- no Python per-sample loops; a handful of
small-kernel convolutions per tick (microseconds for N=180).  A future stage 2
(temporal *rate* filters, 2-32 Hz) can be added the same way along time.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import convolve1d
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "live_demo_cortical.cortex requires scipy (scipy.ndimage). "
        "Install with `pip install scipy`.") from exc


class CortexFrontEnd:
    """Multi-scale spectral wavelet front end; maps a mel drive to a
    cortically spread drive of the same shape."""

    def __init__(self, cfg, scales_cyc_oct=(0.5, 1.0, 2.0, 4.0),
                 weights=None, mix: float = 1.0):
        self.cfg = cfg
        N = cfg.n_channels
        # bins per octave of the mel frequency axis
        self.bins_per_oct = N / np.log2(cfg.fmax / cfg.fmin)
        self.scales = tuple(scales_cyc_oct)
        w = np.ones(len(self.scales)) if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()
        self.mix = float(mix)                      # 0 = raw mel, 1 = full cortical
        self._kernels = [self._gabor(s) for s in self.scales]

    def _gabor(self, scale: float):
        """Zero-mean complex Gabor wavelet (cos, sin) along frequency for a
        given spectral scale (cyc/oct)."""
        period = self.bins_per_oct / scale         # bins per cycle
        half = max(1, int(round(1.5 * period)))
        d = np.arange(-half, half + 1)
        env = np.exp(-0.5 * (d / (0.5 * period)) ** 2)
        cos = np.cos(2 * np.pi * d / period) * env
        sin = np.sin(2 * np.pi * d / period) * env
        return cos - cos.mean(), sin - sin.mean()  # band-pass (drop DC)

    def process(self, drive: np.ndarray) -> np.ndarray:
        """``(N, k)`` mel drive -> ``(N, k)`` cortically spread drive.

        ``mix`` blends raw and cortical so the change is tunable live."""
        if drive.shape[1] == 0:
            return drive
        cort = np.zeros_like(drive)
        for (cos, sin), w in zip(self._kernels, self.weights):
            c = convolve1d(drive, cos, axis=0, mode="reflect")
            s = convolve1d(drive, sin, axis=0, mode="reflect")
            cort += w * np.sqrt(c * c + s * s)     # scale magnitude (envelope)
        # keep the cortical drive on the same scale as the input
        cort *= (drive.max() + 1e-9) / (cort.max() + 1e-9)
        if self.mix >= 1.0:
            return cort
        return (1.0 - self.mix) * drive + self.mix * cort
