"""ISO 226:2003 equal-loudness weighting.

Presenting every tone at the same SPL does not present them at the same
loudness: at 60 phon a 200 Hz tone needs ~12 dB more than a 1 kHz tone, and
the 3-4 kHz region needs several dB less.  Left uncorrected the stimulus is
dominated by its mid-highs, which makes it both harsh to listen to and
unbalanced as an experiment -- the figure's salience would depend on where in
the pool it happened to land.  O'Sullivan et al. (2015) apply the same
correction with an outer/middle-ear filter.
"""

from __future__ import annotations

import numpy as np

_F = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
               400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
               5000, 6300, 8000, 10000, 12500], float)
_AF = np.array([.532, .506, .480, .455, .432, .409, .387, .367, .349, .330,
                .315, .301, .288, .276, .267, .259, .253, .250, .246, .244,
                .243, .243, .243, .242, .242, .245, .254, .271, .301])
_LU = np.array([-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2,
                -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1,
                -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1])
_TF = np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
                14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3,
                -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3])


def spl_for_phon(freq: np.ndarray, phon: float) -> np.ndarray:
    """SPL, in dB, at which each frequency is heard at `phon` phons."""
    lg = np.log10(np.clip(freq, _F[0], _F[-1]))
    af = np.interp(lg, np.log10(_F), _AF)
    lu = np.interp(lg, np.log10(_F), _LU)
    tf = np.interp(lg, np.log10(_F), _TF)
    a = (4.47e-3 * (10 ** (0.025 * phon) - 1.15)
         + (0.4 * 10 ** ((tf + lu) / 10 - 9)) ** af)
    return (10 / af) * np.log10(a) - lu + 94


def weights(freq: np.ndarray, phon: float) -> np.ndarray:
    """Amplitude per channel so every channel is equally loud.

    Normalised to 1 kHz, so the weighting changes the balance across the
    pool without changing the overall level much.
    """
    db = spl_for_phon(np.asarray(freq, float), phon) - spl_for_phon(
        np.array([1000.0]), phon)
    return 10 ** (db / 20.0)
