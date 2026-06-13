"""
live_demo_speech.speech
=======================

Two-talker speech material for the segregation demo.

The talkers are the three **LibriSpeech example excerpts** shipped via
``librosa`` (open / CC, cached locally on first use) -- three different readers
with well-separated fundamentals, so a two-talker mixture is segregable by
temporal coherence + pitch:

    libri1   male,   median F0 ~78 Hz   (Ashiel Mystery, G. Comira)
    libri2   male,   median F0 ~147 Hz  (Age of Chivalry, A. Lankford)
    libri3   female, median F0 ~208 Hz  (Sense & Sensibility, H. Barnett)

``mix_two_talkers`` returns the mixture AND the two clean sources (ground truth
for validation -- on the same common scale, so ``mix == a + b``).  Real speech,
no synthetic shortcuts.
"""
from __future__ import annotations

import numpy as np

from .config import LiveConfig

_TALKERS = ("libri1", "libri2", "libri3")
_EPS = 1e-9


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)) + _EPS)


def load_talker(name: str, sr: int) -> np.ndarray:
    """Load one LibriSpeech example talker, resampled to ``sr`` (mono)."""
    import librosa
    if name not in _TALKERS:
        raise ValueError(f"talker must be one of {_TALKERS}; got {name!r}")
    y, _ = librosa.load(librosa.ex(name), sr=sr, mono=True)
    return y.astype(np.float64)


def mix_two_talkers(sr: int, names=("libri1", "libri3"), seconds: float | None = None,
                    snr_db: float = 0.0, seed: int = 0):
    """Mix two talkers at ``snr_db`` (talker A relative to B), RMS-matched then
    summed.  Returns ``(mix, a, b)`` on a common scale with ``mix == a + b``;
    ``a``/``b`` are the clean sources (validation ground truth)."""
    a = load_talker(names[0], sr)
    b = load_talker(names[1], sr)
    n = min(a.size, b.size)
    if seconds is not None:
        n = min(n, int(round(seconds * sr)))
    a, b = a[:n], b[:n]
    a = a / _rms(a)
    b = b / _rms(b)
    a *= 10.0 ** (snr_db / 20.0)             # talker A louder by snr_db
    mix = a + b
    pk = float(np.max(np.abs(mix))) + _EPS    # common scale -> mix == a + b
    mix, a, b = mix / pk * 0.95, a / pk * 0.95, b / pk * 0.95
    return mix.clip(-1.0, 1.0), a, b


class TwoTalkerSource:
    """Looping paced source: a two-talker mixture, with the clean sources kept
    in ``self.sources = (a, b)`` for validation."""

    paced = True

    def __init__(self, cfg: LiveConfig, names=("libri1", "libri3"),
                 seconds: float | None = None, snr_db: float = 0.0,
                 seed: int = 0):
        mix, a, b = mix_two_talkers(cfg.sr, names=names, seconds=seconds,
                                    snr_db=snr_db, seed=seed)
        self._y = mix
        self.sources = (a, b)
        self.names = names
        self._pos = 0

    def start(self):
        self._pos = 0

    def read(self, max_samples: int) -> np.ndarray:
        if self._pos >= self._y.size:
            self._pos = 0
        end = min(self._pos + max_samples, self._y.size)
        out = self._y[self._pos:end]
        self._pos = end
        return out

    def stop(self):
        pass
