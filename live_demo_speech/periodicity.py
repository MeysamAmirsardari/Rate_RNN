"""
live_demo_speech.periodicity
============================

Pitch / periodicity analysis and **time-resolved** two-talker assignment -- the
cue that actually separates concurrent voices (Licklider 1951; Meddis & Hewitt
1992; Wang & Brown CASA).

Biologically: each cochlear channel is band-pass filtered, the hair cell
half-wave rectifies, and a low-pass keeps the periodicity envelope (the
unresolved harmonics beat at the fundamental, and every channel carrying a
harmonic ``k*F0`` has an autocorrelation peak at lag ``1/F0``).  An
**autocorrelation** of each channel (the brainstem delay-line / periodicity
detector) therefore reports the fundamental of whichever talker dominates that
channel; the **summary autocorrelation** (SACF, summed over channels) shows a
peak per concurrent F0.  Tracking the two strongest F0s and, at each moment,
routing each channel to the F0 its local autocorrelation supports, segregates
the two voices -- a soft time-frequency mask.

``PeriodicitySeparator`` is streaming: ``push(audio_block)`` advances the
per-channel band-pass / rectify / low-pass filters and a down-sampled envelope
ring buffer; ``compute()`` returns ``(masks, sacf, C)`` -- the per-channel soft
talker masks, the summary autocorrelation, and a channel x channel periodicity
coincidence matrix (channels sharing an F0 -> high C; the harmonic grouping).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .config import LiveConfig


class PeriodicitySeparator:
    def __init__(self, cfg: LiveConfig, n_streams: int = 2, ds: int = 4,
                 acf_window_s: float = 0.06):
        self.cfg = cfg
        self.K = n_streams
        self.N = cfg.n_channels
        self.sr = cfg.sr
        self.ds = ds                                   # envelope decimation
        self.fs2 = cfg.sr // ds
        self.centers = np.geomspace(cfg.fmin, cfg.fmax, self.N)

        # per-channel cochlear band-pass (constant-Q ~ ERB) + shared low-pass
        self._band = []
        self._zb = []
        for fc in self.centers:
            bw = max(40.0, fc / 8.0)
            lo = max(20.0, fc - bw)
            hi = min(self.sr / 2.0 - 1.0, fc + bw)
            sos = butter(2, [lo, hi], btype="band", fs=self.sr, output="sos")
            self._band.append(sos)
            self._zb.append(sosfilt_zi(sos) * 0.0)
        self._lp = butter(2, 1000.0, btype="low", fs=self.sr, output="sos")
        self._zl = [sosfilt_zi(self._lp) * 0.0 for _ in range(self.N)]

        self._win = max(64, int(round(acf_window_s * self.fs2)))
        self._env = np.zeros((self.N, self._win))      # down-sampled envelopes
        # pitch lag range (samples at fs2)
        self.lmin = max(2, int(round(self.fs2 / cfg.pitch_fmax)))
        self.lmax = int(round(self.fs2 / cfg.pitch_fmin))
        self._lags = np.arange(self.lmin, self.lmax)
        self._trackA = self.lmax - 1                   # low-F0 talker (long lag)
        self._trackB = self.lmin                        # high-F0 talker (short lag)
        self._init = False
        self.masks = np.zeros((self.N, self.K), dtype=np.float32)
        self.masks[:, 0] = 0.5
        if self.K > 1:
            self.masks[:, 1] = 0.5
        self.sacf = np.zeros(self._lags.size, dtype=np.float32)
        self.C = np.zeros((self.N, self.N), dtype=np.float32)

    # -----------------------------------------------------------------
    def push(self, audio: np.ndarray):
        """Advance the cochlear filters on a raw audio block and append the new
        down-sampled envelope samples to the ring buffer."""
        if audio.size == 0:
            return
        x = np.asarray(audio, dtype=np.float64)
        new = np.empty((self.N, 0))
        cols = []
        for i in range(self.N):
            yb, self._zb[i] = sosfilt(self._band[i], x, zi=self._zb[i])
            r = np.maximum(yb, 0.0)                     # half-wave rectify
            yl, self._zl[i] = sosfilt(self._lp, r, zi=self._zl[i])
            cols.append(yl[::self.ds])
        new = np.asarray(cols)                          # (N, k_ds)
        k = new.shape[1]
        if k == 0:
            return
        if k >= self._win:
            self._env[:] = new[:, -self._win:]
        else:
            self._env[:, :-k] = self._env[:, k:]
            self._env[:, -k:] = new

    # -----------------------------------------------------------------
    def _acf(self):
        """Per-channel normalized autocorrelation over the pitch lags."""
        e = self._env - self._env.mean(1, keepdims=True)
        norm = (e * e).sum(1) + 1e-9
        acf = np.empty((self.N, self._lags.size))
        for j, lag in enumerate(self._lags):
            acf[:, j] = (e[:, :-lag] * e[:, lag:]).sum(1)
        acf = np.maximum(acf, 0.0) / norm[:, None]
        return acf

    def compute(self):
        """Return ``(masks (N,K), sacf, C)`` from the current envelope buffer."""
        acf = self._acf()
        sacf = acf.sum(0)
        self.sacf = sacf.astype(np.float32)
        # two F0 lags = the two strongest, well-separated SACF peaks of THIS
        # window (no temporal smoothing -- assignment adapts to who is voiced
        # now; identity is kept by ordering A = longer lag = lower F0).
        p1 = int(np.argmax(sacf))
        guard = max(2, (self.lmax - self.lmin) // 12)
        s2 = sacf.copy()
        s2[max(0, p1 - guard):p1 + guard + 1] = -np.inf
        p2 = int(np.argmax(s2))
        self._trackA, self._trackB = sorted((self._lags[p1], self._lags[p2]),
                                            reverse=True)
        ia = int(np.clip(self._trackA - self.lmin, 0, self._lags.size - 1))
        ib = int(np.clip(self._trackB - self.lmin, 0, self._lags.size - 1))
        a = acf[:, ia]
        b = acf[:, ib]
        wA = a / (a + b + 1e-9)
        self.masks[:, 0] = wA.astype(np.float32)
        if self.K > 1:
            self.masks[:, 1] = (1.0 - wA).astype(np.float32)
        # periodicity coincidence: channels with similar ACF profiles share F0
        z = acf - acf.mean(1, keepdims=True)
        zn = np.sqrt((z * z).sum(1)) + 1e-9
        C = (z @ z.T) / np.outer(zn, zn)
        np.fill_diagonal(C, 0.0)
        self.C = np.clip(C, 0.0, 1.0).astype(np.float32)
        return self.masks, self.sacf, self.C

    def f0_hz(self):
        """Current tracked fundamentals (Hz) for the two talkers."""
        return (self.fs2 / max(self._trackA, 1), self.fs2 / max(self._trackB, 1))

    def reset(self):
        for i in range(self.N):
            self._zb[i] *= 0.0
            self._zl[i] *= 0.0
        self._env[:] = 0.0
        self._init = False
        self.masks[:] = 0.5
        self.C[:] = 0.0
