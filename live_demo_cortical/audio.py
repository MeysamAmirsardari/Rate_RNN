"""
live_demo_cortical.audio
===============

Streaming cochleo-thalamic front end + audio sources for the live demo.

``MelFrontEnd``
    Causal, frame-by-frame log-mel encoder with automatic gain control.
    Buffers raw samples and emits one mel frame per ``hop_length`` samples
    (one model step).  The dB mapping mirrors ``tasks.auditory.audio`` but
    replaces the fixed global reference with a running AGC peak tracker
    (fast attack, slow release, absolute floor).  Output ``drive`` is the
    per-channel thalamo-cortical input in ``[0, drive_gain]``.

Sources (all expose ``read(max_samples) -> np.ndarray``)
    ``MicSource``        live microphone via ``sounddevice``.
    ``WavSource``        a WAV file, optionally looping (for headless tests).
    ``SyntheticSource``  tone bursts / sweeps (no mic, no files).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

try:
    import librosa
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "live_demo_cortical requires `librosa`. Install with `pip install librosa`."
    ) from exc

from .config import LiveConfig

_EPS = 1e-12


# =====================================================================
#  Streaming mel front end with AGC
# =====================================================================
class MelFrontEnd:
    """Turn a stream of raw audio samples into per-frame model drive.

    Parameters
    ----------
    cfg : LiveConfig

    Notes
    -----
    The mel filterbank and analysis window are precomputed once.  Each
    call to :meth:`push` consumes the internal buffer in ``hop_length``
    strides, computing for every stride a Hann-windowed power spectrum,
    projecting it onto the mel bands (``power=2``, Slaney norm -- the
    librosa default, matching the offline encoder), then log/AGC mapping
    to a ``[0, drive_gain]`` drive.
    """

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self._mel_fb = librosa.filters.mel(
            sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_channels,
            fmin=cfg.fmin, fmax=cfg.fmax)                    # (N, n_freqs)
        self._window = np.hanning(cfg.n_fft).astype(np.float64)
        self._buf = np.zeros(0, dtype=np.float64)
        self._ref = float(cfg.ref_floor)                     # running AGC peak
        self._nf = float(cfg.ref_floor)                      # noise-floor estimate
        self._last_level_db = -cfg.top_db
        self._gate_open = False

    @property
    def ref(self) -> float:
        return self._ref

    @property
    def level_db(self) -> float:
        """Most recent frame's peak level relative to the AGC reference."""
        return self._last_level_db

    @property
    def gate_open(self) -> bool:
        """True if the last frame passed the noise gate (sound present)."""
        return self._gate_open

    def mel_frequencies(self) -> np.ndarray:
        return librosa.mel_frequencies(
            n_mels=self.cfg.n_channels, fmin=self.cfg.fmin, fmax=self.cfg.fmax)

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float64)
        self._ref = float(self.cfg.ref_floor)
        self._nf = float(self.cfg.ref_floor)
        self._gate_open = False

    def push(self, samples: np.ndarray):
        """Consume new samples; return (drive, db) frame blocks.

        Returns
        -------
        drive : (n_channels, k) float
            Per-channel TC drive in [0, drive_gain] for the ``k`` whole
            frames newly available (``k`` may be 0).
        db : (n_channels, k) float
            The same frames as dB relative to the running AGC reference
            (in ``[-top_db, 0]``), for the input spectrogram display.
        """
        cfg = self.cfg
        if samples.size:
            self._buf = np.concatenate([self._buf, np.asarray(samples, float)])

        hop = cfg.hop_length
        n_fft = cfg.n_fft
        if self._buf.size < n_fft:
            return (np.empty((cfg.n_channels, 0)),
                    np.empty((cfg.n_channels, 0)))

        # number of whole frames available given a sliding n_fft window.
        k = 1 + (self._buf.size - n_fft) // hop
        drive = np.empty((cfg.n_channels, k))
        db = np.empty((cfg.n_channels, k))

        atk, rel, floor = cfg.agc_attack, cfg.agc_release, cfg.ref_floor
        nf_rise, gate_factor = cfg.nf_rise, cfg.gate_factor
        inv_top = 1.0 / cfg.top_db

        for i in range(k):
            lo = i * hop
            frame = self._buf[lo:lo + n_fft] * self._window
            spec = np.fft.rfft(frame)
            power = (spec.real ** 2 + spec.imag ** 2)        # (n_freqs,)
            mel_p = self._mel_fb @ power                     # (n_channels,)

            peak = float(mel_p.max())

            # noise-floor tracker: instant drop to a new minimum, slow rise.
            self._nf = min(peak, self._nf * nf_rise)
            if self._nf < _EPS:
                self._nf = _EPS
            # noise gate WITH HYSTERESIS: once open it holds until the peak
            # falls well below the floor, so the dynamic dips of a continuous
            # song don't keep snapping it shut.
            open_thr = gate_factor * self._nf
            gate_open = peak > (0.5 * open_thr if self._gate_open else open_thr)

            # AGC: fast attack toward a louder peak, slow release otherwise.
            if peak > self._ref:
                self._ref += atk * (peak - self._ref)
            else:
                self._ref *= rel
            if self._ref < floor:
                self._ref = floor

            # DISPLAY is ALWAYS continuous: render the dB level every frame so
            # the gate cannot punch black holes into a continuous input.  Quiet
            # passages render dark via the dB mapping itself, not by blanking.
            d_db = 10.0 * np.log10(np.maximum(mel_p, _EPS) / self._ref)
            d_db = np.maximum(d_db, -cfg.top_db)
            db[:, i] = d_db
            # DRIVE is gated, so true silence / room noise still emits nothing.
            drive[:, i] = (np.clip((d_db + cfg.top_db) * inv_top, 0.0, 1.0)
                           if gate_open else 0.0)
            self._gate_open = gate_open

        drive *= cfg.drive_gain
        self._last_level_db = float(db[:, -1].max()) if k else self._last_level_db

        # retain the tail that does not yet complete a new hop
        consumed = k * hop
        self._buf = self._buf[consumed:]
        return drive, db


# =====================================================================
#  Audio sources
# =====================================================================
class MicSource:
    """Live microphone capture via ``sounddevice`` (non-blocking, queued)."""

    paced = False    # the audio callback delivers samples in real time

    def __init__(self, cfg: LiveConfig, device: Optional[int] = None):
        self.cfg = cfg
        self._device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):   # pragma: no cover
        # indata: (frames, channels) float32 -- take channel 0, copy out.
        self._q.put(indata[:, 0].astype(np.float64).copy())

    def start(self):                                           # pragma: no cover
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.cfg.sr, channels=self.cfg.in_channels,
            blocksize=self.cfg.blocksize, dtype="float32",
            device=self._device, callback=self._callback)
        self._stream.start()

    def read(self, max_samples: int = 1 << 20) -> np.ndarray:  # pragma: no cover
        chunks, n = [], 0
        while n < max_samples:
            try:
                c = self._q.get_nowait()
            except queue.Empty:
                break
            chunks.append(c)
            n += c.size
        return np.concatenate(chunks) if chunks else np.empty(0)

    def stop(self):                                            # pragma: no cover
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class WavSource:
    """A WAV file as a sample stream (headless testing / canned demos)."""

    paced = True     # no natural clock -> the app paces reads by wall time

    def __init__(self, cfg: LiveConfig, path: str, loop: bool = True,
                 gap_s: float = 0.4):
        self.cfg = cfg
        y, _ = librosa.load(path, sr=cfg.sr, mono=True)
        gap = np.zeros(int(round(gap_s * cfg.sr)), dtype=y.dtype)
        self._y = np.concatenate([y.astype(np.float64), gap.astype(np.float64)])
        self._pos = 0
        self._loop = loop

    def start(self):
        self._pos = 0

    def read(self, max_samples: int) -> np.ndarray:
        if self._pos >= self._y.size:
            if not self._loop:
                return np.empty(0)
            self._pos = 0
        end = min(self._pos + max_samples, self._y.size)
        out = self._y[self._pos:end]
        self._pos = end
        return out

    def stop(self):
        pass


class SyntheticSource:
    """Tone bursts and sweeps -- a mic-free, file-free stimulus generator."""

    paced = True     # generated on demand -> the app paces reads by wall time

    def __init__(self, cfg: LiveConfig, freqs=(440.0, 1320.0, 2640.0),
                 burst_s: float = 0.18, gap_s: float = 0.12, sweep: bool = True):
        self.cfg = cfg
        self._freqs = list(freqs)
        self._burst = int(round(burst_s * cfg.sr))
        self._gap = int(round(gap_s * cfg.sr))
        self._sweep = sweep
        self._t = 0

    def start(self):
        self._t = 0

    def _sample_at(self, n0: int, n: int) -> np.ndarray:
        sr = self.cfg.sr
        period = self._burst + self._gap
        idx = np.arange(n0, n0 + n)
        out = np.zeros(n, dtype=np.float64)
        for j, i in enumerate(idx):
            phase = i % period
            if phase < self._burst:
                k = (i // period) % len(self._freqs)
                f = self._freqs[k]
                if self._sweep:
                    f = f * (1.0 + 0.5 * phase / self._burst)
                env = np.sin(np.pi * phase / self._burst)        # raised cosine
                out[j] = 0.6 * env * np.sin(2 * np.pi * f * i / sr)
        return out

    def read(self, max_samples: int) -> np.ndarray:
        out = self._sample_at(self._t, max_samples)
        self._t += max_samples
        return out

    def stop(self):
        pass


class SFGSource:
    """Rate-matched Stochastic Figure-Ground stream (audio.sfg.make_sfg) -- a
    chord cloud with coherent figures of stepping coherence emerging and
    dissolving.  Pre-generated once and looped; the figure shows up as the
    coherent block in the live coincidence matrix (and in the learned W)."""

    paced = True

    def __init__(self, cfg: LiveConfig, seconds: float = 60.0, seed: int = 0):
        from audio.sfg import make_sfg          # project-root package
        self._y = make_sfg(sr=cfg.sr, total_s=seconds, seed=seed).astype(np.float64)
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
