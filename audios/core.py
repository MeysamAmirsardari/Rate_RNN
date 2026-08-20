"""
audios.core
===========

The shared machinery: gating, jitter, level, and the checks that read the
timing back off the rendered signal instead of trusting the code that wrote
it.

Jitter
------
Onsets are jittered **around a fixed grid**, not by drawing inter-onset
intervals.  The difference matters.  Drawing intervals lets the sequence
random-walk, so the rate drifts and the number of events in a given time is
itself random; tying each onset to ``k * period`` and displacing it by at most
J keeps the long-run rate *exactly* at nominal while destroying isochrony.
The inter-onset interval then lands on a triangular distribution over
``period +- 2J``, which is what bounds how close two events can come.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 48_000
RAMP_MS = 5.0
PEAK_DBFS = -3.0


def samples(ms: float) -> int:
    """Milliseconds to samples, checked to be exact."""
    n = ms * SR / 1000.0
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{ms} ms is {n} samples at {SR} Hz -- not exact")
    return int(round(n))


def tone(freq: float, dur_ms: float, ramp_ms: float = RAMP_MS) -> np.ndarray:
    """One gated sine: zero starting phase, raised-cosine on and off."""
    n, r = samples(dur_ms), samples(ramp_ms)
    if 2 * r > n:
        raise ValueError("ramps do not fit in the tone")
    x = np.sin(2.0 * np.pi * freq * np.arange(n) / SR)
    env = np.ones(n)
    k = np.arange(r, dtype=np.float64)
    rise = np.sin(0.5 * np.pi * (k + 0.5) / r) ** 2
    env[:r], env[n - r:] = rise, rise[::-1]
    return x * env


def doublet(f1: float, f2: float, tone_ms: float, gap_ms: float) -> np.ndarray:
    """Two gated tones, the second beginning ``gap_ms`` after the first ends."""
    return np.concatenate([tone(f1, tone_ms),
                           np.zeros(samples(gap_ms)),
                           tone(f2, tone_ms)])


def jittered_onsets(n: int, period_ms: float, jitter_ms: float,
                    rng: np.random.Generator, phase_ms: float = 0.0
                    ) -> np.ndarray:
    """Onset samples: ``k * period + phase``, displaced by up to ``jitter``.

    Uniform on ``[-jitter, +jitter]`` and independent per event, so the mean
    rate is exactly ``1000 / period`` however large the jitter is.
    """
    grid = np.arange(n) * period_ms + phase_ms
    if jitter_ms > 0:
        grid = grid + rng.uniform(-jitter_ms, jitter_ms, size=n)
    return np.round(grid * SR / 1000.0).astype(int)


def place(onsets: np.ndarray, event: np.ndarray, total: int) -> np.ndarray:
    """Add one event at each onset, into a buffer of ``total`` samples."""
    out = np.zeros(total)
    for o in onsets:
        out[o:o + event.size] += event[:max(0, min(event.size, total - o))]
    return out


#: One scale factor for every file in the set, so that a tone is at the same
#: level wherever it appears.  Peak-normalising each file instead would tie
#: its level to whatever its loudest moment happened to be -- in a two-stream
#: file that is the rare instant when both streams sum, which would drag every
#: tone down 6 dB relative to the single-stream files and make the set
#: unusable for comparing them.  Equal SPL per tone is the convention.
#:
#: 0.354 is 10^(-3/20) / 2: a lone tone peaks at -9 dBFS, and two coincident
#: tones at -3 dBFS, so nothing in the set can clip.
TONE_AMP = 0.354


def scale(x: np.ndarray, amp: float = TONE_AMP) -> np.ndarray:
    """Apply the shared level.  Tones are generated at unit amplitude."""
    peak = float(np.max(np.abs(x)) * amp)
    if peak >= 1.0:
        raise ValueError(f"clipping: peak would be {peak:.3f}")
    return x * amp


def normalise(x: np.ndarray, peak_dbfs: float = PEAK_DBFS) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    return x * (10.0 ** (peak_dbfs / 20.0) / m) if m > 0 else x


def envelope(x: np.ndarray, win_ms: float = 2.0) -> np.ndarray:
    """Sliding maximum of |x|: longer than one carrier cycle so it does not
    follow the carrier, short enough to resolve the notch inside a doublet.

    A *sliding* window, not a blocked one.  Reshaping into fixed blocks ties
    the estimate to absolute sample positions, so a jittered onset lands at a
    different phase within the blocks each time and the same doublet is
    sometimes read as one event and sometimes as two.  That is an artefact of
    the measurement, and it is exactly the kind that would otherwise be
    reported as jitter in the signal.
    """
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(np.abs(x), size=samples(win_ms), mode="constant")


def events(x: np.ndarray, floor_db: float = -40.0) -> tuple:
    """Start and stop samples of every run above ``floor_db`` of peak."""
    e = envelope(x)
    on = e > 10 ** (floor_db / 20.0) * e.max()
    d = np.diff(np.concatenate([[0], on.view(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def write_wav24(path: Path, x: np.ndarray) -> Path:
    """24-bit PCM, two identical channels (diotic)."""
    q = np.clip(np.round(x * (2 ** 23 - 1)), -(2 ** 23), 2 ** 23 - 1)
    b = q.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(np.repeat(b[:, None, :], 2, axis=1).tobytes())
    return path


def to_mp3(wav: Path, mp3: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)], check=True)
    return mp3


def render(stem: Path, x: np.ndarray, keep_wav: bool = False) -> Path:
    wav = write_wav24(stem.with_suffix(".wav"), x)
    mp3 = to_mp3(wav, stem.with_suffix(".mp3"))
    if not keep_wav:
        wav.unlink()
    return mp3


def levels(x: np.ndarray) -> str:
    """Peak, RMS while sounding, and RMS over the whole file.

    The two RMS figures answer different questions and both are worth having.
    RMS-while-sounding is the level of a tone, which is what should match
    across a stimulus set; whole-file RMS also counts the silence, so it
    tracks how loud the file seems, and a file with twice the tone density is
    legitimately louder even when its tones are at the same SPL.
    """
    body = x[np.abs(x) > 1e-6]
    return (f"peak {20 * np.log10(np.max(np.abs(x))):+.2f} dBFS | "
            f"RMS sounding {20 * np.log10(np.sqrt(np.mean(body ** 2))):+.2f}"
            f" dBFS | RMS overall "
            f"{20 * np.log10(np.sqrt(np.mean(x ** 2))):+.2f} dBFS")
