"""
audios.two_tone
===============

Repeating AB doublets, for listening.  Two files that differ in one thing
only: the frequency separation between A and B.

    file1  ab_df01.mp3   df = 1 semitone
    file2  ab_df09.mp3   df = 9 semitones

The sequence
------------
The specification was "rate 5 Hz, tone duration 40 ms, tone gap 0 ms, A then
B".  Those three numbers only fit together one way, and it is worth being
explicit about which:

    A  0-40 ms      B  40-80 ms      silence 80-200 ms      then repeat

The **doublet** repeats at 5 Hz, i.e. every 200 ms.  The 0 ms gap is the gap
*inside* the doublet -- B begins the sample after A ends.  (Read the other
way, a 5 Hz tone rate would put 160 ms between tones, which contradicts the
0 ms gap; and abutting 40 ms tones with no silence at all would be a 25 Hz
tone rate, not 5 Hz.)

Every duration is an exact number of samples at 48 kHz -- 40 ms = 1920,
200 ms = 9600, 5 ms = 240 -- so nothing is rounded and every onset lands on a
sample boundary.

Acoustic details, and why each one is what it is
------------------------------------------------
**Ramps.**  Each tone is gated with a 5 ms raised-cosine (cos-squared) ramp on
and off, leaving 30 ms at full amplitude.  A rectangular gate on a 40 ms tone
splatters energy across the spectrum and is audible as a click at both ends;
5 ms is the usual compromise for tones this short, long enough to suppress
the splatter and short enough to leave a clear steady state.

**The A-to-B junction.**  Because the tones abut, A's 5 ms fall runs straight
into B's 5 ms rise, so there is a 10 ms dip in amplitude at the boundary
rather than an instantaneous switch.  This is deliberate: each tone is gated
independently, which is what makes A and B two events rather than one
frequency-stepped event.  Cross-fading to hold the amplitude constant would
remove exactly the cue the paradigm is about.

**Phase.**  Every tone starts at sine phase zero, so the waveform leaves
silence from zero with zero slope once the ramp is applied.  No DC step, no
transient beyond the ramp itself.

**Frequencies.**  A is fixed at 1000 Hz in both files and only B moves, so the
two files share an identical A and the separation is the only difference.
1000 Hz sits in the flat part of the equal-loudness contours and is the
conventional reference.

    df = 1 st   B = 1000 * 2^(1/12)  = 1059.46 Hz
    df = 9 st   B = 1000 * 2^(9/12)  = 1681.79 Hz

**Level.**  Both tones are generated at the same amplitude and each file is
peak-normalised to -3 dBFS, so the two files are identical in peak and in RMS
and differ only in frequency content.  One caveat that normalisation cannot
fix: equal SPL is not equal loudness.  At 1682 Hz the ear is roughly 1-2 phon
more sensitive than at 1000 Hz, so the wide-separation file's B tone will
sound very slightly louder than its A.  Correcting that would mean applying an
equal-loudness contour, which is not standard for this paradigm and would put
a level difference between the two files; equal SPL is the convention and the
residual is small.

**Presentation.**  Diotic -- the same signal in both ears -- which is what
headphone presentation of a single stream means.  Sixty doublets, 12 s, which
is long enough for streaming to build up (build-up typically takes 5-10 s).

**Format.**  The masters are 24-bit 48 kHz WAV.  The MP3s are 320 kbps CBR
from those masters.  MP3 is lossy and adds a little pre-echo around sharp
onsets; at 320 kbps with 5 ms ramps it is inaudible, but the WAV is the file
to use if these are ever run as an actual experiment.

Run
---
    python -m audios.two_tone                    # the two files above
    python -m audios.two_tone --df 1 3 6 9       # any set of separations
    python -m audios.two_tone --keep-wav         # keep the WAV masters
"""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
#  The specification
# ---------------------------------------------------------------------
SR = 48_000            # Hz
TONE_MS = 40           # per tone
RAMP_MS = 5            # raised-cosine, on and off
GAP_MS = 0             # between A and B
PAIR_HZ = 5.0          # doublet repetition rate
N_PAIRS = 60           # 12 s
F_A = 1000.0           # Hz, fixed in every file
LEAD_MS = 500          # silence before the first doublet
TAIL_MS = 700          # silence after the last
PEAK_DBFS = -3.0


def _samples(ms: float) -> int:
    """Milliseconds to samples, checked to be exact."""
    n = ms * SR / 1000.0
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"{ms} ms is {n} samples at {SR} Hz -- not exact")
    return int(round(n))


def tone(freq: float, dur_ms: float = TONE_MS,
         ramp_ms: float = RAMP_MS) -> np.ndarray:
    """One gated sine: zero phase, raised-cosine on and off."""
    n = _samples(dur_ms)
    r = _samples(ramp_ms)
    if 2 * r > n:
        raise ValueError("ramps do not fit in the tone")
    t = np.arange(n, dtype=np.float64) / SR
    x = np.sin(2.0 * np.pi * freq * t)

    env = np.ones(n)
    # cos-squared, i.e. sin^2 rising -- 0 to 1 with zero slope at both ends
    k = np.arange(r, dtype=np.float64)
    rise = np.sin(0.5 * np.pi * (k + 0.5) / r) ** 2
    env[:r] = rise
    env[n - r:] = rise[::-1]
    return x * env


def build(df_semitones: float) -> np.ndarray:
    """The whole sequence for one frequency separation."""
    f_b = F_A * 2.0 ** (df_semitones / 12.0)

    pair = np.concatenate([tone(F_A), np.zeros(_samples(GAP_MS)), tone(f_b)])
    period = _samples(1000.0 / PAIR_HZ)
    if pair.size > period:
        raise ValueError("the doublet is longer than its repetition period")

    slot = np.zeros(period)
    slot[:pair.size] = pair

    return np.concatenate([np.zeros(_samples(LEAD_MS)),
                           np.tile(slot, N_PAIRS),
                           np.zeros(_samples(TAIL_MS))])


def normalise(x: np.ndarray, peak_dbfs: float = PEAK_DBFS) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    return x * (10.0 ** (peak_dbfs / 20.0) / m) if m > 0 else x


def write_wav24(path: Path, x: np.ndarray) -> Path:
    """24-bit PCM, two identical channels (diotic)."""
    q = np.clip(np.round(x * (2 ** 23 - 1)), -(2 ** 23), 2 ** 23 - 1)
    q = q.astype("<i4")
    b = q.view(np.uint8).reshape(-1, 4)[:, :3]          # little-endian, drop MSB
    frames = np.repeat(b[:, None, :], 2, axis=1).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(frames)
    return path


def to_mp3(wav: Path, mp3: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
         "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)],
        check=True)
    return mp3


def measure(x: np.ndarray) -> dict:
    """Recover the timing from the signal itself, rather than trusting it.

    The envelope is a running maximum over 1 ms -- longer than one cycle of
    the lowest tone, so it does not follow the carrier, and short enough to
    resolve the 10 ms notch where A meets B.  Thresholding the raw waveform
    would count zero crossings instead of tones.

    Note what this finds: at a 0 ms gap the doublet is ONE 80 ms acoustic
    event with a notch in the middle, not two separate sounds.  The notch is
    measured rather than assumed, because how deep and how wide it is
    determines whether the ear hears two tones or one frequency step.
    """
    n_win = _samples(1.0)
    pad = (-x.size) % n_win
    env = np.abs(np.concatenate([x, np.zeros(pad)])).reshape(-1, n_win).max(1)
    env = np.repeat(env, n_win)[:x.size]
    peak = env.max()

    on = env > 10 ** (-40 / 20) * peak
    d = np.diff(np.concatenate([[0], on.view(np.int8), [0]]))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)

    # the notch: the envelope minimum in the middle third of each event
    notch_db, notch_ms = [], []
    for s0, s1 in zip(starts, stops):
        mid = env[s0 + (s1 - s0) // 3: s1 - (s1 - s0) // 3]
        if mid.size:
            notch_db.append(20 * np.log10(max(mid.min(), 1e-12) / peak))
            notch_ms.append(np.sum(mid < 0.5 * peak) / SR * 1000.0)

    return dict(n_events=len(starts),
                event_ms=(stops - starts) / SR * 1000.0,
                soa_ms=np.diff(starts) / SR * 1000.0,
                notch_db=np.array(notch_db), notch_ms=np.array(notch_ms))


def report(df: float, x: np.ndarray) -> str:
    f_b = F_A * 2.0 ** (df / 12.0)
    m = measure(x)
    body = x[np.abs(x) > 1e-6]
    rms = float(np.sqrt(np.mean(body ** 2)))

    def sd(v, unit="ms"):
        return f"{v.mean():.1f} +- {v.std():.3f} {unit}"

    return (
        f"df {df:>4.1f} st | A {F_A:7.2f} Hz   B {f_b:7.2f} Hz   "
        f"ratio {f_b / F_A:.4f}\n"
        f"            {m['n_events']} doublets (expected {N_PAIRS}); "
        f"each {sd(m['event_ms'])} long, one every {sd(m['soa_ms'])}\n"
        f"            A-B notch: down to {m['notch_db'].mean():.0f} dB, "
        f"{sd(m['notch_ms'])} below half amplitude\n"
        f"            {x.size / SR:5.2f} s | peak "
        f"{20 * np.log10(np.max(np.abs(x))):+.2f} dBFS | "
        f"RMS {20 * np.log10(rms):+.2f} dBFS")


def figure(dfs, xs, stem: str = "two_tone_check") -> Path:
    """Waveform and spectrogram of the first second, as a visual check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    show = _samples(LEAD_MS) + _samples(1000.0)
    fig, axes = plt.subplots(2, len(dfs), figsize=(4.6 * len(dfs), 5.0),
                             constrained_layout=True)
    axes = np.atleast_2d(axes.reshape(2, -1))

    for c, (df, x) in enumerate(zip(dfs, xs)):
        seg = x[_samples(LEAD_MS):show]
        ts = np.arange(seg.size) / SR * 1000.0

        ax = axes[0, c]
        ax.plot(ts, seg, lw=0.4, color="#2D3748")
        ax.set_title(f"df = {df:g} semitone{'s' if df != 1 else ''}  "
                     f"(B = {F_A * 2 ** (df / 12):.0f} Hz)")
        ax.set_ylabel("amplitude" if c == 0 else "")
        ax.set_xlim(0, 620)
        ax.set_ylim(-0.75, 0.75)
        for k in range(4):
            ax.axvspan(k * 200, k * 200 + 40, color="#BD6B6B", alpha=0.18, lw=0)
            ax.axvspan(k * 200 + 40, k * 200 + 80, color="#2166AC",
                       alpha=0.18, lw=0)

        ax = axes[1, c]
        f, tt, S = spectrogram(seg, fs=SR, nperseg=1024, noverlap=960,
                               window="hann")
        ax.pcolormesh(tt * 1000.0, f, 10 * np.log10(S + 1e-14),
                      shading="gouraud", cmap="magma", vmin=-140, vmax=-50)
        ax.set_ylim(400, 2400)
        ax.set_xlim(0, 620)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)" if c == 0 else "")
        for hz in (F_A, F_A * 2 ** (df / 12)):
            ax.axhline(hz, color="w", lw=0.4, ls=":", alpha=0.5)

    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--df", type=float, nargs="*", default=[1.0, 9.0],
                   help="frequency separations in semitones")
    p.add_argument("--keep-wav", action="store_true")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    print(f"{SR} Hz | tone {TONE_MS} ms ({_samples(TONE_MS)} samples), "
          f"ramp {RAMP_MS} ms cos^2 | A-B gap {GAP_MS} ms | "
          f"doublet every {1000 / PAIR_HZ:.0f} ms ({PAIR_HZ} Hz)\n")

    made = []
    for df in args.df:
        x = normalise(build(df))
        made.append(x)
        stem = OUT_DIR / f"ab_df{int(round(df)):02d}"
        wav = write_wav24(stem.with_suffix(".wav"), x)
        mp3 = to_mp3(wav, stem.with_suffix(".mp3"))
        if not args.keep_wav:
            wav.unlink()
        print(report(df, x))
        print(f"          -> {mp3.name}"
              + ("" if args.keep_wav else "  (WAV master removed; "
                                          "--keep-wav to retain)"))
    if not args.no_figure:
        print(f"\n          -> {figure(args.df, made).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
