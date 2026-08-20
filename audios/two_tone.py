"""
audios.two_tone
===============

One stream of AB doublets, at two frequency separations, with **jittered
onsets**.

    file1  ab_df01.mp3   df = 1 semitone
    file2  ab_df09.mp3   df = 9 semitones

The sequence
------------
    A  0-40 ms      B  40-80 ms      then silence until the next doublet

The doublet repeats at a mean rate of 5 Hz, and the 0 ms gap is the one
*inside* it -- B begins the sample after A ends.  (A 5 Hz tone rate would put
160 ms between tones, contradicting the 0 ms gap; abutting 40 ms tones with no
silence would be a 25 Hz tone rate.)

Jitter
------
Each doublet onset is displaced uniformly by up to +-50 ms from its grid
position, so inter-onset intervals run over 100-300 ms with a mean of exactly
200 ms.  Tying the jitter to the grid rather than drawing intervals is what
keeps the **overall rate exactly 5 Hz** -- drawn intervals would random-walk
and the rate would drift.

+-50 ms is a quarter of the period, which is as far as the jitter can go
before doublets start to collide: the minimum interval, 200 - 2 x 50 = 100 ms,
still clears the 80 ms doublet.

What the jitter costs, stated because it is not free: isochrony is itself a
grouping cue, and removing it weakens streaming build-up a little.  What it
buys is that nothing can be following the rhythm instead of the frequencies.

Acoustic details
----------------
**Ramps.**  5 ms raised-cosine (cos-squared) on and off, leaving 30 ms at full
amplitude.  A rectangular gate on a 40 ms tone splatters energy across the
spectrum and clicks at both ends.

**The A-to-B junction.**  The tones abut, so A's fall runs into B's rise and
there is a 10 ms notch rather than a switch.  Each tone is gated
independently on purpose: cross-fading to hold amplitude constant would remove
the cue the paradigm is about.

**Phase.**  Every tone starts at sine phase zero.

**Frequencies.**  A is fixed at 1000 Hz in both files and only B moves, so the
two files share an identical A and the separation is the only difference.

**Level.**  Equal amplitude per tone; each file peak-normalised to -3 dBFS, so
the two are identical in peak and RMS.  Equal SPL is not equal loudness -- the
ear is 1-2 phon more sensitive at 1682 Hz than at 1000 Hz -- but correcting
that would put a level difference between the files.

Run
---
    python -m audios.two_tone
    python -m audios.two_tone --df 1 3 6 9
    python -m audios.two_tone --jitter-ms 0     # the isochronous version
    python -m audios.two_tone --keep-wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from . import core
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audios import core  # type: ignore

OUT_DIR = Path(__file__).resolve().parent

TONE_MS = 40.0
GAP_MS = 0.0
PAIR_HZ = 5.0
JITTER_MS = 50.0
N_PAIRS = 60
F_A = 1000.0
LEAD_MS = 500.0
TAIL_MS = 700.0


def build(df_semitones: float, jitter_ms: float = JITTER_MS,
          seed: int = 0) -> tuple:
    f_b = F_A * 2.0 ** (df_semitones / 12.0)
    ev = core.doublet(F_A, f_b, TONE_MS, GAP_MS)
    period = 1000.0 / PAIR_HZ

    rng = np.random.default_rng(seed)
    on = core.jittered_onsets(N_PAIRS, period, jitter_ms, rng,
                              phase_ms=LEAD_MS)
    total = int(on[-1]) + ev.size + core.samples(TAIL_MS)
    return core.place(on, ev, total), f_b, on


def report(df: float, x: np.ndarray, f_b: float, on: np.ndarray) -> str:
    starts, stops = core.events(x)
    soa = np.diff(starts) / core.SR * 1000.0
    dur = (stops - starts) / core.SR * 1000.0
    dur_s = x.size / core.SR
    return (
        f"df {df:>4.1f} st | A {F_A:7.2f} Hz   B {f_b:7.2f} Hz   "
        f"ratio {f_b / F_A:.4f}\n"
        f"            {len(starts)} doublets (expected {N_PAIRS}), each "
        f"{dur.mean():.1f} +- {dur.std():.2f} ms, over {dur_s:.2f} s\n"
        f"            inter-onset {soa.mean():.1f} +- {soa.std():.1f} ms "
        f"(range {soa.min():.0f}-{soa.max():.0f}) -> mean rate "
        f"{1000.0 / soa.mean():.3f} Hz; silence between doublets >= "
        f"{(soa.min() - dur.mean()):.0f} ms\n"
        f"            {core.levels(x)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--df", type=float, nargs="*", default=[1.0, 9.0])
    p.add_argument("--jitter-ms", type=float, default=JITTER_MS)
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)

    print(f"{core.SR} Hz | tone {TONE_MS:.0f} ms, ramp {core.RAMP_MS:.0f} ms "
          f"cos^2 | A-B gap {GAP_MS:.0f} ms | doublet mean rate {PAIR_HZ} Hz, "
          f"onset jitter +-{args.jitter_ms:.0f} ms\n")

    for df in args.df:
        x, f_b, on = build(df, args.jitter_ms)
        x = core.scale(x)
        mp3 = core.render(OUT_DIR / f"ab_df{int(round(df)):02d}", x,
                          args.keep_wav)
        print(report(df, x, f_b, on))
        print(f"            -> {mp3.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
