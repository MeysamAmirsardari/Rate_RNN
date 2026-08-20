"""
audios.two_stream
=================

Two doublet streams, AB and CD, that share a frequency region and are held
apart by **onset incoherence alone**.

    ab_cd_incoherent.mp3

The frequencies
---------------
All four tones are one semitone apart, in a chain:

    A 1000.00 Hz  ->  B 1059.46 Hz  ->  C 1122.46 Hz  ->  D 1189.21 Hz
                1 st              1 st              1 st

So AB is close to itself, CD is close to itself, and B to C is the same
1 semitone.  The four span three semitones in total, which is far too little
for frequency separation to segregate them.  That is deliberate: it removes
the usual cue, so anything heard as two streams has to come from the timing.

The timing
----------
Both streams run at a mean rate of 5 Hz, and they are **anti-phase**: CD's
grid sits half a period -- 100 ms -- from AB's.  For two streams at the same
rate, half a period is the furthest apart their onsets can be, so this is
maximal incoherence rather than merely some.

On top of that each stream is jittered independently, so the relative phase
also wanders rather than sitting at a fixed offset.

Why the jitter is smaller here (+-25 ms, against +-50 ms in ``two_tone``)
-------------------------------------------------------------------------
Because it has a job to do that unbounded jitter would undo.  With a 100 ms
nominal offset and independent jitter of +-J, the asynchrony between the
streams runs over ``100 +- 2J``.  At J = 50 that reaches zero: the two streams
would occasionally start together, and simultaneous onsets fuse -- which is
the one thing this file must never do.  At J = 25 the asynchrony is bounded to
50-150 ms, so the streams are never closer than 50 ms.

50 ms is comfortably past the point where onset asynchrony works as a
segregation cue; asynchronies beyond roughly 30 ms are enough to stop two
components fusing into one event.

What to listen for
------------------
Whether it splits into two interleaved streams or stays one dense warble --
and whether that flips back and forth, which is what bistable streaming does.
The control is ``two_tone``'s ``ab_df01``: the same 1-semitone doublet on its
own, with nothing to segregate from.

Run
---
    python -m audios.two_stream
    python -m audios.two_stream --jitter-ms 10   # tighter, more regular
    python -m audios.two_stream --phase-ms 0     # coherent: should fuse
    python -m audios.two_stream --keep-wav
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
JITTER_MS = 25.0
PHASE_MS = 100.0            # half a period: maximal incoherence
N_PAIRS = 60
F_A = 1000.0
DF_ST = 1.0                 # every step in the chain
LEAD_MS = 500.0
TAIL_MS = 700.0


def frequencies() -> tuple:
    return tuple(F_A * 2.0 ** (k * DF_ST / 12.0) for k in range(4))


def build(jitter_ms: float = JITTER_MS, phase_ms: float = PHASE_MS,
          seed: int = 0) -> dict:
    fa, fb, fc, fd = frequencies()
    ab = core.doublet(fa, fb, TONE_MS, GAP_MS)
    cd = core.doublet(fc, fd, TONE_MS, GAP_MS)
    period = 1000.0 / PAIR_HZ

    rng = np.random.default_rng(seed)
    on_ab = core.jittered_onsets(N_PAIRS, period, jitter_ms, rng,
                                 phase_ms=LEAD_MS)
    on_cd = core.jittered_onsets(N_PAIRS, period, jitter_ms, rng,
                                 phase_ms=LEAD_MS + phase_ms)

    total = int(max(on_ab[-1], on_cd[-1])) + ab.size + core.samples(TAIL_MS)
    x_ab = core.place(on_ab, ab, total)
    x_cd = core.place(on_cd, cd, total)
    return dict(x=x_ab + x_cd, ab=x_ab, cd=x_cd,
                on_ab=on_ab, on_cd=on_cd, freqs=(fa, fb, fc, fd))


def report(b: dict) -> str:
    fa, fb, fc, fd = b["freqs"]
    ms = 1000.0 / core.SR
    a, c = b["on_ab"] * ms, b["on_cd"] * ms

    # nearest AB onset to every CD onset, and vice versa
    asyn = np.concatenate([np.min(np.abs(c[:, None] - a[None, :]), axis=1),
                           np.min(np.abs(a[:, None] - c[None, :]), axis=1)])
    soa_a, soa_c = np.diff(a), np.diff(c)

    # how much of the time both streams are sounding at once
    on_a = core.envelope(b["ab"]) > 1e-3 * np.max(np.abs(b["ab"]))
    on_c = core.envelope(b["cd"]) > 1e-3 * np.max(np.abs(b["cd"]))
    both = float(np.mean(on_a & on_c))

    return (
        f"  A {fa:7.2f}   B {fb:7.2f}   C {fc:7.2f}   D {fd:7.2f} Hz   "
        f"(each step {DF_ST:.0f} st, span "
        f"{12 * np.log2(fd / fa):.0f} st)\n"
        f"  AB: {len(a)} doublets, inter-onset {soa_a.mean():.1f} +- "
        f"{soa_a.std():.1f} ms -> {1000 / soa_a.mean():.3f} Hz\n"
        f"  CD: {len(c)} doublets, inter-onset {soa_c.mean():.1f} +- "
        f"{soa_c.std():.1f} ms -> {1000 / soa_c.mean():.3f} Hz\n"
        f"  onset asynchrony between streams: {asyn.mean():.1f} +- "
        f"{asyn.std():.1f} ms, minimum {asyn.min():.1f} ms "
        f"(needs > ~30 ms to segregate)\n"
        f"  both streams sounding at once {both * 100:.0f}% of the time\n"
        f"  {core.levels(b['x'])}")


def figure(b: dict, stem: str = "two_stream_check") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    lead = core.samples(LEAD_MS)
    show = lead + core.samples(1400.0)
    ts = (np.arange(show - lead)) / core.SR * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.4),
                             constrained_layout=True)
    ax = axes[0]
    ax.plot(ts, b["ab"][lead:show], lw=0.5, color="#BD6B6B", label="AB")
    ax.plot(ts, b["cd"][lead:show] - 1.6, lw=0.5, color="#2166AC", label="CD")
    ax.set_yticks([0, -1.6])
    ax.set_yticklabels(["AB", "CD"])
    ax.set_xlim(0, ts[-1])
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("The two streams, drawn apart (they are mixed in the file)")

    ax = axes[1]
    # 1024 points is 21 ms -- half a tone, so the tones stay separate in
    # time, at 47 Hz resolution.  The steps here are 59-67 Hz, so they are
    # only just resolvable: 40 ms tones a semitone apart sit at the
    # time-frequency limit, which is the point of the stimulus rather than a
    # failure of the plot.
    f, tt, S = spectrogram(b["x"][lead:show], fs=core.SR, nperseg=1024,
                           noverlap=1000, window="hann")
    S_db = 10 * np.log10(S + 1e-14)
    top = S_db.max()
    ax.pcolormesh(tt * 1000.0, f, S_db, shading="gouraud", cmap="magma",
                  vmin=top - 55, vmax=top)
    ax.set_ylim(880, 1360)
    ax.set_xlim(0, ts[-1])
    for hz, col in zip(b["freqs"], ["#BD6B6B", "#BD6B6B", "#2166AC", "#2166AC"]):
        ax.axhline(hz, color=col, lw=0.6, ls=":", alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")

    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jitter-ms", type=float, default=JITTER_MS)
    p.add_argument("--phase-ms", type=float, default=PHASE_MS)
    p.add_argument("--keep-wav", action="store_true")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    print(f"{core.SR} Hz | tone {TONE_MS:.0f} ms, ramp {core.RAMP_MS:.0f} ms "
          f"cos^2 | both streams {PAIR_HZ} Hz | CD offset "
          f"{args.phase_ms:.0f} ms | jitter +-{args.jitter_ms:.0f} ms\n")

    b = build(args.jitter_ms, args.phase_ms)
    b["x"] = core.scale(b["x"])
    name = "ab_cd_incoherent" if args.phase_ms else "ab_cd_coherent"
    mp3 = core.render(OUT_DIR / name, b["x"], args.keep_wav)
    print(report(b))
    print(f"  -> {mp3.name}")
    if not args.no_figure:
        print(f"  -> {figure(b).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
