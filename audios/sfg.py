"""
audios.sfg
==========

The classic figure, and the same figure sheared into a staircase.

    sfg_coherent.mp3   five tones, all together, every 200 ms
    sfg_stair10.mp3    the same five, delayed 0 10 20 30 40 ms
    sfg_switch.mp3     coherent for twelve seconds, then the staircase,
                       one cloud across the join
    sfg_check.png

The stimulus
------------
A **figure** of five tones on a semitone grid, five semitones apart, repeating
at **5 Hz** inside a balanced cloud.  Nothing about it moves except the lag of
each tone behind the first:

    coherent    0  0  0  0  0 ms     one chord, the classic figure
    staircase   0 10 20 30 40 ms     the same chord sheared

Forty-millisecond tones against a ten-millisecond step means successive tones
still overlap by 30 ms, so the staircase never becomes a sequence of separate
events -- it is the *same* event with its onsets pulled apart.  The whole
figure then spans 80 ms of each 200 ms period rather than 40.

Nothing else differs.  Same five frequencies, same five tones per token, same
token rate, same cloud, same per-tone level.  Total energy is identical; only
the peak differs, and only because coincident tones sum.

Why a switch file
-----------------
Two separate files ask "can you find the figure in this one?" twice, and the
answer depends as much on how long you listened as on the stimulus.  One file
that changes partway asks the question that matters -- whether the figure you
are already holding onto survives the shear -- and the listener is their own
control.  The cloud is scheduled once, across the join, so nothing in the
background marks the moment.

The cloud, unchanged from ``audios.word``
-----------------------------------------
It sounds the figure's own channels, so frequency identifies nothing; every
channel is used equally in total and through the stream; the total number of
tones sounding never moves, so the envelope marks nothing.  See
``audios/README.md`` for why each of those is necessary.

The one thing that is *not* carried over is the word-onset jitter.  This is
the classic figure and it repeats isochronously, which is a real cue -- the
figure can be tracked by rhythm as well as by pattern.  That is deliberate
here, since the question is about the shear rather than about how hard the
figure is to find; ``--jitter-ms`` turns it on for the harder version.

Run
---
    python -m audios.sfg
    python -m audios.sfg --step-ms 20 --jitter-ms 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from . import cloud, core
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audios import cloud, core  # type: ignore

OUT_DIR = Path(__file__).resolve().parent

F_REF = 1000.0
POOL_ST = (-24.0, 36.0)        # 250 Hz to 8 kHz

#: Twenty-one channels, three semitones apart, and the count is forced rather
#: than chosen.  Every channel has to be used at the same rate, and a figure
#: channel is used ``5/s`` by the figure alone -- once per token at 5 Hz.  The
#: cloud supplies ``ceiling / tone`` tones per second across ``C`` channels, so
#: balance needs ``ceiling / (tone * C) > 5``.  At the sixty-one channel pool
#: of ``audios.word`` that needs a ceiling of fourteen; the figure's channels
#: are otherwise used twice as often as the rest, which identifies the figure
#: without listening to it at all.  Twenty-one channels at a ceiling of six
#: gives 7.1 tones per channel per second, so the cloud still supplies 30% of
#: what sounds in a figure channel and the background stays thin.
GRID_ST = 3.0
TONE_MS = 40.0

RATE_HZ = 5.0                  # figure repetition
N_TONES = 5
SPAN_ST = 24.0                 # 5 tones, 6 semitones apart, on the grid
BASE_ST = -12.0
STEP_MS = 10.0                 # the shear

N_TOKENS = 60                  # 12 s per condition at 5 Hz
CEILING = 6
SCHED_STEP_MS = 2.5
LEAD_MS, TAIL_MS = 400.0, 600.0


def tokens(step_ms: float, n_tokens: int, phase_ms: float,
           n_tones: int = N_TONES, jitter_ms: float = 0.0,
           rng: np.random.Generator | None = None) -> list:
    """(semitone, onset sample, token index) for one condition.

    ``jitter_ms`` displaces whole tokens, in whole tone lengths so the cloud's
    tiling still lands on them; the figure's internal shape never changes.
    """
    d_st = SPAN_ST / max(n_tones - 1, 1)
    st = [BASE_ST + k * d_st for k in range(n_tones)]
    period = 1000.0 / RATE_HZ
    q = core.samples(TONE_MS)
    k_max = int(core.samples(jitter_ms) // q)

    ev = []
    for t in range(n_tokens):
        o0 = core.samples(phase_ms + t * period)
        if k_max and rng is not None:
            o0 += int(rng.integers(-k_max, k_max + 1)) * q
        for k in range(n_tones):
            ev.append((st[k], o0 + core.samples(k * step_ms), t))
    return ev


def render(ev: list, tail_ms: float = TAIL_MS) -> np.ndarray:
    total = max(o for _, o, _ in ev) + core.samples(TONE_MS + tail_ms)
    x = np.zeros(total)
    pips: dict = {}
    for st, o, _t in ev:
        f = F_REF * 2.0 ** (st / 12.0)
        if st not in pips:
            pips[st] = core.tone(f, TONE_MS)
        x[o:o + pips[st].size] += pips[st]
    return x


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-ms", type=float, default=STEP_MS)
    p.add_argument("--n-tokens", type=int, default=N_TOKENS)
    p.add_argument("--n-tones", type=int, default=N_TONES)
    p.add_argument("--jitter-ms", type=float, default=0.0,
                   help="displace whole tokens, in units of the tone length")
    p.add_argument("--ceiling", type=int, default=CEILING)
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)

    period = 1000.0 / RATE_HZ
    S, N, nt = args.step_ms, args.n_tokens, args.n_tones
    rng = np.random.default_rng(3)

    # the two conditions, and the switch that runs one into the other
    coh = tokens(0.0, N, LEAD_MS, nt, args.jitter_ms, rng)
    stair = tokens(S, N, LEAD_MS, nt, args.jitter_ms, rng)
    join = LEAD_MS + N * period
    switch = tokens(0.0, N, LEAD_MS, nt, args.jitter_ms, rng) + \
        tokens(S, N, join, nt, args.jitter_ms, rng)

    cases = [("sfg_coherent", coh, "coherent, 0 ms"),
             (f"sfg_stair{S:g}", stair, f"staircase, {S:g} ms step"),
             ("sfg_switch", switch, f"coherent -> {S:g} ms at {join / 1000:.1f} s")]

    freqs = cloud.channels([], f_ref=F_REF, st_lo=POOL_ST[0],
                           st_hi=POOL_ST[1], grid_st=GRID_ST)
    grid = 12.0 * np.log2(freqs / F_REF)
    chan = {int(round(v)): k for k, v in enumerate(grid)}
    n_tone = core.samples(TONE_MS)

    # One ceiling for every file, taken across all of them: the coherent
    # figure peaks at five simultaneous tones and the staircase at two, so a
    # per-file ceiling would make the background density itself the difference
    # between the conditions.
    xs = {nm: render(ev) for nm, ev, _ in cases}
    peak = max(int(cloud.concurrency([o for _, o, _ in ev], xs[nm].size,
                                     n_tone).max())
               for nm, ev, _ in cases)
    peak = max(peak, args.ceiling)

    clouds = {nm: cloud.schedule(
        xs[nm].size, freqs, [(chan[int(round(st))], o) for st, o, _ in ev],
        peak, tone_ms=TONE_MS, step_ms=SCHED_STEP_MS)
        for nm, ev, _ in cases}

    print(f"{core.SR} Hz | {nt}-tone figure at {RATE_HZ:g} Hz "
          f"({period:.0f} ms), tones {TONE_MS:.0f} ms, "
          f"{SPAN_ST / (nt - 1):.0f} semitones apart")
    print(f"          shear {S:g} ms per tone -> figure spans "
          f"{(nt - 1) * S + TONE_MS:.0f} ms, successive tones overlap by "
          f"{TONE_MS - S:.0f} ms")
    print(f"          {freqs.size} cloud channels "
          f"{freqs[0]:.0f}-{freqs[-1]:.0f} Hz, ceiling {peak}, "
          f"token jitter +-{args.jitter_ms:g} ms\n")

    mixes = {nm: xs[nm] + clouds[nm]["x"][:xs[nm].size] for nm, _, _ in cases}
    amp = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(x))) for x in xs.values())
    amp_c = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(m))) for m in mixes.values())

    for nm, ev, ttl in cases:
        core.render(OUT_DIR / f"{nm}_alone", xs[nm] * amp, args.keep_wav)
        core.render(OUT_DIR / nm, mixes[nm] * amp_c, args.keep_wav)
        T = xs[nm].size
        lo, hi = core.samples(LEAD_MS), T - core.samples(TAIL_MS)
        tot = (cloud.concurrency([o for _, o, _ in ev], T, n_tone)
               + cloud.concurrency([o for _, o in clouds[nm]["events"]],
                                   T, n_tone))[lo:hi]
        c = clouds[nm]["counts"]
        isfig = clouds[nm]["fig_counts"] > 0
        print(f"  {nm}  ({ttl})")
        print(f"    {len(ev)} figure tones, {T / core.SR:.1f} s; "
              f"{core.levels(xs[nm] * amp)}")
        print(f"    concurrency {tot.min()}-{tot.max()}, mean "
              f"{tot.mean():.2f} +- {tot.std():.2f}, deepest dip "
              f"{10 * np.log10(peak / max(tot.min(), 1)):.1f} dB for "
              f"{np.mean(tot < peak - 1) * 100:.1f}% of the time")
        print(f"    channel use {c.min()}-{c.max()}: figure "
              f"{c[isfig].min()}-{c[isfig].max()} ({isfig.sum()} ch), other "
              f"{c[~isfig].min()}-{c[~isfig].max()} ({(~isfig).sum()} ch)")
        print(f"    -> {nm}.mp3   {nm}_alone.mp3\n")

    return figure_png(cases, clouds, join, S)


def figure_png(cases, clouds, join_ms: float, step_ms: float) -> int:
    """Two conditions side by side, and the join drawn across the switch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dur = core.samples(TONE_MS)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True,
                             constrained_layout=True)

    windows = [(LEAD_MS - 100.0, 1400.0), (LEAD_MS - 100.0, 1400.0),
               (join_ms - 700.0, 1400.0)]
    for ax, (nm, ev, ttl), (t_lo, span_ms) in zip(axes, cases, windows):
        t0, t1 = core.samples(t_lo), core.samples(t_lo + span_ms)
        for f, o in clouds[nm]["events"]:
            if t0 <= o < t1:
                y = 12 * np.log2(f / F_REF)
                ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR], [y, y],
                        color="#111111", lw=2.4, solid_capstyle="butt")
        for st, o, _t in ev:
            if t0 <= o < t1:
                ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR],
                        [st, st], color="#E8121A", lw=3.0,
                        solid_capstyle="butt")
        if nm == "sfg_switch":
            ax.axvline((core.samples(join_ms) - t0) / core.SR, color="#1B7837",
                       lw=1.6, ls="--", zorder=6)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, span_ms / 1000.0)
        ax.set_ylim(POOL_ST[0] - 2, POOL_ST[1] + 2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(f"Semitones re {F_REF:.0f} Hz")

    out = OUT_DIR / "sfg_check.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
