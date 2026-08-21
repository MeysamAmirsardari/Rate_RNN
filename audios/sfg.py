"""
audios.sfg
==========

The classic figure, and the same figure sheared into a staircase.

    sfg_coherent.mp3   n tones, all together, every 200 ms
    sfg_stair10.mp3    the same n, delayed 0 10 20 ... ms
    sfg_check.png      with --plot

The stimulus
------------
A **figure** of ``--n-tones`` tones on a frequency grid, repeating at 5 Hz
inside a balanced cloud.  Nothing about it moves except the lag of each tone
behind the one below it:

    coherent    0  0  0  0 ... ms     one chord, the classic figure
    staircase   0 10 20 30 ... ms     the same chord sheared

Forty-millisecond tones against a ten-millisecond step means successive tones
still overlap by 30 ms, so the staircase is not a sequence of separate events
-- it is the *same* event with its onsets pulled apart.  ``--order fall``
shears it the other way, highest tone first.

Nothing else differs between the conditions.  Same frequencies, same tones per
token, same token rate, same cloud, same per-tone level.  Total energy is
identical; only the peak differs, and only because coincident tones sum.

Token onsets are jittered
-------------------------
Every token is displaced independently, so the figure never arrives on a beat
and cannot be tracked by rhythm instead of by pattern.  The displacement is in
**whole tone lengths**, which is what keeps the cloud flat: the cloud tiles in
40 ms blocks, and a token landing mid-block leaves the part of that block
before it able to hold only one tone, opening a hole immediately in front of
the figure -- the worst possible place, and deepest in the coherent condition.

At 5 Hz that quantum is coarse against a 200 ms period, so the jitter has few
distinct values; +-40 ms gives three arrival times and five distinct
inter-token intervals, which is enough to destroy isochrony while keeping the
mean rate exact.  ``--jitter-ms 0`` restores the isochronous version.

Keeping the cloud balanced at any size
--------------------------------------
Three things have to hold at once, and two of them fight each other as the
figure grows.  The module solves for them rather than leaving them to a
constant that happened to work at one setting:

**The ceiling cannot be below the figure's own peak.**  A coherent figure of
``n`` tones sounds ``n`` tones at once, and nothing the cloud does can take one
away, so the total is at least ``n`` at that instant -- and to keep the
envelope flat it must be ``n`` everywhere else too.  A ten-tone chord
therefore *requires* a ten-tone background.  This is not a choice; it is what
hiding a ten-tone burst costs.

**Every channel must be used at the same rate.**  A figure channel is used
once per token, so at ``rate`` tones per second, whatever ``n`` is.  The cloud
supplies ``ceiling / tone`` tones per second spread over ``C`` channels, so
the average channel rate is ``ceiling / (tone * C)`` and balance needs

    ceiling / (tone_s * C)  >=  rate

Rearranged, ``C <= ceiling / (tone_s * rate)``.  This is a **bound on the
number of channels**, and it tightens as the figure gets faster, not as it
gets bigger.  Exceed it and the figure's channels are simply busier than the
rest -- which identifies the figure without listening to the timing at all.

**The cloud has to sound the figure's channels.**  If it cannot, every tone at
those frequencies is a figure tone and frequency alone gives the answer.  That
needs headroom above the bound, not merely equality: ``--min-share`` sets how
much of a figure channel's traffic the cloud must supply, and the bound
becomes ``C <= (1 - share) * ceiling / (tone_s * rate)``.

The grid spacing is then the finest that fits inside the bound, so the cloud
stays as spectrally rich as the balance permits.  The figure's own tones are
placed on grid indices, evenly spread across the middle of the pool, so they
are always on the grid however many there are.

Run
---
    python -m audios.sfg --plot
    python -m audios.sfg --n-tones 5 --step-ms 20 --order fall
    python -m audios.sfg --duration 20 --jitter-ms 80 --plot
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
TONE_MS = 40.0
RATE_HZ = 5.0                  # figure repetition

N_TONES = 10
STEP_MS = 10.0                 # the shear, per tone
DURATION_S = 12.0              # per condition
JITTER_MS = 40.0               # token onset, in whole tone lengths
CEILING_MIN = 6
MIN_SHARE = 0.25               # of a figure channel's tones, supplied by cloud
FIG_MARGIN = 0.15              # of the pool left clear above and below

#: Candidate grid spacings, finest first.
GRIDS = tuple(np.arange(0.5, 12.5, 0.5))

SCHED_STEP_MS = 2.5
LEAD_MS, TAIL_MS = 400.0, 600.0


def layout(n_tones: int, rate_hz: float = RATE_HZ, *,
           ceiling: int = 0, min_share: float = MIN_SHARE) -> dict:
    """Solve for the ceiling, the grid and the figure's channels.

    Returns everything downstream needs, plus the numbers behind it so the
    choice can be printed rather than taken on trust.
    """
    tone_s = TONE_MS / 1000.0
    ceil = ceiling or max(n_tones, CEILING_MIN)
    c_max = int(np.floor((1.0 - min_share) * ceil / (tone_s * rate_hz)))
    if c_max < n_tones:
        raise SystemExit(
            f"{n_tones} tones at {rate_hz:g} Hz cannot be balanced: the pool "
            f"would need {n_tones} channels but the rate allows {c_max}.  "
            f"Raise --ceiling (>= {int(np.ceil(n_tones * tone_s * rate_hz / (1 - min_share)))}) "
            f"or lower --min-share.")

    span = POOL_ST[1] - POOL_ST[0]
    grid = next(g for g in GRIDS if int(np.floor(span / g)) + 1 <= c_max)
    freqs = cloud.channels([], f_ref=F_REF, st_lo=POOL_ST[0],
                           st_hi=POOL_ST[1], grid_st=grid)
    st_grid = 12.0 * np.log2(freqs / F_REF)

    # The figure sits on grid indices, evenly spread across the middle of the
    # pool, so its tones are on the grid for any n and the cloud keeps a
    # margin of channels above and below it.
    lo = int(round(FIG_MARGIN * (freqs.size - 1)))
    hi = int(round((1.0 - FIG_MARGIN) * (freqs.size - 1)))
    idx = np.unique(np.round(np.linspace(lo, hi, n_tones)).astype(int))
    if idx.size < n_tones:                      # pool too coarse to separate
        idx = np.arange(lo, lo + n_tones)
    return dict(ceiling=ceil, grid_st=grid, freqs=freqs, st_grid=st_grid,
                fig_idx=idx, fig_st=st_grid[idx], c_max=c_max,
                avg_rate=ceil / (tone_s * freqs.size), rate_hz=rate_hz,
                share=1.0 - rate_hz * freqs.size * tone_s / ceil)


def tokens(lay: dict, step_ms: float, n_tokens: int, phase_ms: float, *,
           order: str = "rise", jitter_ms: float = 0.0,
           rng: np.random.Generator | None = None) -> list:
    """(semitone, onset sample, token index) for one condition.

    ``jitter_ms`` displaces whole tokens, in whole tone lengths so the cloud's
    tiling still lands on them; the figure's internal shape never changes.
    """
    st = list(lay["fig_st"])
    lag = np.arange(len(st)) * step_ms
    if order == "fall":
        lag = lag[::-1]
    elif order != "rise":
        raise ValueError(f"unknown order {order!r}")

    period = 1000.0 / lay["rate_hz"]
    q = core.samples(TONE_MS)
    k_max = int(core.samples(jitter_ms) // q)

    ev = []
    for t in range(n_tokens):
        o0 = core.samples(phase_ms + t * period)
        if k_max and rng is not None:
            o0 += int(rng.integers(-k_max, k_max + 1)) * q
        for k in range(len(st)):
            ev.append((st[k], o0 + core.samples(float(lag[k])), t))
    return ev


def render(ev: list, tail_ms: float = TAIL_MS) -> np.ndarray:
    total = max(o for _, o, _ in ev) + core.samples(TONE_MS + tail_ms)
    x = np.zeros(total)
    pips: dict = {}
    for st, o, _t in ev:
        if st not in pips:
            pips[st] = core.tone(F_REF * 2.0 ** (st / 12.0), TONE_MS)
        x[o:o + pips[st].size] += pips[st]
    return x


def snapshot(cases, clouds, lay, span_ms: float = 1400.0) -> Path:
    """The figure-ground picture: cloud in black, figure in red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dur = core.samples(TONE_MS)
    fig, axes = plt.subplots(1, len(cases), figsize=(5.0 * len(cases), 4.8),
                             sharey=True, constrained_layout=True,
                             squeeze=False)
    t_lo = LEAD_MS - 100.0
    for ax, (nm, ev, ttl) in zip(axes[0], cases):
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
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, span_ms / 1000.0)
        ax.set_ylim(POOL_ST[0] - 2, POOL_ST[1] + 2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel(f"Semitones re {F_REF:.0f} Hz")

    out = OUT_DIR / "sfg_check.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-tones", type=int, default=N_TONES,
                   help="tones in the figure (default 10)")
    p.add_argument("--step-ms", type=float, default=STEP_MS,
                   help="shear per tone; 0 is the coherent chord")
    p.add_argument("--order", choices=("rise", "fall"), default="rise",
                   help="staircase direction: lowest tone first, or highest")
    p.add_argument("--duration", type=float, default=DURATION_S,
                   help="seconds per condition")
    p.add_argument("--jitter-ms", type=float, default=JITTER_MS,
                   help="displace whole tokens, rounded down to a whole "
                        "number of tone lengths; 0 is isochronous")
    p.add_argument("--plot", action="store_true",
                   help="write sfg_check.png, the figure in red on the cloud "
                        "in black")
    p.add_argument("--ceiling", type=int, default=0,
                   help="tones sounding at once; 0 solves for it")
    p.add_argument("--min-share", type=float, default=MIN_SHARE,
                   help="least fraction of a figure channel's tones that the "
                        "cloud must supply")
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)

    lay = layout(args.n_tones, ceiling=args.ceiling,
                 min_share=args.min_share)
    period = 1000.0 / RATE_HZ
    N = max(1, int(round(args.duration * RATE_HZ)))
    S, freqs = args.step_ms, lay["freqs"]
    rng = np.random.default_rng(3)
    kw = dict(order=args.order, jitter_ms=args.jitter_ms, rng=rng)

    cases = [
        ("sfg_coherent", tokens(lay, 0.0, N, LEAD_MS, **kw), "coherent, 0 ms"),
        (f"sfg_stair{S:g}", tokens(lay, S, N, LEAD_MS, **kw),
         f"staircase, {S:g} ms step, {args.order}"),
    ]

    chan = {int(round(v * 2)): k for k, v in enumerate(lay["st_grid"])}
    n_tone = core.samples(TONE_MS)
    xs = {nm: render(ev) for nm, ev, _ in cases}

    # One ceiling for every file, taken across all of them: the coherent
    # figure peaks at n simultaneous tones and the staircase at far fewer, so
    # a per-file ceiling would make the background density itself the
    # difference between the conditions.
    peak = max(int(cloud.concurrency([o for _, o, _ in ev], xs[nm].size,
                                     n_tone).max())
               for nm, ev, _ in cases)
    ceil = max(peak, lay["ceiling"])
    clouds = {nm: cloud.schedule(
        xs[nm].size, freqs, [(chan[int(round(st * 2))], o) for st, o, _ in ev],
        ceil, tone_ms=TONE_MS, step_ms=SCHED_STEP_MS)
        for nm, ev, _ in cases}

    print(f"{core.SR} Hz | {args.n_tones}-tone figure at {RATE_HZ:g} Hz "
          f"({period:.0f} ms), tones {TONE_MS:.0f} ms, "
          f"{args.duration:g} s per condition ({N} tokens)")
    print(f"          shear {S:g} ms/tone ({args.order}) -> figure spans "
          f"{(args.n_tones - 1) * S + TONE_MS:.0f} ms, successive tones "
          f"{'overlap by' if S < TONE_MS else 'gap by'} "
          f"{abs(TONE_MS - S):.0f} ms")
    print(f"          balance: ceiling {ceil} (>= the {peak}-tone peak), "
          f"at most {lay['c_max']} channels at {RATE_HZ:g} Hz -> grid "
          f"{lay['grid_st']:g} st, {freqs.size} channels "
          f"{freqs[0]:.0f}-{freqs[-1]:.0f} Hz")
    print(f"          every channel {lay['avg_rate']:.1f} tones/s, of which "
          f"the figure supplies {RATE_HZ:g} in its own "
          f"({lay['share'] * 100:.0f}% left to the cloud); "
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
        q = np.zeros((freqs.size, 4), dtype=int)
        for k, o in ([(chan[int(round(st * 2))], o) for st, o, _ in ev]
                     + [(int(np.argmin(np.abs(freqs - f))), o)
                        for f, o in clouds[nm]["events"]]):
            q[k, min(3, int(4 * o / T))] += 1
        print(f"  {nm}  ({ttl})")
        print(f"    {len(ev)} figure tones, {T / core.SR:.1f} s; "
              f"{core.levels(xs[nm] * amp)}")
        print(f"    concurrency {tot.min()}-{tot.max()}, mean "
              f"{tot.mean():.2f} +- {tot.std():.2f}, deepest dip "
              f"{10 * np.log10(ceil / max(tot.min(), 1)):.1f} dB for "
              f"{np.mean(tot < ceil - 1) * 100:.1f}% of the time")
        print(f"    channel use {c.min()}-{c.max()}: figure "
              f"{c[isfig].min()}-{c[isfig].max()} ({isfig.sum()} ch), other "
              f"{c[~isfig].min()}-{c[~isfig].max()} ({(~isfig).sum()} ch); "
              f"per quarter {q.min()}-{q.max()}")
        print(f"    -> {nm}.mp3   {nm}_alone.mp3\n")

    if args.plot:
        print(f"  -> {snapshot(cases, clouds, lay).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
