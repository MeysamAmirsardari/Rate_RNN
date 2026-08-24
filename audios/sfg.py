"""
audios.sfg
==========

The classic figure, and the same figure sheared into a staircase.

Everything is written to ``audios/sfg_out/``:

    sfg_coherent.mp3   n tones, all together, every 200 ms
    sfg_stair10.mp3    the same n, delayed 0 10 20 ... ms
    sfg_check.png      with --plot

plus an ``_alone`` copy of each condition without the cloud.

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

What makes the figure audible
-----------------------------
In this paradigm the figure **is** the set of channels that recur more often
than the background.  An earlier version of this module levelled every
channel's total count -- the same number of tones in a figure channel as in a
background one -- and that silently removed the figure: measured, 79 tones per
figure channel against 79 per background channel, with the cloud arriving as
synchronous ten-tone chords that the figure was simply another instance of.

Two things restore it, and both are measured rather than assumed.

**The cloud levels its own counts, not the totals.**  It treats every channel
alike, never avoiding or favouring the figure's -- so frequency alone still
gives nothing away -- and the figure's tones sit on top.  ``--contrast`` sets
how many times more often a figure channel then sounds than a background one,
and asking for a ratio fixes the pool size:

    cloud_rate = (ceiling - n * tone_s * rate) / (tone_s * C)
    contrast   = (rate + cloud_rate) / cloud_rate

A big ratio needs a big pool.  Four needs about 120 channels, which a
half-semitone grid over five octaves just provides; classic figure-ground runs
nearer thirteen, which would need 480 channels an eighth of a semitone apart,
finer than the ear resolves.  Four is where this span tops out.

**The background has to be irregular.**  Dealing strictly least-used-first
spaces a channel's returns far more evenly than chance -- CV 0.41 against 1.00
for a random process -- so the background acquires exactly the regularity the
figure is supposed to own alone.  ``DEALER_SLACK`` buckets the counts before
ranking and hands the choice back to the rng: CV 0.97, counts still level,
concurrency still exactly flat.

Nothing in the cloud repeats.  Over a 13 s stream: 262 simultaneous groups, all
262 distinct, no group ever reused; 5897 of 7260 possible channel pairs occur,
the commonest 8 times.

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

#: Everything this module writes goes here, not into ``audios/`` itself,
#: which by now holds several unrelated stimulus sets.  Named ``sfg_out``
#: rather than ``sfg`` because a directory ``audios/sfg/`` sitting beside the
#: module ``audios/sfg.py`` is a package/module name clash: Python resolves it
#: in the module's favour today, but it is the kind of ambiguity that breaks
#: the moment something adds an ``__init__.py`` or globs directories.
OUT_DIR = Path(__file__).resolve().parent / "sfg_out"

F_REF = 1000.0
POOL_ST = (-24.0, 36.0)        # 250 Hz to 8 kHz
TONE_MS = 20.0                 # set from --tone-ms at run time
RATE_HZ = 3.0

N_TONES = 7
STEP_MS = 20.0
DURATION_S = 20.0
JITTER_MS = 40.0
CEILING_MIN = 6
CLOUD_TONES = 3
CONTRAST = 4.0
DEALER_SLACK = 5               # how loosely the cloud levels its channels
FIG_MARGIN = 0.15









#: ``CONTRAST`` is what makes the figure audible, and it replaced an earlier
#: rule that levelled every channel's *total* count.  That rule was a mistake:
#: in this paradigm the figure IS the set of channels that recur more often
#: than the background, so equalising the totals sets the contrast to 1 and
#: leaves nothing to hear.  Measured on the levelled version: 79 tones per
#: figure channel against 79 per background channel, and no figure.
#:
#: A figure channel sounds ``RATE_HZ`` times a second from the figure plus its
#: share of the cloud; a background channel gets the cloud's share alone.
#: Asking for a ratio therefore fixes the pool size, and a big ratio needs a
#: big pool: 4 needs about 120 channels, which a half-semitone grid over five
#: octaves just provides.  Classic figure-ground runs nearer 13, which would
#: need 480 channels an eighth of a semitone apart -- finer than the ear
#: resolves, so 4 is where this span tops out.

#: Candidate grid spacings, finest first.
GRIDS = tuple(np.arange(0.5, 12.5, 0.5))

SCHED_STEP_MS = 2.5
LEAD_MS, TAIL_MS = 400.0, 600.0


def _set_tone(ms: float) -> None:
    """Tone duration is a module constant that several helpers read directly,
    so ``--tone-ms`` rebinds it once here rather than being threaded through
    every signature."""
    global TONE_MS
    TONE_MS = ms


def layout(n_tones: int, rate_hz: float = RATE_HZ, *,
           ceiling: int = 0, contrast: float = CONTRAST,
           share: bool = False, cloud_tones: int = 0) -> dict:
    """Solve for the ceiling, the grid and the figure's channels.

    Returns everything downstream needs, plus the numbers behind it so the
    choice can be printed rather than taken on trust.
    """
    tone_s = TONE_MS / 1000.0
    ceil = ceiling or max(n_tones, CEILING_MIN)

    # The pool is as large as the span and the grid allow, and the grid is set
    # by ``--contrast``: how many times more often a figure channel sounds
    # than a background one.  A figure channel sounds ``rate`` times a second
    # from the figure plus its share of the cloud; a background channel gets
    # the cloud's share alone.  The cloud has ``ceil - fig_load`` voices to
    # spread over C channels, so
    #
    #     cloud_rate = (ceil - n*tone_s*rate) / (tone_s * C)
    #     contrast   = (rate + cloud_rate) / cloud_rate
    #
    # and asking for a contrast fixes C.  This is the opposite of levelling
    # the totals, which is what removed the figure: at equal totals the
    # contrast is 1 and there is nothing to hear.
    fig_load = n_tones * tone_s * rate_hz          # mean concurrency of figure
    if cloud_tones:
        # Sparse mode.  The cloud holds its own steady concurrency and the
        # figure sounds on top, so the pool follows from the cloud's density
        # alone: bg_rate = cloud_tones / (tone_s * (C - n)).
        c_want = cloud_tones * contrast / (tone_s * rate_hz) + n_tones
        span = POOL_ST[1] - POOL_ST[0]
        grid = min(GRIDS, key=lambda g: abs(int(np.floor(span / g)) + 1 - c_want))
        freqs = cloud.channels([], f_ref=F_REF, st_lo=POOL_ST[0],
                               st_hi=POOL_ST[1], grid_st=grid)
        st_grid = 12.0 * np.log2(freqs / F_REF)
        lo = int(round(FIG_MARGIN * (freqs.size - 1)))
        hi = int(round((1.0 - FIG_MARGIN) * (freqs.size - 1)))
        idx = np.unique(np.round(np.linspace(lo, hi, n_tones)).astype(int))
        cr = cloud_tones / (tone_s * max(freqs.size - n_tones, 1))
        return dict(ceiling=cloud_tones, grid_st=grid, freqs=freqs,
                    st_grid=st_grid, fig_idx=idx, fig_st=st_grid[idx],
                    c_max=int(c_want), cloud_rate=cr, fig_rate=rate_hz,
                    rate_hz=rate_hz, contrast=rate_hz / cr, sparse=True)
    # With the figure's channels off limits the cloud spreads over ``C - n``
    # channels and a figure channel carries nothing but the figure, so the
    # ratio is ``rate / cloud_rate``; when they are shared the cloud's own
    # tones land in figure channels too and it is ``(rate + cloud_rate) /
    # cloud_rate``.  Solving the wrong one leaves the printed contrast a
    # prediction about a stimulus that was not built.
    cloud_rate = (rate_hz / max(contrast - 1.0, 1e-9) if share
                  else rate_hz / max(contrast, 1e-9))
    c_want = (ceil - fig_load) / (tone_s * cloud_rate) + (0 if share else n_tones)
    span = POOL_ST[1] - POOL_ST[0]
    grid = min(GRIDS, key=lambda g: abs(int(np.floor(span / g)) + 1 - c_want))
    c_max = int(c_want)
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
    n_bg = freqs.size if share else freqs.size - n_tones
    cr = (ceil - fig_load) / (tone_s * max(n_bg, 1))
    fr = rate_hz + cr if share else rate_hz
    return dict(ceiling=ceil, grid_st=grid, freqs=freqs, st_grid=st_grid,
                fig_idx=idx, fig_st=st_grid[idx], c_max=c_max,
                cloud_rate=cr, fig_rate=fr, rate_hz=rate_hz,
                contrast=fr / cr, sparse=False)


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
    p.add_argument("--tone-ms", type=float, default=TONE_MS,
                   help="tone duration (default 25)")
    p.add_argument("--cloud-tones", type=int, default=CLOUD_TONES,
                   help="the cloud's own concurrency; the figure sounds on "
                        "top of it, so the total rises during a figure burst")
    p.add_argument("--flat", action="store_true",
                   help="hold the TOTAL flat instead, which forces the cloud "
                        "up to the figure's peak -- 10 tones for a coherent "
                        "10-tone figure, and no longer sparse")
    p.add_argument("--share-channels", action="store_true",
                   help="let the cloud sound the figure's own channels; it "
                        "then inserts a tone near the midpoint of the "
                        "figure's period and doubles its apparent rate")
    p.add_argument("--contrast", type=float, default=CONTRAST,
                   help="how many times more often a figure channel sounds "
                        "than a background one; 1 means no figure")
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _set_tone(args.tone_ms)
    lay = layout(args.n_tones, ceiling=args.ceiling,
                 contrast=args.contrast, share=args.share_channels,
                 cloud_tones=0 if args.flat else args.cloud_tones)
    period = 1000.0 / RATE_HZ
    N = max(1, int(round(args.duration * RATE_HZ)))
    S, freqs = args.step_ms, lay["freqs"]
    # A fresh generator per condition, not one shared between them: sharing
    # it advanced the stream, so the coherent and sheared files were built on
    # different token onsets and differed by more than the shear.
    def kw():
        return dict(order=args.order, jitter_ms=args.jitter_ms,
                    rng=np.random.default_rng(3))

    cases = [
        ("sfg_coherent", tokens(lay, 0.0, N, LEAD_MS, **kw()),
         "coherent, 0 ms"),
        (f"sfg_stair{S:g}", tokens(lay, S, N, LEAD_MS, **kw()),
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
    ceil = lay["ceiling"] if lay["sparse"] else max(peak, lay["ceiling"])
    # The cloud is kept out of the figure's own channels unless asked
    # otherwise, and the reason is measured rather than aesthetic.  The
    # ceiling equals the figure's peak, so no cloud tone may sound during a
    # figure burst; a cloud tone in a figure channel is therefore forced into
    # the gap between two figure tones, and the guard pins it near the middle
    # -- measured phase 0.48 +- 0.12 of the interval.  Every such tone is
    # heard as a beat exactly halfway, so the figure appears to run at twice
    # its rate.  Sharing the channels was meant to stop frequency alone
    # identifying the figure; it doubles the figure's tempo instead.
    banned = (set() if args.share_channels
              else {chan[int(round(st * 2))] for st in lay["fig_st"]})
    # In sparse mode the cloud is scheduled WITHOUT the figure, so it holds
    # its own steady concurrency and the figure sounds on top of it.  In flat
    # mode it fills the figure's complement instead, which pins the total but
    # forces the cloud up to the figure's own peak.
    clouds = {nm: cloud.schedule(
        xs[nm].size, freqs,
        [] if lay["sparse"] else
        [(chan[int(round(st * 2))], o) for st, o, _ in ev],
        ceil, tone_ms=TONE_MS, step_ms=SCHED_STEP_MS, count_figure=False,
        slack=DEALER_SLACK, banned=banned)
        for nm, ev, _ in cases}

    print(f"{core.SR} Hz | {args.n_tones}-tone figure at {RATE_HZ:g} Hz "
          f"({period:.0f} ms), tones {TONE_MS:.0f} ms, "
          f"{args.duration:g} s per condition ({N} tokens)")
    print(f"          shear {S:g} ms/tone ({args.order}) -> figure spans "
          f"{(args.n_tones - 1) * S + TONE_MS:.0f} ms, successive tones "
          f"{'overlap by' if S < TONE_MS else 'gap by'} "
          f"{abs(TONE_MS - S):.0f} ms")
    print(f"          cloud {ceil} tones at once, figure peaks at {peak} "
          f"on top of it" if lay["sparse"] else
          f"          ceiling {ceil} (>= the {peak}-tone peak)")
    print(f"          grid "
          f"{lay['grid_st']:g} st, {freqs.size} channels "
          f"{freqs[0]:.0f}-{freqs[-1]:.0f} Hz")
    nm0, ev0 = cases[0][0], cases[0][1]
    dur = N * period / 1000.0        # the active window, not lead + tail
    isf0 = np.zeros(freqs.size, dtype=bool)
    isf0[list({chan[int(round(st * 2))] for st in lay["fig_st"]})] = True
    tot0 = clouds[nm0]["counts"].copy()
    if lay["sparse"]:                  # the cloud never saw the figure
        for st, o, _t in ev0:
            tot0[chan[int(round(st * 2))]] += 1
    f_rate = tot0[isf0].mean() / dur
    b_rate = tot0[~isf0].mean() / dur
    print(f"          MEASURED: a figure channel sounds {f_rate:.1f} times/s "
          f"against {b_rate:.1f} for a background one: "
          f"contrast {f_rate / max(b_rate, 1e-9):.1f}x"
          f"{'' if args.share_channels else ', cloud kept off figure channels'}"
          f"; token jitter +-{args.jitter_ms:g} ms\n")

    mixes = {nm: xs[nm] + clouds[nm]["x"][:xs[nm].size] for nm, _, _ in cases}
    # ONE gain for every file, figure-only and mixed alike.  Scaling the
    # figure-only files to their own peak made them louder than the figure
    # inside the mix, so ``_alone`` was not the figure you hear in the cloud
    # -- it was a different presentation of it.  With a common gain the
    # ``_alone`` file is exactly the figure component of its mix, sample for
    # sample, and the two can be A/B'd.
    amp = amp_c = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(m))) for m in mixes.values())

    for nm, ev, ttl in cases:
        core.render(OUT_DIR / f"{nm}_alone", xs[nm] * amp, args.keep_wav)
        core.render(OUT_DIR / nm, mixes[nm] * amp_c, args.keep_wav)
        T = xs[nm].size
        lo, hi = core.samples(LEAD_MS), T - core.samples(TAIL_MS)
        tot = (cloud.concurrency([o for _, o, _ in ev], T, n_tone)
               + cloud.concurrency([o for _, o in clouds[nm]["events"]],
                                   T, n_tone))[lo:hi]
        c = clouds[nm]["counts"].copy()
        isfig = np.zeros(freqs.size, dtype=bool)
        for st, o, _t in ev:                    # count the figure's own tones
            k = chan[int(round(st * 2))]
            isfig[k] = True
            if lay["sparse"]:
                c[k] += 1
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
        print(f"    -> {OUT_DIR.name}/{nm}.mp3   "
              f"{OUT_DIR.name}/{nm}_alone.mp3\n")

    if args.plot:
        out = snapshot(cases, clouds, lay)
        print(f"  -> {out.parent.name}/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
