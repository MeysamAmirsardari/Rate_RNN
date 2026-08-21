"""
audios6.word
============

The same experiment as ``audios.word`` at a larger grain: **six-tone
syllables, five to a word**.

A sibling rather than a subclass, because almost every constant moves --
the frequency layout, the repetition period, the jitter -- and a version that
tried to be both would be readable as neither.  The shared machinery
(``audios.core``, ``audios.cloud``) is imported, not copied; only the design
lives here.

The hierarchy
-------------
A **tone** is 40 ms.  A **syllable** is six tones on a rising frequency ramp,
each delayed a fixed step behind the one below it -- a staircase.  A **word**
is five syllables, the same staircase transposed up in frequency, each
delayed a fixed step behind the last.  The word repeats on a fixed period.

What changed from ``audios.word``, and why
------------------------------------------
    tones per syllable      5 -> 6      the syllable now spans 0-100 ms at
                                        the same 20 ms tone step
    syllables per word      3 -> 5      thirty figure tones per word
    figure channels        15 -> 30     half the pool at the old size, so the
                                        pool widens from 61 to 67 channels
    word period          1600 -> 2000   five syllables at the speech rate is
                                        a 900 ms word; 1600 ms left no gap
    word jitter           240 -> 280    kept near 14% of the period, still a
                                        whole number of 40 ms blocks

The frequency layout is the one constraint that genuinely had to be solved
rather than scaled.  Thirty figure tones must not collide in the channel grid,
which needs ``syl_step * dj != tone_spacing * dk`` for every pair in range;
five semitones between tones against three between syllables needs ``dj = 5``,
impossible with five syllables.  It is also the most compact arrangement that
works -- thirty channels inside thirty-seven semitones, where the obvious
alternatives need forty-three.

Everything about the pattern is **frozen**: the same tones at the same
frequencies with the same lags, every repetition.  That is what makes it a
figure and what makes it learnable.  The only thing that ever changes is the
size of one step.

The experiment
--------------
The same manipulation exists at both levels, and the module sweeps either:

    --sweep tone       the step between successive TONES inside a syllable
    --sweep syllable   the step between successive SYLLABLES inside a word

At a step of zero the five tones are simultaneous: a chord, which fuses by
common onset and is the classic figure.  As the step grows the chord shears
into a diagonal.  The question nobody has answered is where along that
continuum the diagonal stops being heard as one object.

Three regimes are worth naming in advance, because they predict different
break points:

    step < 40 ms     successive tones OVERLAP.  Simultaneous grouping is
                     available; onset asynchrony this small is known to leave
                     fusion largely intact.
    step = 40 ms     tones exactly abut.  Overlap ends here, and with it the
                     usual account of why the figure is one object.
    step > 40 ms     a gap opens.  Nothing is simultaneous any more, so if the
                     figure still pops out it is doing so **sequentially** --
                     as a repeated melodic pattern rather than as a chord.

The interesting outcome is the third one.  Because the pattern is frozen and
repeated, it can in principle be bound by its recurrence rather than by
simultaneity, and there is no a priori reason that should fail at 40 ms.  If
pop-out survives well past the overlap limit, the figure is not a chord that
has been smeared -- it is an object defined by its repeating spectrotemporal
shape, and the coherence account is incomplete.

Controls
--------
The sweep alone cannot distinguish three explanations, so:

``--order perm``   the same lags, assigned to frequencies in a fixed random
                   order instead of a rising ramp.  Still frozen, still
                   repeated, but no trajectory.  Separates "the figure is a
                   glide" from "the figure is a fixed pattern".
``--redraw``       lags redrawn on every repetition.  Same spread, same tone
                   count, same marginal spectrum, nothing repeats.  This is
                   the floor: whatever survives here is not the pattern.

Why the frequencies are inharmonic
----------------------------------
The tones sit on a semitone grid at even semitone spacings, which are not
harmonics of anything.  Using a harmonic series here would confound the
result completely: harmonicity fuses a complex on its own, so a fall-off in
pop-out with step size could be the loss of harmonic fusion rather than the
loss of pattern grouping.  The whole point is to measure the timing.

The cloud
---------
The figure sits in a scheduled cloud that fills its complement, so the total
number of tones sounding is flat and the figure cannot be found from the
envelope.  The ceiling is taken across **all conditions of a sweep**, not per
condition: otherwise the background density would co-vary with the step size
and the manipulation would be confounded by loudness.

Run
---
    python -m audios.word --sweep tone
    python -m audios.word --sweep syllable
    python -m audios.word --sweep tone --order perm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audios import cloud, core  # noqa: E402  the shared machinery

OUT_DIR = Path(__file__).resolve().parent

F_REF = 1000.0
POOL_ST = (-30.0, 36.0)        # 178 Hz to 8 kHz, 67 channels
GRID_ST = 1.0
GUARD_ST = 1.0
TONE_MS = 40.0

#: Six tones and five syllables means thirty figure tones per word, and the
#: spacings have to be chosen so that no two of them land in the same channel.
#: A collision needs ``syl_step * dj == tone_spacing * dk`` for some pair
#: within range; five semitones between tones against three between syllables
#: needs ``dj = 5``, which cannot happen with only five syllables.  It is also
#: the most compact layout that manages it: thirty distinct channels inside
#: thirty-seven semitones, against forty-three for the obvious alternatives.
N_TONES = 6                    # tones per syllable
N_SYL = 5                      # syllables per word
SYL_SPAN_ST = 25.0             # 6 tones, 5 semitones apart
BASE_ST = -18.0                # the lowest tone of the lowest syllable
SYL_FREQ_STEP_ST = 3.0         # transposition between syllables

#: The two sweeps.  Each fixes its own repetition period, held constant across
#: the series: the step is the only thing allowed to move.
SWEEPS = {
    "tone": dict(n_syl=1, period_ms=1000.0, n_words=14, jitter_ms=120.0,
                 syl_step_ms=0.0,
                 steps=(0.0, 10.0, 20.0, 40.0, 80.0, 160.0)),
    # Five syllables spanning a second at the speech rate, so the period is
    # two seconds rather than the 1.6 that three syllables needed.
    "syllable": dict(n_syl=N_SYL, period_ms=2000.0, n_words=10,
                     jitter_ms=280.0, tone_step_ms=20.0,
                     steps=(80.0, 130.0, 200.0, 300.0, 450.0,
                            500.0, 700.0, 850.0, 1000.0)),
}

#: Consecutive words are allowed to interleave once the step grows past the
#: repetition period, and the period is deliberately *not* stretched to
#: prevent it.  Stretching it would drop the figure's tone rate in step with
#: the manipulation -- five tones per second at a step of 40 ms against one
#: per second at 1000 ms -- so the figure would grow sparser exactly as it
#: grew slower, and the two could never be told apart.  Holding the period
#: fixed keeps the figure's density, its repetition rate and the cloud's
#: density all constant, leaving the step as the only variable.  Each instance
#: is still one frozen shape; several are simply in flight at once, which is
#: what the classic figure-ground picture shows.
#: The word-onset jitter, in whole tone lengths, fixed per sweep.  It used to
#: be derived from "words must never overlap", which no longer holds now that
#: the step is allowed to run past the repetition period; a fixed value also
#: keeps it from co-varying with the manipulation, which is what a control has
#: to avoid.  Twelve and fifteen per cent of the period, three and six blocks:
#: seven and thirteen distinct arrival times, enough to destroy isochrony.

SCHED_STEP_MS = 2.5
LEAD_MS, TAIL_MS = 400.0, 600.0


def template(n_tones: int, n_syl: int, tone_step_ms: float,
             syl_step_ms: float, order: str = "rise",
             seed: int = 0) -> list:
    """One word as (semitone, lag in ms) pairs, relative to its own onset.

    The lag pattern is built once here and reused for every syllable and every
    repetition; ``order`` decides how lags map onto frequencies.
    """
    lag = np.arange(n_tones) * tone_step_ms
    if order == "fall":
        lag = lag[::-1]
    elif order == "perm":
        lag = lag[np.random.default_rng(seed).permutation(n_tones)]
    elif order != "rise":
        raise ValueError(f"unknown order {order!r}")

    d_st = SYL_SPAN_ST / max(n_tones - 1, 1)
    return [(BASE_ST + j * SYL_FREQ_STEP_ST + k * d_st,
             j * syl_step_ms + lag[k])
            for j in range(n_syl) for k in range(n_tones)]


def build(tpl: list, period_ms: float, n_words: int, *,
          onset_jitter_ms: float = 0.0, redraw: bool = False,
          seed: int = 1) -> dict:
    """Place the word ``n_words`` times.

    ``redraw`` replaces the frozen lags with a fresh uniform draw over the
    same range on every repetition.  Tone count, frequencies and spread are
    untouched; only the repetition is destroyed.
    """
    rng = np.random.default_rng(seed)
    st = np.array([s for s, _ in tpl])
    lag = np.array([t for _, t in tpl])
    span = float(lag.max())

    # The word onset is quantised to the cloud's tone length, and this is the
    # whole reason the background can be thin.  The cloud tiles in 40 ms
    # blocks; a word landing mid-block leaves the part of that block before it
    # able to hold only one tone, because a 40 ms cloud tone starting there
    # would run into the figure and break the ceiling.  The hole is
    # ``onset mod 40 ms`` wide -- 20 ms on average, up to 40 -- and it sits
    # immediately before the figure, which is the worst place for it.  Landing
    # the word on a block boundary removes it exactly.  Seven distinct jitter
    # values at +-120 ms is still ample to destroy isochrony, and the mean
    # rate stays exact because the draw is symmetric.
    q = core.samples(TONE_MS)
    grid = (np.arange(n_words) * core.samples(period_ms)
            + core.samples(LEAD_MS))
    k = int(core.samples(onset_jitter_ms) // q)
    onsets = grid + (rng.integers(-k, k + 1, size=n_words) * q if k else 0)
    total = int(onsets[-1]) + core.samples(span + TONE_MS + TAIL_MS)
    x = np.zeros(total)

    freqs = F_REF * 2.0 ** (st / 12.0)
    pips = {float(f): core.tone(f, TONE_MS) for f in np.unique(freqs)}

    # Lags are held in SAMPLES from here on: the frozen ones come from an
    # exact millisecond grid, but a redrawn one is continuous and would not
    # land on a whole sample.
    lag_n = np.array([core.samples(float(v)) for v in lag])
    span_n = core.samples(span)

    ev = []                    # (semitone, onset sample, word index)
    for w, o0 in enumerate(onsets):
        d = (rng.integers(0, span_n + 1, size=lag_n.size)
             if redraw and span_n > 0 else lag_n)
        for i in range(st.size):
            o = int(o0) + int(d[i])
            p = pips[float(freqs[i])]
            x[o:o + p.size] += p
            ev.append((st[i], o, w))
    return dict(x=x, events=ev, onsets=onsets, st=st, span=span)


def report(b: dict, name: str, step_ms: float, level: str,
           unit_ms: float) -> str:
    """``unit_ms`` is the length of the thing being stepped: one tone when
    sweeping tones, one whole syllable when sweeping syllables.  The gap that
    matters is the step minus that, not the step minus a tone."""
    gap = step_ms - unit_ms
    rel = "overlap" if gap < 0 else ("abut" if gap == 0 else "gap")
    n = len(b["events"])
    # the pattern really is frozen: every word must place its tones at the
    # same offsets from its own onset
    per = {}
    for s, o, w in b["events"]:
        per.setdefault(w, []).append(round(o - b["onsets"][w]))
    shapes = {tuple(sorted(v)) for v in per.values()}
    return (f"  {name}\n"
            f"    {level} step {step_ms:.0f} ms -> successive {level}s "
            f"{rel} by {abs(gap):.0f} ms; word spans {b['span']:.0f} ms, "
            f"{n} tones\n"
            f"    distinct word shapes across {len(per)} repetitions: "
            f"{len(shapes)}; up to {b['in_flight']} in flight at once\n"
            f"    {core.levels(b['x'])}")


def figure(shown: list, stem: str, span_ms: float,
           ncols: int = 5) -> Path:
    """The figure-ground picture: cloud in black, figure in red.

    One time window for every panel, wide enough for the longest condition,
    so the slopes can actually be compared; wrapped onto rows of ``ncols``
    once the sweep gets long.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(shown)
    ncols = min(ncols, k)
    nrows = int(np.ceil(k / ncols))
    dur = core.samples(TONE_MS)
    span = core.samples(span_ms)

    fig, axes = plt.subplots(nrows, ncols, sharey=True, sharex=True,
                             figsize=(3.0 * ncols + 0.8, 4.6 * nrows),
                             constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)

    for ax, (b, cl, title) in zip(axes.ravel(), shown):
        t0 = int(b["onsets"][0]) - core.samples(120.0)
        t1 = t0 + span
        for f, o in cl["events"]:
            if t0 <= o < t1:
                y = 12 * np.log2(f / F_REF)
                ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR], [y, y],
                        color="#111111", lw=2.0, solid_capstyle="butt")
        for st, o, _w in b["events"]:
            if t0 <= o < t1:
                ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR],
                        [st, st], color="#E8121A", lw=2.6,
                        solid_capstyle="butt")
        for o in b["onsets"]:
            if t0 <= o < t1:
                ax.axvline((o - t0) / core.SR, color="#888", lw=0.6,
                           ls=(0, (3, 3)))
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, span / core.SR)
        ax.set_ylim(POOL_ST[0] - 2, POOL_ST[1] + 2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for row in axes:
        row[0].set_ylabel(f"Semitones re {F_REF:.0f} Hz")

    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def channel_check(counts: np.ndarray, isfig: np.ndarray, quarters: np.ndarray
                  ) -> str:
    """Is every channel used equally -- figure channels included?

    Two questions, not one.  **Overall**: do the figure's channels carry the
    same number of tones as the rest, once their figure tones are counted?  If
    they carry fewer they are conspicuously quiet and if they carry more they
    are conspicuously busy, and either way the figure can be found without
    listening to the timing.  **Over time**: is each channel used at the same
    rate throughout, or does the dealer drift?  A channel that is busy early
    and idle late is uniform in total and non-uniform where it matters.

    ``quarters`` is (n_channels, 4): tones per channel per quarter of the
    stream, figure and cloud together.
    """
    tot = counts
    f, o = tot[isfig], tot[~isfig]
    per_q = quarters.sum(axis=0)
    return (f"  channel use, figure and cloud together, {tot.size} channels\n"
            f"    all      {tot.min()}-{tot.max()}, mean {tot.mean():.1f} "
            f"+- {tot.std():.1f}\n"
            f"    figure   {f.min()}-{f.max()}, mean {f.mean():.1f}   "
            f"({isfig.sum()} channels)\n"
            f"    other    {o.min()}-{o.max()}, mean {o.mean():.1f}   "
            f"({(~isfig).sum()} channels)\n"
            f"    per quarter of the stream, per channel: "
            f"{quarters.min()}-{quarters.max()} "
            f"(quarter totals {per_q.min()}-{per_q.max()})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", choices=sorted(SWEEPS), default="tone")
    p.add_argument("--steps", type=float, nargs="+", default=None,
                   help="ms; overrides the sweep's own list")
    p.add_argument("--n-tones", type=int, default=N_TONES)
    p.add_argument("--n-syl", type=int, default=0,
                   help="override the syllables per word")
    p.add_argument("--order", choices=("rise", "fall", "perm"),
                   default="rise")
    p.add_argument("--redraw", action="store_true",
                   help="the floor control: redraw the lags every repetition")
    p.add_argument("--word-jitter-ms", type=float, default=-1.0,
                   help="displacement of each whole word; -1 takes the "
                        "sweep's own value, 0 turns it off (isochronous, and "
                        "then findable by rhythm alone)")
    p.add_argument("--exclusive", action="store_true",
                   help="keep the cloud off the figure's channels (leaks: "
                        "frequency alone then identifies the figure)")
    p.add_argument("--ceiling", type=int, default=6,
                   help="tones sounding at once, figure plus cloud; raised to "
                        "the figure's own peak if that is larger")
    p.add_argument("--cloud-db", type=float, default=0.0)
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)

    sw = dict(SWEEPS[args.sweep])
    steps = args.steps if args.steps is not None else list(sw.pop("steps"))
    sw.pop("steps", None)
    n_syl = args.n_syl or sw["n_syl"]
    period, n_words = sw["period_ms"], sw["n_words"]
    tag = {"tone": "tone", "syllable": "syl"}[args.sweep]
    suffix = ("" if args.order == "rise" else f"_{args.order}") + \
             ("_redraw" if args.redraw else "")

    built, templates = [], []
    for s in steps:
        kw = (dict(tone_step_ms=s, syl_step_ms=sw.get("syl_step_ms", 0.0))
              if args.sweep == "tone" else
              dict(tone_step_ms=sw["tone_step_ms"], syl_step_ms=s))
        tpl = template(args.n_tones, n_syl, order=args.order, **kw)
        templates.append((s, tpl))

    jit = (sw["jitter_ms"] if args.word_jitter_ms < 0
           else args.word_jitter_ms)
    for s, tpl in templates:
        b = build(tpl, period, n_words, redraw=args.redraw,
                  onset_jitter_ms=jit)
        built.append((f"word_{tag}{s:g}ms{suffix}", b, s))

    # measured, not assumed: how many instances of the figure are in flight
    for nm, b, _ in built:
        gap = np.diff(b["onsets"]) / core.SR * 1000.0
        b["in_flight"] = int(np.ceil((b["span"] + TONE_MS) / gap.min()))

    # One ceiling for the whole sweep, so density never co-varies with step.
    n_tone = core.samples(TONE_MS)
    # One ceiling for the sweep AND across sweeps: the tone sweep's figure
    # peaks at five simultaneous tones and the syllable sweep's at three, so
    # letting each take its own peak would make the two levels incomparable.
    peak = max(int(cloud.concurrency([o for _, o, _ in b["events"]],
                                     b["x"].size, n_tone).max())
               for _, b, _ in built)
    peak = max(peak, args.ceiling)
    figure_st = sorted({float(s) for _, b, _ in built for s, _, _ in
                        b["events"]})
    # The pool is the whole grid.  Reserving the figure's own channels would
    # make every tone in them a figure tone, so the figure could be found by
    # frequency alone and the timing -- the thing being measured -- would
    # never have to be used.  --exclusive restores that, for comparison only.
    freqs = cloud.channels(figure_st if args.exclusive else [],
                           f_ref=F_REF, st_lo=POOL_ST[0], st_hi=POOL_ST[1],
                           guard_st=GUARD_ST, grid_st=GRID_ST)
    grid_st = 12.0 * np.log2(freqs / F_REF)
    chan = {int(round(v)): k for k, v in enumerate(grid_st)}
    if not args.exclusive and any(int(round(v)) not in chan
                                  for v in figure_st):
        raise SystemExit("a figure tone is off the cloud grid")
    clouds = {nm: cloud.schedule(
        b["x"].size, freqs,
        ([] if args.exclusive else
         [(chan[int(round(st))], o) for st, o, _ in b["events"]]),
        peak, tone_ms=TONE_MS, step_ms=SCHED_STEP_MS)
        for nm, b, _ in built}

    level = "tone" if args.sweep == "tone" else "syllable"
    fixed = (f"syllable step {sw['tone_step_ms']:.0f} ms"
             if args.sweep == "syllable" else "one syllable")
    print(f"{core.SR} Hz | {n_syl} syllable(s) x {args.n_tones} tones = "
          f"{n_syl * args.n_tones} figure channels, {len(figure_st)} distinct; "
          f"{freqs.size} cloud channels {freqs[0]:.0f}-{freqs[-1]:.0f} Hz")
    iti = np.concatenate([np.diff(b["onsets"]) for _, b, _ in built])
    iti = iti / core.SR * 1000.0
    print(f"          sweeping the {level} step, {fixed}; word every "
          f"{period:.0f} ms +-{jit:.0f} ms jitter "
          f"(measured {iti.min():.0f}-{iti.max():.0f} ms, mean "
          f"{iti.mean():.0f}), lags "
          f"{'REDRAWN each repetition' if args.redraw else 'frozen'}, "
          f"order {args.order}; cloud ceiling {peak}\n")

    gain = 10.0 ** (args.cloud_db / 20.0)
    mixes = [(nm, b["x"] + gain * clouds[nm]["x"][:b["x"].size])
             for nm, b, _ in built]
    amp = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(b["x"]))) for _, b, _ in built)
    amp_c = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(m))) for _, m in mixes)

    for (nm, b, s), (_, m) in zip(built, mixes):
        b["x"] = b["x"] * amp
        core.render(OUT_DIR / f"{nm}_alone", b["x"], args.keep_wav)
        core.render(OUT_DIR / nm, m * amp_c, args.keep_wav)
        unit = (TONE_MS if args.sweep == "tone" else
                (args.n_tones - 1) * sw["tone_step_ms"] + TONE_MS)
        print(report(b, nm, s, level, unit))
        T = b["x"].size
        lo, hi = core.samples(LEAD_MS), T - core.samples(TAIL_MS)
        tot = (cloud.concurrency([o for _, o, _ in b["events"]], T, n_tone)
               + cloud.concurrency([o for _, o in clouds[nm]["events"]],
                                   T, n_tone))[lo:hi]
        cnt, cc = clouds[nm]["counts"], clouds[nm]["cloud_counts"]
        isfig = clouds[nm]["fig_counts"] > 0
        share = (f"figure channels {cnt[isfig].min()}-{cnt[isfig].max()}, "
                 f"others {cnt[~isfig].min()}-{cnt[~isfig].max()}"
                 if isfig.any() else
                 f"cloud {cnt.min()}-{cnt.max()} vs "
                 f"{n_words * n_syl * args.n_tones // len(figure_st)} figure")
        # The notch is what matters, and it matters in dB: power goes with
        # the number of tones sounding, so the same shortfall is loud against
        # a thin cloud and inaudible against a thick one.
        dip = 10 * np.log10(peak / max(tot.min(), 1))
        print(f"    in cloud: concurrency {tot.min()}-{tot.max()}, mean "
              f"{tot.mean():.2f} +- {tot.std():.2f}, deepest dip "
              f"{dip:.1f} dB for {np.mean(tot < peak - 1) * 100:.1f}% of "
              f"the time; tones per channel {share}")
        print(f"    -> {nm}.mp3   {nm}_alone.mp3\n")

    # uniformity across every channel, checked on the built streams
    nm0, b0 = built[0][0], built[0][1]
    cl0 = clouds[nm0]
    T0 = b0["x"].size
    ev = [(chan[int(round(st))], o) for st, o, _ in b0["events"]] + \
         [(int(np.argmin(np.abs(freqs - f))), o) for f, o in cl0["events"]]
    quarters = np.zeros((freqs.size, 4), dtype=int)
    for k, o in ev:
        quarters[k, min(3, int(4 * o / T0))] += 1
    print(channel_check(cl0["counts"], cl0["fig_counts"] > 0, quarters))

    shown = [(b, clouds[nm], f"{level} step {s:g} ms")
             for nm, b, s in built]
    # one window for every panel: the longest span plus a repetition, so the
    # shallow and the steep staircases are drawn to the same time scale
    win = max(2.2 * period, max(b["span"] for _, b, _ in built) + period)
    print(f"  -> {figure(shown, f'word_{tag}{suffix}_check', win).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
