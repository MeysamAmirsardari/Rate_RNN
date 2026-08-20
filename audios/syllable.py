"""
audios.syllable
===============

One coherent complex against ten incoherent near-copies of its own partials.

    syl_coherent.mp3     A's ten partials always simultaneous
    syl_scrambled.mp3    the control: A's partials jittered too
    syl_check.png        the picture

The stimulus
------------
Twenty channels, in two sets of ten.

**A -- the figure.**  Ten partials, harmonics 1 to 10 of 400 Hz, so 400 Hz to
4 kHz.  Every one of them starts at exactly the same instant, every time the
token repeats: **zero onset jitter, fully coherent**.  Simultaneous onset plus
harmonicity is the strongest fusion cue there is, so the ten arrive as one
object rather than ten tones -- which is what makes it syllable-like.

**S -- the ten shifted copies.**  One per partial of A, displaced in both
dimensions:

    frequency   0.5 to 1 semitone away, sign and size drawn once per channel
                and then FIXED, because these are channels and a channel does
                not move
    time        20 to 40 ms away, sign and size drawn **afresh on every
                repetition**, because that is what stops them ever becoming
                an object of their own

That last distinction is the whole design.  If the time shifts were fixed too,
the ten copies would themselves be perfectly coherent -- just a second
syllable offset from the first -- and there would be two objects rather than
one object and a mess.  Redrawing them per repetition leaves A as the only
thing in the stimulus with a stable temporal signature.

The shifts are 20-40 ms against a 40 ms tone, so a copy always overlaps its
partner in time; it is never a separate event that could be grouped by
proximity, it is a smeared version of the same event.

The control
-----------
``syl_scrambled`` jitters A's own partials the same way.  Nothing then is
coherent, and nothing should fuse.  It is the comparison that says whether the
coherence in ``syl_coherent`` is doing the work, rather than the harmonic
series or the spectrum, both of which are identical between the two.

Level
-----
Every channel at equal amplitude, the figure-ground convention, rather than a
1/h roll-off.  With a roll-off the tenth partial would be 20 dB below the
first and the top half of the figure would contribute almost nothing to
whether it fuses; here each of the twenty channels carries the same weight.

Not yet included
----------------
No cloud.  These twenty channels are meant to sit inside a larger set later --
the sketch has them as ten of seventy -- and the cloud construction from
``audios.cloud`` will drop straight in when it does.

Run
---
    python -m audios.syllable
    python -m audios.syllable --n-partials 10 --shift-ms 20 40 --df-st 0.5 1.0
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

F0 = 400.0
N_PARTIALS = 10
TONE_MS = 40.0
PERIOD_MS = 400.0          # token repetition, 2.5 Hz
N_TOKENS = 30              # 12 s
DF_ST = (0.5, 1.0)         # frequency offset of a copy, semitones
SHIFT_MS = (20.0, 40.0)    # time offset of a copy, magnitude
LEAD_MS, TAIL_MS = 400.0, 600.0


def layout(n: int, df_st, seed: int = 0) -> dict:
    """Channel frequencies: n coherent partials and n fixed shifted copies."""
    rng = np.random.default_rng(seed)
    harm = np.arange(1, n + 1, dtype=float)
    f_a = F0 * harm
    off = rng.uniform(df_st[0], df_st[1], size=n) * rng.choice([-1.0, 1.0], n)
    return dict(f_a=f_a, f_s=f_a * 2.0 ** (off / 12.0), offset_st=off,
                harm=harm)


def build(lay: dict, shift_ms, scramble_a: bool = False,
          seed: int = 1) -> dict:
    """Place every token; A coherent unless ``scramble_a``."""
    rng = np.random.default_rng(seed)
    n = lay["f_a"].size
    pips_a = [core.tone(f, TONE_MS) for f in lay["f_a"]]
    pips_s = [core.tone(f, TONE_MS) for f in lay["f_s"]]

    base = core.samples(LEAD_MS) + np.arange(N_TOKENS) * core.samples(PERIOD_MS)
    total = int(base[-1]) + core.samples(max(shift_ms) + TONE_MS + TAIL_MS)
    x = np.zeros(total)

    def draw(k):
        mag = rng.uniform(shift_ms[0], shift_ms[1], size=k)
        return np.round(mag * rng.choice([-1.0, 1.0], k)
                        * core.SR / 1000.0).astype(int)

    ev = []      # (channel, set, onset sample, token index)
    for t, b in enumerate(base):
        d_a = draw(n) if scramble_a else np.zeros(n, dtype=int)
        d_s = draw(n)
        for i in range(n):
            for d, pip, tag in ((d_a[i], pips_a[i], "A"),
                                (d_s[i], pips_s[i], "S")):
                o = int(b) + int(d)
                x[o:o + pip.size] += pip
                ev.append((i, tag, o, t))
    return dict(x=x, events=ev, base=base, lay=lay)


def report(b: dict, name: str) -> str:
    ev = b["events"]
    lay = b["lay"]
    ms = 1000.0 / core.SR

    # Grouped by TOKEN INDEX, not by absolute time.  Binning on time puts a
    # tone shifted across a token boundary into its neighbour's group and
    # reports the period as if it were jitter.
    per_tok: dict = {}
    for i, tag, o, tok in ev:
        per_tok.setdefault((tag, tok), []).append(o)

    def spread(tag):
        v = [np.ptp(np.array(o)) * ms for (tg, _), o in per_tok.items()
             if tg == tag and len(o) > 1]
        return float(np.mean(v)) if v else 0.0

    da = np.abs(lay["offset_st"])
    return (f"  {name}\n"
            f"    A: {lay['f_a'].size} partials, "
            f"{lay['f_a'][0]:.0f}-{lay['f_a'][-1]:.0f} Hz (harmonics of "
            f"{F0:.0f} Hz); onset spread within a token "
            f"{spread('A'):.1f} ms\n"
            f"    S: offsets {da.min():.2f}-{da.max():.2f} st "
            f"({np.mean(lay['f_s'] - lay['f_a']):+.0f} Hz mean), "
            f"onset spread within a token {spread('S'):.1f} ms\n"
            f"    {N_TOKENS} tokens every {PERIOD_MS:.0f} ms; "
            f"{core.levels(b['x'])}")


def figure(coh: dict, scr: dict, stem: str = "syl_check") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lay = coh["lay"]
    n = lay["f_a"].size
    C_A, C_S = "#7C102A", "#2166AC"
    t0, t1 = core.samples(LEAD_MS) - core.samples(120.0), \
        core.samples(LEAD_MS) + core.samples(4 * PERIOD_MS)
    dur = core.samples(TONE_MS)

    fig = plt.figure(figsize=(12.6, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.5, 2.5, 1.0])

    for col, (b, title) in enumerate(
            ((coh, "coherent: A's partials always simultaneous"),
             (scr, "scrambled control: A jittered too"))):
        ax = fig.add_subplot(gs[:, col])
        for i, tag, o, _tok in b["events"]:
            if not (t0 <= o < t1):
                continue
            f = lay["f_a"][i] if tag == "A" else lay["f_s"][i]
            y = 12 * np.log2(f / F0)
            ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR], [y, y],
                    color=C_A if tag == "A" else C_S,
                    lw=4.2 if tag == "A" else 2.6,
                    solid_capstyle="butt", alpha=0.95 if tag == "A" else 0.75,
                    zorder=4 if tag == "A" else 3)
        for bb in b["base"]:
            if t0 <= bb < t1:
                ax.axvline((bb - t0) / core.SR, color="#999", lw=0.6,
                           ls=(0, (3, 3)), zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, (t1 - t0) / core.SR)
        ax.set_ylim(-3, 12 * np.log2(n) + 3)
        if col == 0:
            ax.set_ylabel("Semitones re 400 Hz")
            ax.plot([], [], color=C_A, lw=4.2, label="A (10 partials)")
            ax.plot([], [], color=C_S, lw=2.6, label="shifted copies")
            ax.legend(loc="upper right", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ax.barh(np.arange(n) + 1, lay["offset_st"],
            color=[C_S if v > 0 else "#6699cc" for v in lay["offset_st"]])
    ax.axvline(0, color="#333", lw=0.7)
    for s in (-1, -0.5, 0.5, 1):
        ax.axvline(s, color="#999", lw=0.5, ls=":")
    ax.set_xlabel("Offset (st)")
    ax.set_ylabel("Partial")
    ax.set_title("Frequency shift", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    ms = 1000.0 / core.SR
    d = [(o - coh["base"][tok]) * ms
         for i, tag, o, tok in coh["events"] if tag == "S"]
    ax.hist(d, bins=np.arange(-45, 46, 2.5), color=C_S)
    ax.set_xlabel("Time shift (ms)")
    ax.set_ylabel("Tones")
    ax.set_title("Time shift", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-partials", type=int, default=N_PARTIALS)
    p.add_argument("--df-st", type=float, nargs=2, default=list(DF_ST))
    p.add_argument("--shift-ms", type=float, nargs=2, default=list(SHIFT_MS))
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)

    lay = layout(args.n_partials, args.df_st)
    coh = build(lay, args.shift_ms, scramble_a=False)
    scr = build(lay, args.shift_ms, scramble_a=True)

    amp = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(b["x"]))) for b in (coh, scr))
    print(f"{core.SR} Hz | {2 * args.n_partials} channels: "
          f"{args.n_partials} coherent partials + {args.n_partials} copies "
          f"shifted {args.df_st[0]}-{args.df_st[1]} st and "
          f"{args.shift_ms[0]:.0f}-{args.shift_ms[1]:.0f} ms\n")

    for b, nm in ((coh, "syl_coherent"), (scr, "syl_scrambled")):
        b["x"] = b["x"] * amp
        core.render(OUT_DIR / nm, b["x"], args.keep_wav)
        print(report(b, nm))
        print(f"    -> {nm}.mp3\n")
    print(f"  -> {figure(coh, scr).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
