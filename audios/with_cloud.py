"""
audios.with_cloud
=================

The same three stimuli, each buried in an unpredictable tone cloud.

    ab_df01_cloud.mp3         one stream, 1 semitone,   in cloud
    ab_df09_cloud.mp3         one stream, 9 semitones,  in cloud
    ab_cd_incoherent_cloud.mp3   two streams, in cloud

The targets are untouched -- same frequencies, same jittered onsets, same
gating -- so each cloud file pairs exactly with its clean counterpart and the
cloud is the only difference.

The cloud
---------
See ``audios.cloud``.  Four tones sounding at every instant with no
fluctuation, every channel used equally often, and no channel reliably
following any other.  It spans just under three octaves on a semitone grid
around the targets, with the target semitones removed so the cloud can never
sound a target frequency.

Level
-----
Cloud tones are at the **same level as target tones** by default, which is the
convention in figure-ground stimuli and is what makes the task about temporal
structure rather than about loudness.  ``--cloud-db`` moves the cloud if a
gentler version is wanted; at 0 dB the target is one of five concurrent tones
and is genuinely hard to hold on to.

The three cloud files share one scale factor, as the three clean files share
another.  Within each set a tone is at the same SPL wherever it appears.  The
two sets cannot share a scale: a cloud file has six tones sounding where a
clean file has one, so matching per-tone level across both would either clip
the cloud files or leave the clean ones needlessly quiet.  Per-tone level is
reported for both so the difference is on the record rather than implied.

Run
---
    python -m audios.with_cloud
    python -m audios.with_cloud --cloud-db -6     # a quieter background
    python -m audios.with_cloud --keep-wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from . import cloud, core, two_stream, two_tone
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audios import cloud, core, two_stream, two_tone  # type: ignore

OUT_DIR = Path(__file__).resolve().parent


def targets() -> list:
    """The three stimuli, unscaled, with the semitones the cloud must avoid.

    ``pips`` carries (frequency, onset sample, colour) for every target tone,
    so the check figure can draw the targets inside the cloud rather than
    inferring them back out of the mixture.
    """
    step = core.samples(two_tone.TONE_MS)
    out = []
    for df in (1.0, 9.0):
        x, f_b, on = two_tone.build(df)
        pips = ([(two_tone.F_A, int(o), "#7C102A") for o in on]
                + [(f_b, int(o) + step, "#BD6B6B") for o in on])
        out.append(dict(name=f"ab_df{int(df):02d}", x=x, pips=pips,
                        exclude=[0.0, df],
                        note=f"one stream, df = {df:.0f} st"))
    b = two_stream.build()
    fa, fb, fc, fd = b["freqs"]
    pips = ([(fa, int(o), "#7C102A") for o in b["on_ab"]]
            + [(fb, int(o) + step, "#BD6B6B") for o in b["on_ab"]]
            + [(fc, int(o), "#1B4F8A") for o in b["on_cd"]]
            + [(fd, int(o) + step, "#2166AC") for o in b["on_cd"]])
    out.append(dict(name="ab_cd_incoherent", x=b["x"], pips=pips,
                    exclude=[0.0, 1.0, 2.0, 3.0],
                    note="two streams, 1 st apart throughout"))
    return out


def figure(made: list, stem: str = "with_cloud_check") -> Path:
    """Every tone as a dash: cloud dark, targets coloured.

    Drawn from the placements rather than from the mixture, because a
    spectrogram cannot separate tones a semitone apart at 40 ms and would
    show the targets smeared into the cloud -- which is true of the sound and
    is exactly why the picture has to come from the score instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0, t1 = core.samples(600.0), core.samples(2600.0)
    slot = core.samples(cloud.SLOT_MS)
    step = slot // cloud.N_VOICES
    dur = core.samples(cloud.TONE_MS)

    fig, axes = plt.subplots(1, len(made), figsize=(4.7 * len(made), 5.6),
                             constrained_layout=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), made):
        st = lambda f: 12 * np.log2(f / cloud.F_REF)
        for s in range(m["idx"].shape[0]):
            for v in range(cloud.N_VOICES):
                o = s * slot + v * step
                if t0 <= o < t1:
                    y = st(m["freqs"][m["idx"][s, v]])
                    ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR],
                            [y, y], color="#2D3748", lw=3.0,
                            solid_capstyle="butt")
        for f, o, col in m["pips"]:
            if t0 <= o < t1:
                ax.plot([(o - t0) / core.SR, (o - t0 + dur) / core.SR],
                        [st(f)] * 2, color=col, lw=4.0,
                        solid_capstyle="butt", zorder=5)
        ax.set_title(m["note"])
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, (t1 - t0) / core.SR)
        ax.set_ylim(cloud.ST_LO - 1, cloud.ST_HI + 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Semitones re 1000 Hz")

    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cloud-db", type=float, default=0.0,
                   help="cloud level relative to the target tones")
    p.add_argument("--keep-wav", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    gain = 10.0 ** (args.cloud_db / 20.0)
    print(f"{core.SR} Hz | cloud: tone {cloud.TONE_MS:.0f} ms, slot "
          f"{cloud.SLOT_MS:.0f} ms, {cloud.N_VOICES} voices -> "
          f"{cloud.N_VOICES - 1} sounding at all times | cloud level "
          f"{args.cloud_db:+.0f} dB re target\n")

    made = []
    for t in targets():
        c, freqs, idx = cloud.build(t["x"].size, t["exclude"], seed=args.seed)
        made.append(dict(t, mix=t["x"] + gain * c, freqs=freqs, idx=idx,
                         cloud=gain * c))

    # One scale for the set: the loudest file decides, so a tone is at the
    # same level in all three and none of them clips.
    top = max(float(np.max(np.abs(m["mix"]))) for m in made)
    amp = 10.0 ** (core.PEAK_DBFS / 20.0) / top

    for m in made:
        x = m["mix"] * amp
        mp3 = core.render(OUT_DIR / f"{m['name']}_cloud", x, args.keep_wav)
        tone_db = 20 * np.log10(amp)
        print(f"{m['note']}")
        print(cloud.report(m["cloud"] / max(gain, 1e-9), m["freqs"], m["idx"]))
        print(f"  {core.levels(x)} | a single tone is at {tone_db:+.1f} dBFS "
              f"(clean set: {20 * np.log10(core.TONE_AMP):+.1f})")
        print(f"  -> {mp3.name}\n")
    print(f"  -> {figure(made).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
