"""
audios.complex_set
==================

The same three stimuli, and their clouds, built from **harmonic complexes**
instead of pure tones.

    cx_tones_demo.mp3            A alone, B alone, then the AB sequence
    cx_ab_df01.mp3               one stream, F0 separation 1 semitone
    cx_ab_df09.mp3               one stream, 9 semitones
    cx_ab_cd_incoherent.mp3      two streams, 1 semitone apart throughout
    cx_*_cloud.mp3               each of the three, inside a complex cloud

Why complexes
-------------
A pure tone carries frequency and almost nothing else.  A harmonic complex
carries a **pitch**, which is a far stronger perceptual attribute: it is
carried redundantly by every harmonic, it survives masking that removes any
one of them, and two complexes differing in F0 segregate where two pure tones
of the same separation do not.  Harmonicity is itself a grouping cue -- the
harmonics of one F0 fuse into one object -- so the tokens arrive as objects
rather than as points on a frequency axis.

Every complex has harmonics up to a **fixed 4 kHz ceiling** with 1/h
amplitudes, so all of them occupy the same band and differ in periodicity
rather than in brightness.  A fixed harmonic *count* would make the higher-F0
tone audibly brighter and brightness, not pitch, would be doing the work.
Each is RMS-normalised, so a complex with twenty harmonics is no louder than
one with four.

F0 is 400 Hz rather than the 1000 Hz of the pure-tone set.  A 40 ms tone at
400 Hz holds 16 periods, which is enough for a solid pitch; at 200 Hz it would
hold 8, and the pitch of so short a tone starts to weaken.

Everything else is unchanged
----------------------------
Same 40 ms tones and 5 ms ramps, same 0 ms gap inside the doublet, same 5 Hz
mean rate with the same jitter, same anti-phase offset between the two
streams, same cloud construction -- four tones sounding at every instant,
every channel used equally often, no channel following any other reliably,
and never a target F0.

Run
---
    python -m audios.complex_set
    python -m audios.complex_set --gap-ms 20    # separate the two tones
    python -m audios.complex_set --cloud-db -6
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
PAIR_HZ = 5.0
N_PAIRS = 60
F0_A = 400.0
LEAD_MS, TAIL_MS = 500.0, 700.0

CLOUD_SLOT_MS = 50.0
CLOUD_VOICES = 5                 # tone/slot = 4/5 -> exactly 4 sounding
CLOUD_ST_LO, CLOUD_ST_HI = -12, 15


def f_of(st: float) -> float:
    return F0_A * 2.0 ** (st / 12.0)


def doublet(st1: float, st2: float, gap_ms: float) -> np.ndarray:
    return np.concatenate([core.complex_tone(f_of(st1), TONE_MS),
                           np.zeros(core.samples(gap_ms)),
                           core.complex_tone(f_of(st2), TONE_MS)])


def one_stream(df: float, gap_ms: float, jitter_ms: float = 50.0,
               seed: int = 0) -> dict:
    ev = doublet(0.0, df, gap_ms)
    rng = np.random.default_rng(seed)
    on = core.jittered_onsets(N_PAIRS, 1000.0 / PAIR_HZ, jitter_ms, rng,
                              phase_ms=LEAD_MS)
    total = int(on[-1]) + ev.size + core.samples(TAIL_MS)
    step = core.samples(TONE_MS + gap_ms)
    pips = ([(0.0, int(o)) for o in on] + [(df, int(o) + step) for o in on])
    return dict(x=core.place(on, ev, total), pips=pips, exclude=[0.0, df])


def two_streams(gap_ms: float, jitter_ms: float = 25.0, phase_ms: float = 100.0,
                seed: int = 0) -> dict:
    ab, cd = doublet(0.0, 1.0, gap_ms), doublet(2.0, 3.0, gap_ms)
    rng = np.random.default_rng(seed)
    per = 1000.0 / PAIR_HZ
    on_ab = core.jittered_onsets(N_PAIRS, per, jitter_ms, rng, phase_ms=LEAD_MS)
    on_cd = core.jittered_onsets(N_PAIRS, per, jitter_ms, rng,
                                 phase_ms=LEAD_MS + phase_ms)
    total = int(max(on_ab[-1], on_cd[-1])) + ab.size + core.samples(TAIL_MS)
    step = core.samples(TONE_MS + gap_ms)
    pips = ([(0.0, int(o)) for o in on_ab] + [(1.0, int(o) + step) for o in on_ab]
            + [(2.0, int(o)) for o in on_cd] + [(3.0, int(o) + step) for o in on_cd])
    asyn = np.min(np.abs(on_cd[:, None] - on_ab[None, :]), axis=1) / core.SR * 1e3
    return dict(x=core.place(on_ab, ab, total) + core.place(on_cd, cd, total),
                pips=pips, exclude=[0.0, 1.0, 2.0, 3.0], asyn=asyn)


def cloud(total: int, exclude_st, seed: int = 0) -> dict:
    """Complex-tone cloud: uniform in time, uniform in F0, unpredictable."""
    grid = np.arange(CLOUD_ST_LO, CLOUD_ST_HI + 1, dtype=float)
    keep = [not any(abs(s - e) < 0.5 for e in exclude_st) for s in grid]
    sts = grid[keep]
    pips = [core.complex_tone(f_of(s), TONE_MS) for s in sts]

    slot = core.samples(CLOUD_SLOT_MS)
    step = slot // CLOUD_VOICES
    n_slots = int(np.ceil(total / slot)) + 1

    rng = np.random.default_rng(seed)
    idx = np.empty((n_slots, CLOUD_VOICES), dtype=int)
    pack, i, prev = rng.permutation(sts.size), 0, set()
    for s in range(n_slots):
        group: list[int] = []
        for _ in range(CLOUD_VOICES):
            if i >= pack.size:
                pack, i = rng.permutation(sts.size), 0
            j = i
            while j < pack.size and (pack[j] in prev or pack[j] in group):
                j += 1
            if j >= pack.size:
                j = i
            pack[i], pack[j] = pack[j], pack[i]
            group.append(int(pack[i]))
            i += 1
        idx[s] = group
        prev = set(group)

    x = np.zeros(n_slots * slot + (CLOUD_VOICES - 1) * step + pips[0].size)
    for s in range(n_slots):
        for v in range(CLOUD_VOICES):
            o = s * slot + v * step
            x[o:o + pips[0].size] += pips[idx[s, v]]
    return dict(x=x[:total], sts=sts, idx=idx)


def alternating(df: float, jitter_ms: float = 0.0, seed: int = 0) -> dict:
    """A B A B ... evenly spaced -- the classical streaming stimulus.

    Why this exists, when it was not asked for
    ------------------------------------------
    The specified sequence puts A and B hard against each other and then 120 ms
    of silence: 0 ms between the two tones of a doublet, 160 ms between one
    doublet and the next.  Temporal proximity is one of the strongest grouping
    cues there is, so that spacing *binds* A to B and separates doublets from
    each other.  It is a stimulus built to be heard as one stream of two-note
    events, and no amount of frequency separation will readily split it,
    because splitting it means separating the two tones that are closest
    together in time.

    Stream segregation is normally shown with tones spaced **evenly**: A and B
    each every 200 ms, interleaved, so a tone is 100 ms from each of its
    neighbours and proximity no longer favours either grouping.  Then a small
    separation is heard as one galloping stream and a large one splits into
    two, which is the effect the doublet version cannot easily show.

    Each tone type still occurs at 5 Hz, so this keeps the rate that was
    specified and changes only how the two are spaced within it.
    """
    ev_a = core.complex_tone(f_of(0.0), TONE_MS)
    ev_b = core.complex_tone(f_of(df), TONE_MS)
    per = 1000.0 / PAIR_HZ                       # 200 ms per tone TYPE
    rng = np.random.default_rng(seed)
    on_a = core.jittered_onsets(N_PAIRS, per, jitter_ms, rng, phase_ms=LEAD_MS)
    on_b = core.jittered_onsets(N_PAIRS, per, jitter_ms, rng,
                                phase_ms=LEAD_MS + per / 2.0)
    total = int(max(on_a[-1], on_b[-1])) + ev_a.size + core.samples(TAIL_MS)
    pips = ([(0.0, int(o)) for o in on_a] + [(df, int(o)) for o in on_b])
    return dict(x=core.place(on_a, ev_a, total) + core.place(on_b, ev_b, total),
                pips=pips, exclude=[0.0, df])


def demo(gap_ms: float) -> np.ndarray:
    """A alone, then B alone, then the pair -- so the tones can be heard."""
    def run(st, n, period_ms):
        ev = core.complex_tone(f_of(st), TONE_MS)
        on = np.arange(n) * core.samples(period_ms)
        return core.place(on, ev, int(on[-1]) + ev.size
                          + core.samples(period_ms))
    gap = np.zeros(core.samples(600.0))
    ab = doublet(0.0, 9.0, gap_ms)
    seq_on = np.arange(20) * core.samples(200.0)
    seq = core.place(seq_on, ab, int(seq_on[-1]) + ab.size
                     + core.samples(400.0))
    return np.concatenate([np.zeros(core.samples(300.0)),
                           run(0.0, 6, 400.0), gap,
                           run(9.0, 6, 400.0), gap, seq,
                           np.zeros(core.samples(500.0))])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gap-ms", type=float, default=0.0)
    p.add_argument("--cloud-db", type=float, default=0.0)
    p.add_argument("--keep-wav", action="store_true")
    args = p.parse_args(argv)
    g = args.gap_ms

    print(f"{core.SR} Hz | harmonic complexes, F0 base {F0_A:.0f} Hz, "
          f"harmonics to 4 kHz (1/h), RMS-matched | tone {TONE_MS:.0f} ms, "
          f"ramp {core.RAMP_MS:.0f} ms | A-B gap {g:.0f} ms\n")
    for st, nm in ((0.0, "A"), (1.0, "B (df 1)"), (9.0, "B (df 9)"),
                   (2.0, "C"), (3.0, "D")):
        print(f"  {nm:10s} F0 {f_of(st):7.2f} Hz  "
              f"{core.n_harmonics(f_of(st))} harmonics")

    items = [("cx_ab_df01", one_stream(1.0, g), "one stream, df 1 st"),
             ("cx_ab_df09", one_stream(9.0, g), "one stream, df 9 st"),
             ("cx_ab_cd_incoherent", two_streams(g), "two streams, 1 st apart"),
             ("cx_abab_df01", alternating(1.0),
              "A B A B evenly spaced, df 1 st"),
             ("cx_abab_df09", alternating(9.0),
              "A B A B evenly spaced, df 9 st")]

    clean = [dict(name=n, note=d, **b) for n, b, d in items]
    amp_clean = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(c["x"]))) for c in clean)

    gain = 10.0 ** (args.cloud_db / 20.0)
    mixed = []
    for c in clean:
        cl = cloud(c["x"].size, c["exclude"])
        mixed.append(dict(c, mix=c["x"] + gain * cl["x"], cl=cl))
    amp_cloud = 10 ** (core.PEAK_DBFS / 20) / max(
        float(np.max(np.abs(m["mix"]))) for m in mixed)

    print()
    core.render(OUT_DIR / "cx_tones_demo",
                core.scale(demo(g), amp_clean), args.keep_wav)
    print("  -> cx_tones_demo.mp3   (A x6, B x6, then the AB sequence)")

    for c, m in zip(clean, mixed):
        core.render(OUT_DIR / c["name"], c["x"] * amp_clean, args.keep_wav)
        core.render(OUT_DIR / f"{c['name']}_cloud", m["mix"] * amp_cloud,
                    args.keep_wav)
        counts = np.bincount(m["cl"]["idx"].ravel())
        extra = ""
        if "asyn" in c:
            extra = (f"; stream asynchrony {c['asyn'].mean():.0f} +- "
                     f"{c['asyn'].std():.0f} ms, min {c['asyn'].min():.0f}")
        print(f"\n  {c['note']}{extra}")
        print(f"    clean {core.levels(c['x'] * amp_clean)}")
        print(f"    cloud {m['cl']['sts'].size} F0s "
              f"{f_of(m['cl']['sts'][0]):.0f}-{f_of(m['cl']['sts'][-1]):.0f} Hz, "
              f"per-F0 use spread {counts.max() - counts.min()}, "
              f"{CLOUD_VOICES - 1} sounding at all times")
        print(f"    cloud {core.levels(m['mix'] * amp_cloud)}")
        print(f"    -> {c['name']}.mp3   {c['name']}_cloud.mp3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
