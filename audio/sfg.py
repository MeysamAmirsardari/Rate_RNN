"""
audio.sfg
=========

Faithful **Stochastic Figure-Ground (SFG)** stimulus generator and MP3
renderer (Teki, Chait, Kumar, Shamma & Griffiths, *eLife* 2013, e00699;
Teki et al., *J Neurosci* 2011).

The SFG stimulus -- exactly as in the paper's Materials & Methods:

  * a sequence of **50 ms chords** with **0 ms inter-chord interval**;
  * each chord is the sum of a **random number of equal-amplitude pure tones**
    (default average 10, uniformly 5-15), with **random phase**;
  * tone frequencies are drawn from a pool of **129 values spaced 1/24 octave
    apart between 179 and 7246 Hz**;
  * the onset and offset of **each chord** are shaped by a **10 ms raised-cosine
    ramp** (so repeated tones pulse at the chord rate -> the figure "warbles");
  * a **figure** is a set of ``coherence`` fixed frequencies **repeated across
    ``duration`` consecutive chords**, added on top of the random background --
    a temporally coherent object that pops out of the incoherent cloud.

Rendered at **44.1 kHz** (the paper's rate).  Rather than isolated 2 s trials,
``make_sfg`` renders a *long, continuous scene*: an ongoing chord cloud in which
figures of varied coherence appear and dissolve, so you can hear the object
emerge.  ``make_sfg_trial`` builds one canonical experiment trial.

Run
    python -m audio.sfg                       # ~120 s scene -> audio/sfg.mp3
    python -m audio.sfg --seconds 60 --seed 3
    python -m audio.sfg --format wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_OUT_DIR = Path(__file__).resolve().parent

# ---- stimulus constants (Teki et al. 2013, Methods) ----
F_MIN_HZ   = 179.0          # pool low edge
F_MAX_HZ   = 7246.0         # pool high edge
OCT_STEP   = 1.0 / 24.0     # 1/24-octave spacing between successive components
CHORD_MS   = 50.0           # chord duration
RAMP_MS    = 10.0           # raised-cosine onset/offset ramp per chord
BG_MIN     = 5              # background tones per chord (min)
BG_MAX     = 15             # background tones per chord (max; average 10)
SR_DEFAULT = 44100


def freq_pool() -> np.ndarray:
    """The 129-frequency pool: 1/24-octave spaced, 179 -> ~7246 Hz."""
    n = int(round(np.log2(F_MAX_HZ / F_MIN_HZ) / OCT_STEP)) + 1   # 129
    return F_MIN_HZ * 2.0 ** (np.arange(n) * OCT_STEP)


def _chord(freqs_hz: np.ndarray, sr: int, chord_s: float, ramp_s: float,
           rng: np.random.Generator) -> np.ndarray:
    """One chord: sum of equal-amplitude random-phase sines, raised-cosine ramped."""
    n = int(round(chord_s * sr))
    t = np.arange(n) / sr
    x = np.zeros(n)
    for f in freqs_hz:
        x += np.sin(2 * np.pi * f * t + rng.uniform(0.0, 2 * np.pi))
    r = int(round(ramp_s * sr))
    if r > 0 and 2 * r <= n:
        env = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        x[:r] *= env
        x[-r:] *= env[::-1]
    return x


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Robust peak normalisation (99.9th percentile) + hard clip."""
    p = float(np.percentile(np.abs(x), 99.9)) if x.size else 0.0
    return np.clip(x / p * peak, -1.0, 1.0) if p > 0 else x


# =====================================================================
#  One canonical experiment trial  (2 s, 40 chords, one figure)
# =====================================================================
def make_sfg_trial(coherence: int, duration: int, sr: int = SR_DEFAULT,
                   n_chords: int = 40, fig_onset_chords: int | None = None,
                   seed: int = 0) -> np.ndarray:
    """One SFG trial exactly as in experiment 1: ``n_chords`` 50 ms chords; a
    figure of ``coherence`` repeated frequencies spanning ``duration`` chords,
    onset jittered 15-20 chords (750-1000 ms) post onset unless given."""
    rng = np.random.default_rng(seed)
    pool = freq_pool()
    chord_s, ramp_s = CHORD_MS / 1000.0, RAMP_MS / 1000.0
    if fig_onset_chords is None:
        fig_onset_chords = int(rng.integers(15, 21))
    fig = rng.choice(pool, size=coherence, replace=False)
    fig_chords = range(fig_onset_chords, min(n_chords, fig_onset_chords + duration))
    parts = []
    for c in range(n_chords):
        nbg = int(rng.integers(BG_MIN, BG_MAX + 1))
        freqs = rng.choice(pool, size=nbg, replace=False)
        if c in fig_chords:
            freqs = np.union1d(freqs, fig)
        parts.append(_chord(freqs, sr, chord_s, ramp_s, rng))
    return _norm(np.concatenate(parts))


# =====================================================================
#  Long continuous scene  (figures of varied coherence emerging/dissolving)
# =====================================================================
def make_sfg(sr: int = SR_DEFAULT, total_s: float = 120.0, seed: int = 0,
             coherence_cycle: tuple[int, ...] = (12, 10, 8, 6, 5, 4, 3, 2),
             fig_chords: int = 16, gap_chords: int = 28,
             lead_s: float = 2.5) -> np.ndarray:
    """A long ongoing SFG cloud with figures appearing and dissolving.

    After ``lead_s`` of pure background, figures are inserted on a
    ``fig_chords`` (default 16 = 800 ms) on / ``gap_chords`` (default 28 =
    1.4 s) off cycle, their coherence stepping through ``coherence_cycle``
    (salient -> subtle) and repeating -- so the listener hears the figure pop
    out clearly at high coherence and only faintly at low coherence, against an
    unbroken random background.  All other acoustics are the paper's exactly."""
    rng = np.random.default_rng(seed)
    pool = freq_pool()
    chord_s, ramp_s = CHORD_MS / 1000.0, RAMP_MS / 1000.0
    n_total = int(round(total_s / chord_s))

    # per-chord figure plan: None (background only) or the frozen figure freqs
    plan: list[np.ndarray | None] = [None] * n_total
    c = int(round(lead_s / chord_s))
    k = 0
    while c < n_total:
        coh = coherence_cycle[k % len(coherence_cycle)]
        k += 1
        fig = rng.choice(pool, size=coh, replace=False)
        for j in range(fig_chords):
            if c + j < n_total:
                plan[c + j] = fig
        c += fig_chords + gap_chords

    parts = []
    for ci in range(n_total):
        nbg = int(rng.integers(BG_MIN, BG_MAX + 1))
        freqs = rng.choice(pool, size=nbg, replace=False)
        if plan[ci] is not None:
            freqs = np.union1d(freqs, plan[ci])
        parts.append(_chord(freqs, sr, chord_s, ramp_s, rng))
    return _norm(np.concatenate(parts))


# =====================================================================
#  Driver
# =====================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a faithful SFG scene.")
    ap.add_argument("--seconds", type=float, default=120.0, help="scene length")
    ap.add_argument("--sr", type=int, default=SR_DEFAULT, help="sample rate (Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["mp3", "wav"], default="mp3")
    ap.add_argument("--outdir", default=str(_OUT_DIR))
    args = ap.parse_args(argv)

    from .generate import write_audio                       # lazy: avoid import cycle
    x = make_sfg(sr=args.sr, total_s=args.seconds, seed=args.seed)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    path = write_audio(out / f"sfg.{args.format}", x, args.sr)
    print(f"  wrote {path}  ({x.size / args.sr:.1f} s, {len(freq_pool())} freqs, "
          f"{CHORD_MS:.0f} ms chords)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
