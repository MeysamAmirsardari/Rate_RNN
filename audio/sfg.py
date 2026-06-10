"""
audio.sfg
=========

Faithful **Stochastic Figure-Ground (SFG)** stimulus generator and MP3
renderer (Teki, Chait, Kumar, Shamma & Griffiths, *eLife* 2013, e00699;
Teki et al., *J Neurosci* 2011).

The SFG stimulus -- exactly as in the paper's Materials & Methods:

  * a sequence of **50 ms chords** with **0 ms inter-chord interval**;
  * each chord is the sum of a **constant number of equal-amplitude, random-
    phase pure tones** (``tones_per_chord``, default 22) -- constant so the
    **time marginal is flat** (Teki uses a random 5-15; we don't, on purpose);
  * tone frequencies are drawn from a pool of **129 values spaced 1/24 octave
    apart between 179 and 7246 Hz**;
  * the onset and offset of **each chord** are shaped by a **10 ms raised-cosine
    ramp** (so repeated tones pulse at the chord rate -> the figure "warbles");
  * a **figure** is a set of ``coherence`` channels that fire **synchronously**
    at the **same per-channel rate as the background** (it REPLACES background
    tones, never adds energy) -- so the **channel marginal stays flat** and the
    figure is a pure *temporal-coherence* cue, not an energy cue (rate-matched,
    as in tasks/sfg2; see the rate-matching note in the code).

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
# Teki's background is 5-15 tones/chord (random).  We instead hold the count
# CONSTANT so the time marginal is flat and the figure can't be heard as a
# denser interval -- see the rate-matching note below.  K must be >= the
# largest coherence, and sets the figure rate (gamma = K/N).
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
#  Rate matching  (why the figure must NOT add energy)
# =====================================================================
# A figure must be detectable ONLY by temporal coherence, never by energy.
# Two marginals are therefore held flat:
#   * time marginal  -- every chord has the SAME number of tones K; the figure
#     REPLACES background tones, it never adds on top (no denser interval);
#   * channel marginal -- the figure channels fire at the SAME per-channel rate
#     as the background, differing only in that they fire SYNCHRONOUSLY.
# Keeping K tones/chord while the m figure channels co-fire on a fraction gamma
# of chords forces, for EVERY channel, rate = gamma = K/N (background rate =
# (K - gamma*m)/(N - m) = gamma  =>  gamma = K/N).  So a figure fires
# synchronously every ~N/K chords -- exactly the background rate -- and
# coherence is the only cue.  This matches tasks/sfg2 (rate-matched) and is
# STRICTER than Teki et al., whose figure is on every chord (an energy
# confound they balance only across trials).


# =====================================================================
#  One canonical experiment trial  (2 s, 40 chords, one figure)
# =====================================================================
def make_sfg_trial(coherence: int, duration: int, sr: int = SR_DEFAULT,
                   n_chords: int = 40, tones_per_chord: int = 22,
                   fig_onset_chords: int | None = None,
                   seed: int = 0) -> np.ndarray:
    """One SFG trial: ``n_chords`` 50 ms chords, each with exactly
    ``tones_per_chord`` tones (flat time marginal).  A figure of ``coherence``
    channels spans ``duration`` chords, firing SYNCHRONOUSLY at the background
    per-channel rate gamma=K/N (flat channel marginal) -- so it is detectable
    only by temporal coherence.  Onset jittered 15-20 chords unless given."""
    rng = np.random.default_rng(seed)
    pool = freq_pool(); N = len(pool); K = tones_per_chord
    gamma = K / N
    chord_s, ramp_s = CHORD_MS / 1000.0, RAMP_MS / 1000.0
    if fig_onset_chords is None:
        fig_onset_chords = int(rng.integers(15, 21))
    fig = rng.choice(N, size=coherence, replace=False)
    non_fig = np.setdiff1d(np.arange(N), fig)
    fig_win = set(range(fig_onset_chords, min(n_chords, fig_onset_chords + duration)))
    parts = []
    for c in range(n_chords):
        if c not in fig_win:                                 # pure background
            idx = rng.choice(N, K, replace=False)
        elif rng.random() < gamma:                           # figure fires (synchronous)
            idx = np.concatenate([fig, rng.choice(non_fig, K - coherence, replace=False)])
        else:                                                # figure silent this chord
            idx = rng.choice(non_fig, K, replace=False)
        parts.append(_chord(pool[idx], sr, chord_s, ramp_s, rng))
    return _norm(np.concatenate(parts))


# =====================================================================
#  Long continuous scene  (figures of varied coherence emerging/dissolving)
# =====================================================================
def make_sfg(sr: int = SR_DEFAULT, total_s: float = 120.0, seed: int = 0,
             tones_per_chord: int = 22,
             coherence_cycle: tuple[int, ...] = (12, 9, 6, 4, 3, 2),
             fig_chords: int = 48, gap_chords: int = 22,
             lead_s: float = 2.0, return_active: bool = False):
    """A long ongoing SFG cloud -- every chord exactly ``tones_per_chord``
    tones -- with RATE-MATCHED coherent figures emerging and dissolving.

    After ``lead_s`` of background, figures are inserted on a ``fig_chords``
    on / ``gap_chords`` off cycle, their coherence stepping through
    ``coherence_cycle`` (salient -> subtle).  Each figure's channels fire
    SYNCHRONOUSLY at the background per-channel rate (gamma=K/N, ~every N/K
    chords), so the channel and time marginals stay flat -- the figure is a
    pure temporal-coherence cue, not an energy cue (see the note above)."""
    rng = np.random.default_rng(seed)
    pool = freq_pool(); N = len(pool); K = tones_per_chord
    gamma = K / N
    all_idx = np.arange(N)
    chord_s, ramp_s = CHORD_MS / 1000.0, RAMP_MS / 1000.0
    n_total = int(round(total_s / chord_s))

    # per-chord figure channel set (None outside figure windows)
    fig_at: list[np.ndarray | None] = [None] * n_total
    c = int(round(lead_s / chord_s))
    k = 0
    while c < n_total:
        m = coherence_cycle[k % len(coherence_cycle)]
        k += 1
        fig = rng.choice(N, size=m, replace=False)
        for j in range(min(fig_chords, n_total - c)):
            fig_at[c + j] = fig
        c += fig_chords + gap_chords

    active = np.zeros((n_total, N), dtype=bool)              # chord x channel
    parts = []
    for ci in range(n_total):
        fig = fig_at[ci]
        if fig is None:                                      # pure background
            idx = rng.choice(N, K, replace=False)
        else:
            non_fig = np.setdiff1d(all_idx, fig)
            if rng.random() < gamma:                         # figure fires (synchronous)
                idx = np.concatenate([fig, rng.choice(non_fig, K - len(fig), replace=False)])
            else:                                            # figure silent this chord
                idx = rng.choice(non_fig, K, replace=False)
        active[ci, idx] = True
        parts.append(_chord(pool[idx], sr, chord_s, ramp_s, rng))
    x = _norm(np.concatenate(parts))
    return (x, active, fig_at) if return_active else x


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
