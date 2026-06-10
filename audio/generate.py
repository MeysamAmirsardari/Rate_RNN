"""
audio.generate
==============

Synthesise short, **playable** WAV clips for each model0 paradigm so you can
*hear* the stimuli (talks, intuition, sanity checks).

These are sonifications of the paradigm *structure*: the model's abstract
tonotopic channels are voiced as pure sine tones (with raised-cosine ramps so
there are no clicks), and the syllable paradigm uses the real recordings in
``tasks/syllables_wav/``.  Timings mirror the task configs:

    ab_ba.wav            two-tone streaming -- mostly AB, a rare BA
                         (50 ms tones, 30 ms intra-gap, 500 ms between pairs)
    oddball_ssa.wav      stimulus-specific adaptation -- frequent standard f1,
                         rare deviant f2 (50 ms tones, 150 ms SOA, 10% deviant)
    local_global.wav     5-tone sequences -- standard xxxxy, rare xxxxx
                         (50 ms tones, 100 ms intra-gap, 1 s between sequences)
    roving_tones.wav     roving oddball -- 3-tone words; the repeated standard
                         "roves" to a new word every block (180 ms tones, 1 s gap)
    roving_syllables.wav the same roving structure, voiced with real syllables
    sfg.wav              stochastic figure-ground -- a random tone cloud, then a
                         coherent figure (synchronous chords) pops in, then out

Run
    python -m audio.generate                  # all clips -> audio/
    python -m audio.generate --sr 44100
    python -m audio.generate --only sfg roving_syllables
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_WAV_DIR = _ROOT / "tasks" / "syllables_wav"
_OUT_DIR = Path(__file__).resolve().parent


# =====================================================================
#  Synthesis primitives
# =====================================================================
def tone(freq: float, dur_s: float, sr: int, amp: float = 0.6,
         ramp_s: float = 0.006) -> np.ndarray:
    """A sine tone with raised-cosine on/off ramps (click-free)."""
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    r = int(round(min(ramp_s, dur_s / 2) * sr))
    if r > 0:
        env = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r))
        x[:r] *= env
        x[-r:] *= env[::-1]
    return x


def silence(dur_s: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(dur_s * sr)), dtype=float)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0)


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 0 else x


def write_wav(path: Path, x: np.ndarray, sr: int) -> Path:
    """Write a mono 16-bit PCM WAV (stdlib only)."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def _load_syllable(name: str, sr: int) -> np.ndarray:
    """Load a real syllable recording, resampled to ``sr`` (needs librosa)."""
    import librosa
    y, _ = librosa.load(str(_WAV_DIR / f"{name}.wav"), sr=sr, mono=True)
    return normalize(y.astype(float), 0.9)


# =====================================================================
#  Paradigm clips  (each returns a float waveform in [-1, 1])
# =====================================================================
def make_ab_ba(sr: int, n_pairs: int = 24, p_BA: float = 0.18,
               seed: int = 0) -> np.ndarray:
    """Two-tone streaming: a stream of AB pairs with occasional BA."""
    rng = np.random.default_rng(seed)
    fA, fB = 494.0, 740.0                          # ~B4, ~F#5
    tone_d, intra, inter = 0.050, 0.030, 0.500
    out = [silence(0.3, sr)]
    for _ in range(n_pairs):
        a, b = (fB, fA) if rng.random() < p_BA else (fA, fB)
        out += [tone(a, tone_d, sr), silence(intra, sr),
                tone(b, tone_d, sr), silence(inter, sr)]
    return normalize(concat(*out))


def make_oddball(sr: int, n_tones: int = 150, p_dev: float = 0.10,
                 seed: int = 1) -> np.ndarray:
    """SSA: frequent standard f1, rare deviant f2 (never two deviants in a row)."""
    rng = np.random.default_rng(seed)
    f1, f2 = 1000.0, 1300.0
    tone_d, gap = 0.050, 0.100                      # SOA 150 ms
    out = [silence(0.3, sr)]
    prev_dev = False
    for _ in range(n_tones):
        dev = (not prev_dev) and (rng.random() < p_dev)
        out += [tone(f2 if dev else f1, tone_d, sr), silence(gap, sr)]
        prev_dev = dev
    return normalize(concat(*out))


def make_local_global(sr: int, n_seq: int = 16, p_dev: float = 0.2,
                      seed: int = 2) -> np.ndarray:
    """5-tone sequences: standard xxxxy (80%), rare deviant xxxxx (20%)."""
    rng = np.random.default_rng(seed)
    fx, fy = 800.0, 1300.0
    tone_d, intra, inter = 0.050, 0.100, 1.000
    out = [silence(0.3, sr)]
    for _ in range(n_seq):
        last = fx if rng.random() < p_dev else fy   # deviant = xxxxx
        seq = []
        for i in range(5):
            seq += [tone(fx if i < 4 else last, tone_d, sr)]
            seq += [silence(intra if i < 4 else inter, sr)]
        out += seq
    return normalize(concat(*out))


# tone characters -> frequency (log-spaced); A,B shared, C,D,E variable
_ROVING_FREQS = {"A": 400.0, "B": 600.0, "C": 900.0, "D": 1350.0, "E": 2000.0}
# tone characters -> real syllable recordings (A,B shared, C,D,E variable)
_ROVING_SYLL = {"A": "ba", "B": "da", "C": "ga", "D": "ka", "E": "ma"}


def _roving_words(deviant_pos: int = 1) -> list[str]:
    """3-tone words: the variable tone (C/D/E) sits at ``deviant_pos`` (1-based),
    A and B fill the other two slots in order -> CAB, DAB, EAB for pos 1."""
    shared_slots = [p for p in (1, 2, 3) if p != deviant_pos]
    words = []
    for var in ("C", "D", "E"):
        slots = [""] * 3
        slots[deviant_pos - 1] = var
        for s, p in zip(("A", "B"), shared_slots):
            slots[p - 1] = s
        words.append("".join(slots))
    return words


def make_roving_tones(sr: int, reps_per_block: int = 5,
                      deviant_pos: int = 1) -> np.ndarray:
    """Roving oddball: each word repeats ``reps_per_block`` times (a standard),
    then the standard 'roves' to the next word -- the change is the deviant."""
    tone_d, seq_gap = 0.180, 1.000
    out = [silence(0.3, sr)]
    for word in _roving_words(deviant_pos):
        for _ in range(reps_per_block):
            for ch in word:
                out.append(tone(_ROVING_FREQS[ch], tone_d, sr))
            out.append(silence(seq_gap, sr))
    return normalize(concat(*out))


def make_roving_syllables(sr: int, reps_per_block: int = 5,
                          deviant_pos: int = 1) -> np.ndarray:
    """Roving oddball voiced with real syllable recordings."""
    cache = {ch: _load_syllable(name, sr) for ch, name in _ROVING_SYLL.items()}
    syll_gap, seq_gap = 0.040, 1.000
    out = [silence(0.3, sr)]
    for word in _roving_words(deviant_pos):
        for _ in range(reps_per_block):
            for ch in word:
                out += [cache[ch], silence(syll_gap, sr)]
            out.append(silence(seq_gap - syll_gap, sr))
    return normalize(concat(*out))


def make_sfg(sr: int, pre_s: float = 3.0, fig_s: float = 4.0, post_s: float = 3.0,
             n_fig: int = 8, rate_hz: float = 4.0, seed: int = 7) -> np.ndarray:
    """Stochastic figure-ground: a random tone cloud over 37 log-spaced
    channels; during the middle epoch a fixed subset fires SYNCHRONOUS chords
    (the 'figure') so it perceptually pops out of the cloud."""
    rng = np.random.default_rng(seed)
    freqs = 1600.0 * 2.0 ** (np.arange(-12, 25) / 12.0)   # 37 ch, 800..6400 Hz
    N = len(freqs)
    total = pre_s + fig_s + post_s
    pip = 0.050
    grid = 0.050
    p = rate_hz * grid                                    # per-grid-slot prob
    n_slots = int(total / grid)
    fig_start, fig_end = pre_s, pre_s + fig_s
    out = silence(total + pip, sr)

    def add_pip(freq, t0, amp):
        i0 = int(round(t0 * sr))
        seg = tone(freq, pip, sr, amp=amp)
        out[i0:i0 + seg.size] += seg[:max(0, out.size - i0)]

    # background cloud: independent pips per channel
    for f in freqs:
        for k in range(n_slots):
            if rng.random() < p:
                t0 = k * grid + rng.uniform(-0.012, 0.012)
                add_pip(f, max(0.0, t0), amp=0.18)

    # figure: a fixed subset firing on a SHARED (synchronous) onset list,
    # only during the middle epoch
    fig_ch = rng.choice(N, size=n_fig, replace=False)
    k0, k1 = int(fig_start / grid), int(fig_end / grid)
    for k in range(k0, k1):
        if rng.random() < p:
            t0 = k * grid + rng.uniform(-0.012, 0.012)
            for ci in fig_ch:
                add_pip(freqs[ci], max(fig_start, t0), amp=0.22)
    return normalize(out)


# =====================================================================
#  Driver
# =====================================================================
CLIPS = {
    "ab_ba":            make_ab_ba,
    "oddball_ssa":      make_oddball,
    "local_global":     make_local_global,
    "roving_tones":     make_roving_tones,
    "roving_syllables": make_roving_syllables,
    "sfg":              make_sfg,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate playable paradigm audio.")
    ap.add_argument("--sr", type=int, default=22050, help="sample rate (Hz)")
    ap.add_argument("--outdir", default=str(_OUT_DIR), help="output directory")
    ap.add_argument("--only", nargs="+", choices=list(CLIPS), default=list(CLIPS),
                    help="generate only these clips")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        try:
            x = CLIPS[name](args.sr)
        except Exception as e:                            # e.g. librosa/wavs missing
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue
        path = write_wav(out / f"{name}.wav", x, args.sr)
        print(f"  wrote {path.name:22s} {x.size / args.sr:6.1f} s")
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
