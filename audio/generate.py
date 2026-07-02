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
    python -m audio.generate                  # all clips (mp3) -> audio/
    python -m audio.generate --format wav     # 16-bit PCM WAV instead
    python -m audio.generate --only sfg roving_syllables --sr 44100
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


def write_audio(path: Path, x: np.ndarray, sr: int) -> Path:
    """Write mono audio; the format is chosen by the file suffix:
    ``.mp3`` via soundfile/libsndfile (libsndfile >= 1.1, no ffmpeg needed),
    ``.wav`` via the stdlib ``wave`` writer (16-bit PCM)."""
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    if path.suffix.lower() == ".mp3":
        import soundfile as sf
        sf.write(str(path), x, sr, format="MP3")
    else:
        pcm = (x * 32767.0).astype("<i2")
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
def make_ab_ba(sr: int, n_pairs: int = 200, p_BA: float = 0.18,
               seed: int = 0) -> np.ndarray:
    """Two-tone streaming: a stream of AB pairs with occasional BA."""
    rng = np.random.default_rng(seed)
    fA, fB = 494.0, 2140.0                          # ~B4, ~F#5
    tone_d, intra, inter = 0.050, 0.030, 0.500
    out = [silence(0.3, sr)]
    for _ in range(n_pairs):
        a, b = (fB, fA) if rng.random() < p_BA else (fA, fB)
        out += [tone(a, tone_d, sr), silence(intra, sr),
                tone(b, tone_d, sr), silence(inter, sr)]
    return normalize(concat(*out))


def make_oddball(sr: int, n_tones: int = 350, p_dev: float = 0.10,
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


def make_local_global(sr: int, n_seq: int = 200, p_dev: float = 0.2,
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


def _roving_words(deviant_pos: int = 3) -> list[str]:
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


def _block_order(words: list[str], n_blocks_per_word: int,
                 rng: np.random.Generator) -> list[str]:
    """Constrained-random block order: STRICTLY no two consecutive same-word blocks.
    Uses a backtracking approach to guarantee the constraint is never violated."""
    while True:
        counts = {w: n_blocks_per_word for w in words}
        order: list[str] = []
        stuck = False

        for _ in range(len(words) * n_blocks_per_word):
            # Find available words that have count > 0 AND do not repeat the previous word
            avail = [w for w in words if counts[w] > 0 and (not order or w != order[-1])]

            if not avail:
                # The generator painted itself into a corner. Backtrack and restart.
                stuck = True
                break

            w = avail[int(rng.integers(len(avail)))]
            order.append(w)
            counts[w] -= 1

        if not stuck:
            return order


def make_roving_tones(sr: int, n_blocks_per_word: int = 10,
                      n_reps_per_block: int = 10, deviant_pos: int = 3,
                      seed: int = 0) -> np.ndarray:
    """Roving oddball (Audio): 3 words x ``n_blocks_per_word`` blocks in a
    strictly constrained random order. Each block repeats ``n_reps_per_block`` times.
    180 ms tones, no intra-word gap, 1 s between sequences."""
    words = _roving_words(deviant_pos)
    order = _block_order(words, n_blocks_per_word, np.random.default_rng(seed))

    tone_d, seq_gap = 0.180, 1.000
    out = [silence(0.3, sr)]

    for word in order:
        for _ in range(n_reps_per_block):
            for ch in word:
                out.append(tone(_ROVING_FREQS[ch], tone_d, sr))
            out.append(silence(seq_gap, sr))

    return normalize(concat(*out))


def make_roving_syllables(sr: int, n_blocks_per_word: int = 10,
                          n_reps_per_block: int = 10, deviant_pos: int = 3,
                          seed: int = 0) -> np.ndarray:
    """The same roving block structure as make_roving_tones, voiced with the
    real syllable recordings. Includes a natural 40ms intra-syllable gap."""
    words = _roving_words(deviant_pos)
    order = _block_order(words, n_blocks_per_word, np.random.default_rng(seed))
    cache = {ch: _load_syllable(name, sr) for ch, name in _ROVING_SYLL.items()}

    syll_gap, seq_gap = 0.040, 1.000
    out = [silence(0.3, sr)]

    for word in order:
        for _ in range(n_reps_per_block):
            for ch in word:
                out += [cache[ch], silence(syll_gap, sr)]
            # Subtract the trailing syll_gap to ensure exactly 1.0s between sequences
            out.append(silence(seq_gap - syll_gap, sr))

    return normalize(concat(*out))


# =====================================================================
#  Stochastic Figure-Ground (SFG) - Full Session
# =====================================================================
def _sfg_cloud_onsets(rng: np.random.Generator, dur_ms: int = 5000) -> np.ndarray:
    """Generate independent cloud onsets (ms) for a single channel."""
    onsets = []
    # 8 onsets per 2-second window
    for w0 in range(0, dur_ms, 2000):
        picks = np.sort(rng.choice(np.arange(0, 2000, 50), size=8, replace=False))
        jit = rng.uniform(-25, 25, size=8)
        for cand in (w0 + picks + jit):
            if 0 <= cand and cand + 50 <= dur_ms:
                onsets.append(cand)
    return np.array(onsets)


def _sfg_figure_onsets(rng: np.random.Generator) -> np.ndarray:
    """Generate the shared synchronous chord onsets for the figure channels."""
    onsets = {0.0, 4950.0}  # Forced first and last
    target_n = 20
    grid = np.arange(50, 4950, 50)
    rng.shuffle(grid)
    for cand in grid:
        if len(onsets) >= target_n:
            break
        if all(abs(cand - o) >= 50 for o in onsets):
            onsets.add(cand + rng.uniform(-25, 25))

    valid = [o for o in onsets if 0 <= o and o + 50 <= 5000]
    return np.array(sorted(valid))


def make_sfg(sr: int, n_reps: int = 3, n_fig: int = 8,
             base_seed: int = 0, fig_seed: int = 12345) -> np.ndarray:
    """Stochastic figure-ground (Session Audio): Multiple 16-second presentations.
    Each presentation = 0.5s silence + 5s pre + 5s figure + 5s post + 0.5s silence.
    The figure channels and their chord onsets are FROZEN across reps.
    The background cloud is drawn FRESH every rep.
    """
    # 1. Setup global figure structure (FROZEN across all reps)
    rng_fig = np.random.default_rng(fig_seed)
    shared_fig_onsets = _sfg_figure_onsets(rng_fig)

    # 37 log-spaced frequencies from 800 to 6400 Hz
    freqs = 1600.0 * 2.0 ** (np.arange(-12, 25) / 12.0)
    N = len(freqs)

    # Target figure channels (exact matches from stimulus.py)
    fig_channels_10 = [16, 35, 23, 9, 27, 2, 21, 4, 32, 13]
    fig_idx = set(fig_channels_10[:n_fig])

    # 2. Audio buffer setup
    presentation_dur_s = 16.0
    total_s = n_reps * presentation_dur_s
    out = silence(total_s + 0.1, sr)
    pip_dur = 0.050

    def add_pip(freq: float, t0_s: float, amp: float):
        i0 = int(round(t0_s * sr))
        seg = tone(freq, pip_dur, sr, amp=amp)
        out[i0:i0 + seg.size] += seg[:max(0, out.size - i0)]

    # 3. Build session loop
    for r in range(n_reps):
        # Fresh RNGs for this presentation's background
        rng_pre = np.random.default_rng(base_seed + 1000 * (r + 1))
        rng_figc = np.random.default_rng(base_seed + 1000 * (r + 1) + 1)
        rng_post = np.random.default_rng(base_seed + 1000 * (r + 1) + 2)

        offset_s = r * presentation_dur_s

        # --- PRE EPOCH (0.5s to 5.5s) ---
        pre_start = offset_s + 0.5
        for ch in range(N):
            for on_ms in _sfg_cloud_onsets(rng_pre, 5000):
                add_pip(freqs[ch], pre_start + on_ms / 1000.0, amp=0.18)

        # --- FIGURE EPOCH (5.5s to 10.5s) ---
        fig_start = offset_s + 5.5
        for ch in range(N):
            if ch in fig_idx:
                # Rate-matched: Figure channels ONLY play the synchronous chord
                for on_ms in shared_fig_onsets:
                    add_pip(freqs[ch], fig_start + on_ms / 1000.0, amp=0.22)
            else:
                # Ground channels play fresh independent cloud
                for on_ms in _sfg_cloud_onsets(rng_figc, 5000):
                    add_pip(freqs[ch], fig_start + on_ms / 1000.0, amp=0.18)

        # --- POST EPOCH (10.5s to 15.5s) ---
        post_start = offset_s + 10.5
        for ch in range(N):
            for on_ms in _sfg_cloud_onsets(rng_post, 5000):
                add_pip(freqs[ch], post_start + on_ms / 1000.0, amp=0.18)

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
    ap.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                    help="output audio format (default mp3)")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        try:
            x = CLIPS[name](args.sr)
        except Exception as e:                            # e.g. librosa/wavs missing
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue
        path = write_audio(out / f"{name}.{args.format}", x, args.sr)
        print(f"  wrote {path.name:22s} {x.size / args.sr:6.1f} s")
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
