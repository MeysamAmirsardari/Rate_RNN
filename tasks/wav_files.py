#!/usr/bin/env python3
"""
Generate normalized English-syllable .wav stimuli for an auditory neuroscience task.

Per syllable pipeline:
  1. Synthesize speech with a FEMALE neural TTS voice (Microsoft Edge TTS - free, no API key).
  2. Decode -> mono, resample to a fixed sample rate (same SR for every file).
  3. Trim leading/trailing silence so the duration reflects the syllable, not padding.
  4. Set duration to EXACTLY TARGET_MS:
        - "stretch": pitch-preserving time-stretch (keeps the whole syllable)   [default]
        - "trim_pad": hard trim/zero-pad (truncates content if syllable is longer)
  5. Apply onset/offset raised-cosine ramps (avoids spectral splatter at edges).
  6. RMS-normalize all stimuli to a common level, then apply ONE shared anti-clip
     scale factor -> equal loudness (equal RMS) preserved across all files, no clipping.

Install:
    pip install edge-tts librosa soundfile numpy

Scientific notes:
  * 180 ms is short for a natural syllable. Inspect every output and report the
    duration method and (for "stretch") the per-file stretch factor in your methods.
  * Edge TTS voices can change over time; archive the generated .wav files, since
    they are your actual stimuli (the wavs, not the script, are what you report/share).
"""

import asyncio
import os

import numpy as np
import soundfile as sf
import librosa
import edge_tts

# ------------------------------- Configuration -------------------------------
SYLLABLES = ["pee", "tah", "coo", "bay", "doe", "guy", "pow", "toy"] # 8 syllables
VOICE           = "en-US-JennyNeural"  # female. Alts: en-US-AriaNeural, en-GB-SoniaNeural
OUT_DIR         = "syllables_wav"
SAMPLE_RATE     = 44100      # Hz, identical for every file
TARGET_MS       = 180.0      # required duration (milliseconds)
RAMP_MS         = 10.0       # onset/offset cosine ramp length (ms)
TARGET_RMS      = 0.10       # target RMS amplitude (linear; ~ -20 dBFS)
DURATION_METHOD = "stretch"  # "stretch" (recommended) or "trim_pad"
SILENCE_TOP_DB  = 30         # silence-trim threshold (dB below peak)
PEAK_CEILING    = 0.99       # max allowed |sample| after normalization
# -----------------------------------------------------------------------------


async def synthesize(text, path):
    """Render one syllable to an mp3 file with a female Edge TTS voice."""
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(path)


async def synth_all(items):
    await asyncio.gather(*(synthesize(s, p) for s, p in items))


def set_duration(y, sr, target_ms, method):
    """Force the signal to exactly target_ms samples."""
    target_len = int(round(sr * target_ms / 1000.0))
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if method == "stretch":
        rate = len(y) / target_len            # rate > 1 -> faster -> shorter
        y = librosa.effects.time_stretch(y, rate=rate)
    elif method != "trim_pad":
        raise ValueError("DURATION_METHOD must be 'stretch' or 'trim_pad'")

    # Make length exact (handles stretch rounding and trim_pad case).
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y, (len(y) and (rate if method == "stretch" else 1.0)) or 1.0


def apply_ramps(y, sr, ramp_ms):
    """Apply raised-cosine onset/offset ramps."""
    n = int(round(sr * ramp_ms / 1000.0))
    if n == 0 or 2 * n >= len(y):
        return y
    t = np.linspace(0.0, np.pi / 2.0, n)
    onset = np.sin(t) ** 2          # smooth 0 -> 1
    offset = onset[::-1]            # smooth 1 -> 0
    y = y.copy()
    y[:n] *= onset
    y[-n:] *= offset
    return y


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Synthesize
    items = [(s, os.path.join(tmp_dir, f"{s}.mp3")) for s in SYLLABLES]
    asyncio.run(synth_all(items))

    # 2-5. Decode, trim, set duration, ramp
    processed, stretch_factors = [], {}
    for s, p in items:
        y, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)   # decode + resample + mono
        y, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)  # remove TTS silence
        y, factor = set_duration(y, SAMPLE_RATE, TARGET_MS, DURATION_METHOD)
        y = apply_ramps(y, SAMPLE_RATE, RAMP_MS)
        processed.append((s, y.astype(np.float32)))
        stretch_factors[s] = factor

    # 6. RMS-normalize to a common level, then one shared anti-clip factor
    scaled = [(s, y * (TARGET_RMS / rms(y))) for s, y in processed]
    global_peak = max(float(np.max(np.abs(y))) for _, y in scaled)
    safety = (PEAK_CEILING / global_peak) if global_peak > PEAK_CEILING else 1.0

    print(f"Voice={VOICE}  SR={SAMPLE_RATE} Hz  dur={TARGET_MS} ms  "
          f"method={DURATION_METHOD}  global_safety_scale={safety:.4f}")
    print("-" * 72)
    for s, y in scaled:
        out = (y * safety).astype(np.float32)
        path = os.path.join(OUT_DIR, f"{s}.wav")
        sf.write(path, out, SAMPLE_RATE, subtype="PCM_16")  # change to PCM_24 if needed
        print(f"{s:>3}.wav  dur={len(out)/SAMPLE_RATE*1000:6.1f} ms  "
              f"rms={rms(out):.4f}  peak={np.max(np.abs(out)):.3f}  "
              f"stretch x{stretch_factors[s]:.2f}")

    # Clean up temporary mp3s
    for _, p in items:
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(tmp_dir):
        os.rmdir(tmp_dir)

    print("-" * 72)
    print(f"Done. {len(SYLLABLES)} files written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()