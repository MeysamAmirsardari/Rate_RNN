# audios — stimuli to listen to

Playable versions of the paradigms, built to psychoacoustic spec rather than
sonified loosely. Masters are 24-bit 48 kHz WAV; the MP3s are 320 kbps CBR
from those masters.

```bash
python -m audios.two_tone                 # the two files below
python -m audios.two_tone --df 1 3 6 9    # any set of separations
python -m audios.two_tone --keep-wav      # keep the WAV masters
```

## `ab_df01.mp3` / `ab_df09.mp3` — repeating AB doublets

Two files differing in one thing: the separation between A and B.

    A  0-40 ms      B  40-80 ms      silence 80-200 ms      repeat

| | |
|---|---|
| sample rate | 48 000 Hz (every duration an exact sample count) |
| tone duration | 40 ms, of which 30 ms steady |
| ramps | 5 ms raised-cosine (cos²) on and off |
| gap A→B | 0 ms — B begins the sample after A ends |
| doublet rate | 5 Hz (200 ms onset-to-onset) |
| length | 60 doublets, 13.2 s |
| A | 1000 Hz, **identical in both files** |
| B | 1059.46 Hz (1 st) / 1681.79 Hz (9 st) |
| level | equal amplitude per tone; each file peak-normalised to −3 dBFS |
| presentation | diotic (same signal both ears) |

Measured back off the rendered signal, both files: 60 doublets, each
80.0 ± 0.000 ms, one every 200.0 ± 0.000 ms; peak −3.00 dBFS, RMS −6.65 dBFS.
Zero jitter — nothing is rounded.

### Reading of the spec

"5 Hz, 40 ms tones, 0 ms gap" only fits together one way: the **doublet**
repeats at 5 Hz and the 0 ms gap is the one *inside* it. A 5 Hz tone rate
would put 160 ms between tones, contradicting the 0 ms gap; abutting 40 ms
tones with no silence at all would be a 25 Hz tone rate.

### Things worth knowing before you listen

**The doublet is one 80 ms acoustic event, not two.** With a 0 ms gap the only
thing between A and B is where A's 5 ms fall meets B's 5 ms rise. Measured,
that notch reaches −25 dB and spends 4 ms below half amplitude. Each tone is
still gated independently — cross-fading to hold amplitude constant would
remove the very cue the paradigm is about — but do not expect silence between
them.

**At 1 semitone the two tones are barely resolvable, by design.** 59 Hz of
separation on a 40 ms tone is close to the time-frequency limit; the
spectrogram in `two_tone_check.png` cannot cleanly separate them either. That
is the point of the contrast — at 9 semitones (682 Hz) they split apart
immediately.

**Equal SPL is not equal loudness.** Both files are identical in peak and RMS
and differ only in frequency. The ear is roughly 1–2 phon more sensitive at
1682 Hz than at 1000 Hz, so in the wide file B may sound slightly louder than
A. Correcting it would need an equal-loudness contour, which is not the
convention here and would introduce a level difference between the files.

**Use the WAV for real experiments.** MP3 is lossy and adds slight pre-echo at
sharp onsets — inaudible at 320 kbps with 5 ms ramps, but the master is the
master.

`two_tone_check.png` shows the first 600 ms of each: waveform with A and B
shaded, and a spectrogram.
