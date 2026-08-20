# audios — stimuli to listen to

Playable versions of the paradigms, built to psychoacoustic spec rather than
sonified loosely. Masters are 24-bit 48 kHz WAV; the MP3s are 320 kbps CBR.

```bash
python -m audios.two_tone      # one stream, two frequency separations
python -m audios.two_stream    # two streams held apart by timing alone
```

## Shared conventions

| | |
|---|---|
| sample rate | 48 000 Hz — every duration an exact sample count |
| tone | 40 ms, gated with 5 ms raised-cosine (cos²) on and off, 30 ms steady |
| phase | every tone starts at sine phase zero |
| gap A→B | 0 ms — the second tone begins the sample after the first ends |
| presentation | diotic (same signal both ears) |
| level | **one scale factor for the whole set**, so a tone is at the same SPL in every file |

Level deserves a note. Peak-normalising each file separately would tie its
level to whatever its loudest moment happened to be — in the two-stream file
that is the rare instant when both streams sum, which would drag every tone
down 6 dB relative to the single-stream files and make the set useless for
comparing them. Equal SPL per tone is the convention, so a lone tone peaks at
−9 dBFS everywhere and two coincident tones at −3 dBFS. The two-stream file is
legitimately louder overall (twice the tone density) at −14.1 dBFS against
−17.1, while a tone in it is at the same level: −12.6 vs −12.7 dBFS.

## 1. One stream — `ab_df01.mp3`, `ab_df09.mp3`

    A  0–40 ms      B  40–80 ms      silence      then the next doublet

A is **1000 Hz in both files**; only B moves, so the separation is the only
difference. B = 1059.46 Hz (1 st) or 1681.79 Hz (9 st).

**Onsets are jittered.** Each doublet is displaced uniformly by up to ±50 ms
from its grid position. Measured: inter-onset 199.0 ± 42.7 ms, range
118–285 ms, 60 doublets. The jitter is tied to the grid rather than drawn as
intervals — drawn intervals random-walk and the rate drifts, whereas
grid-tied jitter keeps the **overall rate exactly 5 Hz** however large it is.

±50 ms is a quarter of the period, which is as far as it can go before
doublets collide: the shortest possible interval, 200 − 2×50 = 100 ms, still
clears the 80 ms doublet. Measured minimum silence between doublets: 36 ms.

Worth stating: isochrony is itself a grouping cue, so removing it weakens
streaming build-up slightly. What it buys is that nothing can be following the
rhythm instead of the frequencies.

## 2. Two streams — `ab_cd_incoherent.mp3`

Four tones in a chain, each step one semitone:

    A 1000.00  →  B 1059.46  →  C 1122.46  →  D 1189.21 Hz

AB is close to itself, CD is close to itself, and B→C is the same 1 semitone.
The four span **three semitones in total** — far too little for frequency to
segregate them. That is deliberate: it removes the usual cue, so anything
heard as two streams has to come from timing.

**The streams are anti-phase.** Both run at 5 Hz and CD's grid sits half a
period — 100 ms — from AB's. For two streams at the same rate, half a period
is the furthest their onsets can be, so this is maximal incoherence rather
than merely some. Each stream is then jittered independently, so the relative
phase wanders too.

Measured: AB 5.012 Hz, CD 5.007 Hz, onset asynchrony **85.1 ± 14.3 ms with a
minimum of 57.7 ms**, and the two streams sound together only 2% of the time.

### Why the jitter is smaller here (±25 ms, against ±50)

Because it has a job that unbounded jitter would undo. With a 100 ms offset
and independent jitter of ±J, the asynchrony runs over 100 ± 2J. At J = 50
that reaches **zero** — the streams would occasionally start together, and
simultaneous onsets fuse, which is the one thing this file must never do. At
J = 25 the asynchrony is bounded to 50–150 ms.

The measured minimum, 57.7 ms, is comfortably past the point where onset
asynchrony works as a segregation cue — beyond roughly 30 ms is enough to stop
two components fusing into one event.

### What to listen for

Whether it splits into two interleaved streams or stays one dense warble, and
whether that flips back and forth, which is what bistable streaming does. The
control is `ab_df01` — the same 1-semitone doublet alone, with nothing to
segregate from. `--phase-ms 0` builds the coherent version, which should fuse.

## 3. The same three, in an unpredictable cloud

    ab_df01_cloud.mp3    ab_df09_cloud.mp3    ab_cd_incoherent_cloud.mp3

```bash
python -m audios.with_cloud                # all three
python -m audios.with_cloud --cloud-db -6  # a gentler background
```

The targets are untouched — same frequencies, same jittered onsets, same
gating — so each cloud file pairs exactly with its clean counterpart and the
cloud is the only difference.

### Uniform in time

The cloud has to be equally dense at every instant, or its envelope becomes a
cue and a target could be found by where the background thins out. Five
voices staggered by a fifth of the slot, each tone lasting four fifths of it:

    slot 50 ms, voices at 0 10 20 30 40 ms, tone 40 ms  →  4 sounding, always

Measured on every file: concurrency **4–4**, no fluctuation at all.

### Uniform in frequency

Every channel must be used equally often, or the rare ones become salient and
the common ones a drone. Channels are dealt from a pack reshuffled only when
exhausted, so counts are level to within one pass. Measured: **32 channels
(30 for the two-stream file), 500–3364 Hz on a semitone grid, per-channel use
spread of 1** (40–41 uses, or 44–45).

### Unpredictable

The pack is reshuffled every pass, so no channel reliably follows any other —
there are no recurring pairs anywhere in the cloud. The targets are the only
thing in the file with a repeating temporal structure, which is the point.

### What the cloud is not allowed to do

**Never sound a target frequency.** The target semitones are removed from the
grid, so a target tone is never in doubt as to whether it belongs to the
figure. If the cloud could sound the same frequency, "is that the target"
would become a question about frequency instead of about timing.

**Never repeat a channel in adjacent slots.** The voices are a fifth of a slot
apart, so a channel used twice nearby would run into itself and sound as one
long tone — a duration cue the cloud is not supposed to have.

### Level

Cloud tones sit at the **same level as target tones** by default, the
figure-ground convention, which is what keeps the task about temporal
structure rather than loudness. At 0 dB the target is one of five concurrent
tones and is genuinely hard to hold on to; `--cloud-db -6` gives a gentler
version.

The three cloud files share one scale, as the three clean files share another.
Within each set a tone is at the same SPL. The two sets **cannot** share one:
a cloud file has six tones sounding where a clean file has one, so matching
per-tone level across both would either clip the cloud files or leave the
clean ones needlessly quiet. A tone is at −17.7 dBFS in the cloud set and
−9.0 dBFS in the clean set — on the record rather than implied.

`with_cloud_check.png` draws every tone as a dash, cloud dark and targets
coloured. It is drawn from the placements, not from the mixture, because no
spectrogram can separate tones a semitone apart at 40 ms — true of the sound,
and exactly why the picture has to come from the score.

## Checks

The timing in both READMEs is measured back off the rendered signal, not taken
from the code that wrote it. `two_tone_check.png` and `two_stream_check.png`
show waveforms and spectrograms.

Two limits the checks make visible rather than hide:

- **A doublet is one 80 ms acoustic event, not two.** With a 0 ms gap the only
  thing between the tones is where one ramp meets the next — a notch reaching
  −25 dB, 4 ms below half amplitude. Each tone is gated independently on
  purpose; cross-fading to hold amplitude constant would delete the cue the
  paradigm is about.
- **A semitone at 40 ms is near the time-frequency limit.** 60 Hz of
  separation on a 40 ms tone cannot be cleanly resolved by any spectrogram,
  and that is the point of the 1-vs-9 contrast, not a defect of the plot.
