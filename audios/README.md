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


## 4. Complex-tone versions — `cx_*.mp3`

```bash
python -m audios.complex_set
python -m audios.complex_set --gap-ms 20    # separate the two tones
python -m audios.complex_set --cloud-db -6
```

Built after the pure-tone set turned out to be **unobservable**. Two separate
reasons for that, and only one of them is about timbre.

### Complexes instead of pure tones

A pure tone carries frequency and little else. A harmonic complex carries a
**pitch** — represented redundantly across every harmonic, robust to masking
of any one of them — and harmonicity is itself a grouping cue, so a token
arrives as an object rather than as a point on a frequency axis.

Every complex takes its harmonics up to a **fixed 4 kHz ceiling** with 1/h
amplitudes, so all of them occupy the same band and differ in periodicity, not
brightness. A fixed harmonic *count* would make the higher-F0 tone audibly
brighter and brightness would do the work instead of pitch. Each is
RMS-normalised, so a complex with 20 harmonics is no louder than one with 4.

F0 base is 400 Hz, not 1000: a 40 ms tone at 400 Hz holds 16 periods, enough
for a solid pitch, where at 200 Hz it would hold 8 and the pitch of so short a
tone starts to weaken.

| | F0 | harmonics |
|---|---|---|
| A | 400.00 Hz | 10 |
| B (df 1) | 423.79 Hz | 9 |
| B (df 9) | 672.72 Hz | 5 |
| C, D | 448.98, 475.68 Hz | 8, 8 |

`cx_tones_demo.mp3` plays A six times, then B six times, then the sequence, so
the tones can be heard before they are buried in anything.

### The spacing is the other reason, and probably the bigger one

The specified sequence puts A and B hard against each other and then 120 ms of
silence: **0 ms between the two tones of a doublet, 160 ms between doublets**.
Temporal proximity is among the strongest grouping cues there is, so that
spacing *binds* A to B and separates one doublet from the next. It is a
stimulus built to be heard as one stream of two-note events, and no frequency
separation will readily split it, because splitting it means separating the
two tones that are **closest together in time**.

Stream segregation is normally shown with tones spaced **evenly** — A and B
each every 200 ms, interleaved, so every tone is 100 ms from each neighbour
and proximity favours neither grouping. `cx_abab_df01` and `cx_abab_df09` are
that: same 5 Hz per tone type, same tones, only the spacing changed. At 1
semitone it should gallop as one stream; at 9 it should split into two. That
is the contrast the doublet version cannot easily show.

`--gap-ms 20` is the intermediate step if you want to keep the doublet
structure but stop the two tones fusing into one 80 ms event.


## 5. A coherent syllable against copies of itself — `syl_*.mp3`

```bash
python -m audios.syllable
python -m audios.syllable --shift-ms 10 40 --onset-jitter-ms 100
```

Twenty channels, in two sets of ten. No cloud yet — these are meant to sit
inside a larger set later (ten of seventy in the sketch), and the cloud
construction from `audios.cloud` drops straight in when they do.

**A — the figure.** Ten partials, harmonics 1–10 of 400 Hz (400 Hz–4 kHz).
Every one starts at exactly the same instant, every repetition: **onset spread
measured at 0.0 ms**. Simultaneous onset plus harmonicity is the strongest
fusion cue there is, so the ten arrive as one object rather than ten tones.

**S — ten shifted copies**, one per partial, drawn on two different schedules:

| | drawn | when |
|---|---|---|
| frequency | 0.5–1 semitone, random sign | **once per channel, then fixed** |
| time | **10–40 ms lag, always positive** | **afresh every repetition** |

Measured: offsets 0.51–0.97 st, lag 10–40 ms, spread within a token 24.3 ms.

That split is the whole design. If the time shifts were fixed too, the ten
copies would themselves be perfectly coherent — a second syllable offset from
the first — and there would be two objects rather than one object and a mess.
Redrawing them per repetition leaves A as the only thing with a stable
temporal signature.

The lag is a **lag**, never a lead: a copy always follows its partner. And it
is shorter than the 40 ms tone over almost all of its range, so **a copy
overlaps its partner** rather than being a separate event that proximity could
group. At the top of the range, 40 ms, the two abut exactly.

### The four versions

| file | A's partials | figure onset |
|---|---|---|
| `syl_coherent` | simultaneous (0.0 ms) | regular, 400.0 ± 0.0 ms |
| `syl_coherent_jit` | simultaneous (0.0 ms) | **jittered, 403.2 ± 86.4 ms** |
| `syl_scrambled` | jittered (23.8 ms) | regular |
| `syl_scrambled_jit` | jittered (24.6 ms) | jittered |

`syl_coherent_jit` is the interesting one: the whole figure is displaced by up
to ±100 ms per repetition, so the token stops being periodic, while everything
*inside* it keeps its relative timing — A stays exactly as coherent as before
and only its arrival becomes unpredictable. It separates "the object is held
together by simultaneous onset" from "the object is found because it arrives
on the beat". The jitter is grid-tied, so the mean rate is unchanged (2.480 vs
2.500 Hz, the difference being the endpoint draws).

The scrambled files jitter A's own partials the same way, so nothing is
coherent while the harmonic series and the spectrum stay identical. That is
what says coherence rather than spectrum is doing the work.

### Level

Every channel at equal amplitude, the figure-ground convention, rather than a
1/h roll-off. With a roll-off the tenth partial would be 20 dB below the first
and the top half of the figure would contribute almost nothing to whether it
fuses; here each of the twenty channels carries the same weight.

`syl_check.png` shows three of the four rasters, the per-channel frequency
offsets, and the lag distribution against the tone length.
