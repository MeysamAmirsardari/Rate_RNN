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

### The cloud

Every version also comes as `*_cloud.mp3` — same figure, plus a background.

**The figure is a burst, and a grid cloud cannot hide it.** Twenty tones
inside an eighty-millisecond window, then silence for 80% of the time. Against
a flat two-tone background the total swung **2 to 22**, mean 4.05 ± 4.68 — the
figure was findable from the envelope alone, with no grouping involved. That
is a real defect and it is what this cloud fixes.

**The cloud is scheduled against the figure**, not on a grid of its own:
denser between the tokens, thinner inside them, so the total never moves.
Measured, total concurrency **10–20, mean 19.72 ± 0.92**, below 18 for only
4.1% of the time — a five-fold reduction in envelope modulation.

The residual dips are unavoidable and worth naming. A tone already sounding
cannot be withdrawn when the burst arrives, so the scheduler refuses to start
one that would overshoot, and the period just before each burst is left
slightly under-filled. Finer scheduling helps and then plateaus (SD 1.24 at a
10 ms grid, 0.92 at 2.5 ms, 0.89 at 1 ms); it is intrinsic to fixed-length
tones meeting a step-function target.

**Fifty channels**, 100–5702 Hz on a semitone grid running from two octaves
below the fundamental to just above the tenth partial. With the twenty figure
channels that is **seventy in total** — the seventy in the sketch.

**Uniform in time and uniform in frequency cannot both hold here, and the
figure is why.** Hiding a twenty-tone burst requires a twenty-tone background,
which needs 108–121 tones per cloud channel against 30 per figure channel.
Equalising that would take **189 cloud channels**, not 50 — `--grid-st 0.37`
gets close, at the price of cloud channels spaced a third of a semitone apart.
At seventy channels the flat envelope is the one worth having, since a 10×
envelope burst is a far stronger cue than a 4× difference in how often a
channel is used.

**Nothing within a semitone of a figure channel.** The guard is a real design
parameter, not a rounding tolerance: too small and the cloud sits on top of
the figure and masks it, too large and the cloud is spectrally elsewhere so
segregating the figure stops being a task.

**Unpredictable**: the pack reshuffles every pass, so no cloud channel
reliably follows any other. The figure is the only thing with a stable
structure.

### Level

Every channel at equal amplitude, the figure-ground convention, rather than a
1/h roll-off. With a roll-off the tenth partial would be 20 dB below the first
and the top half of the figure would contribute almost nothing to whether it
fuses; here each of the twenty channels carries the same weight. `--cloud-db`
moves the background if a gentler version is wanted.

`syl_check.png` shows three of the versions, the same coherent figure with the
cloud drawn in grey, the per-channel frequency offsets, and the concurrency
profile — figure in red, cloud in grey, filling to a flat ceiling.

### The figure is frozen

Every channel's position in the token -- its frequency offset **and its lag**
-- is drawn once, in `layout`, and every repetition uses that same set.  The
token is one fixed pattern in time and frequency, repeated thirty times on a
regular 2.5 Hz beat.  That is what makes it a figure and what makes it
learnable: a receptive field averaged over repetitions converges on the
pattern rather than on the marginal spectrum.

`--redraw` is the control.  It draws fresh lags every repetition, so nothing
ever repeats and there is no pattern to converge on, while the marginal
spectrum, the channel count and the tone count are all unchanged.

### The jitter series (`--series`)

    syl_frozen_j0-0.mp3     fully coherent, every channel on the onset
    syl_frozen_j0-10.mp3
    syl_frozen_j0-20.mp3
    syl_frozen_j0-40.mp3
    syl_frozen_j0-80.mp3    each also as *_cloud.mp3
    syl_frozen_check.png

One pattern at five widths.  The per-channel positions are stored as fractions
of the range, so the five files are the **same shape stretched**, not five
independent draws -- the only thing that differs across the series is how far
apart the channels are pulled.  The sounding subset is rescaled to span the
range exactly, so a file called 0-40 really does run 0 to 40 ms.

`--jitter-ranges LO,HI ...` sets the widths.

All five are normalised by one common gain rather than individually, because
coherence changes the peak: the 0-0 file genuinely is the loudest-peaked of
the set, and levelling that away would remove part of what the manipulation
does.  Overall RMS is constant at -23.4 dBFS across the series.

At five channels the cloud hides the figure almost exactly: total concurrency
is 5.00 +- 0.00 for the coherent file and 4.89 +- 0.33 for the widest, with
28-29 cloud tones per channel against 30 per figure channel.

### Copies only (`--drop-a`)

    syl_copies.mp3          the ten shifted copies, token onset regular
    syl_copies_jit.mp3      the same, whole-token onset jittered +-100 ms
    syl_copies_*_cloud.mp3  each inside its own scheduled cloud
    syl_copies_check.png

The coherent set removed; only its ten near-copies sound.  Each keeps a fixed
frequency offset and is redrawn in time on every repetition, so nothing in the
stimulus has a stable temporal signature -- the token is a smear that never
repeats itself.

The coherent/scrambled pair does **not** survive this: those two differ only in
whether A's partials are jittered, and A is gone.  What is left to vary is
whether the smear arrives on the beat, hence the two files.

Ten channels instead of twenty, so the cloud's ceiling drops from twenty to
ten.  The cloud still avoids A's frequencies even though they no longer sound,
which keeps the available spectrum identical to the twenty-channel version at
the cost of a total of sixty channels rather than seventy.

`--n-sound k` thins the figure to k of the partials, chosen evenly across the
harmonic series so the 400 Hz to 4 kHz span is preserved -- a truncated series
of the same size would sit entirely below 2 kHz and change the spectrum as well
as the channel count.  At `--n-sound 5` the frequency imbalance essentially
goes away: 26-28 tones per cloud channel against 30 per figure channel, where
ten figure channels needed 53-57.  A smaller figure needs a smaller cloud to
hide it, and fifty channels is then close to enough.

One 10.8 ms dip in 11.7 s, at the first token: a 40 ms cloud tone cannot fit a
10 ms gap without crossing the ceiling.  Intrinsic, and the same limit as the
shallow pre-burst dips above.

## Words, and when a staircase stops being one thing (`word.py`)

    python -m audios.word --sweep tone
        word_tone{0,10,20,40,80,160,500,700,850,1000}ms.mp3
    python -m audios.word --sweep syllable
        word_syl{80,130,200,300,450,500,700,850,1000}ms.mp3
    python -m audios.word --sweep tone --order perm      the trajectory control
    python -m audios.word --sweep tone --redraw          the floor control

A **tone** is 40 ms.  A **syllable** is five tones on a rising frequency ramp,
each delayed a fixed step behind the one below.  A **word** is three
syllables, the same staircase transposed up in frequency, each delayed a fixed
step behind the last.  The word repeats on a fixed period, and every part of
the pattern is frozen -- verified per condition by counting distinct word
shapes across repetitions, which is 1 for every frozen file and equals the
repetition count for every `--redraw` file.

The same manipulation exists at both levels, and one step is the only thing
that ever moves.

### The regimes

    step < 40 ms    successive units OVERLAP; simultaneous grouping available
    step = 40 ms    they exactly abut; overlap ends here
    step > 40 ms    a gap opens; nothing is simultaneous, so anything that
                    still pops out is doing so sequentially

The third regime is the one worth listening to.  A frozen repeating pattern
can in principle be bound by its recurrence rather than by simultaneity, and
nothing says that has to fail at 40 ms.  If pop-out survives well past the
overlap limit, the figure is not a smeared chord -- it is an object defined by
its repeating spectrotemporal shape.

### The word onset is jittered

Each whole word is displaced independently, uniform on `+-jitter`, on a tied
grid so the mean rate is exact however large the jitter.  Without it the words
arrive isochronously and the figure can be found by rhythm alone -- a
different cue from the pattern, and the easier one.  The internal shape is
untouched: the displacement moves the token, not its contents, which the
distinct-shape count confirms stays at 1.

The jitter is chosen automatically as the largest that still keeps consecutive
words from touching, since the shortest interval a jittered grid can produce
is `period - 2*jitter` and a word occupies its span plus a tone:

    tone sweep       every 1000 ms +-140 ms   (measured 774-1225 ms)
    syllable sweep   every 1600 ms +-270 ms   (measured 1165-2034 ms)

One value for the whole sweep, never per condition -- letting it shrink as the
step grows would confound the jitter with the manipulation.  The periods were
raised from 800 and 1200 ms to make room for a jitter worth having.  Words
never overlapping is checked on the built streams, not assumed.
`--word-jitter-ms 0` restores the isochronous version.

### Past the repetition period

The **syllable** sweep runs to a 1 s step, which is longer than the word
repeats.  The tone step is not swept that far and should not be: it is the
syllable's own frozen shape, fixed at a 20 ms step so the five tones span
0-80 ms, and moving it changes what a syllable *is* rather than how syllables
are strung together.  The
period is deliberately **not** stretched to keep the instances apart.
Stretching it would drop the figure's tone rate in step with the manipulation
-- five tones per second at a 40 ms step against one per second at 1000 ms --
so the figure would grow sparser exactly as it grew slower and the two could
never be told apart.  Holding the period fixed keeps the figure's density, its
repetition rate and the cloud's density all constant, leaving the step as the
only thing that moves.

Consecutive instances therefore interleave, which is what the classic
figure-ground picture shows anyway.  Each instance is still one frozen shape;
several are simply in flight at once, and the count is reported per condition:

    syllable step   80-500   700   850   1000 ms
    in flight          1       2     2      2

### What is held constant

**Per-tone level.**  One common gain across a sweep, never per file.  Total
energy is then identical across conditions (overall RMS constant to 0.3 dB)
and each figure tone has the same level everywhere.  Peak level is *not*
equalised: at step 0 the five tones sum coherently and the file genuinely
peaks 13 dB higher, which is a consequence of coincidence, not an artifact.

**Background density.**  The cloud ceiling is one number for the whole sweep
*and across both sweeps* (default 6).  The tone sweep's figure peaks at five
simultaneous tones and the syllable sweep's at three, so letting each take its
own peak would make the two levels incomparable and would let density co-vary
with the step.  Measured concurrency is 6.00 +- 0.04 to 0.07 throughout.

**The cloud sounds the figure's own channels.**  Reserving channels for the
figure leaks the answer: every tone in a reserved channel is a figure tone, so
the figure can be picked out by frequency alone and the timing -- the thing
being measured -- never has to be used.  Sharing the pool makes frequency
uninformative.  Two things follow and both are enforced: a cloud tone never
lands on a figure tone in the same channel, and each channel's figure tones
count toward its total, with the dealer working down the **total** so figure
channels are neither rarer nor commoner than any other.  Measured: 38-39 tones per figure channel
against 38-39 for the rest.  `--exclusive` restores the reserved version for
comparison.

The count that governs the dealer has to be the count **so far**.  Charging a
figure channel upfront for figure tones it has not played yet made those
channels look fully used at time zero, so they were starved early and
over-supplied late: level in total, but fluctuating three times as much as a
plain cloud channel from second to second -- uniform where it was measured and
lumpy where it is heard.  Counting each figure tone as it arrives holds
per-quarter use at 8-11 tones per channel against 3-13 before, with figure and
cloud channels finally indistinguishable, SD 0.51 against 0.53.

    channel use, figure and cloud together, 61 channels
      all      38-39, mean 38.4 +- 0.5
      figure   38-39, mean 38.1   (15 channels)
      other    38-39, mean 38.4   (46 channels)
      per quarter of the stream, per channel: 8-11

**The word onset is quantised to the cloud's tone length**, and this is why
the background can stay thin.  The cloud tiles in 40 ms blocks; a word landing
mid-block leaves the part of that block before it able to hold only one tone,
because a 40 ms cloud tone starting there would run into the figure and break
the ceiling.  The hole is `onset mod 40 ms` wide -- 20 ms on average, up to 40
-- and it sits immediately before the figure, which is the worst possible
place, and it is *deepest in the coherent condition*, so it would flatter the
reference and exaggerate the fall-off.

Landing every word on a block boundary removes it exactly:

    tone step        0    10    20    40    80   160 ms
    concurrency    6-6   3-6   5-6   6-6   6-6   6-6
    deepest dip    0.0   3.0   0.8   0.0   0.0   0.0 dB

Seven distinct jitter values at +-120 ms still destroys isochrony, and the
mean rate stays exact because the draw is symmetric.  The earlier fix for this
-- doubling the cloud to twelve concurrent tones so the same shortfall was
shallower in dB -- worked on the number and ruined the stimulus; alignment
costs nothing.

**A caveat that survives.**  The residual belongs to 10 and 20 ms, the two
steps that are not multiples of the cloud's 40 ms block, so their figure tones
sit off the cloud's onset grid while 0, 40, 80 and 160 sit on it.  Being
on-grid or off-grid therefore co-varies weakly with the step.  It is small --
3.0 dB for 0.2% of the stream at step 10, nothing anywhere else -- but for a
comparison with no such asymmetry at all, use `--steps 0 40 80 120 160`.  The
syllable sweep does not have the problem: every condition there measures
0.8 dB for 0.0% of the time, identically.

**Frequencies are inharmonic** -- even semitone spacings, harmonics of
nothing.  A harmonic series would confound the result completely: harmonicity
fuses a complex on its own, so a fall-off with step size could be the loss of
harmonic fusion rather than the loss of pattern grouping.

### Controls

`--order perm` keeps the same lags but assigns them to frequencies in a fixed
random order, so the pattern is still frozen and still repeated but has no
rising trajectory.  It separates "the figure is a glide" from "the figure is a
fixed pattern".

`--redraw` redraws the lags every repetition: same spread, same tone count,
same marginal spectrum, nothing repeats.  At tone step 0 it degenerates -- a
chord of zero spread has no lags to scramble -- so that one file is identical
to its frozen counterpart, by construction rather than by accident.

## Classic SFG, and the same figure sheared (`sfg.py`)

    python -m audios.sfg
        sfg_coherent.mp3   five tones together, every 200 ms
        sfg_stair10.mp3    the same five, delayed 0 10 20 30 40 ms
        sfg_switch.mp3     coherent for 12 s, then the staircase, one cloud
                           scheduled across the join
        sfg_check.png

A 5-tone figure repeating at 5 Hz in a balanced cloud, and the same figure
with each tone delayed 10 ms behind the one below it.  Forty-millisecond tones
against a ten-millisecond step still overlap by 30 ms, so the staircase is not
a sequence of separate events -- it is the same event with its onsets pulled
apart, spanning 80 ms of each 200 ms period instead of 40.

The **switch** file is the one worth listening to.  Two separate files ask
"can you find the figure in this one?" twice, and the answer depends as much
on how long you listened as on the stimulus; one file that changes partway
asks whether the figure you are *already holding onto* survives the shear, and
the listener is their own control.  The cloud is scheduled once, across the
join, so nothing in the background marks the moment.

### Why this pool is 21 channels and not 61

The channel count is forced by the repetition rate, not chosen.  Every channel
has to be used at the same rate, and a figure channel is used **5 times a
second by the figure alone** -- once per token at 5 Hz.  The cloud supplies
`ceiling / tone` tones per second spread over `C` channels, so balance needs

    ceiling / (tone_ms * C)  >  rate

At the 61-channel pool of `word.py` that needs a ceiling of **fourteen**;
below it the figure's channels are used twice as often as every other channel,
which identifies the figure without listening to it at all.  Measured on the
first attempt: 60-61 tones per figure channel against 28-29 for the rest.

Twenty-one channels three semitones apart, at a ceiling of six, gives 7.1
tones per channel per second.  The cloud then still supplies 30% of what
sounds in a figure channel, the background stays thin, and the counts come out
level: **91-92 tones per channel, figure and cloud channels alike**.

The cost is a coarser frequency grid -- three semitones between neighbouring
cloud channels rather than one -- which is what a 5 Hz figure buys at this
density.  A ceiling of 14 would keep the fine grid; `--ceiling` allows it.

### Residual

    coherent    concurrency 6-6, mean 6.00 +- 0.00, no dip
    staircase   concurrency 3-6, mean 5.99 +- 0.11, 3.0 dB for 0.2% of the time

The staircase's residual is the familiar one: its tones at +10, +20 and +30 ms
are off the cloud's 40 ms tiling, where the coherent figure's are on it.  It
amounts to about 0.4 ms per token and it does co-vary with the condition, so
it is worth knowing about even though it is far too brief to hear.

Token onsets are **isochronous** here, unlike `word.py`.  This is the classic
figure and the rhythm is part of it; `--jitter-ms` turns the jitter on for the
harder version.
