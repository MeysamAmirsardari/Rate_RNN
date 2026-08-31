# How far can two tone streams be pulled apart before they stop being one thing?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeysamAmirsardari/Rate_RNN/blob/main/audios/streaming/Streaming_playground.ipynb)
&nbsp;The notebook builds every stimulus in the browser, plays the whole sweep,
and runs the adaptive rule and the analysis on a listener whose answer is known.
No install, no account.

A direct replication of the human psychophysics in

> Elhilali M, Ma L, Micheyl C, Oxenham AJ, Shamma SA (2009). Temporal coherence
> in the perceptual organization and cortical representation of auditory
> scenes. *Neuron* 61:317–329.

and one extension it points at but never measured.

Two pure tones, A at 1000 Hz and B above it, each repeating in its own train.
Played alternately they are two streams; played together they are one, even at
a separation wider than an octave. The paper establishes that with four
conditions. **The question it leaves open is what happens in between**: slide
the B train a little and the two are still one thing, slide it to alternation
and they are two, and nobody has measured where the change happens. Figure 8
of the paper simulates exactly that sweep in the model and predicts a smooth
transition. This directory measures it in a listener.

```
python -m audios.streaming selftest                  # before anyone sits down
python -m audios.streaming check                     # the acoustic battery
python -m audios.streaming calibrate                 # set the level once
python -m audios.streaming run S01                   # the replication
python -m audios.streaming run S01 --mode sweep      # the extension
python -m audios.streaming analyse S01
```

## What the listener does

Two sequences, 500 ms apart. In one of them the very last B tone comes early
or late. Which sequence was it. Two-interval two-alternative forced choice, so
there is no criterion to worry about, and an adaptive rule that converges on
79.4% correct.

The threshold is the whole point, and it is not a threshold for streaming: it
is a threshold for **noticing that one tone moved**. The logic is the paper's
and it is the reason this design is worth copying. A listener who hears the A
and B tones as one object compares the target B against the A tone that should
have been simultaneous with it, and gets thresholds of a few ms. A listener who
has segregated them into two streams cannot make that comparison — across
streams, relative timing is not available — and has to fall back on hearing an
irregularity inside the B train alone, which is worth 10–20 ms. So the
threshold reports the percept without ever asking the listener how many streams
they heard. No subjective report, nothing to be biased about.

## The stimulus, exactly as published

Methods, pp. 11–12 of the author manuscript:

| | |
|---|---|
| A tone | 1000 Hz, fixed |
| B tone | 0.5, 0.75 or 1.25 octaves above: **6, 9, 15 semitones** |
| tone | **100 ms including** 10 ms raised-cosine ramps, so 80 ms steady |
| sequence | 5 precursor tones at each frequency, then one target at each |
| B gaps | 50 ms throughout, always |
| A gaps | **30, 50 or 70 ms** |
| control | A tones off, B tones exactly as before |
| alignment | the two trains are positioned so the **target** A and B are synchronous in the standard interval |
| ISI | 500 ms |
| track | 3-down 1-up, dT from 20 ms, step 4 → 2 → √2, ends at the sixth √2 reversal, threshold = geometric mean of those six |
| runs | at least four per condition per listener |
| ear | left only, Sennheiser HD 580, in a booth |

A 50 ms A gap makes both trains run at 150 ms per tone, so they are
synchronous from beginning to end. A 30 or 70 ms A gap makes the A train run
faster or slower than the B train, so the two drift apart — and because they
are pinned at the target, the drift is in the precursors. That is the published
way of producing asynchrony, and section *What this replication cannot settle*
says why we also do it a second way.

`check` draws the schedule from the real onsets:

```
python -m audios.streaming check          # -> out/stimulus_replicate.png
```

`demo` writes one trial as a pair of wavs, and `--seconds` stretches it for
listening. A trial is about a second, which is right for a measurement and
wrong for a demonstration: the one-thing or two-things percept takes several
seconds to build. `--seconds` is refused for the 30 and 70 ms A gaps, because
those are made asynchronous by a tempo difference and the drift accumulates, so
a longer version sweeps through every phase relationship instead of holding
one. For a long asynchronous sequence use `--mode sweep --pct 100`, which holds
the lag constant.

```
python -m audios.streaming demo --mode sweep --pct 0   --seconds 8   # together
python -m audios.streaming demo --mode sweep --pct 100 --seconds 8   # taking turns
```

### Three places the paper is silent, and what we did

**The rule contradicts itself.** The methods say "three-down one-up ... tracked
the 79.4%-correct point" and then "divided by a factor c after two consecutive
correct responses". Those cannot both be true: two-down converges on 70.7% and
three-down on 0.5^(1/3) = 79.37%. The stated convergence point is the
unambiguous half, so the rule here is **three-down**, and `n_down` is in the
config for anyone who reads it the other way.

**No level is given** for the human experiment. 70 dB SPL, which is what they
used in the ferret and what this stimulus is usually presented at. `calibrate`
plays a 1000 Hz tone at the level of one stimulus tone; set the system so a
meter at the headphone reads 70 dB SPL and then leave it alone.

**The track has no stated ceiling**, and it needs one. A backward shift of the
target B eats into the 50 ms gap in front of it, and at 50 ms the two would abut
into a single long tone. dT is capped at **40 ms**, which leaves 10 ms of
silence — plainly two events — and is twice the largest threshold the paper
reports. The published procedure must have run into the same wall (20 ms × 4 on
the first error is 80 ms) and does not say what it did about it. The cost is
measured rather than assumed, and it is in `selftest`:

| true threshold | recovered | bias | runs that hit the ceiling |
|---|---|---|---|
| 2.6 ms | 2.72 | +0.40 dB | 10% |
| 3.2 ms | 3.33 | +0.34 dB | 9% |
| 11.5 ms | 11.21 | −0.22 dB | 29% |
| 15.0 ms | 14.27 | −0.43 dB | 50% |
| 21.0 ms | 18.66 | −1.03 dB | 77% |

So the synchronous conditions are recovered to within half a dB and the
asynchronous ones are compressed by up to 11%. The claim is a factor of five
between them, so an 11% compression on the larger half does not threaten it,
but the number belongs in the methods.

## What is controlled, and measured

`check` builds many trials of every condition and measures each of these on the
rendered waveform. Rows read standard / signal; anything that differs between
them is a way of being right without hearing the asynchrony.

| control | how |
|---|---|
| tone count and total on-time | identical by construction; measured, identical |
| total energy | **0.0000 dB** between the two intervals, at every condition |
| long-term spectrum | **0.000 dB** in every third-octave band that carries anything (bands more than 40 dB down hold the gate's numerical skirt and are excluded, with their count reported) |
| where the intervals first differ | measured on the Hilbert envelope: **at the target and nowhere earlier**. The precursors are bit-identical |
| shift direction | forward on exactly 50% of trials, balanced in blocks of four rather than in expectation |
| interval | the signal is in interval 1 on exactly 50% of trials, balanced the same way |
| dT actually rendered | onsets land on samples, so the shift is quantised; measured, the rendered dT differs from the asked dT by **0.0000 ms** at every level (48 kHz gives 0.021 ms per sample and the floor is 0.25 ms) |
| gate splatter | energy of one tone outside ±3 ERB of itself: **−73 dB**. A 10 ms raised cosine does not click, and a click at an onset is the one artefact this task cannot survive |
| clipping | never |
| starting phase | drawn per trial, shared by the two intervals (see below) |

### Why the phases are randomised, which the paper did not do

Two tones summed overlap at some relative phase, and with a fixed starting
phase that alignment is *the same in every trial of a condition* for a whole
session. It shows up in the peak level: at a 60 ms lag the two intervals differ
by 0.57 dB in peak, because the shift changes where in the beat pattern the
overlap falls. Drawing the phase per trial makes that difference vary from
trial to trial instead — measured, the trial-to-trial spread is 0.52 dB against
a 0.57 dB difference, so it is noise rather than a cue. The phase is shared by
the two intervals of a trial, so the standard and the signal still differ only
in when the target lands.

### The one confound that cannot be removed, and how to read it

**Cubic difference tones.** Two loud tones generate a distortion product at
2f_A − f_B inside the ear, and it exists only while they overlap. Shifting the
target B changes the overlap by dT, so the distortion product's duration
changes with the signal.

| ΔF | f_B | 2f_A − f_B | |
|---|---|---|---|
| 6 st | 1414 Hz | **586 Hz** | audible |
| 9 st | 1682 Hz | **318 Hz** | audible |
| 15 st | 2378 Hz | −378 Hz | none generated |

Nothing in the waveform can remove this — it is made in the cochlea. Two things
make it liveable. The ratio f_B/f_A is 1.41 or more, well past the ~1.2 where
distortion products are strongest, so it is weak. And **15 semitones generates
none at all**, which turns the confound into an internal control: the paper's
result is that thresholds are small at *every* separation including 15 st, so
if the effect held at 6 and 9 st but not 15, distortion would be the reason.
It does not. Keep the level at 70 dB and no higher; report it.

## The extension: sliding the B stream

`--mode sweep`. Both trains run at the same tempo, 150 ms per tone, and the
whole B train is displaced by a constant lag from 0 to half a period. At 0 the
tones are synchronous. At half a period they alternate. Every lag in between is
a degree of asynchrony, and this is the axis Fig. 8 of the paper predicts and
does not measure.

**Tones are 75 ms here, not 100.** Against a 150 ms period, 75 ms is the one
length at which the far end of the axis is exactly alternation — A fills the
first half of the period, B the second, no overlap and no gap. With 100 ms
tones the two would still overlap by 25 ms at 100% and the sweep would never
reach its own endpoint. Fig. 8 used 75 ms tones for the same reason. The period
is unchanged, so each frequency repeats at the same rate as in the replication.

Lags at 0, 10, 20, 30, 40, 50, 65, 80 and 100% of the way to alternation, at 6
and 15 semitones. The measurement is identical: the threshold for spotting a
shift of the last B tone. Two references make it readable, and both should be
measured in the same listener:

* their **synchronous** threshold, which is where one stream lives;
* their **B-only** threshold, which is where two streams live — a listener who
  has segregated cannot use the A tones at all, so they should land on it.

`analyse` reports where the threshold crosses twice the listener's own
synchronous value. That criterion is named rather than discovered, because the
transition is not a step: a factor of two is well outside the run-to-run spread
of one cell (0.3–0.5 in log2, so a factor of 1.2–1.4) and well below the
asynchronous plateau, which the paper puts at four to six times synchronous.
Interpolation is linear in log threshold against lag, which is the space the
data live in.

### What the sweep will and will not tell you

At the far end of the axis the target A and B tones abut rather than overlap,
so a shift makes either a gap or an overlap, and those are not the same cue.
The threshold there averages two things. This is inherent to running the axis
all the way to alternation, and it is one more reason to read the far end
against the B-only control rather than against zero.

## Sessions and data on disk

`run` opens a panel first: participant id, age, sex, handedness, self-reported
hearing, years of musical training, headphone model, experimenter initials and
a line confirming consent under your own protocol. A returning subject confirms
what is on file rather than typing it again.

Every run is a new session, seeded on subject and session number together, so
the same person run twice gets two different orders. BIDS-style layout:

```
data/
  participants.tsv
  sub-S01/
    sub-S01_participant.json
    ses-01/
      sub-S01_ses-01_task-streaming_beh.tsv     one row per trial, with dT
      sub-S01_ses-01_task-streaming_runs.tsv    one row per track, with its reversals
      sub-S01_ses-01_task-streaming_beh.json    design, participant, provenance
      sub-S01_ses-01_task-streaming_events.log
```

Every trial is written as it is answered and every track as it finishes, so a
crash costs the track in progress and nothing else. `run S01 --resume` reopens
the newest unfinished session and skips the tracks it already did; it refuses
if the design has changed, because the run order would no longer be the one
that was interrupted.

**The full replication is 48 tracks and about three hours.** That is not one
sitting, and it is not meant to be — the paper's listeners were experienced and
did at least four runs in each of twelve conditions. Run it over several
sessions; `analyse` pools them. Runs are ordered so that one repeat of every
condition happens before any condition gets its second, which puts the same
amount of practice and the same amount of fatigue on every cell instead of on
whichever was last.

An unanswered trial is not scored as wrong. It is not evidence about the
listener, and feeding it to the track would drive the level up for a reason
that has nothing to do with hearing.

## Analysis

Everything is done in log space. These thresholds are log-normal — a listener
who is twice as good is a factor, not a difference — which is why the paper
takes geometric means, and it is why the intervals, the tests and the
interpolation are all on log dT.

* Per cell: geometric mean of that listener's runs, with a **t interval
  computed in log space** and back-transformed, plus the run-to-run spread in
  log2 so a reader can see how much of the interval is measurement.
* The comparisons the claim rests on, Benjamini-Hochberg corrected. Not
  "synchronous is small" — that is not a claim — but **synchronous is smaller
  than each asynchronous condition**, and **each asynchronous condition is no
  better than having no A tones at all**. The second is the one the paper leans
  on and the one worth checking: if the asynchronous thresholds sit below
  B-only, the A tones were still doing something and the streams had not fully
  segregated.
* `_figure2.png` redraws Fig. 2 with the published values behind ours in grey.
* `_sweep.png` is the extension, with the boundary marked.
* `_tracks.png` is every adaptive run, level against trial. A run that never
  settled, or sat on the floor, or walked off at the end is visible here and
  nowhere else.

### Check the analysis before a person sits down

```
python -m audios.streaming simulate SIM --root /tmp/sim
python -m audios.streaming analyse SIM --root /tmp/sim
```

writes a whole session from a listener whose thresholds are the paper's, runs
it through the real adaptive rule, and hands it to the real analysis. Nothing
in it is data. What it shows is worth knowing before committing anyone's
afternoon: **with four runs per cell, one listener's cell carries a 95%
interval spanning roughly a factor of two.** One listener cannot reproduce
Fig. 2 point by point. The pattern — a few ms synchronous against 10–20 ms
everywhere else — comes out decisively at q < 0.002. Plan for the pattern, and
pool listeners for the points.

For the sweep, `--boundary 25 --width 6` sets where the simulated listener
comes apart, and the analysis recovers it to within about 10 ms from four runs
per cell. If the boundary is the number you want to report, budget more runs
per cell there than the replication needs, or more listeners.

## What this replication cannot settle

* **The published asynchronous conditions change two things at once.** A 30 or
  70 ms A gap makes the A stream asynchronous with the B stream *and* gives it
  a different tempo, and a tempo difference is itself a segregation cue
  (Bregman 1990). The paper's own conclusion does not depend on separating
  them, but a reader will ask. `--mode sweep` is the version that does not have
  this problem: both streams keep the same tempo and only the phase moves. Run
  both and the comparison is available.

* **The listener is judging the end of a sequence.** Shifting the last B tone
  changes when the sequence stops, by dT. That is a duration cue rather than an
  asynchrony cue, and it is exactly what the B-only control bounds: with no A
  tones at all, a listener still manages 14–16 ms on that cue alone. It is the
  floor the asynchronous conditions have to be compared against, and it is why
  the interesting comparison is against B-only rather than against chance.

* **Nine experienced listeners in a booth.** The published thresholds are
  geometric means across nine listeners with extensive practice on the task.
  Numbers from one naive listener at a desk will be higher and noisier, and the
  right thing to report is the pattern and the ratios, not the values.
