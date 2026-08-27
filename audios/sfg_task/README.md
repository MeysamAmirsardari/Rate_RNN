# Does a figure survive being sheared in time?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeysamAmirsardari/Rate_RNN/blob/main/audios/sfg_task/SFG_playground.ipynb)
&nbsp;The notebook builds the real stimuli in the browser and plays every step of the
sweep. No install, no account.

A two-interval figure-ground detection experiment whose only variable is the onset
asynchrony between successive tones of the figure. At 0 ms the seven tones are a
coherent chord, the classic stochastic figure-ground stimulus. At 50 ms they are a
350 ms staircase. Everything else is held identical, and measured to be identical.

The question: temporal coherence theory says common onset is what binds components into
one object. How far can the onsets be pulled apart before the figure stops being one
thing?

## The stimulus

A cloud of 50 ms tones with 10 ms raised-cosine ramps, drawn from 117 channels spanning
250 to 7246 Hz on a 1/24 octave grid, all at the same level. Exactly five tones sound at
every instant, by a start pattern that puts `bg_sounding` starts in every tone length. That works for any count, not only for multiples of it.

No two tones ever sound at once inside one critical band. A 1/24 octave pool puts four
to eight channels inside every ERB, and without this rule two thirds of the tones
acquire a beating partner, which at the bottom of the pool is a 5 Hz throb. A listener
hears that as a repeated beep rather than as a cloud.

### Why the levels are flat and not equal-loudness weighted

ISO 226 corrects for the threshold of hearing, and nothing in this stimulus is anywhere
near threshold. Measured, the masking the cloud throws at each frequency sits 30 to
45 dB above the absolute threshold at every point in the pool, so masking and not
audibility is what limits every channel. The correction therefore buys no evenness:
modelled against threshold and cloud masking together, audibility spreads 6.0 dB across
the pool flat and 6.9 dB weighted.

What it cost was substantial. At 60 phon the weighting spans 12.6 dB, and because the
pool covers exactly the two rising arms of the curve, its two loudest bands land on the
two pool edges, which are also the places with least masking on one side. Measured on
rendered stimuli, long-term power varied 12.4 dB across channels and the top of the pool
stood 2.4 dB above its own 3-ERB neighbourhood. That is audible as a beep. Worse, the
seven tones of one figure element came out a median 5.9 dB and up to 12.6 dB apart, when
the whole premise is that they are seven equally audible tones that bind by common
onset.

Flat, as in Teki 2013: long-term power now varies 1.4 dB across channels, the worst
prominence is 1.0 dB, and the tones of an element are identical in level. The residual
1.0 dB sits at the bottom edge of the pool, which has nothing below it to mask it.

Onsets fall on a 5 ms grid. A finer grid costs nothing, since the cloud starts
`bg_sounding / tone_ms` tones a second whatever the grid is, so the grid is set by the
finest step to be tested and then held fixed for the whole experiment. If it moved with
the step, so would the background, and every condition would face a different cloud.

The figure is a discrete element: seven channels drawn at random inside a band 30
semitones wide, each delayed `step_ms` behind the one below it. Eleven elements per 6 s
interval, at irregular spacing. The interval is long because the figure has to be found
by accumulating evidence across elements rather than caught in one. The minimum gap is
set by the longest element in the experiment (350 ms), not by the current condition, so
element timing comes from one distribution in every condition. Elements never overlap,
so the number of coherent components sounding at once is the same at every step.

### Flat loudness, not a flat tone count

Holding the count constant while the figure switches seven tones on at once would need
at least `coherence` tones starting in every slot. That is 1400 onsets a second at this
resolution, and it leaves the figure with no contrast at all. So the count is allowed to
rise and the level of everything is scaled against the analytic power envelope instead:
the sum of the tones' own squared amplitude envelopes, which is the expected power
because the tones are mutually incoherent. Measured, the element leaves 0.0 dB behind at
every step and in both intervals, against 2.5 dB before.

It has to be the power and not the count. Counting is exact only if every tone carries
the same power, and under equal loudness they do not. The critical-band rule blocks the
channels around the figure, which pushes the background towards the loud edges of the
pool for as long as an element lasts, and leaves 0.9 dB behind.

## The comparison

Both intervals of a trial are built from the same element onsets and the same delays,
drawn once for the trial. They differ in one respect:

* figure present: the same seven channels on every element
* figure absent: seven channels drawn afresh for each element, at the same spacing and
  the same spectral width, never sharing more than one channel with any other element

So the listener hears the same rhythm, the same number of tones, the same element shape
and the same loudness in both intervals, and has to judge which one kept coming back at
the same pitches.

Every element is scaled to the same power. Without that, a frozen figure has constant
element loudness while a redrawn one varies, and the listener could win on modulation
depth without hearing a figure at all.

## What is controlled, and measured

`python -m audios.sfg_task check` builds many trials per step and measures every one of
these. Construction arguments are not evidence.

| control | how |
|---|---|
| tone count | identical by construction; measured, 784 in both |
| figure tones at once | identical in both intervals at every delay |
| figure duty cycle | identical in both intervals at every delay |
| tones sounding | identical range at every step |
| long-term level | RMS-normalised, 0.000 dB apart |
| long-term spectrum | within 0.4 dB in any third-octave band |
| channel balance | long-term power varies 1.4 dB across the pool; no channel stands more than 1.0 dB above its own 3-ERB neighbourhood |
| within an element | the seven tones are identical in level, so none of them carries the figure on its own |
| element loudness pulse | element-locked power against the window before it: -0.06 to -0.02 dB at every step, the same in both |
| element loudness cue | present minus absent, against the same measure split within one condition. The difference (0.10 to 0.13 dB) sits below the noise floor of the measurement (0.14 to 0.17 dB) |
| element power | equalised per draw, against the power the background actually realised. CV 0.0000 in both |
| beating | no two tones inside one critical band at once: 0 pairs, at every step. Six tones sounding, each holding a band clear either side, is 47% of the pool's 25 ERB, and over 1500 trials the rule never once had to bend. At 50 ms tones it was 79% and bent on one slot in a thousand at the shortest delay; the background takes the channel furthest from what is sounding rather than ending the session, and the battery counts how often |
| envelope at the figure rate | 0.02 dB in both intervals, at every delay: the levelling removes the 5 Hz pulse entirely |
| accidental coherence | no two elements of a figure-absent interval share more than one channel |
| figure position | uniform over the allowed range, redrawn every trial, so it is never in a learnable place |
| overall level | roved 3 dB either way per interval |
| interval position | figure in interval 1 on exactly half the trials of every step |
| order | randomised, never more than three trials in a row at the same step |

## The task

Two intervals, 6 s each, 500 ms apart. Which one had the figure. 2IFC because it needs
no criterion; proportion correct maps straight to d'. Yes/no is available
(`--task yesno`), which doubles the trials per step so false alarms can be estimated per
step.

Practice runs at 0 ms with feedback until 8 of the last 10 are right, and says so if it
ends below criterion rather than sliding into the main block. Then 20 trials at each of
7 steps, 140 trials, about 35 minutes with three self-paced breaks.

The trial line does not show the step. A subject who can see the condition will use it.
`--show-step` puts it back for when you are testing the runner yourself, and the data
records the step either way.

Calibrate once. `calibrate` plays 1 kHz at the stimulus level; set the system so a meter
at the headphone reads 65 dB SPL, then leave it.

## Sessions and data on disk

`run` opens a panel first: participant id, age, sex, handedness, self-reported hearing,
years of musical training, headphone model, experimenter initials, and a line confirming
that consent was obtained under your own protocol. A returning subject is shown what is
already on file and confirms it rather than typing it again.

Every run is a new session. The randomisation is seeded on the subject and the session
number together, so the same person run twice gets two different trial orders and two
different sets of stimuli. Nothing is ever appended to an earlier session by accident.

The layout follows the way BIDS lays out behaviour, so it is readable by someone who has
never seen this code:

```
data/
  participants.tsv                                 one row per subject
  sub-S01/
    sub-S01_participant.json                       the panel, as entered
    ses-01/
      sub-S01_ses-01_task-sfg_beh.tsv              one row per trial
      sub-S01_ses-01_task-sfg_beh.json             design, participant, provenance
      sub-S01_ses-01_task-sfg_events.log           start, breaks, quits, finish
```

The `_beh.json` holds every field of `Design`, the participant record as it stood that
day, and the provenance: git commit and whether the tree was dirty, host, platform,
Python, numpy and scipy versions, and the start time. Any stimulus in the session can be
rebuilt from the seed in the TSV plus that file.

Trials are appended as they are answered, so an interrupted session costs nothing.
`run <subject> --resume` reopens the newest unfinished session of that task and skips
the trials already answered. It refuses if the design has changed since the session was
started, because the trial list would no longer be the one that was interrupted.

`subjects` prints what has been recorded. `analyse` pools every session of a subject by
default, or takes `--session N`.

## The control session

`run <subject> --controls`, 20 trials per cell at 20 and 40 ms:

* rise: the sweep itself, measured by the same ears in the same session
* perm: the same set of delays, frozen, but not a monotonic sweep. Separates asynchrony
  from the frequency sweep it draws
* redraw: the same channels and the same elements, delays redrawn every element.
  Separates the frozen pattern from channel recurrence
* scatter: the same channels at the same rate, never grouped into elements. Long-term
  spectrum identical to the figure's, no temporal coherence at all. This is what
  separates temporal coherence from spectral prominence, and it is the control a
  reviewer will ask for

## Analysis

```
python -m audios.sfg_task analyse S01           # one subject, all sessions
python -m audios.sfg_task analyse S01 --session 2
python -m audios.sfg_task analyse S01 --controls
python -m audios.sfg_task group                 # across subjects
```

`analyse` prints a table and writes three figures and three csvs beside the subject's
sessions. It pools sessions by default, since a session is a sitting and not a
condition.

### The number the experiment is for

Accuracy is turned into d' (for 2IFC, `d' = sqrt(2) z(pc)`, which needs no criterion),
a descending logistic is fitted to the seven points by maximum likelihood, and the
threshold reported is **the delay at which d' falls to 1**. That level is chosen because
the task can resolve it; a half-way point often sits off the end of the range that was
tested. Its 95% CI comes from a parametric bootstrap over the per-cell binomials.

### Error bars and tests

* Each point carries a **Wilson 95% interval**, which behaves at 0 and 1 where the
  normal approximation does not.
* d' carries a standard error by the delta method, from the binomial on pc.
* Stars above each point are a one-sided binomial test **against chance**, corrected
  across the seven delays with Benjamini-Hochberg.
* `vs best` in the table is a Fisher exact test of each delay **against the 0 ms
  chord**, also FDR corrected. This is the column that says where performance first
  falls away.
* `step effect` is a single-trial logistic of correct on delay, with guessing built into
  the link, tested by likelihood ratio against the same model with the slope removed.
  That is the one test of whether delay matters at all, and it uses every trial rather
  than seven summary points.

### The figures

**`_accuracy.png`** The figure the experiment is for, on its own: accuracy against delay,
Wilson intervals, chance shaded, and the delay effect in one line under the title. The
fitted curve and the threshold appear only when the fit is worth believing.

**`_psychometric.png`** The same accuracy panel beside d', for when you want both.

**`_timecourse.png`** Accuracy against position in the experiment, one colour per delay,
with the pooled curve in black and its standard error band, and each delay's overall
accuracy on the right on the same axis. Dashed verticals are session boundaries. A trial
of any one delay happens once every few trials, so a plain sliding window would hold
four or five of them and be unreadable; each delay's trials are weighted by a Gaussian in
trial index instead, and the error band uses the effective N from the sum of the weights.
`--window` sets the width, in trials.

This is where a session goes wrong visibly: a subject who is still learning shows every
curve climbing early, one who is tiring shows the black curve sagging late, and one who
has stopped trying shows the 0 ms anchor coming down to meet the rest.

**`_checks.png`** The checks as a forest plot, and response time against delay.

### When the fit refuses itself

A threshold is only printed and only drawn when it means something. It is withheld, with
the reason, when the bootstrap interval comes out wider than the range of delays that
were tested, or when accuracy does not peak at the smallest delay, because a logistic
fitted to a non-monotonic function will report a number and that number will be noise.

### What to check before believing the threshold

The table's second half is there to be read, not skipped. Only the response bias has
chance for a null; the rest come in pairs, and the question about a pair is whether its
two halves agree.

| check | what a failure means |
|---|---|
| said interval 1 | an interval preference. 2IFC is robust to a mild one, but a strong one means the subject is not using both intervals |
| figure in 1 vs in 2 | the two intervals are not equally good, which should not happen if the stimulus is matched |
| first vs second half | practice or fatigue. Also tested on single trials as `drift over the session` |
| after a correct vs after an error | sequential effects, usually post-error slowing |
| anchor, first vs second half | whether the subject is still trying. The 0 ms condition should stay easy all the way through |

Response time should rise with delay. If it is flat, the subject may be answering on a
fixed schedule rather than on the evidence.

### Across subjects

`group` fits each subject separately, prints the per-subject thresholds, and gives the
mean with its SEM and a t-based 95% CI. Per-subject thresholds then mean, rather than
pooling raw trials, because subjects differ and the claim is about listeners. On
simulated data with true thresholds of 17 to 33 ms, five subjects recovered 13 to 37 ms
and a group mean within 1 ms of the truth.

At 20 trials per step, one subject's threshold carries a CI about 14 ms wide. Two
sessions halve that. Decide which you need before running twenty people once.

## What is not controlled, and cannot be

* **The figure changes character across the sweep, not just its asynchrony.** A 5 ms
  element is 60 ms long and puts six tones on at once; a 45 ms element is 300 ms long
  and never has more than one. Measured, a figure tone is sounding 16% of the time at
  5 ms and 57% at 45 ms. Same tone count, same rate, same contrast, different object.
  This is inherent to spreading tones in time. `perm` holds the extent and removes the
  order; `scatter` is the limit where the asynchrony is unbounded. Those two bound the
  alternative explanations, and this is the first limitation a referee will find.

* **The other interval is not a plain cloud, and it cannot be.** Take the long-term
  spectrum of each interval, measure how far its loudest channels stand above their
  neighbours, and pick the interval with the taller peaks. Against a plain cloud that
  gives **d' 8 to 11, which is 100% correct in 2IFC at every delay**, without ever
  hearing a figure. Six seconds is long enough to read seven elevated channels off the
  spectrum, and no scheduling removes it: one interval has seven elevated channels and
  the other does not, and that is what "figure" means.

  So the other interval carries **the same seven channels**, coming back at the same
  rate and with the same regularity, each on its own schedule. Both intervals then have
  the same channels, the same tone count, the same six tones sounding at every instant,
  the same level, the same long-term spectrum and the same contrast, and differ in
  whether the seven fire *together*. Measured, the observer who ignores time falls to
  **d' 0.01 to 0.08, which is 50 to 52% correct**: chance.

  Three things had to be right for that. The figure substitutes background tones rather
  than adding to them, so the tone count does not move. The scattered channels are
  nudged off each other's slots, so the control never forms a momentary chord and never
  needs a levelling gain the coherent side does not. And a coherent element that would
  collide with its overlapping neighbour slides whole, pattern intact, rather than
  doubling a slot. With all three the levelling gain is **0.00 dB in both intervals at
  every delay**, and the leveller is inert.

  Trial-by-trial feedback matters here: it does not bias 2IFC, but it teaches whatever
  cue works, so it is only safe once the only cue that works is the intended one.

  `--absent cloud` restores the classic comparison, for continuity with the published
  task. Run it once as a control precisely because it is the confounded one: if a
  listener is much better on it than on the matched version, that difference is what
  the spectral cue is worth to a human.

## Usage

```
python -m audios.sfg_task check                    # the battery, plus figures
python -m audios.sfg_task demo --step-ms 20 --play # listen to one trial
python -m audios.sfg_task calibrate                # set the level once
python -m audios.sfg_task run                      # asks who is sitting down
python -m audios.sfg_task run S01 --resume         # finish an interrupted one
python -m audios.sfg_task run S01 --controls       # the control session
python -m audios.sfg_task subjects                 # what has been recorded
python -m audios.sfg_task analyse S01
```

Everything is set in `config.py` and nothing is decided anywhere else. `check` and
`demo` write to `out/`, sessions to `data/`, both relative to this directory rather than
to where you ran the command. `--root` puts the data somewhere else.

The runner needs a real terminal for the keypresses, and says so rather than failing
half way through if it does not have one.

`SFG_playground.ipynb` is the Colab notebook behind the badge above. It imports this
package rather than reimplementing anything, so what it plays is what a subject hears.

## Where this sits in the literature

Teki et al. 2013 (eLife) and O'Sullivan et al. 2015 (J Neurosci) both ramp the figure in
frequency, with coherent tones stepping up 1 to 4 bands per chord, and both find the
figure survives it. Shearing it in time is the harder question, because onset asynchrony
is precisely what temporal coherence theory says should destroy binding. We found no
published treatment of onset asynchrony as a figure-ground manipulation. Cite the
frequency-ramp work regardless: it is the nearest prior art, and it predicts that the
figure ought to be robust.
