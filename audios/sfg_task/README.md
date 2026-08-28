# Does a figure survive being sheared in time?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeysamAmirsardari/Rate_RNN/blob/main/audios/sfg_task/SFG_playground.ipynb)
&nbsp;The notebook builds the real stimuli in the browser and plays every step of the
sweep. No install, no account.

A two-interval figure-ground detection experiment whose only variable is the onset
asynchrony between successive tones of the figure. At 5 ms the seven tones start
inside 30 ms of each other and are effectively the coherent chord of the classic
stochastic figure-ground stimulus. At 40 ms they are a 280 ms staircase with no overlap
left at all. Everything else is held identical, and measured to be identical.

The question: temporal coherence theory says common onset is what binds components into
one object. How far can the onsets be pulled apart before the figure stops being one
thing?

## The stimulus

A cloud of 40 ms tones with 10 ms raised-cosine ramps, drawn from 234 channels spanning
250 to 7246 Hz on a 1/48 octave grid, all at the same level. Exactly eight tones sound at
every instant, by a start pattern that puts `bg_sounding` starts in every tone length.
That works for any count, not only for multiples of it.

The grid is finer than the published 1/24 octave because what the figure has to stand
out from is the background's rate *per channel*, and a finer grid thins that without
touching the figure: contrast 2.7x at 1/24 octave, 4.7x at 1/48.

No two tones ever sound at once inside one critical band. Without this rule two thirds
of the tones acquire a beating partner, which at the bottom of the pool is a slow throb.
A listener hears that as a repeated beep rather than as a cloud.

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

Onsets fall on a 5 ms grid, held fixed for the whole experiment. If it moved with the
step, so would the background, and every condition would face a different cloud.

**5 ms is the floor of the paradigm, and it is worth knowing why.** The figure
substitutes background tones rather than adding to them, so a figure tone can only take
a slot a background tone would have started in, which forces `bg_sounding = tone_ms /
hop_ms`. Halving the grid to 2.5 ms therefore doubles the tones sounding at once to 16,
and at 16 the critical-band rule runs out of pool: measured, the dealer is forced off
its first choice 1301 times in 60 trials, against 0 at the present setting. So the
finest asynchrony this design can present is one slot, and the psychometric function
cannot be sampled below 5 ms.

The figure is a discrete element: seven channels drawn at random inside a band 30
semitones wide, each delayed `step_ms` behind the one below it. Seventeen elements per
6 s interval at 3.07 Hz, spaced 326 ms apart on average with +-20 ms of jitter, so the
rhythm is not isochronous and the figure cannot be followed by predicting when it is
due. The interval is long because the figure has to be found by accumulating evidence
across elements rather than caught in one.

Element timing comes from one distribution in every condition: the gap is set by the
longest element in the experiment (280 ms), not by the current one. Two floors, and they
are different things. `rest_gap` is one tone plus its rest and may never be crossed,
because every element uses the same seven channels and a shorter gap is one channel
sounding twice at once. `min_gap` is only where the widest element stops touching the
next, and a negative jitter is allowed to eat into it, because rejecting those draws let
the floor and not the jitter decide the bottom of the distribution. Gaps come out
290-360 ms, sd 11.34 against the 11.55 of the ideal uniform. Measured, elements graze at
all on 0.2% of gaps and by at most 5 ms, and no two figure tones ever start in the same
slot above the 5 ms anchor.

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

* figure present: the same seven channels on every element, arriving together
* figure absent (the default, `--absent scattered`): **the same seven channels**, coming
  back at the same rate and with the same regularity, each on its own schedule and never
  grouped into an element
* `--absent cloud` is the classic alternative: seven channels drawn afresh for each
  element. It is confounded, and the reason is in *What is not controlled* below

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
| tone count | identical by construction; measured, 1192 in both |
| figure tones at once | identical in both intervals at every delay |
| figure duty cycle | how much of the interval the seven channels sound at all. It is set by the delay (20% at 5 ms, 80% at 40 ms), so the control is spread by the figure's own step to follow it: 46/47, 63/64, 80/80. The residue is 12 points at the 5 ms anchor, where the figure is a chord and the control must not be, and it does not grow with the delay |
| tones sounding | identical range at every step |
| long-term level | RMS-normalised, 0.000 dB apart |
| long-term spectrum | within 0.4 dB in any third-octave band |
| channel balance | long-term power varies 1.4 dB across the pool; no channel stands more than 1.0 dB above its own 3-ERB neighbourhood |
| within an element | the seven tones are identical in level, so none of them carries the figure on its own |
| element loudness pulse | element-locked power against the window before it: -0.06 to -0.02 dB at every step, the same in both |
| element loudness cue | present minus absent, against the same measure split within one condition. The difference (0.10 to 0.13 dB) sits below the noise floor of the measurement (0.14 to 0.17 dB) |
| element power | equalised per draw, against the power the background actually realised. CV 0.0000 in both |
| beating | no two tones inside one critical band at once: 0 pairs, at every step. Eight tones sounding, each holding a band clear either side, is about half of the pool's 25 ERB, and over 1120 trials across both absent modes and all four variants the rule never once had to bend. If it ever runs out the background takes the channel furthest from what is sounding rather than ending the session, and the battery counts how often |
| envelope at the figure rate | 0.07 to 0.16 dB between the intervals, at every delay: the levelling removes the element pulse entirely |
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

Practice runs at the smallest delay with feedback until 8 of the last 10 are right, and
says so if it ends below criterion rather than sliding into the main block. Then 24
trials at each of 5 steps, 120 trials, about 30 minutes with two self-paced breaks.

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

`run <subject> --controls`, 20 trials per cell at 10 and 30 ms:

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
a descending logistic is fitted to the five points by maximum likelihood, and the
threshold reported is **the delay at which d' falls to 1**. That level is chosen because
the task can resolve it; a half-way point often sits off the end of the range that was
tested. Its 95% CI comes from a parametric bootstrap over the per-cell binomials.

### Error bars and tests

* Each point carries a **Wilson 95% interval**, which behaves at 0 and 1 where the
  normal approximation does not.
* d' carries a standard error by the delta method, from the binomial on pc.
* Stars above each point are a one-sided binomial test **against chance**, corrected
  across the five delays with Benjamini-Hochberg.
* `vs best` in the table is a Fisher exact test of each delay **against the smallest
  delay**, also FDR corrected. This is the column that says where performance first
  falls away.
* `step effect` is a single-trial logistic of correct on delay, with guessing built into
  the link, tested by likelihood ratio against the same model with the slope removed.
  That is the one test of whether delay matters at all, and it uses every trial rather
  than five summary points.

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
has stopped trying shows the 5 ms anchor coming down to meet the rest.

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
| anchor, first vs second half | whether the subject is still trying. The smallest delay should stay easy all the way through |

Response time should rise with delay. If it is flat, the subject may be answering on a
fixed schedule rather than on the evidence.

### Across subjects

`group` fits each subject separately, prints the per-subject thresholds, and gives the
mean with its SEM and a t-based 95% CI. Per-subject thresholds then mean, rather than
pooling raw trials, because subjects differ and the claim is about listeners. On
simulated data with true thresholds of 17 to 33 ms, five subjects recovered 13 to 37 ms
and a group mean within 1 ms of the truth.

At 24 trials per step, one subject's threshold carries a CI about 13 ms wide. Two
sessions halve that. Decide which you need before running twenty people once.

## What is not controlled, and cannot be

* **The figure changes character across the sweep, not just its asynchrony.** A 5 ms
  element is 70 ms long and puts seven tones on at once; a 40 ms element is 280 ms long
  and never has more than one. Same tone count, same rate, same contrast, same duty,
  different object. This is inherent to spreading tones in time, and it is the first
  limitation a referee will find: *stopped binding* and *became a longer, slower object*
  are the same manipulation.

  The control that settles it is not in the sweep and should be run: hold the element
  extent fixed and trade tones against delay. Seven tones 25 ms apart, four 50 ms apart
  and three 75 ms apart are all about 190 ms long with very different asynchrony, and
  `coherence` is already a field in `Design`. If performance tracks the asynchrony and
  not the extent, the objection is dead. It is also the classic Teki coherence
  manipulation, so it connects to the published work for free.

* **The sweep has to be placed where the function falls, and one pilot says that is
  narrow.** With 30 ms tones a subject scored 92% at 5 ms and chance at 15, 25, 35 and
  45: the whole psychometric function lived between two adjacent points, because seven
  tones stop overlapping at all past `6 * step = tone_ms`. 40 ms tones spread that
  ladder over 87.5, 75, 50, 25 and 0% overlap, which is what the present sweep samples.
  Pilot four points before committing 30 minutes an ear, and if the fall is still inside
  10 ms, report a bound rather than a threshold: the grid cannot resolve below 5 ms.

* **The other interval is not a plain cloud, and it cannot be.** Take the long-term
  spectrum of each interval, measure how far its loudest channels stand above their
  neighbours, and pick the interval with the taller peaks. Against a plain cloud that
  gives **d' 8 to 11, which is 100% correct in 2IFC at every delay**, without ever
  hearing a figure. Six seconds is long enough to read seven elevated channels off the
  spectrum, and no scheduling removes it: one interval has seven elevated channels and
  the other does not, and that is what "figure" means.

  So the other interval carries **the same seven channels**, coming back at the same
  rate and with the same regularity, each on its own schedule. Both intervals then have
  the same channels, the same tone count, the same eight tones sounding at every instant,
  the same level, the same long-term spectrum, the same contrast and the same duty, and
  differ in whether the seven fire *together*. Measured, the observer who ignores time
  falls to **d' 0.005 to 0.54, which is 50 to 60% correct**: chance.

  Four things had to be right for that. The figure substitutes background tones rather
  than adding to them, so the tone count does not move. The scattered channels are
  nudged off each other's slots, so the control never forms a momentary chord and never
  needs a levelling gain the coherent side does not. A coherent element that would
  collide with a neighbour slides whole, pattern intact, rather than doubling a slot.
  And the control is spread by the figure's own step, so the two intervals match on how
  busy the seven channels are. With all four the levelling gain is **0.00 dB in both
  intervals at every delay**, and the leveller is inert.

  That last one sets the jitter. Both intervals draw their onsets from the same
  distribution, and how often the seven trains coincide falls as the jitter decorrelates
  them, so a wider jitter pulls the control's duty off the figure's at the narrow
  delays. At the 10 ms step the two are 0.6 points apart at +-10 ms of jitter, 2.9 at
  +-20, 7.1 at +-30 and 11.1 at +-40. +-20 ms is the most jitter that keeps every sweep
  point inside 3 points.

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
