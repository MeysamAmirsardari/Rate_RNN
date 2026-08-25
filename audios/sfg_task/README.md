# Does a figure survive being sheared in time?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MeysamAmirsardari/Rate_RNN/blob/main/audios/sfg_task/SFG_playground.ipynb)
&nbsp;**Listen to it first** — the notebook builds the real stimuli in the browser, plays
every step of the sweep, and lets you turn the knobs. No install, no account.

A two-interval figure-ground detection experiment whose only variable is the
**onset asynchrony between successive tones of the figure**.  At 0 ms the
seven tones are a coherent chord — the classic stochastic figure-ground
stimulus.  At 50 ms they are a 350 ms staircase.  Everything else is held
identical, and measured to be identical.

The question: temporal coherence theory says common onset is what binds
components into one object.  How far can the onsets be pulled apart before
the figure stops being one thing?

## The stimulus

A cloud of 50 ms tones (10 ms raised-cosine ramps) drawn from 117 channels
spanning 250–7246 Hz on a 1/24-octave grid, weighted to equal loudness at
60 phon (ISO 226:2003).  Eight tones sound at every instant — exactly eight,
by a start pattern that puts `bg_sounding` starts in every tone length, which
works for any count and not only for multiples of it.

**No two tones ever sound at once inside one critical band.**  A
1/24-octave pool puts four to eight channels inside every ERB, and without
this rule two thirds of the tones acquire a beating partner; at the bottom
of the pool, where the channels are 5 Hz apart and the equal-loudness curve
adds 11 dB, that is a 5 Hz throb, and it is what a listener hears as a
repeated beep rather than as a cloud.  Teki's pool starts at 179 Hz, but
Teki does not weight for equal loudness; with the weighting that octave
takes over the stimulus, so the pool starts at 250 Hz here.

Onsets fall on a 5 ms grid.  A finer grid costs nothing — the cloud starts
`bg_sounding / tone_ms` tones a second whatever the grid is — so the grid is
set by the finest step to be tested and then **held fixed for the whole
experiment**.  If it moved with the step, so would the background, and every
condition would face a different cloud.

The **figure** is a discrete element: seven channels drawn at random inside
a band 30 semitones wide, each delayed `step_ms` behind the one below it.
Eleven elements per 6 s interval, at irregular spacing.  The interval is
long because the figure has to be found by accumulating evidence across
elements rather than caught in one.  The minimum gap is set by the *longest*
element in the experiment (350 ms), not by the current condition, so element
timing is drawn from one distribution in every condition.  Elements never
overlap, so the number of coherent components sounding at once is the same
at every step.

**The loudness is flat, not the tone count.**  Holding the count constant
while the figure switches seven tones on at once would need at least
`coherence` tones starting in every slot — 1400 onsets a second at this
resolution, which leaves the figure with no contrast at all.  So the count
is allowed to rise and the level of everything is scaled against the
analytic power envelope instead: the sum of the tones' own squared amplitude
envelopes, which is the expected power because the tones are mutually
incoherent.  Measured, the element leaves 0.0 dB behind at every step, in
both intervals, against 2.5 dB before.  It has to be the power and not the
count: counting is exact only if every tone carries the same power, and
under equal loudness they do not — the critical-band rule blocks the
channels around the figure, which pushes the background towards the loud
edges of the pool for as long as an element lasts, and leaves 0.9 dB behind.

## The comparison

Both intervals of a trial are built from the **same element onsets and the
same delays**, drawn once for the trial.  They differ in one respect:

* **figure present** — the same seven channels on every element
* **figure absent** — seven channels drawn afresh for each element, at the
  same spacing and the same spectral width, never sharing more than one
  channel with any other element

So the listener hears the same rhythm, the same number of tones, the same
element shape and the same loudness in both intervals, and must judge which
one kept coming back at the same pitches.

Every element is scaled to the same summed power.  Without that, a frozen
figure has constant element loudness while a redrawn one varies, and the
listener could win on modulation depth without hearing a figure at all.

## What is controlled, and measured

`python -m audios.sfg_task check` builds many trials per step and measures
every one of these.  Construction arguments are not evidence.

| control | how |
|---|---|
| tone count | identical by construction; measured, 732 in both |
| tones sounding | identical range at every step |
| long-term level | RMS-normalised; 0.000 dB apart |
| long-term spectrum | ≤ 0.2 dB in any third-octave band |
| element loudness pulse | element-locked power of both intervals against the window before them: −0.06 to −0.02 dB at every step, and the same in both |
| element loudness cue | present − absent, against the same measure split within one condition — the difference (0.10–0.13 dB) sits **below** the noise floor of the measurement (0.14–0.17 dB) |
| element power | equalised per draw, against the power the background actually realised rather than the pool average; CV 0.0000 in both |
| beating | no two tones inside one critical band at once: 0 pairs, at every step |
| accidental coherence | no two elements of a figure-absent interval share more than one channel |
| figure position | uniform over the allowed range, redrawn every trial, so it is never in a learnable place |
| overall level | roved ±3 dB per interval |
| interval position | figure in interval 1 on exactly half the trials of every step |
| order | randomised, never more than three trials in a row at the same step |

## The task

Two intervals, 3.5 s each, 500 ms apart; **which one had the figure**.
2IFC because it needs no criterion — proportion correct maps straight to d'.
Yes/no is available (`--task yesno`, which doubles the trials per step so
false alarms can be estimated per step).

Practice runs at 0 ms with feedback until 8 of the last 10 are right.
Then 30 trials at each of 7 steps = 210 trials, ≈ 34 min including four
self-paced breaks.  Responses are appended to CSV as they arrive, so
stopping costs nothing that has been run.

Calibrate once: `calibrate` plays 1 kHz at the stimulus level; set the
system so a meter at the headphone reads 65 dB SPL and leave it.

## The control session

`run <subject> --controls`, 20 trials per cell at 20 and 40 ms:

* **rise** — the sweep itself, measured by the same ears in the same session
* **perm** — the same set of delays, frozen, but not a monotonic sweep.
  Separates *asynchrony* from *the frequency sweep it draws*.
* **redraw** — the same channels and the same elements, delays redrawn every
  element.  Separates *the frozen pattern* from *channel recurrence*.
* **scatter** — the same channels at the same rate, never grouped into
  elements.  Long-term spectrum identical to the figure's, no temporal
  coherence at all.  **This is what separates temporal coherence from
  spectral prominence**, and it is the control a reviewer will ask for.

## Analysis

`analyse <subject>` scores d' per step, fits a descending logistic by
maximum likelihood, and reports the step at which d' falls to 1, with a
bootstrap CI.  On simulated data with a true midpoint of 25 ms, 30 trials
per step recovers the threshold to about ±7 ms; double the trials for ±5.
At 20 trials per step the session is 35 minutes, which is already long —
`run` picks up where it stopped, so split it over two sittings rather than
cutting trials.

## What is *not* controlled, and cannot be

* **Element extent grows with step.**  A 0 ms element is 50 ms long, a 50 ms
  element is 350 ms.  This is the manipulation, not a confound, but it means
  the conditions differ in more than asynchrony.  `perm` and `scatter` are
  what bound the alternative explanations.
* **The loudness pulse shrinks with step**, 2.5 dB at 0 ms to 0.8 dB at
  50 ms.  It is matched between the two intervals at every step, so it
  cannot support the discrimination — but the acoustic salience of *an
  element happening at all* does fall with step.
* **Contrast is 2.5×**, against roughly 10× for chord-grid figure-ground.
  A 350 ms element cannot repeat at 20 Hz; a slow figure has a low contrast
  and there is no way around it.  Detection at 0 ms should be good, not
  perfect — which is what you want, or the sweep starts at ceiling.
* **Levelling the power jitters the tones.**  The compensating gain wanders
  over about 7 dB, 1.6 dB SD, so each tone sits within a decibel or two of
  its equal-loudness level rather than exactly on it.  That is the price of
  a flat envelope, it is matched between the intervals, and the
  equal-loudness shape survives it: measured by octave, the realised levels
  still trace the ISO 226 curve.

## Usage

```
python -m audios.sfg_task check                    # the battery + figures
python -m audios.sfg_task demo --step-ms 20 --play # listen to one trial
python -m audios.sfg_task calibrate
python -m audios.sfg_task run S01                   # resumes if interrupted
python -m audios.sfg_task run S01 --controls
python -m audios.sfg_task analyse S01
```

`SFG_playground.ipynb` is the Colab notebook behind the badge above; it imports this
package rather than reimplementing anything, so what it plays is what a subject hears.

Everything is set in `config.py`; nothing is decided anywhere else.
`check` and `demo` write to `out/`, sessions to `data/<subject>/`, both
relative to this directory rather than to where you ran the command.

## Where this sits in the literature

Teki et al. 2013 (eLife) and O'Sullivan et al. 2015 (J Neurosci) both ramp
the figure **in frequency** — coherent tones stepping up 1–4 bands per
chord — and both find the figure survives it.  Shearing it **in time** is
the harder question, because onset asynchrony is precisely what temporal
coherence theory says should destroy binding.  We found no published
treatment of onset asynchrony as a figure-ground manipulation.  Cite the
frequency-ramp work regardless: it is the nearest prior art and it predicts
that the figure ought to be robust.
