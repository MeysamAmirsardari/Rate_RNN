# Does a figure survive being sheared in time?

A two-interval figure-ground detection experiment whose only variable is the
**onset asynchrony between successive tones of the figure**.  At 0 ms the
seven tones are a coherent chord — the classic stochastic figure-ground
stimulus.  At 50 ms they are a 350 ms staircase.  Everything else is held
identical, and measured to be identical.

The question: temporal coherence theory says common onset is what binds
components into one object.  How far can the onsets be pulled apart before
the figure stops being one thing?

## The stimulus

A cloud of 50 ms tones drawn from 129 channels spanning 179–7246 Hz on a
1/24-octave grid (Teki et al. 2013), weighted to equal loudness at 60 phon
(ISO 226:2003).  Ten tones sound at every instant.

Onsets fall on a 5 ms grid; a tone lasts `k = tone_ms / hop_ms` slots and is
one ramp longer than its slots, with power-complementary ramps, so tones
ramping out are matched by tones ramping in and the background envelope is
flat by construction rather than by smoothing.  A finer grid costs nothing:
the cloud starts `bg_sounding / tone_ms` tones a second whatever the grid is,
so the grid is set by the finest step to be tested and then **held fixed for
the whole experiment**.  If it moved with the step, so would the background,
and every condition would face a different cloud.

The **figure** is a discrete element: seven channels evenly spread over
30 semitones, each delayed `step_ms` behind the one below it.  Six elements
per 3.5 s interval, at irregular spacing.  The minimum gap is set by the
*longest* element in the experiment (350 ms), not by the current condition,
so element timing is drawn from one distribution in every condition.
Elements never overlap, so the number of coherent components sounding at
once is the same at every step.

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
| element loudness pulse | element-locked envelope of both intervals, present − absent, against the same measure split within one condition — the difference sits **below** the noise floor of the measurement |
| element power | equalised per draw; CV 0.0000 in both |
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

## What is *not* controlled, and cannot be

* **Element extent grows with step.**  A 0 ms element is 50 ms long, a 50 ms
  element is 350 ms.  This is the manipulation, not a confound, but it means
  the conditions differ in more than asynchrony.  `perm` and `scatter` are
  what bound the alternative explanations.
* **The loudness pulse shrinks with step**, 2.5 dB at 0 ms to 0.8 dB at
  50 ms.  It is matched between the two intervals at every step, so it
  cannot support the discrimination — but the acoustic salience of *an
  element happening at all* does fall with step.
* **Contrast is 2.1×**, against roughly 10× for chord-grid figure-ground.
  A 350 ms element cannot repeat at 20 Hz; a slow figure has a low contrast
  and there is no way around it.  Detection at 0 ms should be good, not
  perfect — which is what you want, or the sweep starts at ceiling.

## Usage

```
python -m audios.sfg_task check                    # the battery + figures
python -m audios.sfg_task demo --step-ms 20 --play # listen to one trial
python -m audios.sfg_task calibrate
python -m audios.sfg_task run S01
python -m audios.sfg_task run S01 --controls
python -m audios.sfg_task analyse S01
```

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
