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
250 to 7246 Hz on a 1/24 octave grid, weighted to equal loudness at 60 phon
(ISO 226:2003). Exactly eight tones sound at every instant, by a start pattern that puts
`bg_sounding` starts in every tone length. That works for any count, not only for
multiples of it.

No two tones ever sound at once inside one critical band. A 1/24 octave pool puts four
to eight channels inside every ERB, and without this rule two thirds of the tones
acquire a beating partner. At the bottom of the pool, where the channels are 5 Hz apart
and the equal-loudness curve adds 11 dB, that is a 5 Hz throb, and it is what a listener
hears as a repeated beep rather than as a cloud. Teki's pool starts at 179 Hz, but Teki
does not weight for equal loudness; with the weighting, that octave takes over the
stimulus, so the pool starts at 250 Hz here.

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
| tone count | identical by construction; measured, 1029 in both |
| tones sounding | identical range at every step |
| long-term level | RMS-normalised, 0.000 dB apart |
| long-term spectrum | within 0.4 dB in any third-octave band |
| element loudness pulse | element-locked power against the window before it: -0.06 to -0.02 dB at every step, the same in both |
| element loudness cue | present minus absent, against the same measure split within one condition. The difference (0.10 to 0.13 dB) sits below the noise floor of the measurement (0.14 to 0.17 dB) |
| element power | equalised per draw, against the power the background actually realised. CV 0.0000 in both |
| beating | no two tones inside one critical band at once: 0 pairs, at every step |
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

Practice runs at 0 ms with feedback until 8 of the last 10 are right. Then 20 trials at
each of 7 steps, 140 trials, about 35 minutes with three self-paced breaks.

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

`analyse <subject>` scores d' per step, fits a descending logistic by maximum
likelihood, and reports the step at which d' falls to 1 with a bootstrap CI. On
simulated data with a true midpoint of 25 ms, 30 trials per step recovers the threshold
to about 7 ms either way; double the trials for 5. At 20 trials per step the session is
already 35 minutes, so split it over two sittings rather than cutting trials.

## What is not controlled, and cannot be

* Element extent grows with step. A 0 ms element is 50 ms long, a 50 ms element is
  350 ms. That is the manipulation, not a confound, but it does mean the conditions
  differ in more than asynchrony. `perm` and `scatter` are what bound the alternative
  explanations.
* Contrast is 2.5x, against roughly 10x for chord-grid figure-ground. A 350 ms element
  cannot repeat at 20 Hz, and a slow figure has a low contrast. Detection at 0 ms should
  be good rather than perfect, which is what you want, or the sweep starts at ceiling.
* PsychoPy is not used. On this Python it resolves to 2023.1.3 and builds from source,
  which is not a thing to hand a booth machine, and its value here would be a dialog
  box. All the timing that matters is baked into the pre-rendered stimuli, and playback
  goes through sounddevice.
* Levelling the power jitters the tones. The compensating gain wanders over about 7 dB,
  1.6 dB SD, so each tone sits within a decibel or two of its equal-loudness level
  rather than exactly on it. That is the price of a flat envelope, it is matched between
  the intervals, and the equal-loudness shape survives it: measured by octave, the
  realised levels still trace the ISO 226 curve.

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
