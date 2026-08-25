# sfg — stochastic figure-ground

    python -m audios.sfg                       # figure, no-figure, lags-redrawn
    python -m audios.sfg --step-ms 10 --play
    python -m audios.sfg --coherence 4 --n-sounding 12

Six wavs and a raster land in `out/`. Every run prints the control battery,
measured on the built stimulus.

## The design

Following Teki 2013 / O'Sullivan 2015 / Lu 2025, with the choices set out in
the plan:

**The figure is a re-labelling, never an addition.** `n_sounding` tones are up
at every instant whatever the figure does — O'Sullivan hold it at 15 "to keep
the broadband power and all other low-level features constant". Figure-present
and figure-absent then match in tone count, energy and envelope with no
patching. Hold it fixed across conditions.

**Time runs on a grid finer than a tone.** A tone lasts `k` slots and the same
number start in every slot, so concurrency is exactly constant while onsets
can fall anywhere the grid allows. This matters for the shear: on a chord grid
a sheared figure would be *the only thing starting off-grid*, and listeners
would detect asynchrony rather than loss of coherence. The grid is solved as
the coarsest that both the tone and the shear land on, since a finer one costs
density for nothing.

**Power-complementary crossfades.** Tones are one ramp longer than their slots
and ramp with sin/cos amplitude, so tones ramping out are matched by those
ramping in. Independent ramps over abutting slots leave a dip at every
boundary — a flutter at the slot rate.

**Equal loudness, and the confound it creates.** ISO 226 weighting at
`phon`, so no spectral region dominates; without it the stimulus is harsh and
the figure's salience depends on where it lands. But the weighting spans
several dB, so a figure on loud channels adds more than the background tones
it replaces — recurring at the figure's own rate. Measured, that put +0.14 dB
of envelope modulation at the figure rate, right at the modulation-detection
threshold. The figure's position is therefore chosen among placements whose
summed weight matches what the background would have contributed, which takes
the difference to **+0.008 dB**.

**Loose channel dealing.** Least-used-first keeps the counts level but makes
each channel return at suspiciously even intervals — the one property the
figure should own alone. The rank is bucketed so the choice falls back to the
rng.

## Controls

| flag | what it rules out |
|---|---|
| `coherent=False` | energy, tone count, timing — the canonical control: the same extras at the same times, redrawn every repetition |
| `redraw_lags=True` | the pattern, keeping the spread |
| `order='perm'` | "it's a rising glide" rather than a fixed pattern |
| `coherence=0` | figure absent entirely |

## Measured

Defaults, three seeds, 8 s:

    sounding                 10-10 in every condition
    env @ figure rate        0.175 figure / 0.277 no-figure
    figure channel           4.90/s vs background 2.20/s   (2.2x)
    background channel use   16-20 tones, 2-8 per quarter

Contrast trades against how much of the chord the figure occupies: at
`coherence` 7 the chord cannot be sparser than 7, which bounds it near 3x.
Classic SFG reaches ~15x only because its figure repeats *every chord* at
20–25 Hz; a 5 Hz token is a sparser figure and a harder task.

`step_ms = 0` makes `redraw_lags` degenerate — a chord has no lags to redraw —
so that column matches the figure column by construction, not by accident.
