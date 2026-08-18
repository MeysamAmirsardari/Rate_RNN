# Supplementary Figure 3 — AB/BA circuit-state snapshots

A supplementary mechanism figure showing the predictive cascade as circuit
states at four moments in a trial: **node fill is a cell's rate at that
instant, line width is a learned recurrent weight.**

## Build

```bash
python -m final_figures.figure_3.make_circuit_snapshots
```

`--force-data` repeats both simulated runs. The cache key is a hash of the full
configuration, so changing a model parameter forces a rerun.

## Every number is simulator output

Nothing in this figure is drawn by hand, idealised, or chosen for effect. Node
fills, printed rates, line widths and quantitative estimates are all read
from `model0.simulate` output via `circuit_data.py`, which does nothing but run
the committed AB/BA paradigm and slice it. The values are written to
`data/cascade_values.csv` so any number in the figure can be checked against the
file.

The visual system follows the manuscript figures: neutral gray and soft peach
mark the two stimulus windows; excitatory projections are always warm and
inhibitory projections are always cool; and frequent and rare conditions use
the established manuscript slate-blue and oxblood pair. Absolute excitatory
and inhibitory activity use restrained sequential ramps derived from the same
palette. Signed activity uses the standard symmetric, colorblind-safe
ColorBrewer PuOr scale. Panel letters and titles use the same physical
alignment as Figures 2 and 4.

## The contrast is identity controlled

Both rows show the **same physical stimulus** — an A tone, a 30 ms gap, a B
tone — on the **same two channels**. Only the context that preceded it differs:

| | AB frequent | AB rare |
|---|---|---|
| AB share of trials | 90% | 10% |
| W[B←A] learned | 0.212 | 0.000 |
| W[A←B] learned | 0.000 | 0.213 |

So anything that differs between the rows is attributable to the learned weight,
not to tone identity, adaptation or run drift. Rows B and C are exact mirror
images in their connectivity, which is why the thick arc sits above the
excitatory pair in one and below it in the other.

## The signed difference row

Panel D is computed directly as **AB frequent − AB rare** at the same four
prespecified times. It is not a second simulation and no value is selected for
display. Excitatory and inhibitory differences use separate symmetric,
zero-centred scales because their physical magnitudes differ by roughly an
order of magnitude. Thus node-fill colour sign has one meaning in both scales:

- orange: frequent > rare;
- purple: frequent < rare;
- near-white: no difference.

Neuron class remains independent of the signed fill: excitatory populations
are warm-outlined circles and inhibitory populations are blue-outlined rounded
squares. Recurrent E→E differences remain warm because the connection is
excitatory; sign is shown by warm shade, solid versus dashed style and the
printed signed value.

At 65 ms in the silent gap, ΔI_B = +0.181 a.u.; at 105 ms, the target response
has reversed sign, ΔE_B = −0.330 a.u. These are the mechanistic intermediate
and endpoint the raw positive-only rows make difficult to compare visually.

## The cascade, measured

| Step | AB frequent | AB rare |
|---|---|---|
| 1. The link, W[B←A] | **0.212** | 0.000 |
| 2. Pre-activation, peak E_B during tone A | **1.18** | 0.00 |
| 3. Standing inhibition, mean I_B in the gap | **0.379** | 0.208 |
| 4. Target response, peak E_B during tone B | **9.88** | 10.20 |

Read down: the learned link pre-activates B while A is still sounding; that
pre-activation drives B's own interneuron; because `tau_I` (80 ms) exceeds
`tau_E` (20 ms), the interneuron is still elevated through the silent gap; and
when B finally arrives it arrives into standing inhibition and is **3.1%**
smaller than when the link was never learned.

The 3.1% is small, and deliberately so — `model0` is calibrated to the ~5%
mismatch polarity reported in A1 (Ulanovsky et al. 2003; Nieto-Diego &
Malmierca 2016) rather than to a large effect.

## Two encoding decisions, stated because either could mislead

**Only plastic weights are mapped to line width.** The fixed E→I and I→E
matrices are identical in every snapshot and both conditions, so mapping them to
width would spend the most salient visual channel on a constant. They are drawn
at constant width and quoted in the key. Width therefore means exactly one
thing: what the network learned.

**Excitatory and inhibitory rates have separate colour scales.** They differ by
roughly an order of magnitude (E peaks near 10, I near 1.8). A shared scale would
render every interneuron black and hide the effect the figure exists to show.
Two colourbars are drawn and the figure states that they are not comparable.

## Regime note

The AB/BA task overrides the shared `model0` defaults — `w_IE_self = 3.0`,
`w_EI_self = 0.40`, `W_norm = 4.0` — for the reasons documented in
`tasks/ab_ba_model0/ab_ba.py`: the cascade needs W[B←A] near its asymptote
within a 400-trial horizon and a strong I→E delivery arm. This is a
task-specific regime and the manuscript's single-parameter-set claim has to
acknowledge it.

## Outputs

- `outputs/ab_ba_circuit_snapshots.{pdf,svg,png}`
- `data/ab_ba_circuit.npz` — the cached traces and weights.
- `data/cascade_values.csv` — the four cascade steps.
- `data/snapshot_differences.csv` — signed E, I and W differences at every
  displayed snapshot.
- `data/ab_ba_circuit_provenance.json` — configuration, settings hash, design.
