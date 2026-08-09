# Figure 7 — Frozen-core causal synthesis

## Claim

**A coordinated set of mechanisms supports distinct computations within one
frozen core, while the present circuit lacks a spectral-distance axis.**

This figure is intentionally stricter than the task-specific figures. Roving,
AB/BA, stochastic figure-ground, and Saffran-style composition are rerun with
the same pre-existing directional core:

- `w_EI_self = 0.40`
- `w_IE_self = 3.0`
- `W_norm = 4.0`
- every other scalar parameter is the `model0.A1Config` default.

Only the number of input channels changes with the stimulus. These values were
already used by the AB/BA and Saffran analyses; they were not chosen from the
Figure-7 results. The SFG-specific recalibration and the original roving
configuration are not mixed into this synthesis.

## Panels

### A — Quantitative causal map

Rows are computations; columns are pure lesions. Every measured cell contains
`1 − condition / intact` for that row's prespecified primary effect:

- `1`: complete loss;
- `0`: intact-like;
- `<0`: enhancement after the lesion.

The downstream single-rate lesion is defined only for sequence composition.
The other three cells are marked `n/a`; they are not assigned a convenient
zero or one.

`Δ` means the paired lesion differs from intact after an exact within-task
max-T correction and exceeds the ±20% smallest effect of interest. `≈` means
the paired effect passes a Holm-corrected TOST inside that ±20% equivalence
region. A tiny but extremely consistent change can therefore be statistically
different and still practically equivalent; equivalence takes precedence in
the displayed status.

### B — Seed-level evidence

Every paired simulated session is visible. Diamonds are condition means and
bars are paired-bootstrap 95% confidence intervals for the ratio of condition
mean to intact mean. The gray band is the prespecified equivalence region,
`0.8–1.2`.

Native effects are never pooled across tasks:

| Task | Primary effect |
|---|---|
| Roving | repetition-suppression index, 0–180 ms variable-tone window, averaged across positions within seed |
| AB/BA | same physical second-tone response, unexpected minus predicted, after separate probability-context training and on an identical balanced held-out test stream |
| SFG | final `W_FF − W_GG`, ten-tone coherent figure |
| Saffran | fraction of four words held completely in temporal order |

### C — No-retuning ABA– falsification test

The heatmap is learned A–B cross-coupling divided by A/B self-coupling after
80 ABA– cycles. It is a model quantity, not a perceptual report. Rate matters,
but nominal A–B channel separation does not: the largest numerical spread over
the complete separation axis is about `1.7 × 10⁻¹⁶`.

That invariance is required by the current equations. Inputs are one-hot and
all off-diagonal recurrent and inhibitory connections have the same weight, so
every nonzero separation is a permutation of every other separation. A classic
van Noorden boundary cannot be claimed until a fixed spectral front end or a
distance-dependent connectivity rule is specified independently and rerun
without tuning on the sweep.

### D — Synthesis

The last panel maps the supported operations to the paradigms that identify
them. It deliberately says “coordinated toolkit,” not “one sufficient unified
mechanism.” The missing spectral-distance computation remains visible at full
size.

## Replication and inference

There are 12 paired simulated sessions per task.

- **Roving:** constrained block-order seed; the three designed deviant
  positions are averaged within seed.
- **AB/BA:** training/test order seed. Lesions and learned contexts receive the
  same balanced test sequence. Test learning is disabled, so no held-out event
  can alter the context weights.
- **SFG:** cloud and coherent-onset seed. The ten figure channels are fixed;
  they are not mislabeled as redrawn across seeds.
- **Saffran:** exposure-order and downstream-initialization seed.

Change tests are two-sided exact paired sign-flip tests with a maximum-
studentized statistic across all applicable lesions within each task.
Equivalence uses paired TOST with a ±20% retention bound and Holm correction
within task. Cross-task inferential comparisons are not performed.

“No recurrent learning” holds the zero-initialized recurrent matrix at zero.
It therefore tests the absence of learned recurrence as a composite, not the
acute removal of the plasticity rule after a learned matrix has formed.

Uniform inhibition separately preserves the intact E→I and I→E row sums at
each input dimension. It changes selectivity without changing either arm's
total weight.

## Build

```bash
python -m final_figures.figure_7.make_figure_7
```

Use `--force-data` to repeat the full 12-seed factorial and ABA– sweep. The data
cache key hashes the complete run specification. The accelerated simulator is
checked against `model0.model.simulate` before a new cache is written.

## Outputs

- `outputs/figure_7_causal_synthesis.pdf` — submission vector
- `outputs/figure_7_causal_synthesis.svg` — editable vector
- `outputs/figure_7_causal_synthesis.png` — 600 dpi review image
- `data/figure_7_frozen_core_data.npz` — complete figure data contract
- `data/figure_7_runs.csv` — one row per simulated task/condition/session
- `data/figure_7_summary.csv` — native-unit summaries
- `data/figure_7_inference.csv` — normalized effects and corrected inference
- `data/*provenance.json` — run specifications, source/output hashes, and
  replication definitions

