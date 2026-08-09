# Figure 4 — stochastic figure-ground

Temporal coherence binds a set of tones into a figure that stands out from a
random cloud. The reference physiology is

> Lu, Dutta, Mohammed, Elhilali & Shamma (2025), *iScience* **28**:111991,
> "Temporal-coherence induces binding of responses to sound sequences in ferret
> auditory cortex" — `docs_rate_rnn/TC_sound_seq.pdf`.

**Every panel in this figure is model output.** The ferret data are cited in
the text and the caption, not reproduced here: *iScience* is CC-BY-NC-ND, so
re-laying-out their panels is not covered even with attribution.

## Build

```bash
python -m final_figures.figure_4.make_figure_4
```

`--force-data` repeats every simulated session. The cache key covers the
paradigm, the conditions *and* the reduction logic (`READOUT_VERSION`), so
changing how a quantity is measured forces a rerun.

## The measurement, and why it is built this way

Everything is a **tone-triggered average**: the mean over the 50 ms following
each of a channel's *own* pip onsets. A channel is driven by a 50 ms pip in the
cloud epochs exactly as it is by a figure chord, so the thalamic input entering
the average is matched and any difference is cortical.

This matters. An earlier version of this figure measured chord windows against a
whole-epoch baseline. Figure channels are silent for most of the figure epoch,
so that contrast compared *driven* against *mostly silent* and returned a
thalamic term of **+5.7 a.u.** — a restatement of the stimulus, which then
dominated the net drive (+5.96 intact against +5.92 with `W` ≡ 0) and buried the
cortical effect. Panel E now shows the thalamic term at exactly zero, which is
the check that the comparison is clean.

The causal contrast is **plasticity on versus off**:

    modulation (%) = 100 * (E_plastic - E_frozen) / E_frozen

on runs that share a seed and therefore an identical stimulus, one learning and
one with `W` held at zero. Short-term depression, adaptation drift and epoch
transients are the same in both runs and cancel exactly; what survives is the
learned recurrent weight.

## What the model reproduces

| Lu et al. (2025) | Model |
|---|---|
| Coherent tones become mutually enhanced (Fig. 1, 4) | Figure channels +3.63% during the figure epoch (10 tones) |
| Incoherent responses are competitively suppressed (Fig. 1, 2) | Ground channels −0.42% against their own cloud epochs |
| Enhancement grows with figure size, 4 → 10 tones (Fig. 4C, 4D) | +1.26, +2.11, +2.90, +3.63% for 4, 6, 8, 10 tones |
| Enhancement is rapid and builds up (Fig. 5A, 6C) | Builds over presentations, faster for bigger figures |
| The 4-tone figure produces little or no net enhancement (Fig. 4A, 4D) | Smallest figure gives the smallest effect, +1.26% |

## What the model does not reproduce, and where the axes differ

**Buildup timescale.** The ferret binding builds *within* one 5 s figure,
peaking around the mid-epoch (their Fig. 5A). This model's rate-STDP is slower:
the modulation is flat within a presentation and accumulates across them. The
buildup is therefore plotted on the axis where the model actually has it (panel
F) and the discrepancy is stated in the caption. It is a real difference in
timescale, not a difference in mechanism — both are Hebbian growth of recurrent
excitation among co-active channels.

**Persistent post-figure enhancement.** The paper's headline measure is
Δ<sub>post−pre</sub>, a within-trial step: STRFs measured after the figure are
enhanced relative to before it. This model shows **no such step** — the pre and
post cloud epochs are modulated equally (panel C). The learned weight does not
decay between epochs, so the trace it leaves is a standing figure-versus-ground
offset present in *both* cloud epochs (panel G), not a post-minus-pre
difference. Panel G reports that offset for what it is. Reproducing the step
would require a plasticity process with a decay constant between one epoch and
one presentation, which this model does not have.

**Inhibitory structure is irrelevant here.** Uniform and selective inhibition
give assembly drives that agree within 1.1% (panel D, dashed). The figure-ground
effect does not depend on inhibition being tone-selective — which is a negative
result worth stating, since the same control matters in Figure 2 and not here.

## Panels

| | Claim | Source |
|---|---|---|
| **A** | Figure and ground differ in temporal coherence, not drive | exemplar stimulus, plus the per-channel pip count as a drive check |
| **B** | Coherent co-firing strengthens the figure block | learned `W`, and the `W_FF`/`W_GG`/`W_FG` trajectory |
| **C** | Enhancement tracks the figure, on and off | 500 ms tone-triggered time course across pre/figure/post |
| **D** | Structure and its consequence both scale with figure size | assembly drive `(n−1)·W_FF`, and figure-epoch change |
| **E** | The enhancement is recurrent, not thalamic | tone-triggered Δcurrents, with the thalamic term at exactly zero |
| **F** | Binding accumulates over repeated exposures | modulation per presentation, one curve per figure size |
| **G** | A weaker trace survives into the cloud | cloud-epoch modulation vs figure size |

## Statements this figure must not make

- It must not claim the model reproduces Δ<sub>post−pre</sub>. It does not.
- It must not claim within-figure buildup dynamics. The model's buildup is
  across presentations.
- It must not claim selective inhibition is required. It is not.
- It must not present the thalamic current as evidence of anything except that
  the comparison is drive-matched.
- It must not describe ground channels as suppressed *relative to the
  no-plasticity control* — they are marginally enhanced there (+0.077 net
  drive), through cross weights. They are suppressed relative to their own
  cloud-epoch responses, which is the comparison the paper makes and the one
  panel D plots.
- It must not treat the rate matching as exact: figure channels receive 3.3%
  more pips per channel (panel A). The plastic-minus-frozen contrast is immune
  to this, since both runs see the identical stimulus, but a raw
  figure-versus-ground rate comparison would not be — and none is made.

## Regime note

`tasks/sfg2/sfg2.py` recalibrates three parameters for N = 37 — `W_max` 0.17,
`W_decay` 2e-3, and lateral inhibition rescaled to a disynaptic loop gain of
6.3. The reasons are in that module's header: model0's presets were tuned at
N = 2–5, where the recurrent assembly is automatically subcritical and the
inhibitory loop gain does not grow as N². This is a task-specific regime and the
manuscript's single-parameter-set claim has to acknowledge it.

## Outputs

- `outputs/figure_4_sfg.{pdf,svg,png}`
- `data/sfg_figure4_data.npz` — the reduced factorial.
- `data/sfg_figure4_runs.csv` — one row per simulated session.
- `data/figure_4_summary.csv` — every number quoted in the caption.
- `data/figure_4_drive_check.csv` — all 37 per-channel pip counts shown in
  Panel A.
- `data/figure_4_inference.{npz,csv}` — exact paired-seed tests and every
  corrected inferential mark drawn in panels C–G.
- `data/figure_4_inference_provenance.json` — test families, resampling unit
  and multiplicity control.
- `data/sfg_provenance.json`, `data/figure_4_provenance.json`.

## Inferential marks

All model inference uses the eight matched session seeds and exhaustive `2^8`
paired-seed sign flips. Panel C uses a two-sided temporal cluster-mass test.
Panels D–G use maximum-|t| correction within the explicitly stated panel
family. Panels A and B are single-session exemplars and remain descriptive;
stimulus channels are not promoted to independent replicates.
