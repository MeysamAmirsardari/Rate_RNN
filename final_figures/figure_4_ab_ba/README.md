# Figure 4 - probability-reversal sequence oddball

## Scientific question

Figure 4 asks whether learned sequence probability changes the response to an
otherwise identical two-item sequence. The only biological and model
contrasts are:

- AB when rare versus the same AB sequence when regular; and
- BA when rare versus the same BA sequence when regular.

No panel treats AB versus BA as the decoding target. The design is called a
**probability-reversal sequence oddball** or, with an explicit qualifier, a
**local-global-like sequence paradigm**. It is not a fully factorial
local-global experiment because local and global factors were not
independently crossed.

The figure-level claim is: **learned transition statistics generate sequence
surprisal**. Formal surprisal is `-log2 P(sequence | context)`, in bits. The
model's *mismatch response* is a separate, physiological quantity: the
rare-minus-regular excitatory response to a held-fixed sequence.

## ECoG-matched model regime

The configuration is centralized in `tasks/ab_ba_model0/ab_ba.py`:

```python
AB_BA_OVERRIDES = dict(
    w_IE_self=3.0,
    w_EI_self=1.0,
    tau_I=0.250,
    W_norm=4.0,
)

P_REGULAR = 0.85
TIMING_LONG = dict(tone_dur=0.180, intra_gap=0.000, inter_gap=1.500)
TIMING_SHORT = dict(tone_dur=0.050, intra_gap=0.100, inter_gap=1.500)
```

Figure 4 displays and simulates only the long, zero-gap Experiment 1 regime.
The short timing remains documented for the gapped-tone experiment and is not
silently mixed into this figure.

## Build

From the repository root:

```bash
python -m final_figures.figure_4_ab_ba.make_figure_4
```

Use `--force-data` to rerun all paired model sessions, exact inferential
calculations and the retained non-displayed model decoder. Without this flag,
the cache is reused only when the complete analysis specification and source
hashes match.

The submission canvas is fixed at **183 x 225 mm**. Every time-resolved panel
uses the complete **0-600-ms sequence clock**: item 1 occupies 0-180 ms and
item 2 occupies 180-360 ms in the displayed long-timing regime.

## Panel map

- **A:** intentionally blank, with its panel tag and full-row allocation
  retained for the final paradigm illustration.
- **B:** recalculated Experiment-1 ECoG responses for held-fixed AB and
  held-fixed BA. Odd acquisition blocks select one contact per sequence;
  complementary even blocks provide the displayed mean, s.e.m. and jointly
  cluster-corrected inference.
- **C:** cross-session deviance decoding in both directions across the reversed
  physical-sequence mapping.
- **D:** AB-rich and BA-rich recurrent-weight trajectories with their final
  2 x 2 recurrent matrices. Matrix cells report their values directly, and
  both matrices use one common normalization without a separate colorbar.
- **E:** full-clock target-channel model responses for held-fixed AB
  (population B) and held-fixed BA (population A), with rare-minus-regular
  target-channel differences beneath. Ribbons are s.e.m. across 12 paired
  seeds; violet rails are jointly cluster-FWER-corrected across both physical
  sequences and the complete time family.
- **F:** a full-width vertical repeated-measures summary of the absolute
  target-channel mismatch under selective inhibition (intact), no short-term
  depression, no recurrent learning and uniform inhibition. All paired-seed
  observations remain visible; diamonds and vertical intervals show means
  and 95% Student-*t* intervals, and labels give exact lesion-versus-selective
  max-|t|-corrected *P*<sub>FWER</sub> values without stars.

## ECoG analysis contract

Only Experiment 1 is used: 180-ms tones, zero gap within the pair and a 1.5-s
between-sequence gap. The recovered Baphy playback tables provide all 400
binary rare/regular marks in each recording. Both runs specify the same design
probability, `P(regular)=0.85` and `P(rare)=0.15`; only the physical sequence
assigned to each role reverses.

The analysis follows the attached MATLAB data contract. Every reconstructed
observation has six leading `allM2` metadata fields—current run length,
previous run length, repetition number, current stimulus, previous stimulus
and acquisition block—and neural time begins only after those six fields. The
tags are exported separately and cannot enter the ERP clock. Raw ECoG is
filtered 1–250 Hz, then each epoch is centred and divided by its sample baseline
SD over −100 to 0 ms, matching `scripts_AB_BA.m` and `Gen_M2Mat.m`.

Panel B uses an explicit discovery/inference split. Odd acquisition blocks
select the contact with the largest RMS rare-minus-regular discovery effect in
the prespecified 180–360-ms second-tone window. Even blocks are untouched until
estimation and inference, yielding eight held-out block means per role. The
drawn curves and ribbons are mean ± s.e.m.; a symmetric 2-ms Gaussian is
applied to held-out block traces for display only. The test uses unsmoothed
block means and exact enumeration of all `C(16,8)=12,870` assignments, with a
single two-sided maximum-cluster family across AB, BA and the complete
0–600-ms clock. No corrected cluster survives in the present recordings.

At the experimenter's instruction, the two acquisition days are provisionally
treated as one session pending re-recording. Because probability role is still
implemented between recordings, this is conditional within-animal evidence;
it is not an independently replicated day effect or an animal-population claim.

## Model protocol and leakage controls

For each of 12 paired seeds, the analysis:

1. trains one network for 400 pairs with `P(AB)=0.85` and `P(BA)=0.15`;
2. trains a separate network for 400 pairs with `P(AB)=0.15` and
   `P(BA)=0.85`;
3. presents the identical balanced 120-pair held-out stream, in the same
   order, to both learned networks; and
4. disables plasticity throughout testing.

Thus AB-rare versus AB-regular and BA-rare versus BA-regular differ only in
learned probability context. Test acoustics, test order and perturbations are
paired within seed. Trial observations are averaged within seed before
inference; trials are never treated as independent replicates.

### Inferential families

- **D:** weight trajectories are mean +/- s.e.m. Final directional alignment
  uses exhaustive two-sided paired sign flips with max-|t| correction over
  the two prespecified contexts.
- **E:** held-fixed regular/rare target-channel response curves and their
  rare-minus-regular differences are mean +/- s.e.m. Significance rails come
  from exhaustive two-sided paired sign-flip cluster-mass tests, with one
  maximum-cluster family spanning AB, BA and all 601 time samples.
- **F:** faint lines preserve the four absolute condition estimates from each
  paired seed; diamonds and vertical bars are means and 95% Student-*t*
  intervals. The three perturbations are compared with selective inhibition
  by exhaustive two-sided paired sign flips with max-|t| correction over the
  planned lesion-versus-selective family. Values are printed directly; no
  stars are used.

A significance rail marks the extent of a corrected cluster, not an exact
physiological onset or offset. No statistical test pools ECoG and model units
or directly compares the two systems.

### Retained, non-displayed model decoder

The Figure 2-style model context decoder remains archived and contract-tested
but is not a Figure 4 panel. Ridge logistic regression (`lambda = 0.01`) is fit
separately at each 5-ms sample from 0 to 600 ms using the two excitatory
population activities. Rare and regular trials are averaged within paired
seed; five fixed folds keep both endpoints from a seed together, while
centering and sample-SD scaling are learned from training folds only. Curves
are unsmoothed balanced accuracy, intervals use 4,999 paired-seed bootstrap
resamples, and significance uses all `2^12 = 4,096` full-refit within-seed
label swaps with one one-sided maximum-cluster family over AB, BA and all time
samples.

## Outputs

### Figure files

- `outputs/figure_4_ab_ba.pdf`
- `outputs/figure_4_ab_ba.svg`
- `outputs/figure_4_ab_ba.png`
- `outputs/figure_4_metadata.json`

### ECoG panel data

- `ECoG/ab_ba/results/ab_ba_channel_erp/ab_ba_channel_erp.npz` - selected
  contacts, raw held-out block traces, displayed means/s.e.m., test statistics,
  corrected probabilities and significance masks.
- `ECoG/ab_ba/results/ab_ba_channel_erp/timecourses.csv.gz` - tidy Panel-B
  figure data.
- `ECoG/ab_ba/results/ab_ba_channel_erp/clusters.csv` - every observed cluster,
  including non-significant clusters and exact corrected probabilities.
- `ECoG/ab_ba/results/ab_ba_channel_erp/allm2_tags.csv.gz` - the reconstructed
  six leading MATLAB metadata fields plus the binary deviant mark; no neural
  time samples are stored in this table.
- `ECoG/ab_ba/results/ab_ba_channel_erp/provenance.json` - source paths and
  hashes, preprocessing, split, smoothing and joint inferential family.
- `data/ecog_exp1_legacy_panel_data.csv.gz` - retained historical source-figure
  extraction; it is no longer used by Panel B.

### Model data and inference

- `data/model_figure4_ab_ba.npz` - complete cached model arrays.
- `data/model_figure4_runs.csv` - seed-level item-2 response summaries.
- `data/model_figure4_weight_trajectory.csv` - all displayed checkpoints and
  recurrent weights.
- `data/model_figure4_surprisal_response.csv` - retained non-displayed formal
  surprisal values with seed-level target-channel and all-channel response
  summaries.
- `data/model_figure4_surprise_timecourses.csv` - target-channel and
  all-channel rare-minus-regular time courses, corrected cluster
  probabilities and significance masks.
- `data/model_figure4_mechanism_timecourses.csv` - retained non-displayed
  mechanism-difference time courses.
- `data/figure_4_summary.csv` - condition-level effect estimates and 95%
  intervals.
- `data/figure_4_inference.csv` - exact max-|t|-corrected scalar tests for the
  condition, perturbation and directional-weight families.
- `data/model_figure4_context_decoder.csv` - retained non-displayed model
  decoder, interval and corrected cluster results.
- `data/model_figure4_provenance.json` - full specification, software
  versions, source hashes, simulator self-test and output hashes.

`final_figures/figure_4/` contains the separate stochastic figure-ground
figure and is intentionally untouched.
