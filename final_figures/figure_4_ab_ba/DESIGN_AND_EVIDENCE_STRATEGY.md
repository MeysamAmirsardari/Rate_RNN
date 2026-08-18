# Figure 4 design and evidence strategy

## Editorial sentence

The page should be understandable before the caption is opened:

**probability reversal -> same-sequence ECoG response and decoding -> reversed
transition weights -> synaptic origin of sequence surprisal -> spatial recording
context and causal requirements.**

The critical comparison never changes physical sequence identity. It is AB
rare versus AB regular and, separately, BA rare versus BA regular.

## Naming and interpretation

- Paradigm: **probability-reversal sequence oddball**.
- Acceptable qualifier: **local-global-like sequence paradigm**.
- Figure claim: **learned transition statistics generate sequence
  surprisal**.
- Formal surprisal: `-log2 P(sequence | context)`, in bits.
- Model mismatch response: rare-minus-regular excitatory activity.
- Empirical terminology: **same-sequence response** and **same-sequence
  context decoding**, not mismatch negativity.
- Temporal labels: **item 1** and **item 2** only.

The experiment is not called a fully factorial local-global paradigm because
local and global factors were not independently orthogonalized.

## Model regime represented on the page

- Regular and rare probabilities are exactly 0.85 and 0.15.
- Intact overrides are `w_IE_self=3.0`, `w_EI_self=1.0`, `tau_I=0.250 s` and
  `W_norm=4.0`.
- Long timing is 180-ms item, 0-ms within-pair gap and 1.5-s inter-pair gap.
- Short timing is 50-ms item, 100-ms within-pair gap and 1.5-s inter-pair gap.
- The figure displays and simulates only the long, zero-gap Experiment 1
  regime. The short timing is retained in the specification for provenance,
  not pooled into the figure.

## Page architecture

The submission page is fixed at **183 x 245 mm**, exactly matching Figure 2,
and uses five editorial rows
on a 12-column grid.

1. **A, full width:** intentionally blank for the final paradigm illustration;
   only the aligned panel tag is retained.
2. **B-C:** B uses five columns and C uses six, reproducing Figure 2's empirical
   balance. Each contains two aligned Experiment 1 facets, held-fixed AB and
   held-fixed BA. B shows independently selected, held-out block ERPs; C shows
   cross-session deviance decoding.
3. **D, full width:** two trajectory-to-matrix units. AB-rich and BA-rich
   learning histories sit immediately beside their terminal 2 x 2 recurrent
   matrices. Every matrix cell is labelled directly, and both matrices use one
   common normalization without a separate colorbar.
4. **E, full width:** recurrent excitation, inhibition and sequence surprisal
   form three aligned mechanism axes. This is the principal model result and
   therefore receives the complete editorial row.
5. **F-G:** F reproduces Figure 2's spatial ECoG treatment with two 4 x 8 maps,
   an A1/PEG boundary and white marks at the independently selected ERP
   contacts. G uses Figure 2's compact causal-panel footprint, with seed
   points, means, 95% intervals and corrected asterisks.

This allocation gives the mechanistic surprisal result visual priority while
preserving Figure 2's compact spatial and causal summaries in the closing row.

## Visual grammar

- rare: deep oxblood (`#7C102A`);
- regular: deep slate blue (`#2166AC`);
- decoder and primary mismatch structure: matte violet (`#685994`);
- additional model mechanisms and perturbations: harmonized terracotta, blue
  and teal accents;
- item 1: neutral light gray;
- item 2: soft peach (`#FDDBC7`);
- axes, headings and annotation text: dark charcoal.

Role colors repeat across response panels. Violet remains reserved for decoding and
the principal mismatch trajectory, separating representational readout from
response magnitude. D uses direct structural labels, common matrix
normalization and printed cell values rather than a redundant weight
colorbar. G combines categorical position, translucent seed points, mean
markers and intervals, so no conclusion depends on color alone. Large fields
remain neutral; green is not used as a stimulus background.

Panel headings are concise, title-only statements. Analysis units, intervals,
smoothing and inferential status belong in the caption and methods, not in
small gray subtitles beneath headings. All labels must clear plotted data,
cluster rails and neighboring panels at final 183-mm width.

Figure 4 inherits Figure 2's header geometry directly: each panel letter and
title share one physical baseline, with a fixed point-based gap that is
independent of subplot height. Facet labels are short, sentence-case labels;
dates use typographic arrows, differences use plain language, and exact
inferential detail is kept out of the plotting field.

## Evidence hierarchy visible on the page

### ECoG: Experiment 1, held out and jointly corrected

1. B and C display only the first continuous-tone experiment: two 180-ms
   items, no within-pair gap.
2. Both panels hold physical sequence fixed. AB and BA are never pooled or
   contrasted as decoder classes.
3. The recovered Baphy playback table supplies the binary stimulus marks. The
   six `allM2` tag fields are retained separately and neural time starts only
   after them.
4. B uses odd acquisition blocks for contact selection and even blocks for
   estimation and inference; selection never sees the displayed data.
5. B applies a fixed zero-phase Gaussian kernel (sigma = 2 ms) only to the
   displayed held-out block traces. Unsmoothed block means enter inference.
6. Exact two-sided cluster-mass inference enumerates all `C(16,8)=12,870`
   assignments and uses one maximum-cluster family over AB, BA and 0–600 ms.
7. F maps the discovery-half response magnitude over the verified A1/PEG grid
   and marks the independently selected contacts used in B.
8. Both recordings used the same planned 85/15 probability. Contextual role
   remains assigned between recordings, so the provisional same-session test
   is conditional on the experimenter's assumption and this one animal.

### Model: paired, held out and jointly corrected

1. Each seed learns AB-rich and BA-rich contexts in separate simulations.
2. Both networks receive the identical balanced held-out test stream in the
   identical order, with plasticity disabled.
3. Comparisons are within paired seed and within physical sequence.
4. Trials are averaged within seed before inference; paired seeds, not trials,
   are the replication units.
5. D shows the complete prespecified 0-400-pair learning history at 25-pair
   checkpoints and all entries of each terminal recurrent matrix.
6. E displays recurrent excitation and inhibition for both population channels
   together with the intact rare-minus-regular population response. The exact
   cluster family spans both physical sequence identities and all 601 samples.
7. G shows the four absolute condition effects, so the intact selective-
   inhibition estimate remains visible rather than being reduced to a zero
   reference. Seed points remain visible and mean diamonds carry 95% intervals.
8. Asterisks in G use exact max-|t| correction over the three planned
   lesion-versus-intact contrasts; no bracket forest is added.
9. Native ECoG and model units remain separate. No cross-system
   normalization, pooled uncertainty or ECoG-versus-model test is used.

The leakage-safe Figure 2-style model decoder remains archived and
contract-tested but is deliberately not displayed. It retains grouped folds,
training-fold-only standardization, unsmoothed balanced accuracy, 4,999
paired-seed bootstrap resamples, exhaustive full-refit within-seed label swaps
and a single maximum-cluster family over held-fixed AB, held-fixed BA and time.

## Inferential display rules

- Model weight and response time courses are mean +/- s.e.m. across 12 paired
  seeds.
- Model time-resolved significance uses exhaustive two-sided paired sign
  flips, cluster mass and joint maximum-cluster FWER control over the complete
  prespecified family.
- A cluster rail denotes corrected cluster extent, not an exact biological
  onset or offset.
- Scalar markers show means and 95% Student-*t* intervals; translucent points
  or paired lines preserve all seed-level estimates.
- Scalar probabilities are exact, family-wise max-|t| corrected and printed
  as *P*<sub>FWER</sub>. Star notation is not used.
- ECoG ribbons are s.e.m. over held-out acquisition blocks; ECoG cluster rails
  use the jointly corrected unsmoothed block test. If no cluster survives, no
  rail is drawn and the facet is marked `cluster-FWER n.s.`.

## Empirical update rule

The current playback reconstruction is complete for Experiment 1. Future
re-recording should retain the same six-tag-plus-neural-samples contract and
interleave probability assignments within a recording. Panel B can then keep
the same discovery/inference split and joint cluster family while removing the
remaining between-recording context confound. The 0–600-ms clock, palette and
model panels should remain unchanged.
