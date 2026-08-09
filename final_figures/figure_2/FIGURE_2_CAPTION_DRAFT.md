# Figure 2 | Multiscale synaptic depression captures key roving novelty-response dynamics.

**a,** Reserved space for the roving-paradigm illustration. **b,**
Baseline-normalized ECoG event-related potentials (ERPs) at the independently
selected strongest contact in each Panel **e** map, shown for the first and
fifteenth presentation of a roving block and aligned to variable-tone onset.
Within every ordered previous/current stimulus stratum, alternating intact
blocks formed a discovery set used only to estimate Panel **e** and select its
maximum; the complementary blocks estimated and tested the ERP (discovery/test
blocks: 65/60, 64/61 and 64/61 for positions 1-3). Lines and shading denote
mean +/- s.e.m. across held-out blocks. For display only, each full block trace
was convolved with a symmetric zero-phase Gaussian kernel (sigma = 2 ms; FWHM
approximately 4.7 ms) before averaging and cropping; all inference used the
unsmoothed held-out block traces. The peach field denotes the prespecified
0-180-ms variable-tone window. Bars denote latencies where the unsmoothed
paired Rep-1 minus Rep-15 contrast survived a two-sided whole-block sign-flip
cluster-mass test, with family-wise correction applied jointly over all
latencies and all three recordings (4,999 randomizations; teal bars denote
surviving clusters). Positions 1 and 2 had no surviving cluster; position 3 showed a
Rep-15-greater-than-Rep-1 cluster at 48-107 ms
(*P*<sub>FWER</sub> = 0.0068). **c,** Leakage-safe Rep-1-versus-Rep-15 decoding,
quantified as cross-validated balanced accuracy. Complete roving blocks were
held out, scaling was estimated only from training blocks, and the same folds
were retained over time. Shading denotes a paired, transition-stratified
whole-block bootstrap 95% percentile interval conditional on the fitted
out-of-fold decoders. Bars denote temporal clusters surviving a one-sided
above-chance, full-refit, whole-block label-swap test with joint family-wise
correction over positions and time (`4,999` permutations; 5-ms inference
grid). Tone orders are X–B–C, A–X–C, and A–B–X for positions 1–3,
respectively. The dashed line denotes chance. **d,** Rep-1 posterior probability over
repetitions 1–15 and sequence time for each position recording. Maps share a
posterior scale centered at 0.5. Contours denote clusters surviving a
two-sided, four-neighbour repetition-by-time test corrected jointly over all
three maps. This map-level test starts from unsmoothed blockwise logits,
applies the prespecified Gaussian kernel to each block surface, and then
performs conditional whole-surface sign flips. It assumes sign symmetry of
frozen cross-fitted logit surfaces and is not a full-refit decoder
permutation. Contours establish corrected clusters, not cellwise significance
or exact cluster boundaries. **e,** Absolute Haufe activation patterns in the
prespecified 0-180-ms variable-tone window, estimated only from the discovery
blocks, normalized within recording and shown on the verified physical 8 x 4
electrode grid. The line separates A1 (channels 1-16) from PEG (channels
17-32). The small white dot marks the independently selected map maximum used
for the held-out ERP in **b** (MATLAB channels 27, 12 and 18 for positions 1-3,
respectively). Patterns represent distributed discriminative information, not
causal channel importance. **f,** Excitatory
response of the intact rate RNN for Rep 1 and Rep 15, shown separately for the
same three positions and 0–600 ms sequence clock as in **c**. Lines and
shading show the mean ± s.e.m. across eight paired session-order seeds.
**g,** ECoG Rep-1 posterior probability and rate-RNN response across
repetitions. ECoG curves are blockwise out-of-fold posteriors averaged over the
position-specific 0–180 ms variable-tone window; model curves are percentage
response change from Rep 1 for the four perturbation conditions. Error bars
denote s.e.m. over blocks within an ECoG recording, or over eight paired order
seeds after averaging the three designed positions within each seed. Filled
ECoG markers denote conditional, two-sided max-|t| whole-block logit sign-flip
tests against neutral evidence, corrected jointly over positions and
repetitions. Filled model markers denote exhaustive `2^8` paired-seed sign
flips corrected jointly over all four conditions and repetitions 2–15. No test
compares ECoG with the model. **h,** Suppression index,
`(Rep 1 − Rep 15) / Rep 1`, for the intact model and causal perturbations. Dots
are position-averaged order seeds, diamonds are means, and vertical bars are
two-sided 95% Student-t intervals across the eight seeds; filled markers at
the top denote jointly corrected, exact paired-seed contrasts with the intact
model. The uniform-inhibition control preserves the intact E-to-I and I-to-E
row sums.

All ECoG inference is within-animal and block-level. These recordings do not
provide animal-population inference. Model seeds vary constrained roving-block
order while sharing the committed zero recurrent-weight initialization. The
corresponding 32-contact global field power analysis is shown in Supplementary
Fig. 1.
