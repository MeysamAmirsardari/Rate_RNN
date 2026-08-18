# Figure 4 | Learned transition statistics generate sequence surprisal.

**a,** Space reserved for the final probability-reversal paradigm
illustration. **b,** Same-sequence ECoG responses for Experiment 1, the
continuous tone pair with 180-ms items and no within-pair gap. AB (5.3 to
9.4 kHz) and BA (9.4 to 5.3 kHz) are each held physically fixed and compared
when regular versus rare across the reversed probability contexts. Both runs
used the same planned probabilities (0.85 regular and 0.15 rare). Playback
marks were read from the recovered Baphy event tables; the six leading
`allM2` metadata columns were retained as tags and excluded from neural time.
Raw ECoG was filtered 1–250 Hz and baseline-SD normalized as in
`Gen_M2Mat.m`. Odd acquisition blocks selected the strongest contact in the
prespecified 180–360-ms second-tone window (AB, contact 10; BA, contact 29);
the complementary even blocks supplied the displayed mean ± s.e.m. over eight
held-out blocks per role. A symmetric zero-phase Gaussian kernel (sigma,
2 ms; full-width at half-maximum, approximately 4.7 ms) was applied only to
the drawn held-out block traces. Significance used unsmoothed block means and
an exact two-sided cluster-mass test over all `C(16,8)=12,870` assignments,
with the maximum cluster taken jointly over AB, BA and the full 0–600-ms
family. No cluster survived correction (smallest corrected probability: AB,
*P*<sub>FWER</sub> = 0.547; BA, *P*<sub>FWER</sub> = 0.975).
Gray and peach fields denote item 1 and item 2. **c,** Cross-session deviance
decoding in both directions. A ridge-logistic classifier was trained on one
recording and tested on the other, where the physical sequence assigned to
the rare role was reversed; the dashed line marks chance balanced accuracy
(0.5). No cluster survived correction.

**d,** Recurrent learning in the rate RNN. Each of 12 paired seeds learned
separate AB-rich and BA-rich contexts for 400 sequence pairs. The intact
ECoG-matched regime used `w_IE_self = 3.0`, `w_EI_self = 1.0`,
`tau_I = 0.250 s` and `W_norm = 4.0`. Curves show the frequent-transition,
rare-transition and mean self-weights at 25-pair checkpoints; lines and
shading denote mean +/- s.e.m. across paired seeds. Matrices show the
across-seed mean terminal recurrent weights (rows, postsynaptic population;
columns, presynaptic population). Both matrices use one common normalization,
and every cell reports its weight directly; no separate color scale is
required. Directional weight alignment was tested by an exhaustive two-sided
paired sign-flip test with max-|t| correction jointly over the two planned
contexts.

**e,** Synaptic origin of sequence surprisal. The first two axes show recurrent
excitatory and inhibitory drive to populations A and B; solid and dashed traces
denote AB and BA input order. The third axis shows the intact rare-minus-regular
population response, with positive and negative regions filled by context and
the violet curve and band denoting mean +/- s.e.m. across paired seeds. The rail
marks time points belonging to a corrected cluster common to both physical
sequence identities.

**f,** Spatial distribution of the discovery-half rare-regular ECoG response
magnitude for held-fixed AB and BA. Each map is normalized to its own maximum
on the verified 4 x 8 auditory-cortical electrode grid. The horizontal line
separates A1 (contacts 1-16) from PEG (contacts 17-32); white dots identify the
map maxima selected independently for the held-out ERPs in panel b.

**g,** Causal perturbations of the target-channel mismatch response. The
rare-minus-regular response is shown under four prespecified conditions: the
intact network, no short-term depression, frozen recurrent plasticity and
row-sum-matched uniform inhibition. Translucent points show individual paired
seeds; diamonds and vertical bars show means and 95% Student-*t* intervals.
Asterisks compare each perturbation with the intact network using exhaustive
two-sided paired sign flips and max-|t| family-wise correction over the three
planned contrasts (***, *P*<sub>FWER</sub> < 0.001; exact corrected probability
= 0.000488 for each contrast).

The paired training/test seed is the replication unit for every model test;
trials are averaged within seed and are never counted as independent
replicates. No statistical test compares ECoG with the model. The biological
panels contain one animal and are conditional on the two acquisition days, so
they do not support animal-population inference or a causal mechanism claim.
