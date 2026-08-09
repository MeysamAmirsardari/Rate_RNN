# Figure 7 | Frozen-core perturbations reveal a coordinated but incomplete computational toolkit.

**A, Quantitative causal sensitivity map.** Four auditory computations were
rerun under one frozen layer-1 parameter set and four prespecified lesions.
Each cell reports the ablation effect, defined as `1 − E_lesion/E_intact`,
where `E` is that paradigm's primary effect. Thus 1 denotes complete loss, 0
denotes an intact-like effect, and negative values denote enhancement. Roving
novelty depended selectively on multiscale thalamocortical depression.
Directional prediction required learned recurrence and tone-selective
inhibition; removing depression enhanced rather than abolished its held-out
context effect. Coherent figure–ground connectivity required both depression
and learned recurrence but was practically preserved when inhibition was
spatially homogenized. Complete ordered word representations survived every
layer-1 lesion and collapsed only when the multiscale readout was replaced by
an otherwise matched single-rate readout. The single-rate intervention has no
defined output for the three layer-1-only paradigms, so those cells are marked
not applicable. `Δ`, lesion differs from intact after exact paired within-task
max-T correction and exceeds the ±20% smallest effect of interest; `≈`, paired
effect is equivalent to intact within that bound after Holm-corrected TOST.

**B, Paired session-level evidence.** Points show all 12 paired simulated
sessions, diamonds show means, and horizontal bars show paired-bootstrap 95%
confidence intervals. The gray band is the prespecified practical-equivalence
region, 0.8–1.2 of the intact effect. Roving effects are repetition-suppression
indices in the 0–180 ms variable-tone window, averaged over the three deviant
positions within each order seed. Directional effects compare the same
physical second-tone sequence after separate 90:10 and 10:90 probability-
context training, evaluated with learning disabled on one identical balanced
held-out stream. The SFG effect is final `W_FF − W_GG` for a ten-tone coherent
figure. The composition effect is the fraction of four words held completely
in temporal order. Normalization is descriptive and performed within task;
native effects are not compared inferentially across paradigms.

**C, No-retuning ABA– test.** Learned A–B cross-coupling relative to A/B self-
coupling increased with tone-slot rate but was numerically invariant across
all nonzero nominal channel separations (maximum separation-dependent spread,
`1.7 × 10⁻¹⁶`). The current one-hot input and distance-independent off-diagonal
connectivity make every separation a relabeling of every other separation.
Consequently this panel is a falsification test, not evidence for a perceptual
streaming boundary: a classic Δf-by-rate boundary requires an independently
specified spectral front end or distance-dependent connectivity.

**D, Mechanistic synthesis.** The paired interventions support a coordinated
sequence of operations—multiscale depression gates novelty, recurrent learning
forms dependency maps, selective inhibition expresses directional prediction,
and a downstream multiscale readout composes events. The ABA– result exposes
spectral-distance structure as the explicit missing computation, bounding the
claim to a coordinated toolkit rather than a complete unified account.

All layer-1 paradigms used the same pre-existing directional core
(`w_EI_self=0.40`, `w_IE_self=3.0`, `W_norm=4.0`; all remaining scalar values
at `model0` defaults); only the input dimensionality changed. Recurrent weights
began at zero. Uniform-inhibition controls separately preserved the intact E→I
and I→E row sums. Roving replicates vary constrained block order, AB/BA
replicates vary training/test order, SFG replicates vary cloud and coherent-
onset realizations with fixed figure channels, and Saffran replicates vary
exposure order and downstream initialization. Change tests used exact paired
sign flips with a maximum-studentized statistic across the prespecified lesion
family within each task; equivalence used paired TOST with Holm correction.

