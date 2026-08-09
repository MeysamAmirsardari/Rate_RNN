# Figure 4 — caption draft

## Caption

**Figure 4 | Temporal coherence binds a tone set into a figure through recurrent
excitation, and the effect scales with figure size.**

All panels are model output. The physiological result the model is compared
against is Lu *et al.* (2025) *iScience* **28**:111991, cited throughout and not
reproduced here.

**a**, One stochastic figure-ground presentation: 37 channels over three octaves,
5 s of random tone cloud, 5 s in which a subset of channels (green) is made
temporally coherent by a synchronous chord recurring at about 4 s⁻¹, and 5 s of
cloud again. The stimulus is rate matched — figure channels carry only the
coherent chord and ground channels an independent cloud at the same mean rate —
so the two sets differ primarily in *when* their tones occur, not in how many.
Right, the complete per-channel count distributions for the exemplar stimulus,
with diamonds and whiskers denoting mean ± s.e.m. (figure: 56.1 ± 0.9;
ground: 54.1 ± 0.5 pips per channel; difference 2.0 pips). Channels within
this single constructed stimulus are shown descriptively and are not treated as
independent inferential replicates.

**b**, Learned recurrent weights at the end of an exemplar session with a
10-tone figure, reordered so the figure block is top-left. The complete matrix
is shown, including learned self-connections on the diagonal (mean 0.0202).
Coherent co-firing grows the off-diagonal figure-to-figure block selectively:
W<sub>FF</sub> = 0.0112 against W<sub>GG</sub> = 0.0084 and
W<sub>FG</sub> = 0.0077, a figure-to-ground weight ratio of 1.34. Right, the
three blocks over the session.

**c**, Response modulation attributable to plasticity, in 500 ms bins across the
presentation. Each value is a **tone-triggered** average — the mean over the
50 ms following each of a channel's own tone onsets — expressed as
100 × (E<sub>plastic</sub> − E<sub>frozen</sub>)/E<sub>frozen</sub> on two runs
that share a seed and therefore an identical stimulus. Figure channels rise from
+3.0% during the cloud to +5.7% in the first 500 ms bin of the figure and a
+6.7% figure-epoch mean, and return to +2.8% in the first bin after it; ground
channels fall from +2.3% to +1.9%. Shading, s.e.m. over eight session seeds.
The bracket denotes a figure-minus-ground temporal cluster spanning the
complete 0–15-s presentation (*P*<sub>FWER</sub> = 0.0078).

**d**, Left, the assembly drive (*n* − 1)·W<sub>FF</sub>, the total recurrent
excitation one figure channel receives from its fellows, against figure size;
zero without plasticity, and unchanged when tone-selective inhibition is
replaced by uniform inhibition (dashed; the two agree to within 1.1%). Right,
the consequence: each channel set's figure-epoch modulation minus its own
cloud-epoch modulation. Figure channels +1.26, +2.11, +2.90 and +3.63% for
4-, 6-, 8- and 10-tone figures; ground channels −0.07, −0.16, −0.27 and −0.42%.
Both are monotonic in figure size and opposite in sign. This descriptive panel
contains no inferential marks.

**e**, Tone-triggered currents during the figure epoch, plastic minus frozen, for
the 10-tone figure. The thalamic term is exactly zero because the two runs
receive the identical stimulus — the check that the comparison is drive matched.
The enhancement is carried by recurrent excitation (+0.39 ± 0.01 for figure
channels against +0.20 for ground), partly offset by inhibition (+0.15 and
+0.12), leaving a net drive of +0.24 ± 0.01 against +0.08. Asterisks
denote non-thalamic currents different from zero after maximum-|*t*|
correction over the six displayed group-by-current tests (all
*P*<sub>FWER</sub> = 0.0078). The thalamic control is identically zero and was
not tested.

**f**, Figure-epoch modulation against presentation number, one curve per figure
size. Binding accumulates over exposures and does so faster for larger figures,
reaching +4.1% (4 tones) to +7.1% (10 tones) by the thirtieth presentation.
Endpoint asterisks denote significant per-seed linear buildup slopes,
maximum-|*t*| corrected over the four figure sizes (all
*P*<sub>FWER</sub> = 0.0078).

**g**, Modulation during the pre- and post-figure cloud epochs, when no chord is
present. A weaker, size-graded trace persists — figure channels +2.64 to +3.04%
and ground channels +2.48 to +2.31% across sizes — separating the two sets by
0.16 to 0.73 percentage points. Brackets and asterisks denote paired
figure-minus-ground contrasts, maximum-|*t*| corrected over the four sizes
(4 tones, *P*<sub>FWER</sub> = 0.031; 6–10 tones,
*P*<sub>FWER</sub> = 0.0078).

Mean ± s.e.m. over eight session seeds throughout; each seed redraws the ground
cloud and the figure's chord times. All inferential tests use the paired seed
as the unit and exhaustive `2^8` sign flips. Multiplicity is controlled within
each stated panel family by the maximum absolute *t* statistic or, for the
time course in **c**, the maximum two-sided cluster mass. Exemplar panels
**a** and **b** are descriptive and receive no inferential marks. Asterisks
encode corrected probabilities throughout: *P*<sub>FWER</sub> < 0.05 (*),
< 0.01 (**), and < 0.001 (***).

## Numbers quoted, and where they come from

Every value above is in `data/figure_4_summary.csv`; per-session values are in
`data/sfg_figure4_runs.csv`.

| Quantity | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Assembly drive (*n*−1)·W<sub>FF</sub> | 0.0355 | 0.0581 | 0.0798 | 0.1010 |
| Figure channels, figure epoch vs own cloud (%) | +1.26 | +2.11 | +2.90 | +3.63 |
| Ground channels, figure epoch vs own cloud (%) | −0.07 | −0.16 | −0.27 | −0.42 |
| Figure channels, cloud epochs (%) | +2.64 | +2.79 | +2.94 | +3.04 |
| Ground channels, cloud epochs (%) | +2.48 | +2.44 | +2.38 | +2.31 |

Currents, 10-tone figure, plastic minus frozen (a.u.):

| | Thalamic | Recurrent | Inhibition | Net |
|---|---|---|---|---|
| Figure | 0.000 | +0.389 | +0.154 | +0.236 |
| Ground | 0.000 | +0.199 | +0.122 | +0.077 |

## Statements this figure must not make

- **It does not reproduce Δ<sub>post−pre</sub>.** The paper's headline measure is
  a within-trial step, post-figure responses enhanced relative to pre-figure.
  Here the two cloud epochs are modulated equally (panel c): the learned weight
  does not decay between them, so what it leaves is a standing offset (panel g),
  not a step. Panel g must be described as a standing trace, never as a
  post-minus-pre enhancement.
- **The buildup axis is different.** Ferret binding builds within one 5 s figure
  and peaks near its middle; this model's modulation is flat within a
  presentation and accumulates across them (panel f). Same mechanism — Hebbian
  growth of recurrent excitation among co-active channels — on a slower
  timescale.
- **Ground channels are not suppressed relative to the frozen control.** They
  are marginally *enhanced* there (+0.077 net drive, panel e), through cross
  weights. They are suppressed relative to their own cloud-epoch responses,
  which is the comparison the paper makes and the one panel d plots.
- **Selective inhibition is not required.** Uniform inhibition gives the same
  assembly drive to within 1.1%. The figure-ground effect does not test
  inhibitory structure.
- **The thalamic bar in panel e is a control, not a result.** It is zero by
  construction and demonstrates only that the two runs share a stimulus.

## Known limitations

1. **N = 37 regime.** `tasks/sfg2/sfg2.py` recalibrates `W_max` (0.17),
   `W_decay` (2e-3) and the lateral inhibitory weights (disynaptic loop gain
   6.3) for 37 channels. model0's shared presets were tuned at N = 2–5, where
   the assembly is automatically subcritical and the inhibitory loop gain does
   not grow as N². This is a task-specific regime.
2. **Absolute effect size.** The modulations are a few per cent. The paper
   reports the proportion of *tests* with an enhanced response rising from about
   33% to 57% with figure size, which is not the same quantity and cannot be
   compared directly. The comparable claim is the monotonic size dependence and
   its sign, not the magnitude.
3. **Passive listening only.** The paper's experiment I turns on selective
   attention; there is no attentional mechanism in this model, so only
   experiment II is addressed.
4. **A residual drive difference.** Figure channels receive 3.3% more pips per
   channel than ground channels (panel a). The plastic-minus-frozen contrast is
   immune to this — both runs receive the identical stimulus — but the raw
   figure-versus-ground rate comparison would not be, and is not made anywhere
   in this figure.
