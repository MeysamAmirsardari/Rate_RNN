# Manuscript visualization strategy

## Editorial objective

The figure should make the scientific argument legible before the caption is
read:

**paradigm → empirical response → decodable repetition state → posterior
geometry → distributed anatomy → matched model response → buildup and causal
perturbation**.

The system is restrained enough for a Nature-family article but warmer than a
default plotting package. White space, direct labels, repeated visual grammar,
and displays of the actual sampling units take priority over decorative
density.

## Figure 2 composition

Figure 2 is exported at a fixed **183 × 245 mm** page size.

1. **Row 0 — panel A:** the entire row is blank, apart from its panel letter
   and title, for the final roving-paradigm illustration.
2. **Row 1 — panels B and C:** the ECoG Rep-1/Rep-15 response and leakage-safe
   decoder. Positions 1–3 occupy three separate vertical levels.
3. **Row 2 — panel D:** three separate posterior maps on one common scale.
4. **Row 3 — panels E and F:** verified ECoG electrode patterns and the
   position-matched intact rate-RNN response. The RNN uses the same position
   order and 0–600 ms sequence clock as the decoder.
5. **Row 4 — panels G and H:** repetition-dependent change with uncertainty
   and corrected inference, followed by a vertical perturbation summary.

Decoder and model traces use absolute sequence time, **0–600 ms**, with ticks at
0, 180, 360, 540, and 600 ms. Shared tones are neutral gray and the position-specific
variable tone is pale peach. The sequence order is X–A–B, A–X–B, or A–B–X for
positions 1, 2, or 3. Position panels share a y-scale within each measurement.

## Master palette

| Role | Name | Hex |
|---|---|---|
| Novel / Rep 1 / reset | Deep Oxblood | `#7C102A` |
| Adapted / Rep 15 | Deep Slate Blue | `#2166AC` |
| Decoder / posterior | Matte Violet | `#685994` |
| Intact model | Muted Pine | `#2C6E5A` |
| Deviant tone | Soft Peach | `#FDDBC7` |
| Shared tone | Muted Sage | `#CCEBC5` |
| Main text and axes | Dark Charcoal | `#2D3748` |
| Secondary structure | Ash Grey | `#999999` |
| No depression | Slate Grey | `#7E8792` |
| Plasticity frozen | Deep Slate Blue | `#2166AC` |
| Uniform inhibition | Faded Terracotta | `#BD6B6B` |

Color is semantic, not decorative. Rep 1 is always oxblood, Rep 15 always slate
blue, empirical decoding is violet, and an intact model is pine. Labels,
marker shapes, and line styles keep interpretation from depending on color
alone.

## Evidence and uncertainty policy

Uncertainty follows the sampling unit.

- The three ECoG datasets are three deviant-position recordings from **one
  animal (Zaatar)**. They are never treated as three animals. ECoG uncertainty
  and tests are within-animal, whole-block analyses.
- Decoder folds keep a complete roving block together, reuse folds across
  time, and estimate scaling from training blocks only.
- Decoder uncertainty uses paired, transition-stratified whole-block
  resampling and is conditional on the fitted out-of-fold decoders. Decoder
  significance swaps Rep-1/Rep-15 labels within blocks, refits every fold and
  all training-only operations for every permutation, and uses one maximum
  cluster statistic across all positions and time.
- Decoder inference is performed on a prespecified 5-ms grid with 4,999
  permutations. Bars mark a corrected temporal cluster; they do not claim the
  exact onset or that every sample in the bar is individually significant.
- Posterior contours start from unsmoothed, out-of-fold block logits. The same
  Gaussian smoothing kernel used by the display is applied to each blockwise
  logit surface, whole surfaces are sign-flipped, and four-neighbour
  repetition-by-time clusters are corrected jointly over the three maps. This
  conditional test assumes sign symmetry of frozen cross-fitted logit surfaces
  and is separate from the full-refit endpoint decoder test. A contour marks a
  corrected cluster, not cellwise significance or an exact boundary.
- The panel-G ECoG curves are derived directly from the posterior maps:
  blockwise out-of-fold Rep-1 posterior is averaged over each position's
  variable-tone window. Uncertainty is SEM over blocks; filled markers denote
  conditional, jointly corrected block-logit tests against neutral evidence.
- Model uncertainty is calculated across the eight order seeds only after
  averaging the three designed deviant positions within each seed. The 24
  position-by-seed combinations are not counted as independent replicates.
- Model significance in panel G uses exact paired seed sign flips with joint
  correction and is encoded by marker fill rather than additional rails.
- Perturbation markers in panel H test paired seed-level differences from the
  intact condition, jointly corrected over the three planned contrasts.
- No significance test is attached to visual ECoG–model agreement.

Panel G uses the interpretable endpoint-free metric
`100 × (response at repetition r / response at Rep 1 − 1)`. The earlier
endpoint-anchored 1-to-0 normalization was retired because it forces Rep 15 to
zero even when a perturbation causes facilitation.

## Map policy

- Posterior maps use a perceptually ordered diverging scale centered exactly at
  posterior 0.5:
  `#2166AC → #B8CCE0 → #F4F0E8 → #D98C75 → #7C102A`.
- The three posterior maps share symmetric limits selected once from the full
  set of maps.
- Significant posterior clusters use a linen/white halo under a fine charcoal
  contour so the boundary remains visible at both color extremes.
- Electrode maps use the verified 8 × 4 serpentine grid and the sequential map
  `#F5F2EC → #D4CCE2 → #9B88BC → #685994 → #33284E`.
- A1 is channels 1–16 and PEG is channels 17–32. Their boundary is drawn
  between grid rows 4 and 5 (Matplotlib image coordinate `y = 3.5`) with a
  white halo and charcoal core.
- Haufe patterns indicate distributed discriminative information, not causal
  electrode importance.

## Typography and export

- Sans serif: Arial/Helvetica with DejaVu Sans fallback.
- Panel letters: 10–11 pt, bold.
- Group titles: 8.2 pt, semibold.
- Axis labels: about 7 pt; ticks and compact annotations at least 5.5–6.7 pt.
- Primary traces: 1.1–1.3 pt at final size.
- Uncertainty fields: 14–15% opacity, with no boundary stroke.
- Error bars are thin and capped.
- Axes are unboxed; only necessary left and bottom spines remain.
- PDF and SVG text remains editable. The review PNG is 600 dpi.
- Export bounds are fixed rather than content-tight, preventing panel letters
  or annotations from changing the submission page size.

Every final PDF is rendered back to a high-resolution image and inspected for
clipping, crowding, overlap, legibility, color balance, anatomical orientation,
and consistency between plotted claims and saved inferential data.

## Figure 5 composition

Figure 5 uses the same fixed 183 × 245 mm page and tells one continuous causal
story: statistical stream → word-selective activity → learned multiscale
templates → metric dissociation → pairwise and temporal mechanisms → controls
and synthesis.

The learned-mask panel does not devote most of the page to thirteen empty
current-channel rows. For each exemplar unit, it takes the empirically measured
driving row of the complete 14 × 84 mask and reshapes all 84 weights, without
averaging, into context token × filter timescale. Outlined cells identify the
strongest filter carrying each earlier word token. This is a representational
view of the mask, not a new analysis or a time axis.

Condition color is stable throughout Figure 5: single-timescale is slate gray,
multiscale is manuscript violet, and scrambled exposure is faded terracotta.
Word identity uses four muted categorical hues only in the stream, activity,
and template annotations. Seed-level points remain visible beneath a stronger
mean ± SEM mark; marker shape duplicates color identity. Pairwise weights and
mask strength use one sequential linen-to-violet map so color never implies a
signed effect.

The synthesis panel places ordered composition on the horizontal axis and
familiarity on the vertical axis. This makes the central dissociation spatial:
multiscale structure moves right, while scrambled exposure moves below the
horizontal chance boundary.

## Figure 7 causal synthesis

Figure 7 is the filled causal roadmap. It does not inherit the model/data band
used by experimental figures because it is a model-only synthesis. The page is
organized as evidence density: quantitative map, paired evidence, explicit
generalization test, then mechanism synthesis.

The map encodes **ablation effect**, `1 − condition/intact`, rather than a
binary “required/not needed” judgment. Zero is exactly white, positive loss
runs through faded terracotta to oxblood, and enhancement runs toward slate
blue. Every cell retains a printed number and a statistical/equivalence glyph,
so neither hue nor significance alone carries the claim. Undefined downstream
lesions are hatched ash and labeled `n/a`.

Panel B separates statistical difference from practical equivalence. Every
paired session is plotted; condition means use diamonds with paired-bootstrap
95% intervals. The prespecified ±20% equivalence region is a neutral gray band.
Lesion colors are stable: depression is terracotta, recurrent learning is
violet, inhibition is teal, and the downstream readout is slate blue.

The ABA– sweep uses one sequential linen-to-violet scale because it encodes one
unsigned coupling quantity. Its complete lack of Δf variation is the result,
not a plotting failure. The adjacent permutation schematic explains why with
equal off-diagonal weights, and the caption explicitly forbids a perceptual-
streaming claim.

The final mechanism boxes use neutral fills and semantic colored edges. They
end with a direct boundary statement: the frozen core supports a coordinated
toolkit, while spectral-distance structure remains missing. This negative
boundary is given the same visual weight as the positive dissociations.
