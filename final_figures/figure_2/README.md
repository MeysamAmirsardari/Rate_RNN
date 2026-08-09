# Figure 2 — Roving novelty responses

## Scientific role

Figure 2 connects the leakage-safe ECoG signature of repetition-dependent
novelty to the rate-RNN mechanism. It is deliberately organized as:

1. a full-width paradigm placeholder;
2. separate position-level empirical responses and decoder curves;
3. three position-level posterior maps;
4. verified A1/PEG patterns and position-matched model dynamics;
5. empirical and model repetition-dependent change; and
6. a vertical causal-perturbation summary.

Panel A reserves the final footprint but contains no paradigm drawing, as
requested.

## Build

From the repository root:

```bash
python -m final_figures.figure_2.make_figure_2
```

Use `--force-ecog` to repeat the endpoint extraction from the original `.mat`
files, `--force-model` to repeat the model sessions, and `--force-inference` to
repeat the bootstrap/permutation analyses. Without these flags, validated
figure data in `data/` are reused.

## Non-negotiable analysis safeguards

- ECoG endpoint responses use **actual Rep 1 and Rep 15**, not the numeric Rep
  14 used in the legacy decoder exports.
- All decoder/posterior results come from the leakage-safe regression analysis:
  whole roving blocks are held out and standardization is learned on training
  folds only.
- Decoder significance refits the complete grouped decoder after each
  within-block endpoint-label swap and uses one cluster maximum across all
  three recordings and time.
- Posterior contours use a separately identified cross-fitted block-level
  cluster test over repetition and time; they are not inferred from the
  display mean.
- The three ECoG recordings are shown as three deviant-position recordings from
  one animal. They do not constitute `n = 3` animals.
- No uncorrected pointwise significance marks are shown.
- Electrode maps use the verified 8 × 4 serpentine layout, remain separate
  across recordings, and mark the A1 (channels 1–16) / PEG (17–32) boundary.
  They are estimated only from a context-stratified discovery half. Small
  white dots identify the map maxima for positions 1–3 (MATLAB channels 27,
  12 and 18), which are used to estimate ERPs only in the complementary intact
  blocks; they are not causal channel-importance claims.
- Panel B ERP means and SEMs use only held-out intact blocks. Symmetric
  zero-phase Gaussian smoothing (sigma 2 ms) is applied to individual full
  traces for display only. Its cluster-mass test uses unsmoothed paired block
  differences and controls family-wise error jointly over time and all three
  recordings.
- The original all-contact GFP analysis is preserved as Supplementary Fig. 1
  in `../sup1/`.
- Model seeds are labelled simulated session/order realizations, not independent
  network initializations.
- Designed positions are averaged within each seed before model SEM or
  inferential tests; position × seed combinations are not counted as
  independent observations.
- ECoG buildup in panel G is derived directly from blockwise posterior-map
  values in the position-specific variable-tone window. Model buildup is
  percentage change from Rep 1; endpoint anchoring that mechanically forces
  Rep 15 to zero is not used.
- The uniform-inhibition perturbation is row-sum matched to the intact model
  (`E→I = 0.08` and `I→E = 0.29` per entry for five channels).

## Outputs

- `outputs/figure_2_roving.pdf` — submission vector figure.
- `outputs/figure_2_roving.svg` — editable vector figure.
- `outputs/figure_2_roving.png` — 600 dpi review image.
- `data/*.csv` and `data/*.npz` — panel-level figure data.
- `data/*provenance.json` — exact source and computation metadata.
