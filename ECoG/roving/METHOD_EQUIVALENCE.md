# Method equivalence and leakage audit

## Source scope

This package translates:

- `/Users/eminent/Projects/ECoG/decoding_analysis_xbc.m`
- `/Users/eminent/Projects/ECoG/decoding_analysis.m`
- `/Users/eminent/Projects/ECoG/decoding_aspen_08_26_abxd.m` (inspected and
  rejected as the position-3 source after resolving the actual recording)
- `/Users/eminent/Projects/ECoG/wave_analysis.m` and `pe_vs_pc.m` for explicit
  position-3 recording identification and extraction settings
- the SRH/SRV branch of `Gen_M2Mat.m` and `Gen_M2Mat_sp.m`
- `generate_full_mat_info_v2.m`

Only logic exercised by these three analyses is included. No outlier removal is
performed because the source passes empty arguments, which select
`outlier_removal=0`.

## Literal computational mapping

| MATLAB operation | Python operation |
|---|---|
| 100-ms per-epoch baseline subtraction and division by baseline `std` | `nanmean`, then sample SD (`ddof=1`) per channel |
| `erp_win = 101:901` | Python slice `100:901` (801 samples, 0–800 ms) |
| concatenate all ordered current/previous stimulus combinations | identical current-major, context-minor order |
| truncate each class to the smaller count within each context | identical |
| `rng(11); randperm(...)` | legacy MT19937 `RandomState(11)` permutations |
| add `randn * 1e-6` at every time | Gaussian noise with SD `1e-6` |
| `zscore(X_t)` | column sample z-score (`ddof=1`) |
| `fitclinear(...,'logistic','ridge','Lambda',1e-2)` | BFGS minimization of mean logistic deviance plus `lambda/2 * sum(beta^2)`, with unpenalized bias, 1e-6 gradient tolerance, 1e-4 relative-beta tolerance, and 1000-iteration limit |
| `KFold`, 5; `1-kfoldLoss` | stratified five-fold misclassification accuracy |
| `A = cov(X) * W` | sample covariance (`ddof=1`) times the full-data coefficient vector |
| `movmean(accuracy,20)` | even 20-sample MATLAB-centered moving mean with shrunken endpoints |
| first maximum; ±25 samples | identical first maximum and inclusive ±25-sample window |

The random-number streams and optimizer stopping path are not guaranteed to be
bit-for-bit identical across MATLAB and NumPy/SciPy. The objective, data
selection, scaling convention, folds, regularization strength, activation
pattern, smoothing, and selection rules are matched.

## Source issues preserved rather than silently corrected

1. Both Zaatar scripts label the late class “Rep 15” but set `rep_15 = 14`.
   The translation preserves numeric repetition 14 and records it in every
   provenance file.
2. The initially guessed Aspen script requests an absent recording and uses a
   two-stimulus, `Gen_M2Mat`, numeric-repetition-15 configuration. It is not the
   position-3 member of the Zaatar series. The resolved position-3 recording is
   `Zaatar_2024-11-27_xx.mat`: its three stimuli differ only in tone 3, and the
   explicit position-3 scripts use `Gen_M2Mat_sp`, `nStim=3`, and repetitions
   1 and 14.
3. `channelsToUse` is declared but unused in MATLAB. The helpers always use all
   32 channels; the translation does the same.
4. The source calls selected-channel ERPs “true drivers.” Decoder activation
   patterns are distributed information patterns, not causal channel effects.

## Leakage modes

`matlab-faithful` exists only for reproduction:

- z-scoring happens before cross-validation, so each test fold contributes to
  its training-fold means and standard deviations;
- repetitions from the same roving block can be divided between train and test;
- a decoder peak is selected and summarized on the same observations.

`leakage-safe` is the default manuscript analysis:

- the two decoded repetitions from a roving block remain in the same fold;
- z-scoring is fit on each training fold and applied to its test fold;
- one fixed split is reused across time;
- the spatial window is prespecified as 0–180 ms relative to deviant onset
  (configurable on the command line);
- the peak is saved as descriptive QC only, not as a valid inferentially
  selected latency.

The leakage-safe mode deliberately changes only operations required to prevent
information leakage and circular selection. It must not be described as
numerically identical to the MATLAB output.

## Inferential boundary

These scripts decode within individual recordings. Trials and channels are not
independent biological replicates. Manuscript confidence intervals and
corrected time-resolved inference must use independent animals or recordings.
This package therefore exports recording-level curves without manufacturing
trial-level confidence intervals or p-values.
