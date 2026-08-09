# Method equivalence and leakage audit

## Source scope

The translation is based on:

- `/Users/eminent/Projects/ECoG/AB_BA/scripts_AB_BA.m`
- `/Users/eminent/Projects/ECoG/Gen_M2Mat.m` (the `SEQ` branch)
- four legacy Open Ephys recordings in
  `AB_BA/Nutmeg_2026-04-30_2026-05-01_SEQ1_3`
- six supplied MATLAB `.fig` reference outputs.

The source calls `ft_oe_list`, but that custom function and the per-sequence
playback table it creates are not in the supplied folder. The raw TTL event
files delimit complete 25-sequence trials only. Therefore, raw neural data are
validated directly, while exact stimulus labels are imported through the
lossless MATLAB bridge. No ECoG-derived clustering is used to manufacture
labels.

## Literal mapping

| MATLAB source operation | Translation |
|---|---|
| `Gen_M2Mat(..., hilbert_t=0)` | unchanged in the MATLAB export bridge |
| 100-ms per-epoch baseline subtraction and baseline sample-SD division | unchanged inside `Gen_M2Mat` |
| rare stimulus = minimum `pctList` | minimum observed `allM2(:,4)` count; if `pctList` exists, both indices must agree |
| deviant rows: `allM2(:,4)==rare` | identical source rows |
| standard rows: `find(other-day rare)+1` | identical source rows plus an assertion that every row is the same physical target sequence |
| `erp_win = basetime+1 : basetime+seqDur+1000` | identical samples; scientific exports additionally use 0-based time while retaining the source 1-based labels |
| truncate both classes to `min(n)` | identical stable first-`n` truncation |
| concatenate deviant then standard | identical class order (`0`, then `1`) |
| add `randn*1e-6`, then `zscore(X_t)` | identical in `matlab-faithful` |
| ridge logistic, `Lambda=1e-2` | same mean logistic-deviance plus L2 objective and unpenalized bias as `ECoG/roving` |
| five-fold misclassification accuracy | identical in the audit profile |
| `A = cov(X)*W` | identical in the audit profile |
| `movmean(...,20)` | MATLAB-centered moving mean with shrunken endpoints |
| first maximum and inclusive ±25 samples | identical in the audit profile |

## Source defects kept visible

1. “Rep 1 vs Rep 15” is stale text. The arrays contain deviant sequences and
   same-identity standards immediately after the opposite deviant.
2. The source script never calls `rng`; both added noise and fold assignments
   depend on an unknown workspace state. The supplied `.fig` files therefore
   cannot be recreated deterministically from the script alone.
3. The local `Gen_M2Mat.m` shown with the dataset does not populate
   `info.pctList`, although `scripts_AB_BA.m` reads it. The bridge uses the
   mathematically identical minimum stimulus count and asserts agreement when
   a MATLAB-path version does provide `pctList`.
4. Source time labels start at 1 even though the extracted first sample is the
   stimulus-onset sample. Both source labels and scientifically correct
   zero-based time are saved.
5. “Top driver channels” are distributed decoder activation patterns, not
   causal drivers.

## Leakage-safe changes

The manuscript profile changes only choices that would invalidate
cross-validated interpretation:

- all events from one acquisition trial remain in one fold;
- folds are stratified by class and fixed across time;
- means and sample SDs are fit on each training fold and applied to its test
  fold;
- flat training features receive scale 1, so no random noise is injected;
- Haufe patterns are computed within training folds and averaged;
- the spatial summary window is fixed to the complete two-item sequence
  (0–360 ms for experiments 1–2; 0–200 ms for experiment 3);
- the decoder peak remains descriptive and is never used for safe feature or
  latency selection.

## Remaining scientific limitation

The deviant and matched-standard classes come from different acquisition days.
Grouped CV prevents within-trial dependence and preprocessing leakage, but it
cannot distinguish sequence-role information from session/day differences.
This decoder must be described as cross-day condition decoding, supported by
the within-day controls appropriate to the final inference. Trials and
electrodes are not independent biological replicates; animal-level uncertainty
cannot be manufactured from this single recording pair.
