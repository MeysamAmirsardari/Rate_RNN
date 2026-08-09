# AB/BA ECoG analysis

This directory is the Nutmeg conventional-oddball counterpart of
`ECoG/roving`. It translates `AB_BA/scripts_AB_BA.m`, preserves its exact
signal/class extraction in a compact MATLAB bridge, and separates literal
source reproduction from the leakage-safe manuscript analysis.

## What is compared

For each of the three experiments and each sequence order, the two classes are:

1. that physical sequence when it is the rare deviant on one acquisition day;
2. the same physical sequence when it is the standard immediately after the
   opposite deviant on the other acquisition day.

The source figure labels “Rep 1” and “Rep 15” do **not** describe these arrays.
The new outputs use the actual condition names.

## Data boundary

The supplied legacy Open Ephys files contain 32 neural channels at 4 kHz and
whole-trial TTL edges. They do not contain the per-sequence `stimat` playback
table. That table is created by the custom, currently absent `ft_oe_list.m`.
Sequence identity is never inferred from ECoG responses because that would
make the labels outcome-dependent.

First audit the raw archive and supplied MATLAB figures:

```bash
cd /Users/eminent/Projects/Rate_RNN_py
python -m ECoG.ab_ba.run inventory
python -m ECoG.ab_ba.run reference
```

Then, in the original MATLAB environment that has `ft_oe_list.m` and its Open
Ephys dependencies, create the lossless preprocessing export:

```matlab
addpath('/Users/eminent/Projects/ECoG')
addpath('/Users/eminent/Projects/Rate_RNN_py/ECoG/ab_ba/matlab')
export_ab_ba_preprocessed( ...
    '/Users/eminent/Projects/ECoG/AB_BA', ...
    '/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat')
```

For the Figure 4 ECoG analysis, which uses only experiment 1 (180-ms tones,
zero within-sequence gap), the exporter can skip experiments 2 and 3 entirely.
This is useful when those experiments are unavailable in the MATLAB source
environment:

```matlab
addpath('/Users/eminent/Projects/ECoG')
addpath('/Users/eminent/Projects/Rate_RNN_py/ECoG/ab_ba/matlab')
export_ab_ba_preprocessed( ...
    '/Users/eminent/Projects/ECoG/AB_BA', ...
    '/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat', ...
    1)
```

The third argument is optional. Omitting it preserves the original behavior
and exports experiments 1--3. The experiment-1-only MAT file contains
`comparisons.exp1_day1_deviant` and `comparisons.exp1_day2_deviant`. Each
comparison must contain the following fields used by the Python analysis:

- `x_deviant` and `x_standard_after_deviant` (channel x trial x time arrays);
- `time_ms`, `source_time_labels_ms`, `baseline_samples`, and
  `sequence_duration_samples`;
- `deviant_groups` and `standard_groups` for leakage-safe grouped folds;
- `deviant_trials`, `standard_trials`, and both MATLAB source-row fields for
  observation-level provenance;
- `target_sequence`, `expnum`, `deviant_day`, and `standard_source_day`;
- `note_duration_ms`, `note_gap_ms`, and `n_keep_per_class`;
- the three MATLAB stimulus-index fields, both pre-balance class counts, and
  both source stimulus-count vectors for selection and balancing audits.

At file level, `export_metadata.selected_experiments` records the requested
experiment numbers and `export_metadata.export_scope` records whether the
export contains all experiments or a subset.

Run the two experiment-1 comparisons from that subset export with:

```bash
python -m ECoG.ab_ba.run exp1_day1_deviant --mode leakage-safe
python -m ECoG.ab_ba.run exp1_day2_deviant --mode leakage-safe
```

Do not use the `all` target with a subset export: by design, the loader will
reject comparisons whose experiment was not exported.

The exporter stops if `find(other-day deviant)+1` crosses an acquisition-trial
boundary, is not the same physical sequence, if channel metadata differ, or if
an epoch is too short. It does not silently “repair” the source selection.

Run all six comparisons:

```bash
python -m ECoG.ab_ba.run all --mode leakage-safe
python -m ECoG.ab_ba.run all --mode both
```

Outputs are isolated under
`ECoG/ab_ba/results/<comparison>/<mode>/`, including arrays, CSV figure data,
observation-level provenance, hashes, method warnings, and a QC figure.

## Profiles

- `matlab-faithful` preserves all-observation z-scoring, per-time random
  five-fold repartitioning, tiny random noise, full-data Haufe patterns, and
  peak-centered channel selection. It is an audit output, not the manuscript
  result. Because the supplied script never seeds its RNG, the package uses a
  declared seed (11 by default) rather than claiming bit-identical recovery of
  an unknowable random state.
- `leakage-safe` keeps every acquisition trial in one fold, balances folds by
  class, fixes folds across time, fits scaling only on training observations,
  removes the artificial-noise workaround, forms spatial patterns only from
  training folds, and uses the prespecified 0–sequence-duration window.

Read `METHOD_EQUIVALENCE.md` before interpreting either profile.

## Tests

```bash
python -m unittest discover -s ECoG/ab_ba/tests -v
```
