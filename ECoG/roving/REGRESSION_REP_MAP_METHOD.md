# Regression translation of `SVM_rep_map.m`

## What changed

The source script defines Rep 1 as class 1, Rep 15 as class 0, trains an
L2-ridge linear SVM at every millisecond, and projects repetitions 2–14 onto
the fitted anchor axis. The requested translation changes only the learner in
the source-faithful profile:

- source: L2-ridge linear SVM;
- translation: L2-ridge logistic regression, lambda 0.01.

The saved value is the logistic class-1 probability
`P(Rep 1-like | ECoG)`. It is a model posterior score, not an animal-level
Bayesian posterior. A value of 0.5 is the decision boundary.

## Preserved source operations

- `Gen_M2Mat.m`, including `cutting=1`;
- raw ERP (`hilbert_t=0`);
- loader baseline subtraction and division by the sample standard deviation
  of the 100-ms baseline;
- all six previous/current stimulus transitions;
- repetitions 1 through 15;
- MATLAB samples 101:901, corresponding to 0–800 ms;
- global minimum trial count across all repetitions;
- `rng(42)` and one independent `randperm` call per repetition;
- Rep 1/Rep 15 anchor standardization using the sample standard deviation plus
  `1e-6`;
- five-fold anchor cross-validation;
- full anchor model projection of repetitions 2–14;
- Gaussian map smoothing with repetition sigma 0.8 and time sigma 3, default
  MATLAB filter size, and replicate boundary padding.

All three requested recordings contain exactly 125 retained blocks at every
repetition. Therefore, the source balancing permutations change row order but
do not discard observations. The extractor verifies equal counts and identical
block identities across all 15 repetitions.

## Two analysis profiles

### `matlab-faithful`

This reproduces the source operation order with logistic regression replacing
the SVM. Rep 1 and Rep 15 receive cross-validated posteriors. Repetitions 2–14
are projected through a model trained on all endpoint observations.

This profile retains three source leakage paths:

1. standardization is fitted before anchor cross-validation;
2. random folds can split Rep 1 and Rep 15 from the same roving block;
3. intermediate repetitions are evaluated using a model trained on endpoint
   observations from the same blocks.

It is retained for method auditing, not primary inference.

### `leakage-safe`

The 125 roving blocks are divided into five fixed folds, stratified over the
six previous/current stimulus transitions. For each fold and time point:

1. the model sees only Rep 1 and Rep 15 from training blocks;
2. scaling is fitted only on those training observations;
3. the fitted model predicts all 15 repetitions of held-out blocks.

This produces a fully out-of-fold array with dimensions
`block × repetition × time`. Every block and all its repetitions stay together
in one fold. Folds remain fixed over time.

The map is descriptive within a recording. Blocks are repeated measurements
from one animal, and the three position recordings are all from Zaatar; they
must not be treated as independent animals for population inference.

## Outputs

Each dataset and mode contains:

- `RegressionRepMap.mat`: MATLAB-compatible cache;
- `regression_rep_map_arrays.npz`: trial-level posteriors and metadata;
- `figure_data/posterior_map.csv`: mean, SEM, and smoothed figure data;
- `figure_data/endpoint_performance.csv`: out-of-fold Rep 1/Rep 15 accuracy
  and ROC AUC;
- `figure_data/block_index.csv`: source block identity and fold assignment;
- `regression_rep_map.png`: repetition-by-time posterior map;
- `provenance.json`: data, MATLAB-source, and Python-source SHA-256 hashes.

