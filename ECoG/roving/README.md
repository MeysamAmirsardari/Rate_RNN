# Roving ECoG analysis

This directory is the empirical-data side of the planned roving ECoG/model
figure. It contains a careful Python translation of the three requested MATLAB
decoders, their loader behavior, reproducible outputs, and an explicit leakage
audit.

## Run

From `/Users/eminent/Projects/Rate_RNN_py`:

```bash
python -m ECoG.roving.run available --mode leakage-safe
```

To reproduce the original scripts' analysis choices for audit purposes:

```bash
python -m ECoG.roving.run available --mode matlab-faithful
```

To run one recording and save both profiles:

```bash
python -m ECoG.roving.run zaatar_pos2 --mode both
```

Outputs are contained in `results/<analysis>/<mode>/`. Source `.mat` files are
read from `/Users/eminent/Projects/ECoG` by default and are not duplicated.
Set `ECOG_SOURCE_DIR` or pass `--source-dir` to use another verified source
directory.

After running the available recordings, create one recording-level,
deviant-aligned figure table (without invalid pooled confidence intervals):

```bash
python -m ECoG.roving.aggregate
```

To translate `SVM_rep_map.m` with L2-ridge logistic regression and generate
Rep-1-like posterior maps for all 15 repetitions:

```bash
python -m ECoG.roving.repetition_run all --mode both
```

The `leakage-safe` profile keeps every roving block and all its repetitions in
one fold. It is the primary figure profile; `matlab-faithful` retains the
source preprocessing and fold behavior for auditing. Build the three-position
deviant-aligned figure and combined table with:

```bash
python -m ECoG.roving.repetition_aggregate
```

See `REGRESSION_REP_MAP_METHOD.md` for the preserved operations, the single
requested learner change, and the interpretation of the posterior.

The position-3 input is `Zaatar_2024-11-27_xx.mat`. Its stimulus metadata and
the explicit comments/settings in `wave_analysis.m` and `pe_vs_pc.m` resolve
the ambiguity created by the initially guessed Aspen decoder script. See
`METHOD_EQUIVALENCE.md` before interpreting or publishing results.

## Tests

```bash
python -m unittest discover -s ECoG/roving/tests -v
```
