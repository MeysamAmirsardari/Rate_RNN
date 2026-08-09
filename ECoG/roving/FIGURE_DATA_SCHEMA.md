# Figure 2 data contract

Each completed analysis writes:

- `figure_data/decoding_timecourse.csv`: native and deviant-aligned time,
  unsmoothed accuracy, and the source-compatible 20-sample display smoothing;
- `figure_data/channel_erps.csv`: all 32 channel ERPs for the first and numeric
  late repetition, in native ECoG units after the source baseline correction;
- `figure_data/spatial_pattern.csv`: Haufe-style activation-pattern magnitude
  in the declared spatial window;
- `analysis_arrays.npz`: the complete balanced class tensors, folds, decoder
  patterns, and ERPs;
- `provenance.json`: source hash, complete analysis settings, software
  versions, leakage mode, recording identity, and warnings;
- `decoder_qc.png`: a diagnostic reconstruction of the source composite.

For the planned ECoG/model Figure 2:

1. use `deviant_aligned_time_ms` for every recording and simulation;
2. retain native units in separate ECoG and model panels;
3. compute a separately labeled normalized suppression/reset effect for the
   direct comparison panel;
4. aggregate uncertainty only across independent recordings/animals for ECoG
   and across independent initialized networks for the model;
5. use the leakage-safe decoder curve for manuscript claims;
6. treat selected-channel traces and decoder topographies as descriptive.

No model data are copied into this directory automatically. A later combined
figure should read these immutable exports and the simulation exports, rather
than recomputing either analysis inside plotting code.

