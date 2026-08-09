# Final manuscript figures

This directory is the reproducible home for manuscript figures. Each figure has
its own directory containing:

- the figure-building code;
- a frozen, machine-readable panel dataset;
- provenance and methodological notes;
- vector and high-resolution raster exports; and
- a draft caption that states the correct inferential scope.

The shared visual system is documented in
[`VISUALIZATION_STRATEGY.md`](VISUALIZATION_STRATEGY.md) and implemented in
[`style.py`](style.py).

## Figures

- [`figure_2/`](figure_2/) — roving novelty responses in ECoG and the rate RNN.
- [`figure_4/`](figure_4/) — stochastic figure-ground connectivity and its
  functional boundary.
- [`figure_5/`](figure_5/) — statistical learning and multiscale sequence
  composition.
- [`figure_7/`](figure_7/) — frozen-core perturbation map and the no-retuning
  ABA– falsification test.

Generated outputs are intended to be versioned with the manuscript. Source
recordings remain in the project ECoG directory and are never copied here.
