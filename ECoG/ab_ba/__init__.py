"""Two-sequence AB/BA ECoG analysis.

The package has two intentionally separate profiles:

``matlab-faithful``
    Audits the operations in ``AB_BA/scripts_AB_BA.m``, including its
    all-sample standardization and peak-selected spatial map.

``leakage-safe``
    Keeps every recording trial in one fold, fits scaling on training data
    only, fixes folds across time, and uses a prespecified spatial window.
"""

from .config import COMPARISONS, EXPERIMENTS

__all__ = ["COMPARISONS", "EXPERIMENTS"]
