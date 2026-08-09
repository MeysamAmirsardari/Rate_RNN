"""Dependency-light equivalence and leakage tests (run with unittest)."""

from __future__ import annotations

import unittest

import numpy as np

from ECoG.roving.decoder import (
    _group_folds,
    matlab_movmean,
    matlab_zscore,
)


class MatlabPrimitiveTests(unittest.TestCase):
    def test_zscore_uses_sample_standard_deviation(self) -> None:
        x = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]])
        standardized, mean, scale = matlab_zscore(x)
        np.testing.assert_allclose(mean, [2.0, 8.0])
        np.testing.assert_allclose(scale, [1.0, 4.0])
        np.testing.assert_allclose(
            standardized, [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
        )

    def test_movmean_even_window_matches_matlab_centering(self) -> None:
        values = np.arange(1.0, 7.0)
        # movmean(x,4): two samples behind, current, one sample ahead.
        expected = np.array([1.5, 2.0, 2.5, 3.5, 4.5, 5.0])
        np.testing.assert_allclose(matlab_movmean(values, 4), expected)

    def test_group_folds_never_split_a_roving_block(self) -> None:
        groups = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        folds = _group_folds(groups, 2, np.random.RandomState(11))
        fold_id = np.full(len(groups), -1)
        for index, test in enumerate(folds):
            fold_id[test] = index
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(fold_id[groups == group])), 1)


if __name__ == "__main__":
    unittest.main()

