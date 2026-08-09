from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from ECoG.ab_ba.config import COMPARISONS
from ECoG.ab_ba.data import ABBAEpochs, load_export
from ECoG.ab_ba.decoder import run_decoder, stratified_group_folds


class GroupFoldTests(unittest.TestCase):
    def test_no_recording_trial_crosses_folds(self) -> None:
        y = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
        groups = np.r_[np.repeat(np.arange(6), 2), np.repeat(np.arange(10, 16), 2)]
        folds = stratified_group_folds(y, groups, 3, np.random.RandomState(4))
        membership = {}
        for fold, test in enumerate(folds):
            self.assertEqual(set(np.unique(y[test])), {0, 1})
            for group in np.unique(groups[test]):
                self.assertNotIn(group, membership)
                membership[group] = fold
        self.assertEqual(set(membership), set(np.unique(groups)))


class ExportTests(unittest.TestCase):
    def test_compact_export_round_trip(self) -> None:
        spec = COMPARISONS["exp1_day1_deviant"]
        rng = np.random.RandomState(2)
        n, time = 12, 18
        item = {
            "x_deviant": rng.normal(size=(32, n, time)),
            "x_standard_after_deviant": rng.normal(size=(32, n, time)),
            "deviant_groups": 100000 + np.repeat(np.arange(1, 7), 2),
            "standard_groups": 200000 + np.repeat(np.arange(1, 7), 2),
            "deviant_trials": np.repeat(np.arange(1, 7), 2),
            "standard_trials": np.repeat(np.arange(1, 7), 2),
            "deviant_source_rows_matlab": np.arange(1, n + 1),
            "standard_source_rows_matlab": np.arange(31, 31 + n),
            "time_ms": np.arange(time),
            "source_time_labels_ms": np.arange(1, time + 1),
            "target_sequence": spec.expected_target_sequence,
            "n_deviant_before_balance": n + 1,
            "n_standard_before_balance": n,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.mat"
            savemat(path, {"comparisons": {spec.key: item}, "export_metadata": {}})
            loaded = load_export(path, spec)
        self.assertEqual(loaded.deviant.shape, (32, n, time))
        self.assertTrue(np.array_equal(loaded.time_ms, np.arange(time)))
        self.assertEqual(loaded.metadata["n_observations_per_class"], n)


class DecoderTests(unittest.TestCase):
    def test_safe_decoder_is_reproducible_and_grouped(self) -> None:
        spec = COMPARISONS["exp3_day1_deviant"]
        rng = np.random.RandomState(7)
        n_groups, per_group, n_time = 5, 2, 6
        n = n_groups * per_group
        deviant = rng.normal(size=(32, n, n_time))
        standard = rng.normal(size=(32, n, n_time))
        deviant[0, :, 2:4] += 0.8
        epochs = ABBAEpochs(
            deviant=deviant,
            standard_after_deviant=standard,
            deviant_groups=100000 + np.repeat(np.arange(n_groups), per_group),
            standard_groups=200000 + np.repeat(np.arange(n_groups), per_group),
            deviant_trials=np.repeat(np.arange(n_groups), per_group),
            standard_trials=np.repeat(np.arange(n_groups), per_group),
            deviant_source_rows_matlab=np.arange(1, n + 1),
            standard_source_rows_matlab=np.arange(101, 101 + n),
            time_ms=np.arange(n_time),
            source_time_labels_ms=np.arange(1, n_time + 1),
            metadata={},
        )
        first = run_decoder(epochs, spec, mode="leakage-safe")
        second = run_decoder(epochs, spec, mode="leakage-safe")
        np.testing.assert_allclose(first.accuracy, second.accuracy)
        np.testing.assert_allclose(first.activation_patterns, second.activation_patterns)
        groups = np.r_[epochs.deviant_groups, epochs.standard_groups]
        for group in np.unique(groups):
            self.assertEqual(np.unique(first.fold_ids[groups == group]).size, 1)
        self.assertEqual(first.standardization_scope.split(";")[0], "training fold only")


if __name__ == "__main__":
    unittest.main()
