"""Regression repetition-map unit and leakage tests."""

from __future__ import annotations

import unittest

import numpy as np

from ECoG.roving.config import ANALYSES
from ECoG.roving.matlab_io import RovingRepetitionEpochs
from ECoG.roving.repetition_map import (
    RepetitionMapConfig,
    _context_stratified_block_folds,
    matlab_imgaussfilt,
    run_repetition_map,
)


class RepetitionMapTests(unittest.TestCase):
    def test_gaussian_smoothing_preserves_constant_map(self) -> None:
        image = np.full((15, 21), 0.37)
        np.testing.assert_allclose(
            matlab_imgaussfilt(image, 0.8, 3.0),
            image,
            atol=1e-15,
        )

    def test_context_folds_cover_blocks_once(self) -> None:
        stimuli = np.tile([1, 1, 2, 2, 3, 3], 5)
        contexts = np.tile([2, 3, 1, 3, 1, 2], 5)
        folds = _context_stratified_block_folds(
            stimuli, contexts, 5, np.random.RandomState(42)
        )
        assigned = np.concatenate(folds)
        np.testing.assert_array_equal(np.sort(assigned), np.arange(30))
        for fold in folds:
            pairs = set(zip(contexts[fold], stimuli[fold]))
            self.assertEqual(len(pairs), 6)

    def test_safe_map_predicts_every_repetition_out_of_fold(self) -> None:
        rng = np.random.RandomState(7)
        n_blocks, n_repetitions, n_channels, n_time = 30, 15, 4, 3
        repetitions = np.arange(1, n_repetitions + 1)
        signal = np.linspace(1.2, -1.2, n_repetitions)
        data = rng.normal(
            scale=0.35,
            size=(n_blocks, n_repetitions, n_channels, n_time),
        )
        data[:, :, 0, :] += signal[None, :, None]
        stimuli = np.tile([1, 1, 2, 2, 3, 3], 5)
        contexts = np.tile([2, 3, 1, 3, 1, 2], 5)
        epochs = RovingRepetitionEpochs(
            epochs=data,
            group_ids=np.arange(1, n_blocks + 1),
            trials=np.arange(1, n_blocks + 1),
            stimuli=stimuli,
            contexts=contexts,
            repetitions=repetitions,
            time_ms=np.arange(n_time),
            metadata={"deviant_onset_ms": 360},
        )
        result = run_repetition_map(
            epochs,
            ANALYSES["zaatar_pos3"],
            mode="leakage-safe",
            config=RepetitionMapConfig(n_folds=5),
        )
        self.assertTrue(np.all(np.isfinite(result.posterior_trials)))
        self.assertGreater(
            float(np.mean(result.posterior_trials[:, 0])),
            float(np.mean(result.posterior_trials[:, -1])),
        )
        self.assertEqual(set(result.fold_ids_by_block.tolist()), set(range(5)))
        np.testing.assert_array_equal(
            result.anchor_fold_ids[:, :n_blocks],
            result.anchor_fold_ids[:, n_blocks:],
        )


if __name__ == "__main__":
    unittest.main()
