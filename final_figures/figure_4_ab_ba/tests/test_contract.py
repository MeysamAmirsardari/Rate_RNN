from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from final_figures.figure_4_ab_ba.ecog_data import (
    ERP_DISPLAY_SMOOTHING_SIGMA_MS,
    load_reference,
    smooth_erp_for_display,
)
from final_figures.figure_4_ab_ba.make_figure_4 import DISPLAYED_PANELS
from final_figures.figure_4_ab_ba.model_data import (
    AB_BA_OVERRIDES,
    CONDITIONS,
    P_REGULAR,
    PROVENANCE_PATH,
    TIMING_LONG,
    TIMING_SHORT,
    build,
    condition_config,
)


class Figure4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build(force=False)

    def test_six_counterbalanced_reference_curves_exist(self):
        comparisons = load_reference()
        self.assertEqual(len(comparisons), 6)
        self.assertEqual(len({item.family for item in comparisons}), 3)
        self.assertEqual(
            [item.sequence_code for item in comparisons],
            ["AB", "BA", "AB", "BA", "AB", "BA"],
        )
        for item in comparisons:
            self.assertEqual(item.unexpected.shape, item.predicted.shape)
            self.assertEqual(item.decoder_time_ms.shape, item.decoder_accuracy.shape)
            self.assertEqual(item.time_ms[0], 0.0)
            self.assertEqual(item.decoder_time_ms[0], 0.0)

    def test_main_page_has_contiguous_a_through_f_panel_map(self):
        self.assertEqual(DISPLAYED_PANELS, ("A", "B", "C", "D", "E", "F"))

    def test_erp_smoothing_is_fixed_zero_phase_and_display_only(self):
        self.assertEqual(ERP_DISPLAY_SMOOTHING_SIGMA_MS, 3.0)
        time = np.arange(101, dtype=float)
        constant = np.ones_like(time)
        np.testing.assert_allclose(
            smooth_erp_for_display(time, constant),
            constant,
            rtol=0.0,
            atol=1e-14,
        )
        impulse = np.zeros_like(time)
        impulse[50] = 1.0
        smoothed = smooth_erp_for_display(time, impulse)
        self.assertEqual(int(np.argmax(smoothed)), 50)
        np.testing.assert_allclose(smoothed, smoothed[::-1], rtol=0.0, atol=1e-15)
        self.assertEqual(impulse[50], 1.0)

    def test_recalculated_panel_b_uses_held_out_block_erps(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "ECoG"
            / "ab_ba"
            / "results"
            / "ab_ba_channel_erp"
            / "ab_ba_channel_erp.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_allclose(data["planned_regular_probability"], 0.85)
            np.testing.assert_allclose(data["planned_rare_probability"], 0.15)
            np.testing.assert_array_equal(data["discovery_blocks"], np.arange(1, 17, 2))
            np.testing.assert_array_equal(data["inference_blocks"], np.arange(2, 17, 2))
            self.assertEqual(int(data["n_exact_assignments"]), 12870)
            self.assertEqual(float(data["time_ms"][0]), 0.0)
            self.assertEqual(float(data["time_ms"][-1]), 599.75)
            for key in ("5300_9400", "9400_5300"):
                self.assertEqual(data[f"{key}_rare_blocks_raw"].shape, (8, 2400))
                self.assertEqual(data[f"{key}_regular_blocks_raw"].shape, (8, 2400))
                self.assertEqual(data[f"{key}_significant"].dtype, np.bool_)
                self.assertGreaterEqual(int(data[f"{key}_channel_matlab"]), 1)
                self.assertLessEqual(int(data[f"{key}_channel_matlab"]), 32)

    def test_panel_b_provenance_records_matlab_tags_and_joint_correction(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "ECoG"
            / "ab_ba"
            / "results"
            / "ab_ba_channel_erp"
            / "provenance.json"
        )
        provenance = json.loads(path.read_text())
        self.assertEqual(
            provenance["probabilities"]["baphy_deviant_pct"],
            {"nmg038a01": 15.0, "nmg039a01": 15.0},
        )
        self.assertIn("six allM2 metadata columns", provenance["data_contract"])
        self.assertIn("1-250 Hz", provenance["preprocessing"])
        self.assertIn("C(16,8)=12,870", provenance["inference"])
        self.assertIn("jointly over AB, BA", provenance["inference"])
        self.assertIn("display only", provenance["display_smoothing"])

    def test_held_out_contract_has_seed_level_data(self):
        response = self.model["response"]
        self.assertEqual(response.shape[:4], (4, 2, 2, 12))
        self.assertTrue(np.isfinite(response).all())

    def test_sequence_clock_covers_zero_through_600_ms(self):
        target_aligned_time = self.model["time_ms"]
        np.testing.assert_array_equal(
            target_aligned_time,
            np.arange(-180, 421, dtype=float),
        )
        np.testing.assert_array_equal(
            target_aligned_time + 180,
            np.arange(0, 601, dtype=float),
        )
        self.assertEqual(self.model["response"].shape[-1], 601)

    def test_surprise_response_is_intact_rare_minus_regular(self):
        intact = CONDITIONS.index("intact")
        roles = self.model["roles"].tolist()
        predicted = roles.index("predicted")
        unexpected = roles.index("unexpected")
        expected = (
            self.model["response"][intact, :, unexpected]
            - self.model["response"][intact, :, predicted]
        )
        self.assertEqual(self.model["surprise_response"].shape, (2, 12, 601))
        np.testing.assert_array_equal(self.model["surprise_response"], expected)

        population_expected = (
            self.model["population_response"][intact, :, unexpected]
            - self.model["population_response"][intact, :, predicted]
        )
        self.assertEqual(
            self.model["population_surprise_response"].shape,
            (2, 12, 601),
        )
        np.testing.assert_array_equal(
            self.model["population_surprise_response"],
            population_expected,
        )

    def test_surprisal_bits_are_formal_context_probabilities(self):
        expected = -np.log2(np.asarray([P_REGULAR, 1.0 - P_REGULAR]))
        self.assertEqual(self.model["surprisal_bits"].shape, (2,))
        np.testing.assert_allclose(
            self.model["surprisal_bits"], expected, rtol=0.0, atol=1e-15
        )

    def test_context_decoder_contract_is_seed_grouped_and_finite(self):
        decoder_time = self.model["model_decoder_time_ms"]
        np.testing.assert_array_equal(decoder_time, np.arange(0, 601, 5))
        expected_shape = (2, decoder_time.size)
        for key in (
            "model_decoder_accuracy",
            "model_decoder_ci_low",
            "model_decoder_ci_high",
            "model_decoder_significant",
            "model_decoder_p_corrected",
        ):
            self.assertEqual(self.model[key].shape, expected_shape)

        ci_low = self.model["model_decoder_ci_low"]
        ci_high = self.model["model_decoder_ci_high"]
        self.assertTrue(np.isfinite(ci_low).all())
        self.assertTrue(np.isfinite(ci_high).all())
        self.assertTrue(np.all(ci_low <= ci_high))
        self.assertTrue(np.all((self.model["model_decoder_p_corrected"] >= 0.0)))
        self.assertTrue(np.all((self.model["model_decoder_p_corrected"] <= 1.0)))
        self.assertEqual(self.model["model_decoder_significant"].dtype, np.bool_)

        fold_ids = self.model["model_decoder_fold_ids"]
        self.assertEqual(fold_ids.shape, self.model["seeds"].shape)
        folds, counts = np.unique(fold_ids, return_counts=True)
        np.testing.assert_array_equal(folds, np.arange(5))
        self.assertTrue(np.all(counts >= 2))

    def test_context_decoder_provenance_records_leakage_guards(self):
        with PROVENANCE_PATH.open() as stream:
            provenance = json.load(stream)
        self.assertEqual(
            provenance["leakage_guard"],
            "identical balanced held-out test stream; learning disabled during test",
        )
        self.assertEqual(
            provenance["replication_unit"],
            "paired training/test seed; trials averaged within seed; sequence "
            "identities retained for time-resolved analyses and averaged only "
            "for prespecified scalar effects",
        )
        specification = provenance["specification"]
        self.assertEqual(specification["test_p_ab"], 0.5)
        self.assertEqual(specification["model_decoder_folds"], 5)
        self.assertEqual(
            specification["model_decoder_unit"],
            "paired seed; trial-averaged before decoding",
        )

    def test_no_recurrent_learning_has_zero_context_effect(self):
        index = CONDITIONS.index("no_recurrent_learning")
        np.testing.assert_allclose(self.model["condition_effect"][index], 0.0, atol=1e-12)

    def test_ecog_matched_configuration_is_exact(self):
        self.assertEqual(P_REGULAR, 0.85)
        self.assertEqual(
            TIMING_LONG,
            dict(tone_dur=0.180, intra_gap=0.000, inter_gap=1.500),
        )
        self.assertEqual(
            TIMING_SHORT,
            dict(tone_dur=0.050, intra_gap=0.100, inter_gap=1.500),
        )
        cfg, learn = condition_config("intact")
        self.assertTrue(learn)
        for key, value in AB_BA_OVERRIDES.items():
            self.assertEqual(getattr(cfg, key), value)

    def test_weight_history_is_complete_and_matches_terminal_state(self):
        trajectory = self.model["weight_trajectory"]
        checkpoints = self.model["weight_checkpoints"]
        self.assertEqual(trajectory.shape, (2, 12, 17, 2, 2))
        self.assertEqual(self.model["weights"].shape, (4, 2, 12, 2, 2))
        np.testing.assert_array_equal(checkpoints, np.arange(0, 401, 25))
        np.testing.assert_allclose(trajectory[:, :, 0], 0.0, atol=1e-12)
        intact = CONDITIONS.index("intact")
        np.testing.assert_allclose(trajectory[:, :, -1], self.model["weights"][intact])

    def test_time_resolved_mechanism_has_seed_level_replication(self):
        mechanism = self.model["mechanism_time_difference"]
        self.assertEqual(mechanism.shape, (3, 12, self.model["time_ms"].size))
        self.assertTrue(np.isfinite(mechanism).all())
        self.assertEqual(
            self.model["mechanism_time_significant"].shape,
            (3, self.model["time_ms"].size),
        )

    def test_paired_perturbation_contract_is_seed_level(self):
        difference = self.model["lesion_difference"]
        self.assertEqual(difference.shape, (3, 12))
        np.testing.assert_allclose(
            difference,
            self.model["condition_effect"][1:] - self.model["condition_effect"][0],
        )
        self.assertTrue(np.isfinite(difference).all())

    def test_exhaustive_randomization_probabilities_use_exact_denominator(self):
        for key in (
            "model_decoder_p_corrected",
            "time_cluster_p_fwer",
            "population_time_cluster_p_fwer",
            "condition_effect_p_fwer",
            "lesion_vs_intact_p_fwer",
            "weight_alignment_p_fwer",
        ):
            probabilities = np.asarray(self.model[key], dtype=float)
            scaled = probabilities * (2 ** self.model["seeds"].size)
            np.testing.assert_allclose(scaled, np.rint(scaled), atol=1e-12)
            self.assertFalse(np.any(probabilities < 0.0))
            self.assertFalse(np.any(probabilities > 1.0))


if __name__ == "__main__":
    unittest.main()
