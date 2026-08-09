"""Contract tests for the publication Figure 4 SFG inference layer."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"


def _inference() -> dict[str, np.ndarray]:
    with np.load(DATA / "figure_4_inference.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def test_corrected_probabilities_match_marks_and_exact_grid() -> None:
    inference = _inference()
    families = (
        ("d_p_corrected", "d_significant"),
        ("d_slope_p_corrected", "d_slope_significant"),
        ("e_p_corrected", "e_significant"),
        ("f_slope_p_corrected", "f_slope_significant"),
        ("g_p_corrected", "g_significant"),
    )
    for probability_key, significant_key in families:
        probability = np.asarray(inference[probability_key], dtype=float)
        significant = np.asarray(inference[significant_key], dtype=bool)
        assert np.array_equal(significant, probability < 0.05)
        assert np.allclose(probability * 256, np.round(probability * 256))


def test_timecourse_cluster_contract() -> None:
    inference = _inference()
    significant = np.asarray(inference["c_significant"], dtype=bool)
    probability = np.asarray(inference["c_p_corrected"], dtype=float)
    assert significant.shape == (30,)
    assert np.array_equal(significant, probability < 0.05)
    assert np.all(significant)


def test_drive_check_uses_complete_exemplar_channel_distributions() -> None:
    with np.load(DATA / "sfg_exemplar.npz", allow_pickle=False) as source:
        stimulus = np.asarray(source["stim"], dtype=float)
        figure_index = np.asarray(source["figure_index"], dtype=int)
    onsets = np.diff(
        (stimulus > 0).astype(np.int8),
        axis=1,
        prepend=np.int8(0),
    ) == 1
    counts = onsets.sum(axis=1)
    figure = np.zeros(stimulus.shape[0], dtype=bool)
    figure[figure_index] = True
    assert counts[figure].size == 10
    assert counts[~figure].size == 27
    assert np.isclose(counts[figure].mean(), 56.1)
    assert np.isclose(counts[~figure].mean(), 54.074074074074076)

    with (DATA / "figure_4_drive_check.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 37
    exported = np.asarray([int(row["pips_per_channel"]) for row in rows])
    assert np.array_equal(exported, counts)


def test_provenance_declares_seed_unit_and_panel_families() -> None:
    provenance = json.loads(
        (DATA / "figure_4_inference_provenance.json").read_text()
    )
    assert provenance["unit"] == "paired simulated session seed"
    assert provenance["n_seeds"] == 8
    assert provenance["randomization"] == "exhaustive 2^8 paired-seed sign flips"
    assert set(provenance["multiplicity"]) == {
        "C", "D_points", "D_trends", "E", "F", "G"
    }
