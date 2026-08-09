"""Integrity checks for the independent-contact Figure 2 ERP contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / name, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def test_selected_contacts_are_panel_e_maxima_and_splits_are_disjoint() -> None:
    ecog = _load("ecog_figure2_data.npz")
    for position in (1, 2, 3):
        channel = int(ecog[f"pos{position}_erp_selected_channel"])
        pattern = np.asarray(ecog[f"pos{position}_topography"], dtype=float)
        discovery = np.asarray(
            ecog[f"pos{position}_erp_discovery_mask"], dtype=bool
        )
        inference = np.asarray(
            ecog[f"pos{position}_erp_inference_mask"], dtype=bool
        )
        assert channel == int(np.argmax(pattern)) + 1
        assert not np.any(discovery & inference)
        assert np.all(discovery | inference)


def test_erp_statistics_and_display_contract_are_consistent() -> None:
    erp = _load("channel_erp_inference.npz")
    assert np.array_equal(erp["time_ms"], np.arange(361))
    for position in (1, 2, 3):
        significant = np.asarray(erp[f"pos{position}_significant"], dtype=bool)
        probability = np.asarray(
            erp[f"pos{position}_p_corrected"], dtype=float
        )
        assert np.array_equal(significant, probability < 0.05)
        for repetition in (1, 15):
            mean = erp[f"pos{position}_rep{repetition}_mean"]
            sem = erp[f"pos{position}_rep{repetition}_sem"]
            assert mean.shape == (361,)
            assert np.all(np.isfinite(mean))
            assert np.all(np.isfinite(sem))
            assert np.all(sem >= 0)


def test_smoothing_is_declared_display_only() -> None:
    provenance = json.loads((DATA / "channel_erp_provenance.json").read_text())
    assert "used only" in provenance["display_smoothing"]
    assert "never for inference" in provenance["display_smoothing"]
    assert "unsmoothed" in provenance["inference"]
