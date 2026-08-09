"""Prepare leakage-safe, true Rep-1/Rep-15 ECoG data for Figure 2.

The source decoder scripts display their numeric repetition 14 as “Rep 15”.
Figure 2 does not inherit that mismatch. Endpoint ERPs and activation patterns
are re-extracted from the translated MATLAB loader using actual repetitions 1
and 15. The repetition map and endpoint AUC are read from the leakage-safe
regression export, where whole roving blocks are held out and fold-local
standardization is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from ECoG.roving.config import ANALYSES, AnalysisSpec
from ECoG.roving.decoder import fit_matlab_ridge_logistic
from ECoG.roving.matlab_io import (
    _context_order,
    _extract_selected_events,
    extract_repetition_epochs,
)
from final_figures.figure_2.erp_selection import (
    context_stratified_discovery_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECOG_RESULT_ROOT = PROJECT_ROOT / "ECoG" / "roving" / "results"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
POSITION_KEYS = ("zaatar_pos1", "zaatar_pos2", "zaatar_pos3")
GRID_MAP = np.array(
    [
        [4, 3, 2, 1],
        [8, 7, 6, 5],
        [12, 11, 10, 9],
        [16, 15, 14, 13],
        [17, 18, 19, 20],
        [24, 23, 22, 21],
        [28, 27, 26, 25],
        [32, 31, 30, 29],
    ],
    dtype=int,
)


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_true_endpoints(
    spec: AnalysisSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract actual repetitions 1 and 15 in exact MATLAB context order."""

    data_path = spec.data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing source recording: {data_path}")
    loaded = loadmat(data_path, variable_names=["xx"], simplify_cells=True)
    if "xx" not in loaded or not isinstance(loaded["xx"], dict):
        raise ValueError(f"{data_path} does not contain struct variable 'xx'")
    events, metadata = _extract_selected_events(
        loaded["xx"],
        spec,
        selected_reps=(1, 15),
        loader_cutting=1,
    )
    del loaded

    ordered: dict[int, list[Any]] = {1: [], 15: []}
    context_counts: list[dict[str, int]] = []
    for current, context in _context_order(spec.n_stim):
        per_rep: dict[int, list[Any]] = {}
        for repetition in (1, 15):
            selected = [
                event
                for event in events
                if event.rep == repetition
                and event.stimulus == current
                and event.context == context
            ]
            per_rep[repetition] = selected
            ordered[repetition].extend(selected)
        if len(per_rep[1]) != len(per_rep[15]):
            raise ValueError(
                f"{spec.key}: context {context}->{current} has unequal "
                f"Rep-1/Rep-15 counts ({len(per_rep[1])}, {len(per_rep[15])})"
            )
        context_counts.append(
            {
                "current_stimulus": current,
                "previous_stimulus": context,
                "n_blocks": len(per_rep[1]),
            }
        )

    groups_first = np.asarray([event.group for event in ordered[1]], dtype=int)
    groups_last = np.asarray([event.group for event in ordered[15]], dtype=int)
    if not np.array_equal(groups_first, groups_last):
        raise ValueError(f"{spec.key}: Rep-1 and Rep-15 roving blocks do not align")

    start, stop = 100, 901  # MATLAB 101:901, inclusive.
    rep1 = np.stack(
        [event.epoch[:, start:stop] for event in ordered[1]], axis=0
    )
    rep15 = np.stack(
        [event.epoch[:, start:stop] for event in ordered[15]], axis=0
    )
    if rep1.shape != rep15.shape or rep1.shape[1:] != (32, 801):
        raise AssertionError(
            f"{spec.key}: unexpected endpoint arrays {rep1.shape}, {rep15.shape}"
        )
    metadata.update(
        {
            "source_data_file": str(data_path.resolve()),
            "endpoint_repetitions": [1, 15],
            "source_loader": "Gen_M2Mat.m translation",
            "source_erp_window_matlab": [101, 901],
            "n_aligned_blocks": int(rep1.shape[0]),
            "context_counts": context_counts,
        }
    )
    return rep1, rep15, np.arange(801, dtype=int), metadata


def _activation_pattern(
    rep1: np.ndarray,
    rep15: np.ndarray,
    deviant_time_ms: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """Full-data Haufe pattern in the prespecified 0–180 ms window."""

    indices = np.flatnonzero(
        (deviant_time_ms >= 0) & (deviant_time_ms < 180)
    )
    if indices.size != 180:
        raise AssertionError(f"Expected 180 deviant-window samples, got {indices.size}")
    labels = np.r_[
        np.ones(rep1.shape[0], dtype=int),
        np.zeros(rep15.shape[0], dtype=int),
    ]
    patterns = np.empty((rep1.shape[1], indices.size), dtype=float)
    for out_index, time_index in enumerate(indices):
        raw = np.concatenate(
            [rep1[:, :, time_index], rep15[:, :, time_index]], axis=0
        )
        mean = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0, ddof=1) + 1e-6
        standardized = (raw - mean) / scale
        beta, _ = fit_matlab_ridge_logistic(
            standardized, labels, ridge_lambda
        )
        patterns[:, out_index] = (
            np.cov(standardized, rowvar=False, ddof=1) @ beta
        )
    pattern = np.mean(np.abs(patterns), axis=1)
    maximum = float(np.max(pattern))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Activation pattern has no finite positive range")
    return pattern / maximum


def _save_long_form_csvs(data: dict[str, np.ndarray], data_dir: Path) -> None:
    import csv

    def write(name: str, header: list[str], rows: list[list[Any]]) -> None:
        path = data_dir / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    response_rows: list[list[Any]] = []
    sequence_response_rows: list[list[Any]] = []
    auc_rows: list[list[Any]] = []
    posterior_rows: list[list[Any]] = []
    posterior_sequence_rows: list[list[Any]] = []
    topography_rows: list[list[Any]] = []
    buildup_rows: list[list[Any]] = []
    response_buildup_rows: list[list[Any]] = []
    for position in (1, 2, 3):
        time = data[f"pos{position}_time_ms"]
        for repetition in (1, 15):
            values = data[f"pos{position}_gfp_rep{repetition}"]
            response_rows.extend(
                [[position, repetition, int(t), float(v)] for t, v in zip(time, values)]
            )
            sequence_time = data[f"pos{position}_sequence_time_ms"]
            sequence_values = data[
                f"pos{position}_gfp_rep{repetition}_sequence"
            ]
            sequence_response_rows.extend(
                [
                    [position, repetition, int(t), float(v)]
                    for t, v in zip(sequence_time, sequence_values)
                ]
            )
        auc = data[f"pos{position}_auc"]
        auc_rows.extend(
            [[position, int(t), float(v)] for t, v in zip(time, auc)]
        )
        posterior = data[f"pos{position}_posterior"]
        posterior_sequence = data[f"pos{position}_posterior_sequence"]
        repetitions = data["repetitions"]
        for rep_index, repetition in enumerate(repetitions):
            posterior_rows.extend(
                [
                    [position, int(repetition), int(t), float(v)]
                    for t, v in zip(time, posterior[rep_index])
                ]
            )
            posterior_sequence_rows.extend(
                [
                    [position, int(repetition), int(t), float(v)]
                    for t, v in zip(sequence_time, posterior_sequence[rep_index])
                ]
            )
        pattern = data[f"pos{position}_topography"]
        topography_rows.extend(
            [
                [position, channel, float(pattern[channel - 1])]
                for channel in range(1, 33)
            ]
        )
        buildup = data[f"pos{position}_buildup"]
        buildup_rows.extend(
            [
                [position, int(repetition), float(value)]
                for repetition, value in zip(repetitions, buildup)
            ]
        )
        block_response = data[f"pos{position}_block_gfp_response"]
        for block_index in range(block_response.shape[0]):
            response_buildup_rows.extend(
                [
                    [
                        position,
                        block_index + 1,
                        int(repetition),
                        float(block_response[block_index, repetition_index]),
                    ]
                    for repetition_index, repetition in enumerate(repetitions)
                ]
            )

    write(
        "ecog_response_timecourses.csv",
        ["deviant_position", "repetition", "time_from_deviant_ms", "gfp_au"],
        response_rows,
    )
    write(
        "ecog_sequence_response_timecourses.csv",
        ["deviant_position", "repetition", "sequence_time_ms", "gfp_au"],
        sequence_response_rows,
    )
    write(
        "ecog_endpoint_auc.csv",
        ["deviant_position", "time_from_deviant_ms", "auc"],
        auc_rows,
    )
    write(
        "ecog_posterior_map.csv",
        [
            "deviant_position",
            "repetition",
            "time_from_deviant_ms",
            "rep1_posterior",
        ],
        posterior_rows,
    )
    write(
        "ecog_posterior_sequence_map.csv",
        [
            "deviant_position",
            "repetition",
            "sequence_time_ms",
            "rep1_posterior",
        ],
        posterior_sequence_rows,
    )
    write(
        "ecog_activation_patterns.csv",
        ["deviant_position", "channel", "relative_abs_activation"],
        topography_rows,
    )
    write(
        "ecog_normalized_buildup.csv",
        ["deviant_position", "repetition", "normalized_rep1_to_rep15"],
        buildup_rows,
    )
    write(
        "ecog_block_gfp_buildup.csv",
        [
            "deviant_position",
            "roving_block",
            "repetition",
            "mean_single_block_gfp_0_180_ms",
        ],
        response_buildup_rows,
    )


def build_ecog_cache(
    *,
    force: bool = False,
    data_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Build or load the frozen panel data and provenance."""

    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "ecog_figure2_data.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as cached:
            required = {
                "pos1_sequence_time_ms",
                "pos1_block_gfp_response",
                "pos1_posterior_sequence",
                "pos2_sequence_time_ms",
                "pos2_block_gfp_response",
                "pos2_posterior_sequence",
                "pos3_sequence_time_ms",
                "pos3_block_gfp_response",
                "pos3_posterior_sequence",
                "pos1_erp_selected_channel",
                "pos2_erp_selected_channel",
                "pos3_erp_selected_channel",
            }
            if required.issubset(cached.files):
                return {key: cached[key] for key in cached.files}

    result: dict[str, np.ndarray] = {
        "positions": np.array([1, 2, 3], dtype=int),
        "repetitions": np.arange(1, 16, dtype=int),
        "grid_map": GRID_MAP,
    }
    provenance: dict[str, Any] = {
        "scope": (
            "descriptive recordings from one animal; no biological population "
            "confidence interval or significance inference"
        ),
        "animal": "Zaatar",
        "endpoint_repetitions": [1, 15],
        "response_measure": (
            "global field power: population standard deviation across 32 "
            "baseline-SD-normalized channel ERPs"
        ),
        "spatial_measure": (
            "absolute Haufe activation pattern from a discovery-half ridge "
            "logistic fit, averaged over the prespecified 0-180 ms window "
            "and normalized within recording; the complementary blocks are "
            "reserved for channel-level ERP estimation and inference"
        ),
        "recordings": {},
    }

    for position, key in enumerate(POSITION_KEYS, start=1):
        spec = ANALYSES[key]
        map_dir = (
            ECOG_RESULT_ROOT
            / key
            / "regression_rep_map"
            / "leakage-safe"
        )
        map_path = map_dir / "regression_rep_map_arrays.npz"
        map_provenance_path = map_dir / "provenance.json"
        if not map_path.exists():
            raise FileNotFoundError(f"Missing leakage-safe map: {map_path}")
        with np.load(map_path) as source:
            sequence_time_from_map = source["time_ms"].astype(int)
            sequence_from_map = (
                (sequence_time_from_map >= 0)
                & (sequence_time_from_map <= 600)
            )
            posterior_sequence = source["posterior_smoothed"][
                :, sequence_from_map
            ]
            aligned_time = source["deviant_aligned_time_ms"].astype(int)
            common = (aligned_time >= 0) & (aligned_time <= 360)
            map_time = aligned_time[common]
            posterior = source["posterior_smoothed"][:, common]
            auc = source["endpoint_auc"][common]
        if not np.array_equal(map_time, np.arange(361)):
            raise AssertionError(f"{key}: common deviant time is not 0:360 ms")

        all_repetitions = extract_repetition_epochs(spec.data_path(), spec)
        rep1 = all_repetitions.epochs[:, 0]
        rep15 = all_repetitions.epochs[:, -1]
        source_time = all_repetitions.time_ms
        endpoint_metadata = all_repetitions.metadata
        deviant_time = source_time - spec.deviant_onset_ms
        common_endpoint = (deviant_time >= 0) & (deviant_time <= 360)
        if not np.array_equal(deviant_time[common_endpoint], map_time):
            raise AssertionError(f"{key}: endpoint and map time axes differ")
        erp1 = np.mean(rep1, axis=0)
        erp15 = np.mean(rep15, axis=0)
        gfp1 = np.std(erp1, axis=0, ddof=0)[common_endpoint]
        gfp15 = np.std(erp15, axis=0, ddof=0)[common_endpoint]
        sequence = (source_time >= 0) & (source_time <= 600)
        sequence_time = source_time[sequence]
        if not np.array_equal(sequence_time, np.arange(601)):
            raise AssertionError(f"{key}: sequence time is not 0:600 ms")
        gfp1_sequence = np.std(erp1, axis=0, ddof=0)[sequence]
        gfp15_sequence = np.std(erp15, axis=0, ddof=0)[sequence]

        single_block_gfp = np.std(
            all_repetitions.epochs, axis=2, ddof=0
        )
        response_window = (
            (source_time >= spec.deviant_onset_ms)
            & (source_time < spec.deviant_onset_ms + 180)
        )
        block_gfp_response = np.mean(
            single_block_gfp[:, :, response_window], axis=2
        )
        discovery_mask = context_stratified_discovery_mask(
            all_repetitions.stimuli,
            all_repetitions.contexts,
        )
        inference_mask = ~discovery_mask
        topography = _activation_pattern(
            rep1[discovery_mask],
            rep15[discovery_mask],
            deviant_time,
            spec.lambda_ridge,
        )
        selected_channel = int(np.argmax(topography)) + 1

        window = (map_time >= 0) & (map_time < 180)
        posterior_window = np.mean(posterior[:, window], axis=1)
        denominator = posterior_window[0] - posterior_window[-1]
        if denominator <= 0:
            raise ValueError(f"{key}: Rep-1 posterior is not above Rep-15")
        buildup = (posterior_window - posterior_window[-1]) / denominator

        result[f"pos{position}_time_ms"] = map_time
        result[f"pos{position}_gfp_rep1"] = gfp1
        result[f"pos{position}_gfp_rep15"] = gfp15
        result[f"pos{position}_sequence_time_ms"] = sequence_time
        result[f"pos{position}_gfp_rep1_sequence"] = gfp1_sequence
        result[f"pos{position}_gfp_rep15_sequence"] = gfp15_sequence
        result[f"pos{position}_block_gfp_response"] = block_gfp_response
        result[f"pos{position}_posterior"] = posterior
        result[f"pos{position}_posterior_sequence"] = posterior_sequence
        result[f"pos{position}_auc"] = auc
        result[f"pos{position}_topography"] = topography
        result[f"pos{position}_erp_selected_channel"] = np.asarray(
            selected_channel, dtype=np.int64
        )
        result[f"pos{position}_erp_discovery_mask"] = discovery_mask
        result[f"pos{position}_erp_inference_mask"] = inference_mask
        result[f"pos{position}_buildup"] = buildup

        map_provenance = json.loads(map_provenance_path.read_text())
        provenance["recordings"][key] = {
            "deviant_position": position,
            "recording": spec.recording,
            "source_data_file": str(spec.data_path().resolve()),
            "source_data_sha256": map_provenance["data_sha256"],
            "map_arrays": str(map_path.resolve()),
            "map_arrays_sha256": _sha256(map_path),
            "map_method": map_provenance["method"],
            "fold_strategy": map_provenance["fold_strategy"],
            "standardization_scope": map_provenance["standardization_scope"],
            "n_blocks": endpoint_metadata["global_min_trials"],
            "erp_contact_selection": {
                "rule": (
                    "maximum of the displayed discovery-half absolute Haufe "
                    "activation pattern"
                ),
                "selected_channel_matlab": selected_channel,
                "n_discovery_blocks": int(np.count_nonzero(discovery_mask)),
                "n_inference_blocks": int(np.count_nonzero(inference_mask)),
                "split": (
                    "alternating intact blocks within every ordered "
                    "previous/current stimulus stratum; inference blocks are "
                    "never used to select the contact"
                ),
            },
        }
        del (
            rep1,
            rep15,
            single_block_gfp,
            all_repetitions,
        )

    np.savez_compressed(cache_path, **result)
    _save_long_form_csvs(result, data_dir)
    provenance["cache_file"] = str(cache_path.resolve())
    provenance["cache_sha256"] = _sha256(cache_path)
    provenance["generator"] = str(Path(__file__).resolve())
    provenance["generator_sha256"] = _sha256(Path(__file__).resolve())
    (data_dir / "ecog_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    build_ecog_cache()
