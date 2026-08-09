"""Independent-contact ERP estimation and inference for manuscript Figure 2."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t

from ECoG.roving.config import ANALYSES
from ECoG.roving.matlab_io import extract_repetition_epochs
from final_figures.figure_2.erp_selection import (
    context_stratified_discovery_mask,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
NPZ_NAME = "channel_erp_inference.npz"
PROVENANCE_NAME = "channel_erp_provenance.json"
CLUSTERS_CSV = "channel_erp_clusters.csv"
TIMECOURSES_CSV = "channel_erp_timecourses.csv"
POSITION_KEYS = ("zaatar_pos1", "zaatar_pos2", "zaatar_pos3")
TIME_MS = np.arange(0, 361, dtype=np.int64)
N_RANDOMIZATIONS = 4_999
RANDOM_SEED = 2_026_080_9
ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
DISPLAY_SMOOTH_SIGMA_MS = 2.0
SCHEMA_VERSION = "figure-2-independent-contact-erp-v1"


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _analysis_id(ecog: Mapping[str, np.ndarray]) -> tuple[str, dict[str, Any]]:
    selected = {
        str(position): int(ecog[f"pos{position}_erp_selected_channel"])
        for position in (1, 2, 3)
    }
    topographies = {
        str(position): _array_sha256(ecog[f"pos{position}_topography"])
        for position in (1, 2, 3)
    }
    sources = {
        key: {
            "path": str(ANALYSES[key].data_path().resolve()),
            "sha256": _sha256(ANALYSES[key].data_path()),
        }
        for key in POSITION_KEYS
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "selected_channels_matlab": selected,
        "discovery_topography_sha256": topographies,
        "sources": sources,
        "settings": {
            "time_ms": TIME_MS.tolist(),
            "n_randomizations": N_RANDOMIZATIONS,
            "random_seed": RANDOM_SEED,
            "alpha": ALPHA,
            "cluster_forming_alpha": CLUSTER_FORMING_ALPHA,
            "display_smoothing_sigma_ms": DISPLAY_SMOOTH_SIGMA_MS,
        },
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), identity


def _clusters(trace: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for sign in (1.0, -1.0):
        active = sign * trace > threshold
        padded = np.r_[False, active, False].astype(np.int8)
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for start, stop in zip(starts, stops):
            clusters.append(
                {
                    "start": int(start),
                    "stop": int(stop),
                    "sign": int(sign),
                    "mass": float(np.sum(np.abs(trace[start:stop]))),
                }
            )
    return clusters


def _paired_t(difference: np.ndarray) -> np.ndarray:
    n_blocks = difference.shape[0]
    mean = difference.mean(axis=0)
    scale = difference.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    return np.divide(mean, scale, out=np.zeros_like(mean), where=scale > 0)


def _null_t(difference: np.ndarray, signs: np.ndarray) -> np.ndarray:
    n_blocks = difference.shape[0]
    means = (signs @ difference) / n_blocks
    sum_squares = np.sum(difference * difference, axis=0)
    variances = (sum_squares[None, :] - n_blocks * means * means) / (
        n_blocks - 1
    )
    variances = np.maximum(variances, 0.0)
    standard_errors = np.sqrt(variances / n_blocks)
    return np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0,
    )


def _signs(n_randomizations: int, n_blocks: int, position: int) -> np.ndarray:
    generator = np.random.default_rng(RANDOM_SEED + 10_007 * position)
    signs = generator.choice(
        np.array([-1.0, 1.0]),
        size=(n_randomizations, n_blocks),
        replace=True,
    )
    return signs


def _write_csvs(
    data_dir: Path,
    arrays: Mapping[str, np.ndarray],
    cluster_rows: list[dict[str, Any]],
) -> list[Path]:
    cluster_path = data_dir / CLUSTERS_CSV
    with cluster_path.open("w", newline="") as handle:
        fieldnames = [
            "deviant_position",
            "channel_matlab",
            "start_ms",
            "end_ms",
            "direction",
            "cluster_mass",
            "p_corrected",
            "significant",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cluster_rows)

    timecourse_path = data_dir / TIMECOURSES_CSV
    with timecourse_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "deviant_position",
                "channel_matlab",
                "time_ms",
                "rep1_mean_smoothed",
                "rep1_sem_smoothed",
                "rep15_mean_smoothed",
                "rep15_sem_smoothed",
                "raw_rep1_minus_rep15",
                "raw_paired_t",
                "cluster_p_corrected",
                "cluster_significant",
            ]
        )
        for position in (1, 2, 3):
            channel = int(arrays[f"pos{position}_channel_matlab"])
            for index, time_ms in enumerate(arrays["time_ms"]):
                writer.writerow(
                    [
                        position,
                        channel,
                        int(time_ms),
                        float(arrays[f"pos{position}_rep1_mean"][index]),
                        float(arrays[f"pos{position}_rep1_sem"][index]),
                        float(arrays[f"pos{position}_rep15_mean"][index]),
                        float(arrays[f"pos{position}_rep15_sem"][index]),
                        float(arrays[f"pos{position}_difference"][index]),
                        float(arrays[f"pos{position}_t"][index]),
                        float(arrays[f"pos{position}_p_corrected"][index]),
                        bool(arrays[f"pos{position}_significant"][index]),
                    ]
                )
    return [cluster_path, timecourse_path]


def build_channel_erp(
    ecog: Mapping[str, np.ndarray],
    *,
    force: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Build independently selected ERP traces and joint cluster inference."""

    destination = Path(data_dir or DEFAULT_DATA_DIR).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    analysis_id, identity = _analysis_id(ecog)
    npz_path = destination / NPZ_NAME
    provenance_path = destination / PROVENANCE_NAME
    if npz_path.exists() and provenance_path.exists() and not force:
        try:
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("analysis_id") == analysis_id:
                with np.load(npz_path, allow_pickle=False) as cached:
                    return {key: cached[key] for key in cached.files}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    arrays: dict[str, np.ndarray] = {
        "time_ms": TIME_MS.copy(),
        "positions": np.arange(1, 4, dtype=np.int64),
        "analysis_id": np.asarray(analysis_id),
    }
    null_traces: list[np.ndarray] = []
    thresholds: list[float] = []
    observed_clusters: list[list[dict[str, Any]]] = []
    recording_provenance: dict[str, Any] = {}

    for position, key in enumerate(POSITION_KEYS, start=1):
        spec = ANALYSES[key]
        epochs = extract_repetition_epochs(spec.data_path(), spec)
        discovery = context_stratified_discovery_mask(
            epochs.stimuli, epochs.contexts
        )
        cached_discovery = np.asarray(
            ecog[f"pos{position}_erp_discovery_mask"], dtype=bool
        )
        if not np.array_equal(discovery, cached_discovery):
            raise AssertionError(f"position {position}: discovery split mismatch")
        inference = ~discovery

        channel = int(ecog[f"pos{position}_erp_selected_channel"])
        topography = np.asarray(ecog[f"pos{position}_topography"], dtype=float)
        if channel != int(np.argmax(topography)) + 1:
            raise AssertionError(
                f"position {position}: selected contact is not the Panel E maximum"
            )
        channel_index = channel - 1

        deviant_time = np.asarray(epochs.time_ms) - spec.deviant_onset_ms
        time_indices = np.searchsorted(deviant_time, TIME_MS)
        if not np.array_equal(deviant_time[time_indices], TIME_MS):
            raise AssertionError(f"position {position}: missing 0-360 ms ERP axis")

        raw_rep1 = epochs.epochs[inference, 0, channel_index][:, time_indices]
        raw_rep15 = epochs.epochs[inference, -1, channel_index][:, time_indices]
        difference = np.asarray(raw_rep1 - raw_rep15, dtype=np.float64)
        observed_t = _paired_t(difference)
        threshold = float(
            student_t.ppf(
                1.0 - CLUSTER_FORMING_ALPHA / 2.0,
                difference.shape[0] - 1,
            )
        )
        signs = _signs(N_RANDOMIZATIONS, difference.shape[0], position)
        null_t = _null_t(difference, signs).astype(np.float32)

        full_rep1 = gaussian_filter1d(
            epochs.epochs[inference, 0, channel_index].astype(np.float64),
            sigma=DISPLAY_SMOOTH_SIGMA_MS,
            axis=-1,
            mode="reflect",
            truncate=4.0,
        )
        full_rep15 = gaussian_filter1d(
            epochs.epochs[inference, -1, channel_index].astype(np.float64),
            sigma=DISPLAY_SMOOTH_SIGMA_MS,
            axis=-1,
            mode="reflect",
            truncate=4.0,
        )
        smooth_rep1 = full_rep1[:, time_indices]
        smooth_rep15 = full_rep15[:, time_indices]
        n_blocks = difference.shape[0]

        arrays[f"pos{position}_channel_matlab"] = np.asarray(
            channel, dtype=np.int64
        )
        arrays[f"pos{position}_n_discovery_blocks"] = np.asarray(
            np.count_nonzero(discovery), dtype=np.int64
        )
        arrays[f"pos{position}_n_inference_blocks"] = np.asarray(
            n_blocks, dtype=np.int64
        )
        arrays[f"pos{position}_rep1_mean"] = smooth_rep1.mean(axis=0)
        arrays[f"pos{position}_rep1_sem"] = smooth_rep1.std(
            axis=0, ddof=1
        ) / np.sqrt(n_blocks)
        arrays[f"pos{position}_rep15_mean"] = smooth_rep15.mean(axis=0)
        arrays[f"pos{position}_rep15_sem"] = smooth_rep15.std(
            axis=0, ddof=1
        ) / np.sqrt(n_blocks)
        arrays[f"pos{position}_difference"] = difference.mean(axis=0)
        arrays[f"pos{position}_t"] = observed_t
        arrays[f"pos{position}_cluster_threshold_abs_t"] = np.asarray(
            threshold
        )
        null_traces.append(null_t)
        thresholds.append(threshold)
        observed_clusters.append(_clusters(observed_t, threshold))
        recording_provenance[key] = {
            "deviant_position": position,
            "selected_channel_matlab": channel,
            "n_discovery_blocks": int(np.count_nonzero(discovery)),
            "n_inference_blocks": int(n_blocks),
            "cluster_forming_abs_t": threshold,
        }

    null_maximum = np.zeros(N_RANDOMIZATIONS, dtype=np.float64)
    for position_index, null_t in enumerate(null_traces):
        threshold = thresholds[position_index]
        for permutation in range(N_RANDOMIZATIONS):
            clusters = _clusters(null_t[permutation], threshold)
            if clusters:
                null_maximum[permutation] = np.maximum(
                    null_maximum[permutation],
                    max(cluster["mass"] for cluster in clusters),
                )

    cluster_rows: list[dict[str, Any]] = []
    for position_index, clusters in enumerate(observed_clusters):
        position = position_index + 1
        significant = np.zeros(TIME_MS.size, dtype=bool)
        p_corrected = np.ones(TIME_MS.size, dtype=float)
        channel = int(arrays[f"pos{position}_channel_matlab"])
        for cluster in clusters:
            probability = (
                1.0 + np.count_nonzero(null_maximum >= cluster["mass"])
            ) / (N_RANDOMIZATIONS + 1.0)
            start, stop = cluster["start"], cluster["stop"]
            p_corrected[start:stop] = probability
            significant[start:stop] = probability < ALPHA
            cluster_rows.append(
                {
                    "deviant_position": position,
                    "channel_matlab": channel,
                    "start_ms": int(TIME_MS[start]),
                    "end_ms": int(TIME_MS[stop - 1]),
                    "direction": (
                        "rep1_above_rep15"
                        if cluster["sign"] > 0
                        else "rep15_above_rep1"
                    ),
                    "cluster_mass": float(cluster["mass"]),
                    "p_corrected": float(probability),
                    "significant": bool(probability < ALPHA),
                }
            )
        arrays[f"pos{position}_significant"] = significant
        arrays[f"pos{position}_p_corrected"] = p_corrected

    _atomic_npz(npz_path, arrays)
    csv_paths = _write_csvs(destination, arrays, cluster_rows)
    provenance = {
        **identity,
        "analysis_id": analysis_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "At the independently selected strongest Panel E contact, does "
            "the Rep-1 ERP differ from the Rep-15 ERP at any latency from "
            "0 to 360 ms?"
        ),
        "selection": (
            "Within each ordered previous/current stimulus stratum, "
            "alternating intact roving blocks form a discovery half. Panel E "
            "and its maximum contact use only that half. The complementary "
            "blocks estimate and test the displayed ERP."
        ),
        "estimation": (
            "Mean and SEM across intact held-out roving blocks at the selected "
            "baseline-SD-normalized contact."
        ),
        "display_smoothing": (
            "Symmetric zero-phase Gaussian smoothing (sigma 2 ms; FWHM "
            "approximately 4.7 ms) is applied to each full held-out block "
            "trace before averaging and cropping. It is used only for the "
            "drawn mean and SEM, never for inference."
        ),
        "inference": (
            "Two-sided paired t cluster mass on unsmoothed held-out block "
            "differences; 4,999 whole-block sign flips; p<0.05 two-sided t "
            "cluster-forming threshold; maximum cluster jointly over every "
            "latency and all three displayed recordings."
        ),
        "scope": (
            "Three recordings from one animal; within-recording inference "
            "only, not an animal-population claim."
        ),
        "recordings": recording_provenance,
        "outputs": {
            "npz": str(npz_path),
            "npz_sha256": _sha256(npz_path),
            "csv": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in csv_paths
            },
        },
    }
    _atomic_json(provenance_path, provenance)
    return arrays


__all__ = ["build_channel_erp"]
