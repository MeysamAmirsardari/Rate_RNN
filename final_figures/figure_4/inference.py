"""Exact paired-seed inference for manuscript stochastic figure-ground Figure 4."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import t as student_t

from final_figures.figure_4.sfg_data import (
    CLOUD_BINS,
    FIGURE_BINS,
    FIGURE_SIZES,
    N_REPS,
    N_SEEDS,
    REFERENCE_PRESET,
    difference,
    modulation,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
NPZ_NAME = "figure_4_inference.npz"
CSV_NAME = "figure_4_inference.csv"
PROVENANCE_NAME = "figure_4_inference_provenance.json"
ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
TRACE_E_INDEX = 0
SCHEMA_VERSION = "figure-4-sfg-exact-paired-v1"


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _all_signs(n: int) -> np.ndarray:
    integers = np.arange(2**n, dtype=np.uint16)[:, None]
    bits = (integers >> np.arange(n, dtype=np.uint16)[None, :]) & 1
    return np.where(bits == 1, 1.0, -1.0)


def _paired_t(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    se = sd / np.sqrt(np.maximum(n, 1))
    result = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
    constant_nonzero = (se == 0) & (np.abs(mean) > 0)
    result[constant_nonzero] = np.sign(mean[constant_nonzero]) * np.inf
    return result


def _permuted_t(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.stack([_paired_t(values * sign[:, None]) for sign in signs])


def _max_t(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != N_SEEDS:
        raise ValueError(f"Expected ({N_SEEDS}, tests), got {values.shape}")
    signs = _all_signs(N_SEEDS)
    observed = _paired_t(values)
    null = _permuted_t(values, signs)
    maximum = np.max(np.abs(null), axis=1)
    probabilities = np.array([
        np.mean(maximum >= abs(value) - 1e-12)
        for value in observed
    ])
    return observed, probabilities, probabilities < ALPHA


def _clusters(
    trace: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for sign in (1.0, -1.0):
        active = sign * trace > thresholds
        padded = np.r_[False, active, False].astype(np.int8)
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for start, stop in zip(starts, stops):
            clusters.append({
                "start": int(start),
                "stop": int(stop),
                "sign": int(sign),
                "mass": float(np.sum(np.abs(trace[start:stop]))),
            })
    return clusters


def _cluster_test(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    values = np.asarray(values, dtype=float)
    counts = np.sum(np.isfinite(values), axis=0)
    if np.any(counts < 3):
        raise ValueError("Cluster inference requires at least three seeds per bin")
    thresholds = student_t.ppf(
        1.0 - CLUSTER_FORMING_ALPHA / 2.0,
        counts - 1,
    )
    observed = _paired_t(values)
    signs = _all_signs(N_SEEDS)
    null = _permuted_t(values, signs)
    null_maximum = np.zeros(signs.shape[0], dtype=float)
    for permutation, trace in enumerate(null):
        clusters = _clusters(trace, thresholds)
        if clusters:
            null_maximum[permutation] = max(c["mass"] for c in clusters)

    significant = np.zeros(values.shape[1], dtype=bool)
    corrected = np.ones(values.shape[1], dtype=float)
    rows: list[dict[str, Any]] = []
    for cluster in _clusters(observed, thresholds):
        probability = float(
            np.mean(null_maximum >= cluster["mass"] - 1e-12)
        )
        start, stop = cluster["start"], cluster["stop"]
        corrected[start:stop] = probability
        significant[start:stop] = probability < ALPHA
        rows.append({**cluster, "p_corrected": probability,
                     "significant": probability < ALPHA})
    return observed, corrected, significant, rows


def _course(data: Mapping[str, np.ndarray], n_fig: int, group: str) -> np.ndarray:
    return modulation(
        data,
        REFERENCE_PRESET,
        n_fig,
        f"course_{group}",
    )[:, :, TRACE_E_INDEX]


def _window(course: np.ndarray, bins: np.ndarray | slice) -> np.ndarray:
    return np.nanmean(course[:, bins], axis=1)


def _slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    centred = x - x.mean()
    return np.sum((y - y.mean(axis=1, keepdims=True)) * centred, axis=1) / np.sum(
        centred * centred
    )


def build_inference(
    data: Mapping[str, np.ndarray],
    *,
    data_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    destination = Path(data_dir or DEFAULT_DATA_DIR).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "analysis_schema": np.asarray(SCHEMA_VERSION),
        "positions": np.arange(1, N_REPS + 1),
        "figure_sizes": np.asarray(FIGURE_SIZES),
    }
    rows: list[dict[str, Any]] = []

    # Panel C: paired figure-minus-ground trace, cluster-corrected over 30 bins.
    c_values = _course(data, 10, "fig") - _course(data, 10, "gnd")
    c_t, c_p, c_sig, c_clusters = _cluster_test(c_values)
    arrays.update(c_t=c_t, c_p_corrected=c_p, c_significant=c_sig)
    for cluster in c_clusters:
        rows.append({
            "panel": "C",
            "family": "figure_minus_ground_timecourse",
            "contrast": "figure channels - ground channels",
            "index_start": cluster["start"],
            "index_stop": cluster["stop"] - 1,
            "direction": "positive" if cluster["sign"] > 0 else "negative",
            "statistic": cluster["mass"],
            "p_corrected": cluster["p_corrected"],
            "significant": cluster["significant"],
        })

    # Panel D response: each plotted point against zero, one eight-point family.
    d_values = []
    d_labels = []
    per_group_size: dict[str, list[np.ndarray]] = {"fig": [], "gnd": []}
    for group in ("fig", "gnd"):
        for n_fig in FIGURE_SIZES:
            course = _course(data, n_fig, group)
            value = _window(course, FIGURE_BINS) - _window(course, CLOUD_BINS)
            d_values.append(value)
            d_labels.append(f"{group}_{n_fig}")
            per_group_size[group].append(value)
    d_t, d_p, d_sig = _max_t(np.stack(d_values, axis=1))
    arrays.update(d_t=d_t, d_p_corrected=d_p, d_significant=d_sig)
    for label, statistic, probability, significant in zip(
        d_labels, d_t, d_p, d_sig
    ):
        rows.append({
            "panel": "D",
            "family": "response_points_vs_zero",
            "contrast": f"{label} vs zero",
            "index_start": "",
            "index_stop": "",
            "direction": "two-sided",
            "statistic": statistic,
            "p_corrected": probability,
            "significant": bool(significant),
        })

    d_slopes = np.stack([
        _slope(np.stack(per_group_size[group], axis=1), np.asarray(FIGURE_SIZES))
        for group in ("fig", "gnd")
    ], axis=1)
    d_slope_t, d_slope_p, d_slope_sig = _max_t(d_slopes)
    arrays.update(
        d_slope_t=d_slope_t,
        d_slope_p_corrected=d_slope_p,
        d_slope_significant=d_slope_sig,
    )
    for group, statistic, probability, significant in zip(
        ("fig", "gnd"), d_slope_t, d_slope_p, d_slope_sig
    ):
        rows.append({
            "panel": "D",
            "family": "size_trends",
            "contrast": f"{group} linear slope vs zero",
            "index_start": "",
            "index_stop": "",
            "direction": "two-sided",
            "statistic": statistic,
            "p_corrected": probability,
            "significant": bool(significant),
        })

    # Panel E: six non-thalamic plastic-minus-frozen currents against zero.
    e_values = []
    e_labels = []
    for group in ("fig", "gnd"):
        for current in ("rec", "inh", "net"):
            e_values.append(difference(
                data, REFERENCE_PRESET, 10, f"{current}_{group}_figure"
            ))
            e_labels.append(f"{current}_{group}")
    e_t, e_p, e_sig = _max_t(np.stack(e_values, axis=1))
    arrays.update(e_t=e_t, e_p_corrected=e_p, e_significant=e_sig)
    for label, statistic, probability, significant in zip(
        e_labels, e_t, e_p, e_sig
    ):
        rows.append({
            "panel": "E",
            "family": "non_thalamic_currents_vs_zero",
            "contrast": f"{label} vs zero",
            "index_start": "",
            "index_stop": "",
            "direction": "two-sided",
            "statistic": statistic,
            "p_corrected": probability,
            "significant": bool(significant),
        })

    # Panel F: per-seed linear buildup slope, corrected over four figure sizes.
    f_slopes = []
    for n_fig in FIGURE_SIZES:
        buildup = modulation(data, REFERENCE_PRESET, n_fig, "buildup_fig")
        f_slopes.append(_slope(buildup, np.arange(1, N_REPS + 1)))
    f_t, f_p, f_sig = _max_t(np.stack(f_slopes, axis=1))
    arrays.update(f_slope_t=f_t, f_slope_p_corrected=f_p,
                  f_slope_significant=f_sig)
    for n_fig, statistic, probability, significant in zip(
        FIGURE_SIZES, f_t, f_p, f_sig
    ):
        rows.append({
            "panel": "F",
            "family": "buildup_slopes",
            "contrast": f"size {n_fig} linear slope vs zero",
            "index_start": "",
            "index_stop": "",
            "direction": "two-sided",
            "statistic": statistic,
            "p_corrected": probability,
            "significant": bool(significant),
        })

    # Panel G: paired figure-minus-ground cloud modulation at each size.
    g_values = []
    for n_fig in FIGURE_SIZES:
        g_values.append(
            _window(_course(data, n_fig, "fig"), CLOUD_BINS)
            - _window(_course(data, n_fig, "gnd"), CLOUD_BINS)
        )
    g_t, g_p, g_sig = _max_t(np.stack(g_values, axis=1))
    arrays.update(g_t=g_t, g_p_corrected=g_p, g_significant=g_sig)
    for n_fig, statistic, probability, significant in zip(
        FIGURE_SIZES, g_t, g_p, g_sig
    ):
        rows.append({
            "panel": "G",
            "family": "figure_minus_ground_cloud",
            "contrast": f"size {n_fig}: figure - ground",
            "index_start": "",
            "index_stop": "",
            "direction": "two-sided",
            "statistic": statistic,
            "p_corrected": probability,
            "significant": bool(significant),
        })

    npz_path = destination / NPZ_NAME
    np.savez_compressed(npz_path, **arrays)
    csv_path = destination / CSV_NAME
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_path = destination / "sfg_figure4_data.npz"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "paired simulated session seed",
        "n_seeds": N_SEEDS,
        "randomization": "exhaustive 2^8 paired-seed sign flips",
        "multiplicity": {
            "C": "two-sided cluster mass over all 30 time bins",
            "D_points": "maximum absolute t over eight group-by-size points",
            "D_trends": "maximum absolute t over two group slopes",
            "E": "maximum absolute t over six non-thalamic currents",
            "F": "maximum absolute t over four buildup slopes",
            "G": "maximum absolute t over four size-specific group contrasts",
        },
        "alpha": ALPHA,
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "outputs": {
            NPZ_NAME: {"path": str(npz_path), "sha256": _sha256(npz_path)},
            CSV_NAME: {"path": str(csv_path), "sha256": _sha256(csv_path)},
        },
    }
    (destination / PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return arrays


__all__ = ["build_inference"]
