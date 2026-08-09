"""Auditable inferential statistics for manuscript Figure 2.

This module is deliberately separate from both data preparation and plotting.
It consumes the frozen dictionaries returned by :mod:`ecog_data` and
:mod:`model_data`, re-extracts the raw ECoG endpoint features when required,
and writes a single cache with every uncertainty interval and corrected
significance result used by Figure 2.

The biological unit is one animal.  Consequently, the ECoG tests below are
within-recording randomization tests over intact roving blocks; they do not
support a population-level animal inference.  Decoder label permutations
refit every fold after swapping Rep-1/Rep-15 labels *within blocks*.  The
posterior-map test is the strongest computationally practical test for the
full 15-repetition map: a two-sided whole-block sign-flip test on cross-fitted
logit evidence, with the exact display smoothing applied before inference.

Public entry point
------------------
``load_or_build_inference(ecog, model, force=False, data_dir=None)``

The returned dictionary contains only NumPy arrays and is safe to load with
``allow_pickle=False``.  Important axis arrays are ``decoder_time_ms``,
``posterior_time_ms``, ``posterior_repetition``, ``rnn_trace_time_ms``, and
``buildup_repetition``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numba
import numpy as np
from numba import njit, prange
from scipy.ndimage import convolve1d, label
from scipy.special import expit, logit
from scipy.stats import t as student_t

from ECoG.roving.config import ANALYSES
from ECoG.roving.decoder import fit_matlab_ridge_logistic
from ECoG.roving.matlab_io import extract_repetition_epochs
from ECoG.roving.repetition_map import (
    _context_stratified_block_folds,
    _matlab_gaussian_kernel,
)


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_ECOG_RESULTS = _REPO_ROOT / "ECoG" / "roving" / "results"

DEFAULT_DATA_DIR = _THIS_FILE.parent / "data"
NPZ_NAME = "figure_2_inference.npz"
PROVENANCE_NAME = "figure_2_inference_provenance.json"
DECODER_CLUSTERS_CSV = "decoder_significant_clusters.csv"
POSTERIOR_CLUSTERS_CSV = "posterior_significant_clusters.csv"
RNN_CLUSTERS_CSV = "rnn_trace_significant_clusters.csv"
POINTWISE_CSV = "buildup_and_suppression_inference.csv"

SCHEMA_VERSION = "figure-2-inference-v3-erp-clusters"
ERP_CLUSTERS_CSV = "figure_2_erp_clusters.csv"
POSITION_KEYS = ("zaatar_pos1", "zaatar_pos2", "zaatar_pos3")
DECODER_TIME_MS = np.arange(0, 601, 5, dtype=np.int64)
#: Deviant-aligned axis of the panel-B evoked traces, in milliseconds.
ERP_TIME_MS = np.arange(0, 361, dtype=np.int64)
#: Permutation batch size for the evoked-response null.
ERP_CHUNK = 512
POSTERIOR_TIME_MS = np.arange(0, 601, 5, dtype=np.int64)
POSTERIOR_REPETITIONS = np.arange(1, 16, dtype=np.int64)
N_RANDOMIZATIONS = 4_999
N_BOOTSTRAPS = 4_999
RANDOM_SEED = 2_026_073_1
ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
SCALE_EPSILON = 1e-6
POSTERIOR_LOGIT_CLIP = 1e-6
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Small deterministic utilities
# ---------------------------------------------------------------------------
def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _source_stamp(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _condition_names(model: Mapping[str, np.ndarray]) -> list[str]:
    return [str(value) for value in np.asarray(model["conditions"]).tolist()]


def _settings() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decoder_time_ms": DECODER_TIME_MS.tolist(),
        "erp_time_ms": [int(ERP_TIME_MS[0]), int(ERP_TIME_MS[-1]),
                        int(ERP_TIME_MS.size)],
        "posterior_time_ms": POSTERIOR_TIME_MS.tolist(),
        "n_folds": N_FOLDS,
        "n_randomizations": N_RANDOMIZATIONS,
        "n_bootstraps": N_BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "alpha": ALPHA,
        "cluster_forming_alpha": CLUSTER_FORMING_ALPHA,
        "scale_epsilon": SCALE_EPSILON,
        "posterior_logit_clip": POSTERIOR_LOGIT_CLIP,
        "posterior_smoothing": {
            "sigma_repetition_samples": 0.8,
            "sigma_time_samples_at_1_ms": 3.0,
            "padding": "nearest/replicate",
            "order": "repetition then time",
        },
    }


def _input_identity(
    ecog: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    recordings: dict[str, Any] = {}
    for key in POSITION_KEYS:
        spec = ANALYSES[key]
        recordings[key] = _source_stamp(spec.data_path())

    model_keys = [
        key
        for key in (
            "analysis_id",
            "conditions",
            "positions",
            "seeds",
            "sequence_responses",
            "responses_full",
            "window_response",
            "suppression_index",
        )
        if key in model
    ]
    ecog_keys = [
        key
        for key in (
            "positions",
            "repetitions",
            "pos1_block_gfp_response",
            "pos2_block_gfp_response",
            "pos3_block_gfp_response",
        )
        if key in ecog
    ]
    return {
        "recordings": recordings,
        "model_arrays": {
            key: _array_sha256(np.asarray(model[key])) for key in model_keys
        },
        "ecog_arrays": {
            key: _array_sha256(np.asarray(ecog[key])) for key in ecog_keys
        },
    }


def _analysis_id(
    ecog: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    identity = _input_identity(ecog, model)
    core = {
        "settings": _settings(),
        "inputs": identity,
        "generator_sha256": _sha256(_THIS_FILE),
    }
    return hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest(), core


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# ---------------------------------------------------------------------------
# Fast, objective-equivalent ridge logistic regression
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def _ridge_newton_dense(
    x: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
    tolerance: float = 1e-9,
    max_iterations: int = 50,
) -> tuple[np.ndarray, int]:
    """Newton solve of the same average-deviance ridge objective as MATLAB.

    The bias is the last coefficient and is not penalized.  This dense helper
    exists both for validation and for modest-sized auxiliary analyses.
    """

    n_samples, n_features = x.shape
    theta = np.zeros(n_features + 1, dtype=np.float64)
    gradient = np.zeros(n_features + 1, dtype=np.float64)
    hessian = np.zeros(
        (n_features + 1, n_features + 1), dtype=np.float64
    )

    completed = 0
    for iteration in range(max_iterations):
        gradient[:] = 0.0
        hessian[:] = 0.0
        for sample in range(n_samples):
            score = theta[n_features]
            for feature in range(n_features):
                score += x[sample, feature] * theta[feature]
            probability = 1.0 / (1.0 + np.exp(-score))
            residual = probability - y[sample]
            curvature = probability * (1.0 - probability)
            gradient[n_features] += residual
            hessian[n_features, n_features] += curvature
            for feature in range(n_features):
                feature_value = x[sample, feature]
                gradient[feature] += feature_value * residual
                hessian[n_features, feature] += (
                    curvature * feature_value
                )
                for other in range(feature + 1):
                    hessian[feature, other] += (
                        curvature
                        * feature_value
                        * x[sample, other]
                    )

        for feature in range(n_features):
            gradient[feature] = (
                gradient[feature] / n_samples
                + ridge_lambda * theta[feature]
            )
            for other in range(feature + 1):
                hessian[feature, other] /= n_samples
                hessian[other, feature] = hessian[feature, other]
            hessian[feature, feature] += ridge_lambda
            hessian[n_features, feature] /= n_samples
            hessian[feature, n_features] = hessian[n_features, feature]
        gradient[n_features] /= n_samples
        hessian[n_features, n_features] = (
            hessian[n_features, n_features] / n_samples + 1e-12
        )

        step = np.linalg.solve(hessian, gradient)
        maximum_step = 0.0
        for coefficient in range(n_features + 1):
            theta[coefficient] -= step[coefficient]
            if abs(step[coefficient]) > maximum_step:
                maximum_step = abs(step[coefficient])
        completed = iteration + 1
        if maximum_step <= tolerance:
            break
    return theta, completed


@njit(cache=True, fastmath=False)
def _ridge_newton_paired_training(
    standardized: np.ndarray,
    swaps: np.ndarray,
    fold_ids: np.ndarray,
    held_out_fold: int,
    ridge_lambda: float,
) -> np.ndarray:
    """Fit one permuted paired endpoint training set.

    ``standardized`` has block × endpoint × feature dimensions.  Rep 1 is
    endpoint 0 and Rep 15 is endpoint 1.  ``swaps[b]`` exchanges their labels
    within block ``b``.  All observations from the held-out blocks are
    excluded.
    """

    n_blocks, _, n_features = standardized.shape
    n_train = 0
    for block in range(n_blocks):
        if fold_ids[block] != held_out_fold:
            n_train += 2

    theta = np.zeros(n_features + 1, dtype=np.float64)
    gradient = np.zeros(n_features + 1, dtype=np.float64)
    hessian = np.zeros(
        (n_features + 1, n_features + 1), dtype=np.float64
    )

    for _ in range(50):
        gradient[:] = 0.0
        hessian[:] = 0.0
        for block in range(n_blocks):
            if fold_ids[block] == held_out_fold:
                continue
            swapped = swaps[block]
            for endpoint in range(2):
                target = (
                    1.0 - swapped if endpoint == 0 else float(swapped)
                )
                score = theta[n_features]
                for feature in range(n_features):
                    score += (
                        standardized[block, endpoint, feature]
                        * theta[feature]
                    )
                probability = 1.0 / (1.0 + np.exp(-score))
                residual = probability - target
                curvature = probability * (1.0 - probability)
                gradient[n_features] += residual
                hessian[n_features, n_features] += curvature
                for feature in range(n_features):
                    feature_value = standardized[
                        block, endpoint, feature
                    ]
                    gradient[feature] += feature_value * residual
                    hessian[n_features, feature] += (
                        curvature * feature_value
                    )
                    for other in range(feature + 1):
                        hessian[feature, other] += (
                            curvature
                            * feature_value
                            * standardized[block, endpoint, other]
                        )

        for feature in range(n_features):
            gradient[feature] = (
                gradient[feature] / n_train
                + ridge_lambda * theta[feature]
            )
            for other in range(feature + 1):
                hessian[feature, other] /= n_train
                hessian[other, feature] = hessian[feature, other]
            hessian[feature, feature] += ridge_lambda
            hessian[n_features, feature] /= n_train
            hessian[feature, n_features] = hessian[n_features, feature]
        gradient[n_features] /= n_train
        hessian[n_features, n_features] = (
            hessian[n_features, n_features] / n_train + 1e-12
        )
        step = np.linalg.solve(hessian, gradient)
        maximum_step = 0.0
        for coefficient in range(n_features + 1):
            theta[coefficient] -= step[coefficient]
            if abs(step[coefficient]) > maximum_step:
                maximum_step = abs(step[coefficient])
        if maximum_step <= 1e-8:
            break
    return theta


@njit(cache=True, fastmath=False)
def _observed_decoder(
    standardized: np.ndarray,
    fold_ids: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed grouped-CV accuracy and paired block correctness."""

    n_time, n_folds, n_blocks, _, n_features = standardized.shape
    swaps = np.zeros(n_blocks, dtype=np.uint8)
    block_correct = np.zeros((n_blocks, n_time), dtype=np.float64)
    for time_index in range(n_time):
        for fold in range(n_folds):
            theta = _ridge_newton_paired_training(
                standardized[time_index, fold],
                swaps,
                fold_ids,
                fold,
                ridge_lambda,
            )
            for block in range(n_blocks):
                if fold_ids[block] != fold:
                    continue
                for endpoint in range(2):
                    score = theta[n_features]
                    for feature in range(n_features):
                        score += (
                            standardized[
                                time_index,
                                fold,
                                block,
                                endpoint,
                                feature,
                            ]
                            * theta[feature]
                        )
                    predicted = 1 if score >= 0.0 else 0
                    target = 1 if endpoint == 0 else 0
                    if predicted == target:
                        block_correct[block, time_index] += 0.5
    accuracy = np.zeros(n_time, dtype=np.float64)
    for time_index in range(n_time):
        total = 0.0
        for block in range(n_blocks):
            total += block_correct[block, time_index]
        accuracy[time_index] = total / n_blocks
    return accuracy, block_correct


@njit(cache=True, fastmath=False, parallel=True)
def _permuted_decoder_accuracy(
    standardized: np.ndarray,
    fold_ids: np.ndarray,
    permutations: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """Refit all five grouped folds for every within-block label swap."""

    n_permutations = permutations.shape[0]
    n_time, n_folds, n_blocks, _, n_features = standardized.shape
    accuracy = np.empty(
        (n_permutations, n_time), dtype=np.float32
    )
    denominator = float(2 * n_blocks)
    for permutation in prange(n_permutations):
        swaps = permutations[permutation]
        for time_index in range(n_time):
            correct = 0
            for fold in range(n_folds):
                theta = _ridge_newton_paired_training(
                    standardized[time_index, fold],
                    swaps,
                    fold_ids,
                    fold,
                    ridge_lambda,
                )
                for block in range(n_blocks):
                    if fold_ids[block] != fold:
                        continue
                    swapped = swaps[block]
                    for endpoint in range(2):
                        score = theta[n_features]
                        for feature in range(n_features):
                            score += (
                                standardized[
                                    time_index,
                                    fold,
                                    block,
                                    endpoint,
                                    feature,
                                ]
                                * theta[feature]
                            )
                        predicted = 1 if score >= 0.0 else 0
                        target = (
                            1 - swapped if endpoint == 0 else swapped
                        )
                        if predicted == target:
                            correct += 1
            accuracy[permutation, time_index] = correct / denominator
    return accuracy


def _validate_fast_solver(
    x: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
) -> dict[str, Any]:
    """Benchmark the compiled solver against the committed BFGS solver."""

    import time

    x = np.asarray(x, dtype=np.float64)
    y_float = np.asarray(y, dtype=np.float64)
    # Compile before timing.
    fast_theta, iterations = _ridge_newton_dense(
        x, y_float, ridge_lambda
    )
    started = time.perf_counter()
    fast_theta, iterations = _ridge_newton_dense(
        x, y_float, ridge_lambda
    )
    fast_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reference_beta, reference_bias = fit_matlab_ridge_logistic(
        x, y_float.astype(int), ridge_lambda
    )
    reference_seconds = time.perf_counter() - started
    reference = np.r_[reference_beta, reference_bias]
    design = np.column_stack([x, np.ones(x.shape[0])])
    score_difference = design @ (fast_theta - reference)
    prediction_agreement = np.mean(
        (design @ fast_theta >= 0.0)
        == (design @ reference >= 0.0)
    )
    maximum_score_difference = float(
        np.max(np.abs(score_difference))
    )
    if maximum_score_difference > 1e-3 or prediction_agreement < 1.0:
        raise AssertionError(
            "Compiled ridge solver does not reproduce the committed BFGS "
            f"solution: max score delta={maximum_score_difference:.3g}, "
            f"prediction agreement={prediction_agreement:.6f}"
        )
    return {
        "objective": (
            "mean logistic deviance + lambda/2*||beta||^2; "
            "unpenalized bias"
        ),
        "reference_solver": (
            "ECoG.roving.decoder.fit_matlab_ridge_logistic (BFGS)"
        ),
        "fast_solver": "compiled Newton/IRLS, tolerance 1e-9",
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "iterations": int(iterations),
        "maximum_absolute_linear_score_difference": (
            maximum_score_difference
        ),
        "classification_prediction_agreement": float(
            prediction_agreement
        ),
        "fast_seconds": float(fast_seconds),
        "reference_seconds": float(reference_seconds),
        "speed_ratio_reference_over_fast": float(
            reference_seconds / max(fast_seconds, np.finfo(float).tiny)
        ),
    }


# ---------------------------------------------------------------------------
# ECoG preparation and decoder inference
# ---------------------------------------------------------------------------
def _fold_ids_for_recording(
    stimuli: np.ndarray,
    contexts: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    folds = _context_stratified_block_folds(
        np.asarray(stimuli),
        np.asarray(contexts),
        N_FOLDS,
        np.random.RandomState(random_seed),
    )
    ids = np.full(len(stimuli), -1, dtype=np.int64)
    for fold, blocks in enumerate(folds):
        if np.any(ids[blocks] >= 0):
            raise AssertionError("Grouped test folds overlap")
        ids[blocks] = fold
    if np.any(ids < 0):
        raise AssertionError("Grouped folds do not cover all blocks")
    return ids


def _standardize_endpoint_folds(
    endpoint: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    """Training-fold-only z-scoring for every time and fold.

    Parameters
    ----------
    endpoint
        block × endpoint × channel × time.
    """

    n_blocks, n_endpoint, n_features, n_time = endpoint.shape
    if n_endpoint != 2:
        raise ValueError("Endpoint array must contain Rep 1 and Rep 15")
    standardized = np.empty(
        (n_time, N_FOLDS, n_blocks, 2, n_features),
        dtype=np.float64,
    )
    for time_index in range(n_time):
        raw = endpoint[..., time_index]
        for fold in range(N_FOLDS):
            train = raw[fold_ids != fold].reshape(-1, n_features)
            mean = np.mean(train, axis=0)
            scale = np.std(train, axis=0, ddof=1) + SCALE_EPSILON
            standardized[time_index, fold] = (raw - mean) / scale
    return standardized


def _whole_block_bootstrap_ci(
    block_values: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval from transition-stratified block resampling.

    Every resample draws the original number of intact blocks with
    replacement inside each context→current stratum.  The resulting block
    weights are reused across all time points.
    """

    n_blocks = block_values.shape[0]
    strata = np.asarray(strata)
    if strata.shape != (n_blocks, 2):
        raise ValueError("Bootstrap strata must be block × 2")
    weights = np.zeros((n_bootstraps, n_blocks), dtype=np.int32)
    for transition in np.unique(strata, axis=0):
        indices = np.flatnonzero(
            np.all(strata == transition, axis=1)
        )
        weights[:, indices] = rng.multinomial(
            indices.size,
            np.full(indices.size, 1.0 / indices.size),
            size=n_bootstraps,
        )
    if np.any(np.sum(weights, axis=1) != n_blocks):
        raise AssertionError(
            "Transition-stratified bootstrap changed the block count"
        )
    bootstrapped = weights @ np.asarray(block_values, dtype=float)
    bootstrapped /= n_blocks
    low, high = np.quantile(
        bootstrapped, [ALPHA / 2.0, 1.0 - ALPHA / 2.0], axis=0
    )
    return low, high


def _clusters_1d_above_threshold(
    statistic: np.ndarray,
    threshold: np.ndarray,
) -> list[tuple[int, int, float]]:
    selected = np.asarray(statistic) > np.asarray(threshold)
    padded = np.r_[False, selected, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (
            int(start),
            int(stop),
            float(
                np.sum(
                    statistic[start:stop] - threshold[start:stop]
                )
            ),
        )
        for start, stop in zip(starts, stops)
    ]


def _joint_decoder_clusters(
    observed: np.ndarray,
    null: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """One-sided cluster-mass FWER correction over time and recordings."""

    n_positions, n_permutations, n_time = null.shape
    threshold = np.quantile(
        null, 1.0 - CLUSTER_FORMING_ALPHA, axis=1
    )
    null_maximum = np.zeros(n_permutations, dtype=float)
    for position in range(n_positions):
        for permutation in range(n_permutations):
            clusters = _clusters_1d_above_threshold(
                null[position, permutation], threshold[position]
            )
            if clusters:
                null_maximum[permutation] = max(
                    null_maximum[permutation],
                    max(item[2] for item in clusters),
                )

    significant = np.zeros((n_positions, n_time), dtype=bool)
    p_corrected = np.ones((n_positions, n_time), dtype=float)
    rows: list[dict[str, Any]] = []
    for position in range(n_positions):
        for start, stop, mass in _clusters_1d_above_threshold(
            observed[position], threshold[position]
        ):
            probability = (
                1.0 + np.count_nonzero(null_maximum >= mass)
            ) / (n_permutations + 1.0)
            p_corrected[position, start:stop] = probability
            significant[position, start:stop] = probability < ALPHA
            rows.append(
                {
                    "deviant_position": position + 1,
                    "start_ms": int(DECODER_TIME_MS[start]),
                    "end_ms": int(DECODER_TIME_MS[stop - 1]),
                    "cluster_mass": mass,
                    "p_corrected": probability,
                    "significant": probability < ALPHA,
                }
            )
    return significant, p_corrected, rows


def _extract_ecog_recording(
    position: int,
) -> dict[str, np.ndarray | float | str]:
    key = POSITION_KEYS[position - 1]
    spec = ANALYSES[key]
    epochs = extract_repetition_epochs(spec.data_path(), spec)
    source_time = np.asarray(epochs.time_ms, dtype=np.int64)
    time_indices = np.searchsorted(source_time, DECODER_TIME_MS)
    if not np.array_equal(source_time[time_indices], DECODER_TIME_MS):
        raise AssertionError(f"{key}: ECoG source does not contain 0:5:600 ms")
    endpoint = np.stack(
        [
            np.take(epochs.epochs[:, 0], time_indices, axis=-1),
            np.take(epochs.epochs[:, -1], time_indices, axis=-1),
        ],
        axis=1,
    )
    expected = (
        epochs.epochs.shape[0],
        2,
        spec.n_channels,
        DECODER_TIME_MS.size,
    )
    if endpoint.shape != expected:
        raise AssertionError(
            f"{key}: endpoint shape {endpoint.shape}, expected {expected}"
        )

    fold_ids = _fold_ids_for_recording(
        epochs.stimuli, epochs.contexts, spec.random_seed
    )
    deviant_time = source_time - spec.deviant_onset_ms
    erp_window = np.searchsorted(deviant_time, ERP_TIME_MS)
    if not np.array_equal(deviant_time[erp_window], ERP_TIME_MS):
        raise AssertionError(f"{key}: ECoG source does not contain 0:1:360 ms")

    channel_gfp = np.std(epochs.epochs, axis=2, ddof=0)
    response_window = (
        (source_time >= spec.deviant_onset_ms)
        & (source_time < spec.deviant_onset_ms + 180)
    )
    block_gfp = np.mean(channel_gfp[..., response_window], axis=-1)
    return {
        "key": key,
        "endpoint": np.asarray(endpoint, dtype=np.float64),
        "fold_ids": fold_ids,
        "block_gfp": np.asarray(block_gfp, dtype=np.float64),
        "stimuli": np.asarray(epochs.stimuli),
        "contexts": np.asarray(epochs.contexts),
        "ridge_lambda": float(spec.lambda_ridge),
        "n_blocks": np.asarray(epochs.epochs.shape[0], dtype=np.int64),
        # Block x channel x time, first and last repetition of every block.
        "erp_rep1": np.take(epochs.epochs[:, 0], erp_window, axis=-1).astype(
            np.float64
        ),
        "erp_rep15": np.take(epochs.epochs[:, -1], erp_window, axis=-1).astype(
            np.float64
        ),
    }


def _decoder_inference(
    recordings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    n_positions = len(recordings)
    observed = np.empty((n_positions, DECODER_TIME_MS.size), dtype=float)
    ci_low = np.empty_like(observed)
    ci_high = np.empty_like(observed)
    null = np.empty(
        (n_positions, N_RANDOMIZATIONS, DECODER_TIME_MS.size),
        dtype=np.float32,
    )
    fold_ids_all = np.empty(
        (n_positions, int(recordings[0]["n_blocks"])), dtype=np.int64
    )
    solver_validation: dict[str, Any] | None = None
    permutation_digests: dict[str, str] = {}

    for position_index, recording in enumerate(recordings):
        endpoint = np.asarray(recording["endpoint"], dtype=np.float64)
        fold_ids = np.asarray(recording["fold_ids"], dtype=np.int64)
        fold_ids_all[position_index] = fold_ids
        standardized = _standardize_endpoint_folds(endpoint, fold_ids)

        if solver_validation is None:
            validation_fold = 0
            raw = endpoint[..., 0]
            training = raw[fold_ids != validation_fold].reshape(
                -1, raw.shape[-1]
            )
            mean = np.mean(training, axis=0)
            scale = np.std(training, axis=0, ddof=1) + SCALE_EPSILON
            x = (training - mean) / scale
            n_training_blocks = np.count_nonzero(
                fold_ids != validation_fold
            )
            y = np.r_[
                np.ones(n_training_blocks, dtype=int),
                np.zeros(n_training_blocks, dtype=int),
            ]
            # The reshape order is block-major; match it explicitly.
            y = np.tile(np.array([1, 0], dtype=int), n_training_blocks)
            solver_validation = _validate_fast_solver(
                x, y, float(recording["ridge_lambda"])
            )

        accuracy, block_correct = _observed_decoder(
            standardized,
            fold_ids,
            float(recording["ridge_lambda"]),
        )
        observed[position_index] = accuracy
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence(
                [RANDOM_SEED, 10_000, position_index + 1]
            )
        )
        (
            ci_low[position_index],
            ci_high[position_index],
        ) = _whole_block_bootstrap_ci(
            block_correct,
            np.column_stack(
                [recording["contexts"], recording["stimuli"]]
            ),
            bootstrap_rng,
        )

        permutation_rng = np.random.default_rng(
            np.random.SeedSequence(
                [RANDOM_SEED, 20_000, position_index + 1]
            )
        )
        permutations = permutation_rng.integers(
            0,
            2,
            size=(N_RANDOMIZATIONS, endpoint.shape[0]),
            dtype=np.uint8,
        )
        permutation_digests[f"pos{position_index + 1}"] = (
            _array_sha256(permutations)
        )
        null[position_index] = _permuted_decoder_accuracy(
            standardized,
            fold_ids,
            permutations,
            float(recording["ridge_lambda"]),
        )
        del standardized, permutations

    significant, p_corrected, cluster_rows = _joint_decoder_clusters(
        observed, null
    )
    arrays: dict[str, np.ndarray] = {
        "decoder_time_ms": DECODER_TIME_MS.copy(),
        "decoder_accuracy": observed,
        "decoder_ci_low": ci_low,
        "decoder_ci_high": ci_high,
        "decoder_significant": significant,
        "decoder_p_corrected": p_corrected,
        "decoder_fold_ids": fold_ids_all,
    }
    for position in range(1, 4):
        index = position - 1
        arrays[f"decoder_pos{position}_accuracy"] = observed[index]
        arrays[f"decoder_pos{position}_ci_low"] = ci_low[index]
        arrays[f"decoder_pos{position}_ci_high"] = ci_high[index]
        arrays[f"decoder_pos{position}_significant"] = significant[index]
        arrays[f"decoder_pos{position}_p_corrected"] = p_corrected[index]
    provenance = {
        "measure": (
            "classification accuracy for actual Rep 1 versus actual Rep 15 "
            "from raw 32-channel endpoint features"
        ),
        "time_reference": (
            "sequence onset, not deviant onset; 0–600 ms at a declared "
            "5-ms inference grid"
        ),
        "cross_validation": (
            "five fixed context-stratified folds grouped by intact roving "
            "block; both endpoints from a block always share a fold"
        ),
        "scaling": (
            "channel mean and sample SD fitted only on training blocks, "
            "separately for every fold and time point; SD + 1e-6"
        ),
        "bootstrap": (
            "95% percentile interval from 4,999 paired whole-block resamples "
            "within each of the six context→current transition strata; one "
            "set of block weights is reused across all times. Decoder fits "
            "are held fixed, so this is conditional predictive uncertainty"
        ),
        "randomization": (
            "4,999 Monte Carlo within-block Rep-1/Rep-15 label swaps; all "
            "five folds and training-only scaling are refit for every swap "
            "and time point"
        ),
        "multiple_comparison": (
            "one-sided cluster mass with pointwise 95th-percentile null "
            "cluster-forming thresholds; maximum cluster jointly over all "
            "three recordings controls FWER at 0.05"
        ),
        "permutation_sha256": permutation_digests,
        "solver_validation": solver_validation,
    }
    return arrays, cluster_rows, provenance


# ---------------------------------------------------------------------------
# Posterior-map inference
# ---------------------------------------------------------------------------
_FOUR_CONNECTED = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)


def _smooth_posterior_evidence(evidence: np.ndarray) -> np.ndarray:
    smoothed = convolve1d(
        evidence,
        _matlab_gaussian_kernel(0.8),
        axis=1,
        mode="nearest",
    )
    return convolve1d(
        smoothed,
        _matlab_gaussian_kernel(3.0),
        axis=2,
        mode="nearest",
    )


def _t_statistic(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return np.divide(
        np.mean(values, axis=0),
        sem,
        out=np.zeros_like(sem),
        where=sem > 0,
    )


def _two_sided_clusters_2d(
    statistic: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for sign, selected in (
        (1, statistic > threshold),
        (-1, statistic < -threshold),
    ):
        labels, count = label(selected, structure=_FOUR_CONNECTED)
        for cluster_id in range(1, count + 1):
            mask = labels == cluster_id
            mass = float(np.sum(np.abs(statistic[mask]) - threshold))
            coordinates = np.argwhere(mask)
            clusters.append(
                {
                    "mask": mask,
                    "mass": mass,
                    "sign": sign,
                    "rep_start_index": int(np.min(coordinates[:, 0])),
                    "rep_stop_index": int(np.max(coordinates[:, 0])),
                    "time_start_index": int(np.min(coordinates[:, 1])),
                    "time_stop_index": int(np.max(coordinates[:, 1])),
                }
            )
    return clusters


def _posterior_inference() -> tuple[
    dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]
]:
    displays: list[np.ndarray] = []
    observed_statistics: list[np.ndarray] = []
    null_maximum = np.zeros(N_RANDOMIZATIONS, dtype=float)
    observed_clusters: list[list[dict[str, Any]]] = []
    evidence_digests: dict[str, str] = {}
    n_blocks_by_position: list[int] = []

    for position, key in enumerate(POSITION_KEYS, start=1):
        path = (
            _ECOG_RESULTS
            / key
            / "regression_rep_map"
            / "leakage-safe"
            / "regression_rep_map_arrays.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing leakage-safe map: {path}")
        with np.load(path, allow_pickle=False) as source:
            source_time = np.asarray(source["time_ms"], dtype=np.int64)
            time_indices = np.searchsorted(source_time, POSTERIOR_TIME_MS)
            if not np.array_equal(
                source_time[time_indices], POSTERIOR_TIME_MS
            ):
                raise AssertionError(
                    f"{key}: posterior source does not contain 0:5:600 ms"
                )
            display = np.asarray(
                source["posterior_smoothed"][:, time_indices],
                dtype=float,
            )
            probabilities = np.asarray(
                source["posterior_trials"][:, :, source_time <= 600],
                dtype=float,
            )
            probability_time = source_time[source_time <= 600]
        if not np.array_equal(probability_time, np.arange(601)):
            raise AssertionError(f"{key}: posterior full grid is not 0:600 ms")
        evidence = logit(
            np.clip(
                probabilities,
                POSTERIOR_LOGIT_CLIP,
                1.0 - POSTERIOR_LOGIT_CLIP,
            )
        )
        evidence = _smooth_posterior_evidence(evidence)[..., ::5]
        if evidence.shape[1:] != (
            POSTERIOR_REPETITIONS.size,
            POSTERIOR_TIME_MS.size,
        ):
            raise AssertionError(
                f"{key}: unexpected posterior evidence {evidence.shape}"
            )
        displays.append(display)
        observed = _t_statistic(evidence)
        observed_statistics.append(observed)
        n_blocks = evidence.shape[0]
        n_blocks_by_position.append(n_blocks)
        threshold = float(
            student_t.ppf(
                1.0 - CLUSTER_FORMING_ALPHA / 2.0,
                df=n_blocks - 1,
            )
        )
        observed_clusters.append(
            _two_sided_clusters_2d(observed, threshold)
        )

        sign_rng = np.random.default_rng(
            np.random.SeedSequence([RANDOM_SEED, 30_000, position])
        )
        signs = sign_rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(N_RANDOMIZATIONS, n_blocks),
        )
        evidence_digests[f"pos{position}"] = _array_sha256(evidence)
        flattened = evidence.reshape(n_blocks, -1)
        sums_of_squares = np.sum(flattened**2, axis=0)
        means = (signs @ flattened) / n_blocks
        variances = (
            sums_of_squares[None, :] - n_blocks * means**2
        ) / (n_blocks - 1)
        denominator = np.sqrt(
            np.maximum(variances, 0.0) / n_blocks
        )
        null_t = np.divide(
            means,
            denominator,
            out=np.zeros_like(means),
            where=denominator > 0,
        ).reshape(
            N_RANDOMIZATIONS,
            POSTERIOR_REPETITIONS.size,
            POSTERIOR_TIME_MS.size,
        )
        for permutation in range(N_RANDOMIZATIONS):
            clusters = _two_sided_clusters_2d(
                null_t[permutation], threshold
            )
            if clusters:
                null_maximum[permutation] = max(
                    null_maximum[permutation],
                    max(cluster["mass"] for cluster in clusters),
                )
        del probabilities, evidence, flattened, means, variances, null_t

    arrays: dict[str, np.ndarray] = {
        "posterior_time_ms": POSTERIOR_TIME_MS.copy(),
        "posterior_repetition": POSTERIOR_REPETITIONS.copy(),
    }
    rows: list[dict[str, Any]] = []
    for position in range(1, 4):
        statistic = observed_statistics[position - 1]
        significant = np.zeros_like(statistic, dtype=bool)
        p_corrected = np.ones_like(statistic, dtype=float)
        for cluster in observed_clusters[position - 1]:
            probability = (
                1.0
                + np.count_nonzero(null_maximum >= cluster["mass"])
            ) / (N_RANDOMIZATIONS + 1.0)
            mask = cluster["mask"]
            p_corrected[mask] = probability
            significant[mask] = probability < ALPHA
            rows.append(
                {
                    "deviant_position": position,
                    "sign": (
                        "positive" if cluster["sign"] > 0 else "negative"
                    ),
                    "repetition_start": int(
                        POSTERIOR_REPETITIONS[
                            cluster["rep_start_index"]
                        ]
                    ),
                    "repetition_end": int(
                        POSTERIOR_REPETITIONS[
                            cluster["rep_stop_index"]
                        ]
                    ),
                    "time_start_ms": int(
                        POSTERIOR_TIME_MS[
                            cluster["time_start_index"]
                        ]
                    ),
                    "time_end_ms": int(
                        POSTERIOR_TIME_MS[
                            cluster["time_stop_index"]
                        ]
                    ),
                    "cluster_mass": float(cluster["mass"]),
                    "p_corrected": probability,
                    "significant": probability < ALPHA,
                }
            )
        arrays[f"posterior_pos{position}_display"] = displays[position - 1]
        arrays[f"posterior_pos{position}_t"] = statistic
        arrays[f"posterior_pos{position}_significant"] = significant
        arrays[f"posterior_pos{position}_mask"] = significant
        arrays[f"posterior_pos{position}_p_corrected"] = p_corrected

    provenance = {
        "measure": (
            "cross-fitted block-level logit(P[Rep-1-like]) evidence from the "
            "committed leakage-safe repetition-map decoder"
        ),
        "display": (
            "committed posterior_smoothed probabilities, generated by exact "
            "MATLAB imgaussfilt-equivalent replicate-padded Gaussian "
            "smoothing (sigma repetition=0.8, sigma time=3 ms), then sampled "
            "at 5 ms"
        ),
        "inference_smoothing": (
            "the same exact repetition/time Gaussian operator is applied to "
            "each block's logit evidence on the native 1-ms grid before "
            "sampling at 5 ms"
        ),
        "randomization": (
            "4,999 whole-block Rademacher sign flips; one sign is shared by "
            "all 15 repetitions and all times within a block, preserving the "
            "complete within-block dependence structure"
        ),
        "multiple_comparison": (
            "two-sided 4-neighbor cluster mass, p<0.05 two-sided t "
            "cluster-forming threshold, maximum cluster jointly across all "
            "three maps; corrected Monte Carlo p=(b+1)/(4,999+1)"
        ),
        "scope_and_compromise": (
            "This is a cross-fitted evidence sign-flip test, not a refitted "
            "15-repetition decoder permutation. Refitting all endpoint folds "
            "for every repetition-map cell would redefine intermediate "
            "posterior predictions and is computationally prohibitive. The "
            "test is valid under block-level symmetry of cross-fitted logits."
        ),
        "n_blocks_by_position": n_blocks_by_position,
        "evidence_sha256": evidence_digests,
    }
    return arrays, rows, provenance


# ---------------------------------------------------------------------------
# Paired sign-flip and max-T helpers
# ---------------------------------------------------------------------------
def _random_signs(
    n_randomizations: int,
    n_units: int,
    seed_parts: Sequence[int],
) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence([RANDOM_SEED, *seed_parts])
    )
    return rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_randomizations, n_units),
    )


def _all_exact_signs(n_units: int) -> np.ndarray:
    integers = np.arange(2**n_units, dtype=np.uint64)[:, None]
    bits = (integers >> np.arange(n_units, dtype=np.uint64)) & 1
    return (2.0 * bits.astype(float)) - 1.0


def _signflip_t(
    values: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Studentized sign-flip statistics.

    ``values`` is unit × tests; the output is randomization × tests.
    """

    values = np.asarray(values, dtype=float)
    n_units = values.shape[0]
    flattened = values.reshape(n_units, -1)
    means = signs @ flattened / n_units
    sum_squares = np.sum(flattened**2, axis=0)
    variance = (
        sum_squares[None, :] - n_units * means**2
    ) / (n_units - 1)
    sem = np.sqrt(np.maximum(variance, 0.0) / n_units)
    statistic = np.divide(
        means,
        sem,
        out=np.zeros_like(means),
        where=sem > 0,
    )
    return statistic.reshape((signs.shape[0],) + values.shape[1:])


def _max_t_corrected(
    values: np.ndarray,
    signs: np.ndarray,
    *,
    test_mask: np.ndarray | None = None,
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = _t_statistic(values)
    null = _signflip_t(values, signs)
    if test_mask is None:
        tested_observed = observed.reshape(-1)
        tested_null = null.reshape(null.shape[0], -1)
    else:
        test_mask = np.asarray(test_mask, dtype=bool)
        tested_observed = observed[test_mask]
        tested_null = null[:, test_mask]
    maximum = np.max(np.abs(tested_null), axis=1)
    if exact:
        corrected_tested = np.array(
            [
                np.count_nonzero(
                    maximum
                    >= abs(value)
                    - 1e-10 * max(1.0, abs(value))
                )
                / len(maximum)
                for value in tested_observed
            ],
            dtype=float,
        )
    else:
        corrected_tested = np.array(
            [
                (
                    np.count_nonzero(maximum >= abs(value)) + 1.0
                )
                / (len(maximum) + 1.0)
                for value in tested_observed
            ],
            dtype=float,
        )
    p_corrected = np.ones_like(observed, dtype=float)
    if test_mask is None:
        p_corrected[...] = corrected_tested.reshape(observed.shape)
    else:
        p_corrected[test_mask] = corrected_tested
    return observed, p_corrected, p_corrected < ALPHA


def _erp_gfp_contrast(
    rep1: np.ndarray,
    rep15: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Sign-flip null for the plotted Rep-1 minus Rep-15 GFP difference.

    The statistic evaluated here is *exactly* the quantity panel B draws: the
    global field power of the block-averaged Rep-1 evoked response minus that
    of the Rep-15 response.  It is not a per-block summary that happens to
    resemble the plotted curve.

    Under the null, the Rep-1/Rep-15 label is exchangeable *within* a block,
    so flipping the sign of a block's within-block difference is a valid
    relabelling.  Writing ``rep1 = M + D`` and ``rep15 = M - D`` with
    ``M = (rep1 + rep15) / 2`` and ``D = (rep1 - rep15) / 2``, a relabelling
    with signs ``s`` leaves the block mean ``M`` untouched and maps the mean
    difference to ``mean(s * D)``.  Every randomisation is then one matrix
    product, and ``signs = +1`` reproduces the observed curve exactly.
    """

    n_blocks = rep1.shape[0]
    block_mean = 0.5 * (rep1 + rep15).mean(axis=0)
    half_difference = 0.5 * (rep1 - rep15)
    flattened = half_difference.reshape(n_blocks, -1)
    shape = half_difference.shape[1:]

    statistic = np.empty((signs.shape[0], rep1.shape[-1]), dtype=np.float64)
    for start in range(0, signs.shape[0], ERP_CHUNK):
        stop = min(start + ERP_CHUNK, signs.shape[0])
        drawn = (signs[start:stop] @ flattened) / n_blocks
        drawn = drawn.reshape((stop - start,) + shape)
        statistic[start:stop] = (
            np.std(block_mean + drawn, axis=1, ddof=0)
            - np.std(block_mean - drawn, axis=1, ddof=0)
        )
    return statistic


def _erp_inference(
    recordings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Cluster-mass FWER test for the evoked Rep-1 versus Rep-15 difference.

    Two-sided, jointly corrected over time *and* over the three deviant
    positions, so the reported probabilities cover the whole family the panel
    displays rather than one trace at a time.
    """

    n_positions = len(recordings)
    n_time = ERP_TIME_MS.size
    observed = np.empty((n_positions, n_time), dtype=float)
    z_observed = np.empty_like(observed)
    z_null = np.empty(
        (n_positions, N_RANDOMIZATIONS, n_time), dtype=np.float32
    )
    thresholds = np.empty(n_positions, dtype=float)
    block_counts = np.empty(n_positions, dtype=np.int64)

    for index, recording in enumerate(recordings):
        rep1 = np.asarray(recording["erp_rep1"], dtype=np.float64)
        rep15 = np.asarray(recording["erp_rep15"], dtype=np.float64)
        n_blocks = rep1.shape[0]
        block_counts[index] = n_blocks

        signs = _random_signs(N_RANDOMIZATIONS, n_blocks, (7, index))
        null = _erp_gfp_contrast(rep1, rep15, signs)
        observed[index] = _erp_gfp_contrast(
            rep1, rep15, np.ones((1, n_blocks), dtype=float)
        )[0]

        # Variance-normalise against the randomisation distribution so the
        # cluster-forming threshold means the same thing at every latency;
        # the identical transform is applied to the observed curve and to
        # every randomisation, so max-statistic control is unaffected.
        centre = null.mean(axis=0)
        scale = null.std(axis=0, ddof=1)
        scale = np.where(scale > 0, scale, np.inf)
        z_observed[index] = (observed[index] - centre) / scale
        z_null[index] = ((null - centre) / scale).astype(np.float32)
        thresholds[index] = float(
            np.quantile(np.abs(z_null[index]), 1.0 - CLUSTER_FORMING_ALPHA)
        )

    # Joint null: the largest cluster mass anywhere in the family, per
    # randomisation.
    null_maximum = np.zeros(N_RANDOMIZATIONS, dtype=float)
    for index in range(n_positions):
        threshold = thresholds[index]
        for permutation in range(N_RANDOMIZATIONS):
            clusters = _clusters_1d_two_sided_t(
                z_null[index, permutation].astype(np.float64), threshold
            )
            if clusters:
                null_maximum[permutation] = max(
                    null_maximum[permutation],
                    max(item["mass"] for item in clusters),
                )

    significant = np.zeros((n_positions, n_time), dtype=bool)
    p_corrected = np.ones((n_positions, n_time), dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(n_positions):
        for cluster in _clusters_1d_two_sided_t(
            z_observed[index], thresholds[index]
        ):
            start, stop = cluster["start"], cluster["stop"]
            probability = (
                1.0 + np.count_nonzero(null_maximum >= cluster["mass"])
            ) / (N_RANDOMIZATIONS + 1.0)
            p_corrected[index, start:stop] = probability
            significant[index, start:stop] = probability < ALPHA
            rows.append(
                {
                    "deviant_position": index + 1,
                    "start_ms": int(ERP_TIME_MS[start]),
                    "end_ms": int(ERP_TIME_MS[stop - 1]),
                    "direction": (
                        "rep1_above_rep15"
                        if cluster["sign"] > 0
                        else "rep15_above_rep1"
                    ),
                    "cluster_mass": cluster["mass"],
                    "p_corrected": probability,
                    "significant": probability < ALPHA,
                }
            )

    arrays: dict[str, np.ndarray] = {
        "erp_time_ms": ERP_TIME_MS.copy(),
        "erp_significant": significant,
        "erp_p_corrected": p_corrected,
        "erp_difference": observed,
        "erp_z": z_observed,
        "erp_cluster_threshold": thresholds,
    }
    for index in range(n_positions):
        position = index + 1
        arrays[f"erp_pos{position}_significant"] = significant[index]
        arrays[f"erp_pos{position}_p_corrected"] = p_corrected[index]
        arrays[f"erp_pos{position}_difference"] = observed[index]
        arrays[f"erp_pos{position}_z"] = z_observed[index]

    provenance = {
        "question": (
            "Does the evoked response to the first repetition differ from "
            "the fifteenth, at any latency in 0-360 ms?"
        ),
        "statistic": (
            "GFP of the block-averaged Rep-1 evoked response minus GFP of "
            "the Rep-15 response - the quantity panel B draws - "
            "variance-normalised at each latency against its own "
            "randomisation distribution."
        ),
        "resampling_unit": "intact roving block",
        "n_blocks_per_recording": block_counts.tolist(),
        "null": (
            "Rep-1/Rep-15 labels are exchangeable within a block; the null "
            "flips the sign of each block's within-block difference. Blocks "
            "are never split, so within-block dependence is preserved."
        ),
        "n_randomizations": N_RANDOMIZATIONS,
        "cluster_forming_threshold_abs_z": thresholds.tolist(),
        "cluster_statistic": "sum of |z| above threshold, two-sided",
        "multiplicity": (
            "Family-wise error controlled by the maximum cluster mass over "
            "all latencies and all three deviant positions jointly."
        ),
        "alpha": ALPHA,
        "scope": (
            "Three recordings from one animal. This is a within-recording "
            "randomisation test and does not license population inference "
            "across animals."
        ),
    }
    return arrays, rows, provenance


def _ecog_buildup_inference(
    recordings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    response_values = np.stack(
        [np.asarray(item["block_gfp"], dtype=float) for item in recordings],
        axis=0,
    )
    n_positions, n_blocks, n_repetitions = response_values.shape
    rep1 = response_values[:, :, [0]]
    if np.any(~np.isfinite(rep1)) or np.any(rep1 <= 0):
        raise ValueError(
            "Every ECoG block must have a finite positive Rep-1 GFP "
            "response before within-block percentage normalization"
        )
    # Panel G estimand: normalize *inside each inferential unit* before any
    # averaging, so a high-amplitude block cannot dominate the percentage.
    percent_change = 100.0 * (response_values / rep1 - 1.0)
    mean = np.mean(percent_change, axis=1)
    sem = (
        np.std(percent_change, axis=1, ddof=1) / np.sqrt(n_blocks)
    )
    # Independent within-recording flips, concatenated into a block-diagonal
    # randomization by calculating each position and taking a joint maximum.
    observed = np.empty((n_positions, n_repetitions), dtype=float)
    null = np.empty(
        (N_RANDOMIZATIONS, n_positions, n_repetitions), dtype=float
    )
    for position in range(n_positions):
        position_values = percent_change[position]
        observed[position] = _t_statistic(position_values)
        signs = _random_signs(
            N_RANDOMIZATIONS,
            n_blocks,
            [40_000, position + 1],
        )
        null[:, position] = _signflip_t(position_values, signs)
    maximum = np.max(np.abs(null[:, :, 1:]), axis=(1, 2))
    p_corrected = np.ones_like(observed)
    for position in range(n_positions):
        for repetition in range(1, n_repetitions):
            p_corrected[position, repetition] = (
                np.count_nonzero(
                    maximum >= abs(observed[position, repetition])
                )
                + 1.0
            ) / (N_RANDOMIZATIONS + 1.0)
    significant = p_corrected < ALPHA
    arrays: dict[str, np.ndarray] = {
        "buildup_repetition": np.arange(
            1, n_repetitions + 1, dtype=np.int64
        ),
        "ecog_buildup_response_values": response_values,
        "ecog_buildup_values": percent_change,
        "ecog_buildup_percent_values": percent_change,
        "ecog_buildup_mean": mean,
        "ecog_buildup_sem": sem,
        "ecog_buildup_percent_mean": mean,
        "ecog_buildup_percent_sem": sem,
        "ecog_buildup_change_from_rep1_mean": mean,
        "ecog_buildup_change_from_rep1_sem": sem,
        "ecog_buildup_t": observed,
        "ecog_buildup_p_corrected": p_corrected,
        "ecog_buildup_significant": significant,
    }
    for position in range(1, 4):
        index = position - 1
        arrays[f"ecog_buildup_pos{position}_mean"] = mean[index]
        arrays[f"ecog_buildup_pos{position}_sem"] = sem[index]
        arrays[f"ecog_buildup_pos{position}_significant"] = (
            significant[index]
        )
        arrays[f"ecog_buildup_pos{position}_p_corrected"] = (
            p_corrected[index]
        )
    return (
        arrays,
        {
            "measure": (
                "mean single-block global field power over the prespecified "
                "0–180 ms post-deviant window, transformed within each block "
                "as 100*(Rep-r response / Rep-1 response - 1)"
            ),
            "uncertainty": (
                "SEM of within-block percentage changes across intact roving "
                "blocks within each recording"
            ),
            "test": (
                "within-block percentage change versus zero, 4,999 whole-"
                "block sign flips, two-sided max-|t| correction jointly over "
                "repetitions 2–15 and all three recordings"
            ),
            "scope": (
                "within-recording inference for three recordings from one "
                "animal; not a biological population inference"
            ),
        },
    )


# ---------------------------------------------------------------------------
# RNN inference
# ---------------------------------------------------------------------------
def _sequence_model_responses(
    model: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if "sequence_responses" in model:
        responses = np.asarray(model["sequence_responses"], dtype=float)
        time = np.asarray(model["sequence_time_ms"], dtype=np.int64)
        return responses, time

    # Compatibility with the first cache schema.  The variable-tone channel
    # is silent outside its own event; place the available deviant-aligned
    # epoch into sequence coordinates and retain NaN where it was not saved.
    full = np.asarray(model["responses_full"], dtype=float)
    full_time = np.asarray(model["time_ms_full"], dtype=np.int64)
    positions = np.asarray(model["positions"], dtype=int)
    time = np.arange(601, dtype=np.int64)
    responses = np.full(full.shape[:-1] + (time.size,), np.nan, dtype=float)
    deviant_onsets = {1: 0, 2: 180, 3: 360}
    for position_index, position in enumerate(positions):
        destination_time = full_time + deviant_onsets[int(position)]
        valid = (destination_time >= 0) & (destination_time < time.size)
        responses[:, position_index, ..., destination_time[valid]] = (
            full[:, position_index, ..., valid]
        )
    return responses, time


def _clusters_1d_two_sided_t(
    statistic: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for sign, selected in (
        (1, statistic > threshold),
        (-1, statistic < -threshold),
    ):
        padded = np.r_[False, selected, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            clusters.append(
                {
                    "start": int(start),
                    "stop": int(stop),
                    "sign": sign,
                    "mass": float(
                        np.sum(
                            np.abs(statistic[start:stop]) - threshold
                        )
                    ),
                }
            )
    return clusters


def _rnn_inference(
    model: Mapping[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    conditions = _condition_names(model)
    sequence, sequence_time = _sequence_model_responses(model)
    n_conditions, n_positions, n_seeds, _, n_time = sequence.shape
    if n_positions != 3:
        raise ValueError("Figure 2 requires three simulated positions")
    if not np.array_equal(sequence_time, np.arange(n_time)):
        raise AssertionError("Model sequence time must start at 0 with 1-ms steps")
    inference_indices = np.flatnonzero(sequence_time % 5 == 0)
    inference_time = sequence_time[inference_indices]

    means = np.empty(
        (n_conditions, n_positions, 2, n_time), dtype=float
    )
    sems = np.empty_like(means)
    difference_mean = np.empty(
        (n_conditions, n_positions, n_time), dtype=float
    )
    difference_sem = np.empty_like(difference_mean)
    observed_t = np.empty(
        (n_conditions, n_positions, inference_time.size), dtype=float
    )
    exact_signs = _all_exact_signs(n_seeds)
    null_t = np.empty(
        (
            exact_signs.shape[0],
            n_conditions,
            n_positions,
            inference_time.size,
        ),
        dtype=float,
    )
    threshold = float(
        student_t.ppf(
            1.0 - CLUSTER_FORMING_ALPHA / 2.0, df=n_seeds - 1
        )
    )

    for condition in range(n_conditions):
        for position in range(n_positions):
            endpoints = sequence[condition, position, :, [0, -1], :]
            # Advanced indexing may move endpoint ahead of seed.
            if endpoints.shape[0] == 2:
                endpoints = np.swapaxes(endpoints, 0, 1)
            means[condition, position] = np.nanmean(endpoints, axis=0)
            sems[condition, position] = (
                np.nanstd(endpoints, axis=0, ddof=1) / np.sqrt(n_seeds)
            )
            difference = endpoints[:, 0] - endpoints[:, 1]
            difference_mean[condition, position] = np.nanmean(
                difference, axis=0
            )
            difference_sem[condition, position] = (
                np.nanstd(difference, axis=0, ddof=1) / np.sqrt(n_seeds)
            )
            infer_difference = difference[:, inference_indices]
            if np.any(~np.isfinite(infer_difference)):
                # Old cache compatibility: only test times actually saved.
                finite = np.all(np.isfinite(infer_difference), axis=0)
                filled = np.zeros_like(infer_difference)
                filled[:, finite] = infer_difference[:, finite]
                infer_difference = filled
            observed_t[condition, position] = _t_statistic(
                infer_difference
            )
            null_t[:, condition, position] = _signflip_t(
                infer_difference, exact_signs
            )

    if "intact" not in conditions:
        raise ValueError("RNN conditions must include 'intact'")
    intact_index = conditions.index("intact")
    null_maximum = np.zeros(exact_signs.shape[0], dtype=float)
    for permutation in range(exact_signs.shape[0]):
        for position in range(n_positions):
            clusters = _clusters_1d_two_sided_t(
                null_t[permutation, intact_index, position], threshold
            )
            if clusters:
                null_maximum[permutation] = max(
                    null_maximum[permutation],
                    max(cluster["mass"] for cluster in clusters),
                )

    significant = np.zeros_like(observed_t, dtype=bool)
    p_corrected = np.ones_like(observed_t, dtype=float)
    cluster_rows: list[dict[str, Any]] = []
    for position in range(n_positions):
        clusters = _clusters_1d_two_sided_t(
            observed_t[intact_index, position], threshold
        )
        for cluster in clusters:
            tolerance = 1e-10 * max(1.0, cluster["mass"])
            probability = np.mean(
                null_maximum >= cluster["mass"] - tolerance
            )
            start, stop = cluster["start"], cluster["stop"]
            p_corrected[intact_index, position, start:stop] = probability
            significant[intact_index, position, start:stop] = (
                probability < ALPHA
            )
            cluster_rows.append(
                {
                    "condition": conditions[intact_index],
                    "deviant_position": position + 1,
                    "sign": (
                        "positive"
                        if cluster["sign"] > 0
                        else "negative"
                    ),
                    "start_ms": int(inference_time[start]),
                    "end_ms": int(inference_time[stop - 1]),
                    "cluster_mass": float(cluster["mass"]),
                    "p_corrected": float(probability),
                    "significant": probability < ALPHA,
                }
            )

    window = np.asarray(model["window_response"], dtype=float)
    # Position is a repeated simulation factor; average positions within
    # session/order seed before treating seeds as the inferential units.
    buildup_response_values = np.mean(
        window, axis=1
    )  # condition × seed × repetition
    buildup_rep1 = buildup_response_values[..., [0]]
    if np.any(~np.isfinite(buildup_rep1)) or np.any(buildup_rep1 <= 0):
        raise ValueError(
            "Every simulated seed must have a finite positive position-"
            "averaged Rep-1 response before percentage normalization"
        )
    # Required Panel-G estimand: first average the repeated position factor
    # within each seed, then normalize inside that seed.
    buildup_values = 100.0 * (
        buildup_response_values / buildup_rep1 - 1.0
    )
    buildup_mean = np.mean(buildup_values, axis=1)
    buildup_sem = (
        np.std(buildup_values, axis=1, ddof=1) / np.sqrt(n_seeds)
    )
    unit_first_change = np.moveaxis(buildup_values, 1, 0)
    test_mask = np.ones(
        (n_conditions, buildup_values.shape[-1]), dtype=bool
    )
    test_mask[:, 0] = False
    buildup_t, buildup_p, buildup_sig = _max_t_corrected(
        unit_first_change,
        exact_signs,
        test_mask=test_mask,
        exact=True,
    )

    suppression = np.asarray(model["suppression_index"], dtype=float)
    suppression_values = np.mean(suppression, axis=1)  # condition × seed
    suppression_mean = np.mean(suppression_values, axis=1)
    suppression_sem = (
        np.std(suppression_values, axis=1, ddof=1) / np.sqrt(n_seeds)
    )
    ci_multiplier = float(
        student_t.ppf(1.0 - ALPHA / 2.0, df=n_seeds - 1)
    )
    suppression_ci_low = suppression_mean - (
        ci_multiplier * suppression_sem
    )
    suppression_ci_high = suppression_mean + (
        ci_multiplier * suppression_sem
    )
    suppression_unit_first = suppression_values.T
    suppression_t, suppression_p, suppression_sig = _max_t_corrected(
        suppression_unit_first, exact_signs, exact=True
    )
    suppression_vs_intact_values = (
        suppression_values - suppression_values[[intact_index]]
    )
    suppression_vs_intact_mean = np.mean(
        suppression_vs_intact_values, axis=1
    )
    suppression_vs_intact_sem = (
        np.std(suppression_vs_intact_values, axis=1, ddof=1)
        / np.sqrt(n_seeds)
    )
    suppression_vs_intact_ci_low = (
        suppression_vs_intact_mean
        - ci_multiplier * suppression_vs_intact_sem
    )
    suppression_vs_intact_ci_high = (
        suppression_vs_intact_mean
        + ci_multiplier * suppression_vs_intact_sem
    )
    contrast_unit_first = suppression_vs_intact_values.T
    contrast_mask = np.ones(n_conditions, dtype=bool)
    contrast_mask[intact_index] = False
    (
        suppression_vs_intact_t,
        suppression_vs_intact_p,
        suppression_vs_intact_sig,
    ) = _max_t_corrected(
        contrast_unit_first,
        exact_signs,
        test_mask=contrast_mask,
        exact=True,
    )

    arrays: dict[str, np.ndarray] = {
        "rnn_trace_time_ms": sequence_time.copy(),
        "rnn_trace_inference_time_ms": inference_time.copy(),
        "rnn_trace_mean": means,
        "rnn_trace_sem": sems,
        "rnn_trace_difference_mean": difference_mean,
        "rnn_trace_difference_sem": difference_sem,
        "rnn_trace_t": observed_t,
        "rnn_trace_significant": significant,
        "rnn_trace_p_corrected": p_corrected,
        "rnn_buildup_response_values": buildup_response_values,
        "rnn_buildup_values": buildup_values,
        "rnn_buildup_percent_values": buildup_values,
        "rnn_buildup_mean": buildup_mean,
        "rnn_buildup_sem": buildup_sem,
        "rnn_buildup_percent_mean": buildup_mean,
        "rnn_buildup_percent_sem": buildup_sem,
        "rnn_change_mean": buildup_mean,
        "rnn_change_sem": buildup_sem,
        "rnn_change_t": buildup_t,
        "rnn_change_p_corrected": buildup_p,
        "rnn_change_significant": buildup_sig,
        "rnn_suppression_values": suppression_values,
        "rnn_suppression_mean": suppression_mean,
        "rnn_suppression_sem": suppression_sem,
        "rnn_suppression_ci_low": suppression_ci_low,
        "rnn_suppression_ci_high": suppression_ci_high,
        "rnn_suppression_t": suppression_t,
        "rnn_suppression_p_corrected": suppression_p,
        "rnn_suppression_significant": suppression_sig,
        "rnn_suppression_vs_intact_values": (
            suppression_vs_intact_values
        ),
        "rnn_suppression_vs_intact_mean": suppression_vs_intact_mean,
        "rnn_suppression_vs_intact_sem": suppression_vs_intact_sem,
        "rnn_suppression_vs_intact_ci_low": (
            suppression_vs_intact_ci_low
        ),
        "rnn_suppression_vs_intact_ci_high": (
            suppression_vs_intact_ci_high
        ),
        "rnn_suppression_vs_intact_t": suppression_vs_intact_t,
        "rnn_suppression_vs_intact_p_corrected": (
            suppression_vs_intact_p
        ),
        "rnn_suppression_vs_intact_significant": (
            suppression_vs_intact_sig
        ),
    }
    for condition_index, condition in enumerate(conditions):
        for position in range(1, 4):
            pos_index = position - 1
            prefix = f"rnn_trace_{condition}_pos{position}"
            arrays[f"{prefix}_mean"] = means[
                condition_index, pos_index
            ]
            arrays[f"{prefix}_sem"] = sems[
                condition_index, pos_index
            ]
            arrays[f"{prefix}_rep1_mean"] = means[
                condition_index, pos_index, 0
            ]
            arrays[f"{prefix}_rep1_sem"] = sems[
                condition_index, pos_index, 0
            ]
            arrays[f"{prefix}_rep15_mean"] = means[
                condition_index, pos_index, 1
            ]
            arrays[f"{prefix}_rep15_sem"] = sems[
                condition_index, pos_index, 1
            ]
            arrays[f"{prefix}_difference_mean"] = difference_mean[
                condition_index, pos_index
            ]
            arrays[f"{prefix}_difference_sem"] = difference_sem[
                condition_index, pos_index
            ]
            arrays[f"{prefix}_significant"] = significant[
                condition_index, pos_index
            ]
            arrays[f"{prefix}_p_corrected"] = p_corrected[
                condition_index, pos_index
            ]
        arrays[f"rnn_change_{condition}_mean"] = buildup_mean[
            condition_index
        ]
        arrays[f"rnn_change_{condition}_sem"] = buildup_sem[
            condition_index
        ]
        arrays[f"rnn_change_{condition}_significant"] = buildup_sig[
            condition_index
        ]
        arrays[f"rnn_change_{condition}_p_corrected"] = buildup_p[
            condition_index
        ]

    point_rows: list[dict[str, Any]] = []
    repetitions = np.asarray(model["repetitions"], dtype=int)
    for condition_index, condition in enumerate(conditions):
        for repetition_index, repetition in enumerate(repetitions):
            point_rows.append(
                {
                    "analysis": "rnn_percent_change_from_rep1",
                    "condition": condition,
                    "repetition": int(repetition),
                    "mean": float(
                        arrays["rnn_change_mean"][
                            condition_index, repetition_index
                        ]
                    ),
                    "sem": float(
                        arrays["rnn_change_sem"][
                            condition_index, repetition_index
                        ]
                    ),
                    "p_corrected": float(
                        buildup_p[condition_index, repetition_index]
                    ),
                    "significant": bool(
                        buildup_sig[condition_index, repetition_index]
                    ),
                }
            )
        point_rows.append(
            {
                "analysis": "rnn_suppression_index_vs_zero",
                "condition": condition,
                "repetition": "",
                "mean": float(suppression_mean[condition_index]),
                "sem": float(suppression_sem[condition_index]),
                "p_corrected": float(suppression_p[condition_index]),
                "significant": bool(suppression_sig[condition_index]),
            }
        )
        point_rows.append(
            {
                "analysis": "rnn_suppression_index_vs_intact",
                "condition": condition,
                "repetition": "",
                "mean": float(
                    suppression_vs_intact_mean[condition_index]
                ),
                "sem": float(
                    suppression_vs_intact_sem[condition_index]
                ),
                "p_corrected": float(
                    suppression_vs_intact_p[condition_index]
                ),
                "significant": bool(
                    suppression_vs_intact_sig[condition_index]
                ),
            }
        )

    provenance = {
        "replication_unit": (
            "simulated session/order seed; positions are averaged within "
            "seed for buildup and suppression summaries"
        ),
        "trace_uncertainty": "SEM across eight session/order seeds",
        "trace_test": (
            "Rep-1 minus Rep-15 paired seed differences at a 5-ms inference "
            "grid; exhaustive 2^8 two-sided seed sign flips; cluster-mass "
            "maximum jointly over the three displayed intact-condition "
            "positions and times"
        ),
        "buildup_test": (
            "within each seed, positions are averaged first and each "
            "repetition is transformed as 100*(Rr/R1-1); exhaustive 2^8 "
            "sign flips test percentage change versus zero with two-sided "
            "max-|t| jointly over all four conditions and repetitions 2–15"
        ),
        "suppression_test": (
            "position-averaged seed-level suppression index versus zero; "
            "exhaustive 2^8 sign flips; two-sided max-|t| over four "
            "conditions. Paired perturbation-minus-intact contrasts are "
            "additionally tested with exhaustive 2^8 sign flips and a "
            "two-sided max-|t| over the three perturbations."
        ),
        "suppression_interval": (
            "two-sided 95% Student-t interval across eight seed-level, "
            "position-averaged values"
        ),
        "exact_randomizations": int(exact_signs.shape[0]),
        "old_cache_fallback": (
            "If sequence_responses is absent, responses_full is placed into "
            "sequence coordinates and unavailable samples remain NaN; no "
            "inference is assigned to unavailable samples."
        ),
    }
    return arrays, cluster_rows, point_rows, provenance


# ---------------------------------------------------------------------------
# Exports and public API
# ---------------------------------------------------------------------------
def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".csv", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_exports(
    data_dir: Path,
    erp_rows: Sequence[Mapping[str, Any]],
    decoder_rows: Sequence[Mapping[str, Any]],
    posterior_rows: Sequence[Mapping[str, Any]],
    rnn_rows: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    erp_path = data_dir / ERP_CLUSTERS_CSV
    decoder_path = data_dir / DECODER_CLUSTERS_CSV
    posterior_path = data_dir / POSTERIOR_CLUSTERS_CSV
    rnn_path = data_dir / RNN_CLUSTERS_CSV
    point_path = data_dir / POINTWISE_CSV
    _write_csv(
        erp_path,
        erp_rows,
        [
            "deviant_position",
            "start_ms",
            "end_ms",
            "direction",
            "cluster_mass",
            "p_corrected",
            "significant",
        ],
    )
    _write_csv(
        decoder_path,
        decoder_rows,
        [
            "deviant_position",
            "start_ms",
            "end_ms",
            "cluster_mass",
            "p_corrected",
            "significant",
        ],
    )
    _write_csv(
        posterior_path,
        posterior_rows,
        [
            "deviant_position",
            "sign",
            "repetition_start",
            "repetition_end",
            "time_start_ms",
            "time_end_ms",
            "cluster_mass",
            "p_corrected",
            "significant",
        ],
    )
    _write_csv(
        rnn_path,
        rnn_rows,
        [
            "condition",
            "deviant_position",
            "sign",
            "start_ms",
            "end_ms",
            "cluster_mass",
            "p_corrected",
            "significant",
        ],
    )
    _write_csv(
        point_path,
        point_rows,
        [
            "analysis",
            "condition",
            "repetition",
            "mean",
            "sem",
            "p_corrected",
            "significant",
        ],
    )
    return [erp_path, decoder_path, posterior_path, rnn_path, point_path]


def _cache_matches(
    npz_path: Path,
    provenance_path: Path,
    expected_analysis_id: str,
) -> bool:
    if not npz_path.exists() or not provenance_path.exists():
        return False
    try:
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("analysis_id") != expected_analysis_id:
            return False
        if provenance.get("npz_sha256") != _sha256(npz_path):
            return False
        with np.load(npz_path, allow_pickle=False) as archive:
            return (
                str(archive["analysis_id"]) == expected_analysis_id
                and "decoder_time_ms" in archive.files
                and "posterior_time_ms" in archive.files
                and "rnn_suppression_values" in archive.files
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def load_or_build_inference(
    ecog: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    force: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load or compute every inferential quantity used in Figure 2.

    Parameters
    ----------
    ecog, model
        Dictionaries returned by ``build_ecog_cache`` and
        ``load_or_build_model_data``.  Raw ECoG endpoint features and
        cross-fitted posterior trials are independently re-extracted so that
        no figure-ready mean can silently replace the resampling unit.
    force
        Recompute even when the analysis ID and output hash match.
    data_dir
        Output directory; defaults to ``figure_2/data``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Flat, pickle-free data contract with explicit axes and both aggregate
        and panel-convenience keys.
    """

    destination = (
        DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)
    ).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    analysis_id, identity_core = _analysis_id(ecog, model)
    npz_path = destination / NPZ_NAME
    provenance_path = destination / PROVENANCE_NAME
    if (
        not force
        and _cache_matches(npz_path, provenance_path, analysis_id)
    ):
        return _load_npz(npz_path)

    recordings = [
        _extract_ecog_recording(position) for position in (1, 2, 3)
    ]
    erp_arrays, erp_rows, erp_provenance = _erp_inference(recordings)
    decoder_arrays, decoder_rows, decoder_provenance = (
        _decoder_inference(recordings)
    )
    posterior_arrays, posterior_rows, posterior_provenance = (
        _posterior_inference()
    )
    ecog_buildup_arrays, ecog_buildup_provenance = (
        _ecog_buildup_inference(recordings)
    )
    (
        rnn_arrays,
        rnn_rows,
        point_rows,
        rnn_provenance,
    ) = _rnn_inference(model)

    conditions = np.asarray(model["conditions"])
    arrays: dict[str, np.ndarray] = {
        **erp_arrays,
        **decoder_arrays,
        **posterior_arrays,
        **ecog_buildup_arrays,
        **rnn_arrays,
        "analysis_id": np.asarray(analysis_id),
        "conditions": conditions.copy(),
        "positions": np.arange(1, 4, dtype=np.int64),
    }
    _atomic_npz(npz_path, arrays)
    csv_paths = _write_exports(
        destination,
        erp_rows,
        decoder_rows,
        posterior_rows,
        rnn_rows,
        point_rows,
    )
    provenance: dict[str, Any] = {
        **identity_core,
        "analysis_id": analysis_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "ECoG: within-recording randomization inference from three "
            "recordings in one animal. RNN: paired inference across eight "
            "simulated session/order seeds."
        ),
        "evoked_response": erp_provenance,
        "decoder": decoder_provenance,
        "posterior_maps": posterior_provenance,
        "ecog_buildup": ecog_buildup_provenance,
        "rnn": rnn_provenance,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "numba_threads": numba.get_num_threads(),
            "platform": platform.platform(),
        },
        "outputs": {
            "npz": str(npz_path),
            "csv": [str(path) for path in csv_paths],
        },
        "npz_sha256": _sha256(npz_path),
        "csv_sha256": {
            path.name: _sha256(path) for path in csv_paths
        },
    }
    _atomic_json(provenance_path, provenance)
    return arrays


__all__ = ["load_or_build_inference"]
