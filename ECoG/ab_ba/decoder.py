"""Time-resolved ridge-logistic decoding for AB/BA ECoG comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np

from ECoG.roving.decoder import (
    fit_matlab_ridge_logistic,
    matlab_movmean,
    matlab_zscore,
)

from .config import ComparisonSpec, EXPERIMENTS
from .data import ABBAEpochs


Mode = Literal["matlab-faithful", "leakage-safe"]


@dataclass(frozen=True)
class DecoderResult:
    mode: Mode
    accuracy: np.ndarray
    accuracy_smoothed: np.ndarray
    activation_patterns: np.ndarray
    spatial_pattern: np.ndarray
    spatial_window_indices: np.ndarray
    peak_index: int
    peak_time_ms: int
    peak_accuracy_smoothed: float
    top_channels_matlab: np.ndarray
    erp_deviant: np.ndarray
    erp_standard_after_deviant: np.ndarray
    time_ms: np.ndarray
    source_time_labels_ms: np.ndarray
    fold_ids: np.ndarray
    fold_strategy: str
    standardization_scope: str
    activation_pattern_scope: str
    warnings: Tuple[str, ...]


def _stratified_folds(
    y: np.ndarray, n_folds: int, rng: np.random.RandomState
) -> List[np.ndarray]:
    parts: List[List[np.ndarray]] = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        shuffled = indices[rng.permutation(len(indices))]
        for fold, split in enumerate(np.array_split(shuffled, n_folds)):
            parts[fold].append(split)
    return [np.sort(np.concatenate(fold)) for fold in parts]


def stratified_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """Assign whole, class-homogeneous recording trials to balanced folds."""

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if y.shape != groups.shape:
        raise ValueError("y and groups must have identical shapes")
    assignments: List[List[object]] = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        label_groups = np.unique(groups[y == label])
        if label_groups.size < n_folds:
            raise ValueError(
                f"Class {label} has {label_groups.size} trial groups; "
                f"cannot form {n_folds} grouped folds"
            )
        sizes = np.array([np.sum(groups == group) for group in label_groups])
        # Random tie order followed by stable descending size makes the split
        # deterministic under the declared RNG while balancing observations.
        tie_order = rng.permutation(label_groups.size)
        ordered = tie_order[np.argsort(-sizes[tie_order], kind="stable")]
        totals = np.zeros(n_folds, dtype=int)
        for index in ordered:
            fold = int(np.argmin(totals))
            group = label_groups[index]
            assignments[fold].append(group)
            totals[fold] += int(sizes[index])
    folds = [np.flatnonzero(np.isin(groups, fold_groups)) for fold_groups in assignments]
    for fold, test in enumerate(folds):
        if test.size == 0 or np.unique(y[test]).size != np.unique(y).size:
            raise AssertionError(f"Grouped fold {fold} does not contain both classes")
    return folds


def _fold_ids(folds: Sequence[np.ndarray], n_samples: int) -> np.ndarray:
    ids = np.full(n_samples, -1, dtype=int)
    for fold, indices in enumerate(folds):
        if np.any(ids[indices] >= 0):
            raise AssertionError("Cross-validation test folds overlap")
        ids[indices] = fold
    if np.any(ids < 0):
        raise AssertionError("Cross-validation test folds do not cover all observations")
    return ids


def _safe_standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=1)
    bad = ~np.isfinite(scale) | (scale <= np.finfo(float).eps)
    scale = scale.copy()
    scale[bad] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _predict(beta: np.ndarray, bias: float, x: np.ndarray) -> np.ndarray:
    return (x @ beta + bias > 0).astype(int)


def _safe_cv_and_pattern(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[np.ndarray],
    ridge_lambda: float,
) -> Tuple[float, np.ndarray]:
    predictions = np.empty_like(y)
    all_indices = np.arange(len(y))
    patterns = []
    weights = []
    for test in folds:
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        x_train, x_test = _safe_standardize_train_test(x[train], x[test])
        beta, bias = fit_matlab_ridge_logistic(x_train, y[train], ridge_lambda)
        predictions[test] = _predict(beta, bias, x_test)
        patterns.append(np.cov(x_train, rowvar=False, ddof=1) @ beta)
        weights.append(len(train))
    pattern = np.average(np.asarray(patterns), axis=0, weights=np.asarray(weights))
    return float(np.mean(predictions == y)), pattern


def _legacy_cv(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[np.ndarray],
    ridge_lambda: float,
) -> float:
    predictions = np.empty_like(y)
    all_indices = np.arange(len(y))
    for test in folds:
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        beta, bias = fit_matlab_ridge_logistic(x[train], y[train], ridge_lambda)
        predictions[test] = _predict(beta, bias, x[test])
    return float(np.mean(predictions == y))


def run_decoder(
    epochs: ABBAEpochs,
    spec: ComparisonSpec,
    *,
    mode: Mode = "leakage-safe",
    random_seed: int | None = None,
) -> DecoderResult:
    """Run the source decoder or the preregistration-ready safe profile.

    The original script does not seed its random stream.  For an auditable
    rerun, ``matlab-faithful`` therefore defaults to the package's declared
    reproducibility seed; this reproduces the operations, not the unknown RNG
    state stored in the supplied ``.fig`` files.
    """

    if mode not in ("matlab-faithful", "leakage-safe"):
        raise ValueError(f"Unknown mode {mode!r}")
    if epochs.deviant.shape != epochs.standard_after_deviant.shape:
        raise ValueError("The two classes must already be balanced")
    if not np.all(np.isfinite(epochs.deviant)) or not np.all(
        np.isfinite(epochs.standard_after_deviant)
    ):
        raise ValueError("Epoch arrays contain NaN or infinite values")

    x_full = np.concatenate(
        [epochs.deviant, epochs.standard_after_deviant], axis=1
    )
    n_per_class = epochs.deviant.shape[1]
    y = np.r_[np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    groups = np.r_[epochs.deviant_groups, epochs.standard_groups]
    n_samples = len(y)
    n_time = x_full.shape[2]
    seed = spec.reproducibility_seed if random_seed is None else random_seed
    rng = np.random.RandomState(seed)
    accuracy = np.empty(n_time, dtype=float)
    patterns = np.empty((x_full.shape[0], n_time), dtype=float)

    if mode == "leakage-safe":
        folds = stratified_group_folds(y, groups, spec.n_folds, rng)
        fold_ids = _fold_ids(folds, n_samples)
        fold_strategy = (
            "stratified 5-fold CV grouped by acquisition-day trial; fixed over time"
        )
        standardization_scope = "training fold only; flat training features scaled by 1"
        pattern_scope = "mean of training-fold Haufe patterns; no held-out samples"
        warnings: Tuple[str, ...] = (
            "The two classes were recorded on different acquisition days; "
            "decoding can include day/session differences despite identical stimuli.",
        )
    else:
        folds = []
        fold_ids = np.full(n_samples, -1, dtype=int)
        fold_strategy = "source-equivalent stratified random 5-fold; repartitioned per time"
        standardization_scope = "all observations before cross-validation"
        pattern_scope = "full-data Haufe pattern around a data-selected peak"
        warnings = (
            "Held-out observations contribute to z-score means and variances.",
            "Events from the same acquisition trial can occur in train and test folds.",
            "The displayed spatial window is selected and summarized on the same data.",
            "The source script did not seed random noise or fold generation; the "
            f"audit rerun uses declared seed {seed} and cannot recover its unknown state.",
            "The two classes were recorded on different acquisition days.",
        )

    for time_index in range(n_time):
        x = x_full[:, :, time_index].T.copy()
        if mode == "matlab-faithful":
            x += rng.normal(scale=spec.noise_sd, size=x.shape)
            x_standardized, _, _ = matlab_zscore(x)
            time_folds = _stratified_folds(y, spec.n_folds, rng)
            if time_index == 0:
                fold_ids = _fold_ids(time_folds, n_samples)
            accuracy[time_index] = _legacy_cv(
                x_standardized, y, time_folds, spec.ridge_lambda
            )
            beta, _ = fit_matlab_ridge_logistic(
                x_standardized, y, spec.ridge_lambda
            )
            patterns[:, time_index] = (
                np.cov(x_standardized, rowvar=False, ddof=1) @ beta
            )
        else:
            accuracy[time_index], patterns[:, time_index] = _safe_cv_and_pattern(
                x, y, folds, spec.ridge_lambda
            )

    smoothed = matlab_movmean(accuracy, spec.smooth_samples)
    peak_index = int(np.argmax(smoothed))
    if mode == "matlab-faithful":
        lo = max(0, peak_index - spec.peak_half_window_samples)
        hi = min(n_time - 1, peak_index + spec.peak_half_window_samples)
        spatial_indices = np.arange(lo, hi + 1)
    else:
        duration = EXPERIMENTS[spec.expnum].sequence_duration_ms
        spatial_indices = np.flatnonzero(
            (epochs.time_ms >= 0) & (epochs.time_ms < duration)
        )
    spatial_pattern = np.mean(np.abs(patterns[:, spatial_indices]), axis=1)
    top_channels = np.argsort(-spatial_pattern, kind="stable")[:3] + 1
    return DecoderResult(
        mode=mode,
        accuracy=accuracy,
        accuracy_smoothed=smoothed,
        activation_patterns=patterns,
        spatial_pattern=spatial_pattern,
        spatial_window_indices=spatial_indices,
        peak_index=peak_index,
        peak_time_ms=int(epochs.time_ms[peak_index]),
        peak_accuracy_smoothed=float(smoothed[peak_index]),
        top_channels_matlab=top_channels,
        erp_deviant=np.mean(epochs.deviant, axis=1),
        erp_standard_after_deviant=np.mean(
            epochs.standard_after_deviant, axis=1
        ),
        time_ms=epochs.time_ms.copy(),
        source_time_labels_ms=epochs.source_time_labels_ms.copy(),
        fold_ids=fold_ids,
        fold_strategy=fold_strategy,
        standardization_scope=standardization_scope,
        activation_pattern_scope=pattern_scope,
        warnings=warnings,
    )
