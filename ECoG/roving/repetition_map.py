"""Ridge-logistic repetition maps translated from ``SVM_rep_map.m``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import expit
from scipy.stats import rankdata

from .config import AnalysisSpec
from .decoder import (
    _fold_id_vector,
    _stratified_folds,
    fit_matlab_ridge_logistic,
)
from .matlab_io import RovingRepetitionEpochs


Mode = Literal["matlab-faithful", "leakage-safe"]


@dataclass(frozen=True)
class RepetitionMapConfig:
    """Parameters copied from ``SVM_rep_map.m`` except for the learner."""

    ridge_lambda: float = 1e-2
    n_folds: int = 5
    random_seed: int = 42
    sigma_time: float = 3.0
    sigma_repetition: float = 0.8
    scale_epsilon: float = 1e-6


@dataclass(frozen=True)
class RepetitionMapResult:
    mode: Mode
    posterior_trials: np.ndarray  # blocks x repetitions x time
    posterior_mean: np.ndarray  # repetitions x time
    posterior_sem: np.ndarray
    posterior_smoothed: np.ndarray
    posterior_centered_smoothed: np.ndarray
    endpoint_accuracy: np.ndarray
    endpoint_auc: np.ndarray
    time_ms: np.ndarray
    deviant_aligned_time_ms: np.ndarray
    repetitions: np.ndarray
    fold_ids_by_block: np.ndarray
    anchor_fold_ids: np.ndarray
    balance_permutations: np.ndarray
    fold_strategy: str
    standardization_scope: str
    inference_scope: str
    warnings: Tuple[str, ...]


def _matlab_anchor_standardize(
    x: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match ``mean`` and ``std(...,0,1)+1e-6`` from the source."""

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=1) + epsilon
    return (x - mean) / scale, mean, scale


def _posterior(
    beta: np.ndarray, bias: float, x: np.ndarray
) -> np.ndarray:
    return expit(x @ beta + bias)


def _binary_metrics(
    positive: np.ndarray, negative: np.ndarray
) -> tuple[float, float]:
    """Accuracy at 0.5 and tie-aware ROC AUC."""

    probability = np.r_[positive, negative]
    labels = np.r_[
        np.ones(positive.size, dtype=int),
        np.zeros(negative.size, dtype=int),
    ]
    accuracy = float(np.mean((probability >= 0.5) == labels))
    ranks = rankdata(probability, method="average")
    n_positive = positive.size
    n_negative = negative.size
    auc = (
        np.sum(ranks[:n_positive])
        - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    return accuracy, float(auc)


def _matlab_gaussian_kernel(sigma: float) -> np.ndarray:
    """Kernel used by default ``imgaussfilt`` filter-size selection."""

    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    radius = int(np.ceil(2.0 * sigma))
    coordinate = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(coordinate**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def matlab_imgaussfilt(
    image: np.ndarray, sigma_repetition: float, sigma_time: float
) -> np.ndarray:
    """Separable ``imgaussfilt`` with default replicate padding."""

    smoothed = convolve1d(
        np.asarray(image, dtype=float),
        _matlab_gaussian_kernel(sigma_repetition),
        axis=0,
        mode="nearest",
    )
    return convolve1d(
        smoothed,
        _matlab_gaussian_kernel(sigma_time),
        axis=1,
        mode="nearest",
    )


def _context_stratified_block_folds(
    stimuli: np.ndarray,
    contexts: np.ndarray,
    n_folds: int,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """Keep blocks intact while distributing each transition over folds."""

    fold_parts: List[List[np.ndarray]] = [[] for _ in range(n_folds)]
    pairs = np.column_stack([contexts, stimuli])
    for pair in np.unique(pairs, axis=0):
        indices = np.flatnonzero(np.all(pairs == pair, axis=1))
        shuffled = indices[rng.permutation(indices.size)]
        for fold, part in enumerate(np.array_split(shuffled, n_folds)):
            fold_parts[fold].append(part)
    folds = [
        np.sort(np.concatenate(parts)) if parts else np.array([], dtype=int)
        for parts in fold_parts
    ]
    _fold_id_vector(folds, len(stimuli))
    return folds


def _run_faithful(
    epochs: RovingRepetitionEpochs,
    config: RepetitionMapConfig,
) -> RepetitionMapResult:
    """Source operation order with only SVM -> logistic changed."""

    data = epochs.epochs
    n_blocks, n_repetitions, _, n_time = data.shape
    rng = np.random.RandomState(config.random_seed)

    # Exact source order: one independent randperm call for every repetition.
    permutations = np.stack(
        [rng.permutation(n_blocks) for _ in range(n_repetitions)],
        axis=0,
    )
    posterior_trials = np.full(
        (n_blocks, n_repetitions, n_time), np.nan, dtype=float
    )
    anchor_fold_ids = np.empty((n_time, 2 * n_blocks), dtype=int)
    endpoint_accuracy = np.empty(n_time, dtype=float)
    endpoint_auc = np.empty(n_time, dtype=float)
    labels = np.r_[
        np.ones(n_blocks, dtype=int),
        np.zeros(n_blocks, dtype=int),
    ]
    all_anchor_indices = np.arange(2 * n_blocks)

    for time_index in range(n_time):
        rep_one = data[permutations[0], 0, :, time_index]
        rep_fifteen = data[permutations[-1], -1, :, time_index]
        anchor_raw = np.concatenate([rep_one, rep_fifteen], axis=0)
        anchor, mean, scale = _matlab_anchor_standardize(
            anchor_raw, config.scale_epsilon
        )

        beta, bias = fit_matlab_ridge_logistic(
            anchor, labels, config.ridge_lambda
        )
        folds = _stratified_folds(labels, config.n_folds, rng)
        fold_ids = _fold_id_vector(folds, 2 * n_blocks)
        anchor_fold_ids[time_index] = fold_ids
        anchor_oof = np.empty(2 * n_blocks, dtype=float)
        for test in folds:
            train = np.setdiff1d(
                all_anchor_indices, test, assume_unique=True
            )
            fold_beta, fold_bias = fit_matlab_ridge_logistic(
                anchor[train], labels[train], config.ridge_lambda
            )
            anchor_oof[test] = _posterior(
                fold_beta, fold_bias, anchor[test]
            )

        posterior_trials[permutations[0], 0, time_index] = (
            anchor_oof[:n_blocks]
        )
        posterior_trials[permutations[-1], -1, time_index] = (
            anchor_oof[n_blocks:]
        )
        for repetition_index in range(1, n_repetitions - 1):
            order = permutations[repetition_index]
            test = (data[order, repetition_index, :, time_index] - mean) / scale
            posterior_trials[order, repetition_index, time_index] = _posterior(
                beta, bias, test
            )

        endpoint_accuracy[time_index], endpoint_auc[time_index] = (
            _binary_metrics(anchor_oof[:n_blocks], anchor_oof[n_blocks:])
        )

    return _finish_result(
        epochs=epochs,
        mode="matlab-faithful",
        posterior_trials=posterior_trials,
        endpoint_accuracy=endpoint_accuracy,
        endpoint_auc=endpoint_auc,
        fold_ids_by_block=np.full(n_blocks, -1, dtype=int),
        anchor_fold_ids=anchor_fold_ids,
        balance_permutations=permutations,
        config=config,
        fold_strategy=(
            "source-style random stratified 5-fold anchor CV, repartitioned "
            "at every time point"
        ),
        standardization_scope=(
            "all Rep-1/Rep-15 anchor observations before cross-validation"
        ),
        inference_scope="descriptive within-recording posterior geometry",
        warnings=(
            "Source-style anchor standardization uses held-out observations.",
            "Source-style folds can place Rep 1 and Rep 15 from the same "
            "roving block on opposite sides of cross-validation.",
            "Intermediate repetitions are predicted by a model trained on "
            "endpoint observations from those same blocks.",
        ),
    )


def _run_safe(
    epochs: RovingRepetitionEpochs,
    config: RepetitionMapConfig,
) -> RepetitionMapResult:
    """Out-of-fold posterior for every block, repetition, and time point."""

    data = epochs.epochs
    n_blocks, n_repetitions, _, n_time = data.shape
    rng = np.random.RandomState(config.random_seed)
    folds = _context_stratified_block_folds(
        epochs.stimuli, epochs.contexts, config.n_folds, rng
    )
    fold_ids_by_block = _fold_id_vector(folds, n_blocks)
    posterior_trials = np.full(
        (n_blocks, n_repetitions, n_time), np.nan, dtype=float
    )
    anchor_fold_ids = np.tile(
        np.r_[fold_ids_by_block, fold_ids_by_block], (n_time, 1)
    )
    endpoint_accuracy = np.empty(n_time, dtype=float)
    endpoint_auc = np.empty(n_time, dtype=float)
    all_blocks = np.arange(n_blocks)

    for time_index in range(n_time):
        for test_blocks in folds:
            train_blocks = np.setdiff1d(
                all_blocks, test_blocks, assume_unique=True
            )
            train_raw = np.concatenate(
                [
                    data[train_blocks, 0, :, time_index],
                    data[train_blocks, -1, :, time_index],
                ],
                axis=0,
            )
            train_labels = np.r_[
                np.ones(train_blocks.size, dtype=int),
                np.zeros(train_blocks.size, dtype=int),
            ]
            train, mean, scale = _matlab_anchor_standardize(
                train_raw, config.scale_epsilon
            )
            beta, bias = fit_matlab_ridge_logistic(
                train, train_labels, config.ridge_lambda
            )
            test_raw = data[test_blocks, :, :, time_index]
            test = (test_raw.reshape(-1, test_raw.shape[-1]) - mean) / scale
            probability = _posterior(beta, bias, test).reshape(
                test_blocks.size, n_repetitions
            )
            posterior_trials[test_blocks, :, time_index] = probability

        endpoint_accuracy[time_index], endpoint_auc[time_index] = (
            _binary_metrics(
                posterior_trials[:, 0, time_index],
                posterior_trials[:, -1, time_index],
            )
        )

    if np.any(~np.isfinite(posterior_trials)):
        raise AssertionError("Leakage-safe predictions do not cover every block")
    return _finish_result(
        epochs=epochs,
        mode="leakage-safe",
        posterior_trials=posterior_trials,
        endpoint_accuracy=endpoint_accuracy,
        endpoint_auc=endpoint_auc,
        fold_ids_by_block=fold_ids_by_block,
        anchor_fold_ids=anchor_fold_ids,
        balance_permutations=np.tile(
            np.arange(n_blocks, dtype=int), (n_repetitions, 1)
        ),
        config=config,
        fold_strategy=(
            "fixed 5-fold cross-validation grouped by roving block and "
            "stratified over the six context/current transitions"
        ),
        standardization_scope="training blocks only, separately in every fold",
        inference_scope=(
            "descriptive out-of-fold posterior for one recording from one animal"
        ),
        warnings=(
            "The 125 roving blocks are repeated observations from one animal, "
            "not independent biological replicates.",
            "No inferential significance is attached to the smoothed map.",
        ),
    )


def _finish_result(
    *,
    epochs: RovingRepetitionEpochs,
    mode: Mode,
    posterior_trials: np.ndarray,
    endpoint_accuracy: np.ndarray,
    endpoint_auc: np.ndarray,
    fold_ids_by_block: np.ndarray,
    anchor_fold_ids: np.ndarray,
    balance_permutations: np.ndarray,
    config: RepetitionMapConfig,
    fold_strategy: str,
    standardization_scope: str,
    inference_scope: str,
    warnings: Tuple[str, ...],
) -> RepetitionMapResult:
    if np.any((posterior_trials < 0.0) | (posterior_trials > 1.0)):
        raise AssertionError("Logistic posterior is outside [0, 1]")
    posterior_mean = np.mean(posterior_trials, axis=0)
    posterior_sem = np.std(
        posterior_trials, axis=0, ddof=1
    ) / np.sqrt(posterior_trials.shape[0])
    posterior_smoothed = matlab_imgaussfilt(
        posterior_mean,
        config.sigma_repetition,
        config.sigma_time,
    )
    return RepetitionMapResult(
        mode=mode,
        posterior_trials=posterior_trials,
        posterior_mean=posterior_mean,
        posterior_sem=posterior_sem,
        posterior_smoothed=posterior_smoothed,
        posterior_centered_smoothed=posterior_smoothed - 0.5,
        endpoint_accuracy=endpoint_accuracy,
        endpoint_auc=endpoint_auc,
        time_ms=epochs.time_ms.copy(),
        deviant_aligned_time_ms=(
            epochs.time_ms - int(epochs.metadata["deviant_onset_ms"])
        ),
        repetitions=epochs.repetitions.copy(),
        fold_ids_by_block=fold_ids_by_block,
        anchor_fold_ids=anchor_fold_ids,
        balance_permutations=balance_permutations,
        fold_strategy=fold_strategy,
        standardization_scope=standardization_scope,
        inference_scope=inference_scope,
        warnings=warnings,
    )


def run_repetition_map(
    epochs: RovingRepetitionEpochs,
    spec: AnalysisSpec,
    *,
    mode: Mode = "leakage-safe",
    config: RepetitionMapConfig | None = None,
) -> RepetitionMapResult:
    """Fit the requested Rep-1-like logistic posterior map."""

    config = RepetitionMapConfig() if config is None else config
    if epochs.epochs.shape[1] != 15:
        raise ValueError("SVM_rep_map translation requires repetitions 1 through 15")
    metadata_deviant = epochs.metadata.get("deviant_onset_ms")
    if metadata_deviant is None:
        raise ValueError("Repetition epochs are missing deviant-onset metadata")
    if int(metadata_deviant) != spec.deviant_onset_ms:
        raise ValueError("Deviant onset metadata does not match the analysis spec")

    if mode == "matlab-faithful":
        return _run_faithful(epochs, config)
    if mode == "leakage-safe":
        return _run_safe(epochs, config)
    raise ValueError(f"Unknown repetition-map mode {mode!r}")
