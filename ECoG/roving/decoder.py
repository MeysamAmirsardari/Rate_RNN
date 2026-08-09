"""Time-resolved MATLAB-equivalent and leakage-safe ridge-logistic decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .config import AnalysisSpec
from .matlab_io import RovingEpochs


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
    erp_first: np.ndarray
    erp_late: np.ndarray
    time_ms: np.ndarray
    deviant_aligned_time_ms: np.ndarray
    fold_ids: np.ndarray
    fold_strategy: str
    standardization_scope: str
    warnings: Tuple[str, ...]


def matlab_zscore(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column z-score with MATLAB's default sample standard deviation."""

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=1)
    return (x - mean) / scale, mean, scale


def matlab_movmean(values: np.ndarray, window: int) -> np.ndarray:
    """MATLAB ``movmean(values, window)`` with default shrink endpoints."""

    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be positive")
    left = window // 2
    right = (window - 1) // 2
    cumulative = np.r_[0.0, np.cumsum(values)]
    out = np.empty_like(values)
    for index in range(len(values)):
        lo = max(0, index - left)
        hi = min(len(values), index + right + 1)
        out[index] = (cumulative[hi] - cumulative[lo]) / (hi - lo)
    return out


def _objective_and_gradient(
    theta: np.ndarray, x: np.ndarray, y_signed: np.ndarray, ridge_lambda: float
) -> Tuple[float, np.ndarray]:
    beta = theta[:-1]
    bias = theta[-1]
    signed_score = y_signed * (x @ beta + bias)
    objective = (
        np.mean(np.logaddexp(0.0, -signed_score))
        + 0.5 * ridge_lambda * np.dot(beta, beta)
    )
    factor = -y_signed * expit(-signed_score) / len(y_signed)
    gradient = np.r_[x.T @ factor + ridge_lambda * beta, np.sum(factor)]
    return float(objective), gradient


def fit_matlab_ridge_logistic(
    x: np.ndarray, y: np.ndarray, ridge_lambda: float
) -> Tuple[np.ndarray, float]:
    """Minimize fitclinear's documented logistic-ridge objective.

    MATLAB uses average deviance plus ``lambda/2 * sum(beta**2)`` and does
    not penalize the bias.  BFGS is the default ridge solver for fewer than
    101 predictors in the source scripts' dimensional regime.
    """

    y_signed = np.where(np.asarray(y) == 1, 1.0, -1.0)
    initial = np.zeros(x.shape[1] + 1, dtype=float)

    class _MatlabBetaTolerance:
        """Reproduce fitclinear's additional default BFGS stopping rule."""

        def __init__(self) -> None:
            self.previous: np.ndarray | None = None
            self.stopped = False

        def __call__(self, current: np.ndarray) -> None:
            if self.previous is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    relative = (current - self.previous) / current
                if np.linalg.norm(relative) < 1e-4:
                    self.stopped = True
                    raise StopIteration
            self.previous = current.copy()

    beta_tolerance = _MatlabBetaTolerance()
    fitted = minimize(
        _objective_and_gradient,
        initial,
        args=(np.asarray(x, dtype=float), y_signed, ridge_lambda),
        method="BFGS",
        jac=True,
        callback=beta_tolerance,
        options={"gtol": 1e-6, "maxiter": 1000},
    )
    if (
        not fitted.success
        and not beta_tolerance.stopped
        and np.linalg.norm(fitted.jac, ord=np.inf) > 1e-5
    ):
        raise RuntimeError(
            f"Ridge-logistic optimization failed: {fitted.message}; "
            f"|gradient|_inf={np.linalg.norm(fitted.jac, ord=np.inf):.3g}"
        )
    return fitted.x[:-1], float(fitted.x[-1])


def _stratified_folds(
    y: np.ndarray, n_folds: int, rng: np.random.RandomState
) -> List[np.ndarray]:
    fold_parts: List[List[np.ndarray]] = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        shuffled = indices[rng.permutation(len(indices))]
        for fold, part in enumerate(np.array_split(shuffled, n_folds)):
            fold_parts[fold].append(part)
    return [np.sort(np.concatenate(parts)) for parts in fold_parts]


def _group_folds(
    groups: np.ndarray, n_folds: int, rng: np.random.RandomState
) -> List[np.ndarray]:
    unique = np.unique(groups)
    unique = unique[rng.permutation(len(unique))]
    group_parts = np.array_split(unique, n_folds)
    return [np.flatnonzero(np.isin(groups, part)) for part in group_parts]


def _fold_id_vector(folds: Sequence[np.ndarray], n_samples: int) -> np.ndarray:
    ids = np.full(n_samples, -1, dtype=int)
    for fold, indices in enumerate(folds):
        if np.any(ids[indices] != -1):
            raise AssertionError("Fold test sets overlap")
        ids[indices] = fold
    if np.any(ids == -1):
        raise AssertionError("Fold test sets do not cover all observations")
    return ids


def _predict(beta: np.ndarray, bias: float, x: np.ndarray) -> np.ndarray:
    return (x @ beta + bias > 0.0).astype(int)


def _cv_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[np.ndarray],
    ridge_lambda: float,
    *,
    fold_local_standardization: bool,
) -> float:
    predictions = np.empty_like(y)
    all_indices = np.arange(len(y))
    for test in folds:
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        if fold_local_standardization:
            x_train, mean, scale = matlab_zscore(x[train])
            x_test = (x[test] - mean) / scale
        else:
            x_train = x[train]
            x_test = x[test]
        beta, bias = fit_matlab_ridge_logistic(
            x_train, y[train], ridge_lambda
        )
        predictions[test] = _predict(beta, bias, x_test)
    return float(np.mean(predictions == y))


def run_decoder(
    epochs: RovingEpochs,
    spec: AnalysisSpec,
    *,
    mode: Mode = "leakage-safe",
    safe_spatial_window_deviant_ms: Tuple[int, int] = (0, 180),
) -> DecoderResult:
    """Run the requested source decoder or its leakage-safe counterpart.

    ``matlab-faithful`` reproduces the source order, including z-scoring all
    observations before CV and selecting the spatial window around the peak
    found on the same data.

    ``leakage-safe`` keeps paired repetitions from one roving block in the
    same fold, fits z-scoring on each training fold, reuses folds over time,
    and uses a prespecified deviant-aligned spatial window.
    """

    if mode not in ("matlab-faithful", "leakage-safe"):
        raise ValueError(f"Unknown mode {mode!r}")
    x_full = np.concatenate([epochs.rep_first, epochs.rep_late], axis=1)
    y = np.r_[
        np.zeros(epochs.rep_first.shape[1], dtype=int),
        np.ones(epochs.rep_late.shape[1], dtype=int),
    ]
    groups = np.r_[epochs.first_groups, epochs.late_groups]
    n_samples = len(y)
    n_time = x_full.shape[2]
    rng = np.random.RandomState(spec.random_seed)
    accuracy = np.empty(n_time, dtype=float)
    patterns = np.empty((spec.n_channels, n_time), dtype=float)

    if mode == "leakage-safe":
        fixed_folds = _group_folds(groups, spec.n_folds, rng)
        fixed_fold_ids = _fold_id_vector(fixed_folds, n_samples)
        fold_strategy = "5-fold grouped by roving block; fixed over time"
        standardization_scope = "training fold only"
        warnings: Tuple[str, ...] = ()
    else:
        # In MATLAB the same RNG stream is seeded immediately before the two
        # randperm balancing calls, then continues into per-time randn and
        # KFold partitioning.  Extraction already applied the permutations;
        # consume analogous calls here to retain that operation order.
        rng.permutation(
            int(epochs.metadata["n_first_before_global_balance"])
        )
        rng.permutation(
            int(epochs.metadata["n_late_before_global_balance"])
        )
        fixed_folds = []
        fixed_fold_ids = np.full(n_samples, -1, dtype=int)
        fold_strategy = "source-equivalent stratified random 5-fold; repartitioned per time"
        standardization_scope = "all observations before cross-validation"
        warnings = (
            "Source-compatible preprocessing leaks held-out feature means and "
            "standard deviations into each fold.",
            "Source-compatible folds can separate repetitions from the same "
            "roving block.",
            "The peak-centered spatial window is selected and summarized on "
            "the same data.",
        )

    for time_index in range(n_time):
        x = x_full[:, :, time_index].T.copy()
        x += rng.normal(scale=spec.noise_sd, size=x.shape)
        if mode == "matlab-faithful":
            x_standardized, _, _ = matlab_zscore(x)
            folds = _stratified_folds(y, spec.n_folds, rng)
            accuracy[time_index] = _cv_accuracy(
                x_standardized,
                y,
                folds,
                spec.lambda_ridge,
                fold_local_standardization=False,
            )
            x_for_pattern = x_standardized
            if time_index == 0:
                fixed_fold_ids = _fold_id_vector(folds, n_samples)
        else:
            accuracy[time_index] = _cv_accuracy(
                x,
                y,
                fixed_folds,
                spec.lambda_ridge,
                fold_local_standardization=True,
            )
            x_for_pattern, _, _ = matlab_zscore(x)

        beta, _ = fit_matlab_ridge_logistic(
            x_for_pattern, y, spec.lambda_ridge
        )
        patterns[:, time_index] = np.cov(
            x_for_pattern, rowvar=False, ddof=1
        ) @ beta

    smoothed = matlab_movmean(accuracy, spec.smooth_samples)
    peak_index = int(np.argmax(smoothed))
    if mode == "matlab-faithful":
        lo = max(0, peak_index - spec.peak_half_window_samples)
        hi = min(n_time - 1, peak_index + spec.peak_half_window_samples)
        spatial_indices = np.arange(lo, hi + 1)
    else:
        deviant_time = epochs.time_ms - spec.deviant_onset_ms
        lo_ms, hi_ms = safe_spatial_window_deviant_ms
        spatial_indices = np.flatnonzero(
            (deviant_time >= lo_ms) & (deviant_time < hi_ms)
        )
        if spatial_indices.size == 0:
            raise ValueError(
                f"Prespecified spatial window {safe_spatial_window_deviant_ms} "
                "does not overlap the epoch"
            )
    spatial_pattern = np.mean(np.abs(patterns[:, spatial_indices]), axis=1)
    # MATLAB channel numbers are 1-based.
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
        erp_first=np.mean(epochs.rep_first, axis=1),
        erp_late=np.mean(epochs.rep_late, axis=1),
        time_ms=epochs.time_ms.copy(),
        deviant_aligned_time_ms=epochs.time_ms - spec.deviant_onset_ms,
        fold_ids=fixed_fold_ids,
        fold_strategy=fold_strategy,
        standardization_scope=standardization_scope,
        warnings=warnings,
    )
