"""Save repetition posterior maps, trial-level values, and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import scipy
from scipy.io import savemat

from .config import AnalysisSpec
from .matlab_io import RovingRepetitionEpochs
from .repetition_map import RepetitionMapConfig, RepetitionMapResult


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_map_csv(path: Path, result: RepetitionMapResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "repetition",
                "time_ms",
                "deviant_aligned_time_ms",
                "posterior_rep1_like_mean",
                "posterior_rep1_like_sem",
                "posterior_rep1_like_smoothed",
                "posterior_centered_smoothed",
            ]
        )
        for repetition_index, repetition in enumerate(result.repetitions):
            for time_index, time_ms in enumerate(result.time_ms):
                writer.writerow(
                    [
                        repetition,
                        time_ms,
                        result.deviant_aligned_time_ms[time_index],
                        result.posterior_mean[repetition_index, time_index],
                        result.posterior_sem[repetition_index, time_index],
                        result.posterior_smoothed[repetition_index, time_index],
                        result.posterior_centered_smoothed[
                            repetition_index, time_index
                        ],
                    ]
                )


def _write_endpoint_csv(path: Path, result: RepetitionMapResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_ms",
                "deviant_aligned_time_ms",
                "endpoint_oof_accuracy",
                "endpoint_oof_auc",
            ]
        )
        writer.writerows(
            zip(
                result.time_ms,
                result.deviant_aligned_time_ms,
                result.endpoint_accuracy,
                result.endpoint_auc,
            )
        )


def _write_block_csv(
    path: Path,
    epochs: RovingRepetitionEpochs,
    result: RepetitionMapResult,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "block_row",
                "source_group_id",
                "source_trial",
                "previous_stimulus",
                "current_stimulus",
                "cross_validation_fold",
            ]
        )
        for index in range(epochs.epochs.shape[0]):
            writer.writerow(
                [
                    index,
                    epochs.group_ids[index],
                    epochs.trials[index],
                    epochs.contexts[index],
                    epochs.stimuli[index],
                    result.fold_ids_by_block[index],
                ]
            )


def _plot_map(
    path: Path,
    spec: AnalysisSpec,
    result: RepetitionMapResult,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3.2, 1]},
        constrained_layout=True,
    )
    maximum_deviation = max(
        float(np.max(np.abs(result.posterior_smoothed - 0.5))),
        np.finfo(float).eps,
    )
    norm = TwoSlopeNorm(
        vmin=0.5 - maximum_deviation,
        vcenter=0.5,
        vmax=0.5 + maximum_deviation,
    )
    image = axes[0].imshow(
        result.posterior_smoothed,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            result.time_ms[0] - 0.5,
            result.time_ms[-1] + 0.5,
            15.5,
            0.5,
        ],
        cmap="RdBu_r",
        norm=norm,
    )
    for boundary in (0, 180, 360, 540):
        axes[0].axvline(
            boundary,
            color="black",
            ls="--" if boundary in (0, 540) else "-",
            lw=1.1,
        )
    deviant_start = spec.deviant_onset_ms
    rectangle = plt.Rectangle(
        (deviant_start, 0.5),
        spec.tone_duration_ms,
        15,
        fill=False,
        edgecolor="black",
        linewidth=1.8,
        linestyle="--",
    )
    axes[0].add_patch(rectangle)
    axes[0].set(
        xlim=(0, 800),
        ylim=(15.5, 0.5),
        yticks=result.repetitions,
        xlabel="Time relative to sequence onset (ms)",
        ylabel="Repetition number",
        title=(
            f"{spec.key}: ridge-logistic Rep-1-like posterior "
            f"({result.mode})"
        ),
    )
    colorbar = figure.colorbar(image, ax=axes[0], pad=0.015)
    colorbar.set_label("P(Rep 1-like | ECoG)")

    axes[1].plot(
        result.time_ms,
        result.endpoint_auc,
        color="#7b3294",
        lw=1.4,
        label="OOF ROC AUC",
    )
    axes[1].plot(
        result.time_ms,
        result.endpoint_accuracy,
        color="#008837",
        lw=1.2,
        alpha=0.85,
        label="OOF accuracy",
    )
    axes[1].axhline(0.5, color="black", ls=":", lw=1)
    for boundary in (0, 180, 360, 540):
        axes[1].axvline(boundary, color="0.65", lw=0.7)
    axes[1].set(
        xlim=(0, 800),
        ylim=(0.25, 0.85),
        xlabel="Time relative to sequence onset (ms)",
        ylabel="Endpoint performance",
    )
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _matlab_struct(
    spec: AnalysisSpec,
    epochs: RovingRepetitionEpochs,
    result: RepetitionMapResult,
    config: RepetitionMapConfig,
) -> dict:
    return {
        "meta": {
            "analysis": spec.key,
            "source_script": "SVM_rep_map.m",
            "source_file": spec.data_file,
            "mode": result.mode,
            "learner_update": (
                "L2 ridge logistic regression replaces the source L2 ridge SVM"
            ),
            "posterior_definition": "P(Rep 1-like | ECoG)",
            "anchor_positive": 1,
            "anchor_negative": 15,
            "lambda": config.ridge_lambda,
            "nFolds": config.n_folds,
            "rngSeed": config.random_seed,
            "sigma_time": config.sigma_time,
            "sigma_rep": config.sigma_repetition,
            "fold_strategy": result.fold_strategy,
            "standardization_scope": result.standardization_scope,
            "inference_scope": result.inference_scope,
        },
        "time_vec": result.time_ms,
        "deviant_aligned_time_ms": result.deviant_aligned_time_ms,
        "repetitions": result.repetitions,
        "posterior_trials": result.posterior_trials,
        "posterior_map": result.posterior_mean,
        "posterior_sem": result.posterior_sem,
        "posterior_map_smoothed": result.posterior_smoothed,
        "posterior_centered_smoothed": result.posterior_centered_smoothed,
        "endpoint_accuracy": result.endpoint_accuracy,
        "endpoint_auc": result.endpoint_auc,
        "block_group_id": epochs.group_ids,
        "block_trial": epochs.trials,
        "block_current_stimulus": epochs.stimuli,
        "block_previous_stimulus": epochs.contexts,
        "fold_ids_by_block": result.fold_ids_by_block,
        "anchor_fold_ids": result.anchor_fold_ids,
        # MATLAB indices for source randperm outputs are 1-based.
        "balance_permutations": result.balance_permutations + 1,
    }


def save_repetition_map(
    output_root: Path,
    source_dir: Path,
    data_path: Path,
    spec: AnalysisSpec,
    epochs: RovingRepetitionEpochs,
    result: RepetitionMapResult,
    config: RepetitionMapConfig,
) -> Path:
    destination = (
        Path(output_root) / spec.key / "regression_rep_map" / result.mode
    )
    figure_data = destination / "figure_data"
    figure_data.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        destination / "regression_rep_map_arrays.npz",
        posterior_trials=result.posterior_trials,
        posterior_mean=result.posterior_mean,
        posterior_sem=result.posterior_sem,
        posterior_smoothed=result.posterior_smoothed,
        posterior_centered_smoothed=result.posterior_centered_smoothed,
        endpoint_accuracy=result.endpoint_accuracy,
        endpoint_auc=result.endpoint_auc,
        time_ms=result.time_ms,
        deviant_aligned_time_ms=result.deviant_aligned_time_ms,
        repetitions=result.repetitions,
        group_ids=epochs.group_ids,
        trials=epochs.trials,
        stimuli=epochs.stimuli,
        contexts=epochs.contexts,
        fold_ids_by_block=result.fold_ids_by_block,
        anchor_fold_ids=result.anchor_fold_ids,
        balance_permutations=result.balance_permutations,
    )
    savemat(
        destination / "RegressionRepMap.mat",
        {
            "RegressionRepMap": _matlab_struct(
                spec, epochs, result, config
            )
        },
        do_compression=True,
    )
    _write_map_csv(figure_data / "posterior_map.csv", result)
    _write_endpoint_csv(figure_data / "endpoint_performance.csv", result)
    _write_block_csv(figure_data / "block_index.csv", epochs, result)
    _plot_map(destination / "regression_rep_map.png", spec, result)

    source_paths = [
        Path(source_dir) / "SVM_rep_map.m",
        Path(source_dir) / "Gen_M2Mat.m",
        Path(source_dir) / "generate_full_mat_info_v2.m",
    ]
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("repetition_map.py"),
        Path(__file__).resolve().with_name("repetition_run.py"),
        Path(__file__).resolve().with_name("matlab_io.py"),
        Path(__file__).resolve().with_name("decoder.py"),
        Path(__file__).resolve().with_name("config.py"),
    ]
    provenance: Dict[str, object] = {
        "analysis_spec": spec.to_dict(),
        "map_config": asdict(config),
        "mode": result.mode,
        "method": {
            "source": "SVM_rep_map.m",
            "preserved": (
                "Gen_M2Mat cutting=1, samples 101:901, all six valid "
                "transitions, repetitions 1:15, rng seed 42, anchor "
                "standardization formula, ridge lambda, five folds, and "
                "Gaussian map smoothing"
            ),
            "requested_update": (
                "fitclinear Learner='svm' replaced by "
                "fitclinear Learner='logistic' equivalent"
            ),
            "posterior": "class-1 logistic probability: P(Rep 1-like | ECoG)",
        },
        "data_sha256": _sha256(data_path),
        "source_file_sha256": {
            str(path.resolve()): _sha256(path)
            for path in source_paths
            if path.exists()
        },
        "python_file_sha256": {
            str(path): _sha256(path)
            for path in code_paths
            if path.exists()
        },
        "source_metadata": epochs.metadata,
        "fold_strategy": result.fold_strategy,
        "standardization_scope": result.standardization_scope,
        "inference_scope": result.inference_scope,
        "warnings": list(result.warnings),
        "posterior_range": [
            float(np.min(result.posterior_trials)),
            float(np.max(result.posterior_trials)),
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (destination / "provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    return destination
