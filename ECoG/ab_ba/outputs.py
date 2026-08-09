"""Save arrays, figure tables, provenance, and a compact QC figure."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy

from .config import ComparisonSpec, EXPERIMENTS
from .data import ABBAEpochs
from .decoder import DecoderResult


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
    ]
)
OXBLOOD = "#7C102A"
VIOLET = "#685994"
BLUE = "#2166AC"
CHARCOAL = "#2D3748"
ASH = "#8A939F"
LIGHT_GRAY = "#EEF0F3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _tone_background(axis, experiment, y_min: float, y_max: float) -> None:
    for tone in range(2):
        start = tone * (experiment.note_duration_ms + experiment.note_gap_ms)
        stop = start + experiment.note_duration_ms
        axis.axvspan(start, stop, color=LIGHT_GRAY, zorder=0)
    axis.set_ylim(y_min, y_max)


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(ASH)
    axis.tick_params(colors=CHARCOAL, width=0.8)
    axis.grid(axis="y", color="#D9DEE5", lw=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def _plot_qc(path: Path, spec: ComparisonSpec, result: DecoderResult) -> None:
    experiment = EXPERIMENTS[spec.expnum]
    fig = plt.figure(figsize=(14.8, 8.4), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(2, 6, height_ratios=(1.0, 1.05))
    ax_accuracy = fig.add_subplot(grid[0, :3])
    ax_spatial = fig.add_subplot(grid[0, 3:])
    erp_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    ax_map = fig.add_subplot(grid[1, 3:])

    _tone_background(ax_accuracy, experiment, 0.3, 1.0)
    spatial_time = result.time_ms[result.spatial_window_indices]
    if result.mode == "matlab-faithful":
        ax_accuracy.axvspan(
            spatial_time[0], spatial_time[-1] + 1,
            color="#D8D0E8", alpha=0.75, zorder=0,
        )
    ax_accuracy.plot(result.time_ms, result.accuracy_smoothed, color=VIOLET, lw=2.2)
    ax_accuracy.axhline(0.5, color=ASH, ls=(0, (4, 3)), lw=1.1)
    ax_accuracy.set(
        xlim=(0, min(600, result.time_ms[-1])),
        xlabel="Time from sequence onset (ms)",
        ylabel="Cross-validated accuracy",
        title="Same sequence: deviant vs standard-after-deviant",
    )
    _style_axis(ax_accuracy)

    channels = np.arange(1, result.spatial_pattern.size + 1)
    bar_colors = np.array([ASH] * len(channels), dtype=object)
    bar_colors[result.top_channels_matlab - 1] = OXBLOOD
    ax_spatial.bar(channels, result.spatial_pattern, color=bar_colors, width=0.76)
    ax_spatial.set(
        xlim=(0, len(channels) + 1),
        xlabel="Electrode",
        ylabel="|activation pattern|",
        title=(
            "Training-fold spatial pattern"
            if result.mode == "leakage-safe"
            else "Legacy peak-selected spatial pattern"
        ),
    )
    _style_axis(ax_spatial)

    for rank, (axis, channel) in enumerate(
        zip(erp_axes, result.top_channels_matlab), start=1
    ):
        index = channel - 1
        values = np.r_[
            result.erp_deviant[index], result.erp_standard_after_deviant[index]
        ]
        limit = max(float(np.nanmax(np.abs(values))) * 1.16, 0.25)
        _tone_background(axis, experiment, -limit, limit)
        axis.plot(result.time_ms, result.erp_deviant[index], color=OXBLOOD,
                  lw=1.65, label="Deviant")
        axis.plot(result.time_ms, result.erp_standard_after_deviant[index],
                  color=BLUE, lw=1.65, label="Standard after opposite deviant")
        axis.axhline(0, color=ASH, lw=0.8)
        axis.set(
            xlim=(0, min(600, result.time_ms[-1])),
            xlabel="Time (ms)",
            ylabel="Baseline-normalized amplitude" if rank == 1 else "",
            title=f"Electrode {channel} · rank {rank}",
        )
        _style_axis(axis)
        if rank == 1:
            axis.legend(frameon=False, fontsize=8, loc="best")

    cmap = LinearSegmentedColormap.from_list(
        "violet_oxblood", ["#F2F0F7", "#B5A8CE", VIOLET, OXBLOOD]
    )
    maximum = max(float(np.max(result.spatial_pattern)), np.finfo(float).eps)
    ax_map.axhspan(0.35, 4.5, color="#F7F8FA", zorder=0)
    ax_map.axhspan(4.5, 8.65, color="#F0F2F5", zorder=0)
    for row in range(8):
        for col in range(4):
            channel = GRID_MAP[row, col]
            value = result.spatial_pattern[channel - 1]
            top = channel in result.top_channels_matlab
            ax_map.scatter(
                col + 1, row + 1, s=210 if top else 145,
                color=cmap(value / maximum), edgecolor=OXBLOOD if top else "white",
                linewidth=2.0 if top else 1.0, zorder=2,
            )
            ax_map.text(col + 1, row + 1, str(channel), ha="center", va="center",
                        fontsize=7.2, color="white" if value / maximum > 0.48 else CHARCOAL,
                        fontweight="bold" if top else "normal", zorder=3)
    ax_map.axhline(4.5, color=CHARCOAL, lw=1.1)
    ax_map.text(0.25, 2.5, "A1", rotation=90, ha="center", va="center",
                color=CHARCOAL, fontweight="bold")
    ax_map.text(0.25, 6.5, "PEG", rotation=90, ha="center", va="center",
                color=CHARCOAL, fontweight="bold")
    ax_map.set(xlim=(-0.05, 4.7), ylim=(8.8, 0.2), aspect="equal",
               xticks=[], yticks=[], title="A1–PEG electrode map")
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{spec.expected_target_sequence} · {result.mode}",
        color=CHARCOAL, fontsize=15, fontweight="bold",
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_analysis(
    output_root: Path,
    spec: ComparisonSpec,
    export_path: Path,
    epochs: ABBAEpochs,
    result: DecoderResult,
) -> Path:
    destination = Path(output_root) / spec.key / result.mode
    figure_data = destination / "figure_data"
    figure_data.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        destination / "analysis_arrays.npz",
        deviant=epochs.deviant,
        standard_after_deviant=epochs.standard_after_deviant,
        deviant_groups=epochs.deviant_groups,
        standard_groups=epochs.standard_groups,
        deviant_trials=epochs.deviant_trials,
        standard_trials=epochs.standard_trials,
        accuracy=result.accuracy,
        accuracy_smoothed=result.accuracy_smoothed,
        activation_patterns=result.activation_patterns,
        spatial_pattern=result.spatial_pattern,
        spatial_window_indices=result.spatial_window_indices,
        erp_deviant=result.erp_deviant,
        erp_standard_after_deviant=result.erp_standard_after_deviant,
        time_ms=result.time_ms,
        source_time_labels_ms=result.source_time_labels_ms,
        fold_ids=result.fold_ids,
    )
    _write_rows(
        figure_data / "decoding_timecourse.csv",
        ["time_ms", "source_time_label_ms", "accuracy", "accuracy_smoothed"],
        zip(result.time_ms, result.source_time_labels_ms,
            result.accuracy, result.accuracy_smoothed),
    )
    _write_rows(
        figure_data / "spatial_pattern.csv",
        ["channel_matlab", "activation_pattern_abs_mean", "top3_exploratory"],
        ((channel, result.spatial_pattern[channel - 1],
          int(channel in result.top_channels_matlab))
         for channel in range(1, result.spatial_pattern.size + 1)),
    )
    _write_rows(
        figure_data / "channel_erps.csv",
        ["channel_matlab", "time_ms", "deviant",
         "standard_after_opposite_deviant"],
        ((channel, result.time_ms[t], result.erp_deviant[channel - 1, t],
          result.erp_standard_after_deviant[channel - 1, t])
         for channel in range(1, result.erp_deviant.shape[0] + 1)
         for t in range(result.time_ms.size)),
    )
    _write_rows(
        figure_data / "observation_provenance.csv",
        ["class", "observation", "group", "recording_trial", "source_row_matlab"],
        (("deviant", i + 1, epochs.deviant_groups[i], epochs.deviant_trials[i],
          epochs.deviant_source_rows_matlab[i])
         for i in range(epochs.deviant.shape[1])),
    )
    # Append the second class without rewriting the header.
    with (figure_data / "observation_provenance.csv").open(
        "a", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerows(
            ("standard_after_opposite_deviant", i + 1,
             epochs.standard_groups[i], epochs.standard_trials[i],
             epochs.standard_source_rows_matlab[i])
            for i in range(epochs.standard_after_deviant.shape[1])
        )

    source_dir = export_path.parent
    source_candidates = [
        source_dir / "scripts_AB_BA.m",
        source_dir.parent / "Gen_M2Mat.m",
        Path(__file__).resolve().parent / "matlab" / "export_ab_ba_preprocessed.m",
    ]
    provenance: Dict[str, object] = {
        "comparison_spec": spec.to_dict(),
        "mode": result.mode,
        "export_file": str(export_path.resolve()),
        "export_sha256": _sha256(export_path),
        "source_file_sha256": {
            str(path.resolve()): _sha256(path) for path in source_candidates if path.exists()
        },
        "source_metadata": epochs.metadata,
        "fold_strategy": result.fold_strategy,
        "standardization_scope": result.standardization_scope,
        "activation_pattern_scope": result.activation_pattern_scope,
        "warnings": list(result.warnings),
        "descriptive_peak_time_ms": result.peak_time_ms,
        "descriptive_peak_accuracy_smoothed": result.peak_accuracy_smoothed,
        "spatial_window_ms": [
            int(result.time_ms[result.spatial_window_indices[0]]),
            int(result.time_ms[result.spatial_window_indices[-1]]),
        ],
        "top_channels_matlab_exploratory": result.top_channels_matlab.tolist(),
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        },
    }
    with (destination / "provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, default=str)
        stream.write("\n")
    _plot_qc(destination / "decoder_qc.png", spec, result)
    return destination
