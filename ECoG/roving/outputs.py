"""Save analysis arrays, manuscript figure data, provenance, and a QC figure."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from .config import AnalysisSpec
from .decoder import DecoderResult
from .matlab_io import RovingEpochs


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


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _plot_qc(
    path: Path, spec: AnalysisSpec, result: DecoderResult
) -> None:
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax_accuracy = fig.add_subplot(grid[0, :2])
    ax_spatial = fig.add_subplot(grid[0, 2:])
    axes_erp = [fig.add_subplot(grid[1, index]) for index in range(3)]
    ax_map = fig.add_subplot(grid[1, 3])

    tone_colors = ["#f2f2f2"] * 3
    tone_colors[spec.deviant_position - 1] = "#d8f0df"
    for tone in range(3):
        start = tone * spec.tone_duration_ms
        stop = start + spec.tone_duration_ms
        ax_accuracy.axvspan(start, stop, color=tone_colors[tone], zorder=0)
    spatial_time = result.time_ms[result.spatial_window_indices]
    ax_accuracy.axvspan(
        spatial_time[0], spatial_time[-1], color="#ffd966", alpha=0.45
    )
    ax_accuracy.plot(result.time_ms, result.accuracy_smoothed, lw=2, color="#2166ac")
    ax_accuracy.axhline(0.5, color="black", ls="--", lw=1)
    ax_accuracy.set(xlim=(0, 800), ylim=(0.3, 1.0), xlabel="Time (ms)",
                    ylabel="Decoding accuracy", title="Rep 1 vs source late repetition")

    channels = np.arange(1, 33)
    colors = np.full((32, 3), 0.55)
    colors[result.top_channels_matlab - 1] = (0.85, 0.1, 0.1)
    ax_spatial.bar(channels, result.spatial_pattern, color=colors, width=0.75)
    ax_spatial.set(
        xlim=(0, 33),
        xlabel="Channel number",
        ylabel="|activation pattern|",
        title=f"Spatial pattern ({result.mode})",
    )

    for rank, (axis, channel) in enumerate(
        zip(axes_erp, result.top_channels_matlab), start=1
    ):
        index = channel - 1
        axis.plot(result.time_ms, result.erp_late[index], color="#2166ac",
                  lw=1.6, label=f"Rep {spec.rep_late}")
        axis.plot(result.time_ms, result.erp_first[index], color="#d73027",
                  lw=1.6, label=f"Rep {spec.rep_first}")
        axis.axhline(0, color="black", lw=0.7)
        axis.set(xlim=(0, 800), xlabel="Time (ms)", ylabel="Amplitude",
                 title=f"Ch {channel} (rank {rank}; exploratory)")
        if rank == 3:
            axis.legend(frameon=False)

    maximum = max(float(np.max(result.spatial_pattern)), np.finfo(float).eps)
    for row in range(8):
        for col in range(4):
            channel = GRID_MAP[row, col]
            value = result.spatial_pattern[channel - 1]
            ax_map.scatter(
                col + 1,
                row + 1,
                s=90 + 260 * value / maximum,
                c=[plt.cm.Reds(value / maximum)],
                edgecolor="black" if channel in result.top_channels_matlab else "0.6",
                linewidth=1.2,
            )
            ax_map.text(col + 1, row + 1, str(channel), ha="center", va="center",
                        fontsize=7)
    ax_map.axhline(4.5, color="black", ls="--", lw=1)
    ax_map.text(0.2, 2.5, "A1", rotation=90, va="center", color="0.35")
    ax_map.text(0.2, 6.5, "PEG", rotation=90, va="center", color="0.35")
    ax_map.set(xlim=(-0.1, 5), ylim=(8.8, 0.2), aspect="equal",
               title="A1 / PEG", xticks=[], yticks=[])

    fig.suptitle(
        f"{spec.key}: {result.mode}; source numeric repetitions "
        f"{spec.rep_first} vs {spec.rep_late}",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_analysis(
    output_root: Path,
    spec: AnalysisSpec,
    data_path: Path,
    epochs: RovingEpochs,
    result: DecoderResult,
) -> Path:
    """Write one self-contained, versioned-by-mode analysis directory."""

    destination = Path(output_root) / spec.key / result.mode
    figure_data = destination / "figure_data"
    figure_data.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        destination / "analysis_arrays.npz",
        rep_first=epochs.rep_first,
        rep_late=epochs.rep_late,
        first_groups=epochs.first_groups,
        late_groups=epochs.late_groups,
        accuracy=result.accuracy,
        accuracy_smoothed=result.accuracy_smoothed,
        activation_patterns=result.activation_patterns,
        spatial_pattern=result.spatial_pattern,
        spatial_window_indices=result.spatial_window_indices,
        erp_first=result.erp_first,
        erp_late=result.erp_late,
        time_ms=result.time_ms,
        deviant_aligned_time_ms=result.deviant_aligned_time_ms,
        fold_ids=result.fold_ids,
    )

    _write_rows(
        figure_data / "decoding_timecourse.csv",
        ["time_ms", "deviant_aligned_time_ms", "accuracy", "accuracy_smoothed"],
        zip(
            result.time_ms,
            result.deviant_aligned_time_ms,
            result.accuracy,
            result.accuracy_smoothed,
        ),
    )
    _write_rows(
        figure_data / "spatial_pattern.csv",
        ["channel_matlab", "activation_pattern_abs_mean", "is_top3_exploratory"],
        (
            (channel, result.spatial_pattern[channel - 1],
             int(channel in result.top_channels_matlab))
            for channel in range(1, 33)
        ),
    )
    _write_rows(
        figure_data / "channel_erps.csv",
        [
            "channel_matlab",
            "time_ms",
            "deviant_aligned_time_ms",
            f"erp_rep_{spec.rep_first}",
            f"erp_rep_{spec.rep_late}",
        ],
        (
            (
                channel,
                result.time_ms[time],
                result.deviant_aligned_time_ms[time],
                result.erp_first[channel - 1, time],
                result.erp_late[channel - 1, time],
            )
            for channel in range(1, 33)
            for time in range(len(result.time_ms))
        ),
    )

    source_candidates = [
        data_path.parent / spec.matlab_script,
        *[
            data_path.parent / name
            for name in spec.supporting_matlab_scripts
        ],
        data_path.parent / (
            "Gen_M2Mat_sp.m" if spec.loader_cutting == 3 else "Gen_M2Mat.m"
        ),
        data_path.parent / "generate_full_mat_info_v2.m",
    ]
    source_files = {
        str(path.resolve()): _sha256(path)
        for path in source_candidates
        if path.exists()
    }
    python_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("config.py"),
        Path(__file__).resolve().with_name("decoder.py"),
        Path(__file__).resolve().with_name("matlab_io.py"),
    ]
    provenance: Dict[str, object] = {
        "analysis_spec": spec.to_dict(),
        "mode": result.mode,
        "data_sha256": _sha256(data_path),
        "source_file_sha256": source_files,
        "python_file_sha256": {
            str(path): _sha256(path) for path in python_files
        },
        "source_metadata": epochs.metadata,
        "fold_strategy": result.fold_strategy,
        "standardization_scope": result.standardization_scope,
        "warnings": list(result.warnings),
        "peak_time_ms": result.peak_time_ms,
        "peak_accuracy_smoothed": result.peak_accuracy_smoothed,
        "spatial_window_time_ms": [
            int(result.time_ms[result.spatial_window_indices[0]]),
            int(result.time_ms[result.spatial_window_indices[-1]]),
        ],
        "spatial_window_deviant_aligned_ms": [
            int(result.deviant_aligned_time_ms[result.spatial_window_indices[0]]),
            int(result.deviant_aligned_time_ms[result.spatial_window_indices[-1]]),
        ],
        "top_channels_matlab_exploratory": result.top_channels_matlab.tolist(),
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
    _plot_qc(destination / "decoder_qc.png", spec, result)
    return destination
