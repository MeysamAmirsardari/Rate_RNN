"""Assemble the three leakage-safe repetition posterior maps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .config import ANALYSES


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def aggregate_repetition_maps(
    results_dir: Path = DEFAULT_RESULTS,
) -> tuple[Path, Path, Path]:
    results_dir = Path(results_dir)
    table_path = results_dir / "regression_rep_map_all_positions.csv"
    manifest_path = results_dir / "regression_rep_map_manifest.json"
    figure_path = results_dir / "regression_rep_map_all_positions.png"

    records = []
    maps = {}
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "analysis",
                "animal",
                "recording",
                "deviant_position",
                "repetition",
                "time_ms",
                "deviant_aligned_time_ms",
                "posterior_rep1_like_mean",
                "posterior_rep1_like_sem",
                "posterior_rep1_like_smoothed",
            ]
        )
        for key, spec in ANALYSES.items():
            directory = (
                results_dir / key / "regression_rep_map" / "leakage-safe"
            )
            arrays_path = directory / "regression_rep_map_arrays.npz"
            provenance_path = directory / "provenance.json"
            if not arrays_path.exists() or not provenance_path.exists():
                continue
            arrays = np.load(arrays_path)
            provenance = json.loads(provenance_path.read_text())
            time_ms = arrays["time_ms"]
            deviant_time = arrays["deviant_aligned_time_ms"]
            repetitions = arrays["repetitions"]
            mean = arrays["posterior_mean"]
            sem = arrays["posterior_sem"]
            smoothed = arrays["posterior_smoothed"]
            maps[key] = (spec, deviant_time, repetitions, smoothed)
            for rep_index, repetition in enumerate(repetitions):
                for time_index, time in enumerate(time_ms):
                    writer.writerow(
                        [
                            key,
                            spec.animal,
                            spec.recording,
                            spec.deviant_position,
                            repetition,
                            time,
                            deviant_time[time_index],
                            mean[rep_index, time_index],
                            sem[rep_index, time_index],
                            smoothed[rep_index, time_index],
                        ]
                    )
            records.append(
                {
                    "analysis": key,
                    "animal": spec.animal,
                    "recording": spec.recording,
                    "deviant_position": spec.deviant_position,
                    "data_sha256": provenance["data_sha256"],
                    "output": str(directory.resolve()),
                }
            )

    if not records:
        raise FileNotFoundError("No leakage-safe regression repetition maps found")
    missing = sorted(set(ANALYSES) - set(maps))
    manifest_path.write_text(
        json.dumps(
            {
                "description": (
                    "Descriptive, block-held-out ridge-logistic repetition "
                    "posterior maps for the three Zaatar recordings. No "
                    "cross-recording population inference is performed."
                ),
                "posterior": "P(Rep 1-like | ECoG)",
                "records": records,
                "missing_analyses": missing,
            },
            indent=2,
        )
        + "\n"
    )

    figure, axes = plt.subplots(
        1, len(ANALYSES), figsize=(15, 5.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    maximum_deviation = max(
        float(np.max(np.abs(posterior - 0.5)))
        for _, _, _, posterior in maps.values()
    )
    maximum_deviation = max(maximum_deviation, np.finfo(float).eps)
    norm = TwoSlopeNorm(
        vmin=0.5 - maximum_deviation,
        vcenter=0.5,
        vmax=0.5 + maximum_deviation,
    )
    image = None
    for axis, key in zip(axes, ANALYSES):
        if key not in maps:
            axis.set_axis_off()
            continue
        spec, time, repetitions, posterior = maps[key]
        image = axis.imshow(
            posterior,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=[time[0] - 0.5, time[-1] + 0.5, 15.5, 0.5],
            cmap="RdBu_r",
            norm=norm,
        )
        for relative_boundary in (
            -spec.deviant_onset_ms,
            180 - spec.deviant_onset_ms,
            360 - spec.deviant_onset_ms,
            540 - spec.deviant_onset_ms,
        ):
            axis.axvline(relative_boundary, color="0.25", lw=0.65)
        axis.axvspan(0, 180, fill=False, edgecolor="black", ls="--", lw=1.4)
        axis.set(
            xlim=(-360, 800),
            ylim=(15.5, 0.5),
            yticks=[1, 5, 10, 15],
            title=f"Deviant position {spec.deviant_position}",
            xlabel="Time from deviant onset (ms)",
        )
    axes[0].set_ylabel("Repetition number")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.9, pad=0.015)
        colorbar.set_label("P(Rep 1-like | ECoG)")
    figure.suptitle(
        "Roving repetition posterior maps (block-held-out regression)",
        fontweight="bold",
    )
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)
    return table_path, manifest_path, figure_path


if __name__ == "__main__":
    table, manifest, figure = aggregate_repetition_maps()
    print(f"Saved {table}")
    print(f"Saved {manifest}")
    print(f"Saved {figure}")
