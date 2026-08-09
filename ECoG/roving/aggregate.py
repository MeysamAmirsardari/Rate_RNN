"""Merge recording-level leakage-safe exports on a deviant-aligned axis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .config import ANALYSES, DEFAULT_SOURCE_DIR


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def aggregate_recordings(results_dir: Path = DEFAULT_RESULTS) -> tuple[Path, Path]:
    """Create a long table and overlay without treating recordings as IID."""

    results_dir = Path(results_dir)
    records: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, str]]] = {}
    for provenance_path in sorted(
        results_dir.glob("*/leakage-safe/provenance.json")
    ):
        with provenance_path.open(encoding="utf-8") as stream:
            provenance = json.load(stream)
        key = provenance["analysis_spec"]["key"]
        curve_path = provenance_path.parent / "figure_data" / "decoding_timecourse.csv"
        curves[key] = _read_csv(curve_path)
        records.append(
            {
                "analysis": key,
                "animal": provenance["analysis_spec"]["animal"],
                "recording": provenance["analysis_spec"]["recording"],
                "deviant_position": provenance["analysis_spec"]["deviant_position"],
                "rep_first": provenance["analysis_spec"]["rep_first"],
                "rep_late_numeric": provenance["analysis_spec"]["rep_late"],
                "data_sha256": provenance["data_sha256"],
                "provenance": str(provenance_path.resolve()),
            }
        )
    if not records:
        raise FileNotFoundError(
            f"No leakage-safe recording exports found below {results_dir}"
        )

    table_path = results_dir / "figure2_ecog_recording_timecourses.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "analysis",
            "animal",
            "recording",
            "deviant_position",
            "rep_first",
            "rep_late_numeric",
            "time_ms",
            "deviant_aligned_time_ms",
            "accuracy",
            "accuracy_smoothed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        by_key = {record["analysis"]: record for record in records}
        for key, rows in curves.items():
            identity = by_key[key]
            for row in rows:
                writer.writerow(
                    {
                        **{
                            field: identity[field]
                            for field in fields[:6]
                        },
                        **row,
                    }
                )

    manifest_path = results_dir / "figure2_ecog_manifest.json"
    completed = {record["analysis"] for record in records}
    missing = [
        {
            "analysis": key,
            "status": "missing exact source recording",
            "expected_data_file": str(
                spec.data_path(DEFAULT_SOURCE_DIR).resolve()
            ),
            "matlab_script": str(
                spec.source_script_path(DEFAULT_SOURCE_DIR).resolve()
            ),
        }
        for key, spec in ANALYSES.items()
        if key not in completed
    ]
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "description": (
                    "Recording-level leakage-safe decoder curves aligned to "
                    "the deviant. No across-recording mean or CI is computed "
                    "because available recordings are not independent animals."
                ),
                "records": records,
                "missing_analyses": missing,
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for key, rows in curves.items():
        x = [float(row["deviant_aligned_time_ms"]) for row in rows]
        y = [float(row["accuracy_smoothed"]) for row in rows]
        axis.plot(x, y, lw=1.8, label=key)
    axis.axhline(0.5, color="black", ls="--", lw=1)
    axis.axvspan(0, 180, color="#ffd966", alpha=0.25)
    axis.set(
        xlim=(-360, 440),
        ylim=(0.3, 1.0),
        xlabel="Time relative to deviant onset (ms)",
        ylabel="Leakage-safe decoding accuracy",
        title="Recording-level ECoG curves (no pooled inference)",
    )
    axis.legend(frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure_path = results_dir / "figure2_ecog_recording_overlay.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    return table_path, manifest_path


if __name__ == "__main__":
    table, manifest = aggregate_recordings()
    print(f"Saved {table}")
    print(f"Saved {manifest}")
