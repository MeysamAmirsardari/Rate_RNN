"""Read the immutable same-sequence AB/BA MATLAB-reference traces for Figure 4.

The supplied Open Ephys archive does not contain the per-sequence playback
table required to reconstruct event labels.  Consequently this module never
refits the decoder and never infers labels from ECoG. It reads only the exact
curves extracted from the six supplied MATLAB ``.fig`` files. Each file
contrasts one physical sequence when rare on one recording day with that same
physical sequence when regular on the reversed-probability day. These curves
are therefore marked as legacy, descriptive, and day-confounded throughout
the data contract.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REFERENCE_DIR = REPO / "ECoG" / "ab_ba" / "results" / "reference_extracted"
MANIFEST = REPO / "ECoG" / "ab_ba" / "results" / "reference_figures.json"
ERP_DISPLAY_SMOOTHING_SIGMA_MS = 3.0
ERP_DISPLAY_SMOOTHING_TRUNCATE = 4.0


@dataclass(frozen=True)
class LegacyComparison:
    key: str
    family: str
    family_short: str
    sequence_code: str
    sequence: str
    sequence_plain: str
    source_name: str
    target_onset_ms: float
    target_duration_ms: float
    selected_channel: int
    time_ms: np.ndarray
    unexpected: np.ndarray
    predicted: np.ndarray
    decoder_time_ms: np.ndarray
    decoder_accuracy: np.ndarray
    source_file: Path
    source_sha256: str


_SPECS = (
    {
        "key": "exp1_day1_deviant",
        "family": "Continuous tones",
        "family_short": "180-ms tones",
        "sequence_code": "AB",
        "sequence": "5.3→9.4 kHz",
        "sequence_plain": "5300-9400",
        "source_name": "Nugmeg_2026-04-30_new_exp1_5300-9400.tsv.gz",
        "target_onset_ms": 180.0,
        "target_duration_ms": 180.0,
    },
    {
        "key": "exp1_day2_deviant",
        "family": "Continuous tones",
        "family_short": "180-ms tones",
        "sequence_code": "BA",
        "sequence": "9.4→5.3 kHz",
        "sequence_plain": "9400-5300",
        "source_name": "Nugmeg_2026-04-30_new_exp1_9400-5300.tsv.gz",
        "target_onset_ms": 180.0,
        "target_duration_ms": 180.0,
    },
    {
        "key": "exp2_day1_deviant",
        "family": "Speech tokens",
        "family_short": "speech",
        "sequence_code": "AB",
        "sequence": "/dɑː/→/peɪ/",
        "sequence_plain": "dah-pey",
        "source_name": "Nugmeg_2026-04-30_new_exp2_dah-pey.tsv.gz",
        "target_onset_ms": 180.0,
        "target_duration_ms": 180.0,
    },
    {
        "key": "exp2_day2_deviant",
        "family": "Speech tokens",
        "family_short": "speech",
        "sequence_code": "BA",
        "sequence": "/peɪ/→/dɑː/",
        "sequence_plain": "pey-dah",
        "source_name": "Nugmeg_2026-04-30_new_exp2_pey-dah.tsv.gz",
        "target_onset_ms": 180.0,
        "target_duration_ms": 180.0,
    },
    {
        "key": "exp3_day1_deviant",
        "family": "Gapped tones",
        "family_short": "50-ms tones",
        "sequence_code": "AB",
        "sequence": "4.0→1.5 kHz",
        "sequence_plain": "4000-1500",
        "source_name": "Nugmeg_2026-04-30_new_exp3_4000-1500.tsv.gz",
        "target_onset_ms": 150.0,
        "target_duration_ms": 50.0,
    },
    {
        "key": "exp3_day2_deviant",
        "family": "Gapped tones",
        "family_short": "50-ms tones",
        "sequence_code": "BA",
        "sequence": "1.5→4.0 kHz",
        "sequence_plain": "1500-4000",
        "source_name": "Nugmeg_2026-04-30_new_exp3_1500-4000.tsv.gz",
        "target_onset_ms": 150.0,
        "target_duration_ms": 50.0,
    },
)


def _one(spec: dict, manifest: dict) -> LegacyComparison:
    source = REFERENCE_DIR / spec["source_name"]
    frame = pd.read_csv(source, sep="\t")
    required = {"object_id", "kind", "display_name", "x", "y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Malformed reference extraction: {source}")

    def series(object_id: int) -> tuple[np.ndarray, np.ndarray]:
        rows = frame.loc[frame["object_id"] == object_id].sort_values("sample")
        if rows.empty:
            raise ValueError(f"Object {object_id} absent from {source.name}")
        return rows["x"].to_numpy(float), rows["y"].to_numpy(float)

    # Objects 4/5 are the rank-1 channel traces in the literal MATLAB figure;
    # object 11 is the displayed 20-sample moving-mean decoder accuracy.
    x_u, unexpected = series(4)
    x_p, predicted = series(5)
    x_d, accuracy = series(11)
    if not np.array_equal(x_u, x_p):
        raise ValueError(f"Unexpected/predicted clocks differ in {source.name}")
    if unexpected.size != predicted.size or x_d.size != accuracy.size:
        raise ValueError(f"Reference array length mismatch in {source.name}")

    entry = manifest["reference_figures"][spec["key"]]
    selected = int(entry["legacy_top_channels_matlab"][0])
    # MATLAB plotted 1:n on the sequence clock. Scientific time is zero based.
    response_time = x_u - 1.0
    decoder_time = x_d - 1.0
    return LegacyComparison(
        key=spec["key"],
        family=spec["family"],
        family_short=spec["family_short"],
        sequence_code=spec["sequence_code"],
        sequence=spec["sequence"],
        sequence_plain=spec["sequence_plain"],
        source_name=spec["source_name"],
        target_onset_ms=float(spec["target_onset_ms"]),
        target_duration_ms=float(spec["target_duration_ms"]),
        selected_channel=selected,
        time_ms=response_time,
        unexpected=unexpected,
        predicted=predicted,
        decoder_time_ms=decoder_time,
        decoder_accuracy=accuracy,
        source_file=Path(entry["file"]),
        source_sha256=str(entry["sha256"]),
    )


def load_reference() -> tuple[LegacyComparison, ...]:
    with MANIFEST.open() as stream:
        manifest = json.load(stream)
    comparisons = tuple(_one(spec, manifest) for spec in _SPECS)
    if len(comparisons) != 6:
        raise AssertionError("Figure 4 requires six counterbalanced comparisons")
    return comparisons


def smooth_erp_for_display(
    time_ms: np.ndarray,
    values: np.ndarray,
    *,
    sigma_ms: float = ERP_DISPLAY_SMOOTHING_SIGMA_MS,
    truncate: float = ERP_DISPLAY_SMOOTHING_TRUNCATE,
) -> np.ndarray:
    """Apply the fixed, zero-phase Gaussian transform used only in Panel B.

    The finite kernel is locally renormalized at the boundaries, so samples
    outside the observed record are neither zero-padded nor reflected.  This
    helper is deliberately separate from every decoder and inferential path.
    """

    time = np.asarray(time_ms, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.ndim != 1 or time.shape != signal.shape:
        raise ValueError("time_ms and values must be matching one-dimensional arrays")
    if time.size < 2 or not np.isfinite(time).all() or not np.isfinite(signal).all():
        raise ValueError("display smoothing requires finite sampled traces")
    step = float(np.median(np.diff(time)))
    if step <= 0 or not np.allclose(np.diff(time), step, rtol=0.0, atol=1e-9):
        raise ValueError("display smoothing requires a uniformly increasing time grid")
    if sigma_ms <= 0 or truncate <= 0:
        raise ValueError("sigma_ms and truncate must be positive")

    sigma_samples = float(sigma_ms) / step
    radius = max(1, int(np.ceil(float(truncate) * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    numerator = np.convolve(signal, kernel, mode="same")
    denominator = np.convolve(np.ones_like(signal), kernel, mode="same")
    return numerator / denominator


def export_panel_data(comparisons: tuple[LegacyComparison, ...], path: Path) -> None:
    """Save immutable source values plus the explicitly labelled display trace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as stream:
        fields = (
            "comparison", "family", "sequence_code", "sequence", "selected_channel",
            "series", "sequence_time_ms", "value", "display_transform", "status",
            "source_file", "source_sha256",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for role, values, status in (
                (
                    "unexpected_mean_raw",
                    comparison.unexpected,
                    "legacy_cross_day_descriptive_source_extracted",
                ),
                (
                    "predicted_mean_raw",
                    comparison.predicted,
                    "legacy_cross_day_descriptive_source_extracted",
                ),
                (
                    "unexpected_mean_display_gaussian_sigma_3ms",
                    smooth_erp_for_display(comparison.time_ms, comparison.unexpected),
                    "display_only_zero_phase_gaussian_sigma_3ms",
                ),
                (
                    "predicted_mean_display_gaussian_sigma_3ms",
                    smooth_erp_for_display(comparison.time_ms, comparison.predicted),
                    "display_only_zero_phase_gaussian_sigma_3ms",
                ),
            ):
                for time, value in zip(comparison.time_ms, values):
                    writer.writerow({
                        "comparison": comparison.key,
                        "family": comparison.family,
                        "sequence_code": comparison.sequence_code,
                        "sequence": comparison.sequence_plain,
                        "selected_channel": comparison.selected_channel,
                        "series": role,
                        "sequence_time_ms": f"{time:.9g}",
                        "value": f"{value:.12g}",
                        "display_transform": (
                            "gaussian_sigma_3ms_radius_4sigma_local_boundary_normalization"
                            if "display_gaussian" in role
                            else "none"
                        ),
                        "status": status,
                        "source_file": str(comparison.source_file),
                        "source_sha256": comparison.source_sha256,
                    })
            for time, value in zip(
                comparison.decoder_time_ms, comparison.decoder_accuracy
            ):
                writer.writerow({
                    "comparison": comparison.key,
                    "family": comparison.family,
                    "sequence_code": comparison.sequence_code,
                    "sequence": comparison.sequence_plain,
                    "selected_channel": comparison.selected_channel,
                    "series": "legacy_smoothed_decoder_accuracy",
                    "sequence_time_ms": f"{time:.9g}",
                    "value": f"{value:.12g}",
                    "display_transform": "source_20_sample_moving_mean",
                    "status": "legacy_cross_day_descriptive_source_20_sample_movmean",
                    "source_file": str(comparison.source_file),
                    "source_sha256": comparison.source_sha256,
                })
