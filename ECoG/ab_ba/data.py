"""Load the compact, lossless MATLAB preprocessing export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import loadmat

from .config import COMPARISONS, ComparisonSpec, EXPERIMENTS


@dataclass(frozen=True)
class ABBAEpochs:
    """Balanced inputs for one physical sequence comparison."""

    deviant: np.ndarray  # channels x observations x time
    standard_after_deviant: np.ndarray  # channels x observations x time
    deviant_groups: np.ndarray
    standard_groups: np.ndarray
    deviant_trials: np.ndarray
    standard_trials: np.ndarray
    deviant_source_rows_matlab: np.ndarray
    standard_source_rows_matlab: np.ndarray
    time_ms: np.ndarray
    source_time_labels_ms: np.ndarray
    metadata: Dict[str, object]


def _vector(value: object, *, dtype=int) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _require_shape(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be channels x observations x time; got {array.shape}")
    return array


def _as_text(value: object) -> str:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return str(value)


def _canonical_sequence(value: str) -> str:
    # The protocol table uses "daa" while the saved figure filenames use
    # "dah" for the same spoken token. This spelling alias is documented and
    # does not relax order or tone-sequence matching.
    return value.strip().casefold().replace("dah", "daa")


def load_export(
    path: Path, comparison: str | ComparisonSpec
) -> ABBAEpochs:
    """Load one comparison without guessing or substituting source data."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing export: {path}\n"
            "Run ECoG/ab_ba/matlab/export_ab_ba_preprocessed.m in the "
            "original MATLAB environment containing ft_oe_list.m."
        )
    spec = COMPARISONS[comparison] if isinstance(comparison, str) else comparison
    loaded = loadmat(path, variable_names=["comparisons", "export_metadata"],
                     simplify_cells=True)
    comparisons = loaded.get("comparisons")
    if not isinstance(comparisons, dict) or spec.key not in comparisons:
        raise ValueError(f"{path} does not contain comparisons.{spec.key}")
    item = comparisons[spec.key]
    if not isinstance(item, dict):
        raise ValueError(f"comparisons.{spec.key} is not a MATLAB struct")

    deviant = _require_shape("x_deviant", item["x_deviant"])
    standard = _require_shape(
        "x_standard_after_deviant", item["x_standard_after_deviant"]
    )
    if deviant.shape != standard.shape:
        raise ValueError(f"Balanced class shapes differ: {deviant.shape} vs {standard.shape}")
    if deviant.shape[0] != spec.n_channels:
        raise ValueError(f"Expected {spec.n_channels} channels; got {deviant.shape[0]}")

    target = _as_text(item["target_sequence"])
    if _canonical_sequence(target) != _canonical_sequence(
        spec.expected_target_sequence
    ):
        raise ValueError(
            f"{spec.key}: exported target {target!r} does not match the "
            f"protocol value {spec.expected_target_sequence!r}"
        )
    n_obs, n_time = deviant.shape[1:]
    time_ms = _vector(item["time_ms"], dtype=int)
    source_labels = _vector(item["source_time_labels_ms"], dtype=int)
    if time_ms.size != n_time or source_labels.size != n_time:
        raise ValueError("Export time vectors do not match the epoch length")
    if not np.array_equal(time_ms, np.arange(n_time)):
        raise ValueError("Scientific time_ms must be zero-based and contiguous")

    arrays = {
        "deviant_groups": _vector(item["deviant_groups"]),
        "standard_groups": _vector(item["standard_groups"]),
        "deviant_trials": _vector(item["deviant_trials"]),
        "standard_trials": _vector(item["standard_trials"]),
        "deviant_rows": _vector(item["deviant_source_rows_matlab"]),
        "standard_rows": _vector(item["standard_source_rows_matlab"]),
    }
    for name, values in arrays.items():
        if values.size != n_obs:
            raise ValueError(f"{name} has {values.size} values for {n_obs} observations")

    metadata: Dict[str, object] = {
        "comparison_key": spec.key,
        "target_sequence": target,
        "expnum": spec.expnum,
        "deviant_day": spec.deviant_day,
        "standard_source_day": spec.standard_source_day,
        "n_observations_per_class": n_obs,
        "experiment": EXPERIMENTS[spec.expnum].to_dict(),
        "export_metadata": loaded.get("export_metadata", {}),
    }
    for key in (
        "deviant_stimulus_index_matlab",
        "standard_stimulus_index_matlab",
        "n_deviant_before_balance",
        "n_standard_before_balance",
    ):
        if key in item:
            value = np.asarray(item[key]).reshape(-1)
            metadata[key] = int(value[0]) if value.size else None

    return ABBAEpochs(
        deviant=deviant,
        standard_after_deviant=standard,
        deviant_groups=arrays["deviant_groups"],
        standard_groups=arrays["standard_groups"],
        deviant_trials=arrays["deviant_trials"],
        standard_trials=arrays["standard_trials"],
        deviant_source_rows_matlab=arrays["deviant_rows"],
        standard_source_rows_matlab=arrays["standard_rows"],
        time_ms=time_ms,
        source_time_labels_ms=source_labels,
        metadata=metadata,
    )
