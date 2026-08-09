"""Literal experiment and analysis settings for the Nutmeg AB/BA data."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


_PROJECTS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = Path(
    os.environ.get("AB_BA_SOURCE_DIR", _PROJECTS_DIR / "ECoG" / "AB_BA")
).expanduser()
DEFAULT_EXPORT_FILE = DEFAULT_SOURCE_DIR / "ab_ba_preprocessed_export.mat"


@dataclass(frozen=True)
class ExperimentSpec:
    expnum: int
    sequences: Tuple[str, str]
    day1_proportions: Tuple[float, float]
    day2_proportions: Tuple[float, float]
    note_duration_ms: int
    note_gap_ms: int
    sequence_duration_ms: int
    sequence_gap_ms: int = 1500
    block_length: int = 25

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    expnum: int
    deviant_day: int
    expected_target_sequence: str
    n_channels: int = 32
    ridge_lambda: float = 1e-2
    n_folds: int = 5
    reproducibility_seed: int = 11
    noise_sd: float = 1e-6
    smooth_samples: int = 20
    peak_half_window_samples: int = 25

    @property
    def standard_source_day(self) -> int:
        return 2 if self.deviant_day == 1 else 1

    def to_dict(self) -> dict:
        return asdict(self)


# Percentages are transcribed from the supplied experiment table.  The first
# and second values refer to the two sequence strings in the same row.
EXPERIMENTS: Dict[int, ExperimentSpec] = {
    1: ExperimentSpec(
        expnum=1,
        sequences=("9400-5300", "5300-9400"),
        day1_proportions=(0.84, 0.16),
        day2_proportions=(0.15, 0.85),
        note_duration_ms=180,
        note_gap_ms=0,
        sequence_duration_ms=360,
    ),
    2: ExperimentSpec(
        expnum=2,
        sequences=("pey-daa", "daa-pey"),
        day1_proportions=(0.855, 0.145),
        day2_proportions=(0.145, 0.855),
        note_duration_ms=180,
        note_gap_ms=0,
        sequence_duration_ms=360,
    ),
    3: ExperimentSpec(
        expnum=3,
        sequences=("1500-4000", "4000-1500"),
        day1_proportions=(0.85, 0.15),
        day2_proportions=(0.1325, 0.8675),
        note_duration_ms=50,
        note_gap_ms=100,
        sequence_duration_ms=200,
    ),
}


# Stable IDs are based on the recording day supplying the deviant class.  The
# MATLAB export validates the physical sequence name instead of trusting these
# labels silently.
COMPARISONS: Dict[str, ComparisonSpec] = {
    "exp1_day1_deviant": ComparisonSpec(
        "exp1_day1_deviant", 1, 1, "5300-9400"
    ),
    "exp1_day2_deviant": ComparisonSpec(
        "exp1_day2_deviant", 1, 2, "9400-5300"
    ),
    "exp2_day1_deviant": ComparisonSpec(
        "exp2_day1_deviant", 2, 1, "daa-pey"
    ),
    "exp2_day2_deviant": ComparisonSpec(
        "exp2_day2_deviant", 2, 2, "pey-daa"
    ),
    "exp3_day1_deviant": ComparisonSpec(
        "exp3_day1_deviant", 3, 1, "4000-1500"
    ),
    "exp3_day2_deviant": ComparisonSpec(
        "exp3_day2_deviant", 3, 2, "1500-4000"
    ),
}
