"""Configuration copied from the three source MATLAB decoder scripts."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


_PROJECTS_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = Path(
    os.environ.get("ECOG_SOURCE_DIR", _PROJECTS_DIR / "ECoG")
).expanduser()


@dataclass(frozen=True)
class AnalysisSpec:
    """All source-script choices that can change the computation."""

    key: str
    matlab_script: str
    supporting_matlab_scripts: Tuple[str, ...]
    data_file: str
    animal: str
    recording: str
    deviant_position: int
    n_stim: int
    loader_cutting: int
    rep_first: int
    rep_late: int
    expnum: int = 1
    hilbert_t: int = 0
    n_channels: int = 32
    epoch_start_sample_matlab: int = 101
    epoch_end_sample_matlab: int = 901
    lambda_ridge: float = 1e-2
    n_folds: int = 5
    random_seed: int = 11
    noise_sd: float = 1e-6
    smooth_samples: int = 20
    peak_half_window_samples: int = 25
    tone_duration_ms: int = 180

    @property
    def deviant_onset_ms(self) -> int:
        return (self.deviant_position - 1) * self.tone_duration_ms

    def source_script_path(self, source_dir: Path = DEFAULT_SOURCE_DIR) -> Path:
        return source_dir / self.matlab_script

    def data_path(self, source_dir: Path = DEFAULT_SOURCE_DIR) -> Path:
        return source_dir / self.data_file

    def to_dict(self) -> dict:
        return asdict(self)


# The values below are intentionally literal translations, including the
# source scripts' rep_late=14 setting for both Zaatar recordings.
ANALYSES: Dict[str, AnalysisSpec] = {
    "zaatar_pos1": AnalysisSpec(
        key="zaatar_pos1",
        matlab_script="decoding_analysis_xbc.m",
        supporting_matlab_scripts=(),
        data_file="Zaatar_2024-11-22_12-08-48_SRH1.mat",
        animal="Zaatar",
        recording="2024-11-22_12-08-48_SRH1",
        deviant_position=1,
        n_stim=3,
        loader_cutting=3,
        rep_first=1,
        rep_late=14,
    ),
    "zaatar_pos2": AnalysisSpec(
        key="zaatar_pos2",
        matlab_script="decoding_analysis.m",
        supporting_matlab_scripts=(),
        data_file="Zaatar_2024-11-26_13-17-55_SRH.mat",
        animal="Zaatar",
        recording="2024-11-26_13-17-55_SRH",
        deviant_position=2,
        n_stim=3,
        loader_cutting=3,
        rep_first=1,
        rep_late=14,
    ),
    "zaatar_pos3": AnalysisSpec(
        key="zaatar_pos3",
        # decoding_analysis.m contains this file as its alternate input and
        # supplies the same requested logistic-decoder method. wave_analysis.m
        # independently identifies the recording as third-tone deviant and
        # confirms Gen_M2Mat_sp, nStim=3, and repetitions [1,14].
        matlab_script="decoding_analysis.m",
        supporting_matlab_scripts=("wave_analysis.m", "pe_vs_pc.m"),
        data_file="Zaatar_2024-11-27_xx.mat",
        animal="Zaatar",
        recording="2024-11-27_10-42-16_SRH",
        deviant_position=3,
        n_stim=3,
        loader_cutting=3,
        rep_first=1,
        rep_late=14,
    ),
}
