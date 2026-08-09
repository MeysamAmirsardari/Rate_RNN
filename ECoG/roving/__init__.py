"""Roving-paradigm ECoG decoding.

The package contains a traceable Python translation of the MATLAB analyses in
the sibling ``ECoG`` repository plus a leakage-safe analysis profile.
"""

from .config import ANALYSES, AnalysisSpec
from .decoder import DecoderResult, run_decoder
from .matlab_io import (
    RovingEpochs,
    RovingRepetitionEpochs,
    extract_repetition_epochs,
    extract_roving_epochs,
)
from .repetition_map import (
    RepetitionMapConfig,
    RepetitionMapResult,
    run_repetition_map,
)

__all__ = [
    "ANALYSES",
    "AnalysisSpec",
    "DecoderResult",
    "RovingEpochs",
    "RovingRepetitionEpochs",
    "RepetitionMapConfig",
    "RepetitionMapResult",
    "extract_repetition_epochs",
    "extract_roving_epochs",
    "run_decoder",
    "run_repetition_map",
]
