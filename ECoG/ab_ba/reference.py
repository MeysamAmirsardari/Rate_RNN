"""Inventory the six supplied MATLAB ``.fig`` reference outputs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import loadmat


FIGURE_TO_COMPARISON = {
    "exp1_5300-9400": "exp1_day1_deviant",
    "exp1_9400-5300": "exp1_day2_deviant",
    "exp2_dah-pey": "exp2_day1_deviant",
    "exp2_pey-dah": "exp2_day2_deviant",
    "exp3_4000-1500": "exp3_day1_deviant",
    "exp3_1500-4000": "exp3_day2_deviant",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_strings(path: Path) -> str:
    loaded = loadmat(path, variable_names=["__function_workspace__"])
    raw = np.asarray(loaded["__function_workspace__"], dtype=np.uint8).tobytes()
    return "\n".join(
        item.decode("latin1") for item in re.findall(rb"[ -~]{4,}", raw)
    )


def reference_manifest(source_dir: Path) -> Dict[str, object]:
    source_dir = Path(source_dir)
    entries: Dict[str, object] = {}
    for path in sorted(source_dir.glob("*.fig")):
        match_key = next(
            (key for key in FIGURE_TO_COMPARISON if key in path.stem), None
        )
        if match_key is None:
            continue
        text = _workspace_strings(path)
        peaks = sorted(set(re.findall(r"Decoder Spatial Pattern @ (\d+) ms", text)))
        channels = sorted(
            set(re.findall(r"Ch (\d+) \(Rank (\d+)\)", text)),
            key=lambda pair: int(pair[1]),
        )
        comparison = FIGURE_TO_COMPARISON[match_key]
        entry = {
            "file": str(path.resolve()),
            "sha256": _sha256(path),
            "legacy_peak_time_source_label_ms": int(peaks[0]) if len(peaks) == 1 else None,
            "legacy_top_channels_matlab": [int(channel) for channel, _ in channels],
            "legacy_top_channel_ranks": [int(rank) for _, rank in channels],
        }
        extracted = (
            Path(__file__).resolve().parent
            / "results"
            / "reference_extracted"
            / f"{path.stem}.tsv.gz"
        )
        if extracted.exists():
            entry["extracted_plot_data"] = str(extracted)
            entry["extracted_plot_data_sha256"] = _sha256(extracted)
        entries[comparison] = entry
    missing = sorted(set(FIGURE_TO_COMPARISON.values()) - set(entries))
    return {
        "reference_figures": entries,
        "plot_data_extractor": {
            "source": str(
                Path(__file__).resolve().parent / "tools" / "ExtractFigData.java"
            ),
            "mat_file_library": "HEBI Robotics MFL 0.5.15",
        },
        "missing_comparisons": missing,
        "interpretation_warnings": [
            "The figures label the classes Rep 1 and Rep 15, but the source arrays "
            "are deviant sequences and same-identity standards immediately after "
            "the opposite deviant on the other acquisition day.",
            "The source MATLAB script does not seed randn or KFold; a rerun cannot "
            "recover the unknown random state used to create these figures.",
            "The reference decoder standardizes all observations before CV and is "
            "not the manuscript-safe inferential result.",
        ],
    }
