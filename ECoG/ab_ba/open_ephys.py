"""Read-only inventory checks for the supplied legacy Open Ephys archive.

This module deliberately does not infer sequence identities from ECoG
responses.  The binary archive contains trial-boundary TTL events, but the
per-sequence playback table used by ``ft_oe_list`` is not embedded in those
events.  Inferring labels from the dependent neural signal would be circular.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


HEADER_BYTES = 1024
CONTINUOUS_RECORD_BYTES = 8 + 2 + 2 + 2 * 1024 + 10
EVENT_DTYPE = np.dtype(
    [
        ("timestamp", "<i8"),
        ("sample_position", "<i2"),
        ("event_type", "u1"),
        ("node_id", "u1"),
        ("event_id", "u1"),
        ("channel", "u1"),
        ("recording_number", "<u2"),
    ]
)


def _header(path: Path) -> Dict[str, str]:
    text = path.read_bytes()[:HEADER_BYTES].decode("ascii", "ignore")
    fields = dict(
        (key, value.strip().strip("'"))
        for key, value in re.findall(r"header\.([A-Za-z_]+)\s*=\s*([^;]+);", text)
    )
    if fields.get("format") != "Open Ephys Data Format":
        raise ValueError(f"Unexpected Open Ephys header in {path}")
    return fields


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_inventory(path: Path, sample_rate_hz: float) -> Dict[str, object]:
    payload = path.read_bytes()[HEADER_BYTES:]
    if len(payload) % EVENT_DTYPE.itemsize:
        raise ValueError(f"Truncated event record in {path}")
    events = np.frombuffer(payload, dtype=EVENT_DTYPE)
    rises = events[(events["event_id"] == 1) & (events["channel"] == 3)]
    falls = events[(events["event_id"] == 0) & (events["channel"] == 3)]
    if rises.size != falls.size:
        raise ValueError(f"Unpaired TTL edges in {path}")
    durations = (falls["timestamp"] - rises["timestamp"]) / sample_rate_hz
    trial_mask = durations > 5.0
    trial_rises = rises["timestamp"][trial_mask]
    trial_falls = falls["timestamp"][trial_mask]
    gaps = (
        (trial_rises[1:] - trial_falls[:-1]) / sample_rate_hz
        if trial_rises.size > 1
        else np.array([], dtype=float)
    )
    return {
        "n_events": int(events.size),
        "ttl_channel": 3,
        "n_ttl_pairs": int(rises.size),
        "n_short_setup_pulses": int(np.sum(~trial_mask)),
        "n_long_trial_pulses": int(np.sum(trial_mask)),
        "trial_duration_s": durations[trial_mask].tolist(),
        "inter_trial_gap_s": gaps.tolist(),
        "event_sha256": _sha256(path),
    }


def inventory_recording(recording_dir: Path) -> Dict[str, object]:
    node = Path(recording_dir) / "Record Node 101"
    if not node.is_dir():
        raise FileNotFoundError(f"Missing Record Node 101 in {recording_dir}")
    continuous = sorted(node.glob("*.continuous"))
    if not continuous:
        raise FileNotFoundError(f"No .continuous files in {node}")
    channels: List[dict] = []
    neural = []
    starts = []
    sample_counts = []
    sample_rate = None
    for path in continuous:
        header = _header(path)
        rate = float(header["sampleRate"])
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Mixed sample rates in {node}")
        payload = path.stat().st_size - HEADER_BYTES
        if payload % CONTINUOUS_RECORD_BYTES:
            raise ValueError(f"Truncated continuous record in {path}")
        n_records = payload // CONTINUOUS_RECORD_BYTES
        with path.open("rb") as stream:
            stream.seek(HEADER_BYTES)
            first_timestamp = int(np.frombuffer(stream.read(8), dtype="<i8")[0])
        n_samples = int(n_records * 1024)
        channel = header.get("channel", path.stem)
        is_neural = bool(re.fullmatch(r"CH(?:[1-9]|[12][0-9]|3[0-2])", channel))
        if is_neural:
            neural.append(channel)
        starts.append(first_timestamp)
        sample_counts.append(n_samples)
        channels.append(
            {
                "file": path.name,
                "header_channel": channel,
                "bit_volts": float(header["bitVolts"]),
                "n_records": int(n_records),
                "n_samples": n_samples,
                "first_timestamp": first_timestamp,
                "is_neural_channel": is_neural,
            }
        )
    if len(neural) != 32:
        raise ValueError(f"Expected 32 neural channels in {node}; found {len(neural)}")
    if len(set(starts)) != 1 or len(set(sample_counts)) != 1:
        raise ValueError(f"Channel timestamp/length mismatch in {node}")
    event_path = node / "all_channels.events"
    result: Dict[str, object] = {
        "recording": Path(recording_dir).name,
        "record_node": str(node.resolve()),
        "sample_rate_hz": sample_rate,
        "n_neural_channels": len(neural),
        "neural_channels": sorted(neural, key=lambda x: int(x[2:])),
        "n_continuous_files": len(continuous),
        "n_samples_per_channel": sample_counts[0],
        "duration_s": sample_counts[0] / float(sample_rate),
        "first_timestamp": starts[0],
        "channels": channels,
    }
    result.update(_event_inventory(event_path, float(sample_rate)))
    return result


def inventory_archive(source_dir: Path) -> Dict[str, object]:
    source_dir = Path(source_dir)
    archive_dirs = sorted(
        path
        for parent in source_dir.glob("Nutmeg_2026-04-30_2026-05-01_SEQ1_3")
        for path in parent.glob("Nutmeg_*_SEQ*")
        if path.is_dir()
    )
    if len(archive_dirs) != 4:
        raise FileNotFoundError(
            f"Expected four Nutmeg recording directories under {source_dir}; "
            f"found {len(archive_dirs)}"
        )
    return {
        "source_dir": str(source_dir.resolve()),
        "recordings": [inventory_recording(path) for path in archive_dirs],
        "labeling_boundary": (
            "Open Ephys TTLs delimit whole 25-sequence trials only. Per-sequence "
            "identity/order is supplied by the absent ft_oe_list playback table; "
            "it is never inferred from ECoG responses."
        ),
    }
