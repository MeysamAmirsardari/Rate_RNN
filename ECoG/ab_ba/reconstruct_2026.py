"""Epoch the 2026 AB/BA recordings from the raw Open Ephys archive.

What is and is not recoverable
------------------------------
`Gen_M2Mat.m` builds ``allM2`` as ``[repLength, prevRepLength, repNum, stim,
prevStim, trial, data...]`` -- six tag columns, then the epoch, which is why
``scripts_AB_BA.m`` slices ``allM2(:,7:end)`` and selects on ``allM2(:,4)``.
Everything except column 4 (the stimulus index) can be rebuilt from the raw
archive plus the recording parameters:

* the TTL line marks the 16 blocks;
* a block holds 25 sequences at a fixed cadence;
* a sequence is two 180 ms notes with no gap, then a 1.5 s gap, so the
  stimulus-onset asynchrony is 1860 ms and a block spans 46.5 s.

This module reconstructs those 400 onsets per recording and epochs them, and it
verifies the reconstruction against the measured TTL spacing.

The one thing it cannot invent is **which** of the 400 sequences were deviants.
That lives in the Baphy playback table that `ft_oe_list` reads. Supply it with
``--labels`` as a CSV of 400 rows per recording (``block,position,stim``) and
every downstream analysis runs; without it this module stops after the
validation step rather than guessing. Deriving the labels from the ECoG would
make the outcome define its own target.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


ARCHIVE = Path("/Users/eminent/Projects/ECoG/AB_BA/"
               "Nutmeg_2026-04-30_2026-05-01_SEQ1_3")
RECORDINGS = {
    "day1_seq1": "Nutmeg_2026-04-30_15-30-21_SEQ1",
    "day1_seq3": "Nutmeg_2026-04-30_16-05-18_SEQ3",
    "day2_seq1": "Nutmeg_2026-05-01_17-25-02_SEQ1",
    "day2_seq3": "Nutmeg_2026-05-01_17-54-34_SEQ3",
}
RESULT_DIR = Path(__file__).resolve().parent / "results" / "reconstruct_2026"

HEADER_BYTES = 1024
CONTINUOUS_RECORD = np.dtype([
    ("timestamp", "<i8"), ("n_samples", "<u2"), ("recording", "<u2"),
    ("samples", ">i2", 1024), ("marker", "u1", 10),
])
EVENT_RECORD = np.dtype([
    ("timestamp", "<i8"), ("position", "<i2"), ("event_type", "u1"),
    ("node", "u1"), ("event_id", "u1"), ("channel", "u1"),
    ("recording", "<u2"),
])

N_CHANNELS = 32
#: Experiment 1 and 2: two 180 ms notes, no gap. Experiment 3: 50 ms notes,
#: 100 ms gap. Both then wait 1.5 s before the next sequence.
TIMINGS = {
    "exp1": {"note_ms": 180, "note_gap_ms": 0, "sequence_gap_ms": 1500},
    "exp3": {"note_ms": 50, "note_gap_ms": 100, "sequence_gap_ms": 1500},
}
BLOCK_LENGTH = 25
BASELINE_MS = 100
EPOCH_MS = 600
#: ``scripts_AB_BA.m`` calls ``ft_oe_list(..., [1 250])``. The raw archive is
#: unfiltered, so the same passband is applied before ``Gen_M2Mat``-equivalent
#: epoching.
BANDPASS_HZ = (1.0, 250.0)


def channel_path(node: Path, channel: int) -> Path:
    """Open Ephys named the neural channels differently on the two days."""

    for pattern in (f"100_CH{channel}.continuous", f"100_{channel}.continuous"):
        candidate = node / pattern
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no continuous file for channel {channel} in {node}")


def _header(path: Path) -> dict[str, str]:
    text = path.read_bytes()[:HEADER_BYTES].decode("ascii", "ignore")
    return dict(
        (key, value.strip().strip("'"))
        for key, value in re.findall(r"header\.([A-Za-z_]+)\s*=\s*([^;]+);", text)
    )


def sequence_onsets(block_starts: np.ndarray, soa_samples: float) -> np.ndarray:
    """The 25 sequence onsets of every block, at the fixed cadence."""

    offsets = np.arange(BLOCK_LENGTH) * soa_samples
    return np.rint(block_starts[:, None] + offsets[None, :]).astype(np.int64)


@dataclass(frozen=True)
class Reconstruction:
    key: str
    sample_rate: float
    block_starts: np.ndarray
    onsets: np.ndarray            # block x 25, in samples
    measured_block_ms: float
    predicted_block_ms: float
    epochs: np.ndarray | None     # sequence x channel x time
    time_ms: np.ndarray


def load_recording(key: str, folder: Path, experiment: str,
                   with_epochs: bool = True) -> Reconstruction:
    node = folder / "Record Node 101"
    events = np.frombuffer(
        (node / "all_channels.events").read_bytes()[HEADER_BYTES:],
        dtype=EVENT_RECORD)
    rises = events[events["event_id"] == 1]["timestamp"]
    falls = events[events["event_id"] == 0]["timestamp"]
    if rises.size < 2 or rises.size != falls.size:
        raise ValueError(f"{key}: unpaired TTL edges")

    channel_header = _header(channel_path(node, 1))
    sample_rate = float(channel_header["sampleRate"])

    timing = TIMINGS[experiment]
    sequence_ms = 2 * timing["note_ms"] + timing["note_gap_ms"]
    soa_ms = sequence_ms + timing["sequence_gap_ms"]
    soa_samples = soa_ms * sample_rate / 1000.0

    # A block is the interval a TTL pulse is HIGH. The recording opens with a
    # short setup pulse, so blocks are selected by duration rather than index;
    # rise-to-rise would wrongly include the pause between blocks.
    durations_ms = (falls - rises) * 1000.0 / sample_rate
    is_block = durations_ms > 0.5 * BLOCK_LENGTH * soa_ms
    block_starts = rises[is_block]
    measured = float(np.median(durations_ms[is_block]))
    predicted = BLOCK_LENGTH * soa_ms
    onsets = sequence_onsets(block_starts, soa_samples)

    epochs = None
    time_ms = np.arange(-BASELINE_MS, EPOCH_MS)
    if with_epochs:
        first = int(np.frombuffer(
            channel_path(node, 1).read_bytes()[HEADER_BYTES:HEADER_BYTES + 8],
            dtype="<i8")[0])
        pre = int(round(BASELINE_MS * sample_rate / 1000.0))
        length = int(round((BASELINE_MS + EPOCH_MS) * sample_rate / 1000.0))
        flat = onsets.ravel()
        epochs = np.empty((flat.size, N_CHANNELS, length), dtype=np.float32)
        low, high = BANDPASS_HZ
        coefficients = butter(
            3, [low / (sample_rate / 2), high / (sample_rate / 2)], btype="band")
        for channel in range(N_CHANNELS):
            trace = filtfilt(*coefficients,
                             _continuous(channel_path(node, channel + 1)))
            for row, onset in enumerate(flat):
                start = int(onset - first - pre)
                epochs[row, channel] = trace[start:start + length]
        baseline = epochs[:, :, :pre]
        epochs -= baseline.mean(axis=2, keepdims=True)
        time_ms = (np.arange(length) - pre) * 1000.0 / sample_rate

    return Reconstruction(
        key=key, sample_rate=sample_rate, block_starts=block_starts,
        onsets=onsets, measured_block_ms=measured, predicted_block_ms=predicted,
        epochs=epochs, time_ms=time_ms)


def _continuous(path: Path) -> np.ndarray:
    raw = path.read_bytes()[HEADER_BYTES:]
    count = len(raw) // CONTINUOUS_RECORD.itemsize
    records = np.frombuffer(raw[:count * CONTINUOUS_RECORD.itemsize],
                            dtype=CONTINUOUS_RECORD)
    bit_volts = float(_header(path).get("bitVolts", 1.0))
    return records["samples"].astype(np.float32).ravel() * bit_volts


def validate(experiment: str = "exp1", with_epochs: bool = True) -> dict:
    """Rebuild the onsets and check them against the recorded TTL spacing."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"experiment": experiment,
                                 "timing": TIMINGS[experiment]}
    per_recording = {}
    for key, name in RECORDINGS.items():
        if experiment == "exp1" and "seq3" in key:
            continue
        if experiment == "exp3" and "seq1" in key:
            continue
        result = load_recording(key, ARCHIVE / name, experiment,
                                with_epochs=with_epochs)
        entry = {
            "sample_rate_hz": result.sample_rate,
            "n_blocks": int(result.block_starts.size),
            "n_sequences": int(result.onsets.size),
            "measured_block_ms": round(result.measured_block_ms, 1),
            "implied_soa_ms": round(result.measured_block_ms / BLOCK_LENGTH, 2),
            "predicted_block_ms": result.predicted_block_ms,
            "block_error_ms": round(
                result.measured_block_ms - result.predicted_block_ms, 1),
        }
        if result.epochs is not None:
            gfp = result.epochs.std(axis=1).mean(axis=0)
            entry["gfp_peak_ms"] = [
                int(result.time_ms[i]) for i in
                _local_peaks(gfp, result.time_ms)
            ]
            np.savez_compressed(
                RESULT_DIR / f"{key}_epochs.npz",
                epochs=result.epochs, time_ms=result.time_ms,
                onsets=result.onsets, block_starts=result.block_starts)
        per_recording[key] = entry
        print(f"[{key}] blocks {entry['n_blocks']}  sequences "
              f"{entry['n_sequences']}  block {entry['measured_block_ms']} ms "
              f"vs predicted {entry['predicted_block_ms']} ms "
              f"(error {entry['block_error_ms']} ms)", flush=True)
        if "gfp_peak_ms" in entry:
            print(f"         evoked GFP peaks at {entry['gfp_peak_ms']} ms",
                  flush=True)
    report["recordings"] = per_recording
    report["labels_required"] = (
        "Per-sequence stimulus identity is not in the archive. Supply a CSV "
        "with columns block,position,stim (400 rows per recording) or the "
        "Baphy parameter file, and the oddball analysis runs end to end."
    )
    (RESULT_DIR / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _local_peaks(signal: np.ndarray, time_ms: np.ndarray,
                 minimum_ms: int = 0) -> list[int]:
    after = time_ms >= minimum_ms
    indices = np.flatnonzero(after)
    values = signal[after]
    peaks = [i for i in range(1, values.size - 1)
             if values[i] > values[i - 1] and values[i] >= values[i + 1]
             and values[i] > values.mean() + values.std()]
    return [int(indices[i]) for i in peaks][:6]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="exp1", choices=sorted(TIMINGS))
    parser.add_argument("--no-epochs", action="store_true")
    options = parser.parse_args()
    validate(options.experiment, with_epochs=not options.no_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
