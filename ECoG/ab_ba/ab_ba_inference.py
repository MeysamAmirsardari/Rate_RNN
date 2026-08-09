"""Trial-level AB/BA analysis with the recovered Baphy playback table.

The identity-controlled contrast
--------------------------------
Each physical sequence appears in both recordings, as the 15% deviant in one
and the 85% standard in the other:

    5300-9400   deviant on 2026-04-30 (n = 56)   standard on 2026-05-01 (n = 340)
    9400-5300   standard on 2026-04-30 (n = 344) deviant on 2026-05-01 (n = 60)

So "deviant minus standard" can be taken with the acoustics held exactly fixed,
which is what Figure 4 panel A promises. Per the experimenter's instruction the
two recordings are treated as one session; see ``ECOG_INFERENCE_AUDIT.md`` for
what that assumption costs and the re-recording that will remove it.

Two statistics are reported:

``direct``       for one physical sequence, deviant blocks against standard
                 blocks. This is what panel B draws. It is confounded with
                 recording, which the same-session assumption sets aside.
``interaction``  ``[seq1 - seq2](day 1) - [seq1 - seq2](day 2)``, which equals
                 the sum of the two deviant-minus-standard effects and cancels
                 any additive recording difference. Reported alongside as the
                 confound-free version of the same question.

The resampling unit is the acquisition block: 16 per recording, each holding 25
sequences. Blocks are never split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from ECoG.ab_ba.baphy_events import RUN_TO_RECORDING, Playback, read_playback
from ECoG.ab_ba.reconstruct_2026 import (
    ARCHIVE, BANDPASS_HZ, EVENT_RECORD, HEADER_BYTES, N_CHANNELS,
    _continuous, channel_path,
)

RESULT_DIR = Path(__file__).resolve().parent / "results" / "ab_ba_inference"

BASELINE_MS = 100
EPOCH_MS = 600
N_PERMUTATIONS = 9_999
RANDOM_SEED = 20_260_809
ALPHA = 0.05
CLUSTER_ALPHA = 0.05
#: Experiment 1 is the pair Figure 4 displays.
EXPERIMENT = "a01"


@dataclass(frozen=True)
class Session:
    run: str
    playback: Playback
    epochs: np.ndarray        # sequence x channel x time
    block: np.ndarray
    name: np.ndarray
    is_deviant: np.ndarray
    time_ms: np.ndarray


def block_starts(node: Path, sample_rate: float, soa_ms: float,
                 sequences: int) -> np.ndarray:
    """TTL rises that begin an acquisition block, by HIGH duration."""

    events = np.frombuffer(
        (node / "all_channels.events").read_bytes()[HEADER_BYTES:],
        dtype=EVENT_RECORD)
    rises = events[events["event_id"] == 1]["timestamp"]
    falls = events[events["event_id"] == 0]["timestamp"]
    duration_ms = (falls - rises) * 1000.0 / sample_rate
    return rises[duration_ms > 0.5 * sequences * soa_ms]


def load_session(run: str) -> Session:
    playback = read_playback(run)
    node = ARCHIVE / RUN_TO_RECORDING[run] / "Record Node 101"
    header = channel_path(node, 1).read_bytes()[:HEADER_BYTES].decode("ascii", "ignore")
    sample_rate = float(
        [line for line in header.split(";") if "sampleRate" in line][0].split("=")[1])

    starts = block_starts(node, sample_rate, playback.soa_ms,
                          playback.sequences_per_trial)
    if starts.size != playback.trial.max():
        raise ValueError(
            f"{run}: {starts.size} TTL blocks but {playback.trial.max()} Baphy trials")

    first = int(np.frombuffer(
        channel_path(node, 1).read_bytes()[HEADER_BYTES:HEADER_BYTES + 8],
        dtype="<i8")[0])
    onsets = (starts[playback.trial - 1]
              + np.rint(playback.onset_s * sample_rate).astype(np.int64))

    pre = int(round(BASELINE_MS * sample_rate / 1000.0))
    length = int(round((BASELINE_MS + EPOCH_MS) * sample_rate / 1000.0))
    low, high = BANDPASS_HZ
    coefficients = butter(3, [low / (sample_rate / 2), high / (sample_rate / 2)],
                          btype="band")

    epochs = np.empty((onsets.size, N_CHANNELS, length), dtype=np.float32)
    for channel in range(N_CHANNELS):
        trace = filtfilt(*coefficients, _continuous(channel_path(node, channel + 1)))
        for row, onset in enumerate(onsets):
            start = int(onset - first - pre)
            epochs[row, channel] = trace[start:start + length]

    baseline = epochs[:, :, :pre]
    # MATLAB ``std`` defaults to the sample denominator (N - 1).
    scale = baseline.std(axis=2, ddof=1, keepdims=True)
    epochs = (epochs - baseline.mean(axis=2, keepdims=True)) / np.where(
        scale > 0, scale, 1.0)

    return Session(
        run=run, playback=playback, epochs=epochs, block=playback.trial,
        name=playback.name, is_deviant=playback.is_deviant,
        time_ms=(np.arange(length) - pre) * 1000.0 / sample_rate)


def block_means(session: Session, sequence: str) -> np.ndarray:
    """Mean GFP per acquisition block for one physical sequence."""

    gfp = session.epochs.std(axis=1)
    keep = session.name == sequence
    blocks = np.unique(session.block)
    return np.stack([gfp[keep & (session.block == b)].mean(axis=0)
                     for b in blocks])


def two_sample_t(values: np.ndarray, is_a: np.ndarray) -> np.ndarray:
    a, b = values[is_a], values[~is_a]
    na, nb = a.shape[0], b.shape[0]
    pooled = ((na - 1) * a.var(axis=0, ddof=1)
              + (nb - 1) * b.var(axis=0, ddof=1)) / (na + nb - 2)
    se = np.sqrt(pooled * (1.0 / na + 1.0 / nb))
    return (a.mean(axis=0) - b.mean(axis=0)) / np.where(se > 0, se, np.inf)


def _clusters(statistic: np.ndarray, threshold: float):
    found = []
    for sign, chosen in ((1, statistic > threshold), (-1, statistic < -threshold)):
        padded = np.r_[False, chosen, False].astype(np.int8)
        changes = np.diff(padded)
        for start, stop in zip(np.flatnonzero(changes == 1),
                               np.flatnonzero(changes == -1)):
            found.append((int(start), int(stop), sign,
                          float(np.sum(np.abs(statistic[start:stop]) - threshold))))
    return found


def cluster_test(contrasts: dict[str, tuple[np.ndarray, np.ndarray]],
                 time_ms: np.ndarray) -> dict:
    """Two-sided cluster mass, family-wise corrected over time and contrasts."""

    rng = np.random.default_rng(RANDOM_SEED)
    observed, nulls, thresholds = {}, {}, {}
    for key, (values, is_a) in contrasts.items():
        observed[key] = two_sample_t(values, is_a)
        n_units, n_a = values.shape[0], int(is_a.sum())
        null = np.empty((N_PERMUTATIONS, observed[key].size))
        for index in range(N_PERMUTATIONS):
            permuted = np.zeros(n_units, dtype=bool)
            permuted[rng.permutation(n_units)[:n_a]] = True
            null[index] = two_sample_t(values, permuted)
        nulls[key] = null
        thresholds[key] = float(np.quantile(np.abs(null), 1.0 - CLUSTER_ALPHA))

    null_max = np.zeros(N_PERMUTATIONS)
    for key, null in nulls.items():
        for index in range(N_PERMUTATIONS):
            found = _clusters(null[index], thresholds[key])
            if found:
                null_max[index] = max(null_max[index],
                                      max(item[3] for item in found))

    result = {}
    for key in contrasts:
        significant = np.zeros(observed[key].size, dtype=bool)
        rows = []
        for start, stop, sign, mass in _clusters(observed[key], thresholds[key]):
            probability = (1.0 + np.count_nonzero(null_max >= mass)) / (
                N_PERMUTATIONS + 1.0)
            significant[start:stop] = probability < ALPHA
            rows.append({"start_ms": float(time_ms[start]),
                         "end_ms": float(time_ms[stop - 1]),
                         "sign": sign, "cluster_mass": mass,
                         "p_fwer": probability,
                         "significant": bool(probability < ALPHA)})
        result[key] = {"t": observed[key], "significant": significant,
                       "threshold": thresholds[key],
                       "clusters": sorted(rows, key=lambda r: -r["cluster_mass"])}
    return result


def run(result_dir: Path | None = None) -> dict:
    result_dir = Path(result_dir or RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in sorted(RUN_TO_RECORDING) if r.endswith(EXPERIMENT)]
    sessions = {run: load_session(run) for run in runs}
    day1, day2 = sessions[runs[0]], sessions[runs[1]]
    time_ms = day1.time_ms

    sequences = sorted(set(day1.name) | set(day2.name))
    contrasts, summary = {}, {}
    for sequence in sequences:
        deviant_session = day1 if day1.playback.deviant == sequence else day2
        standard_session = day2 if deviant_session is day1 else day1
        deviant = block_means(deviant_session, sequence)
        standard = block_means(standard_session, sequence)
        values = np.vstack([deviant, standard])
        is_a = np.r_[np.ones(deviant.shape[0], bool),
                     np.zeros(standard.shape[0], bool)]
        contrasts[sequence] = (values, is_a)
        summary[sequence] = {
            "deviant_run": deviant_session.run,
            "standard_run": standard_session.run,
            "n_deviant_sequences": int((deviant_session.name == sequence).sum()),
            "n_standard_sequences": int((standard_session.name == sequence).sum()),
            "n_blocks_each": int(deviant.shape[0]),
        }

    # Day-cancelling interaction, same blocks.
    first, second = sequences
    d1 = block_means(day1, first) - block_means(day1, second)
    d2 = block_means(day2, first) - block_means(day2, second)
    contrasts["interaction"] = (
        np.vstack([d1, d2]),
        np.r_[np.ones(d1.shape[0], bool), np.zeros(d2.shape[0], bool)])

    tested = cluster_test(contrasts, time_ms)

    arrays = {"time_ms": time_ms}
    for key, entry in tested.items():
        safe = key.replace("-", "_")
        arrays[f"{safe}_t"] = entry["t"]
        arrays[f"{safe}_significant"] = entry["significant"]
    for sequence in sequences:
        values, is_a = contrasts[sequence]
        safe = sequence.replace("-", "_")
        # Keep the per-block curves so the figure can draw dispersion over the
        # resampling unit rather than over sequences.
        arrays[f"{safe}_deviant_blocks"] = values[is_a]
        arrays[f"{safe}_standard_blocks"] = values[~is_a]
        arrays[f"{safe}_deviant_mean"] = values[is_a].mean(0)
        arrays[f"{safe}_standard_mean"] = values[~is_a].mean(0)
    np.savez_compressed(result_dir / "ab_ba_inference.npz", **arrays)

    report = {
        "experiment": EXPERIMENT,
        "runs": {run: {"date": s.playback.date,
                       "standard": s.playback.standard,
                       "deviant": s.playback.deviant}
                 for run, s in sessions.items()},
        "contrasts": summary,
        "assumption": (
            "The two recordings are treated as one session, at the "
            "experimenter's instruction. The 'interaction' contrast is "
            "reported alongside because it cancels any additive difference "
            "between them."
        ),
        "resampling_unit": "acquisition block (16 per recording, 25 sequences each)",
        "n_permutations": N_PERMUTATIONS,
        "results": {key: {"cluster_forming_abs_t": entry["threshold"],
                          "clusters": entry["clusters"][:6]}
                    for key, entry in tested.items()},
    }
    (result_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    return report


if __name__ == "__main__":
    output = run()
    for key, entry in output["results"].items():
        print(f"\n{key}  (|t| threshold {entry['cluster_forming_abs_t']:.2f})")
        for cluster in entry["clusters"][:4]:
            flag = "  ***" if cluster["significant"] else ""
            print(f"  {cluster['start_ms']:7.0f}-{cluster['end_ms']:<7.0f} ms  "
                  f"sign {cluster['sign']:+d}  mass {cluster['cluster_mass']:8.2f}  "
                  f"p_FWER {cluster['p_fwer']:.4f}{flag}")
