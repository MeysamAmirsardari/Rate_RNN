"""Faithful replication of ``scripts_AB_BA.m``.

This module reproduces the supplied MATLAB analysis and nothing else. No
significance testing, no leakage correction, no change of contrast: where the
source leaks, the leak is reproduced, because the point is to match the figure.

Mapping to the source
---------------------
``ft_oe_list(ss, 'SEQ', [1 250])``
    Reads the Open Ephys archive, band-passes 1-250 Hz and **resamples to
    1000 Hz** (every stored ``xx`` on this machine has ``outp.fs == 1000``
    against a 4000 Hz archive). Its ``outp.stimat`` playback table is recovered
    from the Baphy session folder by :mod:`ECoG.ab_ba.baphy_events`.

``Gen_M2Mat(xx, expnum, ch, 0)``
    Epochs each sequence from ``-basetime`` to ``seqDur + postTrial`` with
    ``basetime = 0.1 * fs = 100`` samples, then baseline-corrects each epoch by
    its own pre-onset mean and standard deviation. Rows are chronological, so
    ``allM2`` row order is playback order.

``scripts_AB_BA.m`` lines 76-98
    Class 0, "Rep 1 (Surprise)": the deviant rows of ``allMatrices2`` -- the
    **second** day, 2026-05-01.
    Class 1, "Rep 15 (Adapted)": the row *following* each deviant of
    ``allMatrices1`` -- the **first** day, 2026-04-30.
    ``erp_win`` keeps ``seqDur + 1000`` samples from stimulus onset, which for
    Experiment 1 is 0-1360 ms and matches the published x-axis.

``fitclinear(..., 'ridge', 'Lambda', 1e-2)``
    MATLAB minimises ``mean(loss) + Lambda/2 * ||w||^2``; scikit-learn
    minimises ``C * sum(loss) + 0.5 * ||w||^2``. They agree at
    ``C = 1 / (n_train * Lambda)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ECoG.ab_ba.baphy_events import RUN_TO_RECORDING, read_playback
from ECoG.ab_ba.reconstruct_2026 import (
    ARCHIVE, EVENT_RECORD, HEADER_BYTES, N_CHANNELS, _continuous,
    _header, channel_path,
)

RESULT_DIR = Path(__file__).resolve().parent / "results" / "matlab_replication"

#: ft_oe_list(ss, runclass, [1 250]) and its 1 kHz output rate.
BANDPASS_HZ = (1.0, 250.0)
TARGET_FS = 1000.0
#: Gen_M2Mat: baseline = 0.1 s.
BASETIME_MS = 100
#: scripts_AB_BA.m: win_length = seqDur + 1000, devPos = 1.
EXTRA_MS = 1000
DEV_POS = 1
RIDGE_LAMBDA = 1e-2
N_FOLDS = 5
SMOOTHING = 20
PEAK_HALF_WINDOW = 25
NOISE_SCALE = 1e-6
RANDOM_SEED = 0

#: scripts_AB_BA.m lines 314-324.
GRID_MAP = np.array([
    [4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9], [16, 15, 14, 13],
    [17, 18, 19, 20], [24, 23, 22, 21], [28, 27, 26, 25], [32, 31, 30, 29],
])

#: allMatrices1 is xx1 = the first day; allMatrices2 is xx2 = the second.
#: scripts_AB_BA.m line 77 notes "SWAP allMatrices1 and allMatrices2 for the
#: other condition", and both were run: the saved figures are named after the
#: deviant sequence of whichever recording supplied class 0.
RUN_DAY1 = "nmg038a01"
RUN_DAY2 = "nmg039a01"
#: True reproduces ``Nugmeg_2026-04-30_new_exp1_5300-9400.fig`` -- class 0 is
#: the deviant of the FIRST day, which is the saved figure. False reproduces
#: ``..._9400-5300.fig``, the literal reading of the script.
SWAP = True


def load_epochs(run: str) -> dict:
    """``allM2`` for one recording: epochs plus the six tag columns we need."""

    playback = read_playback(run)
    node = ARCHIVE / RUN_TO_RECORDING[run] / "Record Node 101"
    source_fs = float(_header(channel_path(node, 1))["sampleRate"])
    decimation = int(round(source_fs / TARGET_FS))
    if not np.isclose(source_fs / decimation, TARGET_FS):
        raise ValueError(f"{run}: cannot resample {source_fs} Hz to {TARGET_FS} Hz")

    events = np.frombuffer(
        (node / "all_channels.events").read_bytes()[HEADER_BYTES:],
        dtype=EVENT_RECORD)
    rises = events[events["event_id"] == 1]["timestamp"]
    falls = events[events["event_id"] == 0]["timestamp"]
    durations_ms = (falls - rises) * 1000.0 / source_fs
    block_ms = playback.sequences_per_trial * playback.soa_ms
    starts = rises[durations_ms > 0.5 * block_ms]
    if starts.size != playback.trial.max():
        raise ValueError(f"{run}: {starts.size} blocks, "
                         f"{playback.trial.max()} Baphy trials")

    first = int(np.frombuffer(
        channel_path(node, 1).read_bytes()[HEADER_BYTES:HEADER_BYTES + 8],
        dtype="<i8")[0])
    onset_source = (starts[playback.trial - 1]
                    + np.rint(playback.onset_s * source_fs).astype(np.int64)
                    - first)
    onset_ms = np.rint(onset_source * 1000.0 / source_fs).astype(np.int64)

    sequence_ms = int(round(2 * playback.note_ms + playback.note_gap_ms))
    window = BASETIME_MS + sequence_ms + EXTRA_MS
    low, high = BANDPASS_HZ
    coefficients = butter(3, [low / (source_fs / 2), high / (source_fs / 2)],
                          btype="band")

    epochs = np.empty((onset_ms.size, N_CHANNELS, window), dtype=np.float64)
    for channel in range(N_CHANNELS):
        trace = filtfilt(*coefficients, _continuous(channel_path(node, channel + 1)))
        trace = resample_poly(trace, 1, decimation)
        for row, onset in enumerate(onset_ms):
            start = int(onset) - BASETIME_MS
            epochs[row, channel] = trace[start:start + window]

    baseline = epochs[:, :, :BASETIME_MS]
    epochs = (epochs - baseline.mean(axis=2, keepdims=True)) / baseline.std(
        axis=2, ddof=1, keepdims=True)

    return {
        "run": run, "epochs": epochs, "name": playback.name,
        "deviant": playback.deviant, "sequence_ms": sequence_ms,
        "note_ms": playback.note_ms, "note_gap_ms": playback.note_gap_ms,
    }


def build_classes(day1: dict, day2: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """scripts_AB_BA.m lines 76-107, including the deviant+1 standard."""

    sequence_ms = day1["sequence_ms"]
    window = slice(BASETIME_MS, BASETIME_MS + sequence_ms + EXTRA_MS)

    deviant_rows = np.flatnonzero(day2["name"] == day2["deviant"])
    class_0 = day2["epochs"][deviant_rows][:, :, window]

    following = np.flatnonzero(day1["name"] == day1["deviant"]) + 1
    following = following[following < day1["epochs"].shape[0]]
    class_1 = day1["epochs"][following][:, :, window]

    keep = min(class_0.shape[0], class_1.shape[0])
    return class_0[:keep], class_1[:keep], keep


def _movmean(values: np.ndarray, width: int) -> np.ndarray:
    """MATLAB ``movmean``: centred, with shrinking windows at the ends."""

    half_low, half_high = width // 2, width - width // 2 - 1
    out = np.empty_like(values, dtype=float)
    for index in range(values.size):
        lo = max(0, index - half_low)
        hi = min(values.size, index + half_high + 1)
        out[index] = values[lo:hi].mean()
    return out


def decode(class_0: np.ndarray, class_1: np.ndarray) -> dict:
    """Time-resolved 5-fold accuracy and the activation pattern A = cov(X) w."""

    full = np.concatenate([class_0, class_1], axis=0)       # trial x ch x time
    labels = np.r_[np.zeros(class_0.shape[0], int), np.ones(class_1.shape[0], int)]
    n_time = full.shape[2]
    rng = np.random.default_rng(RANDOM_SEED)

    accuracy = np.empty(n_time)
    patterns = np.empty((N_CHANNELS, n_time))
    for index in range(n_time):
        features = full[:, :, index].copy()
        features += rng.standard_normal(features.shape) * NOISE_SCALE
        features = (features - features.mean(axis=0)) / features.std(axis=0, ddof=1)

        correct = 0
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
        for train, test in splitter.split(features, labels):
            model = LogisticRegression(
                C=1.0 / (train.size * RIDGE_LAMBDA), penalty="l2",
                solver="lbfgs", max_iter=1000)
            model.fit(features[train], labels[train])
            correct += int((model.predict(features[test]) == labels[test]).sum())
        accuracy[index] = correct / labels.size

        model = LogisticRegression(
            C=1.0 / (labels.size * RIDGE_LAMBDA), penalty="l2",
            solver="lbfgs", max_iter=1000)
        model.fit(features, labels)
        patterns[:, index] = np.cov(features, rowvar=False) @ model.coef_.ravel()

    smoothed = _movmean(accuracy, SMOOTHING)
    peak_index = int(np.argmax(smoothed))
    lo = max(0, peak_index - PEAK_HALF_WINDOW)
    hi = min(n_time, peak_index + PEAK_HALF_WINDOW + 1)
    peak_pattern = np.abs(patterns[:, lo:hi]).mean(axis=1)
    top = np.argsort(peak_pattern)[::-1][:3] + 1

    return {
        "accuracy": accuracy, "smoothed": smoothed, "patterns": patterns,
        "peak_index": peak_index, "peak_accuracy": float(smoothed[peak_index]),
        "peak_pattern": peak_pattern, "top_channels": top,
        "decode_window": (lo, hi),
        "erp_0": class_0.mean(axis=0), "erp_1": class_1.mean(axis=0),
    }


#: MATLAB figure colours, taken from scripts_AB_BA.m lines 173-177 and 228.
DEV_COLOR = (0.85, 0.95, 0.85)
STD_COLOR = (0.95, 0.95, 0.95)
PEAK_COLOR = (1.00, 0.85, 0.40)
REP1_COLOR = (0.8500, 0.3250, 0.0980)
REP15_COLOR = (0.0, 0.4470, 0.7410)


def _tone_patches(ax, note_ms, gap_ms, n_notes=2):
    for note in range(n_notes):
        start = (note_ms + gap_ms) * note
        ax.axvspan(start, start + note_ms,
                   color=DEV_COLOR if note == DEV_POS - 1 else STD_COLOR,
                   lw=0, zorder=0)


def plot(result: dict, day1: dict, result_dir: Path) -> Path:
    """Redraw the six panels of scripts_AB_BA.m."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    note_ms, gap_ms = day1["note_ms"], day1["note_gap_ms"]
    time = np.arange(1, result["accuracy"].size + 1)
    lo, hi = result["decode_window"]
    peak_ms = int(time[result["peak_index"]])
    top = np.asarray(result["top_channels"])

    fig = plt.figure(figsize=(18, 9), facecolor="w")
    grid = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0:2])
    _tone_patches(ax, note_ms, gap_ms)
    ax.axvspan(time[lo], time[hi - 1], color=PEAK_COLOR, alpha=0.6, lw=0)
    ax.plot(time, result["smoothed"], "b-", lw=2.5, label="5-Fold CV Accuracy")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(time[0], 0.505, "Chance Level", fontsize=9, va="bottom")
    ax.set_xlim(0, time[-1]); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Time (ms)", fontweight="bold")
    ax.set_ylabel("Decoding Accuracy", fontweight="bold")
    ax.set_title("Time-Resolved Decoding (Rep 1 vs Rep 15)", fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[0, 2:4])
    colours = [(0.55, 0.55, 0.55)] * N_CHANNELS
    for channel in top:
        colours[int(channel) - 1] = (0.85, 0.1, 0.1)
    ax.bar(np.arange(1, N_CHANNELS + 1), result["peak_pattern"], color=colours,
           width=0.75)
    ax.set_xlim(0, 33)
    ax.set_xlabel("Channel Number", fontweight="bold")
    ax.set_ylabel("Activation Weight (A)", fontweight="bold")
    ax.set_title(f"Decoder Spatial Pattern @ {peak_ms} ms", fontweight="bold")
    ax.grid(alpha=0.3)

    for rank, channel in enumerate(top):
        index = int(channel) - 1
        ax = fig.add_subplot(grid[1, rank])
        limit = np.abs(result["erp_0"][index]).max() * 1.3
        _tone_patches(ax, note_ms, gap_ms)
        ax.axvspan(time[lo], time[hi - 1], color=PEAK_COLOR, alpha=0.6, lw=0)
        ax.plot(time, result["erp_1"][index], color=REP15_COLOR, lw=2,
                label="Rep 15 (Adapted)")
        ax.plot(time, result["erp_0"][index], color=REP1_COLOR, lw=2,
                label="Rep 1 (Surprise)")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlim(0, time[-1]); ax.set_ylim(-limit, limit)
        ax.set_xlabel("Time (ms)", fontweight="bold")
        ax.set_ylabel("Amplitude", fontweight="bold")
        ax.set_title(f"Ch {channel} (Rank {rank + 1})", fontweight="bold")
        if rank == 2:
            ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(grid[2, rank])
        difference = result["erp_0"][index] - result["erp_1"][index]
        limit = np.abs(difference).max() * 1.3
        _tone_patches(ax, note_ms, gap_ms)
        ax.axvspan(time[lo], time[hi - 1], color=PEAK_COLOR, alpha=0.6, lw=0)
        ax.plot(time, difference, color="k", lw=2)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlim(0, time[-1]); ax.set_ylim(-limit, limit)
        ax.set_xlabel("Time (ms)", fontweight="bold")
        ax.set_ylabel("Amplitude", fontweight="bold")
        ax.set_title(f"Ch {channel} (Rank {rank + 1})", fontweight="bold")
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(grid[1:3, 3])
    cmap = LinearSegmentedColormap.from_list(
        "matlab_red", [(1, 1, 1), (0.7, 0.05, 0.05)], N=256)
    maximum = float(result["peak_pattern"].max()) or 1.0
    norm = Normalize(0, maximum)
    ax.fill([0.3, 4.7, 4.7, 0.3], [0.3, 0.3, 8.7, 8.7], color=(0.96,) * 3,
            edgecolor=(0.8,) * 3, lw=1.5)
    ax.plot([0.3, 4.7], [4.5, 4.5], "k--", lw=2)
    ax.text(-0.2, 2.5, "A1", color=(0.4,) * 3, fontsize=14, fontweight="bold",
            rotation=90, ha="center")
    ax.text(-0.2, 6.5, "PEG", color=(0.4,) * 3, fontsize=14, fontweight="bold",
            rotation=90, ha="center")
    for row in range(8):
        for column in range(4):
            channel = int(GRID_MAP[row, column])
            weight = result["peak_pattern"][channel - 1]
            fill = cmap(norm(weight))
            highlight = channel in set(int(c) for c in top)
            ax.scatter(column + 1, row + 1, s=550 if highlight else 300,
                       color=fill, edgecolor="k" if highlight else (0.6,) * 3,
                       linewidth=2.5 if highlight else 1, zorder=3)
            ax.text(column + 1, row + 1, str(channel), ha="center", va="center",
                    fontsize=12 if highlight else 9, zorder=4,
                    fontweight="bold" if highlight else "normal",
                    color="w" if norm(weight) > 0.6 else "k")
    ax.set_xlim(-0.6, 5.5); ax.set_ylim(0, 9)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("A1/PEG Spatial Topography", fontweight="bold")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    bar.set_label("Decoder Activation Weight", fontweight="bold")

    fig.suptitle(
        f"Decoder Analysis: Targeting the 1st Tone Deviant "
        f"(0-{int(note_ms)} ms)", fontsize=18, fontweight="bold")
    path = result_dir / "matlab_replication.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run(result_dir: Path | None = None) -> dict:
    result_dir = Path(result_dir or RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    first, second = (RUN_DAY2, RUN_DAY1) if SWAP else (RUN_DAY1, RUN_DAY2)
    day1, day2 = load_epochs(first), load_epochs(second)
    class_0, class_1, keep = build_classes(day1, day2)
    print(f"Extracted {2 * keep} balanced trials "
          f"(Class 0: Rep 1, Class 1: Rep 15).")

    result = decode(class_0, class_1)
    print(f"Decoder peaked at {result['peak_index']} ms with "
          f"{result['peak_accuracy'] * 100:.1f}% accuracy.")
    print(f"Top three channels: {result['top_channels'].tolist()}")

    np.savez_compressed(
        result_dir / "matlab_replication.npz",
        accuracy=result["accuracy"], smoothed=result["smoothed"],
        patterns=result["patterns"], peak_pattern=result["peak_pattern"],
        top_channels=result["top_channels"], erp_0=result["erp_0"],
        erp_1=result["erp_1"], peak_index=result["peak_index"],
        decode_window=np.asarray(result["decode_window"]),
        note_ms=day1["note_ms"], note_gap_ms=day1["note_gap_ms"],
        sequence_ms=day1["sequence_ms"], n_trials=2 * keep,
    )
    figure_path = plot(result, day1, result_dir)
    print(f"figure: {figure_path}")
    summary = {
        "source": "scripts_AB_BA.m, expnum = 1, devPos = 1",
        "swapped": SWAP,
        "class_0": f"deviant rows of {RUN_DAY2} (2026-05-01)",
        "class_1": f"row after each deviant of {RUN_DAY1} (2026-04-30)",
        "n_trials": 2 * keep,
        "peak_time_ms": result["peak_index"],
        "peak_accuracy": result["peak_accuracy"],
        "top_channels": result["top_channels"].tolist(),
        "fs_hz": TARGET_FS,
        "window_ms": [0, day1["sequence_ms"] + EXTRA_MS],
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
