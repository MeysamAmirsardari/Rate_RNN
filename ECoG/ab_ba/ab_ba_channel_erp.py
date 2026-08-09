"""Leakage-controlled same-sequence ERP analysis for AB/BA Figure 4.

This module reconstructs the Experiment-1 analysis directly from the two raw
Open Ephys recordings and the recovered Baphy playback tables.  It mirrors
the relevant ``Gen_M2Mat.m`` contract:

* every observation carries six metadata fields before its neural samples;
* the stimulus mark is read from the playback table, never inferred from ECoG;
* the neural epoch is baseline-centred and divided by its baseline SD; and
* only the 180-ms + 180-ms, zero-within-pair-gap experiment is retained.

For each held-fixed physical sequence, alternating acquisition blocks form a
discovery half.  The strongest contact in the prespecified second-item window
(180--360 ms) is selected from that half only.  The complementary blocks are
used for the displayed mean/SEM and for inference.  Display traces receive the
same modest, symmetric 2-ms Gaussian smoothing used for Figure 2; inference
always uses the unsmoothed block means.

The inferential unit is the acquisition block.  The two-sided cluster-mass
test enumerates all C(16, 8) assignments of the eight held-out blocks from
each recording, preserves the counterbalanced AB/BA role reversal, and uses
one maximum-cluster family across both sequences and the complete 0--600-ms
clock.  As requested by the experimenter, the two recordings are provisionally
treated as one session.  The output provenance states that the resulting
inference remains conditional on this assumption and is not animal-population
evidence.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t

from ECoG.ab_ba.ab_ba_inference import EXPERIMENT, load_session
from ECoG.ab_ba.baphy_events import BAPHY_ROOT, RUN_TO_RECORDING
from ECoG.ab_ba.reconstruct_2026 import ARCHIVE


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results" / "ab_ba_channel_erp"
NPZ_PATH = RESULT_DIR / "ab_ba_channel_erp.npz"
PROVENANCE_PATH = RESULT_DIR / "provenance.json"
TIMECOURSE_CSV = RESULT_DIR / "timecourses.csv.gz"
CLUSTER_CSV = RESULT_DIR / "clusters.csv"
TAG_CSV = RESULT_DIR / "allm2_tags.csv.gz"

SEQUENCES = ("5300-9400", "9400-5300")
SEQUENCE_CODES = {"5300-9400": "AB", "9400-5300": "BA"}
STIMULUS_IDS = {"5300-9400": 1, "9400-5300": 2}
DISPLAY_SIGMA_MS = 2.0
SELECTION_WINDOW_MS = (180.0, 360.0)
DISPLAY_WINDOW_MS = (0.0, 600.0)
ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
SCHEMA_VERSION = "figure-4-ab-ba-independent-contact-erp-v1"
MATLAB_SOURCES = (
    Path("/Users/eminent/Projects/ECoG/AB_BA/scripts_AB_BA.m"),
    Path("/Users/eminent/Projects/ECoG/Gen_M2Mat.m"),
    Path("/Users/eminent/Projects/ECoG/generate_full_mat_info_v2.m"),
)


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(sequence: str) -> str:
    return sequence.replace("-", "_")


def _allm2_tags(session) -> np.ndarray:
    """Rebuild the six leading ``allM2`` columns in playback order.

    The output columns are exactly
    ``current run length, previous run length, repetition number, current
    stimulus ID, previous stimulus ID, acquisition block``.  No neural sample
    is included here, which prevents the six tags from ever being mistaken for
    time points downstream.
    """

    rows: list[list[int]] = []
    last_stimulus = 0
    last_length = 0
    for block in np.unique(session.block):
        names = session.name[session.block == block]
        ids = np.asarray([STIMULUS_IDS[str(name)] for name in names], dtype=int)
        starts = np.r_[0, np.flatnonzero(np.diff(ids) != 0) + 1]
        stops = np.r_[starts[1:], ids.size]
        lengths = stops - starts
        for run_index, (start, stop, length) in enumerate(
            zip(starts, stops, lengths)
        ):
            stimulus = int(ids[start])
            previous_stimulus = (
                last_stimulus if run_index == 0 else int(ids[starts[run_index - 1]])
            )
            previous_length = (
                last_length if run_index == 0 else int(lengths[run_index - 1])
            )
            for repetition in range(1, int(length) + 1):
                rows.append(
                    [
                        int(length),
                        previous_length,
                        repetition,
                        stimulus,
                        previous_stimulus,
                        int(block),
                    ]
                )
        last_stimulus = int(ids[-1])
        last_length = int(lengths[-1])
    tags = np.asarray(rows, dtype=np.int64)
    if tags.shape != (session.name.size, 6):
        raise AssertionError(
            f"{session.run}: reconstructed tags have shape {tags.shape}, "
            f"expected {(session.name.size, 6)}"
        )
    expected_stimulus = np.asarray(
        [STIMULUS_IDS[str(name)] for name in session.name], dtype=np.int64
    )
    if not np.array_equal(tags[:, 3], expected_stimulus):
        raise AssertionError(f"{session.run}: allM2 stimulus tags lost playback order")
    if not np.array_equal(tags[:, 5], session.block):
        raise AssertionError(f"{session.run}: allM2 acquisition-block tags disagree")
    return tags


def _matched_block_means(
    rare_session,
    regular_session,
    sequence: str,
    blocks: np.ndarray,
    time_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return trial-count- and position-matched block means.

    A regular block contains roughly six times as many target presentations as
    a rare block. Averaging every regular presentation would therefore give
    the regular block mean much lower measurement variance. For each rare
    occurrence, this function chooses the nearest still-unused occurrence of
    the same physical sequence in the corresponding regular block. Selection
    depends only on playback position, never on ECoG.
    """

    rare_means, regular_means, counts = [], [], []
    for block in blocks:
        rare_rows = np.flatnonzero(
            (rare_session.block == block) & (rare_session.name == sequence)
        )
        regular_rows = np.flatnonzero(
            (regular_session.block == block) & (regular_session.name == sequence)
        )
        if rare_rows.size == 0 or regular_rows.size < rare_rows.size:
            raise AssertionError(
                f"{sequence}, block {block}: cannot match "
                f"{rare_rows.size} rare to {regular_rows.size} regular trials"
            )
        rare_block_rows = np.flatnonzero(rare_session.block == block)
        regular_block_rows = np.flatnonzero(regular_session.block == block)
        rare_positions = np.searchsorted(rare_block_rows, rare_rows)
        regular_positions = np.searchsorted(regular_block_rows, regular_rows)
        available = list(range(regular_positions.size))
        chosen = []
        for position in rare_positions:
            nearest = min(
                available,
                key=lambda index: (abs(int(regular_positions[index]) - int(position)),
                                   int(regular_positions[index])),
            )
            chosen.append(nearest)
            available.remove(nearest)
        selected_regular_rows = regular_rows[np.asarray(chosen, dtype=int)]
        rare_means.append(rare_session.epochs[rare_rows].mean(axis=0)[:, time_mask])
        regular_means.append(
            regular_session.epochs[selected_regular_rows].mean(axis=0)[:, time_mask]
        )
        counts.append(rare_rows.size)
    return (
        np.stack(rare_means).astype(np.float64),
        np.stack(regular_means).astype(np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def _two_sample_t(values: np.ndarray, group_a: np.ndarray) -> np.ndarray:
    """Studentized two-sample statistic; groups are equal-sized by design."""

    a = values[group_a]
    b = values[~group_a]
    if a.shape[0] != b.shape[0]:
        raise AssertionError("cluster permutation requires equal block counts")
    variance_a = a.var(axis=0, ddof=1)
    variance_b = b.var(axis=0, ddof=1)
    standard_error = np.sqrt(variance_a / a.shape[0] + variance_b / b.shape[0])
    return np.divide(
        a.mean(axis=0) - b.mean(axis=0),
        standard_error,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=standard_error > 0,
    )


def _clusters(statistic: np.ndarray, threshold: float) -> list[dict]:
    clusters: list[dict] = []
    for sign in (1, -1):
        active = sign * statistic > threshold
        padded = np.r_[False, active, False].astype(np.int8)
        changes = np.diff(padded)
        for start, stop in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        ):
            clusters.append(
                {
                    "start": int(start),
                    "stop": int(stop),
                    "sign": int(sign),
                    "mass": float(np.sum(np.abs(statistic[start:stop]))),
                }
            )
    return clusters


def _write_tags(sessions: dict[str, object], tags: dict[str, np.ndarray]) -> None:
    with gzip.open(TAG_CSV, "wt", newline="") as stream:
        fields = (
            "run",
            "row",
            "current_run_length",
            "previous_run_length",
            "repetition_number",
            "current_stimulus_id",
            "previous_stimulus_id",
            "acquisition_block",
            "binary_deviant_mark",
            "physical_sequence",
            "context_role",
            "onset_within_block_s",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run, session in sessions.items():
            for row, (metadata, name, mark, onset) in enumerate(
                zip(tags[run], session.name, session.is_deviant, session.playback.onset_s)
            ):
                writer.writerow(
                    {
                        "run": run,
                        "row": row,
                        "current_run_length": int(metadata[0]),
                        "previous_run_length": int(metadata[1]),
                        "repetition_number": int(metadata[2]),
                        "current_stimulus_id": int(metadata[3]),
                        "previous_stimulus_id": int(metadata[4]),
                        "acquisition_block": int(metadata[5]),
                        "binary_deviant_mark": int(mark),
                        "physical_sequence": str(name),
                        "context_role": "rare" if mark else "regular",
                        "onset_within_block_s": f"{float(onset):.9g}",
                    }
                )


def _write_timecourses(arrays: dict[str, np.ndarray]) -> None:
    with gzip.open(TIMECOURSE_CSV, "wt", newline="") as stream:
        fields = (
            "sequence",
            "sequence_code",
            "contact_matlab",
            "time_ms",
            "regular_mean",
            "regular_sem",
            "rare_mean",
            "rare_sem",
            "unsmoothed_t",
            "cluster_p_fwer",
            "cluster_significant",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sequence in SEQUENCES:
            key = _safe(sequence)
            for index, time in enumerate(arrays["time_ms"]):
                writer.writerow(
                    {
                        "sequence": sequence,
                        "sequence_code": SEQUENCE_CODES[sequence],
                        "contact_matlab": int(arrays[f"{key}_channel_matlab"]),
                        "time_ms": f"{float(time):.9g}",
                        "regular_mean": f"{arrays[f'{key}_regular_mean'][index]:.12g}",
                        "regular_sem": f"{arrays[f'{key}_regular_sem'][index]:.12g}",
                        "rare_mean": f"{arrays[f'{key}_rare_mean'][index]:.12g}",
                        "rare_sem": f"{arrays[f'{key}_rare_sem'][index]:.12g}",
                        "unsmoothed_t": f"{arrays[f'{key}_t'][index]:.12g}",
                        "cluster_p_fwer": f"{arrays[f'{key}_p_fwer'][index]:.12g}",
                        "cluster_significant": int(
                            arrays[f"{key}_significant"][index]
                        ),
                    }
                )


def _write_clusters(rows: list[dict]) -> None:
    with CLUSTER_CSV.open("w", newline="") as stream:
        fields = (
            "sequence",
            "sequence_code",
            "contact_matlab",
            "start_ms",
            "end_ms",
            "direction",
            "cluster_mass",
            "p_fwer",
            "significant",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(result_dir: Path | None = None, *, force: bool = False) -> dict[str, np.ndarray]:
    global RESULT_DIR, NPZ_PATH, PROVENANCE_PATH, TIMECOURSE_CSV, CLUSTER_CSV, TAG_CSV
    if result_dir is not None:
        RESULT_DIR = Path(result_dir)
        NPZ_PATH = RESULT_DIR / "ab_ba_channel_erp.npz"
        PROVENANCE_PATH = RESULT_DIR / "provenance.json"
        TIMECOURSE_CSV = RESULT_DIR / "timecourses.csv.gz"
        CLUSTER_CSV = RESULT_DIR / "clusters.csv"
        TAG_CSV = RESULT_DIR / "allm2_tags.csv.gz"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if NPZ_PATH.exists() and PROVENANCE_PATH.exists() and not force:
        with np.load(NPZ_PATH, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}

    runs = [run for run in sorted(RUN_TO_RECORDING) if run.endswith(EXPERIMENT)]
    if len(runs) != 2:
        raise AssertionError(f"Experiment 1 requires two runs, found {runs}")
    sessions = {run: load_session(run) for run in runs}
    day1, day2 = (sessions[run] for run in runs)
    if day1.playback.deviant_pct != 15 or day2.playback.deviant_pct != 15:
        raise AssertionError("both probability-reversal runs must use 85/15")
    if day1.playback.note_gap_ms != 0 or day2.playback.note_gap_ms != 0:
        raise AssertionError("Figure 4 ECoG must use only the zero-gap experiment")
    if day1.playback.note_ms != 180 or day2.playback.note_ms != 180:
        raise AssertionError("Figure 4 ECoG requires 180-ms tones")
    if not np.array_equal(day1.time_ms, day2.time_ms):
        raise AssertionError("recordings have different neural clocks")

    tags = {run: _allm2_tags(session) for run, session in sessions.items()}
    _write_tags(sessions, tags)

    time_mask = (day1.time_ms >= DISPLAY_WINDOW_MS[0]) & (
        day1.time_ms < DISPLAY_WINDOW_MS[1]
    )
    time_ms = day1.time_ms[time_mask]
    blocks = np.unique(day1.block)
    if not np.array_equal(blocks, np.arange(1, 17)):
        raise AssertionError(f"expected acquisition blocks 1..16, found {blocks}")
    discovery_blocks = blocks[::2]
    inference_blocks = blocks[1::2]
    selection_mask = (time_ms >= SELECTION_WINDOW_MS[0]) & (
        time_ms < SELECTION_WINDOW_MS[1]
    )
    sample_step_ms = float(np.median(np.diff(time_ms)))
    sigma_samples = DISPLAY_SIGMA_MS / sample_step_ms

    arrays: dict[str, np.ndarray] = {
        "time_ms": time_ms,
        "discovery_blocks": discovery_blocks,
        "inference_blocks": inference_blocks,
        "planned_regular_probability": np.asarray(0.85),
        "planned_rare_probability": np.asarray(0.15),
    }
    # These retain both physical identities in both sessions, in fixed
    # day-1/day-2 row order.  A single shared label permutation therefore
    # preserves the reversal mapping and their within-block dependence.
    inference_values: dict[str, np.ndarray] = {}
    observed_rare_masks: dict[str, np.ndarray] = {}

    for sequence in SEQUENCES:
        key = _safe(sequence)
        rare_session = day1 if day1.playback.deviant == sequence else day2
        regular_session = day2 if rare_session is day1 else day1
        rare_discovery, regular_discovery, discovery_counts = _matched_block_means(
            rare_session,
            regular_session,
            sequence,
            discovery_blocks,
            time_mask,
        )
        discovery_difference = rare_discovery.mean(axis=0) - regular_discovery.mean(
            axis=0
        )
        selection_score = np.sqrt(
            np.mean(discovery_difference[:, selection_mask] ** 2, axis=1)
        )
        channel_index = int(np.argmax(selection_score))

        rare_inference, regular_inference, inference_counts = _matched_block_means(
            rare_session,
            regular_session,
            sequence,
            inference_blocks,
            time_mask,
        )
        rare_inference = rare_inference[:, channel_index]
        regular_inference = regular_inference[:, channel_index]
        if rare_session is day1:
            day1_blocks, day2_blocks = rare_inference, regular_inference
        else:
            day1_blocks, day2_blocks = regular_inference, rare_inference
        values = np.vstack([day1_blocks, day2_blocks])
        day1_mask = np.r_[np.ones(8, dtype=bool), np.zeros(8, dtype=bool)]
        rare_mask = day1_mask if rare_session is day1 else ~day1_mask
        rare_raw, regular_raw = values[rare_mask], values[~rare_mask]
        rare_display = gaussian_filter1d(
            rare_raw,
            sigma=sigma_samples,
            axis=-1,
            mode="reflect",
            truncate=4.0,
        )
        regular_display = gaussian_filter1d(
            regular_raw,
            sigma=sigma_samples,
            axis=-1,
            mode="reflect",
            truncate=4.0,
        )

        arrays[f"{key}_channel_matlab"] = np.asarray(channel_index + 1)
        arrays[f"{key}_selection_score"] = selection_score
        arrays[f"{key}_discovery_trials_per_block"] = discovery_counts
        arrays[f"{key}_inference_trials_per_block"] = inference_counts
        arrays[f"{key}_rare_blocks_raw"] = rare_raw
        arrays[f"{key}_regular_blocks_raw"] = regular_raw
        arrays[f"{key}_rare_mean"] = rare_display.mean(axis=0)
        arrays[f"{key}_rare_sem"] = rare_display.std(axis=0, ddof=1) / np.sqrt(8)
        arrays[f"{key}_regular_mean"] = regular_display.mean(axis=0)
        arrays[f"{key}_regular_sem"] = regular_display.std(axis=0, ddof=1) / np.sqrt(
            8
        )
        arrays[f"{key}_t"] = _two_sample_t(values, rare_mask)
        inference_values[sequence] = values
        observed_rare_masks[sequence] = rare_mask

    threshold = float(
        student_t.ppf(1.0 - CLUSTER_FORMING_ALPHA / 2.0, 2 * 8 - 2)
    )
    assignments = list(itertools.combinations(range(16), 8))
    null_maximum = np.zeros(len(assignments), dtype=np.float64)
    for permutation_index, chosen in enumerate(assignments):
        pseudo_day1 = np.zeros(16, dtype=bool)
        pseudo_day1[list(chosen)] = True
        maximum = 0.0
        for sequence in SEQUENCES:
            pseudo_rare = (
                pseudo_day1
                if np.array_equal(
                    observed_rare_masks[sequence],
                    np.r_[np.ones(8, dtype=bool), np.zeros(8, dtype=bool)],
                )
                else ~pseudo_day1
            )
            statistic = _two_sample_t(inference_values[sequence], pseudo_rare)
            clusters = _clusters(statistic, threshold)
            if clusters:
                maximum = max(maximum, max(cluster["mass"] for cluster in clusters))
        null_maximum[permutation_index] = maximum

    cluster_rows: list[dict] = []
    for sequence in SEQUENCES:
        key = _safe(sequence)
        significant = np.zeros(time_ms.size, dtype=bool)
        p_fwer = np.ones(time_ms.size, dtype=np.float64)
        for cluster in _clusters(arrays[f"{key}_t"], threshold):
            probability = float(np.mean(null_maximum >= cluster["mass"]))
            start, stop = cluster["start"], cluster["stop"]
            p_fwer[start:stop] = probability
            significant[start:stop] = probability < ALPHA
            cluster_rows.append(
                {
                    "sequence": sequence,
                    "sequence_code": SEQUENCE_CODES[sequence],
                    "contact_matlab": int(arrays[f"{key}_channel_matlab"]),
                    "start_ms": float(time_ms[start]),
                    "end_ms": float(time_ms[stop - 1]),
                    "direction": (
                        "rare_above_regular"
                        if cluster["sign"] > 0
                        else "regular_above_rare"
                    ),
                    "cluster_mass": cluster["mass"],
                    "p_fwer": probability,
                    "significant": bool(probability < ALPHA),
                }
            )
        arrays[f"{key}_significant"] = significant
        arrays[f"{key}_p_fwer"] = p_fwer
        arrays[f"{key}_minimum_cluster_p_fwer"] = np.asarray(
            min(
                (
                    row["p_fwer"]
                    for row in cluster_rows
                    if row["sequence"] == sequence
                ),
                default=1.0,
            )
        )
    arrays["cluster_forming_abs_t"] = np.asarray(threshold)
    arrays["n_exact_assignments"] = np.asarray(len(assignments), dtype=np.int64)

    np.savez_compressed(NPZ_PATH, **arrays)
    _write_timecourses(arrays)
    _write_clusters(cluster_rows)

    baphy_files = {
        run: BAPHY_ROOT / run[:-3] / "tmp" / f"{run}_p_SEQ.mat" for run in runs
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "matlab_sources": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in MATLAB_SOURCES
        },
        "question": (
            "For held-fixed AB and held-fixed BA, does the independently "
            "selected ECoG contact differ when that same sequence is rare "
            "versus regular?"
        ),
        "experiment": (
            "Experiment 1 only: two 180-ms tones, zero within-pair gap, "
            "1.5-s between-sequence gap"
        ),
        "probabilities": {
            "planned_regular_each_run": 0.85,
            "planned_rare_each_run": 0.15,
            "baphy_deviant_pct": {
                run: float(sessions[run].playback.deviant_pct) for run in runs
            },
        },
        "data_contract": (
            "The recovered playback table provides a binary deviant mark and "
            "the six allM2 metadata columns. Neural time begins only after "
            "those six tags; no tag is plotted or analysed as a sample."
        ),
        "preprocessing": (
            "1-250 Hz third-order zero-phase Butterworth filter; -100 to 0 ms "
            "baseline subtraction and division by baseline SD, matching the "
            "Gen_M2Mat raw-trace path; no outlier removal"
        ),
        "selection": (
            "Odd-numbered acquisition blocks are discovery data. Separately "
            "for each physical sequence, the contact maximizing the RMS "
            "rare-minus-regular discovery effect from 180 to <360 ms is fixed "
            "before held-out estimation."
        ),
        "estimation": (
            "Even-numbered blocks only. Within each block, every rare target "
            "presentation is matched to the nearest-in-position unused regular "
            "presentation of the same physical sequence, so role means use "
            "equal trial counts without consulting ECoG. One mean per block "
            "and role; mean and SEM over eight held-out blocks per role."
        ),
        "display_smoothing": (
            "Symmetric zero-phase Gaussian, sigma 2 ms (FWHM approximately "
            "4.7 ms), applied to held-out block traces for display only."
        ),
        "inference": (
            "Two-sided two-sample t cluster mass on unsmoothed held-out block "
            "means; p<0.05 two-sided t cluster-forming threshold; exact "
            "enumeration of all C(16,8)=12,870 session-label assignments; "
            "shared permutations preserve the counterbalanced AB/BA reversal; "
            "maximum cluster jointly over AB, BA and every 0-600-ms sample."
        ),
        "scope": (
            "At the experimenter's instruction, the two acquisition days are "
            "provisionally treated as one session. Contextual role is still "
            "implemented between recordings, so inference is conditional on "
            "this assumption and this one animal; it is not animal-population "
            "or independently replicated day-level evidence."
        ),
        "runs": {
            run: {
                "date": sessions[run].playback.date,
                "standard": sessions[run].playback.standard,
                "rare": sessions[run].playback.deviant,
                "planned_regular_probability": 0.85,
                "planned_rare_probability": 0.15,
                "observed_regular_count": int((~sessions[run].is_deviant).sum()),
                "observed_rare_count": int(sessions[run].is_deviant.sum()),
                "n_acquisition_blocks": int(np.unique(sessions[run].block).size),
                "playback_path": str(baphy_files[run]),
                "playback_sha256": _sha256(baphy_files[run]),
                "raw_recording_path": str(
                    (ARCHIVE / RUN_TO_RECORDING[run]).resolve()
                ),
            }
            for run in runs
        },
        "selected_contacts_matlab": {
            SEQUENCE_CODES[sequence]: int(
                arrays[f"{_safe(sequence)}_channel_matlab"]
            )
            for sequence in SEQUENCES
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (NPZ_PATH, TIMECOURSE_CSV, CLUSTER_CSV, TAG_CSV)
        },
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n")
    return arrays


if __name__ == "__main__":
    output = run(force=True)
    for sequence in SEQUENCES:
        key = _safe(sequence)
        print(
            f"{SEQUENCE_CODES[sequence]}: contact "
            f"{int(output[f'{key}_channel_matlab'])}, "
            f"minimum cluster P_FWER "
            f"{float(output[f'{key}_minimum_cluster_p_fwer']):.6g}"
        )
