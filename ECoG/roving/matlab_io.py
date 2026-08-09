"""Translate ``Gen_M2Mat[_sp]`` and ``generate_full_mat_info_v2``.

Only the SRH/SRV branch used by the three requested recordings is translated.
The implementation is vectorized across channels but retains MATLAB event
ordering, 1-based sample arithmetic, baseline correction, context filtering,
and per-context truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert

from .config import AnalysisSpec


@dataclass(frozen=True)
class RovingEpochs:
    """Balanced decoder inputs before the source scripts add tiny noise."""

    rep_first: np.ndarray  # channels x trials x time
    rep_late: np.ndarray  # channels x trials x time
    first_groups: np.ndarray
    late_groups: np.ndarray
    first_trials: np.ndarray
    late_trials: np.ndarray
    first_stimuli: np.ndarray
    late_stimuli: np.ndarray
    first_contexts: np.ndarray
    late_contexts: np.ndarray
    time_ms: np.ndarray
    metadata: Dict[str, object]


@dataclass(frozen=True)
class RovingRepetitionEpochs:
    """All 15 repetitions, aligned by roving block before model fitting."""

    epochs: np.ndarray  # blocks x repetitions x channels x time
    group_ids: np.ndarray
    trials: np.ndarray
    stimuli: np.ndarray
    contexts: np.ndarray
    repetitions: np.ndarray
    time_ms: np.ndarray
    metadata: Dict[str, object]


@dataclass(frozen=True)
class _SelectedEvent:
    rep: int
    stimulus: int
    context: int
    trial: int
    group: int
    epoch: np.ndarray  # channels x complete loader epoch


def _matlab_round_int(value: float, name: str) -> int:
    rounded = int(np.rint(value))
    if not np.isclose(value, rounded):
        raise ValueError(f"{name} must resolve to an integer sample; got {value}")
    return rounded


def _as_stim_list(value: object) -> List[str]:
    values = np.atleast_1d(value)
    return [str(item) for item in values.tolist()]


def _first_trial_rep_list(stimat: np.ndarray) -> np.ndarray:
    """Exact Python indexing equivalent of the MATLAB ``rep_list`` line."""

    ids = stimat[stimat[:, 0] == 1, 1]
    changes_after = np.flatnonzero(np.diff(ids) != 0) + 1
    if changes_after.size == 0:
        return np.array([1], dtype=int)
    return np.unique(np.diff(np.r_[0, changes_after])).astype(int)


def _loader_parameters(
    outp: dict, stimat: np.ndarray, stim_list: List[str]
) -> Dict[str, int]:
    fs = _matlab_round_int(float(outp["fs"]), "fs")
    seq_dur = _matlab_round_int(float(outp["dur"]) * fs, "seqDur")
    baseline = _matlab_round_int(0.1 * fs, "basetime")

    hyphen_count = stim_list[0].count("-")
    single_note = hyphen_count == 0
    note_num = 1 if single_note else hyphen_count

    rep_list = _first_trial_rep_list(stimat)
    first_cross_trial = np.flatnonzero(
        (stimat[:-1, 0] == 1) & (stimat[1:, 0] == 2)
    )
    if first_cross_trial.size != 1:
        raise ValueError(
            "MATLAB postTrial expression did not resolve to exactly one event"
        )
    trial_one_last_offset_ms = stimat[first_cross_trial[0], 3]
    post_trial = _matlab_round_int(
        outp["dataMat"].shape[0] - trial_one_last_offset_ms * fs / 1000.0,
        "postTrial",
    )

    if single_note or np.array_equal(rep_list, np.array([1])):
        soa = seq_dur + _matlab_round_int(float(outp["pre"]) * fs, "pre") + post_trial
    else:
        soa = _matlab_round_int(
            (stimat[1, 2] - stimat[0, 2]) * fs / 1000.0, "SOA"
        )

    return {
        "fs": fs,
        "seq_dur": seq_dur,
        "baseline": baseline,
        "note_num": note_num,
        "post_trial": post_trial,
        "soa": soa,
    }


def _cut_epoch(
    data: np.ndarray,
    trial_zero_based: int,
    onset_sample: int,
    rep_zero_based: int,
    params: Dict[str, int],
    cutting: int,
    hilbert_t: int,
) -> np.ndarray:
    """Cut one event using MATLAB's inclusive, 1-based expressions."""

    baseline = params["baseline"]
    soa = params["soa"]
    start = onset_sample + soa * rep_zero_based - baseline
    if cutting == 1:
        stop = (
            onset_sample
            + soa * rep_zero_based
            + params["seq_dur"]
            + params["post_trial"]
        )
    elif cutting == 3:
        stop = onset_sample + soa * (rep_zero_based + 1)
    else:
        raise ValueError(
            f"Only source-used cutting modes 1 and 3 are supported; got {cutting}"
        )

    if start < 0:
        raise IndexError(f"Epoch starts before recorded data: {start}")

    raw = data[start:min(stop, data.shape[0]), :, trial_zero_based].copy()
    if stop > data.shape[0]:
        raw = np.pad(raw, ((0, stop - data.shape[0]), (0, 0)))

    if hilbert_t == 1:
        raw = np.abs(hilbert(raw, axis=0))
    elif hilbert_t != 0:
        raise ValueError(f"hilbert_t must be 0 or 1; got {hilbert_t}")

    base = raw[:baseline]
    # MATLAB uses mean(...,'omitnan') followed by std(...), whose default
    # normalization is N-1 and whose default NaN behavior is propagation.
    raw -= np.nanmean(base, axis=0, keepdims=True)
    raw /= np.std(base, axis=0, ddof=1, keepdims=True)
    return raw.T


def _extract_selected_events(
    xx: dict,
    spec: AnalysisSpec,
    *,
    selected_reps: Iterable[int] | None = None,
    loader_cutting: int | None = None,
) -> Tuple[List[_SelectedEvent], Dict[str, object]]:
    if not str(xx["list_str"]).lower().startswith("sr"):
        raise ValueError(
            f"Expected SRH/SRV data, found list_str={xx['list_str']!r}"
        )
    if spec.expnum != 1:
        raise NotImplementedError(
            "The requested files contain one simplified outp struct; expnum != 1 "
            "would require preserving the original MATLAB cell array."
        )

    outp = xx["outp"]
    data = np.asarray(outp["dataMat"])
    stimat = np.asarray(outp["stimat"])
    stim_list = _as_stim_list(outp["stims"])
    if data.ndim != 3 or data.shape[1] != spec.n_channels:
        raise ValueError(
            f"Expected samples x {spec.n_channels} channels x trials; got {data.shape}"
        )
    if len(stim_list) != spec.n_stim:
        raise ValueError(
            f"{spec.key}: source nStim={spec.n_stim}, file contains "
            f"{len(stim_list)} stimuli"
        )

    params = _loader_parameters(outp, stimat, stim_list)
    selected_rep_set = (
        {spec.rep_first, spec.rep_late}
        if selected_reps is None
        else {int(rep) for rep in selected_reps}
    )
    cutting = spec.loader_cutting if loader_cutting is None else loader_cutting
    events: List[_SelectedEvent] = []
    last_stim_previous_trial = 0
    last_rep_length = 0
    group = 0

    for trial in range(1, data.shape[2] + 1):
        current = stimat[stimat[:, 0] == trial]
        if current.size == 0:
            continue
        block_starts = np.r_[0, np.flatnonzero(np.diff(current[:, 1]) != 0) + 1]
        stimuli = current[block_starts, 1].astype(int)
        onsets = np.rint(current[block_starts, 2] * params["fs"] / 1000.0).astype(int)
        contexts = np.r_[
            last_stim_previous_trial, current[block_starts[1:] - 1, 1]
        ].astype(int)
        lengths = np.r_[
            last_rep_length,
            np.diff(block_starts),
            len(current) - block_starts[-1],
        ].astype(int)

        last_stim_previous_trial = int(stimuli[-1])
        last_rep_length = int(lengths[-1])

        for block_index, onset in enumerate(onsets):
            current_length = int(lengths[block_index + 1])
            group += 1
            for rep in sorted(selected_rep_set):
                if rep < 1 or rep > current_length:
                    continue
                epoch = _cut_epoch(
                    data=data,
                    trial_zero_based=trial - 1,
                    onset_sample=int(onset),
                    rep_zero_based=rep - 1,
                    params=params,
                    cutting=cutting,
                    hilbert_t=spec.hilbert_t,
                )
                events.append(
                    _SelectedEvent(
                        rep=rep,
                        stimulus=int(stimuli[block_index]),
                        context=int(contexts[block_index]),
                        trial=trial,
                        group=group,
                        epoch=epoch,
                    )
                )

    metadata: Dict[str, object] = {
        "list_str": str(xx["list_str"]),
        "experiment": str(xx["experiment"]),
        "fs_hz": params["fs"],
        "baseline_samples": params["baseline"],
        "seq_duration_samples": params["seq_dur"],
        "soa_samples": params["soa"],
        "post_trial_samples": params["post_trial"],
        "stimuli": stim_list,
        "loader_cutting": cutting,
        "n_selected_events_before_context_filter": len(events),
    }
    return events, metadata


def _context_order(n_stim: int) -> Iterable[Tuple[int, int]]:
    """MATLAB order: curr=1:nStim; ctx=setdiff(1:nStim,curr)."""

    for current in range(1, n_stim + 1):
        for context in range(1, n_stim + 1):
            if context != current:
                yield current, context


def _stack_contexts(
    events: List[_SelectedEvent], spec: AnalysisSpec
) -> Tuple[List[_SelectedEvent], List[_SelectedEvent], List[dict]]:
    first_all: List[_SelectedEvent] = []
    late_all: List[_SelectedEvent] = []
    context_counts: List[dict] = []
    for current, context in _context_order(spec.n_stim):
        first = [
            event
            for event in events
            if event.rep == spec.rep_first
            and event.stimulus == current
            and event.context == context
        ]
        late = [
            event
            for event in events
            if event.rep == spec.rep_late
            and event.stimulus == current
            and event.context == context
        ]
        n_trials = min(len(first), len(late))
        context_counts.append(
            {
                "current_stimulus": current,
                "previous_stimulus": context,
                "n_first": len(first),
                "n_late": len(late),
                "n_kept": n_trials,
            }
        )
        first_all.extend(first[:n_trials])
        late_all.extend(late[:n_trials])
    return first_all, late_all, context_counts


def _event_field(events: List[_SelectedEvent], field: str) -> np.ndarray:
    return np.asarray([getattr(event, field) for event in events], dtype=int)


def extract_roving_epochs(
    data_path: Path, spec: AnalysisSpec, *, balance_seed: int | None = None
) -> RovingEpochs:
    """Load and reproduce source extraction through the balancing step."""

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Required source recording is missing: {data_path}\n"
            "Pass --data-file only if you have verified the intended recording; "
            "no similarly named recording is substituted automatically."
        )

    loaded = loadmat(data_path, variable_names=["xx"], simplify_cells=True)
    if "xx" not in loaded or not isinstance(loaded["xx"], dict):
        raise ValueError(f"{data_path} does not contain the expected struct variable 'xx'")
    events, metadata = _extract_selected_events(loaded["xx"], spec)
    first, late, context_counts = _stack_contexts(events, spec)
    if not first or not late:
        raise ValueError(
            f"No matched context trials found for repetitions "
            f"{spec.rep_first} and {spec.rep_late}"
        )

    n_keep = min(len(first), len(late))
    rng = np.random.RandomState(
        spec.random_seed if balance_seed is None else balance_seed
    )
    first_idx = rng.permutation(len(first))[:n_keep]
    late_idx = rng.permutation(len(late))[:n_keep]
    first = [first[index] for index in first_idx]
    late = [late[index] for index in late_idx]

    # MATLAB erp_win=101:901, inclusive.  Python's stop is exclusive.
    start = spec.epoch_start_sample_matlab - 1
    stop = spec.epoch_end_sample_matlab
    first_array = np.stack([event.epoch[:, start:stop] for event in first], axis=1)
    late_array = np.stack([event.epoch[:, start:stop] for event in late], axis=1)
    expected_time = stop - start
    if first_array.shape != (spec.n_channels, n_keep, expected_time):
        raise AssertionError(f"Unexpected first-repetition shape {first_array.shape}")
    if late_array.shape != first_array.shape:
        raise AssertionError(
            f"Class shape mismatch: first={first_array.shape}, late={late_array.shape}"
        )

    metadata.update(
        {
            "source_data_file": str(data_path.resolve()),
            "n_first_before_global_balance": len(first),
            "n_late_before_global_balance": len(late),
            "n_trials_per_class": n_keep,
            "n_labels_total": 2 * n_keep,
            "context_counts": context_counts,
            "source_rep_first": spec.rep_first,
            "source_rep_late": spec.rep_late,
            "source_late_display_label": "Rep 15",
            "source_late_label_numeric_mismatch": spec.rep_late != 15,
        }
    )
    return RovingEpochs(
        rep_first=first_array,
        rep_late=late_array,
        first_groups=_event_field(first, "group"),
        late_groups=_event_field(late, "group"),
        first_trials=_event_field(first, "trial"),
        late_trials=_event_field(late, "trial"),
        first_stimuli=_event_field(first, "stimulus"),
        late_stimuli=_event_field(late, "stimulus"),
        first_contexts=_event_field(first, "context"),
        late_contexts=_event_field(late, "context"),
        time_ms=np.arange(expected_time, dtype=int),
        metadata=metadata,
    )


def extract_repetition_epochs(
    data_path: Path,
    spec: AnalysisSpec,
    *,
    num_repetitions: int = 15,
) -> RovingRepetitionEpochs:
    """Reproduce ``SVM_rep_map`` extraction for repetitions 1 through 15.

    The source script uses ``Gen_M2Mat`` (``cutting=1``), then keeps MATLAB
    samples 101:901 and concatenates the six valid context/current pairs in
    their nested-loop order.  The three requested recordings have one
    observation for every repetition in every retained block, so the source
    balancing step changes order only; this function verifies that invariant
    instead of silently manufacturing block correspondence.
    """

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Required source recording is missing: {data_path}")
    loaded = loadmat(data_path, variable_names=["xx"], simplify_cells=True)
    if "xx" not in loaded or not isinstance(loaded["xx"], dict):
        raise ValueError(f"{data_path} does not contain the expected struct variable 'xx'")

    repetitions = np.arange(1, num_repetitions + 1, dtype=int)
    events, metadata = _extract_selected_events(
        loaded["xx"],
        spec,
        selected_reps=repetitions,
        loader_cutting=1,
    )

    ordered_by_rep: List[List[_SelectedEvent]] = [[] for _ in repetitions]
    context_counts: List[dict] = []
    for current, context in _context_order(spec.n_stim):
        counts = []
        for rep_index, rep in enumerate(repetitions):
            selected = [
                event
                for event in events
                if event.rep == rep
                and event.stimulus == current
                and event.context == context
            ]
            ordered_by_rep[rep_index].extend(selected)
            counts.append(len(selected))
        context_counts.append(
            {
                "current_stimulus": current,
                "previous_stimulus": context,
                "counts_by_repetition": counts,
            }
        )

    counts_by_rep = np.asarray([len(events_rep) for events_rep in ordered_by_rep])
    if np.any(counts_by_rep == 0):
        raise ValueError(
            f"At least one repetition has no valid trials: {counts_by_rep.tolist()}"
        )
    global_min_trials = int(np.min(counts_by_rep))
    if not np.all(counts_by_rep == global_min_trials):
        raise ValueError(
            "The requested recordings were expected to have equal counts at all "
            "15 repetitions. Unequal counts would make the source's independent "
            "randperm balancing incompatible with leakage-safe block alignment: "
            f"{counts_by_rep.tolist()}"
        )

    reference_groups = _event_field(ordered_by_rep[0], "group")
    for rep_index, events_rep in enumerate(ordered_by_rep[1:], start=2):
        if not np.array_equal(_event_field(events_rep, "group"), reference_groups):
            raise ValueError(
                f"Roving blocks do not align between repetition 1 and {rep_index}"
            )

    start = 100  # MATLAB index 101
    stop = 901   # MATLAB index 901, inclusive
    epochs = np.stack(
        [
            np.stack(
                [event.epoch[:, start:stop] for event in events_rep],
                axis=0,
            )
            for events_rep in ordered_by_rep
        ],
        axis=1,
    )
    expected_shape = (
        global_min_trials,
        num_repetitions,
        spec.n_channels,
        stop - start,
    )
    if epochs.shape != expected_shape:
        raise AssertionError(
            f"Unexpected repetition-map epoch shape {epochs.shape}; "
            f"expected {expected_shape}"
        )

    first_rep = ordered_by_rep[0]
    metadata.update(
        {
            "source_data_file": str(data_path.resolve()),
            "source_matlab_script": "SVM_rep_map.m",
            "source_loader": "Gen_M2Mat.m",
            "source_erp_window_matlab": [101, 901],
            "num_repetitions": num_repetitions,
            "deviant_position": spec.deviant_position,
            "deviant_onset_ms": spec.deviant_onset_ms,
            "counts_by_repetition_before_balance": counts_by_rep.tolist(),
            "global_min_trials": global_min_trials,
            "all_repetitions_have_identical_block_order": True,
            "context_counts": context_counts,
        }
    )
    return RovingRepetitionEpochs(
        epochs=epochs,
        group_ids=reference_groups,
        trials=_event_field(first_rep, "trial"),
        stimuli=_event_field(first_rep, "stimulus"),
        contexts=_event_field(first_rep, "context"),
        repetitions=repetitions,
        time_ms=np.arange(stop - start, dtype=int),
        metadata=metadata,
    )
