"""Leakage-safe, held-out model analysis for AB/BA manuscript Figure 4.

Each seed trains two separate probability contexts (85% AB and 85% BA), then
tests both learned networks on the *same balanced stream* with plasticity off.
Thus a physical AB or BA test sequence is identical on both sides of every
predicted-versus-unexpected contrast.  Context, test order and perturbations
are paired within seed; trials are averaged before inference.
"""

from __future__ import annotations

import csv
import dataclasses as dc
import hashlib
import itertools
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats

from final_figures.figure_2.model_data import _run_fast, self_test_fast_simulator
from final_figures.figure_2.inference import (
    _observed_decoder,
    _permuted_decoder_accuracy,
    _standardize_endpoint_folds,
)
from model0.config import A1Config
from model0.model import simulate as simulate_reference
from tasks.ab_ba_model0.ab_ba import (
    AB_BA_OVERRIDES,
    P_REGULAR,
    TIMING_LONG,
    TIMING_SHORT,
    build_stim,
    shuffled_codes,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_DIR = HERE / "data"
NPZ_PATH = DATA_DIR / "model_figure4_ab_ba.npz"
PROVENANCE_PATH = DATA_DIR / "model_figure4_provenance.json"
RUNS_PATH = DATA_DIR / "model_figure4_runs.csv"
SUMMARY_PATH = DATA_DIR / "figure_4_summary.csv"
INFERENCE_PATH = DATA_DIR / "figure_4_inference.csv"
WEIGHT_TRAJECTORY_PATH = DATA_DIR / "model_figure4_weight_trajectory.csv"
MECHANISM_TIME_PATH = DATA_DIR / "model_figure4_mechanism_timecourses.csv"
SURPRISE_TIME_PATH = DATA_DIR / "model_figure4_surprise_timecourses.csv"
SURPRISAL_PATH = DATA_DIR / "model_figure4_surprisal_response.csv"
MODEL_DECODER_PATH = DATA_DIR / "model_figure4_context_decoder.csv"

CONDITIONS = (
    "intact",
    "no_depression",
    "no_recurrent_learning",
    "uniform_inhibition",
)
CONDITION_LABELS = (
    "Intact",
    "No depression",
    "No recurrent learning",
    "Uniform inhibition",
)
CONTEXTS = ("AB_context", "BA_context")
SEQUENCES = ("AB", "BA")
ROLES = ("predicted", "unexpected")
SEEDS = np.arange(12, dtype=int)

TRAIN_TRIALS = 400
TEST_TRIALS = 120
TRAIN_P_AB = {"AB_context": P_REGULAR, "BA_context": 1.0 - P_REGULAR}
#: Figure 4 uses the recorded Experiment-1 timing without substitution:
#: two contiguous 180-ms tones and a 1.5-s inter-sequence interval.  Keeping
#: this as an explicit copy prevents model-side display or cache code from
#: silently drifting away from the ECoG-matched task definition.
TIMING_MODEL = dict(TIMING_LONG)
TONE_DURATION_MS = int(round(1000 * TIMING_MODEL["tone_dur"]))
INTRA_GAP_MS = int(round(1000 * TIMING_MODEL["intra_gap"]))
INTER_GAP_MS = int(round(1000 * TIMING_MODEL["inter_gap"]))
TARGET_ONSET_MS = TONE_DURATION_MS + INTRA_GAP_MS
DISPLAY_PRE_MS = TARGET_ONSET_MS
DISPLAY_POST_MS = 600 - TARGET_ONSET_MS
TARGET_WINDOW_MS = (0, TONE_DURATION_MS)
PREDICTION_WINDOW_MS = (-50, 0)
CLUSTER_FORMING_ALPHA = 0.05
WEIGHT_CHECKPOINTS = np.arange(0, TRAIN_TRIALS + 1, 25, dtype=int)
MODEL_DECODER_TIME_MS = np.arange(0, 601, 5, dtype=int)
MODEL_DECODER_BOOTSTRAPS = 4_999


def condition_config(condition: str) -> tuple[A1Config, bool]:
    """Return the ECoG-matched AB/BA core or one prespecified lesion.

    The intact values come only from ``tasks.ab_ba_model0.ab_ba``. Uniform
    inhibition preserves the E->I and I->E row sums independently, so it
    changes spatial structure without changing total inhibitory gain.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown AB/BA condition: {condition}")
    intact = A1Config(N=2, **AB_BA_OVERRIDES)
    cfg = dc.replace(intact)
    learn = condition != "no_recurrent_learning"
    if condition == "no_depression":
        cfg.stp_enabled = False
    elif condition == "uniform_inhibition":
        ei_row = intact.w_EI_self + intact.w_EI_lat
        ie_row = intact.w_IE_self + intact.w_IE_lat
        cfg.w_EI_self = cfg.w_EI_lat = ei_row / 2.0
        cfg.w_IE_self = cfg.w_IE_lat = ie_row / 2.0
        if not np.isclose(cfg.w_EI_self + cfg.w_EI_lat, ei_row):
            raise AssertionError("Uniform E->I lesion changed row sum")
        if not np.isclose(cfg.w_IE_self + cfg.w_IE_lat, ie_row):
            raise AssertionError("Uniform I->E lesion changed row sum")
    return cfg, learn


def self_test_ab_ba_core() -> dict[str, float | bool]:
    """Verify the compiled simulator under the new load-bearing parameters."""

    cfg, learn = condition_config("intact")
    stimulus = np.zeros((2, 720), dtype=float)
    stimulus[0, 20:200] = 1.0
    stimulus[1, 200:380] = 1.0
    initial = np.array([[0.04, 0.01], [0.08, 0.03]], dtype=float)
    reference = simulate_reference(
        stimulus,
        cfg=cfg,
        W_init=initial,
        learn=learn,
        seed=91,
    )
    fast_e, fast_w = _run_fast(
        stimulus,
        cfg,
        learn=learn,
        session_seed=91,
        weights_initial=initial,
    )
    max_e = float(np.max(np.abs(reference["E"] - fast_e)))
    max_w = float(np.max(np.abs(reference["W_final"] - fast_w)))
    np.testing.assert_allclose(reference["E"], fast_e, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        reference["W_final"], fast_w, atol=2e-11, rtol=2e-11
    )
    return {"passed": True, "max_abs_E": max_e, "max_abs_W": max_w}


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO / "model0" / "config.py",
        REPO / "model0" / "model.py",
        REPO / "tasks" / "ab_ba_model0" / "ab_ba.py",
        REPO / "final_figures" / "figure_2" / "model_data.py",
        REPO / "final_figures" / "figure_2" / "inference.py",
        REPO / "final_figures" / "figure_7" / "figure7_data.py",
    )
    return {str(path.relative_to(REPO)): _sha256(path) for path in paths}


def _specification() -> dict[str, Any]:
    configs = {}
    for condition in CONDITIONS:
        cfg, learn = condition_config(condition)
        configs[condition] = {"learn_during_training": learn, "config": asdict(cfg)}
    return {
        "schema": 4,
        "analysis": "separate context training; identical balanced held-out test",
        "seeds": SEEDS.tolist(),
        "train_trials": TRAIN_TRIALS,
        "test_trials": TEST_TRIALS,
        "train_p_ab": TRAIN_P_AB,
        "test_p_ab": 0.5,
        "tone_duration_ms": TONE_DURATION_MS,
        "intra_gap_ms": INTRA_GAP_MS,
        "inter_gap_ms": INTER_GAP_MS,
        "timing_model_seconds": dict(TIMING_MODEL),
        "timing_long_seconds": dict(TIMING_LONG),
        "timing_short_seconds": dict(TIMING_SHORT),
        "target_onset_ms": TARGET_ONSET_MS,
        "target_window_ms": list(TARGET_WINDOW_MS),
        "prediction_window_ms": list(PREDICTION_WINDOW_MS),
        "weight_checkpoints_training_pairs": WEIGHT_CHECKPOINTS.tolist(),
        "model_decoder_time_ms": MODEL_DECODER_TIME_MS.tolist(),
        "model_decoder_ridge_lambda": 1e-2,
        "model_decoder_folds": 5,
        "model_decoder_bootstraps": MODEL_DECODER_BOOTSTRAPS,
        "model_decoder_randomizations": int(2 ** SEEDS.size),
        "model_decoder_exact_p": "exceedance count / 2^n paired seeds",
        "model_decoder_cluster_family": "both physical sequences and all time samples",
        "model_decoder_unit": "paired seed; trial-averaged before decoding",
        "conditions": configs,
        "source_hashes": _source_hashes(),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _spec_hash(specification: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(specification).encode()).hexdigest()


def _target_epochs(
    history: np.ndarray,
    starts: np.ndarray,
    codes: np.ndarray,
    sequence: str,
) -> np.ndarray:
    """Return trial × time target-channel epochs, aligned to target onset."""

    selected = np.flatnonzero(codes == sequence)
    channel = 1 if sequence == "AB" else 0
    n_time = DISPLAY_PRE_MS + DISPLAY_POST_MS + 1
    epochs = np.empty((selected.size, n_time), dtype=float)
    for row, trial_index in enumerate(selected):
        sequence_start = int(starts[trial_index])
        lo = sequence_start + TARGET_ONSET_MS - DISPLAY_PRE_MS
        hi = lo + n_time
        epochs[row] = history[channel, lo:hi]
    return epochs


def _all_channel_epochs(
    history: np.ndarray,
    starts: np.ndarray,
    codes: np.ndarray,
    sequence: str,
) -> np.ndarray:
    """Return trial × channel × time epochs on the common sequence clock."""

    selected = np.flatnonzero(codes == sequence)
    n_time = DISPLAY_PRE_MS + DISPLAY_POST_MS + 1
    epochs = np.empty((selected.size, history.shape[0], n_time), dtype=float)
    for row, trial_index in enumerate(selected):
        sequence_start = int(starts[trial_index])
        lo = sequence_start
        hi = lo + n_time
        epochs[row] = history[:, lo:hi]
    return epochs


def _fixed_seed_folds(n_seeds: int) -> np.ndarray:
    """Five deterministic grouped folds reused across all decoder times."""

    order = np.random.RandomState(11).permutation(n_seeds)
    fold_ids = np.full(n_seeds, -1, dtype=np.int64)
    for fold, indices in enumerate(np.array_split(order, 5)):
        fold_ids[indices] = fold
    if np.any(fold_ids < 0):
        raise AssertionError("Model decoder folds do not cover every seed")
    return fold_ids


def _above_threshold_runs(
    statistic: np.ndarray,
    threshold: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Return half-open one-dimensional clusters and excess mass."""

    selected = np.asarray(statistic) > np.asarray(threshold)
    padded = np.r_[False, selected, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (
            int(start),
            int(stop),
            float(np.sum(statistic[start:stop] - threshold[start:stop])),
        )
        for start, stop in zip(starts, stops)
    ]


def _exact_joint_decoder_clusters(
    observed: np.ndarray,
    null: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-sided cluster FWER across sequence identity and time.

    Every one of the ``2**n`` paired-seed label assignments is present in the
    null distribution, including the observed assignment. Therefore the exact
    randomization probability is the exceedance count divided by ``2**n``;
    the Monte-Carlo ``+1`` correction is neither required nor appropriate.
    """

    n_sequences, n_randomizations, n_time = null.shape
    # Match the stored randomization precision so that the observed assignment
    # (which is explicitly present in ``null``) cannot be lost to round-off.
    observed_for_test = np.asarray(observed, dtype=null.dtype)
    threshold = np.quantile(
        null,
        1.0 - CLUSTER_FORMING_ALPHA,
        axis=1,
    )
    null_maximum = np.zeros(n_randomizations, dtype=float)
    for sequence_index in range(n_sequences):
        for randomization_index in range(n_randomizations):
            runs = _above_threshold_runs(
                null[sequence_index, randomization_index],
                threshold[sequence_index],
            )
            if runs:
                null_maximum[randomization_index] = max(
                    null_maximum[randomization_index],
                    max(run[2] for run in runs),
                )

    significant = np.zeros((n_sequences, n_time), dtype=bool)
    p_corrected = np.ones((n_sequences, n_time), dtype=float)
    for sequence_index in range(n_sequences):
        for start, stop, mass in _above_threshold_runs(
            observed_for_test[sequence_index], threshold[sequence_index]
        ):
            probability = np.count_nonzero(null_maximum >= mass) / float(
                n_randomizations
            )
            p_corrected[sequence_index, start:stop] = probability
            significant[sequence_index, start:stop] = probability < 0.05
    return significant, p_corrected


def _model_context_decoder(
    trial_epochs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decode rare versus regular context with the Figure-2 estimator.

    The independent grouping unit is the paired simulation seed. Trial
    responses are averaged within seed before fitting, so held-out trials are
    never treated as independent replicates. Endpoint 0 is rare and endpoint
    1 is regular, matching the positive-class convention of Figure 2.
    """

    sequence_time = np.arange(trial_epochs.shape[-1], dtype=int)
    time_indices = np.searchsorted(sequence_time, MODEL_DECODER_TIME_MS)
    if not np.array_equal(sequence_time[time_indices], MODEL_DECODER_TIME_MS):
        raise AssertionError("Model epochs do not contain the 0:5:600 ms grid")

    # sequence × role × seed × trial × channel × time; roles are stored as
    # regular, rare and reordered here to rare, regular.
    endpoints = np.take(trial_epochs, [1, 0], axis=1).mean(axis=3)
    endpoints = np.take(endpoints, time_indices, axis=-1)
    fold_ids = _fixed_seed_folds(endpoints.shape[2])
    observed = np.empty((len(SEQUENCES), MODEL_DECODER_TIME_MS.size), dtype=float)
    ci_low = np.empty_like(observed)
    ci_high = np.empty_like(observed)
    null = np.empty(
        (len(SEQUENCES), 2 ** endpoints.shape[2], MODEL_DECODER_TIME_MS.size),
        dtype=np.float32,
    )
    exact_swaps = np.asarray(
        list(itertools.product((0, 1), repeat=endpoints.shape[2])),
        dtype=np.uint8,
    )
    for sequence_index in range(len(SEQUENCES)):
        endpoint = np.transpose(endpoints[sequence_index], (1, 0, 2, 3))
        standardized = _standardize_endpoint_folds(endpoint, fold_ids)
        accuracy, seed_correct = _observed_decoder(
            standardized, fold_ids, 1e-2
        )
        observed[sequence_index] = accuracy
        bootstrap_rng = np.random.default_rng(44_700 + sequence_index)
        draws = bootstrap_rng.integers(
            0, seed_correct.shape[0],
            size=(MODEL_DECODER_BOOTSTRAPS, seed_correct.shape[0]),
        )
        bootstrap = seed_correct[draws].mean(axis=1)
        ci_low[sequence_index], ci_high[sequence_index] = np.quantile(
            bootstrap, [0.025, 0.975], axis=0
        )
        null[sequence_index] = _permuted_decoder_accuracy(
            standardized, fold_ids, exact_swaps, 1e-2
        )

    significant, p_corrected = _exact_joint_decoder_clusters(observed, null)
    return {
        "model_decoder_time_ms": MODEL_DECODER_TIME_MS.copy(),
        "model_decoder_accuracy": observed,
        "model_decoder_ci_low": ci_low,
        "model_decoder_ci_high": ci_high,
        "model_decoder_significant": significant,
        "model_decoder_p_corrected": p_corrected,
        "model_decoder_fold_ids": fold_ids,
    }


def _latent_histories(
    excitatory: np.ndarray,
    weights: np.ndarray,
    cfg: A1Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct source-model recurrent current and inhibitory rate.

    ``model0`` records E and I after each Euler update, while the recurrent
    current at sample t uses the state before that update.  Shifting the saved
    E history by one sample reproduces that ordering exactly.
    """

    old_e = np.empty_like(excitatory)
    old_e[:, 0] = 0.0
    old_e[:, 1:] = excitatory[:, :-1]
    recurrent = weights @ old_e

    n_channels, n_time = excitatory.shape
    eye = np.eye(n_channels)
    m_ei = (
        cfg.w_EI_lat * np.ones((n_channels, n_channels))
        + (cfg.w_EI_self - cfg.w_EI_lat) * eye
    )
    inhibitory = np.zeros_like(excitatory)
    state = np.zeros(n_channels, dtype=float)
    for time_index in range(n_time):
        drive = m_ei @ old_e[:, time_index]
        state += cfg.dt * (-state + np.maximum(drive, 0.0)) / cfg.tau_I
        inhibitory[:, time_index] = state
    return recurrent, inhibitory


def _role_context(sequence: str, role: str) -> str:
    if sequence == "AB":
        return "AB_context" if role == "predicted" else "BA_context"
    return "BA_context" if role == "predicted" else "AB_context"


def _t_stat(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    sd = values.std(axis=axis, ddof=1)
    n = values.shape[axis]
    denominator = sd / np.sqrt(n)
    out = np.zeros_like(mean, dtype=float)
    np.divide(mean, denominator, out=out, where=denominator > 0)
    out[(denominator == 0) & (mean != 0)] = np.sign(mean[(denominator == 0) & (mean != 0)]) * np.inf
    return out


def _clusters(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1))


def _joint_cluster_test(difference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-sided sign-flip clusters, joint over sequence and time."""

    # sequence × seed × time
    difference = np.asarray(difference, dtype=float)
    n_sequences, n_seeds, n_time = difference.shape
    threshold = stats.t.ppf(1.0 - CLUSTER_FORMING_ALPHA / 2.0, n_seeds - 1)
    observed_t = _t_stat(difference, axis=1)
    observed_clusters: list[tuple[int, np.ndarray, float]] = []
    for sequence_index in range(n_sequences):
        for run in _clusters(np.abs(observed_t[sequence_index]) >= threshold):
            mass = float(np.abs(observed_t[sequence_index, run]).sum())
            observed_clusters.append((sequence_index, run, mass))

    null_max = np.zeros(2**n_seeds, dtype=float)
    for permutation_index, bits in enumerate(itertools.product((-1.0, 1.0), repeat=n_seeds)):
        signed = difference * np.asarray(bits)[None, :, None]
        permuted_t = _t_stat(signed, axis=1)
        maximum = 0.0
        for sequence_index in range(n_sequences):
            for run in _clusters(np.abs(permuted_t[sequence_index]) >= threshold):
                maximum = max(maximum, float(np.abs(permuted_t[sequence_index, run]).sum()))
        null_max[permutation_index] = maximum

    mask = np.zeros((n_sequences, n_time), dtype=bool)
    p_map = np.ones((n_sequences, n_time), dtype=float)
    for sequence_index, run, mass in observed_clusters:
        p_value = np.count_nonzero(null_max >= mass) / float(null_max.size)
        p_map[sequence_index, run] = p_value
        if p_value < 0.05:
            mask[sequence_index, run] = True
    return mask, p_map


def _max_t_p(values: np.ndarray) -> np.ndarray:
    """Exact two-sided family-wise p values for condition-wise effects."""

    values = np.asarray(values, dtype=float)
    n_tests, n_seeds = values.shape
    observed = np.abs(_t_stat(values, axis=1))
    null_max = np.empty(2**n_seeds, dtype=float)
    for index, bits in enumerate(itertools.product((-1.0, 1.0), repeat=n_seeds)):
        signed = values * np.asarray(bits)[None, :]
        null_max[index] = np.max(np.abs(_t_stat(signed, axis=1)))
    return np.asarray([
        np.count_nonzero(null_max >= statistic) / float(null_max.size)
        for statistic in observed
    ])


def _interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=-1)
    sem = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    critical = stats.t.ppf(0.975, values.shape[-1] - 1)
    return mean, mean - critical * sem, mean + critical * sem


def _simulate() -> dict[str, np.ndarray]:
    n_conditions = len(CONDITIONS)
    n_sequences = len(SEQUENCES)
    n_roles = len(ROLES)
    n_seeds = SEEDS.size
    n_time = DISPLAY_PRE_MS + DISPLAY_POST_MS + 1

    response = np.empty((n_conditions, n_sequences, n_roles, n_seeds, n_time))
    population_response = np.empty_like(response)
    recurrent = np.empty_like(response)
    inhibitory = np.empty_like(response)
    channel_response = np.empty((n_sequences, n_roles, n_seeds, 2, n_time))
    channel_recurrent = np.empty_like(channel_response)
    channel_inhibitory = np.empty_like(channel_response)
    weights = np.empty((n_conditions, len(CONTEXTS), n_seeds, 2, 2))
    weight_trajectory = np.zeros(
        (len(CONTEXTS), n_seeds, WEIGHT_CHECKPOINTS.size, 2, 2),
        dtype=float,
    )
    decoder_trial_epochs = np.empty(
        (len(SEQUENCES), len(ROLES), n_seeds, TEST_TRIALS // 2, 2, n_time),
        dtype=float,
    )

    for seed_index, seed in enumerate(SEEDS):
        train_codes = {
            context: shuffled_codes(
                TRAIN_TRIALS,
                TRAIN_P_AB[context],
                np.random.default_rng(int(seed) + 1000 * (context == "AB_context") + 2000 * (context == "BA_context")),
            )
            for context in CONTEXTS
        }
        test_codes = shuffled_codes(
            TEST_TRIALS, 0.5, np.random.default_rng(int(seed) + 3000)
        )

        for condition_index, condition in enumerate(CONDITIONS):
            cfg, learn = condition_config(condition)
            learned: dict[str, np.ndarray] = {}
            for context_index, context in enumerate(CONTEXTS):
                train_stimulus, _, _ = build_stim(
                    train_codes[context], cfg, 0, 1, **TIMING_MODEL
                )
                _, learned[context] = _run_fast(
                    train_stimulus,
                    cfg,
                    learn=learn,
                    session_seed=int(seed),
                )
                weights[condition_index, context_index, seed_index] = learned[context]

            test_stimulus, starts, _ = build_stim(
                test_codes, cfg, 0, 1, **TIMING_MODEL
            )
            test_labels = np.asarray(test_codes)
            histories: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for context in CONTEXTS:
                excitatory, _ = _run_fast(
                    test_stimulus,
                    cfg,
                    learn=False,
                    session_seed=int(seed),
                    weights_initial=learned[context],
                )
                rec, inh = _latent_histories(excitatory, learned[context], cfg)
                histories[context] = (excitatory, rec, inh)

            for sequence_index, sequence in enumerate(SEQUENCES):
                for role_index, role in enumerate(ROLES):
                    context = _role_context(sequence, role)
                    excitatory, rec, inh = histories[context]
                    excitatory_epochs = _all_channel_epochs(
                        excitatory, starts, test_labels, sequence
                    )
                    recurrent_epochs = _all_channel_epochs(
                        rec, starts, test_labels, sequence
                    )
                    inhibitory_epochs = _all_channel_epochs(
                        inh, starts, test_labels, sequence
                    )
                    mean_exc = excitatory_epochs.mean(axis=0)
                    mean_rec = recurrent_epochs.mean(axis=0)
                    mean_inh = inhibitory_epochs.mean(axis=0)
                    target_channel = 1 if sequence == "AB" else 0
                    response[condition_index, sequence_index, role_index, seed_index] = mean_exc[target_channel]
                    population_response[condition_index, sequence_index, role_index, seed_index] = mean_exc.mean(axis=0)
                    recurrent[condition_index, sequence_index, role_index, seed_index] = mean_rec[target_channel]
                    inhibitory[condition_index, sequence_index, role_index, seed_index] = mean_inh[target_channel]
                    if condition_index == 0:
                        decoder_trial_epochs[sequence_index, role_index, seed_index] = excitatory_epochs
                        channel_response[sequence_index, role_index, seed_index] = mean_exc
                        channel_recurrent[sequence_index, role_index, seed_index] = mean_rec
                        channel_inhibitory[sequence_index, role_index, seed_index] = mean_inh

        # Exact prefix reruns provide the intact learning trajectory without
        # resetting latent state inside a session.  A run on the first k pairs
        # is mathematically identical to stopping the full stream after pair k.
        intact_cfg, intact_learn = condition_config("intact")
        for context_index, context in enumerate(CONTEXTS):
            for checkpoint_index, checkpoint in enumerate(WEIGHT_CHECKPOINTS[1:], start=1):
                if checkpoint == TRAIN_TRIALS:
                    checkpoint_weights = weights[0, context_index, seed_index]
                else:
                    prefix_stimulus, _, _ = build_stim(
                        train_codes[context][:int(checkpoint)], intact_cfg, 0, 1,
                        **TIMING_MODEL,
                    )
                    _, checkpoint_weights = _run_fast(
                        prefix_stimulus,
                        intact_cfg,
                        learn=intact_learn,
                        session_seed=int(seed),
                    )
                weight_trajectory[
                    context_index, seed_index, checkpoint_index
                ] = checkpoint_weights

    time_ms = np.arange(n_time, dtype=float) - DISPLAY_PRE_MS
    target = (time_ms >= TARGET_WINDOW_MS[0]) & (time_ms < TARGET_WINDOW_MS[1])
    prediction = (time_ms >= PREDICTION_WINDOW_MS[0]) & (time_ms < PREDICTION_WINDOW_MS[1])
    target_response = response[..., target].mean(axis=-1)
    role_difference = response[:, :, 1] - response[:, :, 0]
    condition_effect = target_response[:, :, 1].mean(axis=1) - target_response[:, :, 0].mean(axis=1)

    intact_index = CONDITIONS.index("intact")
    time_sig, time_cluster_p = _joint_cluster_test(role_difference[intact_index])
    effect_p = _max_t_p(condition_effect)
    lesion_difference = condition_effect[1:] - condition_effect[0]
    lesion_p = _max_t_p(lesion_difference)
    lesion_mean, lesion_ci_low, lesion_ci_high = _interval(lesion_difference)

    weight_alignment = np.empty((2, n_seeds), dtype=float)
    weight_alignment[0] = (
        weights[intact_index, 0, :, 1, 0] - weights[intact_index, 0, :, 0, 1]
    )
    weight_alignment[1] = (
        weights[intact_index, 1, :, 0, 1] - weights[intact_index, 1, :, 1, 0]
    )
    weight_p = _max_t_p(weight_alignment)

    target_index = int(np.flatnonzero(time_ms == 0)[0])
    mechanism = np.empty((3, 2, n_seeds), dtype=float)
    # Average sequence identities within seed before inference.
    mechanism[0] = recurrent[intact_index][..., prediction].mean(axis=(0, -1))
    mechanism[1] = inhibitory[intact_index, :, :, :, target_index].mean(axis=0)
    mechanism[2] = target_response[intact_index].mean(axis=0)
    mechanism_difference = mechanism[:, 1] - mechanism[:, 0]
    mechanism_p = _max_t_p(mechanism_difference)

    mechanism_time_difference = np.empty((3, n_seeds, n_time), dtype=float)
    # Positive values follow the causal reading direction in the panel:
    # prediction-related recurrent drive, prediction-related inhibition, and
    # the resulting unexpected-minus-predicted excitatory response.
    mechanism_time_difference[0] = (
        recurrent[intact_index, :, 0] - recurrent[intact_index, :, 1]
    ).mean(axis=0)
    mechanism_time_difference[1] = (
        inhibitory[intact_index, :, 0] - inhibitory[intact_index, :, 1]
    ).mean(axis=0)
    mechanism_time_difference[2] = (
        response[intact_index, :, 1] - response[intact_index, :, 0]
    ).mean(axis=0)
    mechanism_time_sig, mechanism_time_cluster_p = _joint_cluster_test(
        mechanism_time_difference
    )

    surprise_response = role_difference[intact_index]
    population_surprise_response = (
        population_response[intact_index, :, 1]
        - population_response[intact_index, :, 0]
    )
    population_time_sig, population_time_cluster_p = _joint_cluster_test(
        population_surprise_response
    )
    surprisal_bits = np.asarray(
        [-np.log2(TRAIN_P_AB["AB_context"]), -np.log2(TRAIN_P_AB["BA_context"])],
        dtype=float,
    )
    # Both contexts use 85/15 probabilities. Store regular then rare to match
    # ROLES, independent of physical AB/BA identity.
    surprisal_bits = np.sort(surprisal_bits)
    surprisal_target_response = target_response[intact_index]
    population_target_response = np.compress(
        target,
        population_response[intact_index],
        axis=-1,
    ).mean(axis=-1)
    model_decoder = _model_context_decoder(decoder_trial_epochs)

    effect_mean, effect_ci_low, effect_ci_high = _interval(condition_effect)
    weight_mean, weight_ci_low, weight_ci_high = _interval(weight_alignment)

    return {
        "time_ms": time_ms,
        "conditions": np.asarray(CONDITIONS),
        "condition_labels": np.asarray(CONDITION_LABELS),
        "contexts": np.asarray(CONTEXTS),
        "sequences": np.asarray(SEQUENCES),
        "roles": np.asarray(ROLES),
        "seeds": SEEDS,
        "response": response,
        "population_response": population_response,
        "recurrent": recurrent,
        "inhibitory": inhibitory,
        "channel_response": channel_response,
        "channel_recurrent": channel_recurrent,
        "channel_inhibitory": channel_inhibitory,
        "weights": weights,
        "weight_checkpoints": WEIGHT_CHECKPOINTS,
        "weight_trajectory": weight_trajectory,
        "target_response": target_response,
        "condition_effect": condition_effect,
        "condition_effect_mean": effect_mean,
        "condition_effect_ci_low": effect_ci_low,
        "condition_effect_ci_high": effect_ci_high,
        "condition_effect_p_fwer": effect_p,
        "lesion_vs_intact_p_fwer": lesion_p,
        "lesion_difference": lesion_difference,
        "lesion_difference_mean": lesion_mean,
        "lesion_difference_ci_low": lesion_ci_low,
        "lesion_difference_ci_high": lesion_ci_high,
        "time_significant": time_sig,
        "time_cluster_p_fwer": time_cluster_p,
        "weight_alignment": weight_alignment,
        "weight_alignment_mean": weight_mean,
        "weight_alignment_ci_low": weight_ci_low,
        "weight_alignment_ci_high": weight_ci_high,
        "weight_alignment_p_fwer": weight_p,
        "mechanism": mechanism,
        "mechanism_difference_p_fwer": mechanism_p,
        "mechanism_time_difference": mechanism_time_difference,
        "mechanism_time_significant": mechanism_time_sig,
        "mechanism_time_cluster_p_fwer": mechanism_time_cluster_p,
        "surprise_response": surprise_response,
        "population_surprise_response": population_surprise_response,
        "population_time_significant": population_time_sig,
        "population_time_cluster_p_fwer": population_time_cluster_p,
        "surprisal_bits": surprisal_bits,
        "surprisal_target_response": surprisal_target_response,
        "surprisal_population_response": population_target_response,
        **model_decoder,
    }


def _write_csvs(data: dict[str, np.ndarray]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("condition", "sequence", "role", "seed", "target_response"),
        )
        writer.writeheader()
        for ci, condition in enumerate(CONDITIONS):
            for si, sequence in enumerate(SEQUENCES):
                for ri, role in enumerate(ROLES):
                    for zi, seed in enumerate(SEEDS):
                        writer.writerow({
                            "condition": condition,
                            "sequence": sequence,
                            "role": role,
                            "seed": int(seed),
                            "target_response": float(data["target_response"][ci, si, ri, zi]),
                        })

    with SUMMARY_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("condition", "n_seeds", "unexpected_minus_predicted_mean", "ci95_low", "ci95_high"),
        )
        writer.writeheader()
        for index, condition in enumerate(CONDITIONS):
            writer.writerow({
                "condition": condition,
                "n_seeds": int(SEEDS.size),
                "unexpected_minus_predicted_mean": float(data["condition_effect_mean"][index]),
                "ci95_low": float(data["condition_effect_ci_low"][index]),
                "ci95_high": float(data["condition_effect_ci_high"][index]),
            })

    with INFERENCE_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("family", "contrast", "p_fwer", "method"))
        writer.writeheader()
        for index, condition in enumerate(CONDITIONS):
            writer.writerow({
                "family": "condition_effect_vs_zero",
                "contrast": condition,
                "p_fwer": float(data["condition_effect_p_fwer"][index]),
                "method": "exact paired sign flips; max-|t| over four conditions",
            })
        for index, condition in enumerate(CONDITIONS[1:]):
            writer.writerow({
                "family": "lesion_vs_intact",
                "contrast": condition,
                "p_fwer": float(data["lesion_vs_intact_p_fwer"][index]),
                "method": "exact paired sign flips; max-|t| over three planned lesions",
            })
        for index, context in enumerate(CONTEXTS):
            writer.writerow({
                "family": "directional_weight_alignment",
                "contrast": context,
                "p_fwer": float(data["weight_alignment_p_fwer"][index]),
                "method": "exact paired sign flips; max-|t| over two contexts",
            })

    with WEIGHT_TRAJECTORY_PATH.open("w", newline="") as stream:
        fields = (
            "context", "seed", "training_pair", "w_A_from_A",
            "w_A_from_B", "w_B_from_A", "w_B_from_B",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        trajectory = np.asarray(data["weight_trajectory"])
        checkpoints = np.asarray(data["weight_checkpoints"])
        for context_index, context in enumerate(CONTEXTS):
            for seed_index, seed in enumerate(SEEDS):
                for checkpoint_index, checkpoint in enumerate(checkpoints):
                    matrix = trajectory[context_index, seed_index, checkpoint_index]
                    writer.writerow({
                        "context": context,
                        "seed": int(seed),
                        "training_pair": int(checkpoint),
                        "w_A_from_A": float(matrix[0, 0]),
                        "w_A_from_B": float(matrix[0, 1]),
                        "w_B_from_A": float(matrix[1, 0]),
                        "w_B_from_B": float(matrix[1, 1]),
                    })

    with MECHANISM_TIME_PATH.open("w", newline="") as stream:
        fields = ("metric", "seed", "time_from_target_ms", "difference")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        names = (
            "predicted_minus_unexpected_recurrent_current",
            "predicted_minus_unexpected_inhibitory_rate",
            "unexpected_minus_predicted_excitatory_rate",
        )
        differences = np.asarray(data["mechanism_time_difference"])
        time_ms = np.asarray(data["time_ms"])
        for metric_index, metric in enumerate(names):
            for seed_index, seed in enumerate(SEEDS):
                for time_index, time in enumerate(time_ms):
                    writer.writerow({
                        "metric": metric,
                        "seed": int(seed),
                        "time_from_target_ms": float(time),
                        "difference": float(differences[metric_index, seed_index, time_index]),
                    })

    with SURPRISE_TIME_PATH.open("w", newline="") as stream:
        fields = (
            "sequence", "measure", "seed", "sequence_time_ms",
            "rare_minus_regular_rate", "cluster_p_fwer",
            "cluster_significant",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        sequence_time = np.asarray(data["time_ms"]) + DISPLAY_PRE_MS
        measures = (
            (
                "target_channel",
                np.asarray(data["surprise_response"]),
                np.asarray(data["time_cluster_p_fwer"]),
                np.asarray(data["time_significant"]),
            ),
            (
                "all_channel_mean",
                np.asarray(data["population_surprise_response"]),
                np.asarray(data["population_time_cluster_p_fwer"]),
                np.asarray(data["population_time_significant"]),
            ),
        )
        for measure, difference, p_values, significant in measures:
            for sequence_index, sequence in enumerate(SEQUENCES):
                for seed_index, seed in enumerate(SEEDS):
                    for time_index, time in enumerate(sequence_time):
                        writer.writerow({
                            "sequence": sequence,
                            "measure": measure,
                            "seed": int(seed),
                            "sequence_time_ms": float(time),
                            "rare_minus_regular_rate": float(
                                difference[sequence_index, seed_index, time_index]
                            ),
                            "cluster_p_fwer": float(
                                p_values[sequence_index, time_index]
                            ),
                            "cluster_significant": int(
                                significant[sequence_index, time_index]
                            ),
                        })

    with SURPRISAL_PATH.open("w", newline="") as stream:
        fields = (
            "sequence", "role", "seed", "surprisal_bits",
            "target_channel_rate", "all_channel_mean_rate",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        response = np.asarray(data["surprisal_target_response"])
        population = np.asarray(data["surprisal_population_response"])
        for sequence_index, sequence in enumerate(SEQUENCES):
            for role_index, role in enumerate(ROLES):
                for seed_index, seed in enumerate(SEEDS):
                    writer.writerow({
                        "sequence": sequence,
                        "role": role,
                        "seed": int(seed),
                        "surprisal_bits": float(data["surprisal_bits"][role_index]),
                        "target_channel_rate": float(
                            response[sequence_index, role_index, seed_index]
                        ),
                        "all_channel_mean_rate": float(
                            population[sequence_index, role_index, seed_index]
                        ),
                    })

    with MODEL_DECODER_PATH.open("w", newline="") as stream:
        fields = (
            "sequence", "sequence_time_ms", "balanced_accuracy", "ci95_low",
            "ci95_high", "cluster_p_fwer", "cluster_significant",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sequence_index, sequence in enumerate(SEQUENCES):
            for time_index, time in enumerate(data["model_decoder_time_ms"]):
                writer.writerow({
                    "sequence": sequence,
                    "sequence_time_ms": int(time),
                    "balanced_accuracy": float(data["model_decoder_accuracy"][sequence_index, time_index]),
                    "ci95_low": float(data["model_decoder_ci_low"][sequence_index, time_index]),
                    "ci95_high": float(data["model_decoder_ci_high"][sequence_index, time_index]),
                    "cluster_p_fwer": float(data["model_decoder_p_corrected"][sequence_index, time_index]),
                    "cluster_significant": int(data["model_decoder_significant"][sequence_index, time_index]),
                })


def build(force: bool = False) -> dict[str, np.ndarray]:
    specification = _specification()
    digest = _spec_hash(specification)
    if not force and NPZ_PATH.exists() and PROVENANCE_PATH.exists():
        with PROVENANCE_PATH.open() as stream:
            provenance = json.load(stream)
        if provenance.get("specification_sha256") == digest:
            return dict(np.load(NPZ_PATH, allow_pickle=False))

    self_test = {
        "generic_compiled_equations": self_test_fast_simulator(),
        "ecog_matched_ab_ba_core": self_test_ab_ba_core(),
    }
    data = _simulate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = NPZ_PATH.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, NPZ_PATH)
    _write_csvs(data)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "specification_sha256": digest,
        "specification": specification,
        "self_test": self_test,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "outputs": {
            NPZ_PATH.name: _sha256(NPZ_PATH),
            RUNS_PATH.name: _sha256(RUNS_PATH),
            SUMMARY_PATH.name: _sha256(SUMMARY_PATH),
            INFERENCE_PATH.name: _sha256(INFERENCE_PATH),
            WEIGHT_TRAJECTORY_PATH.name: _sha256(WEIGHT_TRAJECTORY_PATH),
            MECHANISM_TIME_PATH.name: _sha256(MECHANISM_TIME_PATH),
            SURPRISE_TIME_PATH.name: _sha256(SURPRISE_TIME_PATH),
            SURPRISAL_PATH.name: _sha256(SURPRISAL_PATH),
            MODEL_DECODER_PATH.name: _sha256(MODEL_DECODER_PATH),
        },
        "replication_unit": (
            "paired training/test seed; trials averaged within seed; sequence "
            "identities retained for time-resolved analyses and averaged only "
            "for prespecified scalar effects"
        ),
        "leakage_guard": "identical balanced held-out test stream; learning disabled during test",
    }
    temporary_json = PROVENANCE_PATH.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, PROVENANCE_PATH)
    return data


if __name__ == "__main__":
    build(force=True)
