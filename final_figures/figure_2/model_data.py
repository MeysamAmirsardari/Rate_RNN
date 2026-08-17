"""Generate the model-side data for manuscript Figure 2.

This module is intentionally independent of the plotting code.  It runs the
committed ``model0`` equations on paired roving sessions, extracts only the
deviant-channel response needed by Figure 2, and writes a versioned data
contract:

``model_figure2_data.npz``
    Complete seed-level repetition time courses and final weights.
``model_figure2_timecourses.csv``
    Figure-ready Rep-1 and Rep-15 time courses.
``model_figure2_buildup.csv``
    Deviant-window response for repetitions 1--15.
``model_figure2_sessions.csv``
    The exact constrained block order for every position/session seed.
``model_provenance.json``
    Full model/task configurations, source hashes, software versions,
    self-test results, schema, and output hashes.

The replication unit is a *simulated session/order seed*, not an independently
initialized network.  The committed model has ``W_init_scale == 0``, so seeds
change the constrained block order and hence the learning history, while every
session starts from the same all-zero recurrent matrix.

The accelerated simulator below is a literal loop translation of
``model0.model.simulate``.  ``fastmath`` is disabled, and
``self_test_fast_simulator`` compares its complete E history and final W
against the source implementation on short fixed stimuli before any cache is
used or generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numba
import numpy as np
from numba import njit


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model0.config import A1Config  # noqa: E402
from model0.model import simulate as reference_simulate  # noqa: E402
from tasks.roving.config import RovingConfig  # noqa: E402
from tasks.roving.roving import (  # noqa: E402
    build_roving_stim,
    generate_block_order,
)


# ---------------------------------------------------------------------------
# Public data-contract constants
# ---------------------------------------------------------------------------
CONDITIONS: tuple[str, ...] = (
    "intact",
    "no_depression",
    "plasticity_frozen",
    "uniform_inhibition",
)
CONDITION_LABELS: tuple[str, ...] = (
    "Intact",
    "No depression",
    "Plasticity frozen",
    "Uniform inhibition",
)
POSITIONS = np.array([1, 2, 3], dtype=np.int64)
DEFAULT_SESSION_SEEDS = np.array(
    [11, 23, 37, 41, 53, 67, 79, 97], dtype=np.int64
)

N_CHANNELS = 5
N_BLOCKS_PER_WORD = 10
N_REPETITIONS = 15
# Recurrent-weight decay for the roving paradigm: tau = 50 s.  Mirrors
# model0.ROVING_W_DECAY; defined here because this figure runs against the
# committed model0 on main, which predates that constant.
ROVING_W_DECAY = 2e-2
TONE_DURATION_MS = 180
EPOCH_PRE_MS = 100
EPOCH_POST_MS = 360
SEQUENCE_FIGURE_STOP_MS = 600
SEQUENCE_FIGURE_SAMPLES = SEQUENCE_FIGURE_STOP_MS + 1

DEFAULT_DATA_DIR = _THIS_FILE.parent / "data"
NPZ_NAME = "model_figure2_data.npz"
PROVENANCE_NAME = "model_provenance.json"
TIMECOURSE_CSV_NAME = "model_figure2_timecourses.csv"
BUILDUP_CSV_NAME = "model_figure2_buildup.csv"
SESSIONS_CSV_NAME = "model_figure2_sessions.csv"


# Parameter-vector indices used inside the Numba kernel.  A flat numeric
# vector avoids object-mode dataclass access and makes the compiled code
# stable across Numba releases.
_DT = 0
_TAU_E = 1
_TAU_I = 2
_TAU_D = 3
_TAU_F = 4
_U = 5
_A_TC = 6
_TAU_TRACE = 7
_ETA_LTP = 8
_ETA_LTD = 9
_W_DECAY = 10
_W_MAX = 11
_W_MAX_SELF = 12
_W_NORM = 13
_ETA_HET = 14
_U_STD = 15

# Boolean/integer flag-vector indices.
_STP_ENABLED = 0
_MULTISCALE_STD = 1
_BOUNDED_PLASTICITY = 2
_PLASTIC_SELF = 3
_LEARN = 4


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _condition_config(condition: str) -> tuple[A1Config, bool]:
    """Return the committed model configuration and plasticity switch."""

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")

    cfg = A1Config(N=N_CHANNELS, multiscale_std=True)
    # Roving forgetting rate: tau = 50 s rather than the A1Config default's
    # 33 min.  At the default the recurrent weights accumulate ACROSS blocks,
    # so by mid session B predicts C, D and E about equally and the
    # transition matrix carries the session average instead of the block
    # currently being repeated.  See model0/config.py (ROVING_W_DECAY) for
    # the induction-strength argument: 15 pairings at a 1.5 s interval leave
    # short-term potentiation, not consolidated LTP.
    cfg.W_decay = ROVING_W_DECAY

    learn = True
    if condition == "no_depression":
        # Exact source-model lesion: x_eff is fixed at one and the input is
        # A_TC * U * s, matching the first-presentation resting drive.
        cfg.stp_enabled = False
    elif condition == "plasticity_frozen":
        # ``simulate(..., learn=False)`` leaves W at its initial value.
        learn = False
    elif condition == "uniform_inhibition":
        # Row-sum matched to the *current* selective N=5 configuration:
        #   E->I: 0.20 + 4*0.05 = 0.40; 0.40/5 = 0.08
        #   I->E: 0.65 + 4*0.20 = 1.45; 1.45/5 = 0.29
        cfg.w_EI_self = 0.08
        cfg.w_EI_lat = 0.08
        cfg.w_IE_self = 0.29
        cfg.w_IE_lat = 0.29

    if condition == "uniform_inhibition":
        selective = A1Config(N=N_CHANNELS, multiscale_std=True)
        selective_ei = (
            selective.w_EI_self
            + (N_CHANNELS - 1) * selective.w_EI_lat
        )
        selective_ie = (
            selective.w_IE_self
            + (N_CHANNELS - 1) * selective.w_IE_lat
        )
        uniform_ei = cfg.w_EI_self + (N_CHANNELS - 1) * cfg.w_EI_lat
        uniform_ie = cfg.w_IE_self + (N_CHANNELS - 1) * cfg.w_IE_lat
        if not (
            np.isclose(selective_ei, uniform_ei, atol=0.0, rtol=1e-14)
            and np.isclose(selective_ie, uniform_ie, atol=0.0, rtol=1e-14)
        ):
            raise AssertionError("Uniform inhibition is not row-sum matched")
    return cfg, learn


def _inhibitory_matrices(cfg: A1Config) -> tuple[np.ndarray, np.ndarray]:
    eye = np.eye(cfg.N, dtype=np.float64)
    ones = np.ones((cfg.N, cfg.N), dtype=np.float64)
    m_ei = (
        cfg.w_EI_lat * ones
        + (cfg.w_EI_self - cfg.w_EI_lat) * eye
    )
    m_ie = (
        cfg.w_IE_lat * ones
        + (cfg.w_IE_self - cfg.w_IE_lat) * eye
    )
    return m_ei, m_ie


def _pack_parameters(
    cfg: A1Config, *, learn: bool
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    parameters = np.array(
        [
            cfg.dt,
            cfg.tau_E,
            cfg.tau_I,
            cfg.tau_D,
            cfg.tau_F,
            cfg.U,
            cfg.A_TC,
            cfg.tau_trace,
            cfg.eta_LTP,
            cfg.eta_LTD,
            cfg.W_decay,
            cfg.W_max,
            cfg.W_max_self,
            cfg.W_norm,
            cfg.eta_het,
            cfg.U_std,
        ],
        dtype=np.float64,
    )
    flags = np.array(
        [
            int(cfg.stp_enabled),
            int(cfg.multiscale_std),
            int(cfg.bounded_plasticity),
            int(cfg.plastic_self),
            int(learn),
        ],
        dtype=np.int64,
    )
    tau_std = np.asarray(cfg.tau_std, dtype=np.float64)
    w_std = np.asarray(cfg.w_std, dtype=np.float64)
    decay_std = np.exp(-cfg.dt / tau_std)
    m_ei, m_ie = _inhibitory_matrices(cfg)
    return parameters, flags, decay_std, w_std, m_ei, m_ie


def _initial_weights(cfg: A1Config, session_seed: int) -> np.ndarray:
    """Match ``model0.simulate`` initialization, including RNG semantics."""

    rng = np.random.default_rng(int(session_seed))
    weights = (
        cfg.W_init_scale
        * np.abs(rng.standard_normal((cfg.N, cfg.N)))
    )
    if not cfg.plastic_self:
        np.fill_diagonal(weights, 0.0)
    return np.asarray(weights, dtype=np.float64)


@njit(cache=False, fastmath=False)
def _simulate_numba_exact(
    stim: np.ndarray,
    weights_initial: np.ndarray,
    parameters: np.ndarray,
    flags: np.ndarray,
    decay_std: np.ndarray,
    w_std: np.ndarray,
    m_ei: np.ndarray,
    m_ie: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal compiled translation of ``model0.model.simulate``.

    Only the E history and final W are returned because these are the states
    required by Figure 2.  All latent states and currents are nevertheless
    evolved in the same operation order as the reference implementation.
    """

    n_channels, n_time = stim.shape
    n_scales = decay_std.size
    dt = parameters[_DT]

    weights = weights_initial.copy()
    excitatory = np.zeros(n_channels, dtype=np.float64)
    inhibitory = np.zeros(n_channels, dtype=np.float64)
    facilitation = np.empty(n_channels, dtype=np.float64)
    resources = np.ones(n_channels, dtype=np.float64)
    trace = np.zeros(n_channels, dtype=np.float64)
    d_std = np.zeros((n_channels, n_scales), dtype=np.float64)
    for channel in range(n_channels):
        facilitation[channel] = parameters[_U]

    excitatory_history = np.zeros(
        (n_channels, n_time), dtype=np.float64
    )

    x_effective = np.ones(n_channels, dtype=np.float64)
    tc_input = np.zeros(n_channels, dtype=np.float64)
    recurrent = np.zeros(n_channels, dtype=np.float64)
    inhibition_to_e = np.zeros(n_channels, dtype=np.float64)
    net_i = np.zeros(n_channels, dtype=np.float64)
    d_e = np.zeros(n_channels, dtype=np.float64)
    d_i = np.zeros(n_channels, dtype=np.float64)
    d_u = np.zeros(n_channels, dtype=np.float64)
    d_x = np.zeros(n_channels, dtype=np.float64)
    d_trace = np.zeros(n_channels, dtype=np.float64)
    d_weights = np.zeros((n_channels, n_channels), dtype=np.float64)

    for time_index in range(n_time):
        # Thalamocortical input and the effective available resource.
        for post in range(n_channels):
            stimulus_value = stim[post, time_index]
            if flags[_STP_ENABLED] == 0:
                x_effective[post] = 1.0
                tc_input[post] = (
                    parameters[_A_TC]
                    * parameters[_U]
                    * stimulus_value
                )
            elif flags[_MULTISCALE_STD] != 0:
                depression = 0.0
                for scale in range(n_scales):
                    depression += d_std[post, scale] * w_std[scale]
                available = 1.0 - depression
                if available < 0.0:
                    available = 0.0
                x_effective[post] = available
                tc_input[post] = (
                    parameters[_A_TC]
                    * parameters[_U]
                    * available
                    * stimulus_value
                )
            else:
                x_effective[post] = resources[post]
                tc_input[post] = (
                    parameters[_A_TC]
                    * facilitation[post]
                    * resources[post]
                    * stimulus_value
                )

        # W @ E, M_IE @ I, and M_EI @ E.  The summation order matches the
        # source arrays' increasing column order.
        for post in range(n_channels):
            recurrent_value = 0.0
            inhibitory_value = 0.0
            net_inhibitory_value = 0.0
            for pre in range(n_channels):
                recurrent_value += weights[post, pre] * excitatory[pre]
                inhibitory_value += m_ie[post, pre] * inhibitory[pre]
                net_inhibitory_value += m_ei[post, pre] * excitatory[pre]
            recurrent[post] = recurrent_value
            inhibition_to_e[post] = inhibitory_value
            if net_inhibitory_value < 0.0:
                net_inhibitory_value = 0.0
            net_i[post] = net_inhibitory_value

        # Derivatives use the old state, exactly as in the source.
        for channel in range(n_channels):
            net_e = (
                tc_input[channel]
                + recurrent[channel]
                - inhibition_to_e[channel]
            )
            d_e[channel] = (
                -excitatory[channel] + net_e
            ) / parameters[_TAU_E]
            d_i[channel] = (
                -inhibitory[channel] + net_i[channel]
            ) / parameters[_TAU_I]
            if (
                flags[_STP_ENABLED] != 0
                and flags[_MULTISCALE_STD] == 0
            ):
                stimulus_value = stim[channel, time_index]
                d_u[channel] = (
                    (parameters[_U] - facilitation[channel])
                    / parameters[_TAU_F]
                    + parameters[_U]
                    * (1.0 - facilitation[channel])
                    * stimulus_value
                )
                d_x[channel] = (
                    (1.0 - resources[channel]) / parameters[_TAU_D]
                    - facilitation[channel]
                    * resources[channel]
                    * stimulus_value
                )
            d_trace[channel] = (
                -trace[channel] + excitatory[channel]
            ) / parameters[_TAU_TRACE]

        if flags[_LEARN] != 0:
            for post in range(n_channels):
                e_post = excitatory[post] / parameters[_W_NORM]
                tr_post = trace[post] / parameters[_W_NORM]
                for pre in range(n_channels):
                    e_pre = excitatory[pre] / parameters[_W_NORM]
                    tr_pre = trace[pre] / parameters[_W_NORM]
                    if flags[_BOUNDED_PLASTICITY] != 0:
                        d_weights[post, pre] = (
                            parameters[_ETA_LTP]
                            * (parameters[_W_MAX] - weights[post, pre])
                            * e_post
                            * tr_pre
                            - parameters[_ETA_LTD] * tr_post * e_pre
                            - parameters[_ETA_HET]
                            * weights[post, pre]
                            * e_pre
                            - parameters[_W_DECAY] * weights[post, pre]
                        )
                    else:
                        d_weights[post, pre] = (
                            parameters[_ETA_LTP] * e_post * tr_pre
                            - parameters[_ETA_LTD] * tr_post * e_pre
                            - parameters[_W_DECAY] * weights[post, pre]
                        )

        # Euler step.
        for channel in range(n_channels):
            excitatory[channel] += dt * d_e[channel]
            inhibitory[channel] += dt * d_i[channel]
            if flags[_STP_ENABLED] != 0:
                if flags[_MULTISCALE_STD] != 0:
                    stimulus_value = stim[channel, time_index]
                    for scale in range(n_scales):
                        d_std[channel, scale] = (
                            d_std[channel, scale] * decay_std[scale]
                            + parameters[_U_STD] * stimulus_value
                        )
                else:
                    facilitation[channel] += dt * d_u[channel]
                    resources[channel] += dt * d_x[channel]
            trace[channel] += dt * d_trace[channel]

        if flags[_LEARN] != 0:
            for post in range(n_channels):
                for pre in range(n_channels):
                    weights[post, pre] += dt * d_weights[post, pre]
                    if weights[post, pre] < 0.0:
                        weights[post, pre] = 0.0
                    elif weights[post, pre] > parameters[_W_MAX]:
                        weights[post, pre] = parameters[_W_MAX]
            if flags[_PLASTIC_SELF] == 0:
                for channel in range(n_channels):
                    weights[channel, channel] = 0.0
            else:
                for channel in range(n_channels):
                    if (
                        weights[channel, channel]
                        > parameters[_W_MAX_SELF]
                    ):
                        weights[channel, channel] = parameters[_W_MAX_SELF]

        for channel in range(n_channels):
            if (
                flags[_STP_ENABLED] != 0
                and flags[_MULTISCALE_STD] == 0
            ):
                if facilitation[channel] < 0.0:
                    facilitation[channel] = 0.0
                elif facilitation[channel] > 1.0:
                    facilitation[channel] = 1.0
                if resources[channel] < 0.0:
                    resources[channel] = 0.0
                elif resources[channel] > 1.0:
                    resources[channel] = 1.0
            if excitatory[channel] < 0.0:
                excitatory[channel] = 0.0
            excitatory_history[channel, time_index] = excitatory[channel]

    return excitatory_history, weights


def _run_fast(
    stim: np.ndarray,
    cfg: A1Config,
    *,
    learn: bool,
    session_seed: int,
    weights_initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if stim.dtype != np.float64:
        stim = np.asarray(stim, dtype=np.float64)
    if weights_initial is None:
        weights_initial = _initial_weights(cfg, session_seed)
    else:
        weights_initial = np.asarray(weights_initial, dtype=np.float64)
    packed = _pack_parameters(cfg, learn=learn)
    return _simulate_numba_exact(
        np.ascontiguousarray(stim),
        np.ascontiguousarray(weights_initial),
        *packed,
    )


def self_test_fast_simulator(
    *,
    atol: float = 2e-11,
    rtol: float = 2e-11,
) -> dict[str, Any]:
    """Validate the compiled equations against ``model0.simulate``.

    Two short cases exercise the production branches: the intact multiscale
    depression/plasticity path and the depression-off path.  Explicit,
    nonzero recurrent initial weights ensure the recurrent and learning terms
    are tested even though manuscript sessions begin from zero.
    """

    n_time = 620
    stim = np.zeros((N_CHANNELS, n_time), dtype=np.float64)
    stim[0, 10:83] = 1.0
    stim[1, 83:151] = 0.7
    stim[3, 151:239] = 1.0
    stim[2, 300:411] = 0.55
    stim[4, 411:533] = 0.9
    weights_initial = (
        np.arange(N_CHANNELS * N_CHANNELS, dtype=np.float64)
        .reshape(N_CHANNELS, N_CHANNELS)
        * 2e-5
    )

    cases: list[tuple[str, A1Config, bool]] = []
    intact, intact_learn = _condition_config("intact")
    no_depression, no_depression_learn = _condition_config("no_depression")
    cases.append(("intact_multiscale", intact, intact_learn))
    cases.append(
        ("no_depression", no_depression, no_depression_learn)
    )

    case_results: dict[str, dict[str, float]] = {}
    for name, cfg, learn in cases:
        reference = reference_simulate(
            stim,
            cfg=cfg,
            W_init=weights_initial,
            learn=learn,
            seed=123,
        )
        fast_e, fast_w = _run_fast(
            stim,
            cfg,
            learn=learn,
            session_seed=123,
            weights_initial=weights_initial,
        )
        max_abs_e = float(np.max(np.abs(fast_e - reference["E"])))
        max_abs_w = float(np.max(np.abs(fast_w - reference["W_final"])))
        np.testing.assert_allclose(
            fast_e,
            reference["E"],
            atol=atol,
            rtol=rtol,
            err_msg=f"Numba E mismatch in {name}",
        )
        np.testing.assert_allclose(
            fast_w,
            reference["W_final"],
            atol=atol,
            rtol=rtol,
            err_msg=f"Numba W mismatch in {name}",
        )
        case_results[name] = {
            "max_abs_E": max_abs_e,
            "max_abs_W": max_abs_w,
        }

    return {
        "passed": True,
        "atol": atol,
        "rtol": rtol,
        "cases": case_results,
        "reference": "model0.model.simulate",
        "fastmath": False,
    }


def _extract_deviant_channel_epochs(
    excitatory_history: np.ndarray,
    *,
    seq_starts: np.ndarray,
    seq_words: np.ndarray,
    position: int,
) -> np.ndarray:
    """Extract one channel per trial from -100 through +359 ms.

    The channel is the variable/deviant tone in that trial's three-tone word.
    Samples before the beginning of the simulated session are zero padded,
    matching ``tasks.roving.roving.evoked_per_trial``.
    """

    epoch_length = EPOCH_PRE_MS + EPOCH_POST_MS
    epochs = np.zeros(
        (seq_starts.size, epoch_length), dtype=np.float64
    )
    deviant_offset = (position - 1) * TONE_DURATION_MS
    n_time = excitatory_history.shape[1]
    for trial_index, (sequence_start, word) in enumerate(
        zip(seq_starts, seq_words)
    ):
        channel = ord(str(word)[position - 1]) - ord("A")
        epoch_start = int(sequence_start) + deviant_offset - EPOCH_PRE_MS
        epoch_stop = epoch_start + epoch_length
        source_start = max(0, epoch_start)
        source_stop = min(n_time, epoch_stop)
        if source_stop <= source_start:
            continue
        destination_start = source_start - epoch_start
        destination_stop = destination_start + source_stop - source_start
        epochs[trial_index, destination_start:destination_stop] = (
            excitatory_history[channel, source_start:source_stop]
        )
    return epochs


def _extract_sequence_deviant_channel_epochs(
    excitatory_history: np.ndarray,
    *,
    seq_starts: np.ndarray,
    seq_words: np.ndarray,
    position: int,
) -> np.ndarray:
    """Extract the variable-tone channel on the absolute 0--600 ms sequence clock.

    Unlike ``_extract_deviant_channel_epochs``, this preserves the two shared
    tones before a position-3 variable tone and the shared tones after
    position-1/2 variable tones.  It is therefore the appropriate model
    counterpoint to non-deviant-aligned ECoG decoding.
    """

    epochs = np.zeros(
        (seq_starts.size, SEQUENCE_FIGURE_SAMPLES), dtype=np.float64
    )
    n_time = excitatory_history.shape[1]
    for trial_index, (sequence_start, word) in enumerate(
        zip(seq_starts, seq_words)
    ):
        channel = ord(str(word)[position - 1]) - ord("A")
        epoch_start = int(sequence_start)
        epoch_stop = epoch_start + SEQUENCE_FIGURE_SAMPLES
        source_start = max(0, epoch_start)
        source_stop = min(n_time, epoch_stop)
        if source_stop <= source_start:
            continue
        destination_start = source_start - epoch_start
        destination_stop = destination_start + source_stop - source_start
        epochs[trial_index, destination_start:destination_stop] = (
            excitatory_history[channel, source_start:source_stop]
        )
    return epochs


def _session_pack(
    *, position: int, session_seed: int
) -> tuple[RovingConfig, dict[str, Any]]:
    cfg = RovingConfig(
        deviant_tone_pos=int(position),
        n_blocks_per_word=N_BLOCKS_PER_WORD,
        seed=int(session_seed),
    )
    rng = np.random.default_rng(int(session_seed))
    order = generate_block_order(cfg, rng)
    base_a1, _ = _condition_config("intact")
    return cfg, build_roving_stim(order, cfg, base_a1)


def _run_dataset(
    session_seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    n_conditions = len(CONDITIONS)
    n_positions = POSITIONS.size
    n_seeds = session_seeds.size
    full_time = np.arange(
        -EPOCH_PRE_MS, EPOCH_POST_MS, dtype=np.int64
    )
    n_full_time = full_time.size
    n_blocks = 3 * N_BLOCKS_PER_WORD

    responses_full = np.empty(
        (
            n_conditions,
            n_positions,
            n_seeds,
            N_REPETITIONS,
            n_full_time,
        ),
        dtype=np.float64,
    )
    response_sem_blocks_full = np.empty_like(responses_full)
    sequence_responses = np.empty(
        (
            n_conditions,
            n_positions,
            n_seeds,
            N_REPETITIONS,
            SEQUENCE_FIGURE_SAMPLES,
        ),
        dtype=np.float64,
    )
    sequence_response_sem_blocks = np.empty_like(sequence_responses)
    final_weights = np.empty(
        (
            n_conditions,
            n_positions,
            n_seeds,
            N_CHANNELS,
            N_CHANNELS,
        ),
        dtype=np.float64,
    )
    block_orders = np.empty(
        (n_positions, n_seeds, n_blocks), dtype="<U3"
    )
    trials_per_repetition = np.empty(
        (n_positions, n_seeds, N_REPETITIONS), dtype=np.int64
    )

    for position_index, position in enumerate(POSITIONS):
        for seed_index, session_seed in enumerate(session_seeds):
            roving_cfg, pack = _session_pack(
                position=int(position),
                session_seed=int(session_seed),
            )
            block_orders[position_index, seed_index] = np.asarray(
                pack["block_order"], dtype="<U3"
            )

            for condition_index, condition in enumerate(CONDITIONS):
                a1_cfg, learn = _condition_config(condition)
                excitatory_history, weights = _run_fast(
                    pack["stim"],
                    a1_cfg,
                    learn=learn,
                    session_seed=int(session_seed),
                )
                epochs = _extract_deviant_channel_epochs(
                    excitatory_history,
                    seq_starts=np.asarray(pack["seq_starts"]),
                    seq_words=np.asarray(pack["seq_word"]),
                    position=int(position),
                )
                sequence_epochs = _extract_sequence_deviant_channel_epochs(
                    excitatory_history,
                    seq_starts=np.asarray(pack["seq_starts"]),
                    seq_words=np.asarray(pack["seq_word"]),
                    position=int(position),
                )
                repetitions_zero = np.asarray(pack["seq_rep"])
                for repetition_zero in range(N_REPETITIONS):
                    selected = epochs[
                        repetitions_zero == repetition_zero
                    ]
                    selected_sequence = sequence_epochs[
                        repetitions_zero == repetition_zero
                    ]
                    if selected.shape[0] != n_blocks:
                        raise AssertionError(
                            "Every repetition must contain one trial per block"
                        )
                    responses_full[
                        condition_index,
                        position_index,
                        seed_index,
                        repetition_zero,
                    ] = np.mean(selected, axis=0)
                    response_sem_blocks_full[
                        condition_index,
                        position_index,
                        seed_index,
                        repetition_zero,
                    ] = (
                        np.std(selected, axis=0, ddof=1)
                        / np.sqrt(selected.shape[0])
                    )
                    sequence_responses[
                        condition_index,
                        position_index,
                        seed_index,
                        repetition_zero,
                    ] = np.mean(selected_sequence, axis=0)
                    sequence_response_sem_blocks[
                        condition_index,
                        position_index,
                        seed_index,
                        repetition_zero,
                    ] = (
                        np.std(selected_sequence, axis=0, ddof=1)
                        / np.sqrt(selected_sequence.shape[0])
                    )
                    trials_per_repetition[
                        position_index, seed_index, repetition_zero
                    ] = selected.shape[0]
                final_weights[
                    condition_index, position_index, seed_index
                ] = weights

    post_deviant = full_time >= 0
    responses = responses_full[..., post_deviant]
    response_sem_blocks = response_sem_blocks_full[..., post_deviant]
    time_ms = full_time[post_deviant]
    deviant_window = (time_ms >= 0) & (
        time_ms < TONE_DURATION_MS
    )
    window_response = np.mean(
        responses[..., deviant_window], axis=-1
    )
    rep_first = window_response[..., 0]
    rep_late = window_response[..., -1]
    suppression_index = np.divide(
        rep_first - rep_late,
        rep_first,
        out=np.full_like(rep_first, np.nan),
        where=rep_first > 0,
    )

    return {
        "time_ms": time_ms,
        "time_ms_full": full_time,
        "sequence_time_ms": np.arange(SEQUENCE_FIGURE_SAMPLES, dtype=np.int64),
        "repetitions": np.arange(1, N_REPETITIONS + 1, dtype=np.int64),
        "conditions": np.asarray(CONDITIONS, dtype="<U24"),
        "condition_labels": np.asarray(
            CONDITION_LABELS, dtype="<U24"
        ),
        "positions": POSITIONS.copy(),
        "seeds": session_seeds.copy(),
        # Required Figure-2 contract:
        # condition x position x session seed x repetition x time.
        "responses": responses,
        "window_response": window_response,
        # Additional audit/figure data.
        "responses_full": responses_full,
        "response_sem_blocks": response_sem_blocks,
        "response_sem_blocks_full": response_sem_blocks_full,
        "sequence_responses": sequence_responses,
        "sequence_response_sem_blocks": sequence_response_sem_blocks,
        "suppression_index": suppression_index,
        "final_weights": final_weights,
        "block_orders": block_orders,
        "trials_per_repetition": trials_per_repetition,
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        _THIS_FILE,
        _REPO_ROOT / "model0" / "model.py",
        _REPO_ROOT / "model0" / "config.py",
        _REPO_ROOT / "tasks" / "roving" / "roving.py",
        _REPO_ROOT / "tasks" / "roving" / "config.py",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot establish provenance; missing source files: {missing}"
        )
    return {str(path.resolve()): _sha256(path) for path in paths}


def _provenance_core(
    session_seeds: np.ndarray,
) -> dict[str, Any]:
    condition_configs: dict[str, Any] = {}
    for condition in CONDITIONS:
        cfg, learn = _condition_config(condition)
        condition_configs[condition] = {
            "a1_config": asdict(cfg),
            "learn": learn,
        }
    roving_configs = {
        str(position): asdict(
            RovingConfig(
                deviant_tone_pos=int(position),
                n_blocks_per_word=N_BLOCKS_PER_WORD,
                seed=int(session_seeds[0]),
            )
        )
        for position in POSITIONS
    }
    source_hashes = _source_hashes()
    source_bundle_sha256 = hashlib.sha256(
        _canonical_json(source_hashes).encode("utf-8")
    ).hexdigest()
    return {
        "analysis": "Figure 2 roving model data",
        "schema_version": 1,
        "replication_unit": (
            "simulated session/order seed; not an independently initialized "
            "network because W_init_scale is zero"
        ),
        "session_seeds": session_seeds.tolist(),
        "positions": POSITIONS.tolist(),
        "conditions": list(CONDITIONS),
        "condition_labels": list(CONDITION_LABELS),
        "n_blocks_per_word": N_BLOCKS_PER_WORD,
        "n_words": 3,
        "n_blocks_per_session": 3 * N_BLOCKS_PER_WORD,
        "n_repetitions_per_block": N_REPETITIONS,
        "analysis_epoch_ms": {
            "start_inclusive": -EPOCH_PRE_MS,
            "stop_exclusive": EPOCH_POST_MS,
            "saved_primary_start_inclusive": 0,
            "saved_primary_stop_exclusive": EPOCH_POST_MS,
            "deviant_window": [0, TONE_DURATION_MS],
            "sequence_figure_window_inclusive": [0, SEQUENCE_FIGURE_STOP_MS],
        },
        "uniform_inhibition_control": {
            "matching": "row-sum matched separately for E->I and I->E at N=5",
            "w_EI_self": 0.08,
            "w_EI_lat": 0.08,
            "w_IE_self": 0.29,
            "w_IE_lat": 0.29,
            "selective_EI_row_sum": 0.40,
            "uniform_EI_row_sum": 0.40,
            "selective_IE_row_sum": 1.45,
            "uniform_IE_row_sum": 1.45,
        },
        "condition_configs": condition_configs,
        "roving_configs_by_position": roving_configs,
        "equation_source": "model0.model.simulate",
        "accelerator": {
            "implementation": "_simulate_numba_exact",
            "numba_njit": True,
            "fastmath": False,
            "float_dtype": "float64",
        },
        "source_file_sha256": source_hashes,
        "source_bundle_sha256": source_bundle_sha256,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
        },
        "output_schema": {
            "responses": (
                "condition x position x session_seed x repetition x time; "
                "mean deviant-channel E across the 30 blocks; time 0..359 ms"
            ),
            "responses_full": (
                "same dimensions with time -100..359 ms"
            ),
            "window_response": (
                "condition x position x session_seed x repetition; "
                "mean E from 0 through 179 ms"
            ),
            "response_sem_blocks": (
                "descriptive within-session SEM over 30 blocks; not the "
                "model replication uncertainty used for inference"
            ),
            "sequence_responses": (
                "condition x position x session_seed x repetition x time; "
                "mean variable-tone-channel E across 30 blocks on the absolute "
                "sequence clock, time 0..600 ms"
            ),
            "sequence_response_sem_blocks": (
                "descriptive within-session SEM over 30 blocks for the "
                "absolute sequence-clock response"
            ),
            "suppression_index": (
                "condition x position x session_seed; "
                "(Rep1 - Rep15) / Rep1 in the 0..179 ms deviant window"
            ),
        },
    }


def _analysis_id(core: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(core).encode("utf-8")
    ).hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_csv_exports(
    data_dir: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    analysis_id: str,
) -> tuple[Path, Path, Path]:
    time_path = data_dir / TIMECOURSE_CSV_NAME
    buildup_path = data_dir / BUILDUP_CSV_NAME
    sessions_path = data_dir / SESSIONS_CSV_NAME

    conditions = arrays["conditions"]
    positions = arrays["positions"]
    seeds = arrays["seeds"]
    repetitions = arrays["repetitions"]
    time_ms = arrays["time_ms"]
    responses = arrays["responses"]

    def time_rows() -> Iterable[Sequence[Any]]:
        for condition_index, condition in enumerate(conditions):
            for position_index, position in enumerate(positions):
                for seed_index, session_seed in enumerate(seeds):
                    for repetition_index in (0, N_REPETITIONS - 1):
                        for time_index, time in enumerate(time_ms):
                            yield (
                                analysis_id,
                                str(condition),
                                int(position),
                                int(session_seed),
                                int(repetitions[repetition_index]),
                                int(time),
                                float(
                                    responses[
                                        condition_index,
                                        position_index,
                                        seed_index,
                                        repetition_index,
                                        time_index,
                                    ]
                                ),
                            )

    _atomic_csv(
        time_path,
        (
            "analysis_id",
            "condition",
            "deviant_position",
            "session_seed",
            "repetition",
            "time_ms_from_deviant",
            "deviant_channel_E",
        ),
        time_rows(),
    )

    window_response = arrays["window_response"]

    def buildup_rows() -> Iterable[Sequence[Any]]:
        for condition_index, condition in enumerate(conditions):
            for position_index, position in enumerate(positions):
                for seed_index, session_seed in enumerate(seeds):
                    for repetition_index, repetition in enumerate(
                        repetitions
                    ):
                        yield (
                            analysis_id,
                            str(condition),
                            int(position),
                            int(session_seed),
                            int(repetition),
                            float(
                                window_response[
                                    condition_index,
                                    position_index,
                                    seed_index,
                                    repetition_index,
                                ]
                            ),
                        )

    _atomic_csv(
        buildup_path,
        (
            "analysis_id",
            "condition",
            "deviant_position",
            "session_seed",
            "repetition",
            "mean_E_0_180_ms",
        ),
        buildup_rows(),
    )

    block_orders = arrays["block_orders"]

    def session_rows() -> Iterable[Sequence[Any]]:
        for position_index, position in enumerate(positions):
            for seed_index, session_seed in enumerate(seeds):
                for block_index, word in enumerate(
                    block_orders[position_index, seed_index]
                ):
                    yield (
                        analysis_id,
                        int(position),
                        int(session_seed),
                        block_index + 1,
                        str(word),
                    )

    _atomic_csv(
        sessions_path,
        (
            "analysis_id",
            "deviant_position",
            "session_seed",
            "block",
            "word",
        ),
        session_rows(),
    )
    return time_path, buildup_path, sessions_path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _cache_matches(
    npz_path: Path,
    provenance_path: Path,
    *,
    expected_analysis_id: str,
) -> bool:
    if not npz_path.exists() or not provenance_path.exists():
        return False
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("analysis_id") != expected_analysis_id:
            return False
        with np.load(npz_path, allow_pickle=False) as archive:
            cached_id = str(np.asarray(archive["analysis_id"]).item())
        if cached_id != expected_analysis_id:
            return False
        expected_npz_hash = (
            provenance.get("output_file_sha256", {}).get(npz_path.name)
        )
        if expected_npz_hash and _sha256(npz_path) != expected_npz_hash:
            return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def load_or_build_model_data(
    force: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load or generate the complete Figure-2 model data contract.

    Parameters
    ----------
    force
        Recompute even when the fixed-name cache has a matching analysis ID
        and output hash.
    data_dir
        Destination directory.  Defaults to ``figure_2/data`` beside this
        module.

    Returns
    -------
    dict[str, numpy.ndarray]
        Includes the required keys ``time_ms``, ``repetitions``,
        ``conditions``, ``positions``, ``seeds``, ``responses`` with shape
        condition × position × seed × repetition × time, and
        ``window_response`` with the same dimensions minus time.
    """

    destination = (
        DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)
    ).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    session_seeds = DEFAULT_SESSION_SEEDS.copy()

    # Refuse to build or trust a cache if the accelerated equations no longer
    # match the source model in the current software environment.
    self_test = self_test_fast_simulator()
    core = _provenance_core(session_seeds)
    analysis_id = _analysis_id(core)
    npz_path = destination / NPZ_NAME
    provenance_path = destination / PROVENANCE_NAME

    if (
        not force
        and _cache_matches(
            npz_path,
            provenance_path,
            expected_analysis_id=analysis_id,
        )
    ):
        return _load_npz(npz_path)

    arrays = _run_dataset(session_seeds)
    provenance: dict[str, Any] = {
        **core,
        "analysis_id": analysis_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "self_test": self_test,
        "outputs": {
            "npz": str(npz_path),
            "provenance": str(provenance_path),
            "timecourses_csv": str(destination / TIMECOURSE_CSV_NAME),
            "buildup_csv": str(destination / BUILDUP_CSV_NAME),
            "sessions_csv": str(destination / SESSIONS_CSV_NAME),
        },
    }
    arrays["analysis_id"] = np.asarray(analysis_id)
    arrays["metadata_json"] = np.asarray(
        _canonical_json(provenance)
    )
    _atomic_npz(npz_path, arrays)
    csv_paths = _write_csv_exports(
        destination, arrays, analysis_id=analysis_id
    )

    output_paths = (npz_path, *csv_paths)
    provenance["output_file_sha256"] = {
        path.name: _sha256(path) for path in output_paths
    }
    provenance["output_file_bytes"] = {
        path.name: path.stat().st_size for path in output_paths
    }
    _atomic_json(provenance_path, provenance)
    return arrays


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the provenance-preserving Figure-2 model dataset."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="validate the accelerated equations without running sessions",
    )
    args = parser.parse_args(argv)
    if args.self_test_only:
        print(json.dumps(self_test_fast_simulator(), indent=2))
        return 0

    arrays = load_or_build_model_data(
        force=args.force, data_dir=args.data_dir
    )
    print(
        f"Saved/loaded {NPZ_NAME}: responses {arrays['responses'].shape}; "
        f"analysis {str(arrays['analysis_id'].item())[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
