"""Generate the frozen-core perturbation data for manuscript Figure 7.

The synthesis figure is deliberately stricter than the task-specific figures.
Every layer-1 paradigm is rerun with the same pre-existing directional core
(``w_EI_self=.40, w_IE_self=3.0, W_norm=4``); only the number of stimulus
channels changes.  No parameter is selected from the Figure-7 results.

The four primary effects are prespecified and remain in their native units:

* roving: repetition-suppression index in the 0--180 ms variable-tone window;
* AB/BA: held-out same-sequence unexpected-minus-predicted target response;
* SFG: final ``W_FF - W_GG`` for the prespecified ten-tone figure;
* Saffran: fraction of four words held completely in temporal order.

For display only, each lesion is divided by that task's intact mean.  Native
effects are retained in the cache and all inference is paired within task.
The downstream single-rate lesion is meaningful only for Saffran; the other
three cells are encoded as not applicable, never imputed as zero or one.

The ABA_ sweep is a falsification test.  The current one-hot, distance-
independent circuit makes every nonzero channel separation a permutation of
every other one.  We therefore report the learned A--B coupling over rate and
nominal channel separation, and verify the predicted numerical invariance.
It is not labelled a perceptual streaming boundary.
"""

from __future__ import annotations

import csv
import dataclasses as dc
import gc
import hashlib
import itertools
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy import stats


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_figures.figure_2.model_data import (  # noqa: E402
    _extract_deviant_channel_epochs,
    _run_fast,
    _session_pack,
)
from final_figures.figure_5 import saffran_data as saffran  # noqa: E402
from model0 import A1Config  # noqa: E402
from model0.model import simulate as reference_simulate  # noqa: E402
from tasks.ab_ba_model0.ab_ba import (  # noqa: E402
    build_stim as build_ab_stim,
    shuffled_codes,
)
from tasks.sfg2.sfg2 import compute_W_groups  # noqa: E402
from tasks.sfg2.stimulus import (  # noqa: E402
    N_CHANNELS as SFG_CHANNELS,
    build_session as build_sfg_session,
)


TASKS: tuple[str, ...] = ("roving", "ab_ba", "sfg", "saffran")
TASK_LABELS: tuple[str, ...] = (
    "Roving novelty",
    "Directional prediction",
    "Coherent dependency",
    "Sequence composition",
)
METRIC_LABELS: tuple[str, ...] = (
    "suppression index",
    "unexpected - predicted rate",
    "W_FF - W_GG",
    "fraction of words held",
)

CONDITIONS: tuple[str, ...] = (
    "intact",
    "no_depression",
    "no_recurrent_learning",
    "uniform_inhibition",
    "single_rate_readout",
)
CONDITION_LABELS: tuple[str, ...] = (
    "Intact",
    "- depression",
    "- recurrent learning",
    "Uniform inhibition",
    "Single-rate readout",
)
LESIONS: tuple[str, ...] = CONDITIONS[1:]

N_SEEDS = 12
ROVING_SEEDS = np.array(
    [11, 23, 37, 41, 53, 67, 79, 97, 101, 109, 127, 139], dtype=int
)
GENERIC_SEEDS = np.arange(N_SEEDS, dtype=int)

SFG_FIGURE_SIZE = 10
SFG_REPETITIONS = 30
AB_TRAIN_TRIALS = 400
AB_TEST_TRIALS = 120
AB_TARGET_WINDOW_MS = (80, 130)
AB_LEGACY_TIMING = dict(tone_dur=0.050, intra_gap=0.030, inter_gap=0.500)

EQUIVALENCE_BOUND = 0.20
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 72026

ABA_RATES_HZ = np.array([4.0, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25, 12.5])
ABA_SEPARATIONS = np.arange(1, 9, dtype=int)
ABA_CYCLES = 80
ABA_TONE_MS = 50

DEFAULT_DATA_DIR = _THIS_FILE.parent / "data"
NPZ_NAME = "figure_7_frozen_core_data.npz"
RUNS_NAME = "figure_7_runs.csv"
SUMMARY_NAME = "figure_7_summary.csv"
INFERENCE_NAME = "figure_7_inference.csv"
PROVENANCE_NAME = "figure_7_data_provenance.json"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def core_config(n_channels: int) -> A1Config:
    """The single pre-existing directional core used by every paradigm."""

    return A1Config(
        N=int(n_channels),
        w_EI_self=0.40,
        w_IE_self=3.0,
        W_norm=4.0,
    )


def condition_config(n_channels: int, condition: str) -> tuple[A1Config, bool]:
    """Return a pure lesion of the frozen core.

    Uniform inhibition separately preserves the E->I and I->E row sums; it
    changes spatial selectivity without changing either arm's total weight.
    """

    if condition not in CONDITIONS[:4]:
        raise ValueError(f"Layer-1 condition not defined: {condition}")
    intact = core_config(n_channels)
    cfg = dc.replace(intact)
    learn = condition != "no_recurrent_learning"
    if condition == "no_depression":
        cfg.stp_enabled = False
    elif condition == "uniform_inhibition":
        ei_row = intact.w_EI_self + (n_channels - 1) * intact.w_EI_lat
        ie_row = intact.w_IE_self + (n_channels - 1) * intact.w_IE_lat
        cfg.w_EI_self = cfg.w_EI_lat = ei_row / n_channels
        cfg.w_IE_self = cfg.w_IE_lat = ie_row / n_channels
        if not np.isclose(
            cfg.w_EI_self + (n_channels - 1) * cfg.w_EI_lat,
            ei_row,
        ):
            raise AssertionError("E->I row sum changed in uniform lesion")
        if not np.isclose(
            cfg.w_IE_self + (n_channels - 1) * cfg.w_IE_lat,
            ie_row,
        ):
            raise AssertionError("I->E row sum changed in uniform lesion")
    return cfg, learn


def self_test_core() -> dict[str, Any]:
    """Verify the accelerated equations against the source model for this core."""

    cfg = core_config(5)
    stimulus = np.zeros((5, 740), dtype=float)
    stimulus[0, 20:100] = 1.0
    stimulus[3, 150:250] = 0.7
    stimulus[1, 320:500] = 0.55
    initial = np.arange(25, dtype=float).reshape(5, 5) * 1e-5
    reference = reference_simulate(
        stimulus, cfg=cfg, W_init=initial, learn=True, seed=91
    )
    fast_e, fast_w = _run_fast(
        stimulus,
        cfg,
        learn=True,
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


# ---------------------------------------------------------------------------
# Roving
# ---------------------------------------------------------------------------
def run_roving() -> np.ndarray:
    """condition x paired order seed; positions are averaged within seed."""

    values = np.empty((4, N_SEEDS), dtype=float)
    by_position = np.empty((4, 3, N_SEEDS), dtype=float)
    for position_index, position in enumerate((1, 2, 3)):
        for seed_index, seed in enumerate(ROVING_SEEDS):
            _, pack = _session_pack(position=position, session_seed=int(seed))
            repetitions = np.asarray(pack["seq_rep"], dtype=int)
            for condition_index, condition in enumerate(CONDITIONS[:4]):
                cfg, learn = condition_config(5, condition)
                excitatory, _ = _run_fast(
                    pack["stim"], cfg, learn=learn, session_seed=int(seed)
                )
                epochs = _extract_deviant_channel_epochs(
                    excitatory,
                    seq_starts=np.asarray(pack["seq_starts"]),
                    seq_words=np.asarray(pack["seq_word"]),
                    position=position,
                )
                # Epoch index 100 is variable-tone onset; response window is
                # prespecified as 0--180 ms, matching Figure 2.
                response = epochs[:, 100:280].mean(axis=1)
                by_rep = np.array(
                    [response[repetitions == rep].mean() for rep in range(15)]
                )
                by_position[condition_index, position_index, seed_index] = (
                    (by_rep[0] - by_rep[-1]) / by_rep[0]
                )
                del excitatory, epochs
            gc.collect()
    values[:] = by_position.mean(axis=1)
    return values


# ---------------------------------------------------------------------------
# Directional AB/BA
# ---------------------------------------------------------------------------
def _ab_response(
    excitatory: np.ndarray,
    starts: np.ndarray,
    codes: Sequence[str],
) -> dict[str, float]:
    lo, hi = AB_TARGET_WINDOW_MS
    labels = np.asarray(codes)
    out: dict[str, float] = {}
    for code, channel in (("AB", 1), ("BA", 0)):
        selected = np.flatnonzero(labels == code)
        out[code] = float(
            np.mean(
                [
                    excitatory[channel, int(starts[k]) + lo : int(starts[k]) + hi].mean()
                    for k in selected
                ]
            )
        )
    return out


def run_ab_ba() -> np.ndarray:
    """Held-out same-sequence contextual target effect, condition x seed."""

    values = np.empty((4, N_SEEDS), dtype=float)
    for seed_index, seed in enumerate(GENERIC_SEEDS):
        train_codes = {
            "AB_context": shuffled_codes(
                AB_TRAIN_TRIALS, 0.90, np.random.default_rng(int(seed) + 1000)
            ),
            "BA_context": shuffled_codes(
                AB_TRAIN_TRIALS, 0.10, np.random.default_rng(int(seed) + 2000)
            ),
        }
        test_codes = shuffled_codes(
            AB_TEST_TRIALS, 0.50, np.random.default_rng(int(seed) + 3000)
        )
        for condition_index, condition in enumerate(CONDITIONS[:4]):
            cfg, learn = condition_config(2, condition)
            learned: dict[str, np.ndarray] = {}
            for context, codes in train_codes.items():
                stimulus, _, _ = build_ab_stim(
                    codes, cfg, 0, 1, **AB_LEGACY_TIMING
                )
                _, learned[context] = _run_fast(
                    stimulus, cfg, learn=learn, session_seed=int(seed)
                )
                del stimulus

            test_stimulus, starts, _ = build_ab_stim(
                test_codes, cfg, 0, 1, **AB_LEGACY_TIMING
            )
            test_response: dict[str, dict[str, float]] = {}
            for context, weights in learned.items():
                excitatory, _ = _run_fast(
                    test_stimulus,
                    cfg,
                    learn=False,
                    session_seed=int(seed),
                    weights_initial=weights,
                )
                test_response[context] = _ab_response(
                    excitatory, starts, test_codes
                )
                del excitatory

            # Same physical AB and BA sequences compared across learned
            # probability contexts. Positive means unexpected > predicted.
            ab_effect = (
                test_response["BA_context"]["AB"]
                - test_response["AB_context"]["AB"]
            )
            ba_effect = (
                test_response["AB_context"]["BA"]
                - test_response["BA_context"]["BA"]
            )
            values[condition_index, seed_index] = 0.5 * (ab_effect + ba_effect)
            del test_stimulus, learned
        gc.collect()
    return values


# ---------------------------------------------------------------------------
# Stochastic figure-ground
# ---------------------------------------------------------------------------
def run_sfg() -> np.ndarray:
    """Final figure-ground connectivity contrast, condition x cloud seed."""

    values = np.empty((4, N_SEEDS), dtype=float)
    for seed_index, seed in enumerate(GENERIC_SEEDS):
        stimulus, session = build_sfg_session(
            SFG_FIGURE_SIZE,
            SFG_REPETITIONS,
            base_seed=int(seed),
            fig_seed=12345 + int(seed),
            with_silence=True,
            rate_matched=True,
        )
        figure_index = np.asarray(session["fig_idx"], dtype=int)
        ground_index = np.asarray(session["gnd_idx"], dtype=int)
        for condition_index, condition in enumerate(CONDITIONS[:4]):
            if condition == "no_recurrent_learning":
                values[condition_index, seed_index] = 0.0
                continue
            cfg, learn = condition_config(SFG_CHANNELS, condition)
            excitatory, weights = _run_fast(
                stimulus, cfg, learn=learn, session_seed=int(seed) + 7
            )
            w_ff, w_gg, _ = compute_W_groups(
                weights, figure_index, ground_index
            )
            values[condition_index, seed_index] = float(w_ff - w_gg)
            del excitatory, weights
            gc.collect()
        del stimulus
    return values


# ---------------------------------------------------------------------------
# Saffran-style sequence composition
# ---------------------------------------------------------------------------
def _saffran_words_held(seed: int, condition: str) -> float:
    readout = "single_rate" if condition == "single_rate_readout" else "multi_rate"
    layer_condition = "intact" if condition == "single_rate_readout" else condition
    cfg, learn = condition_config(saffran.N_CHANNELS, layer_condition)
    stream = saffran._structured_stream(cfg.dt, int(seed))
    excitatory, _ = _run_fast(
        stream["stim"], cfg, learn=learn, session_seed=int(seed)
    )
    layer, _, taus = saffran._make_readout(readout, int(seed))
    layer.run(excitatory, cfg.dt, learn=True)
    committed = np.flatnonzero(layer.committed)
    unit_chains = [
        saffran.span_chain(layer, int(unit), taus) for unit in committed
    ]
    words = {entry["word"] for entry in unit_chains if entry["spans"]}
    del excitatory, layer, stream
    gc.collect()
    return len(words) / len(saffran.WORDS)


def run_saffran() -> np.ndarray:
    values = np.empty((5, N_SEEDS), dtype=float)
    for condition_index, condition in enumerate(CONDITIONS):
        for seed_index, seed in enumerate(GENERIC_SEEDS):
            values[condition_index, seed_index] = _saffran_words_held(
                int(seed), condition
            )
    return values


# ---------------------------------------------------------------------------
# ABA_ no-retuning symmetry test
# ---------------------------------------------------------------------------
def _aba_stimulus(
    cfg: A1Config,
    *,
    channel_a: int,
    channel_b: int,
    rate_hz: float,
) -> np.ndarray:
    slot_steps = int(round(1.0 / (rate_hz * cfg.dt)))
    tone_steps = min(int(round(ABA_TONE_MS / 1000.0 / cfg.dt)), slot_steps)
    cycle_steps = 4 * slot_steps
    stimulus = np.zeros((cfg.N, ABA_CYCLES * cycle_steps), dtype=float)
    for cycle in range(ABA_CYCLES):
        start = cycle * cycle_steps
        stimulus[channel_a, start : start + tone_steps] = 1.0
        stimulus[
            channel_b,
            start + slot_steps : start + slot_steps + tone_steps,
        ] = 1.0
        stimulus[
            channel_a,
            start + 2 * slot_steps : start + 2 * slot_steps + tone_steps,
        ] = 1.0
    return stimulus


def run_aba_sweep() -> np.ndarray:
    """Learned cross/self coupling; rows rate, columns nominal separation."""

    cfg = core_config(12)
    coupling = np.empty((ABA_RATES_HZ.size, ABA_SEPARATIONS.size), dtype=float)
    for rate_index, rate_hz in enumerate(ABA_RATES_HZ):
        for sep_index, separation in enumerate(ABA_SEPARATIONS):
            channel_a = 1
            channel_b = channel_a + int(separation)
            stimulus = _aba_stimulus(
                cfg,
                channel_a=channel_a,
                channel_b=channel_b,
                rate_hz=float(rate_hz),
            )
            _, weights = _run_fast(
                stimulus, cfg, learn=True, session_seed=0
            )
            cross = weights[channel_a, channel_b] + weights[channel_b, channel_a]
            within = weights[channel_a, channel_a] + weights[channel_b, channel_b]
            coupling[rate_index, sep_index] = cross / within
            del stimulus
    return coupling


# ---------------------------------------------------------------------------
# Inference and display contract
# ---------------------------------------------------------------------------
def _holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    n_values = values.size
    for rank, index in enumerate(order):
        candidate = (n_values - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def _studentized(values: np.ndarray) -> float:
    sem = np.std(values, ddof=1) / np.sqrt(values.size)
    if sem == 0:
        return np.inf if np.mean(values) != 0 else 0.0
    return float(np.mean(values) / sem)


def _max_t_paired(
    normalized: np.ndarray, intact_normalized: np.ndarray
) -> np.ndarray:
    """Two-sided exact max-T p-values across lesions within one task."""

    differences = normalized - intact_normalized[None, :]
    observed = np.array([abs(_studentized(row)) for row in differences])
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=N_SEEDS)))
    null_max = np.empty(signs.shape[0], dtype=float)
    for permutation_index, sign in enumerate(signs):
        permuted = differences * sign[None, :]
        null_max[permutation_index] = max(
            abs(_studentized(row)) for row in permuted
        )
    return np.array(
        [
            (1.0 + np.sum(null_max >= statistic - 1e-14))
            / (1.0 + null_max.size)
            for statistic in observed
        ]
    )


def _tost_p(
    normalized: np.ndarray, intact_normalized: np.ndarray
) -> np.ndarray:
    """Paired TOST p-values for retention within 1 +/- the SESOI."""

    p_values: list[float] = []
    for row in normalized:
        difference = row - intact_normalized
        mean = float(np.mean(difference))
        sem = float(np.std(difference, ddof=1) / np.sqrt(difference.size))
        if sem == 0.0:
            p_values.append(0.0 if abs(mean) < EQUIVALENCE_BOUND else 1.0)
            continue
        lower_t = (mean + EQUIVALENCE_BOUND) / sem
        upper_t = (mean - EQUIVALENCE_BOUND) / sem
        p_lower = float(stats.t.sf(lower_t, difference.size - 1))
        p_upper = float(stats.t.cdf(upper_t, difference.size - 1))
        p_values.append(max(p_lower, p_upper))
    return _holm(np.asarray(p_values))


def _bootstrap_ratio_ci(intact: np.ndarray, lesion: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, intact.size, size=(BOOTSTRAP_DRAWS, intact.size))
    denominator = intact[indices].mean(axis=1)
    numerator = lesion[indices].mean(axis=1)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-15,
    )
    return tuple(float(value) for value in np.nanpercentile(ratio, [2.5, 97.5]))


def analyse(raw_effect: np.ndarray) -> dict[str, np.ndarray]:
    n_tasks = len(TASKS)
    n_lesions = len(LESIONS)
    retention = np.full((n_tasks, n_lesions, N_SEEDS), np.nan)
    mean = np.full((n_tasks, n_lesions), np.nan)
    ci = np.full((n_tasks, n_lesions, 2), np.nan)
    change_q = np.full((n_tasks, n_lesions), np.nan)
    equivalence_q = np.full((n_tasks, n_lesions), np.nan)
    status = np.full((n_tasks, n_lesions), "not_applicable", dtype="<U20")

    for task_index in range(n_tasks):
        intact = raw_effect[task_index, 0]
        intact_mean = float(np.mean(intact))
        applicable: list[int] = []
        for lesion_index in range(n_lesions):
            condition_index = lesion_index + 1
            lesion = raw_effect[task_index, condition_index]
            if np.isnan(lesion).all():
                continue
            applicable.append(lesion_index)
            retention[task_index, lesion_index] = lesion / intact_mean
            mean[task_index, lesion_index] = float(np.mean(lesion) / intact_mean)
            ci[task_index, lesion_index] = _bootstrap_ratio_ci(intact, lesion)

        if not applicable:
            continue
        normalized = retention[task_index, applicable]
        intact_normalized = intact / intact_mean
        q_change = _max_t_paired(normalized, intact_normalized)
        q_equivalence = _tost_p(normalized, intact_normalized)
        for local_index, lesion_index in enumerate(applicable):
            change_q[task_index, lesion_index] = q_change[local_index]
            equivalence_q[task_index, lesion_index] = q_equivalence[local_index]
            deviation = abs(mean[task_index, lesion_index] - 1.0)
            if q_equivalence[local_index] < 0.05:
                status[task_index, lesion_index] = "equivalent"
            elif q_change[local_index] < 0.05 and deviation >= EQUIVALENCE_BOUND:
                status[task_index, lesion_index] = "changed"
            else:
                status[task_index, lesion_index] = "inconclusive"

    return {
        "retention": retention,
        "retention_mean": mean,
        "retention_ci": ci,
        "ablation_effect": 1.0 - mean,
        "change_q": change_q,
        "equivalence_q": equivalence_q,
        "status": status,
    }


def _settings() -> dict[str, Any]:
    core = asdict(core_config(2))
    core.pop("N")
    return {
        "schema_version": 1,
        "tasks": list(TASKS),
        "conditions": list(CONDITIONS),
        "n_seeds": N_SEEDS,
        "roving_seeds": ROVING_SEEDS.tolist(),
        "generic_seeds": GENERIC_SEEDS.tolist(),
        "frozen_core_excluding_input_dimension": core,
        "sfg_figure_size": SFG_FIGURE_SIZE,
        "sfg_repetitions": SFG_REPETITIONS,
        "ab_train_trials": AB_TRAIN_TRIALS,
        "ab_test_trials": AB_TEST_TRIALS,
        "ab_target_window_ms": list(AB_TARGET_WINDOW_MS),
        "ab_legacy_timing_seconds": dict(AB_LEGACY_TIMING),
        "equivalence_bound_retention": EQUIVALENCE_BOUND,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "aba_rates_hz": ABA_RATES_HZ.tolist(),
        "aba_separations_channels": ABA_SEPARATIONS.tolist(),
        "aba_cycles": ABA_CYCLES,
        "aba_tone_ms": ABA_TONE_MS,
    }


def _source_hashes() -> dict[str, str]:
    paths = [
        _THIS_FILE,
        _REPO_ROOT / "model0" / "config.py",
        _REPO_ROOT / "model0" / "model.py",
        _REPO_ROOT / "final_figures" / "figure_2" / "model_data.py",
        _REPO_ROOT / "tasks" / "roving" / "roving.py",
        _REPO_ROOT / "tasks" / "ab_ba_model0" / "ab_ba.py",
        _REPO_ROOT / "tasks" / "sfg2" / "stimulus.py",
        _REPO_ROOT / "final_figures" / "figure_5" / "saffran_data.py",
        _REPO_ROOT / "layer2_multirate" / "layer2.py",
        _REPO_ROOT / "layer2_syllable" / "layer2.py",
    ]
    return {str(path.relative_to(_REPO_ROOT)): _sha256(path) for path in paths}


def _write_exports(
    data_dir: Path,
    raw_effect: np.ndarray,
    analysis: Mapping[str, np.ndarray],
) -> None:
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        for condition_index, condition in enumerate(CONDITIONS):
            for seed_index in range(N_SEEDS):
                value = raw_effect[task_index, condition_index, seed_index]
                if np.isnan(value):
                    continue
                run_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "seed_index": seed_index,
                        "seed": (
                            int(ROVING_SEEDS[seed_index])
                            if task == "roving"
                            else int(GENERIC_SEEDS[seed_index])
                        ),
                        "native_effect": float(value),
                        "metric": METRIC_LABELS[task_index],
                    }
                )
            values = raw_effect[task_index, condition_index]
            if np.isnan(values).all():
                continue
            summary_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "n_seeds": int(np.isfinite(values).sum()),
                    "native_mean": float(np.nanmean(values)),
                    "native_sem": float(
                        np.nanstd(values, ddof=1) / np.sqrt(np.isfinite(values).sum())
                    ),
                    "metric": METRIC_LABELS[task_index],
                }
            )
        for lesion_index, lesion in enumerate(LESIONS):
            if np.isnan(analysis["retention_mean"][task_index, lesion_index]):
                continue
            inference_rows.append(
                {
                    "task": task,
                    "lesion": lesion,
                    "retention_mean": float(
                        analysis["retention_mean"][task_index, lesion_index]
                    ),
                    "retention_ci95_low": float(
                        analysis["retention_ci"][task_index, lesion_index, 0]
                    ),
                    "retention_ci95_high": float(
                        analysis["retention_ci"][task_index, lesion_index, 1]
                    ),
                    "ablation_effect_1_minus_retention": float(
                        analysis["ablation_effect"][task_index, lesion_index]
                    ),
                    "exact_within_task_maxT_q": float(
                        analysis["change_q"][task_index, lesion_index]
                    ),
                    "holm_TOST_q": float(
                        analysis["equivalence_q"][task_index, lesion_index]
                    ),
                    "status": str(analysis["status"][task_index, lesion_index]),
                }
            )

    _atomic_csv(
        data_dir / RUNS_NAME,
        run_rows,
        ("task", "condition", "seed_index", "seed", "native_effect", "metric"),
    )
    _atomic_csv(
        data_dir / SUMMARY_NAME,
        summary_rows,
        ("task", "condition", "n_seeds", "native_mean", "native_sem", "metric"),
    )
    _atomic_csv(
        data_dir / INFERENCE_NAME,
        inference_rows,
        (
            "task",
            "lesion",
            "retention_mean",
            "retention_ci95_low",
            "retention_ci95_high",
            "ablation_effect_1_minus_retention",
            "exact_within_task_maxT_q",
            "holm_TOST_q",
            "status",
        ),
    )


def build_figure7_data(
    *, force: bool = False, data_dir: Path | None = None
) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / NPZ_NAME
    provenance_path = data_dir / PROVENANCE_NAME
    settings = _settings()
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]

    if npz_path.exists() and provenance_path.exists() and not force:
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("settings_hash") == settings_hash:
            with np.load(npz_path, allow_pickle=False) as handle:
                return {key: handle[key] for key in handle.files}

    self_test = self_test_core()
    raw_effect = np.full((len(TASKS), len(CONDITIONS), N_SEEDS), np.nan)

    print("[Figure 7] frozen core: roving")
    raw_effect[0, :4] = run_roving()
    print("[Figure 7] frozen core: held-out AB/BA")
    raw_effect[1, :4] = run_ab_ba()
    print("[Figure 7] frozen core: SFG connectivity")
    raw_effect[2, :4] = run_sfg()
    print("[Figure 7] frozen core: Saffran composition")
    raw_effect[3, :] = run_saffran()
    print("[Figure 7] no-retuning ABA_ symmetry sweep")
    aba_coupling = run_aba_sweep()

    analysis = analyse(raw_effect)
    arrays: dict[str, np.ndarray] = {
        "tasks": np.asarray(TASKS),
        "task_labels": np.asarray(TASK_LABELS),
        "metric_labels": np.asarray(METRIC_LABELS),
        "conditions": np.asarray(CONDITIONS),
        "condition_labels": np.asarray(CONDITION_LABELS),
        "lesions": np.asarray(LESIONS),
        "roving_seeds": ROVING_SEEDS.copy(),
        "generic_seeds": GENERIC_SEEDS.copy(),
        "raw_effect": raw_effect,
        "aba_rates_hz": ABA_RATES_HZ.copy(),
        "aba_separations": ABA_SEPARATIONS.copy(),
        "aba_coupling": aba_coupling,
        **analysis,
    }
    _atomic_npz(npz_path, arrays)
    _write_exports(data_dir, raw_effect, analysis)

    source_hashes = _source_hashes()
    _atomic_json(
        provenance_path,
        {
            "figure": "Figure 7 - frozen-core causal synthesis",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "settings": settings,
            "settings_hash": settings_hash,
            "source_hashes": source_hashes,
            "self_test": self_test,
            "replication_units": {
                "roving": "paired constrained block-order session seed; positions averaged within seed",
                "ab_ba": "paired training/test order seed; identical balanced test stream across probability contexts and lesions",
                "sfg": "paired cloud/onset session seed; fixed ten figure channels",
                "saffran": "paired exposure-order and downstream-initialization seed",
            },
            "inference": {
                "change": "two-sided exact paired max-T sign-flip test, corrected within each prespecified task family",
                "equivalence": "paired TOST within retention 1 +/- 0.20, Holm-corrected within task",
                "uncertainty": "20,000 paired bootstrap resamples of ratio of condition mean to intact mean",
                "cross_task_inference": False,
            },
            "aba_interpretation": (
                "The sweep is a model falsification, not a perceptual result. "
                "With one-hot inputs and distance-independent connectivity, "
                "all nonzero channel separations are permutation-equivalent."
            ),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "output": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        },
    )
    return arrays


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    build_figure7_data(force=arguments.force)
