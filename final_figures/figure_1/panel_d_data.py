"""Simulate and cache the real ``model0`` traces behind Figure 1, panel D.

Panel D is the figure's reading key, and it is measured rather than drawn.  It
shows one canonical two-tone sequence in a network that has already learned the
dependency between its two channels, so the recurrent current genuinely
pre-activates the upcoming channel instead of illustrating that it would.

Protocol
--------
The AB/BA oddball of ``tasks/ab_ba_model0/ab_ba.py``: 400 sequences, 90% AB and
10% BA in shuffled order, tone 50 ms, intra-sequence gap 30 ms, inter-sequence
gap 500 ms, two delta-tuned channels.  The task's own configuration overrides
are reused verbatim (``w_IE_self=3.0``, ``w_EI_self=0.40``, ``W_norm=4.0``);
they exist because the MMN-polarity test needs ``W[B<-A]`` near its asymptote
within a 400-sequence horizon, and Figure 1 must show the same network the AB/BA
results are computed in, not a differently tuned one.

What is plotted is the mean over the **AB sequences in the second half of the
run**, by which point ``W[B<-A]`` has converged.  Averaging over sequences of one
type is the only reduction applied; no trace is smoothed, rescaled or idealised.

The timescale separation the panel exists to show is not asserted from the
configuration.  ``tau_E`` and ``tau_I`` are read back out of the traces as the
measured 1/e fall time from each signal's own peak, and those measured values
are what the panel prints.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model0 import A1Config, simulate  # noqa: E402
from tasks.ab_ba_model0.ab_ba import (  # noqa: E402
    build_stim,
    evoked_per_trial,
    shuffled_codes,
)


#: Reused verbatim from ``tasks.ab_ba_model0.ab_ba.main``.  See the module
#: docstring: the panel must portray the network the AB/BA results come from.
AB_BA_OVERRIDES: dict[str, float] = dict(
    w_IE_self=3.0, w_EI_self=0.40, W_norm=4.0
)

N_TRIALS = 400
P_AB = 0.90
SEED = 1
CH_A, CH_B = 0, 1

TONE_DUR = 50e-3
INTRA_GAP = 30e-3
INTER_GAP = 500e-3

#: Milliseconds of the sequence kept for the panel.  Long enough that the
#: inhibitory tail is visibly still running after the excitation that caused it
#: has returned to zero, short enough that the two tones are not a detail.
WINDOW_MS = 340.0

#: A trace is treated as back at baseline below this fraction of its own peak.
BASELINE_FRACTION = 0.01

DATA_VERSION = "figure1-panelD-1"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
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


def _config() -> A1Config:
    return dataclasses.replace(A1Config(), **AB_BA_OVERRIDES)


def _settings() -> dict[str, Any]:
    cfg = _config()
    return {
        "data_version": DATA_VERSION,
        "paradigm": "ab_ba_model0",
        "n_trials": N_TRIALS,
        "p_AB": P_AB,
        "seed": SEED,
        "ch_A": CH_A,
        "ch_B": CH_B,
        "tone_dur_s": TONE_DUR,
        "intra_gap_s": INTRA_GAP,
        "inter_gap_s": INTER_GAP,
        "window_ms": WINDOW_MS,
        "baseline_fraction": BASELINE_FRACTION,
        "config": dataclasses.asdict(cfg),
        "config_overrides": AB_BA_OVERRIDES,
    }


def _one_over_e_fall_ms(trace: np.ndarray, dt: float,
                        search_from: int = 0) -> tuple[float, float, float]:
    """Measured 1/e fall time of a trace, from its own peak.

    Returns ``(peak_ms, peak_value, fall_ms)``.  This is how the panel obtains
    the two time constants it prints: they are read out of the simulated traces
    rather than copied from the configuration, so the annotation is a
    measurement of the figure's own data.
    """

    segment = trace[search_from:]
    peak_index = int(np.argmax(segment)) + search_from
    peak_value = float(trace[peak_index])
    if peak_value <= 0:
        return float("nan"), 0.0, float("nan")
    threshold = peak_value / np.e
    after = np.arange(peak_index, trace.size)
    below = after[trace[after] < threshold]
    if below.size == 0:
        return peak_index * dt * 1e3, peak_value, float("nan")
    return (peak_index * dt * 1e3, peak_value,
            float((below[0] - peak_index) * dt * 1e3))


def build_panel_d_data(*, force: bool = False,
                       data_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Run (or load) the AB/BA session and reduce it to the panel-D traces."""

    data_dir = Path(data_dir or (_THIS_FILE.parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "panel_d_traces.npz"
    provenance_path = data_dir / "panel_d_provenance.json"

    settings = _settings()
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("settings_hash") == settings_hash:
            with np.load(npz_path, allow_pickle=True) as handle:
                return {key: handle[key] for key in handle.files}

    cfg = _config()
    dt = cfg.dt
    print(f"[figure 1] simulating {N_TRIALS} AB/BA sequences "
          f"({P_AB:.0%} AB) for panel D")

    rng = np.random.default_rng(SEED)
    codes = np.array(shuffled_codes(N_TRIALS, P_AB, rng))
    stim, starts, n_seq = build_stim(
        list(codes), cfg, CH_A, CH_B,
        tone_dur=TONE_DUR, intra_gap=INTRA_GAP, inter_gap=INTER_GAP,
    )
    out = simulate(stim, cfg=cfg, seed=SEED)

    # The learned link needs time to converge, so the panel reports the
    # second half of the run.  Selecting on sequence type is the only
    # selection applied; every AB sequence in that half is averaged.
    half = len(codes) // 2
    late = np.zeros(len(codes), dtype=bool)
    late[half:] = True
    keep = late & (codes == "AB")
    n_kept = int(keep.sum())

    n_window = int(round(WINDOW_MS * 1e-3 / dt))
    arrays: dict[str, np.ndarray] = {}
    for name in ("E", "I", "tm_in", "rec_E", "inh_to_E"):
        epochs = evoked_per_trial(out[name], starts, n_seq)
        arrays[name] = epochs[keep].mean(axis=0)[:, :n_window]
    epochs = evoked_per_trial(stim, starts, n_seq)
    arrays["stim"] = epochs[keep].mean(axis=0)[:, :n_window]

    weights = out["W_final"]
    arrays["W_final"] = weights
    arrays["time_ms"] = np.arange(n_window) * dt * 1e3
    arrays["tone_a_ms"] = np.array([0.0, TONE_DUR * 1e3])
    arrays["tone_b_ms"] = np.array(
        [(TONE_DUR + INTRA_GAP) * 1e3, (2 * TONE_DUR + INTRA_GAP) * 1e3])
    arrays["n_sequences_averaged"] = np.array([n_kept])
    arrays["dt"] = np.array([dt])

    # ---- measured timescales, read back out of the traces ----
    excitation = arrays["E"][CH_A]
    inhibition = arrays["inh_to_E"][CH_B]
    e_peak_ms, e_peak, e_fall_ms = _one_over_e_fall_ms(excitation, dt)
    # The inhibitory current onto B is driven twice: once by the predictive
    # pre-activation during tone A, and again by tone B itself.  Its free decay
    # is only interpretable after the second drive ends, so the search starts
    # at tone-B offset.
    tone_b_off = int(round((2 * TONE_DUR + INTRA_GAP) / dt))
    i_peak_ms, i_peak, i_fall_ms = _one_over_e_fall_ms(
        inhibition, dt, search_from=tone_b_off - int(round(0.02 / dt)))

    tone_b_on = int(round((TONE_DUR + INTRA_GAP) / dt))
    baseline = BASELINE_FRACTION * e_peak
    active = np.flatnonzero(excitation > baseline)
    excitation_off_ms = float(active[-1] * dt * 1e3) if active.size else 0.0

    arrays["tau_E_measured_ms"] = np.array([e_fall_ms])
    arrays["tau_I_measured_ms"] = np.array([i_fall_ms])
    arrays["tau_E_config_ms"] = np.array([cfg.tau_E * 1e3])
    arrays["tau_I_config_ms"] = np.array([cfg.tau_I * 1e3])
    arrays["excitation_peak_ms"] = np.array([e_peak_ms])
    arrays["inhibition_peak_ms"] = np.array([i_peak_ms])
    arrays["excitation_off_ms"] = np.array([excitation_off_ms])
    arrays["inh_at_tone_b_onset"] = np.array([float(inhibition[tone_b_on])])
    arrays["inh_peak"] = np.array([i_peak])
    arrays["exc_at_tone_b_onset"] = np.array([float(excitation[tone_b_on])])
    arrays["exc_peak"] = np.array([e_peak])
    arrays["W_BA"] = np.array([float(weights[CH_B, CH_A])])
    arrays["W_AB"] = np.array([float(weights[CH_A, CH_B])])

    print(f"    averaged {n_kept} late AB sequences")
    print(f"    W[B<-A] = {weights[CH_B, CH_A]:.3f}   "
          f"W[A<-B] = {weights[CH_A, CH_B]:.3f}")
    print(f"    measured 1/e fall: excitation {e_fall_ms:.0f} ms "
          f"(tau_E = {cfg.tau_E * 1e3:.0f} ms), inhibition {i_fall_ms:.0f} ms "
          f"(tau_I = {cfg.tau_I * 1e3:.0f} ms)")
    print(f"    at tone-B onset: excitation {excitation[tone_b_on]:.3f}, "
          f"inhibition {inhibition[tone_b_on]:.3f} "
          f"({inhibition[tone_b_on] / i_peak:.0%} of its own peak)")

    _atomic_npz(npz_path, arrays)
    _atomic_json(provenance_path, {
        "figure": "Figure 1 panel D - measured reading key",
        "settings": settings,
        "settings_hash": settings_hash,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paradigm_source": (
            "tasks/ab_ba_model0/ab_ba.py - 400 sequences, 90% AB / 10% BA in "
            "shuffled order, 50 ms tones, 30 ms intra-sequence gap, 500 ms "
            "inter-sequence gap, two delta-tuned channels."
        ),
        "reduction": (
            f"Mean over the {n_kept} AB sequences in the second half of the "
            "run, by which point W[B<-A] has converged. No smoothing, "
            "rescaling or idealisation is applied to any trace."
        ),
        "measured": {
            "W_BA": float(weights[CH_B, CH_A]),
            "W_AB": float(weights[CH_A, CH_B]),
            "tau_E_measured_ms": e_fall_ms,
            "tau_I_measured_ms": i_fall_ms,
            "tau_E_config_ms": cfg.tau_E * 1e3,
            "tau_I_config_ms": cfg.tau_I * 1e3,
            "excitation_off_ms": excitation_off_ms,
            "excitation_at_tone_b_onset": float(excitation[tone_b_on]),
            "inhibition_at_tone_b_onset": float(inhibition[tone_b_on]),
            "inhibition_peak": i_peak,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "generator": str(_THIS_FILE),
        "generator_sha256": _sha256(_THIS_FILE),
        "npz_sha256": _sha256(npz_path),
    })
    return {key: np.asarray(value) for key, value in arrays.items()}


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Repeat the simulation even if the cache is valid.")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    build_panel_d_data(force=_parse_args(arguments).force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
