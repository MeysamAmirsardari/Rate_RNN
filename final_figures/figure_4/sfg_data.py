"""Generate the model-side data for manuscript Figure 4.

Stochastic figure-ground (SFG), the paradigm of experiment II in Lu, Dutta,
Mohammed, Elhilali & Shamma (2025) *iScience* 28:111991.  A presentation is
5 s of random tone cloud (*pre*), 5 s in which a set of 4, 6, 8 or 10 tones is
made temporally coherent (*figure*), and 5 s of cloud again (*post*).  The
cloud spans 37 channels over three octaves and the figure chords recur at
about 4 per second.  ``tasks/sfg2`` holds the faithful generator; this module
only drives it, reduces it, and caches the result.

The readout: tone-triggered, and matched on drive
-------------------------------------------------
Every quantity below is a **tone-triggered average** -- the mean over the 50 ms
that follow each of a channel's *own* tone onsets.  This is what makes the
comparison legitimate.  A channel is being driven by a 50 ms pip in the cloud
epochs exactly as it is during a figure chord, so the thalamic input entering
the average is the same in both, and any difference is cortical.

An epoch mean would not do this: figure channels are silent for most of the
figure epoch, so their epoch mean is mostly baseline.  Nor would a chord window
measured against a whole-epoch baseline, which compares "driven" against "mostly
silent" and reports the stimulus rather than the response.

The causal contrast: plasticity on versus off
---------------------------------------------
The reported measure throughout is

    modulation (%) = 100 * (E_plastic - E_frozen) / E_frozen

evaluated on matched runs -- **the same seed, hence the identical stimulus**,
with recurrent plasticity either learning or held at ``W = 0``.  The two runs
see the same pips at the same times, so short-term depression, adaptation drift
and epoch transients cancel exactly and the residual is attributable to the
learned recurrent weights alone.  ``tm`` is identical between the two runs by
construction, and the module asserts it.

Quantities
----------
**Connectivity.**  The learned recurrent matrix grouped into figure-figure,
ground-ground and cross blocks.  The quantity that matters functionally is not
mean pairwise ``W_FF`` but the **assembly drive** ``(n_fig - 1) * W_FF``, the
total recurrent excitation one figure channel receives from its fellows; only
that grows with figure size.

**Response dynamics.**  Modulation in 500 ms bins across the whole
pre / figure / post presentation, for the figure and the ground channel sets --
the model's analogue of the response time courses in the paper's Figures 5A
and 6C.

**Size dependence.**  Figure-epoch modulation against figure size, the model's
analogue of the paper's Figure 4C.

**Buildup.**  Figure-epoch modulation against presentation number.  The paper's
binding builds within one 5 s figure; this model's rate-STDP is slower, so its
buildup lives on the across-presentation axis instead.  Reported on the axis
where the model actually has it, and the difference stated.

**Persistence.**  Modulation during the pre- and post-figure tone clouds, when
the coherent chord is absent.  This is the model's counterpart of the paper's
``delta_post-pre`` and it behaves differently: see ``README.md``.

**Mechanism.**  Tone-triggered recurrent, inhibitory and net currents under the
same plastic-minus-frozen contrast.

Conditions
----------
``selective`` / ``uniform``  inhibition structure, both recalibrated for
                             N = 37 in ``tasks.sfg2.sfg2.sfg2_config``.
``plastic`` / ``frozen``     recurrent plasticity on, or W held at zero.  Both
                             are run for both inhibition structures, so each
                             structure has its own matched control.

The replication unit is the **session seed**, which redraws the ground cloud and
the figure channel set.  Eight seeds are run.  The figure is frozen within a
session and redrawn across seeds, matching the experiment's design.
"""

from __future__ import annotations

import argparse
import csv
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

from model0 import simulate  # noqa: E402
from tasks.sfg2.sfg2 import compute_W_groups, sfg2_config  # noqa: E402
from tasks.sfg2.stimulus import (  # noqa: E402
    N_CHANNELS,
    TONE_DUR_MS,
    build_session,
)


FIGURE_SIZES: tuple[int, ...] = (4, 6, 8, 10)
PRESETS: tuple[str, ...] = ("selective", "uniform")
PLASTICITY: tuple[str, ...] = ("plastic", "frozen")

#: Every inhibition structure is run with and without plasticity, so each has
#: its own stimulus-matched control.
CONDITIONS: tuple[tuple[str, str], ...] = tuple(
    (preset, plasticity) for preset in PRESETS for plasticity in PLASTICITY
)
REFERENCE_PRESET = "selective"

N_SEEDS = 8
N_REPS = 30
#: Presentations averaged for every steady-state measure, taken from the end of
#: the session so that the weights have settled.
LATE_REPS = 10

#: Time-course resolution.  30 bins span the 15 s presentation: 10 pre,
#: 10 figure, 10 post.
BIN_MS = 500
N_BINS = 30
FIGURE_BINS = slice(10, 20)
CLOUD_BINS = np.r_[0:10, 20:30]

CURRENTS: tuple[str, ...] = ("tm", "rec", "inh", "net")
GROUPS: tuple[str, ...] = ("fig", "gnd")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tone-triggered averaging
# ---------------------------------------------------------------------------
def tone_onsets(stimulus_row: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Onset samples of one channel's pips that fall wholly inside [lo, hi).

    Only rising edges count.  Taking every non-zero difference would also
    collect the tone *offsets* and average over the decay instead of the
    response.
    """

    active = (stimulus_row[lo:hi] > 0).astype(np.int8)
    onsets = np.flatnonzero(np.diff(np.r_[np.int8(0), active]) == 1) + lo
    return onsets[onsets + TONE_DUR_MS <= hi]


def _accumulate(totals: dict[str, float], traces: Mapping[str, np.ndarray],
                stimulus: np.ndarray, channels: np.ndarray,
                lo: int, hi: int) -> int:
    """Add the pip-weighted tone-triggered sums over a channel set.

    Returns the number of pips added, so the caller can pool across
    presentations before dividing.  Pooling first matters: the figure is
    frozen within a session, so a 500 ms bin that happens to contain no chord
    onset contains none in any presentation, and dividing per presentation
    would turn that into a NaN that then propagates.
    """

    count = 0
    for channel in channels:
        onsets = tone_onsets(stimulus[channel], lo, hi)
        if onsets.size == 0:
            continue
        window = onsets[:, None] + np.arange(TONE_DUR_MS)[None, :]
        for name, trace in traces.items():
            totals[name] += float(trace[channel][window].sum())
        count += onsets.size * TONE_DUR_MS
    return count


def _pooled(traces: Mapping[str, np.ndarray], stimulus: np.ndarray,
            channels: np.ndarray, lo: int, hi: int) -> dict[str, float]:
    """Pip-weighted tone-triggered mean of each trace over a channel set."""

    totals = {name: 0.0 for name in traces}
    count = _accumulate(totals, traces, stimulus, channels, lo, hi)
    if count == 0:
        return {name: np.nan for name in traces}
    return {name: totals[name] / count for name in traces}


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
def run_one(n_fig: int, preset: str, plasticity: str, seed: int, *,
            keep_detail: bool = False) -> dict[str, Any]:
    """Run one SFG session and reduce it to tone-triggered quantities."""

    config = sfg2_config(preset)
    stimulus, session = build_session(n_fig, N_REPS, base_seed=seed,
                                      fig_seed=12345 + seed, with_silence=True)
    out = simulate(
        stimulus,
        cfg=config,
        W_init=np.zeros((config.N, config.N)) if plasticity == "frozen" else None,
        learn=(plasticity == "plastic"),
        record_W_every=max(1, int(round(0.25 / config.dt))) if keep_detail else 0,
        seed=7 + seed,
    )

    figure_index = np.asarray(session["fig_idx"])
    ground_index = np.asarray(session["gnd_idx"])
    index_of = {"fig": figure_index, "gnd": ground_index}

    traces = {
        "E": out["E"],
        "tm": out["tm_in"],
        "rec": out["rec_E"],
        "inh": out["inh_to_E"],
    }
    traces["net"] = traces["tm"] + traces["rec"] - traces["inh"]

    w_ff, w_gg, w_fg = (float(v) for v in
                        compute_W_groups(out["W_final"], figure_index, ground_index))

    # Time course: 500 ms bins spanning the whole presentation, pooled over the
    # late presentations before dividing, so bins are weighted by how many pips
    # they actually contain.
    names = list(traces)
    course = {group: np.zeros((N_BINS, len(names))) for group in GROUPS}
    samples = {group: np.zeros(N_BINS, dtype=int) for group in GROUPS}
    for rep in session["reps"][-LATE_REPS:]:
        start = int(rep["bounds"]["pre"][0])
        for b in range(N_BINS):
            lo, hi = start + b * BIN_MS, start + (b + 1) * BIN_MS
            for group in GROUPS:
                totals = {name: 0.0 for name in names}
                samples[group][b] += _accumulate(
                    totals, traces, stimulus, index_of[group], lo, hi)
                course[group][b] += np.array([totals[n] for n in names])
    for group in GROUPS:
        empty = samples[group] == 0
        course[group] /= np.where(empty, 1, samples[group])[:, None]
        course[group][empty] = np.nan

    # Buildup: figure-epoch tone-triggered E, one value per presentation.
    buildup = {group: np.zeros(N_REPS) for group in GROUPS}
    for r, rep in enumerate(session["reps"]):
        lo, hi = (int(v) for v in rep["bounds"]["figure"])
        for group in GROUPS:
            buildup[group][r] = _pooled({"E": traces["E"]}, stimulus,
                                        index_of[group], lo, hi)["E"]

    result: dict[str, Any] = dict(
        n_fig=n_fig, preset=preset, plasticity=plasticity, seed=seed,
        w_ff=w_ff, w_gg=w_gg, w_fg=w_fg,
        w_ff_over_gg=w_ff / w_gg if w_gg > 0 else np.nan,
        assembly_drive=(n_fig - 1) * w_ff,
    )
    for group in GROUPS:
        result[f"course_{group}"] = course[group]        # (N_BINS, n_traces)
        result[f"samples_{group}"] = samples[group].astype(float)
        result[f"buildup_{group}"] = buildup[group]      # (N_REPS,)
        for j, name in enumerate(names):
            result[f"{name}_{group}_figure"] = float(
                np.nanmean(course[group][FIGURE_BINS, j]))
            result[f"{name}_{group}_cloud"] = float(
                np.nanmean(course[group][CLOUD_BINS, j]))
    result["_trace_names"] = names

    if keep_detail:
        trajectory = np.array([compute_W_groups(w, figure_index, ground_index)
                               for w in out["W_traj"]])
        result["_detail"] = dict(
            out=out, session=session, stimulus=stimulus,
            figure_index=figure_index, ground_index=ground_index,
            trajectory=trajectory, w_t=np.asarray(out["W_t"]),
        )
    return result


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
SCALAR_FIELDS = ("seed", "w_ff", "w_gg", "w_fg", "w_ff_over_gg",
                 "assembly_drive") + tuple(
    f"{name}_{group}_{window}"
    for name in ("E",) + CURRENTS
    for group in GROUPS
    for window in ("figure", "cloud"))

ARRAY_FIELDS = tuple(f"course_{g}" for g in GROUPS) + \
    tuple(f"buildup_{g}" for g in GROUPS) + \
    tuple(f"samples_{g}" for g in GROUPS)

READOUT_VERSION = 3


def _settings() -> dict[str, Any]:
    return dict(
        figure_sizes=list(FIGURE_SIZES), conditions=[list(c) for c in CONDITIONS],
        n_seeds=N_SEEDS, n_reps=N_REPS, late_reps=LATE_REPS,
        n_channels=int(N_CHANNELS), bin_ms=BIN_MS,
        tone_ms=int(TONE_DUR_MS), readout_version=READOUT_VERSION,
    )


def build_sfg_data(*, force: bool = False,
                   data_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Run (or load) the SFG factorial and write the figure data contract."""

    data_dir = Path(data_dir or (_THIS_FILE.parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "sfg_figure4_data.npz"
    provenance_path = data_dir / "sfg_provenance.json"

    settings = _settings()
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("settings_hash") == settings_hash:
            with np.load(npz_path, allow_pickle=True) as handle:
                return {key: handle[key] for key in handle.files}

    total = len(CONDITIONS) * len(FIGURE_SIZES) * N_SEEDS
    print(f"[figure 4] simulating {total} SFG sessions "
          f"({N_REPS} presentations each)")

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for preset, plasticity in CONDITIONS:
        for n_fig in FIGURE_SIZES:
            subset = [run_one(n_fig, preset, plasticity, seed)
                      for seed in range(N_SEEDS)]
            rows.extend(subset)
            key = f"{preset}|{plasticity}|{n_fig}"
            for field in SCALAR_FIELDS:
                arrays[f"{key}|{field}"] = np.array(
                    [r[field] for r in subset], dtype=float)
            for field in ARRAY_FIELDS:
                arrays[f"{key}|{field}"] = np.stack(
                    [np.asarray(r[field], dtype=float) for r in subset])
            print(f"    {preset:<10s} {plasticity:<8s} size {n_fig:>2d}  "
                  f"W_FF/W_GG {np.mean([r['w_ff_over_gg'] for r in subset]):.2f}  "
                  f"assembly {np.mean([r['assembly_drive'] for r in subset]):.4f}  "
                  f"E_fig|figure {np.mean([r['E_fig_figure'] for r in subset]):.4f}")

    arrays["trace_names"] = np.array(rows[0]["_trace_names"])
    _atomic_npz(npz_path, arrays)
    _write_csv(data_dir, rows)
    _atomic_json(provenance_path, {
        "figure": "Figure 4 - stochastic figure-ground",
        "settings": settings,
        "settings_hash": settings_hash,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paradigm_source": (
            "tasks/sfg2 - faithful SFG generator: 37 channels over three "
            "octaves, 5 s pre / 5 s figure / 5 s post, figure chords at ~4/s, "
            "figure frozen within a session and redrawn across seeds. "
            "Rate-matched: figure channels carry only the coherent chord and "
            "ground channels an independent cloud at the same mean rate, so "
            "the two sets differ in temporal coherence and nothing else."
        ),
        "replication_unit": (
            "Session seed: redraws the ground cloud and the figure channel set."
        ),
        "measure_note": (
            "All quantities are tone-triggered: the mean over the 50 ms "
            "following each of a channel's own pip onsets. The reported "
            "modulation is 100*(plastic-frozen)/frozen on runs sharing a seed "
            "and therefore an identical stimulus, so short-term depression and "
            "epoch transients cancel and only the learned recurrent weights "
            "remain."
        ),
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__},
    })
    with np.load(npz_path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def _write_csv(data_dir: Path, rows) -> None:
    path = data_dir / "sfg_figure4_runs.csv"
    fields = ("preset", "plasticity", "n_fig") + SCALAR_FIELDS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Derived contrasts
# ---------------------------------------------------------------------------
def modulation(data: Mapping[str, np.ndarray], preset: str, n_fig: int,
               field: str) -> np.ndarray:
    """Per-seed plasticity-attributable modulation, in per cent.

    ``plastic`` and ``frozen`` share a seed and therefore an identical
    stimulus, so the ratio is taken seed by seed.
    """

    plastic = np.asarray(data[f"{preset}|plastic|{n_fig}|{field}"], dtype=float)
    frozen = np.asarray(data[f"{preset}|frozen|{n_fig}|{field}"], dtype=float)
    # The frozen recurrent current is identically zero, so a ratio is undefined
    # for that column of a course array.  Callers take the E column; use
    # ``difference`` for the currents.
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (plastic - frozen) / frozen


def difference(data: Mapping[str, np.ndarray], preset: str, n_fig: int,
               field: str) -> np.ndarray:
    """Per-seed plastic-minus-frozen difference in the field's own units.

    Used for the currents, where the frozen recurrent term is exactly zero and
    a ratio would be undefined.
    """

    plastic = np.asarray(data[f"{preset}|plastic|{n_fig}|{field}"], dtype=float)
    frozen = np.asarray(data[f"{preset}|frozen|{n_fig}|{field}"], dtype=float)
    return plastic - frozen


def mean_sem(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM over ``axis``, counting only the finite entries.

    A 500 ms bin containing no chord onset is NaN for that seed, so the count
    has to come from the data rather than from the array shape.
    """

    values = np.asarray(values, dtype=float)
    n = np.sum(np.isfinite(values), axis=axis)
    with np.errstate(invalid="ignore"):
        return (np.nanmean(values, axis=axis),
                np.nanstd(values, axis=axis, ddof=1) /
                np.sqrt(np.maximum(n, 1)))


# ---------------------------------------------------------------------------
# Exemplar
# ---------------------------------------------------------------------------
EXEMPLAR_VERSION = 2
EXEMPLAR_SIZE = 10


def build_exemplar(*, force: bool = False,
                   data_dir: Path | None = None) -> dict[str, np.ndarray]:
    """One illustrative session: the stimulus and the learned matrix."""

    data_dir = Path(data_dir or (_THIS_FILE.parent / "data"))
    npz_path = data_dir / "sfg_exemplar.npz"
    provenance_path = data_dir / "sfg_exemplar_provenance.json"
    key = hashlib.sha256(json.dumps(
        {"settings": _settings(), "version": EXEMPLAR_VERSION,
         "size": EXEMPLAR_SIZE}, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("exemplar_hash") == key:
            with np.load(npz_path, allow_pickle=True) as handle:
                return {name: handle[name] for name in handle.files}

    print("[figure 4] simulating the exemplar SFG session")
    result = run_one(EXEMPLAR_SIZE, REFERENCE_PRESET, "plastic", 0,
                     keep_detail=True)
    detail = result["_detail"]
    out, session = detail["out"], detail["session"]

    rep = session["reps"][-1]
    lo = int(rep["bounds"]["pre"][0])
    hi = int(rep["bounds"]["post"][1])
    exemplar = {
        "stim": np.asarray(detail["stimulus"][:, lo:hi], dtype=float),
        "epoch_bounds": np.array([
            [rep["bounds"][name][0] - lo, rep["bounds"][name][1] - lo]
            for name in ("pre", "figure", "post")], dtype=float),
        "figure_index": detail["figure_index"].astype(float),
        "ground_index": detail["ground_index"].astype(float),
        "W_final": np.asarray(out["W_final"], dtype=float),
        "trajectory": detail["trajectory"],
        "w_t": detail["w_t"],
        "dt": np.array([out["cfg"].dt]),
        "n_fig": np.array([EXEMPLAR_SIZE]),
    }
    _atomic_npz(npz_path, exemplar)
    _atomic_json(provenance_path, {
        "exemplar_hash": key, "exemplar_version": EXEMPLAR_VERSION,
        "figure_size": EXEMPLAR_SIZE,
        "condition": [REFERENCE_PRESET, "plastic"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    })
    return exemplar


def load_all(*, force: bool = False,
             data_dir: Path | None = None) -> dict[str, np.ndarray]:
    factorial = build_sfg_data(force=force, data_dir=data_dir)
    return factorial | build_exemplar(force=force, data_dir=data_dir)


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Repeat every simulated SFG session.")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    load_all(force=_parse_args(arguments).force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
