"""Simulator output for the AB/BA circuit-state snapshots.

Every number the figure draws comes from here, and this module does nothing but
run ``model0`` on the committed AB/BA paradigm and slice the result.  No value
is idealised, smoothed, or chosen by hand.

The contrast is **identity controlled**.  Both rows of the figure show the same
physical stimulus -- an A tone then a B tone, 50 ms each with a 30 ms gap -- and
the same two channels.  The only difference is the context that preceded it:

``frequent``   AB was 90% of trials, so the A->B transition was learned.
``rare``       AB was 10% of trials, so it was not.

Anything that differs between the rows is therefore attributable to the learned
weight, not to tone identity, adaptation, or run drift.

State drawn at each snapshot
----------------------------
``E`` (2,)      excitatory rate of channels A and B
``I`` (2,)      inhibitory rate of the two channel-matched interneurons
``W`` (2, 2)    plastic recurrent E->E weights, row = post, column = pre
``M_EI``        fixed E->I matrix, rebuilt exactly as ``model0.model`` does
``M_IE``        fixed I->E matrix, likewise

Only ``W`` differs between conditions and it is the only quantity mapped to line
width; the fixed matrices are drawn at constant width and quoted numerically.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model0 import INH_PRESETS  # noqa: E402
from tasks.ab_ba_model0.ab_ba import (  # noqa: E402
    evoked_per_trial,
    run_experiment,
)


#: The committed AB/BA regime.  ``tasks/ab_ba_model0/ab_ba.py`` documents why
#: this task overrides the shared defaults: the predictive cascade needs
#: W[B<-A] near its asymptote within the 400-trial horizon (W_norm = 4) and a
#: strong I->E delivery arm (w_IE_self = 3.0).
OVERRIDES = dict(w_IE_self=3.0, w_EI_self=0.40, W_norm=4.0)

N_TRIALS = 400
LATE_FROM = 300           # trials averaged for the steady state
CH_A, CH_B = 0, 1

TONE_MS = 50
GAP_MS = 30
INTER_GAP_MS = 500
LEGACY_TIMING = dict(
    tone_dur=TONE_MS / 1000.0,
    intra_gap=GAP_MS / 1000.0,
    inter_gap=INTER_GAP_MS / 1000.0,
)

#: Snapshot times in ms from sequence onset, and what each one is for.
SNAPSHOTS: tuple[tuple[int, str, str], ...] = (
    (10, "Tone A onset", "A is driven; nothing has been predicted yet"),
    (45, "Tone A, late", "the learned link pre-activates B"),
    (65, "Silent gap", "B's interneuron outlives the excitation that drove it"),
    (105, "Tone B", "the target arrives into standing inhibition"),
)

CONDITIONS: tuple[tuple[str, float, int, str], ...] = (
    ("frequent", 0.90, 1, "AB frequent (90%) — A→B learned"),
    ("rare", 0.10, 2, "AB rare (10%) — A→B not learned"),
)

#: Independent seed pairs used for the effect-size panel.  The suppression is
#: small by design -- model0 is calibrated to the ~5% mismatch polarity reported
#: in A1 -- so its reliability, not its magnitude, is what the figure must show.
N_SEED_PAIRS = 6

DATA_DIR = _THIS_FILE.parent / "data"


def _config():
    return dc.replace(INH_PRESETS["selective"](), **OVERRIDES)


def _inhibitory_matrices(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the fixed E->I and I->E matrices exactly as model0 does."""

    n = cfg.N
    ones, eye = np.ones((n, n)), np.eye(n)
    m_ei = cfg.w_EI_lat * ones + (cfg.w_EI_self - cfg.w_EI_lat) * eye
    m_ie = cfg.w_IE_lat * ones + (cfg.w_IE_self - cfg.w_IE_lat) * eye
    return m_ei, m_ie


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build(*, force: bool = False, data_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Run both contexts and cache the traces the figure needs."""

    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "ab_ba_circuit.npz"
    provenance_path = data_dir / "ab_ba_circuit_provenance.json"

    cfg = _config()
    settings: dict[str, Any] = dict(
        overrides=OVERRIDES, n_trials=N_TRIALS, late_from=LATE_FROM,
        tone_ms=TONE_MS, gap_ms=GAP_MS, inter_gap_ms=INTER_GAP_MS,
        timing=LEGACY_TIMING,
        snapshots=[s[0] for s in SNAPSHOTS], n_seed_pairs=N_SEED_PAIRS,
        conditions=[[c[0], c[1], c[2]] for c in CONDITIONS],
        config={k: v for k, v in vars(cfg).items()},
    )
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("settings_hash") == settings_hash:
            with np.load(npz_path, allow_pickle=True) as handle:
                return {key: handle[key] for key in handle.files}

    print("[figure 3] simulating both AB/BA contexts")
    m_ei, m_ie = _inhibitory_matrices(cfg)
    arrays: dict[str, np.ndarray] = {"M_EI": m_ei, "M_IE": m_ie}

    for name, p_ab, seed, _label in CONDITIONS:
        result = run_experiment(
            p_AB=p_ab, n_trials=N_TRIALS, seed=seed, cfg=cfg,
            ch_A=CH_A, ch_B=CH_B, timing=LEGACY_TIMING,
        )
        excitatory = evoked_per_trial(result["E"], result["seq_starts"],
                                      result["n_seq"])
        inhibitory = evoked_per_trial(result["I"], result["seq_starts"],
                                      result["n_seq"])
        codes = result["codes"]
        # The SAME physical sequence in both contexts: AB trials only.
        keep = (np.arange(len(codes)) >= LATE_FROM) & (codes == "AB")
        arrays[f"{name}|E"] = excitatory[keep].mean(axis=0)      # (2, n_seq)
        arrays[f"{name}|I"] = inhibitory[keep].mean(axis=0)
        arrays[f"{name}|W"] = np.asarray(result["W_final"], dtype=float)
        arrays[f"{name}|n_trials"] = np.array([int(keep.sum())], dtype=float)
        print(f"    {name:<9s} n={int(keep.sum()):3d} AB trials   "
              f"W[B<-A] {result['W_final'][1, 0]:.4f}  "
              f"W[A<-B] {result['W_final'][0, 1]:.4f}")

    # Effect size across independent seed pairs.  Same measurement as the
    # exemplar, repeated; nothing here is fitted or selected.
    tone_b = slice(TONE_MS + GAP_MS, 2 * TONE_MS + GAP_MS)
    suppression = np.empty(N_SEED_PAIRS)
    for index in range(N_SEED_PAIRS):
        peaks = []
        for p_ab, seed in ((0.90, 100 + index), (0.10, 200 + index)):
            result = run_experiment(
                p_AB=p_ab, n_trials=N_TRIALS, seed=seed,
                cfg=cfg, ch_A=CH_A, ch_B=CH_B, timing=LEGACY_TIMING,
            )
            excitatory = evoked_per_trial(result["E"], result["seq_starts"],
                                          result["n_seq"])
            codes = result["codes"]
            keep = (np.arange(len(codes)) >= LATE_FROM) & (codes == "AB")
            peaks.append(float(excitatory[keep].mean(axis=0)[1, tone_b].max()))
        suppression[index] = 100.0 * (1.0 - peaks[0] / peaks[1])
    arrays["suppression_pct"] = suppression
    print(f"    suppression {suppression.mean():.3f}% ± "
          f"{suppression.std(ddof=1) / np.sqrt(N_SEED_PAIRS):.3f} SEM "
          f"over {N_SEED_PAIRS} seed pairs")

    arrays["snapshot_ms"] = np.array([s[0] for s in SNAPSHOTS], dtype=float)
    _atomic_npz(npz_path, arrays)
    provenance_path.write_text(json.dumps({
        "figure": "Supplementary Figure 3 - AB/BA circuit-state snapshots",
        "settings": settings,
        "settings_hash": settings_hash,
        "design": (
            "Identity-controlled: both conditions plot the same physical AB "
            "sequence on the same two channels. Only the preceding context, "
            "and hence the learned W[B<-A], differs."
        ),
        "averaging": (
            f"Mean over AB trials from trial {LATE_FROM} onward, so the "
            "recurrent weights have settled."
        ),
    }, indent=2, sort_keys=True) + "\n")

    with np.load(npz_path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


if __name__ == "__main__":
    build(force=True)
