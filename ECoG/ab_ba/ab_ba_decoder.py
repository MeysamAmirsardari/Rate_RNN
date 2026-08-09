"""Cross-session, cross-acoustic decoding of deviant versus standard.

Why not decode within a physical sequence
-----------------------------------------
For one physical sequence every deviant presentation comes from one recording
and every standard presentation from the other. A classifier trained on that
split can separate the classes using any constant difference between the two
recordings -- gain, impedance, drift -- and would score high while learning
nothing about deviance. That is the design the legacy MATLAB decoder used, and
it is not salvageable by better cross-validation.

The design that does work
-------------------------
The two recordings are counterbalanced, so *within* a recording the deviant and
the standard are two different physical sequences, and **which sequence is the
deviant reverses between recordings**:

    2026-04-30   deviant 5300-9400        standard 9400-5300
    2026-05-01   deviant 9400-5300        standard 5300-9400

Train on one recording with deviance labels, test on the other with deviance
labels. Then

* a decoder that learned the **acoustics** scores *below* chance on the test
  set, because the acoustic-to-role mapping is inverted there;
* a decoder that learned a **session** feature scores at chance, because that
  feature is constant across both classes within the test recording and
  carries no discriminative information into it;
* only a genuine **deviance** representation scores above chance.

Both directions are run and averaged, which is the symmetrisation step in the
prespecified recovery analysis.

Everything that could leak is fitted inside the training recording only:
standardisation, the ridge solution, and the class weighting. The null shuffles
deviance labels within each test block, so block composition and the 85/15 class
ratio are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ECoG.ab_ba.ab_ba_inference import EXPERIMENT, load_session
from ECoG.ab_ba.baphy_events import RUN_TO_RECORDING

RESULT_DIR = Path(__file__).resolve().parent / "results" / "ab_ba_decoder"

#: Decoding grid, in milliseconds, matching the displayed sequence window.
GRID_MS = np.arange(-100.0, 600.0, 5.0)
RIDGE_LAMBDA = 1e-2
N_PERMUTATIONS = 999
RANDOM_SEED = 20_260_809
ALPHA = 0.05
CLUSTER_ALPHA = 0.05


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        mask = truth == label
        if mask.any():
            scores.append(float((predicted[mask] == label).mean()))
    return float(np.mean(scores)) if scores else 0.5


def _features(session, grid_index: np.ndarray) -> np.ndarray:
    """Sequence x channel x grid, the decoder's design matrix per latency."""

    return session.epochs[:, :, grid_index]


def decode_direction(train, test, grid_index: np.ndarray,
                     test_labels: np.ndarray | None = None) -> np.ndarray:
    """Train on one recording, score balanced accuracy on the other."""

    train_x = _features(train, grid_index)
    test_x = _features(test, grid_index)
    train_y = train.is_deviant.astype(int)
    test_y = (test.is_deviant.astype(int) if test_labels is None
              else test_labels.astype(int))

    accuracy = np.empty(grid_index.size)
    for index in range(grid_index.size):
        a = train_x[:, :, index]
        b = test_x[:, :, index]
        centre, scale = a.mean(axis=0), a.std(axis=0, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        model = LogisticRegression(
            C=1.0 / RIDGE_LAMBDA, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=400)
        model.fit((a - centre) / scale, train_y)
        accuracy[index] = balanced_accuracy(
            test_y, model.predict((b - centre) / scale))
    return accuracy


def _shuffle_within_blocks(labels: np.ndarray, blocks: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    for block in np.unique(blocks):
        mask = blocks == block
        shuffled[mask] = rng.permutation(labels[mask])
    return shuffled


def _clusters(statistic: np.ndarray, threshold: float):
    padded = np.r_[False, statistic > threshold, False].astype(np.int8)
    changes = np.diff(padded)
    return [(int(start), int(stop),
             float(np.sum(statistic[start:stop] - threshold)))
            for start, stop in zip(np.flatnonzero(changes == 1),
                                   np.flatnonzero(changes == -1))]


def run(result_dir: Path | None = None) -> dict:
    result_dir = Path(result_dir or RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in sorted(RUN_TO_RECORDING) if r.endswith(EXPERIMENT)]
    sessions = {run: load_session(run) for run in runs}
    day1, day2 = sessions[runs[0]], sessions[runs[1]]
    grid_index = np.searchsorted(day1.time_ms, GRID_MS)

    directions = ((day1, day2, f"{runs[0]}->{runs[1]}"),
                  (day2, day1, f"{runs[1]}->{runs[0]}"))
    observed = {name: decode_direction(train, test, grid_index)
                for train, test, name in directions}
    mean_accuracy = np.mean(list(observed.values()), axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    null = np.empty((N_PERMUTATIONS, GRID_MS.size))
    for index in range(N_PERMUTATIONS):
        per_direction = []
        for train, test, _ in directions:
            shuffled = _shuffle_within_blocks(
                test.is_deviant.astype(int), test.block, rng)
            per_direction.append(
                decode_direction(train, test, grid_index, test_labels=shuffled))
        null[index] = np.mean(per_direction, axis=0)
        if (index + 1) % 100 == 0:
            print(f"  permutation {index + 1}/{N_PERMUTATIONS}", flush=True)

    threshold = float(np.quantile(null, 1.0 - CLUSTER_ALPHA))
    null_max = np.zeros(N_PERMUTATIONS)
    for index in range(N_PERMUTATIONS):
        found = _clusters(null[index], threshold)
        if found:
            null_max[index] = max(item[2] for item in found)

    significant = np.zeros(GRID_MS.size, dtype=bool)
    rows = []
    for start, stop, mass in _clusters(mean_accuracy, threshold):
        probability = (1.0 + np.count_nonzero(null_max >= mass)) / (
            N_PERMUTATIONS + 1.0)
        significant[start:stop] = probability < ALPHA
        rows.append({"start_ms": float(GRID_MS[start]),
                     "end_ms": float(GRID_MS[stop - 1]),
                     "cluster_mass": mass, "p_fwer": probability,
                     "significant": bool(probability < ALPHA)})

    np.savez_compressed(
        result_dir / "ab_ba_decoder.npz",
        time_ms=GRID_MS, accuracy=mean_accuracy, significant=significant,
        null_mean=null.mean(axis=0), threshold=np.array([threshold]),
        **{f"accuracy_{name.replace('->', '_to_')}": value
           for name, value in observed.items()})

    report = {
        "design": (
            "Train on one recording with deviance labels, test on the other. "
            "The acoustic-to-role mapping is inverted between them, so an "
            "acoustic decoder scores below chance and a session feature "
            "carries no information across; only deviance can score above "
            "chance. Both directions averaged."
        ),
        "runs": {run: {"date": s.playback.date, "deviant": s.playback.deviant}
                 for run, s in sessions.items()},
        "n_permutations": N_PERMUTATIONS,
        "null_mean_accuracy": float(null.mean()),
        "peak_accuracy": float(mean_accuracy.max()),
        "peak_time_ms": float(GRID_MS[int(mean_accuracy.argmax())]),
        "cluster_forming_accuracy": threshold,
        "clusters": sorted(rows, key=lambda r: -r["cluster_mass"])[:6],
        "per_direction_peak": {name: float(value.max())
                               for name, value in observed.items()},
    }
    (result_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    output = run()
    print(json.dumps(output, indent=2, sort_keys=True))
