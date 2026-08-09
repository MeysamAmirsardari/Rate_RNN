"""Cluster-mass permutation test for the replicated AB/BA decoder curves.

Built to the same recipe as ``final_figures/figure_2/inference.py``: the whole
decoder is refitted for every permutation, clusters are formed on the observed
accuracy against a per-latency null, and the family-wise error rate is
controlled by the maximum cluster mass over **all latencies and both facets
jointly**.

What the null is, and what it is not
------------------------------------
The permuted quantity is the Rep-1/Rep-15 label. Figure 2 could swap labels
*within* an acquisition block because both repetitions occur inside every
block. Here they cannot: ``scripts_AB_BA.m`` draws class 0 from one recording
and class 1 from the other, so every trial in a block carries the same label and
a within-block swap is degenerate. The only available exchangeability is a free
permutation across trials, which tests "these two sets are distinguishable" --
a question that the recording difference can answer on its own. The test is
therefore reported as a test of the replicated curve, not as evidence that
deviance is decodable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import ECoG.ab_ba.matlab_replication as replication

RESULT_DIR = Path(__file__).resolve().parent / "results" / "matlab_replication_inference"

#: 5 ms inference grid, matching the Figure 2 decoder.
GRID_STEP_MS = 5
N_PERMUTATIONS = 499
RANDOM_SEED = 20_260_809
ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
FACETS = (("AB", True), ("BA", False))


def _accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    splitter = StratifiedKFold(n_splits=replication.N_FOLDS, shuffle=True,
                               random_state=replication.RANDOM_SEED)
    for train, test in splitter.split(features, labels):
        model = LogisticRegression(
            C=1.0 / (train.size * replication.RIDGE_LAMBDA), penalty="l2",
            solver="lbfgs", max_iter=1000)
        model.fit(features[train], labels[train])
        correct += int((model.predict(features[test]) == labels[test]).sum())
    return correct / labels.size


def _facet(swap: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised features on the inference grid, and the labels."""

    first, second = ((replication.RUN_DAY2, replication.RUN_DAY1) if swap
                     else (replication.RUN_DAY1, replication.RUN_DAY2))
    day1 = replication.load_epochs(first)
    day2 = replication.load_epochs(second)
    class_0, class_1, _ = replication.build_classes(day1, day2)

    full = np.concatenate([class_0, class_1], axis=0)
    labels = np.r_[np.zeros(class_0.shape[0], int), np.ones(class_1.shape[0], int)]
    grid = np.arange(0, full.shape[2], GRID_STEP_MS)

    rng = np.random.default_rng(replication.RANDOM_SEED)
    features = np.empty((grid.size, full.shape[0], full.shape[1]))
    for index, sample in enumerate(grid):
        block = full[:, :, sample].copy()
        block += rng.standard_normal(block.shape) * replication.NOISE_SCALE
        features[index] = (block - block.mean(axis=0)) / block.std(axis=0, ddof=1)
    return features, labels, grid


def _clusters(statistic: np.ndarray, threshold: np.ndarray):
    selected = statistic > threshold
    padded = np.r_[False, selected, False].astype(np.int8)
    changes = np.diff(padded)
    return [(int(start), int(stop),
             float(np.sum(statistic[start:stop] - threshold[start:stop])))
            for start, stop in zip(np.flatnonzero(changes == 1),
                                   np.flatnonzero(changes == -1))]


def run(result_dir: Path | None = None) -> dict:
    result_dir = Path(result_dir or RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    observed, nulls, grids = {}, {}, {}
    for name, swap in FACETS:
        features, labels, grid = _facet(swap)
        grids[name] = grid
        observed[name] = np.array([_accuracy(features[i], labels)
                                   for i in range(grid.size)])
        print(f"[{name}] observed peak {observed[name].max():.3f} at "
              f"{grid[int(observed[name].argmax())]} ms", flush=True)

        rng = np.random.default_rng(RANDOM_SEED)
        null = np.empty((N_PERMUTATIONS, grid.size))
        for permutation in range(N_PERMUTATIONS):
            shuffled = rng.permutation(labels)
            null[permutation] = [_accuracy(features[i], shuffled)
                                 for i in range(grid.size)]
            if (permutation + 1) % 50 == 0:
                print(f"  [{name}] permutation {permutation + 1}/"
                      f"{N_PERMUTATIONS}", flush=True)
        nulls[name] = null

    thresholds = {name: np.quantile(nulls[name], 1.0 - CLUSTER_FORMING_ALPHA,
                                    axis=0) for name, _ in FACETS}
    null_max = np.zeros(N_PERMUTATIONS)
    for name, _ in FACETS:
        for permutation in range(N_PERMUTATIONS):
            found = _clusters(nulls[name][permutation], thresholds[name])
            if found:
                null_max[permutation] = max(null_max[permutation],
                                            max(item[2] for item in found))

    report, arrays = {}, {}
    for name, _ in FACETS:
        grid = grids[name]
        significant = np.zeros(grid.size, dtype=bool)
        rows = []
        for start, stop, mass in _clusters(observed[name], thresholds[name]):
            probability = (1.0 + np.count_nonzero(null_max >= mass)) / (
                N_PERMUTATIONS + 1.0)
            significant[start:stop] = probability < ALPHA
            rows.append({"start_ms": int(grid[start]),
                         "end_ms": int(grid[stop - 1]),
                         "cluster_mass": mass, "p_fwer": probability,
                         "significant": bool(probability < ALPHA)})
        report[name] = sorted(rows, key=lambda r: -r["cluster_mass"])[:6]
        arrays[f"{name}_time_ms"] = grid.astype(float)
        arrays[f"{name}_accuracy"] = observed[name]
        arrays[f"{name}_threshold"] = thresholds[name]
        arrays[f"{name}_significant"] = significant

    np.savez_compressed(result_dir / "decoder_inference.npz", **arrays)
    summary = {
        "n_permutations": N_PERMUTATIONS,
        "grid_step_ms": GRID_STEP_MS,
        "multiplicity": "maximum cluster mass over latencies and both facets",
        "null": ("free permutation of the Rep-1/Rep-15 label; a within-block "
                 "swap is impossible because the two classes come from "
                 "different recordings"),
        "clusters": report,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, default=float))
