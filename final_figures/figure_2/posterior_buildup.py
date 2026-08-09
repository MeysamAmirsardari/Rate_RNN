"""Block-level posterior summaries for the ECoG half of Figure 2G."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import logit

from ECoG.roving.config import ANALYSES


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
RESULT_ROOT = HERE.parents[1] / "ECoG" / "roving" / "results"
POSITION_KEYS = ("zaatar_pos1", "zaatar_pos2", "zaatar_pos3")
N_RANDOMIZATIONS = 4_999
RANDOM_SEED = 2_026_073_1
ALPHA = 0.05
NPZ_NAME = "posterior_buildup_inference.npz"
PROVENANCE_NAME = "posterior_buildup_provenance.json"
CSV_NAME = "posterior_buildup_inference.csv"


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    return [
        RESULT_ROOT
        / key
        / "regression_rep_map"
        / "leakage-safe"
        / "regression_rep_map_arrays.npz"
        for key in POSITION_KEYS
    ]


def _analysis_id() -> str:
    identity = {
        "generator": _sha256(Path(__file__)),
        "sources": {str(path.resolve()): _sha256(path) for path in _source_paths()},
        "window": "position-specific variable tone, onset inclusive to onset+180 exclusive",
        "randomizations": N_RANDOMIZATIONS,
        "random_seed": RANDOM_SEED,
        "test": "conditional whole-block logit sign flip; joint two-sided max-|t|",
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _t_statistic(values: np.ndarray) -> np.ndarray:
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return np.divide(
        np.mean(values, axis=0),
        sem,
        out=np.zeros_like(sem),
        where=sem > 0,
    )


def _build() -> tuple[dict[str, np.ndarray], dict]:
    probability_values: list[np.ndarray] = []
    logit_values: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    for position, (key, path) in enumerate(
        zip(POSITION_KEYS, _source_paths()), start=1
    ):
        if not path.exists():
            raise FileNotFoundError(path)
        spec = ANALYSES[key]
        with np.load(path, allow_pickle=False) as source:
            time = np.asarray(source["time_ms"], dtype=int)
            window = (time >= spec.deviant_onset_ms) & (
                time < spec.deviant_onset_ms + 180
            )
            if np.count_nonzero(window) != 180:
                raise AssertionError(f"{key}: expected a 180-sample tone window")
            posterior = np.asarray(source["posterior_trials"][..., window], float)
        probability = np.mean(posterior, axis=-1)
        evidence = np.mean(
            logit(np.clip(posterior, 1e-6, 1 - 1e-6)), axis=-1
        )
        probability_values.append(probability)
        logit_values.append(evidence)
        source_hashes[str(path.resolve())] = _sha256(path)

    probabilities = np.stack(probability_values)
    evidence = np.stack(logit_values)
    n_positions, n_blocks, n_repetitions = evidence.shape
    observed = np.stack([_t_statistic(evidence[p]) for p in range(n_positions)])
    null = np.empty(
        (N_RANDOMIZATIONS, n_positions, n_repetitions), dtype=np.float32
    )
    for position in range(n_positions):
        rng = np.random.default_rng(
            np.random.SeedSequence([RANDOM_SEED, 51_000, position + 1])
        )
        signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(N_RANDOMIZATIONS, n_blocks),
        )
        signed_mean = signs @ evidence[position] / n_blocks
        sum_squares = np.sum(evidence[position] ** 2, axis=0)
        variance = (
            sum_squares[None] - n_blocks * signed_mean**2
        ) / (n_blocks - 1)
        signed_sem = np.sqrt(np.maximum(variance, 0) / n_blocks)
        null[:, position] = np.divide(
            signed_mean,
            signed_sem,
            out=np.zeros_like(signed_mean),
            where=signed_sem > 0,
        )
    maximum = np.max(np.abs(null), axis=(1, 2))
    p_corrected = np.empty_like(observed)
    for position in range(n_positions):
        for repetition in range(n_repetitions):
            p_corrected[position, repetition] = (
                1
                + np.count_nonzero(
                    maximum >= abs(observed[position, repetition])
                )
            ) / (N_RANDOMIZATIONS + 1)

    mean = np.mean(probabilities, axis=1)
    sem = np.std(probabilities, axis=1, ddof=1) / np.sqrt(n_blocks)
    significant = p_corrected < ALPHA
    arrays: dict[str, np.ndarray] = {
        "repetitions": np.arange(1, n_repetitions + 1, dtype=int),
        "posterior_probability_values": probabilities,
        "posterior_logit_values": evidence,
        "mean": mean,
        "sem": sem,
        "t": observed,
        "p_corrected": p_corrected,
        "significant": significant,
    }
    for position in range(1, 4):
        index = position - 1
        for name, value in (
            ("mean", mean[index]),
            ("sem", sem[index]),
            ("t", observed[index]),
            ("p_corrected", p_corrected[index]),
            ("significant", significant[index]),
        ):
            arrays[f"posterior_buildup_pos{position}_{name}"] = value
    provenance = {
        "analysis": "Figure 2G ECoG posterior buildup",
        "scope": (
            "conditional within-recording inference for three recordings "
            "from one animal; not animal-population inference"
        ),
        "measure": (
            "blockwise out-of-fold Rep-1 posterior, averaged over the "
            "position-specific 0–180 ms variable-tone window"
        ),
        "uncertainty": "SEM of blockwise posterior probabilities",
        "test_measure": (
            "blockwise mean logit posterior over the same tone window versus 0"
        ),
        "test": (
            "4,999 conditional whole-block sign flips; two-sided max-|t| "
            "correction jointly over three positions and repetitions 1–15"
        ),
        "assumption": (
            "sign symmetry of frozen cross-fitted block logits; decoder "
            "models are not refit"
        ),
        "n_blocks": n_blocks,
        "source_sha256": source_hashes,
    }
    return arrays, provenance


def load_or_build_posterior_buildup(
    *,
    force: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    destination = Path(data_dir or DEFAULT_DATA_DIR).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    npz_path = destination / NPZ_NAME
    provenance_path = destination / PROVENANCE_NAME
    analysis_id = _analysis_id()
    if npz_path.exists() and provenance_path.exists() and not force:
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance.get("analysis_id") == analysis_id
            and provenance.get("npz_sha256") == _sha256(npz_path)
        ):
            with np.load(npz_path, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}

    arrays, provenance = _build()
    arrays["analysis_id"] = np.asarray(analysis_id)
    np.savez_compressed(npz_path, **arrays)
    rows = []
    for position in (1, 2, 3):
        for index, repetition in enumerate(arrays["repetitions"]):
            rows.append(
                [
                    position,
                    int(repetition),
                    float(arrays[f"posterior_buildup_pos{position}_mean"][index]),
                    float(arrays[f"posterior_buildup_pos{position}_sem"][index]),
                    float(
                        arrays[
                            f"posterior_buildup_pos{position}_p_corrected"
                        ][index]
                    ),
                    bool(
                        arrays[
                            f"posterior_buildup_pos{position}_significant"
                        ][index]
                    ),
                ]
            )
    with (destination / CSV_NAME).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "deviant_position",
                "repetition",
                "mean_rep1_posterior",
                "sem",
                "p_corrected",
                "significant",
            ]
        )
        writer.writerows(rows)
    provenance.update(
        {
            "analysis_id": analysis_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "npz_sha256": _sha256(npz_path),
            "csv_sha256": _sha256(destination / CSV_NAME),
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": _sha256(Path(__file__)),
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return arrays


__all__ = ["load_or_build_posterior_buildup"]
