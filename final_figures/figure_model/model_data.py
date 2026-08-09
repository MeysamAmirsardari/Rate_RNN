"""Compute the real quantities behind the two-layer model figure.

The figure explains an architecture, so it would have been defensible to draw
every stage.  It does not.  Each object in the layer-2 pipeline is the actual
array the implementation forms, computed here through the real equations:

``E``       layer-1 excitatory rate, from ``model0.simulate`` driven by the
            genuine Saffran stream excerpt cached for Figure 5.
``s``       the filterbank state, one leaky integrator per (channel, rate),
            stepped with ``Layer2MR``'s own update.
``D``       the coincidence map ``outer(E, s_flat)`` with same-channel entries
            zeroed — formed by calling ``Layer2MR.coincidence`` itself.
``M``       a learned mask, taken unmodified from the Figure 5 exemplar session.

The one thing the figure asserts structurally is the interface: **layer 2 reads
the rate vector E and nothing else.**  That is visible here — ``Layer2MR.step``
takes ``E`` as its only argument from layer 1, never ``W`` — and it is why
composition survives with layer 1 frozen or absent (4.00 / 3.88 / 3.75 words
held in order).

The snapshot instant is the peak of the final token of a four-token word, so
the coincidence map on display actually contains a whole word of context.
"""

from __future__ import annotations

import argparse
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

from layer2_multirate.config import MRConfig  # noqa: E402
from layer2_multirate.layer2 import Layer2MR  # noqa: E402
from layer2_syllable.run_ab_ba import LAYER1, layer1_rates  # noqa: E402
from model0.config import A1Config, selective_inh  # noqa: E402


#: Mirrors final_figures/figure_5/saffran_data.py, which owns the paradigm.
WORDS: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5),
                                      (6, 7, 8, 9), (10, 11, 12, 13))
WORD_NAMES: tuple[str, ...] = ("W1", "W2", "W3", "W4")
N_CHANNELS = 14
N_UNITS = 24
TONE_DUR = 0.050
GAP = 0.030

EXEMPLAR = (_REPO_ROOT / "final_figures" / "figure_5" / "data"
            / "saffran_exemplar.npz")

DATA_VERSION = "model-figure-1"


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


def _settings() -> dict[str, Any]:
    cfg = selective_inh(N=N_CHANNELS, **LAYER1)
    return {
        "data_version": DATA_VERSION,
        "n_channels": N_CHANNELS,
        "n_units": N_UNITS,
        "tone_dur_s": TONE_DUR,
        "gap_s": GAP,
        "words": [list(word) for word in WORDS],
        "layer1_overrides": dict(LAYER1),
        "layer1_config": {key: value for key, value in vars(cfg).items()},
        "layer2_config": {key: value for key, value in
                          vars(MRConfig(n_units=N_UNITS)).items()},
        "exemplar_source": str(EXEMPLAR),
    }


def build_model_data(*, force: bool = False,
                     data_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Run (or load) the real layer-1 and layer-2 state the figure draws."""

    data_dir = Path(data_dir or (_THIS_FILE.parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "model_figure_data.npz"
    provenance_path = data_dir / "model_provenance.json"

    settings = _settings()
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("settings_hash") == settings_hash:
            with np.load(npz_path, allow_pickle=True) as handle:
                return {key: handle[key] for key in handle.files}

    if not EXEMPLAR.exists():
        raise FileNotFoundError(
            f"{EXEMPLAR} is missing. Build Figure 5 first:\n"
            "    python -m final_figures.figure_5.make_figure_5"
        )

    print("[model figure] running layer 1 and the layer-2 filterbank "
          "on the cached Saffran excerpt")

    with np.load(EXEMPLAR, allow_pickle=True) as handle:
        exemplar = {key: handle[key] for key in handle.files}

    stim = np.asarray(exemplar["excerpt_stim"], dtype=float)
    dt = float(np.asarray(exemplar["excerpt_dt"]).ravel()[0])
    stops = np.asarray(exemplar["excerpt_word_stops"], dtype=float)
    order = np.asarray(exemplar["excerpt_order"], dtype=int)
    n_tone = int(np.asarray(exemplar["excerpt_n_tone"]).ravel()[0])
    taus = np.asarray(exemplar["exemplar_taus"], dtype=float)
    masks = np.asarray(exemplar["exemplar_span_masks"], dtype=float)
    mask_now = np.asarray(exemplar["exemplar_span_now"], dtype=int)
    mask_word = np.asarray(exemplar["exemplar_span_word"], dtype=int)
    seed = int(np.asarray(exemplar["seed"]).ravel()[0])

    # ---- layer 1 on the real stream -------------------------------------
    cfg: A1Config = selective_inh(N=N_CHANNELS, **LAYER1)
    rates, _ = layer1_rates(stim, cfg, mode="full", seed=seed)
    rates = np.asarray(rates, dtype=float)

    # ---- layer 2's own filterbank, stepped forward ----------------------
    layer = Layer2MR(N_CHANNELS, MRConfig(n_units=N_UNITS, rates=tuple(taus),
                                          seed=seed))
    # The snapshot is the last token of a four-token word, so the coincidence
    # map on display holds a complete word of context.
    long_words = [index for index, word in enumerate(order)
                  if len(WORDS[word]) == 4]
    if not long_words:
        raise RuntimeError("the cached excerpt contains no four-token word")
    chosen = long_words[len(long_words) // 2]
    snapshot_word = int(order[chosen])
    # Word stops mark the offset of the final token, so its onset is one tone
    # earlier; the snapshot sits late inside that token, where its rate peaks.
    final_onset = int(stops[chosen]) - n_tone
    snapshot = final_onset + int(round(0.8 * n_tone))

    s_history = np.zeros((snapshot + 1, N_CHANNELS, taus.size))
    for step in range(snapshot + 1):
        layer.step(rates[:, step], dt, learn=False)
        s_history[step] = layer.s

    E_snapshot = rates[:, snapshot].copy()
    s_snapshot = layer.s.copy()
    coincidence = layer.coincidence(E_snapshot)

    # ---- the mask that belongs to this word -----------------------------
    matching = np.flatnonzero(mask_word == snapshot_word)
    mask_index = int(matching[0]) if matching.size else 0
    mask = masks[mask_index].copy()

    arrays: dict[str, np.ndarray] = {
        "E": E_snapshot,
        "s": s_snapshot,
        "D": coincidence,
        "M": mask,
        "taus": taus,
        "dt": np.array([dt]),
        "snapshot_index": np.array([snapshot]),
        "snapshot_word": np.array([snapshot_word]),
        "snapshot_channel": np.array([int(mask_now[mask_index])]),
        "mask_unit": np.array([int(np.asarray(
            exemplar["exemplar_span_units"], dtype=int)[mask_index])]),
        "valid_connections": layer.valid_connections.astype(float),
        "n_rates": np.array([taus.size]),
        "n_channels": np.array([N_CHANNELS]),
        "seed": np.array([seed]),
        "tau_E_ms": np.array([cfg.tau_E * 1e3]),
        "tau_I_ms": np.array([cfg.tau_I * 1e3]),
        "tau_std_s": np.asarray(cfg.tau_std, dtype=float),
    }

    # A short window of layer-1 rates around the snapshot, for the inset that
    # shows what the filterbank is integrating.
    window = int(round(0.9 / dt))
    start = max(0, snapshot - window)
    arrays["rate_window"] = rates[:, start:snapshot + 1]
    arrays["rate_window_start_ms"] = np.array([(start - snapshot) * dt * 1e3])
    arrays["s_window"] = s_history[start:snapshot + 1]

    print(f"    snapshot at sample {snapshot} "
          f"({snapshot * dt * 1e3:.0f} ms into the excerpt)")
    print(f"    word {WORD_NAMES[snapshot_word]} = "
          f"{WORDS[snapshot_word]}, final token channel "
          f"{int(mask_now[mask_index])}")
    print(f"    E nonzero channels: {np.flatnonzero(E_snapshot > 1e-6).tolist()}")
    print(f"    D shape {coincidence.shape}, mask shape {mask.shape}, "
          f"{taus.size} rates spanning "
          f"{taus[0] * 1e3:.0f}-{taus[-1] * 1e3:.0f} ms")

    _atomic_npz(npz_path, arrays)
    _atomic_json(provenance_path, {
        "figure": "Two-layer model definition",
        "settings": settings,
        "settings_hash": settings_hash,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stimulus_source": (
            "final_figures/figure_5/data/saffran_exemplar.npz - the cached "
            "late-exposure Saffran stream excerpt, used unmodified."
        ),
        "layer1": (
            "model0.simulate through layer2_syllable.run_ab_ba.layer1_rates "
            "in 'full' mode (recurrent plasticity on), N = 14, overrides "
            f"{dict(LAYER1)}."
        ),
        "layer2": (
            "Layer2MR stepped forward over the layer-1 rate history with "
            "learn=False. s, D come from the class's own update and "
            "coincidence(); the mask is taken unmodified from the Figure 5 "
            "exemplar session."
        ),
        "interface_note": (
            "Layer2MR.step receives E only. It never reads layer 1's "
            "recurrent weight matrix W. This is the structural reason "
            "composition survives the layer-1 controls."
        ),
        "snapshot": {
            "sample": int(snapshot),
            "ms_into_excerpt": snapshot * dt * 1e3,
            "word": WORD_NAMES[snapshot_word],
            "word_tokens": list(WORDS[snapshot_word]),
            "final_token_channel": int(mask_now[mask_index]),
            "mask_unit": int(np.asarray(
                exemplar["exemplar_span_units"], dtype=int)[mask_index]),
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
                        help="Recompute even if the cache is valid.")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    build_model_data(force=_parse_args(arguments).force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
