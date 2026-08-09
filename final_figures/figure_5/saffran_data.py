"""Generate the model-side data for manuscript Figure 5.

Saffran-style statistical word segmentation, run under **one protocol** for
every condition the figure reports.  The strategy document requires that the
single-timescale and multiscale readouts be regenerated with the same word set,
training exposure, unit count, testing procedure, and seed structure, so that
the only difference between them is the quantity under test.  That requirement
is met literally here: ``L2Config`` and ``MRConfig`` already share ``eta``,
``lam``, ``gate_frac``, ``w_init`` and ``commit_frac``, this module forces the
same ``n_units`` on both, and the two readouts differ **only** in the number of
timescales in the trace bank (1 versus 6).

Design
------
Paradigm (Saffran, Aslin & Newport 1996).  Four three-token words built from
twelve channels, concatenated in random order, no word immediately repeating.
The stream is perfectly isochronous: the gap inside a word equals the gap
across a boundary, so nothing but the transition statistics marks a boundary.

    within a word      P(next | current) = 1
    across a boundary  P(next | current) = 1/3

Testing is the infant test: isolated **words** against **part-words**, which are
three-token sequences that did occur in the stream but straddle a boundary.

Factorial
---------
``readout``     ``single_rate`` (one trace, R = 1) or ``multi_rate``
                (filterbank, R = 6).
``layer1_mode`` ``full`` (recurrent plasticity on), ``frozen`` (layer 1 present,
                recurrent weights held at zero) or ``raw`` (no cortex at all;
                layer 2 reads the stimulus directly).  This is the dependency
                control the strategy requires: if ``raw`` matches ``full``, the
                readout is a generic multiscale sequence mechanism and must be
                described as one.
``exposure``    ``structured`` (the Saffran stream) or ``scrambled`` (the same
                twelve tokens, the same isochronous timing, but no word
                structure).  Test items are the original words and part-words in
                both cases, so ``scrambled`` is the floor the effect must clear.

The replication unit is the **exposure order seed**.  Eight seeds are run, and
every seed changes the constrained word order and the layer-1 simulation seed.
Layer 2 masks start from the same small random values in every seed, so these
are independent exposure sessions rather than independent networks; that is
stated rather than glossed.

Metric
------
``span_depth`` is the uniform structural measure applied identically to both
readouts: how many consecutive tokens a unit's mask represents **in order**,
with each older token held at a strictly slower rate.  A unit spans a whole
three-token word when its ordered triple matches a real word *and* the older
predecessor is carried by a slower filter than the recent one.  With one
timescale there is no slower filter available, so ``span_depth`` cannot exceed
two by construction -- the limitation is structural, not a tuning accident.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from layer2_multirate.config import MRConfig  # noqa: E402
from layer2_multirate.layer2 import Layer2MR  # noqa: E402
from layer2_syllable.config import L2Config  # noqa: E402
from layer2_syllable.layer2 import Layer2  # noqa: E402
from layer2_syllable.run_ab_ba import LAYER1, layer1_rates  # noqa: E402
from layer2_syllable.saffran.run_saffran import (  # noqa: E402
    boundary_bigrams,
    within_bigrams,
    word_order,
)
from layer2_syllable.stimulus import build_stream, chunk_windows  # noqa: E402
from model0 import selective_inh  # noqa: E402


# ---------------------------------------------------------------------------
# Protocol — one definition, used by every condition
# ---------------------------------------------------------------------------
#: Two three-token and two four-token words.  Mixed lengths matter: a readout
#: that only ever sees three-token patterns can succeed with a fixed-width
#: template, whereas a vocabulary of different lengths forces the units to
#: discover where each word starts and stops.
WORDS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8, 9),
    (10, 11, 12, 13),
)
WORD_NAMES: tuple[str, ...] = ("W1", "W2", "W3", "W4")
WORD_LENGTHS: tuple[int, ...] = tuple(sorted({len(w) for w in WORDS}))
MAX_WORD_LEN = max(len(w) for w in WORDS)
N_CHANNELS = 14
TONE_DUR = 0.050
GAP = 0.030
N_TRAIN_WORDS = 500
N_TEST_REPS = 8
N_UNITS = 24
N_SEEDS = 8

#: Words in the late-exposure excerpt kept for the unit-activity tape.
EXCERPT_WORDS = 12

READOUTS: tuple[str, ...] = ("single_rate", "multi_rate")
LAYER1_MODES: tuple[str, ...] = ("full", "frozen", "raw")
EXPOSURES: tuple[str, ...] = ("structured", "scrambled")

#: Conditions actually simulated.  The layer-1 dependency control is run under
#: structured exposure only; the scrambled floor is run with the intact layer 1.
CONDITIONS: tuple[tuple[str, str], ...] = (
    ("full", "structured"),
    ("frozen", "structured"),
    ("raw", "structured"),
    ("full", "scrambled"),
)

#: The condition every headline panel reports.
REFERENCE_CONDITION = ("full", "structured")

EXEMPLAR_VERSION = 2
"""Bumped when the exemplar export changes shape or meaning.

The exemplar illustration is cached separately from the factorial, so changing
what the mechanism panels need costs one simulated session rather than all 64.
"""

DETECTOR_CRITERION = 0.90
"""Held-out AUC a unit must reach to count as a detector for a whole item.

A functional criterion, deliberately strict, and applied identically to both
architectures.  It asks only whether some unit separates the complete item from
every other item of the same length; it makes no reference to masks, rates, or
the span analysis.
"""

MASK_FLOOR = 0.02
"""Minimum fraction of a unit's total mask mass for a predecessor to count.

Without a floor, numerical dust in an unused corner of a mask can complete an
ordered triple by accident.  Two percent is far above the decayed background
(``lam`` prunes unused synapses continuously) and far below the weights the
committed units actually place on their chosen predecessors.
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


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


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney area under the ROC curve, ties counted as one half."""

    positive = np.asarray(positive, dtype=float).ravel()
    negative = np.asarray(negative, dtype=float).ravel()
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()
    return float(wins / (positive.size * negative.size))


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
def _structured_stream(dt: float, seed: int, n_words: int = N_TRAIN_WORDS) -> dict:
    rng = np.random.default_rng(1000 + seed)
    order = word_order(len(WORDS), n_words, rng)
    stream = build_stream(
        WORDS,
        [0.25] * len(WORDS),
        n_words,
        N_CHANNELS,
        dt,
        tone_dur=TONE_DUR,
        intra_gap=GAP,
        inter_gap=GAP,
        seed=seed,
        order=order,
    )
    stream["order"] = np.asarray(order, dtype=int)
    stream["vocabulary"] = [tuple(int(t) for t in w) for w in WORDS]
    return stream


def _scrambled_stream(dt: float, seed: int, n_words: int = N_TRAIN_WORDS) -> dict:
    """Same tokens, same isochronous timing, no word structure.

    Tokens are drawn uniformly with the single constraint that a token never
    immediately repeats, which is also true of the structured stream.  Every
    transition therefore occurs with probability 1/11 and no three-token group
    is more probable than any other.
    """

    rng = np.random.default_rng(5000 + seed)
    total = 3 * n_words
    tokens = np.empty(total, dtype=int)
    previous = -1
    for index in range(total):
        choices = [t for t in range(N_CHANNELS) if t != previous]
        tokens[index] = int(rng.choice(choices))
        previous = int(tokens[index])
    triples = [tuple(int(t) for t in row) for row in tokens.reshape(-1, 3)]
    stream = build_stream(
        triples,
        [1.0] * len(triples),
        len(triples),
        N_CHANNELS,
        dt,
        tone_dur=TONE_DUR,
        intra_gap=GAP,
        inter_gap=GAP,
        seed=seed,
        order=np.arange(len(triples)),
    )
    stream["order"] = np.arange(len(triples), dtype=int)
    stream["vocabulary"] = triples
    return stream


def part_words_of_length(length: int) -> list[tuple[int, ...]]:
    """Sequences of ``length`` tokens that occurred but straddle a boundary.

    Built by taking the final ``k`` tokens of one word and the first
    ``length - k`` of another.  Anything that coincides with a real word is
    removed, so a part-word is never accidentally a word.
    """

    vocabulary = {tuple(w) for w in WORDS}
    out: set[tuple[int, ...]] = set()
    for first in WORDS:
        for second in WORDS:
            if first is second:
                continue
            for k in range(1, length):
                if k <= len(first) and (length - k) <= len(second):
                    out.add(tuple(first[-k:]) + tuple(second[: length - k]))
    return sorted(out - vocabulary)


def _test_stream(dt: float, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Isolated words against part-words, blocked by item length.

    With a mixed-length vocabulary a word and a part-word must be compared at
    equal length, otherwise item duration is confounded with word status.  The
    stream therefore contains, for each length present in the vocabulary, the
    words of that length and the part-words of that length; discrimination is
    scored within length and then averaged.
    """

    items: list[tuple[int, ...]] = []
    kind: list[str] = []
    lengths: list[int] = []
    for length in WORD_LENGTHS:
        for word in WORDS:
            if len(word) == length:
                items.append(tuple(word))
                kind.append("word")
                lengths.append(length)
        for partial in part_words_of_length(length):
            items.append(partial)
            kind.append("part_word")
            lengths.append(length)

    rng = np.random.default_rng(77 + seed)
    order = np.concatenate(
        [rng.permutation(len(items)) for _ in range(N_TEST_REPS)]
    )
    stream = build_stream(
        items,
        [1.0] * len(items),
        len(order),
        N_CHANNELS,
        dt,
        tone_dur=TONE_DUR,
        intra_gap=GAP,
        inter_gap=0.500,
        seed=seed,
        order=order,
    )
    stream["order"] = np.asarray(order, dtype=int)
    stream["items"] = items
    return (stream,
            np.asarray(kind)[order],
            np.asarray(lengths)[order])


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------
def _make_readout(readout: str, seed: int):
    """Both readouts, forced onto identical learning parameters."""

    if readout == "single_rate":
        config = L2Config(n_units=N_UNITS, seed=seed)
        return Layer2(N_CHANNELS, config), config, np.array([config.tau_decay])
    if readout == "multi_rate":
        config = MRConfig(n_units=N_UNITS, seed=seed)
        return Layer2MR(N_CHANNELS, config), config, np.asarray(config.rates)
    raise ValueError(f"Unknown readout: {readout!r}")


def _mask_context_by_rate(layer, unit: int, channel: int) -> np.ndarray:
    """A unit's mask row as (context channel, rate), for either readout."""

    if isinstance(layer, Layer2MR):
        return layer.mask_context_rate(unit, channel)
    return np.asarray(layer.M[unit, channel], dtype=float)[:, None]


def span_chain(layer, unit: int, taus: np.ndarray) -> dict[str, Any]:
    """The ordered token chain a unit's mask holds, and the word it matches.

    The same code runs for both readouts.  Starting from the channel driving
    the unit now, predecessors are taken greedily, each required to be on a
    different channel and on a **strictly slower** filter than the one before,
    so that its relative age is recoverable.  With a single timescale no slower
    filter exists and the chain can never exceed length two.

    A unit spans a word when the token driving it is that word's final token
    *and* the tail of the chain is the complete word.  A chain may carry an
    extra older token beyond the word (the boundary context that preceded it);
    that still counts, because the whole word is present, in order, at
    recoverable ages.
    """

    mask = np.asarray(layer.M[unit], dtype=float)
    total = float(mask.sum())
    empty = dict(depth=0, chain=(), now=-1, taus=(), weights=(),
                 spans=False, word=-1, word_len=0)
    if total <= 0:
        return empty

    now = int(np.argmax(mask.sum(axis=1)))
    row = _mask_context_by_rate(layer, unit, now)
    taus = np.asarray(taus, dtype=float)

    chain = [now]
    chain_taus: list[float] = []
    weights: list[float] = []
    used = {now}
    slowest = -1
    for _ in range(MAX_WORD_LEN - 1):
        candidate = row.copy()
        for channel in used:
            candidate[channel, :] = 0.0
        if slowest >= 0:
            candidate[:, taus <= taus[slowest]] = 0.0
        if not np.any(candidate > 0):
            break
        channel, rate = np.unravel_index(int(np.argmax(candidate)),
                                         candidate.shape)
        weight = float(candidate[channel, rate] / total)
        if weight < MASK_FLOOR:
            break
        chain.insert(0, int(channel))
        chain_taus.insert(0, float(taus[rate]))
        weights.insert(0, weight)
        used.add(int(channel))
        slowest = int(rate)

    spans, word_index, word_len = False, -1, 0
    for index, word in enumerate(WORDS):
        if word[-1] != now or len(word) > len(chain):
            continue
        if tuple(chain[-len(word):]) == tuple(word):
            spans, word_index, word_len = True, index, len(word)
            break

    return dict(
        depth=len(chain),
        chain=tuple(chain),
        now=now,
        taus=tuple(chain_taus),
        weights=tuple(weights),
        spans=spans,
        word=word_index,
        word_len=word_len,
    )


def _transition_selectivity(weights: np.ndarray) -> dict[str, float]:
    """Within-word versus boundary mass in a learned 12x12 transition map."""

    within = within_bigrams(WORDS)
    boundary = boundary_bigrams(WORDS)
    within_values = [float(weights[post, pre]) for pre, post in within]
    boundary_values = [float(weights[post, pre]) for pre, post in boundary]
    within_mean = float(np.mean(within_values)) if within_values else np.nan
    boundary_mean = float(np.mean(boundary_values)) if boundary_values else np.nan
    denominator = within_mean + boundary_mean
    return dict(
        within_mean=within_mean,
        boundary_mean=boundary_mean,
        selectivity=(
            float((within_mean - boundary_mean) / denominator)
            if denominator > 0
            else np.nan
        ),
    )


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(
    readout: str,
    layer1_mode: str,
    exposure: str,
    seed: int,
    *,
    keep_detail: bool = False,
) -> dict[str, Any]:
    """Expose, analyse, and test one (readout, layer-1 mode, exposure, seed)."""

    a1_config = selective_inh(N=N_CHANNELS, **LAYER1)
    dt = a1_config.dt

    if exposure == "structured":
        train_stream = _structured_stream(dt, seed)
    elif exposure == "scrambled":
        train_stream = _scrambled_stream(dt, seed)
    else:
        raise ValueError(f"Unknown exposure: {exposure!r}")

    train_E, layer1_weights = layer1_rates(
        train_stream["stim"], a1_config, mode=layer1_mode, seed=seed
    )

    layer, config, taus = _make_readout(readout, seed)
    train_trace = layer.run(train_E, dt, learn=True)

    committed = np.flatnonzero(layer.committed)
    per_unit = [span_chain(layer, int(unit), taus) for unit in committed]
    depths = np.array([entry["depth"] for entry in per_unit], dtype=int)
    spans = np.array([entry["spans"] for entry in per_unit], dtype=bool)
    words_covered = sorted({entry["word"] for entry in per_unit if entry["spans"]})

    test_stream, kind, lengths = _test_stream(dt, seed)
    test_E, _ = layer1_rates(
        test_stream["stim"], a1_config, mode=layer1_mode, seed=seed
    )
    layer.reset_state()
    test = layer.run(test_E, dt, learn=False)
    windows = chunk_windows(test_stream, pad_s=0.05)
    population = np.array(
        [test["y"][committed, start:stop].max(axis=1).sum()
         for start, stop in windows]
    )

    # Words and part-words are compared only at equal length, then averaged,
    # so item duration cannot masquerade as word status.
    per_length: dict[int, float] = {}
    for length in WORD_LENGTHS:
        selector = lengths == length
        per_length[length] = auc(
            population[selector & (kind == "word")],
            population[selector & (kind == "part_word")],
        )
    matched_auc = float(np.nanmean(list(per_length.values())))

    # Functional composition: is there a unit selective for the WHOLE item?
    # This asks nothing about masks, so it applies to both architectures on
    # equal terms.  The unit is chosen on half the presentations and scored on
    # the other half, so the selection cannot inflate the score.
    peaks = np.array(
        [test["y"][committed, start:stop].max(axis=1)
         for start, stop in windows]
    )
    item_of = [tuple(test_stream["items"][index])
               for index in test_stream["order"]]
    selection = np.arange(peaks.shape[0]) % 2 == 0
    detector_auc: list[float] = []
    for word in WORDS:
        length = len(word)
        is_word = np.array([item == tuple(word) for item in item_of])
        rivals = (lengths == length) & ~is_word
        pick = is_word & selection, rivals & selection
        score = is_word & ~selection, rivals & ~selection
        if min(pick[0].sum(), pick[1].sum(), score[0].sum(), score[1].sum()) == 0:
            continue
        candidates = [auc(peaks[pick[0], unit], peaks[pick[1], unit])
                      for unit in range(peaks.shape[1])]
        best = int(np.nanargmax(candidates)) if candidates else -1
        if best < 0:
            continue
        detector_auc.append(
            auc(peaks[score[0], best], peaks[score[1], best])
        )
    detector_auc_array = np.asarray(detector_auc, dtype=float)

    result: dict[str, Any] = dict(
        readout=readout,
        layer1_mode=layer1_mode,
        exposure=exposure,
        seed=seed,
        n_committed=int(committed.size),
        n_span=int(spans.sum()),
        n_words_covered=len(words_covered),
        max_depth=int(depths.max()) if depths.size else 0,
        mean_depth=float(depths.mean()) if depths.size else 0.0,
        auc=matched_auc,
        n_rates=int(np.asarray(taus).size),
        mean_detector_auc=(float(np.nanmean(detector_auc_array))
                           if detector_auc_array.size else np.nan),
        n_word_detectors=int((detector_auc_array >= DETECTOR_CRITERION).sum()),
    )
    for length in WORD_LENGTHS:
        result[f"auc_len{length}"] = float(per_length[length])
    if layer1_weights is not None:
        result.update(_transition_selectivity(np.asarray(layer1_weights)))
    else:
        result.update(within_mean=np.nan, boundary_mean=np.nan,
                      selectivity=np.nan)

    if keep_detail:
        result["_detail"] = dict(
            peaks=peaks,
            item_of=item_of,
            detector_auc=detector_auc_array,
            layer=layer,
            taus=np.asarray(taus),
            per_unit=per_unit,
            committed=committed,
            depths=depths,
            population=population,
            kind=kind,
            lengths=lengths,
            train_stream=train_stream,
            train_activity=train_trace["y"],
            test_stream=test_stream,
            layer1_weights=layer1_weights,
            config=config,
        )
    return result


# ---------------------------------------------------------------------------
# Filterbank illustration
# ---------------------------------------------------------------------------
def filterbank_response(duration_s: float = 1.0) -> dict[str, np.ndarray]:
    """One token through the filterbank, and the elapsed-time code it creates.

    The panel this feeds makes the mechanism concrete: after a single token the
    ratio of the trace across rates is a monotone function of how long ago that
    token fired, which is the only new ingredient the multiscale readout adds.
    """

    config = MRConfig(n_units=N_UNITS)
    layer = Layer2MR(N_CHANNELS, config)
    dt = selective_inh(N=N_CHANNELS, **LAYER1).dt
    n_steps = int(round(duration_s / dt))
    n_tone = int(round(TONE_DUR / dt))

    drive = np.zeros((N_CHANNELS, n_steps))
    drive[0, :n_tone] = 1.0

    traces = np.empty((n_steps, len(config.rates)))
    for step in range(n_steps):
        layer.s += dt * (-layer.s + drive[:, step][:, None]) / layer.tau[None, :]
        traces[step] = layer.s[0]

    time_ms = np.arange(n_steps) * dt * 1e3
    peak = traces.max(axis=0, keepdims=True)
    normalized = traces / np.where(peak > 0, peak, 1.0)
    return dict(
        time_ms=time_ms,
        traces=traces,
        normalized=normalized,
        rates_s=np.asarray(config.rates, dtype=float),
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _settings() -> dict[str, Any]:
    return dict(
        words=[list(w) for w in WORDS],
        word_lengths=list(WORD_LENGTHS),
        n_channels=N_CHANNELS,
        tone_dur_s=TONE_DUR,
        gap_s=GAP,
        n_train_words=N_TRAIN_WORDS,
        n_test_reps=N_TEST_REPS,
        n_units=N_UNITS,
        n_seeds=N_SEEDS,
        readouts=list(READOUTS),
        conditions=[list(c) for c in CONDITIONS],
        mask_floor=MASK_FLOOR,
        layer1=dict(LAYER1),
        single_rate_config=asdict(L2Config(n_units=N_UNITS)),
        multi_rate_config={
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(MRConfig(n_units=N_UNITS)).items()
        },
    )


def _build_exemplar(
    data_dir: Path,
    *,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """The single illustrative session behind the mechanism panels.

    Cached apart from the factorial and keyed on ``EXEMPLAR_VERSION`` as well as
    the protocol, so changing what the illustration exports costs one simulated
    session rather than all of them.
    """

    npz_path = data_dir / "saffran_exemplar.npz"
    provenance_path = data_dir / "saffran_exemplar_provenance.json"
    settings = _settings()
    exemplar_hash = hashlib.sha256(
        json.dumps({"settings": settings, "version": EXEMPLAR_VERSION},
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("exemplar_hash") == exemplar_hash:
            with np.load(npz_path, allow_pickle=True) as handle:
                return {name: handle[name] for name in handle.files}

    print("[figure 5] simulating the exemplar session for the mechanism panels")
    exemplar: dict[str, np.ndarray] = {}
    # Exemplar detail from the reference condition, seed 0, for the mechanism
    # and template panels.  Everything quantitative comes from the factorial
    # above; this only supplies the illustration.
    reference_mode, reference_exposure = REFERENCE_CONDITION
    exemplar = run_one(
        "multi_rate", reference_mode, reference_exposure, 0, keep_detail=True
    )
    detail = exemplar.pop("_detail")
    layer = detail["layer"]
    exemplar["exemplar_taus"] = detail["taus"]
    exemplar["exemplar_depths"] = detail["depths"].astype(float)
    exemplar["exemplar_committed"] = detail["committed"].astype(float)
    exemplar["exemplar_layer1_weights"] = np.asarray(detail["layer1_weights"])

    # Spanning units, ordered by which word they represent, with the mask row
    # that carries the evidence.  ``per_unit`` is index-aligned to
    # ``committed``, so the unit index comes from that pairing.
    spanning = [
        (int(unit), entry)
        for unit, entry in zip(detail["committed"], detail["per_unit"])
        if entry["spans"]
    ]
    spanning.sort(key=lambda pair: pair[1]["word"])
    exemplar["exemplar_span_units"] = np.array(
        [unit for unit, _ in spanning], dtype=float
    )
    exemplar["exemplar_span_now"] = np.array(
        [entry["now"] for _, entry in spanning], dtype=float
    )
    exemplar["exemplar_span_word"] = np.array(
        [entry["word"] for _, entry in spanning], dtype=float
    )
    if spanning:
        # The kernel is the mask over the coincidence map D, shape
        # (channel firing now, context channel x rate).  There is no time axis:
        # the second index is a filter identity, not elapsed time.
        exemplar["exemplar_span_masks"] = np.stack([
            np.asarray(layer.M[unit], dtype=float) for unit, _ in spanning
        ])
        exemplar["exemplar_n_rates"] = np.array(
            [np.asarray(detail["taus"]).size], dtype=float
        )

    # Late-exposure excerpt for the unit-activity tape.  It is taken from the
    # END of exposure, after the units have committed, because an excerpt from
    # the start shows learning rather than the learned response.
    train_stream = detail["train_stream"]
    activity = np.asarray(detail["train_activity"], dtype=float)
    onsets = train_stream["tone_onsets"]
    n_tone = int(train_stream["n_tone"])
    first_word = len(onsets) - EXCERPT_WORDS
    excerpt_start = int(onsets[first_word][0])
    excerpt_stop = min(int(onsets[-1][-1] + n_tone + 0.25 / train_stream["dt"]),
                       activity.shape[1])

    exemplar["excerpt_stim"] = np.asarray(
        train_stream["stim"][:, excerpt_start:excerpt_stop], dtype=float
    )
    exemplar["excerpt_dt"] = np.array([train_stream["dt"]], dtype=float)
    exemplar["excerpt_order"] = np.asarray(
        train_stream["order"][first_word:], dtype=float
    )
    exemplar["excerpt_onsets"] = np.array(
        [row[0] - excerpt_start for row in onsets[first_word:]], dtype=float
    )
    exemplar["excerpt_word_stops"] = np.array(
        [row[-1] + n_tone - excerpt_start for row in onsets[first_word:]],
        dtype=float,
    )
    exemplar["excerpt_n_tone"] = np.array([n_tone], dtype=float)

    # One representative unit per word: the spanning unit, in word order.
    exemplar["excerpt_activity"] = activity[
        [unit for unit, _ in spanning], excerpt_start:excerpt_stop
    ] if spanning else np.zeros((0, excerpt_stop - excerpt_start))

    bank = filterbank_response()
    for name, value in bank.items():
        exemplar[f"filterbank_{name}"] = np.asarray(value, dtype=float)

    _atomic_npz(npz_path, exemplar)
    _atomic_json(provenance_path, {
        "exemplar_hash": exemplar_hash,
        "exemplar_version": EXEMPLAR_VERSION,
        "reference_condition": list(REFERENCE_CONDITION),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kernel_note": (
            "exemplar_span_masks is the mask over the coincidence map, shape "
            "(unit, channel firing now, context channel x rate). The second "
            "axis of the context block is a filter identity, not elapsed time."
        ),
    })
    return exemplar


def build_saffran_data(
    *,
    force: bool = False,
    data_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run (or load) the full factorial and write the figure data contract."""

    data_dir = Path(data_dir or (_THIS_FILE.parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "saffran_figure5_data.npz"
    provenance_path = data_dir / "saffran_provenance.json"

    settings = _settings()
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if npz_path.exists() and provenance_path.exists() and not force:
        stored = json.loads(provenance_path.read_text())
        if stored.get("settings_hash") == settings_hash:
            with np.load(npz_path, allow_pickle=True) as handle:
                factorial = {key: handle[key] for key in handle.files}
            return factorial | _build_exemplar(data_dir, force=force)

    print(
        f"[figure 5] simulating {len(CONDITIONS)} conditions x "
        f"{len(READOUTS)} readouts x {N_SEEDS} seeds "
        f"= {len(CONDITIONS) * len(READOUTS) * N_SEEDS} sessions"
    )

    rows: list[dict[str, Any]] = []
    for layer1_mode, exposure in CONDITIONS:
        for readout in READOUTS:
            for seed in range(N_SEEDS):
                row = run_one(readout, layer1_mode, exposure, seed)
                rows.append(row)
            subset = [
                r for r in rows
                if r["readout"] == readout
                and r["layer1_mode"] == layer1_mode
                and r["exposure"] == exposure
            ]
            print(
                f"    {exposure:<11s} {layer1_mode:<7s} {readout:<12s} "
                f"AUC {np.mean([r['auc'] for r in subset]):.3f}  "
                f"spanning units {np.mean([r['n_span'] for r in subset]):.2f}  "
                f"words covered {np.mean([r['n_words_covered'] for r in subset]):.2f}"
            )

    arrays: dict[str, np.ndarray] = {}
    scalar_fields = (
        "seed", "n_committed", "n_span", "n_words_covered", "max_depth",
        "mean_depth", "auc", "n_rates", "within_mean", "boundary_mean",
        "selectivity", "mean_detector_auc", "n_word_detectors",
    ) + tuple(f"auc_len{length}" for length in WORD_LENGTHS)
    for layer1_mode, exposure in CONDITIONS:
        for readout in READOUTS:
            key = f"{exposure}|{layer1_mode}|{readout}"
            subset = [
                r for r in rows
                if r["readout"] == readout
                and r["layer1_mode"] == layer1_mode
                and r["exposure"] == exposure
            ]
            subset.sort(key=lambda r: r["seed"])
            for field in scalar_fields:
                arrays[f"{key}|{field}"] = np.array(
                    [r[field] for r in subset], dtype=float
                )

    _atomic_npz(npz_path, arrays)
    _write_csv(data_dir, rows)

    provenance = {
        "figure": "Figure 5 - Saffran statistical learning and composition",
        "settings": settings,
        "settings_hash": settings_hash,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "replication_unit": (
            "Exposure order seed. Layer 2 masks start from the same small "
            "random values in every seed, so these are independent exposure "
            "sessions rather than independent networks."
        ),
        "protocol_note": (
            "single_rate and multi_rate share eta, lam, gate_frac, w_init, "
            "commit_frac and n_units; they differ only in the number of "
            "timescales in the trace bank (1 versus 6)."
        ),
        "test_note": (
            "Layer 1 is re-run on the isolated test stream in the same mode "
            "used for exposure, matching the committed protocol in "
            "layer2_multirate.run_saffran."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {},
    }
    for name in ("saffran_figure5_data.npz", "saffran_figure5_runs.csv"):
        path = data_dir / name
        if path.exists():
            provenance["outputs"][name] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
    _atomic_json(provenance_path, provenance)

    with np.load(npz_path, allow_pickle=True) as handle:
        factorial = {key: handle[key] for key in handle.files}
    return factorial | _build_exemplar(data_dir, force=force)


def _write_csv(data_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = data_dir / "saffran_figure5_runs.csv"
    fields = (
        "readout", "layer1_mode", "exposure", "seed", "n_committed", "n_span",
        "n_words_covered", "max_depth", "mean_depth", "auc", "n_rates",
        "within_mean", "boundary_mean", "selectivity", "mean_detector_auc",
        "n_word_detectors",
    ) + tuple(f"auc_len{length}" for length in WORD_LENGTHS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repeat every simulated exposure session.",
    )
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = _parse_args(arguments)
    build_saffran_data(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
