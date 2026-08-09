"""Counterbalanced probability-reversal analysis of the Nugmeg SEQ recordings.

Why this module exists
----------------------
The 2026-04-30/05-01 AB/BA recordings cannot be analysed at the trial level:
their per-sequence playback table (``outp.stimat``, built by the missing
``ft_oe_list``) is absent, the TTL file marks only 16 blocks, and the ADC
channels are sampled at 4 kHz so they cannot carry the 5.3/9.4 kHz tones.
See ``final_figures/figure_4_ab_ba/ECOG_INFERENCE_AUDIT.md``.

Two *complete* ``ft_oe_list`` outputs do exist for the SEQ runclass, in
``/Users/eminent/Projects/ECoG/ECoG_data``:

    Nugmeg_2025-03-03_12-31-33.mat        stim 1 = 20%, stim 2 = 80%
    Nugmeg_2025-03-03_13-35-18_SEQ3.mat   stim 1 = 80%, stim 2 = 20%

Same animal, same day, about one hour apart, 10 acquisition trials of 25
sequences each, 32 channels at 1 kHz, and a full ``stimat`` playback table.
The two physical sequences are

    stim 1  '800--800--800--800-6000'   ends on a 6 kHz note
    stim 2  '800-800-800-800-800'       ends on 800 Hz

and their probabilities are *reversed* between the recordings, so the same
physical sequence appears as rare in one and frequent in the other.

The contrast
------------
Per acquisition trial, ``d = response(stim 1) - response(stim 2)`` holds the
acoustics fixed. The counterbalanced interaction

    d(rec_a) - d(rec_b) = (stim1 rare - stim1 frequent)
                        + (stim2 rare - stim2 frequent)

cancels the acoustic difference and any *additive* recording-level offset. It
is twice the average rarity effect.

What this module concludes
--------------------------
It runs the analysis to completion and reports the result honestly, including
the diagnostics that make the significant term uninterpretable as a probability
effect. See ``SEQ_COUNTERBALANCED_FINDINGS.md``. The headline is a design fact,
not a missing-file fact: probability is manipulated *between* recordings, so
with two recordings the acquisition trials are pseudo-replicates and the
effective replication for the manipulation is one against one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


SOURCE_DIR = Path("/Users/eminent/Projects/ECoG/ECoG_data")
RECORDINGS = {
    "rec_a": SOURCE_DIR / "Nugmeg_2025-03-03_12-31-33.mat",
    "rec_b": SOURCE_DIR / "Nugmeg_2025-03-03_13-35-18_SEQ3.mat",
}
RESULT_DIR = Path(__file__).resolve().parent / "results" / "seq_counterbalanced"

BASELINE_MS = 100          # Gen_M2Mat.m: basetime = 0.1 s
POST_MS = 1500             # inside the 3500 ms SOA
N_PERMUTATIONS = 4_999
RANDOM_SEED = 20_260_809
ALPHA = 0.05
#: The deviant note is the last one; its response peaks near 624 ms.
DEVIANT_WINDOW_MS = (550, 900)


@dataclass(frozen=True)
class Recording:
    key: str
    epochs: np.ndarray         # sequence x channel x time, baseline-normalised
    trial: np.ndarray
    stim: np.ndarray
    time_ms: np.ndarray
    pct: np.ndarray
    names: tuple[str, ...]

    @property
    def rare_stim(self) -> int:
        return int(np.argmin(self.pct)) + 1


def load_recording(key: str, path: Path) -> Recording:
    """Epoch one recording exactly as the SEQ branch of ``Gen_M2Mat.m`` does.

    Epochs are taken at the ``stimat`` onsets, which for regularly spaced
    sequences are precisely the onsets ``Gen_M2Mat`` reconstructs as
    ``tmp(j) + SOA*(ii-1)``; the regular spacing is asserted. Each epoch is
    baseline-corrected by the mean of its 100 ms pre-onset window and divided
    by that window's standard deviation, as in the source.
    """

    out = sio.loadmat(path, struct_as_record=False, squeeze_me=True)["xx"].outp
    if int(out.fs) != 1000:
        raise ValueError(f"{key}: expected 1 kHz, found {out.fs} Hz")

    data = np.asarray(out.dataMat, dtype=np.float64)      # time x channel x trial
    stimat = np.asarray(out.stimat, dtype=np.int64)       # trial, stim, on, off
    trial, stim, onset = stimat[:, 0], stimat[:, 1], stimat[:, 2]

    spacing = np.unique(np.diff(onset[trial == 1]))
    if spacing.size != 1:
        raise ValueError(f"{key}: irregular sequence spacing {spacing}")

    n_time = BASELINE_MS + POST_MS
    epochs = np.empty((stimat.shape[0], data.shape[1], n_time), dtype=np.float64)
    for row, (this_trial, this_onset) in enumerate(zip(trial, onset)):
        lo = this_onset - BASELINE_MS
        segment = data[lo:lo + n_time, :, this_trial - 1].T
        baseline = segment[:, :BASELINE_MS]
        scale = baseline.std(axis=1, ddof=0)
        if not np.all(scale > 0):
            raise ValueError(f"{key}: flat baseline in epoch {row}")
        epochs[row] = (segment - baseline.mean(axis=1, keepdims=True)) / scale[:, None]

    return Recording(
        key=key, epochs=epochs, trial=trial, stim=stim,
        time_ms=np.arange(-BASELINE_MS, POST_MS),
        pct=np.asarray(out.stims.pct, dtype=float),
        names=tuple(str(s) for s in np.atleast_1d(out.stims.stims)),
    )


def per_trial_responses(recording: Recording) -> np.ndarray:
    """Mean response per acquisition trial and stimulus: trial x 2 x ch x time."""

    trials = np.unique(recording.trial)
    return np.stack([
        [recording.epochs[(recording.trial == t) & (recording.stim == s)].mean(axis=0)
         for s in (1, 2)]
        for t in trials
    ])


def two_sample_t(values: np.ndarray, is_a: np.ndarray) -> np.ndarray:
    """Pooled-variance two-sample t over the leading (unit) axis."""

    a, b = values[is_a], values[~is_a]
    na, nb = a.shape[0], b.shape[0]
    pooled = ((na - 1) * a.var(axis=0, ddof=1)
              + (nb - 1) * b.var(axis=0, ddof=1)) / (na + nb - 2)
    se = np.sqrt(pooled * (1.0 / na + 1.0 / nb))
    return (a.mean(axis=0) - b.mean(axis=0)) / np.where(se > 0, se, np.inf)


def max_statistic_test(values: np.ndarray, is_a: np.ndarray, *,
                       n_permutations: int = N_PERMUTATIONS,
                       seed: int = RANDOM_SEED) -> dict[str, np.ndarray]:
    """Permutation max-|t| family-wise test over every element of the map.

    The permuted quantity is the recording label of an acquisition trial. Read
    the caveat in the module docstring before interpreting the result: this
    null says trials are exchangeable between recordings, which is weaker than
    the scientific null that probability had no effect.
    """

    observed = two_sample_t(values, is_a)
    rng = np.random.default_rng(seed)
    n_units, n_a = values.shape[0], int(is_a.sum())
    maxima = np.empty(n_permutations)
    for index in range(n_permutations):
        permuted = np.zeros(n_units, dtype=bool)
        permuted[rng.permutation(n_units)[:n_a]] = True
        maxima[index] = np.abs(two_sample_t(values, permuted)).max()
    critical = float(np.quantile(maxima, 1.0 - ALPHA))
    p_map = np.array([
        (1.0 + np.count_nonzero(maxima >= abs(value))) / (n_permutations + 1.0)
        for value in observed.ravel()
    ]).reshape(observed.shape)
    return {"t": observed, "p_fwer": p_map, "critical": np.array([critical]),
            "null_max": maxima}


def gain_diagnostic(a: Recording, b: Recording, channel: int) -> dict[str, float]:
    """Do the two recordings differ where the stimuli are acoustically identical?

    Both sequences open with the same 800 Hz note, so the 0-130 ms window is
    matched across stimulus *and* across recording. Any difference there is a
    recording-level gain difference, which the additive interaction does not
    remove when the acoustic contrast is large.
    """

    window = (a.time_ms >= 0) & (a.time_ms < 130)
    amplitude_a = float(np.abs(a.epochs[:, channel, window]).mean())
    amplitude_b = float(np.abs(b.epochs[:, channel, window]).mean())
    gfp_a = float(a.epochs.std(axis=1)[:, window].mean())
    gfp_b = float(b.epochs.std(axis=1)[:, window].mean())
    return {
        "channel_amplitude_ratio": amplitude_a / amplitude_b,
        "population_gfp_ratio": gfp_a / gfp_b,
    }


def run(result_dir: Path | None = None) -> dict[str, object]:
    result_dir = Path(result_dir or RESULT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)

    a = load_recording("rec_a", RECORDINGS["rec_a"])
    b = load_recording("rec_b", RECORDINGS["rec_b"])
    if a.names != b.names:
        raise ValueError("recordings do not share a stimulus set")
    if a.rare_stim == b.rare_stim:
        raise ValueError("recordings are not counterbalanced")

    responses_a, responses_b = per_trial_responses(a), per_trial_responses(b)
    contrast = np.vstack([responses_a[:, 0] - responses_a[:, 1],
                          responses_b[:, 0] - responses_b[:, 1]])
    is_a = np.r_[np.ones(responses_a.shape[0], bool),
                 np.zeros(responses_b.shape[0], bool)]

    post = a.time_ms >= 0
    result = max_statistic_test(contrast[:, :, post], is_a)
    significant = result["p_fwer"] < ALPHA
    channels = np.flatnonzero(significant.any(axis=1))
    peak = np.unravel_index(np.abs(result["t"]).argmax(), result["t"].shape)
    time_post = a.time_ms[post]

    diagnostic = gain_diagnostic(a, b, int(peak[0]))
    mean_1 = np.vstack([responses_a[:, 0, peak[0]], responses_b[:, 0, peak[0]]]).mean(0)
    mean_2 = np.vstack([responses_a[:, 1, peak[0]], responses_b[:, 1, peak[0]]]).mean(0)
    acoustic = (mean_1 - mean_2)[post]
    interaction = (contrast[is_a, peak[0]].mean(0)
                   - contrast[~is_a, peak[0]].mean(0))[post]

    summary = {
        "stimuli": list(a.names),
        "probabilities": {"rec_a": a.pct.tolist(), "rec_b": b.pct.tolist()},
        "n_trials": {"rec_a": int(responses_a.shape[0]),
                     "rec_b": int(responses_b.shape[0])},
        "n_sequences_per_trial": 25,
        "critical_abs_t": float(result["critical"][0]),
        "significant_channels": (channels + 1).tolist(),
        "peak": {
            "channel": int(peak[0]) + 1,
            "time_ms": int(time_post[peak[1]]),
            "t": float(result["t"][peak]),
            "p_fwer": float(result["p_fwer"][peak]),
        },
        "diagnostics": {
            **diagnostic,
            "interaction_over_acoustic_at_peak": float(
                interaction[peak[1]] / acoustic[peak[1]]),
        },
        "interpretation": (
            "The max-statistic test is exact for the null that acquisition "
            "trials are exchangeable between recordings. It is NOT a test of "
            "the probability manipulation: probability was varied between "
            "recordings, so trials are pseudo-replicates and the effective "
            "replication is one recording against one. The diagnostics show "
            "the two recordings differ in gain where the stimuli are "
            "acoustically identical, and the surviving term is a large "
            "fraction of the acoustic contrast itself."
        ),
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        result_dir / "interaction.npz",
        time_ms=time_post, t=result["t"], p_fwer=result["p_fwer"],
        critical=result["critical"], contrast=contrast, is_a=is_a,
        acoustic_difference=acoustic, interaction=interaction,
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
