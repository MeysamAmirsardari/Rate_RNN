"""
tasks.sfg2.stimulus
====================

Faithful Stochastic Figure-Ground (SFG) stimulus generator for model0,
ported from the Baphy ``fgFrozen`` deconstruction of

    Lu, Dutta, Mohammed, Elhilali & Shamma (2025), iScience 28:111991,
    "Temporal-coherence induces binding ...", Experiment II.

One PRESENTATION = three concatenated 5 s epochs (the SelectAttention
trial), optionally padded by 0.5 s silences (16 s total):

    [ pre-silence 0.5 s ]
    pre    : 5 s pure random tone-cloud          (no figure)
    figure : 5 s tone-cloud + coherent figure     (synchronous chords)
    post   : 5 s pure random tone-cloud          (no figure)
    [ post-silence 0.5 s ]

The three 5 s blocks concatenate seamlessly (no inter-epoch silence);
the figure is present ONLY in the middle epoch, so the pre/post epochs
are a built-in no-figure control.

Tone-cloud (background) -- independent per channel
    Mean 4 Hz.  Onsets placed on a 50 ms grid in 2 s windows (8 onsets
    per window, drawn without replacement -> >= 50 ms spacing), then
    jittered +/- 25 ms.  Generate 6 s, keep the first 5 s, drop pips
    whose 50 ms duration crosses the 5 s border.

Figure (coherent tone complex)
    A FROZEN subset of {4, 6, 8, 10} channels -- the first ``n`` of ten
    frozen channel indices -- fire a SYNCHRONOUS 50 ms chord at a shared
    list of onset times: mean 4 Hz, >= 50 ms spacing, +/- 25 ms jitter,
    forced to include t = 0 and t = 4950 ms.  Background pips within
    +/- 50 ms of a figure onset are dropped so the overall envelope
    stays flat.

Output: a binary ``(T, n_channels)`` matrix at dt = 1 ms (== model0's
dt), values in {0, 1}.  A pip is a 50 ms ON state in one channel.  Use
``build_session`` for the model-ready ``(n_channels, T)`` transpose.

Frozen vs fresh
---------------
For the *weight* readout the binding mechanism requires the coherent
FIGURE to repeat (frozen onsets + frozen channels, so figure<->figure
co-firings accumulate W_FF) while the random GROUND is drawn FRESH every
presentation (ground co-firings are random pairs that average out, so
W_GG stays flat).  ``build_session`` freezes the figure and re-seeds the
cloud per presentation accordingly.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------
#  Frequency grid (37 log-spaced channels; 1600 Hz center, 3 oct, 12/oct)
# ---------------------------------------------------------------------
N_CHANNELS    = 37
CENTER_FREQ_HZ = 1600.0
STEPS_PER_OCT = 12
FREQS_HZ = CENTER_FREQ_HZ * 2.0 ** (np.arange(-12, 25) / STEPS_PER_OCT)   # 800..6400 Hz

# ---------------------------------------------------------------------
#  Time base (dt = 1 ms == model0 dt, so 1 sample == 1 step == 1 ms)
# ---------------------------------------------------------------------
DT_MS            = 1
TONE_DUR_MS      = 50
TONE_RATE_HZ     = 4.0
MIN_ONSET_GAP_MS = 50
ONSET_JITTER_MS  = 25

# ---------------------------------------------------------------------
#  Epoch durations (ms)
# ---------------------------------------------------------------------
PRE_MS     = 5000
FIGURE_MS  = 5000
POST_MS    = 5000
SILENCE_MS = 500                       # pre/post-trial silence pad

# ---------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------
N_FIG_OPTS = (4, 6, 8, 10)
# Frozen Baphy realization, 0-indexed (Baphy [17 36 24 10 28 3 22 5 33 14] - 1).
# Each condition uses the first n_fig of these ten -> nested figure sets.
FIG_CHANNELS_10     = np.array([16, 35, 23, 9, 27, 2, 21, 4, 32, 13])

# ---------------------------------------------------------------------
#  Drive matching
# ---------------------------------------------------------------------
# Every channel receives exactly this many pips in every 5 s epoch, figure
# and ground alike.  Sampling each channel independently at a nominal 4 Hz
# left a residual imbalance (19.0 pips on figure channels against 18.2 on
# ground, with a +/-1.5 spread across channels), so figure and ground were
# not quite matched on drive and a raw rate comparison was not clean.
PIPS_PER_EPOCH = int(round(TONE_RATE_HZ * PRE_MS / 1000.0))      # 20
FIG_FORCE_FIRST_MS  = 0
FIG_FORCE_LAST_MS   = 4950


# =====================================================================
#  Onset generators
# =====================================================================
#: Slots are drawn at least this far apart so that +/-25 ms of jitter can
#: never make two pips in a channel overlap.  Drawing on the bare 50 ms grid
#: cannot guarantee that, because a tone is itself 50 ms long: adjacent picks
#: are already touching, and any jitter makes them collide.  Repairing the
#: collision afterwards pushed later pips off the end of the epoch and lost
#: two or three per channel, which is what left the counts unequal.
MIN_PICK_SEPARATION_MS = 2 * MIN_ONSET_GAP_MS


def _greedy_slots(rng: np.random.Generator, allowed: np.ndarray,
                  n_onsets: int, separation: float) -> np.ndarray | None:
    order = rng.permutation(allowed)
    chosen: List[float] = []
    for candidate in order:
        if all(abs(candidate - taken) >= separation for taken in chosen):
            chosen.append(float(candidate))
            if len(chosen) == n_onsets:
                return np.sort(np.asarray(chosen))
    return None


def chord_slot_weight(n_fig: int, n_slots: int = PRE_MS // MIN_ONSET_GAP_MS,
                      n_chord: int = PIPS_PER_EPOCH) -> float:
    """How often a ground pip may still land on a chord slot.

    Excluding chord slots outright leaves the chord moments *emptier* than the
    rest of the epoch, which is as good a cue as leaving them fuller. What
    should stay flat is the expected number of simultaneously active channels.
    A slot holds ``target`` active channels on average; the figure already
    supplies ``n_fig`` of them at a chord, so the ground should supply
    ``target - n_fig`` there and the usual amount elsewhere. Solving for the
    sampling weight gives the expression below, which is zero once the figure
    alone already exceeds the target -- for large figures a residual bump is
    unavoidable.
    """

    target = N_CHANNELS * PIPS_PER_EPOCH / n_slots      # 7.4 channels
    wanted = max(0.0, target - n_fig)
    if wanted <= 0.0:
        return 0.0
    share = wanted / (N_CHANNELS - n_fig)               # per ground channel
    others = n_slots - n_chord
    denominator = n_chord * (1.0 - share)
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(share * others / denominator, 0.0, 1.0))


def sample_onsets(rng: np.random.Generator, n_onsets: int, duration_ms: int,
                  avoid: np.ndarray | None = None,
                  avoid_weight: float = 0.0) -> np.ndarray:
    """Exactly ``n_onsets`` pip onsets on the 50 ms grid, then jittered.

    ``avoid`` lists chord onsets whose 50 ms windows the pip should not
    overlap.  Excluding those slots is what lets the ground cloud fill the
    silences *between* chords instead of piling on top of them, which is
    what keeps the number of simultaneously active channels roughly flat
    across the epoch.  Jitter that would push a pip back onto a chord is
    dropped for that pip rather than allowed to reintroduce the overlap.
    """

    slots = np.arange(0, duration_ms - TONE_DUR_MS + 1, MIN_ONSET_GAP_MS,
                      dtype=float)
    slots = slots[(slots >= ONSET_JITTER_MS)
                  & (slots <= duration_ms - TONE_DUR_MS - ONSET_JITTER_MS)]
    if avoid is not None and np.size(avoid):
        clash = np.min(np.abs(slots[:, None] - np.asarray(avoid, float)[None, :]),
                       axis=1) < TONE_DUR_MS + ONSET_JITTER_MS
        keep = ~clash | (rng.random(slots.size) < avoid_weight)
        if keep.sum() >= n_onsets:
            slots = slots[keep]

    picked = None
    for _attempt in range(400):
        picked = _greedy_slots(rng, slots, n_onsets, MIN_PICK_SEPARATION_MS)
        if picked is not None:
            break
    if picked is None:                       # never seen; kept as a guard
        picked = np.sort(rng.choice(slots, size=n_onsets, replace=False))

    jittered = picked + rng.uniform(-ONSET_JITTER_MS, ONSET_JITTER_MS,
                                    size=picked.size)
    return np.sort(jittered)


def channel_cloud_onsets(rng: np.random.Generator,
                         duration_ms: int = PRE_MS) -> np.ndarray:
    """Onset times (ms) for one channel of the tone cloud."""

    return sample_onsets(rng, PIPS_PER_EPOCH, duration_ms)


def figure_onsets(rng: np.random.Generator) -> np.ndarray:
    """Shared figure-CHORD onset times (ms), forced to span 0..4950 ms.

    Mean 4 Hz over the 5 s figure, >= 50 ms spacing, +/- 25 ms jitter.
    """
    forced = np.array([float(FIG_FORCE_FIRST_MS), float(FIG_FORCE_LAST_MS)])
    grid = np.arange(MIN_PICK_SEPARATION_MS,
                     FIGURE_MS - TONE_DUR_MS - MIN_PICK_SEPARATION_MS,
                     MIN_ONSET_GAP_MS, dtype=float)
    picked = None
    for _attempt in range(400):
        picked = _greedy_slots(rng, grid, PIPS_PER_EPOCH - forced.size,
                               MIN_PICK_SEPARATION_MS)
        if picked is not None:
            break
    jittered = picked + rng.uniform(-ONSET_JITTER_MS, ONSET_JITTER_MS,
                                    size=picked.size)
    return np.sort(np.clip(np.r_[forced, jittered], 0.0,
                           FIGURE_MS - TONE_DUR_MS))


# =====================================================================
#  Epoch builders
# =====================================================================
def make_cloud(rng: np.random.Generator, dur_ms: int = 5000,
               drop_onsets_ms: np.ndarray = None,
               drop_radius_ms: int = TONE_DUR_MS,
               avoid: np.ndarray = None,
               avoid_weight: float = 0.0) -> np.ndarray:
    """``(dur_ms, N_CHANNELS)`` binary tone-cloud (independent channels).

    ``drop_onsets_ms`` : figure-chord onsets to avoid -- any background
    pip whose onset lies within ``drop_radius_ms`` of one is dropped
    (envelope flatness).
    """
    M = np.zeros((dur_ms, N_CHANNELS), dtype=np.uint8)
    drops = None if drop_onsets_ms is None else np.asarray(drop_onsets_ms, float)
    for ch in range(N_CHANNELS):
        for onset in sample_onsets(rng, PIPS_PER_EPOCH, dur_ms, avoid=avoid,
                                   avoid_weight=avoid_weight):
            t0 = int(round(onset)); t1 = t0 + TONE_DUR_MS
            if t0 < 0 or t1 > dur_ms:
                continue
            if drops is not None and drops.size and \
               np.any(np.abs(t0 - drops) < drop_radius_ms):
                continue
            M[t0:t1, ch] = 1
    return M


def make_figure_epoch(rng_fig: np.random.Generator,
                      rng_cloud: np.random.Generator,
                      n_fig: int,
                      rate_matched: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(FIGURE_MS, N_CHANNELS)`` cloud + synchronous figure chords.

    ``rng_fig`` (frozen across presentations) sets the figure onsets;
    ``rng_cloud`` (fresh) sets the surrounding background.

    ``rate_matched`` (default)
        COHERENCE-ONLY figure.  Figure channels fire ONLY the synchronous
        chord (~mean 4 Hz) and carry NO independent background; ground
        channels carry an independent cloud at the same mean rate.  Figure
        and ground channels then have matched per-channel statistics
        (same pip count, flat time-marginal) and differ ONLY in temporal
        coherence -- so any figure enhancement is attributable to
        coherence, not to extra thalamic drive.  This is the principled
        substrate for testing "temporal coherence induces binding".

    ``rate_matched = False``
        ADDITIVE figure (the original baphy ``fgFrozen`` form).  The chord
        is added on top of a full independent cloud and background pips
        within +/-50 ms of a chord onset are dropped to flatten the global
        envelope.  Figure channels then fire MORE than ground (chord +
        surviving background) -- the global energy is matched but the
        per-channel rate is not, which confounds the figure-vs-ground
        comparison.  Kept only for fidelity to the recording stimulus.
    """
    fig_channels = FIG_CHANNELS_10[:n_fig]
    onsets = figure_onsets(rng_fig)

    M = np.zeros((FIGURE_MS, N_CHANNELS), dtype=np.uint8)
    starts: List[int] = []
    for onset in onsets:
        t0 = int(round(onset)); t1 = t0 + TONE_DUR_MS
        if t0 < 0 or t1 > FIGURE_MS:
            continue
        M[t0:t1, fig_channels] = 1                     # synchronous chord
        starts.append(t0)
    starts = np.asarray(starts)

    if rate_matched:
        # Ground pips avoid the chord windows, so they fill the silences
        # between chords rather than stacking on them.  Each ground channel
        # still receives exactly PIPS_PER_EPOCH pips, so the per-channel
        # drive stays matched to the figure channels.
        cloud = make_cloud(rng_cloud, FIGURE_MS, avoid=starts,
                           avoid_weight=chord_slot_weight(n_fig))
        gnd_mask = np.ones(N_CHANNELS, dtype=bool)
        gnd_mask[fig_channels] = False
        M[:, gnd_mask] = cloud[:, gnd_mask]            # ground only; figure = chord
    else:
        cloud = make_cloud(rng_cloud, FIGURE_MS, drop_onsets_ms=starts)
        np.maximum(M, cloud, out=M)
    return M, fig_channels, starts


def build_presentation(n_fig: int, fig_seed: int, cloud_seed: int,
                       with_silence: bool = True,
                       rate_matched: bool = True) -> Tuple[np.ndarray, Dict]:
    """One SFG presentation ``(T, N_CHANNELS)`` + epoch metadata.

    The figure (onsets + channels) is determined by ``fig_seed`` only,
    so a constant ``fig_seed`` freezes it across presentations; the
    cloud is seeded by ``cloud_seed`` (vary it per presentation for a
    fresh background).  ``rate_matched`` -> see ``make_figure_epoch``.
    """
    rng_fig  = np.random.default_rng(fig_seed)
    rng_pre  = np.random.default_rng(cloud_seed)
    rng_figc = np.random.default_rng(cloud_seed + 1)
    rng_post = np.random.default_rng(cloud_seed + 2)

    pre  = make_cloud(rng_pre,  PRE_MS)
    figM, fig_channels, fig_starts = make_figure_epoch(
        rng_fig, rng_figc, n_fig, rate_matched=rate_matched)
    post = make_cloud(rng_post, POST_MS)

    parts: List[np.ndarray] = []
    bounds: Dict[str, Tuple[int, int]] = {}
    pos = 0
    if with_silence:
        parts.append(np.zeros((SILENCE_MS, N_CHANNELS), np.uint8)); pos += SILENCE_MS
    bounds["pre"]    = (pos, pos + PRE_MS);    parts.append(pre);  pos += PRE_MS
    bounds["figure"] = (pos, pos + FIGURE_MS); parts.append(figM); pos += FIGURE_MS
    bounds["post"]   = (pos, pos + POST_MS);   parts.append(post); pos += POST_MS
    if with_silence:
        parts.append(np.zeros((SILENCE_MS, N_CHANNELS), np.uint8)); pos += SILENCE_MS

    M = np.vstack(parts)
    meta = dict(bounds=bounds, fig_channels=fig_channels,
                fig_starts=fig_starts + bounds["figure"][0],   # global within presentation
                n_fig=n_fig, T=M.shape[0])
    return M, meta


# =====================================================================
#  Session: repeated presentations (frozen figure, fresh cloud)
# =====================================================================
def build_session(n_fig: int, n_reps: int, base_seed: int = 0,
                  fig_seed: int = 12345, with_silence: bool = True,
                  rate_matched: bool = True
                  ) -> Tuple[np.ndarray, Dict]:
    """Concatenate ``n_reps`` presentations into a model-ready stimulus.

    ``rate_matched`` (default True) -> figure channels carry only the
    coherent chord, ground channels an independent cloud at the same
    mean rate; see ``make_figure_epoch``.

    Returns
    -------
    stim : (N_CHANNELS, T) float in {0, 1}
    sess : dict with per-presentation epoch bounds (global step indices),
           figure/ground channel indices, etc.
    """
    blocks: List[np.ndarray] = []
    reps: List[Dict] = []
    pos = 0
    for r in range(n_reps):
        M, meta = build_presentation(
            n_fig, fig_seed=fig_seed, cloud_seed=base_seed + 1000 * (r + 1),
            with_silence=with_silence, rate_matched=rate_matched)
        gbounds = {k: (lo + pos, hi + pos) for k, (lo, hi) in meta["bounds"].items()}
        reps.append(dict(bounds=gbounds, start=pos, T=meta["T"],
                         fig_starts=meta["fig_starts"] + pos))
        blocks.append(M)
        pos += M.shape[0]

    stim = np.vstack(blocks).T.astype(float)               # (N, T)
    fig_idx = FIG_CHANNELS_10[:n_fig]
    gnd_idx = np.setdiff1d(np.arange(N_CHANNELS), fig_idx)
    sess = dict(reps=reps, n_fig=n_fig, n_reps=n_reps,
                fig_idx=fig_idx, gnd_idx=gnd_idx,
                T=pos, with_silence=with_silence, rate_matched=rate_matched)
    return stim, sess
