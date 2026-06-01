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
FIG_FORCE_FIRST_MS  = 0
FIG_FORCE_LAST_MS   = 4950


# =====================================================================
#  Onset generators
# =====================================================================
def channel_cloud_onsets(rng: np.random.Generator,
                         duration_ms: int = 6000) -> np.ndarray:
    """Onset times (ms) for one channel of the tone cloud.

    Three 2 s windows; in each, 8 onsets are drawn without replacement
    from the 40 50-ms slots (guaranteeing >= 50 ms spacing), then
    jittered +/- 25 ms.
    """
    n_per_window = int(round(2.0 * TONE_RATE_HZ))               # 8
    slots = np.arange(0, 2000, MIN_ONSET_GAP_MS)                # 40 slots
    out: List[float] = []
    for w0 in range(0, duration_ms, 2000):
        picks = np.sort(rng.choice(slots, size=n_per_window, replace=False))
        jit = rng.uniform(-ONSET_JITTER_MS, ONSET_JITTER_MS, size=n_per_window)
        out.extend((w0 + picks + jit).tolist())
    return np.asarray(out)


def figure_onsets(rng: np.random.Generator) -> np.ndarray:
    """Shared figure-CHORD onset times (ms), forced to span 0..4950 ms.

    Mean 4 Hz over the 5 s figure, >= 50 ms spacing, +/- 25 ms jitter.
    """
    onsets = {float(FIG_FORCE_FIRST_MS), float(FIG_FORCE_LAST_MS)}
    target_n = int(round((FIGURE_MS / 1000.0) * TONE_RATE_HZ))  # ~20
    grid = np.arange(MIN_ONSET_GAP_MS, FIGURE_MS - MIN_ONSET_GAP_MS, MIN_ONSET_GAP_MS)
    rng.shuffle(grid)
    for cand in grid:
        if len(onsets) >= target_n:
            break
        if all(abs(cand - o) >= MIN_ONSET_GAP_MS for o in onsets):
            onsets.add(cand + float(rng.uniform(-ONSET_JITTER_MS, ONSET_JITTER_MS)))
    return np.asarray(sorted(onsets))


# =====================================================================
#  Epoch builders
# =====================================================================
def make_cloud(rng: np.random.Generator, dur_ms: int = 5000,
               drop_onsets_ms: np.ndarray = None,
               drop_radius_ms: int = TONE_DUR_MS) -> np.ndarray:
    """``(dur_ms, N_CHANNELS)`` binary tone-cloud (independent channels).

    ``drop_onsets_ms`` : figure-chord onsets to avoid -- any background
    pip whose onset lies within ``drop_radius_ms`` of one is dropped
    (envelope flatness).
    """
    M = np.zeros((dur_ms, N_CHANNELS), dtype=np.uint8)
    drops = None if drop_onsets_ms is None else np.asarray(drop_onsets_ms, float)
    for ch in range(N_CHANNELS):
        for onset in channel_cloud_onsets(rng, 6000):
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
                      n_fig: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(FIGURE_MS, N_CHANNELS)`` cloud + synchronous figure chords.

    ``rng_fig`` (frozen across presentations) sets the figure onsets;
    ``rng_cloud`` (fresh) sets the surrounding background.
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

    cloud = make_cloud(rng_cloud, FIGURE_MS, drop_onsets_ms=starts)
    np.maximum(M, cloud, out=M)
    return M, fig_channels, starts


def build_presentation(n_fig: int, fig_seed: int, cloud_seed: int,
                       with_silence: bool = True) -> Tuple[np.ndarray, Dict]:
    """One SFG presentation ``(T, N_CHANNELS)`` + epoch metadata.

    The figure (onsets + channels) is determined by ``fig_seed`` only,
    so a constant ``fig_seed`` freezes it across presentations; the
    cloud is seeded by ``cloud_seed`` (vary it per presentation for a
    fresh background).
    """
    rng_fig  = np.random.default_rng(fig_seed)
    rng_pre  = np.random.default_rng(cloud_seed)
    rng_figc = np.random.default_rng(cloud_seed + 1)
    rng_post = np.random.default_rng(cloud_seed + 2)

    pre  = make_cloud(rng_pre,  PRE_MS)
    figM, fig_channels, fig_starts = make_figure_epoch(rng_fig, rng_figc, n_fig)
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
                  fig_seed: int = 12345, with_silence: bool = True
                  ) -> Tuple[np.ndarray, Dict]:
    """Concatenate ``n_reps`` presentations into a model-ready stimulus.

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
            with_silence=with_silence)
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
                T=pos, with_silence=with_silence)
    return stim, sess
