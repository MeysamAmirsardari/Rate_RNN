"""One interval of the stimulus, and the two intervals of one trial.

The figure is a *discrete element*: `coherence` tones whose onsets are
delayed by `step_ms` from one to the next.  Elements never overlap, so the
number of coherent components sounding at once is the same at every step.

The two intervals of a trial are built from the *same* element onsets and the
*same* delays.  They differ in one respect and no other: whether the seven
channels are the same on every element, or drawn afresh each time.  Rhythm,
element count, tone count, spectral extent, summed power per element and
long-term level are identical by construction, not by patching afterwards.
"""

from __future__ import annotations

import numpy as np

from ..sfg.loudness import weights
from .config import Design


# ------------------------------------------------------------------- pool
def erb_number(f: np.ndarray) -> np.ndarray:
    """Glasberg & Moore's ERB-rate scale: distance in critical bands."""
    return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)


def make_pool(d: Design) -> dict:
    span = 12 * np.log2(d.f_hi / d.f_lo)
    st = np.arange(0.0, span + 1e-9, d.grid_st)
    f = d.f_lo * 2.0 ** (st / 12.0)
    amp = weights(f, d.phon) if d.equal_loudness else np.ones_like(f)
    e = erb_number(f)
    near = np.abs(e[:, None] - e[None, :]) < d.min_sep_erb
    np.fill_diagonal(near, False)
    return dict(st=st, f=f, n=st.size, amp=amp / amp.max(), erb=e, near=near)


def _draw(d: Design, pl: dict, band: np.ndarray,
          rng: np.random.Generator) -> np.ndarray | None:
    """`coherence` channels from a band, no two inside one critical band."""
    out, free, e = [], np.ones(band.size, bool), pl["erb"][band]
    for _ in range(d.coherence):
        idx = np.flatnonzero(free)
        if idx.size == 0:
            return None
        j = int(rng.choice(idx))
        out.append(band[j])
        free &= np.abs(e - e[j]) >= d.min_sep_erb
    return np.sort(np.array(out))


def comb(d: Design, pl: dict, rng: np.random.Generator,
         avoid: list[np.ndarray] | None = None) -> np.ndarray:
    """`coherence` channels drawn at random inside a band of fixed width.

    Random rather than evenly spread, as in the published stimuli: an even
    comb is a stack of identical intervals, which is a percept of its own,
    and with enough elements there are not enough distinct positions for a
    figure-absent interval to keep redrawing without repeating itself.  The
    band is a fixed width at a uniformly drawn position, so spectral extent
    is not a cue and the figure is never in a learnable place.  `avoid`
    rejects a draw sharing more than one channel with any earlier element.
    """
    lo = int(round(d.fig_edge_st / d.grid_st))
    width = int(round(d.fig_span_st / d.grid_st))
    hi = pl["n"] - 1 - lo - width
    if hi < 0:
        raise ValueError("figure band does not fit inside the pool")
    for _ in range(500):
        b = lo + int(rng.integers(0, hi + 1))
        ch = _draw(d, pl, np.arange(b, b + width + 1), rng)
        if ch is None:
            continue
        if all(np.intersect1d(ch, a).size <= 1 for a in avoid or ()):
            return ch
    raise ValueError("no figure draw left that avoids the earlier elements")


def comb_gain(pl: dict, ch: np.ndarray, target: float) -> float:
    """Level correction giving every element the same power: `target` per
    tone, whatever channels the draw landed on.

    Two things need it.  Under equal-loudness weighting the channels differ
    by several dB, so a frozen figure would have a constant element power
    while a redrawn one varied -- a modulation cue rather than a coherence
    cue, and the one thing a listener could use without hearing a figure at
    all.  And `target` is the power the *background* actually realises, not
    the pool average: the critical-band rule thins the densely packed low
    channels, so a figure levelled to the pool average comes out 1.4 dB hot
    and leaves a loudness pulse at the element rate.
    """
    return float(np.sqrt(ch.size * target / np.sum(pl["amp"][ch] ** 2)))


# ---------------------------------------------------------- the schedule
def delays(d: Design, step_ms: float, rng: np.random.Generator,
           variant: str) -> np.ndarray:
    """Delay of each figure tone behind the first, in slots."""
    lag = np.arange(d.coherence) * step_ms / d.hop_ms
    if variant == "perm":
        lag = lag[rng.permutation(d.coherence)]
    elif d.order == "fall":
        lag = lag[::-1]
    return np.round(lag).astype(int)


def redrawn(d: Design, step_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Fresh delays with the same extent and the same number of tones.

    Keeps everything the figure has -- seven channels arriving inside the
    same window, the same summed power -- and throws away only the frozen
    pattern.
    """
    span = int(round((d.coherence - 1) * step_ms / d.hop_ms))
    if span == 0:
        return np.zeros(d.coherence, int)
    mid = rng.choice(np.arange(1, span), size=d.coherence - 2, replace=False)
    return np.concatenate(([0], mid, [span]))[rng.permutation(d.coherence)]


def element_onsets(d: Design, rng: np.random.Generator) -> np.ndarray:
    """Irregular element onsets, in slots, with the spacing every condition
    shares: the minimum gap is set by the longest element in the experiment,
    not by this condition's."""
    cuts = np.sort(rng.integers(0, d.slack + 1, d.events))
    parts = np.diff(np.concatenate(([0], cuts, [d.slack])))[:d.events]
    return d.lead + np.cumsum(parts) + np.arange(d.events) * d.min_gap


def scatter_onsets(d: Design, rng: np.random.Generator) -> np.ndarray:
    """`events` onsets per figure channel, ungrouped, with the same minimum
    rest between two tones of a channel that the background obeys."""
    room = d.n_slots - d.tail - d.k - d.lead
    sep = d.k + d.guard
    free = room - (d.events - 1) * sep
    if free < 0:
        raise ValueError("scatter control does not fit in the interval")
    out = []
    for _ in range(d.coherence):
        cuts = np.sort(rng.integers(0, free + 1, d.events - 1))
        parts = np.diff(np.concatenate(([0], cuts, [free])))[:d.events]
        out.append(d.lead + np.cumsum(parts) + np.arange(d.events) * sep)
    return np.array(out)


def schedule(d: Design, pl: dict, rng: np.random.Generator, *,
             step_ms: float, ons: np.ndarray, lag: np.ndarray,
             coherent: bool, variant: str) -> dict:
    """Every tone in the interval: channel, slot, level, figure or not."""
    busy = np.zeros((pl["n"], d.n_slots + d.k + d.guard), bool)
    chan, slot, elem = [], [], []

    def put(c: int, s: int, e: int) -> None:
        chan.append(int(c)), slot.append(int(s)), elem.append(e)
        busy[c, max(0, s - d.guard):s + d.k + d.guard] = True
        busy[pl["near"][c], s:s + d.k] = True

    if variant == "scatter":
        # The same channels at the same rate, never grouped into elements.
        # Long-term spectrum identical to the figure's; no temporal
        # coherence at all.
        combs = [comb(d, pl, rng)]
        for c, ts in zip(combs[0], scatter_onsets(d, rng)):
            for s in ts:
                put(c, s, 0)
    else:
        combs = []
        for i, s0 in enumerate(ons):
            ch = (combs[0] if coherent and combs
                  else comb(d, pl, rng, avoid=None if coherent else combs))
            combs.append(ch)
            lg = redrawn(d, step_ms, rng) if variant == "redraw" else lag
            for c, l in zip(ch, lg):
                put(c, s0 + l, i)
    fig_ch = combs[0]

    # Exactly `bg_sounding` background tones sound at every instant: over any
    # k consecutive slots the pattern below starts exactly that many, and a
    # tone lasts exactly k slots.  It works for any count, not only multiples
    # of the tone length.
    #
    # The background is otherwise blind to the figure -- it keeps its own
    # counts, so its statistics are the same whether the figure recurs or
    # not.  It only avoids what is already sounding: the same channel, and
    # anything within `min_sep_erb` of it, so no two tones ever beat.
    seen = np.zeros(pl["n"])
    n_bg, k = d.bg_sounding, d.k
    for s in range(d.n_slots - k):
        for _ in range((s + 1) * n_bg // k - s * n_bg // k):
            score = seen + rng.random(pl["n"]) * d.dealer_slack
            score[busy[:, s:s + k].any(axis=1)] = np.inf
            c = int(np.argmin(score))
            if not np.isfinite(score[c]):
                raise ValueError(
                    f"slot {s}: nothing free more than {d.min_sep_erb:g} ERB "
                    f"from what is sounding; lower bg_sounding or min_sep_erb")
            put(c, s, -1)
            seen[c] += 1

    chan, slot, elem = np.array(chan), np.array(slot), np.array(elem)
    is_fig = elem >= 0
    # Every element carries the power of `coherence` background tones, taken
    # from the background this interval actually laid down rather than from
    # the pool average.
    target = float(np.mean(pl["amp"][chan[~is_fig]] ** 2))
    gain = np.ones(chan.size)
    for i, ch in enumerate(combs if variant != "scatter" else combs * 1):
        gain[elem == i] = comb_gain(pl, ch, target)
    return dict(chan=chan, slot=slot, gain=gain, is_fig=is_fig,
                fig_ch=fig_ch, onsets=ons, lag=lag, step_ms=step_ms,
                coherent=coherent, variant=variant)


# ---------------------------------------------------------------- render
def render(d: Design, pl: dict, sch: dict, rng: np.random.Generator,
           rove_db: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise, and flatten the loudness.

    The number of tones sounding cannot be held constant: keeping it constant
    while the figure switches seven tones on at once would need at least
    `coherence` tones starting in every slot, which is 1400 onsets a second
    at this resolution and leaves the figure with no contrast at all.  So the
    count is allowed to rise and the *level of everything* is scaled to
    compensate, against the analytic power envelope -- the sum of the tones'
    own squared amplitude envelopes, which is the expected power because the
    tones are mutually incoherent.  The result is flat to 0.01 dB at every
    step and in both intervals, so the loudness pulse that used to mark
    every element -- 2.5 dB at 0 ms, and audible as a thump at the element
    rate -- is gone.

    It has to be the *power* and not the number of tones sounding.  Counting
    is exact only if every tone carries the same power, and under equal
    loudness they do not: the critical-band rule blocks the channels around
    the figure, which pushes the background towards the loud edges of the
    pool for as long as an element lasts, and leaves 0.9 dB behind.  The
    price of levelling the power is that the gain wanders, 1.6 dB SD, which
    jitters each tone about its equal-loudness level by the same amount.
    """
    hop = int(round(d.hop_ms * d.fs / 1000.0))
    n = int(round(d.tone_ms * d.fs / 1000.0))
    r = int(round(d.ramp_ms * d.fs / 1000.0))
    env = np.ones(n)
    ramp = np.sin(np.pi / 2 * np.arange(r) / r) ** 2
    env[:r], env[-r:] = ramp, ramp[::-1]

    t = np.arange(n) / d.fs
    ph = rng.uniform(0, 2 * np.pi, pl["n"])
    pips = (pl["amp"][:, None]
            * np.sin(2 * np.pi * pl["f"][:, None] * t + ph[:, None]) * env)

    y = np.zeros(d.n_slots * hop + n)
    yf, pw = np.zeros_like(y), np.zeros_like(y)
    for c, s, g, f in zip(sch["chan"], sch["slot"], sch["gain"],
                          sch["is_fig"]):
        a = s * hop
        y[a:a + n] += g * pips[c]
        pw[a:a + n] += (g * pl["amp"][c] * env) ** 2
        if f:
            yf[a:a + n] += g * pips[c]
    y, yf, pw = (z[:d.n_slots * hop] for z in (y, yf, pw))

    target = np.median(pw[d.k * hop:-d.k * hop])
    flat = np.sqrt(target / np.maximum(pw, 0.3 * target))
    y, yf = y * flat, yf * flat
    inner = flat[d.k * hop:-d.k * hop]
    sch["flat_db"] = float(20 * np.log10(inner.max() / inner.min()))

    e = int(round(d.edge_ms * d.fs / 1000.0))
    ramp = np.sin(np.pi / 2 * np.arange(e) / e) ** 2
    for z in (y, yf):
        z[:e] *= ramp
        z[-e:] *= ramp[::-1]

    # one gain for both, so the figure-only file is exactly the figure you
    # hear inside the mix rather than a louder version of it
    g = 10 ** (d.rms_dbfs / 20) / np.sqrt(np.mean(y ** 2))
    if rove_db:
        g *= 10 ** (rng.uniform(-rove_db, rove_db) / 20)
    return y * g, yf * g


def interval(d: Design, pl: dict, *, step_ms: float, ons: np.ndarray,
             lag: np.ndarray, coherent: bool, variant: str,
             seed: int, rove: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    sch = schedule(d, pl, rng, step_ms=step_ms, ons=ons, lag=lag,
                   coherent=coherent, variant=variant)
    sch["y"], sch["y_fig"] = render(d, pl, sch, rng,
                                    d.rove_db if rove else 0.0)
    return sch


def trial(d: Design, pl: dict, *, step_ms: float, seed: int,
          variant: str = "rise", rove: bool = True) -> tuple[dict, dict]:
    """The figure-present and figure-absent intervals of one trial.

    Both are built from the same element onsets and the same delays, drawn
    once here, so the only difference between them is coherence.
    """
    rng = np.random.default_rng(seed)
    ons = element_onsets(d, rng)
    lag = delays(d, step_ms, rng, variant)
    kw = dict(step_ms=step_ms, ons=ons, lag=lag, variant=variant, rove=rove)
    s1, s2 = rng.integers(0, 2 ** 31, 2)
    return (interval(d, pl, coherent=True, seed=int(s1), **kw),
            interval(d, pl, coherent=False, seed=int(s2), **kw))
