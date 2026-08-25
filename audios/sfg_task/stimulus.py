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
def make_pool(d: Design) -> dict:
    span = 12 * np.log2(d.f_hi / d.f_lo)
    st = np.arange(0.0, span + 1e-9, d.grid_st)
    f = d.f_lo * 2.0 ** (st / 12.0)
    amp = weights(f, d.phon)
    return dict(st=st, f=f, n=st.size, amp=amp / amp.max())


def comb(d: Design, pl: dict, rng: np.random.Generator,
         avoid: list[np.ndarray] | None = None) -> np.ndarray:
    """`coherence` channels evenly spread, at a position drawn uniformly.

    Uniform so the figure is never in a learnable place; evenly spread and
    always the same width so its spacing and spectral extent cannot be the
    cue.  `avoid` rejects a position sharing more than one channel with the
    last one, which is what stops a figure-absent interval from stumbling
    into a two-element figure of its own.
    """
    lo = int(round(d.fig_edge_st / d.grid_st))
    hi = pl["n"] - 1 - lo
    width = int(round(d.fig_span_st / d.grid_st))
    for b in rng.permutation(np.arange(lo, hi - width + 1)):
        ch = np.round(np.linspace(b, b + width, d.coherence)).astype(int)
        if all(np.intersect1d(ch, a).size <= 1 for a in avoid or ()):
            return ch
    raise ValueError("no figure position left that avoids the earlier ones")


def comb_gain(pl: dict, ch: np.ndarray) -> float:
    """Level correction giving every draw the same summed power.

    Under equal-loudness weighting the channels differ by several dB, so a
    frozen figure has a *constant* element power while a redrawn one varies.
    That difference is a modulation cue at the element rate rather than a
    coherence cue, and it is the one thing the listener could use without
    hearing a figure at all.  Equalising the summed power removes it and
    leaves the position free.
    """
    a = pl["amp"]
    return float(np.sqrt(ch.size * np.mean(a ** 2) / np.sum(a[ch] ** 2)))


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
    chan, slot, gain, is_fig = [], [], [], []

    def put(c: int, s: int, g: float, f: bool) -> None:
        chan.append(int(c)), slot.append(int(s))
        gain.append(float(g)), is_fig.append(f)
        busy[c, max(0, s - d.guard):s + d.k + d.guard] = True

    if variant == "scatter":
        # The same channels at the same rate, never grouped into elements.
        # Long-term spectrum identical to the figure's; no temporal
        # coherence at all.
        fig_ch = comb(d, pl, rng)
        g = comb_gain(pl, fig_ch)
        for c, ts in zip(fig_ch, scatter_onsets(d, rng)):
            for s in ts:
                put(c, s, g, True)
    else:
        combs: list[np.ndarray] = []
        gains: list[float] = []
        for s0 in ons:
            if coherent and combs:
                ch, g = combs[0], gains[0]
            else:
                ch = comb(d, pl, rng, avoid=None if coherent else combs)
                g = comb_gain(pl, ch)
            combs.append(ch), gains.append(g)
            lg = redrawn(d, step_ms, rng) if variant == "redraw" else lag
            for c, l in zip(ch, lg):
                put(c, s0 + l, g, True)
        fig_ch = combs[0]

    # The background is blind to the figure: it keeps its own counts, so its
    # statistics are the same whether the figure recurs or not.  It only
    # avoids channels that are still sounding, which is symmetric between
    # the conditions and stops two tones landing on top of each other.
    seen = np.zeros(pl["n"])
    for s in range(d.n_slots - d.k):
        score = seen + rng.random(pl["n"]) * d.dealer_slack
        score[busy[:, s:s + d.k].any(axis=1)] = np.inf
        for c in np.argsort(score)[:d.density]:
            if not np.isfinite(score[c]):
                raise ValueError(f"slot {s}: no free channel, pool too small")
            put(c, s, 1.0, False)
            seen[c] += 1

    return dict(chan=np.array(chan), slot=np.array(slot),
                gain=np.array(gain), is_fig=np.array(is_fig),
                fig_ch=fig_ch, onsets=ons, lag=lag, step_ms=step_ms,
                coherent=coherent, variant=variant)


# ---------------------------------------------------------------- render
def render(d: Design, pl: dict, sch: dict, rng: np.random.Generator,
           rove_db: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A tone is one ramp longer than its k slots, so at every slot boundary
    the tones ramping out are matched by the ones ramping in; with
    power-complementary ramps the total power is constant across the join.
    """
    hop = int(round(d.hop_ms * d.fs / 1000.0))
    n = d.k * hop + hop
    x = np.arange(hop) / hop
    env = np.ones(n)
    env[:hop] = np.sin(np.pi / 2 * x)
    env[-hop:] = np.cos(np.pi / 2 * x)

    t = np.arange(n) / d.fs
    ph = rng.uniform(0, 2 * np.pi, pl["n"])
    pips = (pl["amp"][:, None]
            * np.sin(2 * np.pi * pl["f"][:, None] * t + ph[:, None]) * env)

    y = np.zeros(d.n_slots * hop + n)
    yf = np.zeros_like(y)
    for c, s, g, f in zip(sch["chan"], sch["slot"], sch["gain"],
                          sch["is_fig"]):
        y[s * hop:s * hop + n] += g * pips[c]
        if f:
            yf[s * hop:s * hop + n] += g * pips[c]
    y, yf = y[:d.n_slots * hop], yf[:d.n_slots * hop]

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
