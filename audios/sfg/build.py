"""Building the stimulus: the pool, the figure, the background, the sound."""

from __future__ import annotations

import numpy as np

from .config import SFGConfig
from .loudness import weights


# ---------------------------------------------------------------- the pool
def make_pool(cfg: SFGConfig) -> dict:
    """Channel grid, and which channels the figure uses.

    The grid follows from `contrast`: a figure channel sounds `rate_hz` times
    a second, the background shares what is left between the rest, so asking
    for a ratio fixes how many channels the pool needs.  Coarser than the
    published 1/24 octave, because at these densities a 1/24 octave pool
    would put several simultaneous tones inside one critical band and the
    result beats audibly.
    """
    span_st = 12 * np.log2(cfg.f_hi / cfg.f_lo)
    if cfg.grid_st > 0:
        grid = cfg.grid_st
    else:
        bg_per_s = cfg.density * 1000.0 / cfg.hop - cfg.coherence * cfg.rate_hz
        want = max(bg_per_s, 1.0) * cfg.contrast / cfg.rate_hz + cfg.coherence
        grids = np.arange(cfg.min_grid_st, 6.01, 0.25)
        n = np.floor(span_st / grids) + 1
        grid = float(grids[int(np.argmin(np.abs(n - want)))])

    st = np.arange(0.0, span_st + 1e-9, grid)
    f = cfg.f_lo * 2.0 ** (st / 12.0)
    amp = weights(f, cfg.phon) if cfg.equal_loudness else np.ones_like(f)
    return dict(st=st, f=f, n=st.size, grid_st=grid, amp=amp / amp.max())


def figure_channels(pool: dict, cfg: SFGConfig, rng: np.random.Generator,
                    avoid: list[np.ndarray] | None = None) -> np.ndarray:
    """`coherence` channels, evenly spread, at a position drawn uniformly.

    Redrawn per stimulus so the figure is never in a learnable place, but
    always spanning the same range so its spectral extent is not a cue.
    `avoid` rejects positions sharing more than one channel with any
    earlier draw, which is what stops the figure-absent control from
    stumbling into a figure of its own.
    """
    if cfg.coherence == 0:
        return np.zeros(0, int)
    lo = int(round(cfg.fig_lo_st / pool["grid_st"]))
    hi = pool["n"] - 1 - int(round(cfg.fig_hi_st / pool["grid_st"]))
    width = int(round(cfg.fig_span_st / pool["grid_st"]))
    if lo + width > hi:
        raise ValueError("figure span does not fit inside the pool")

    for b in rng.permutation(np.arange(lo, hi - width + 1)):
        ch = np.round(np.linspace(b, b + width, cfg.coherence)).astype(int)
        if all(np.intersect1d(ch, a).size <= 1 for a in avoid or ()):
            return ch
    raise ValueError("no figure position left that avoids the earlier ones")


def comb_gain(pool: dict, ch: np.ndarray) -> float:
    """Level correction that gives every draw the same total power.

    Under equal-loudness weighting the channels differ by several dB, so a
    figure that happens to sit on loud channels adds more energy than the
    background tones it replaces -- and that difference recurs at the
    figure's own rate, which is a modulation cue rather than a coherence
    cue.  Equalising the summed power of each draw removes it while leaving
    the position free; measured, it takes the figure/no-figure envelope
    difference at the figure rate from +0.14 dB to under 0.02.
    """
    if ch.size == 0:
        return 1.0
    a = pool["amp"]
    return float(np.sqrt(ch.size * np.mean(a ** 2) / np.sum(a[ch] ** 2)))


# ------------------------------------------------------------- the schedule
def _lags(cfg: SFGConfig, n: int, rng: np.random.Generator,
          redraw: bool = False) -> np.ndarray:
    """Each figure tone's delay behind the first, in slots.

    `redraw` keeps the extent and the number of tones but throws away the
    pattern: the delays are drawn afresh, so the figure is still the same
    seven channels arriving inside the same window and is still the same
    thing spectrally, but there is nothing frozen to learn.
    """
    span = int(round((n - 1) * cfg.step_ms / cfg.hop))
    if redraw:
        if span == 0 or n < 3:
            return np.zeros(n, int)
        mid = rng.choice(np.arange(1, span), size=n - 2, replace=False)
        return np.concatenate(([0], mid, [span]))[rng.permutation(n)]

    lag = np.arange(n) * cfg.step_ms
    if cfg.order == "fall":
        lag = lag[::-1]
    elif cfg.order == "perm":
        lag = lag[rng.permutation(n)]
    elif cfg.order != "rise":
        raise ValueError(f"unknown order {cfg.order!r}")
    if cfg.wobble_ms > 0:
        lag = lag + rng.integers(0, int(cfg.wobble_ms / cfg.hop) + 1, n) \
            * cfg.hop
    return np.round(lag / cfg.hop).astype(int)


def schedule(cfg: SFGConfig, pool: dict, rng: np.random.Generator) -> dict:
    """Which channel starts in which slot, for the figure and the background.

    The figure is a *re-labelling* of tones that were going to sound anyway,
    never an addition: the background fills each slot up to the same total
    whatever the figure does, so figure-present and figure-absent stimuli are
    identical in tone count, in energy and in envelope.
    """
    k = cfg.k
    n_slots = int(round(cfg.duration_s * 1000.0 / cfg.hop))
    lead = int(round(cfg.lead_ms / cfg.hop))
    period = max(1, int(round(1000.0 / (cfg.rate_hz * cfg.hop))))
    jit = int(round(cfg.jitter_ms / cfg.hop))

    fig_ch = figure_channels(pool, cfg, rng)
    lag = _lags(cfg, fig_ch.size, rng) if fig_ch.size else np.zeros(0, int)

    occ: list[list[tuple[int, float]]] = [[] for _ in range(n_slots)]
    onsets = []
    seen = [fig_ch]
    for w in range((n_slots - lead) // period + 1):
        s0 = lead + w * period + (int(rng.integers(-jit, jit + 1)) if jit else 0)
        if s0 < 0 or s0 >= n_slots:
            continue
        onsets.append(s0)
        if cfg.coherent:
            ch = fig_ch
        else:
            ch = figure_channels(pool, cfg, rng, avoid=seen[-6:])
            seen.append(ch)
        lg = _lags(cfg, ch.size, rng, redraw=True) if cfg.redraw_lags else lag
        g = comb_gain(pool, ch)
        for c, l in zip(ch, lg):
            if 0 <= s0 + l < n_slots:
                occ[s0 + l].append((int(c), g))

    need = max((len(o) for o in occ), default=0)
    density = cfg.density
    if density < need:
        raise ValueError(
            f"the figure starts {need} tones in one slot but only {density} "
            f"sound per slot; raise n_sounding to at least "
            f"{need * k} (or reduce coherence / jitter_ms)")

    chan = np.zeros(n_slots * density, int)
    slot = np.zeros(n_slots * density, int)
    gain = np.ones(n_slots * density)
    is_fig = np.zeros(n_slots * density, bool)

    # Least-used-first with a loose tie so the counts stay level while the
    # order stays unpredictable; strict ranking makes each channel return at
    # suspiciously even intervals, which is the one property the figure is
    # supposed to have alone.
    used = np.zeros(pool["n"])
    live: list[int] = []
    m = 0
    for s in range(n_slots):
        here = [c for c, _ in occ[s]]
        gs = [g for _, g in occ[s]]
        n_fig = len(here)
        for c in here:
            used[c] += 1
        while len(here) < density:
            for c in np.argsort(used // 4 + rng.random(pool["n"])):
                if c not in here and c not in live:
                    here.append(int(c))
                    gs.append(1.0)
                    used[c] += 1
                    break
            else:
                raise ValueError(f"cannot fill slot {s}: pool too small")
        chan[m:m + density] = here
        slot[m:m + density] = s
        gain[m:m + density] = gs
        is_fig[m:m + density][:n_fig] = True
        live = (live + here)[-k * density:]
        m += density

    return dict(chan=chan, slot=slot, gain=gain, is_fig=is_fig, fig_ch=fig_ch,
                onsets=np.array(onsets), density=density, n_slots=n_slots,
                lag=lag)


# --------------------------------------------------------------- the sound
def render(cfg: SFGConfig, pool: dict, sch: dict) -> dict:
    """Synthesise, with one gain for the mix and the figure alike.

    A tone is one ramp longer than its `k` slots so that at every slot
    boundary the tones ramping out are matched by the ones ramping in; with
    power-complementary ramps the total power is then constant across the
    join.  Ramping each tone independently over abutting slots would leave a
    dip at every boundary, heard as a flutter at the slot rate.
    """
    hop = int(round(cfg.hop * cfg.fs / 1000.0))
    n = cfg.k * hop + hop
    t = np.arange(n) / cfg.fs
    x = np.arange(hop) / hop
    env = np.ones(n)
    env[:hop] = np.sin(np.pi / 2 * x)
    env[-hop:] = np.cos(np.pi / 2 * x)

    phase = np.random.default_rng(cfg.seed + 1).uniform(
        0, 2 * np.pi, pool["n"])
    pips = (pool["amp"][:, None]
            * np.sin(2 * np.pi * pool["f"][:, None] * t + phase[:, None])
            * env)

    N = sch["n_slots"] * hop + n
    y = np.zeros(N)
    yf = np.zeros(N)
    for c, s, g, f in zip(sch["chan"], sch["slot"], sch["gain"],
                          sch["is_fig"]):
        a = s * hop
        y[a:a + n] += g * pips[c]
        if f:
            yf[a:a + n] += g * pips[c]

    g = 10 ** (cfg.peak_dbfs / 20) / np.max(np.abs(y))
    return dict(mix=y * g, figure=yf * g, ground=(y - yf) * g, fs=cfg.fs)


def build(cfg: SFGConfig) -> dict:
    """Pool, schedule and sound for one stimulus."""
    rng = np.random.default_rng(cfg.seed)
    pool = make_pool(cfg)
    sch = schedule(cfg, pool, rng)
    out = render(cfg, pool, sch)
    return dict(cfg=cfg, pool=pool, **sch, **out)
