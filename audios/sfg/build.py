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


def figure_channels(pool: dict, cfg: SFGConfig,
                    rng: np.random.Generator) -> np.ndarray:
    """`coherence` channels, evenly spread over the middle of the pool.

    Redrawn per stimulus so the figure is never in a learnable place, but
    always spanning the same range so its spectral extent is not a cue.
    """
    if cfg.coherence == 0:
        return np.zeros(0, int)
    lo = int(round(cfg.fig_lo_st / pool["grid_st"]))
    hi = pool["n"] - 1 - int(round(cfg.fig_hi_st / pool["grid_st"]))
    width = int(round(cfg.fig_span_st / pool["grid_st"]))
    if lo + width > hi:
        raise ValueError("figure span does not fit inside the pool")

    # Where the figure sits is drawn fresh per stimulus so it is never in a
    # learnable place -- but not uniformly.  Under equal-loudness weighting
    # the channels differ in amplitude by several dB, so a figure that
    # happens to sit on loud channels adds more energy than the background
    # tones it replaces, and that difference recurs at the figure's own rate.
    # Choosing among positions whose summed weight is closest to what the
    # background would have contributed removes the bias; measured, it takes
    # the figure/no-figure envelope difference from +0.14 dB to under 0.02.
    amp = pool["amp"]
    target = cfg.coherence * amp.mean()
    bases = np.arange(lo, hi - width + 1)
    sets = [np.unique(np.round(np.linspace(b, b + width, cfg.coherence))
                      .astype(int)) for b in bases]
    err = np.array([abs(amp[i].sum() - target) for i in sets])
    good = np.flatnonzero(err <= np.quantile(err, 0.15))
    return sets[int(rng.choice(good))]


# ------------------------------------------------------------- the schedule
def _lags(cfg: SFGConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Each figure tone's delay behind the first, in slots."""
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

    occ: list[list[int]] = [[] for _ in range(n_slots)]
    onsets = []
    for w in range((n_slots - lead) // period + 1):
        s0 = lead + w * period + (int(rng.integers(-jit, jit + 1)) if jit else 0)
        if s0 < 0 or s0 >= n_slots:
            continue
        onsets.append(s0)
        ch = fig_ch if cfg.coherent else figure_channels(pool, cfg, rng)
        lg = _lags(cfg, ch.size, rng) if cfg.redraw_lags else lag
        for c, l in zip(ch, lg):
            if 0 <= s0 + l < n_slots:
                occ[s0 + l].append(int(c))

    need = max((len(o) for o in occ), default=0)
    density = cfg.density
    if density < need:
        raise ValueError(
            f"the figure starts {need} tones in one slot but only {density} "
            f"sound per slot; raise n_sounding to at least "
            f"{need * k} (or reduce coherence / jitter_ms)")

    chan = np.zeros(n_slots * density, int)
    slot = np.zeros(n_slots * density, int)
    is_fig = np.zeros(n_slots * density, bool)

    # Least-used-first with a loose tie so the counts stay level while the
    # order stays unpredictable; strict ranking makes each channel return at
    # suspiciously even intervals, which is the one property the figure is
    # supposed to have alone.
    seen = np.zeros(pool["n"])
    live: list[int] = []
    m = 0
    for s in range(n_slots):
        here = list(occ[s])
        n_fig = len(here)
        for c in here:
            seen[c] += 1
        while len(here) < density:
            for c in np.argsort(seen // 4 + rng.random(pool["n"])):
                if c not in here and c not in live:
                    here.append(int(c))
                    seen[c] += 1
                    break
            else:
                raise ValueError(f"cannot fill slot {s}: pool too small")
        chan[m:m + density] = here
        slot[m:m + density] = s
        is_fig[m:m + density][:n_fig] = True
        live = (live + here)[-k * density:]
        m += density

    return dict(chan=chan, slot=slot, is_fig=is_fig, fig_ch=fig_ch,
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
    for c, s, f in zip(sch["chan"], sch["slot"], sch["is_fig"]):
        a = s * hop
        y[a:a + n] += pips[c]
        if f:
            yf[a:a + n] += pips[c]

    g = 10 ** (cfg.peak_dbfs / 20) / np.max(np.abs(y))
    return dict(mix=y * g, figure=yf * g, ground=(y - yf) * g, fs=cfg.fs)


def build(cfg: SFGConfig) -> dict:
    """Pool, schedule and sound for one stimulus."""
    rng = np.random.default_rng(cfg.seed)
    pool = make_pool(cfg)
    sch = schedule(cfg, pool, rng)
    out = render(cfg, pool, sch)
    return dict(cfg=cfg, pool=pool, **sch, **out)
