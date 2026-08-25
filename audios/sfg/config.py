"""Every parameter of the stimulus, in one place."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SFGConfig:
    # --- output ------------------------------------------------------
    fs: int = 48_000
    seed: int = 0

    # --- time base ---------------------------------------------------
    # Onsets fall on a grid of `hop_ms`; a tone lasts a whole number of
    # hops.  Because the same number of tones *starts* in every slot, the
    # number sounding is constant and the envelope is flat by construction.
    # `hop_ms` is the resolution of every timing control below, and it is
    # independent of the tone length -- that is what lets the figure be
    # sheared by an arbitrary amount without becoming the only thing in the
    # stimulus that starts off-grid.
    hop_ms: float = 0.0         # 0 solves it: the coarsest grid that both
                                # the tone and the shear land on exactly
    tone_ms: float = 30.0

    # --- figure ------------------------------------------------------
    coherence: int = 7          # channels in the figure; 0 = figure absent
    rate_hz: float = 5.0        # figure repetition rate
    step_ms: float = 5.0        # shear per tone (0 = a coherent chord)
    order: str = "rise"         # 'rise' | 'fall' | 'perm'
    wobble_ms: float = 0.0      # frozen irregularity of the staircase
    jitter_ms: float = 60.0     # displacement of each figure onset
    coherent: bool = True       # False: redraw the figure's channels every
                                # repetition -- the canonical figure-absent
                                # control, identical in every respect but
                                # coherence
    redraw_lags: bool = False   # redraw the lags too: removes the pattern
                                # while keeping the spread

    # --- background --------------------------------------------------
    # `n_sounding` is the published control -- O'Sullivan et al. hold it at
    # 15 "to keep the broadband power and all other low-level features
    # constant".  Hold it fixed across conditions: it is what makes the
    # figure a re-labelling of tones that would have sounded anyway rather
    # than an addition, so figure-present and figure-absent match in count,
    # energy and envelope without any patching.
    n_sounding: int = 10
    contrast: float = 4.0       # figure/background per-channel rate, which
                                # is what sets the pool size

    # --- pool --------------------------------------------------------
    f_lo: float = 250.0         # narrower than the published 179-7246 Hz:
    f_hi: float = 5000.0        # the equal-loudness weighting spans 11 dB
                                # over that range, and a fixed figure against
                                # a randomly drawn background then differs in
                                # level, which modulates the envelope at the
                                # figure's own rate
    grid_st: float = 0.0        # 0 solves for it from `contrast`
    min_grid_st: float = 0.5    # never finer than 1/24 octave: closer than
                                # that and simultaneous tones beat
    fig_lo_st: float = 8.0      # figure confined to the middle of the pool,
    fig_hi_st: float = 8.0      # this many semitones clear of each edge
    fig_span_st: float = 30.0   # frequency span of the figure

    # --- level -------------------------------------------------------
    equal_loudness: bool = True  # ISO 226 weighting, so no region dominates
    phon: float = 60.0
    peak_dbfs: float = -6.0

    # --- length ------------------------------------------------------
    duration_s: float = 12.0
    lead_ms: float = 500.0
    tail_ms: float = 500.0

    def replace(self, **kw) -> "SFGConfig":
        return replace(self, **kw)

    @property
    def hop(self) -> float:
        """The time grid.  Solved as the coarsest that both the tone length
        and the shear land on exactly, since a finer grid costs density for
        no gain: `n_sounding` is fixed, so a grid k times finer needs k times
        as many tone onsets to hold it."""
        if self.hop_ms > 0:
            return self.hop_ms
        a = round(self.tone_ms * 100)
        b = round(self.step_ms * 100)
        g = math.gcd(int(a), int(b)) if b else int(a)
        return g / 100.0

    @property
    def k(self) -> int:
        """Tone length in slots."""
        k = self.tone_ms / self.hop
        if abs(k - round(k)) > 1e-9:
            raise ValueError(
                f"tone_ms {self.tone_ms} must be a whole number of hops")
        return int(round(k))

    @property
    def density(self) -> int:
        """Tones starting per slot, so that `n_sounding` are always up."""
        return max(1, int(round(self.n_sounding / self.k)))
