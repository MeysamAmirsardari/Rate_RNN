"""The two tone streams, their onsets, and the sound.

One sequence is two parallel trains of pure tones, A and B.  Everything the
listener is asked to do turns on the onset of the very last B tone, so the
onsets are computed exactly, in samples, and never by accumulating floats.
"""

from __future__ import annotations

import numpy as np

from .config import Design


# ------------------------------------------------------------------ onsets
def onsets_ms(d: Design, *, gap_a_ms: float, lag_ms: float, dt_ms: float,
              sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Onset times of the A and B tones, in ms, for one interval.

    The two trains are pinned to each other at their *target* tones, which is
    the pair the listener judges: in the standard interval the target B sits
    `lag_ms` after the target A, and in the signal interval it sits
    `lag_ms + sign*dt_ms` after it.  `lag_ms` is 0 in the replication, where
    the published design makes the target pair synchronous and lets the
    precursors drift apart on their own because the two trains run at
    different tempi; it is the swept variable in `sweep`, where the tempi are
    identical and the whole B train is displaced.

    Pinning at the target rather than at the start is what the paper does,
    and it matters: it keeps the judged pair in the same relationship to the
    end of the sequence in every condition, so the listener is never
    comparing an event 850 ms in with one 750 ms in.
    """
    n = d.n_tones
    soa_a = d.soa_a(gap_a_ms)
    a = np.arange(n) * soa_a
    b_last = a[-1] + lag_ms
    b = b_last - (n - 1 - np.arange(n)) * d.soa_b
    b = b.astype(float)
    b[-1] = b_last + sign * dt_ms
    shift = d.lead_ms - min(a.min(), b.min())
    return a + shift, b + shift


def overlap_ms(d: Design, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """How long each A tone and its nearest B tone sound together."""
    out = []
    for t in a:
        j = int(np.argmin(np.abs(b - t)))
        out.append(max(0.0, min(t, b[j]) + d.tone - max(t, b[j])))
    return np.array(out)


# ------------------------------------------------------------------- sound
def _tone(d: Design, f: float, n: int, phase: float = 0.0) -> np.ndarray:
    """One gated tone, `d.tone` ms long *including* both ramps.

    `phase` is drawn fresh for each trial and shared by the two intervals of
    that trial.  Without it the fine structure of the sequence is frozen for
    the whole experiment: the A and B tones overlap at one fixed phase in
    every trial of a condition, and whatever that particular alignment sounds
    like becomes learnable over a two-hour session.  Sharing it within a
    trial keeps the two intervals identical apart from the shift.
    """
    t = np.arange(n) / d.fs
    y = np.sin(2 * np.pi * f * t + phase)
    r = int(round(d.ramp_ms * d.fs / 1000.0))
    if 2 * r > n:
        raise ValueError("the ramps are longer than the tone")
    # raised cosine, 0.5*(1 - cos(pi x)), which is sin^2 and not sin: the
    # power-complementary sine ramp is a different window and would leave the
    # gated tone 3 dB down at its own half-way point
    w = np.sin(np.pi / 2 * (np.arange(r) + 0.5) / r) ** 2
    y[:r] *= w
    y[-r:] *= w[::-1]
    return y


def render(d: Design, *, f_b: float, a: np.ndarray, b: np.ndarray,
           b_only: bool, n_samples: int, rove_db: float = 0.0,
           phase: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Sum the two trains into one monaural interval of a fixed length.

    Every interval of the experiment is rendered to the same number of
    samples whatever the shift does, so nothing can be read off the file.
    """
    amp = np.sqrt(2.0) * 10 ** (d.rms_dbfs / 20.0) * 10 ** (rove_db / 20.0)
    n = int(round(d.tone * d.fs / 1000.0))
    pip_a = amp * _tone(d, d.f_a, n, phase[0])
    pip_b = amp * _tone(d, f_b, n, phase[1])

    y = np.zeros(n_samples)
    trains = [(b, pip_b)] if b_only else [(a, pip_a), (b, pip_b)]
    for onsets, pip in trains:
        for t in onsets:
            i = int(round(t * d.fs / 1000.0))
            if i < 0 or i + n > n_samples:
                raise ValueError("a tone falls outside the interval")
            y[i:i + n] += pip
    return y


def to_ear(d: Design, y: np.ndarray) -> np.ndarray:
    """Monaural, as in the paper: the left earpiece only."""
    if d.ear == "both":
        return np.column_stack([y, y])
    z = np.zeros_like(y)
    return np.column_stack([y, z] if d.ear == "left" else [z, y])


# ------------------------------------------------------------------- trial
def interval(d: Design, *, df_st: float, gap_a_ms: float, lag_ms: float,
             dt_ms: float, sign: int, b_only: bool,
             n_samples: int | None = None, rove_db: float = 0.0,
             phase: tuple[float, float] = (0.0, 0.0)) -> dict:
    a, b = onsets_ms(d, gap_a_ms=gap_a_ms, lag_ms=lag_ms, dt_ms=dt_ms,
                     sign=sign)
    if n_samples is None:
        n_samples = interval_samples(d, gap_a_ms=gap_a_ms, lag_ms=lag_ms)
    y = render(d, f_b=d.f_b(df_st), a=a, b=b, b_only=b_only,
               n_samples=n_samples, rove_db=rove_db, phase=phase)
    return dict(y=y, a=a, b=b, dt_ms=dt_ms, sign=sign, df_st=df_st,
                gap_a_ms=gap_a_ms, lag_ms=lag_ms, b_only=b_only,
                f_b=d.f_b(df_st))


def interval_samples(d: Design, *, gap_a_ms: float, lag_ms: float) -> int:
    """One length for every interval of a condition, worst case included."""
    return int(round(d.interval_ms(gap_a_ms, lag_ms) * d.fs / 1000.0))


class Balance:
    """Which interval holds the signal, and which way the tone moves.

    Both are drawn in shuffled blocks of four rather than independently per
    trial.  A track runs about 45 trials, and an independent coin lands 60/40
    or worse on one of the two roughly a third of the time; a listener with
    an interval preference then scores above chance on the imbalance alone,
    and the shift direction leaking would let one be learned outright.  In
    blocks of four both are exactly balanced every four trials and the order
    inside a block is still unpredictable.
    """

    CELLS = ((1, -1), (1, 1), (2, -1), (2, 1))

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.queue: list[tuple[int, int]] = []

    def next(self) -> tuple[int, int]:
        if not self.queue:
            self.queue = [self.CELLS[i]
                          for i in self.rng.permutation(len(self.CELLS))]
        return self.queue.pop()


def trial(d: Design, *, df_st: float, gap_a_ms: float, lag_ms: float,
          dt_ms: float, b_only: bool, seed: int, rove: bool = False,
          target: int | None = None,
          sign: int | None = None) -> tuple[list[dict], int]:
    """The two intervals of one trial, and which of them holds the signal.

    Both are built to the same length and from the same tones; they differ
    only in whether the last B tone sits where the train says it should.
    The shift goes forward or backward with equal probability, as in the
    paper, so a listener who has learned that late means signal is at chance.
    Pass `target` and `sign` from a `Balance` to have that hold exactly
    rather than on average.
    """
    rng = np.random.default_rng(seed)
    if sign is None:
        sign = int(rng.choice([-1, 1]))
    if target is None:
        target = int(rng.integers(1, 3))      # 1 or 2
    n = interval_samples(d, gap_a_ms=gap_a_ms, lag_ms=lag_ms)
    kw = dict(df_st=df_st, gap_a_ms=gap_a_ms, lag_ms=lag_ms, b_only=b_only,
              n_samples=n)
    rv = (float(rng.uniform(-d.rove_db, d.rove_db)) if rove and d.rove_db
          else 0.0)
    ph = (tuple(rng.uniform(0, 2 * np.pi, 2)) if d.random_phase
          else (0.0, 0.0))
    std = interval(d, dt_ms=0.0, sign=1, rove_db=rv, phase=ph, **kw)
    sig = interval(d, dt_ms=dt_ms, sign=sign, rove_db=rv, phase=ph, **kw)
    return ([std, sig] if target == 2 else [sig, std]), target
