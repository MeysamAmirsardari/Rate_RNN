"""The adaptive rule, exactly as the paper states it.

  "an adaptive three-down one-up rule, which tracked the 79.4%-correct point
   on the psychometric function ... the tracking variable, dT, was set to
   20 ms.  It was divided by a factor c after [three] consecutive correct
   responses, and multiplied by that same factor after each incorrect
   response.  The value of c was set to 4 at the beginning of the adaptive
   run; it was reduced to 2 after the first reversal in the direction of
   tracking (from decreasing to increasing), and to sqrt(2) after a further
   two reversals.  The procedure stopped after the sixth reversal with the
   sqrt(2) step size.  Threshold was computed as the geometric mean of dT at
   the last six reversal points."

The manuscript says "two consecutive correct" in that sentence and
"three-down one-up ... 79.4%" in the one before it, which cannot both be
true: two-down one-up converges on 70.7 % and three-down on 0.5^(1/3) =
79.37 %.  The stated convergence point is the unambiguous half, so the rule
here is three-down, and `n_down` is in the config for anyone who reads it
the other way.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Design


class Track:
    """One adaptive run.  Ask it for a level, tell it what happened."""

    def __init__(self, d: Design, n_down: int = 3):
        self.d = d
        self.n_down = n_down
        self.dt = float(d.dt_start_ms)
        self.fi = 0                      # index into d.factors
        self.direction = 0               # -1 going down, +1 going up
        self.n_right = 0
        self.reversals: list[float] = []
        self.final: list[float] = []     # reversals taken at the last factor
        self.history: list[tuple[float, bool]] = []
        self.clamped = 0
        self.done = False
        self.why = ""

    # ------------------------------------------------------------------
    @property
    def factor(self) -> float:
        return self.d.factors[self.fi]

    @property
    def at_final_factor(self) -> bool:
        return self.fi == len(self.d.factors) - 1

    def level(self) -> float:
        return self.dt

    # ------------------------------------------------------------------
    def update(self, correct: bool) -> None:
        """Score one trial and move the track."""
        if self.done:
            raise RuntimeError("this run has already finished")
        here = self.dt
        self.history.append((here, bool(correct)))

        if correct:
            self.n_right += 1
            if self.n_right < self.n_down:
                return                    # hold the level, no reversal
            self.n_right = 0
            new_dir, nxt = -1, here / self.factor
        else:
            self.n_right = 0
            new_dir, nxt = +1, here * self.factor

        if self.direction and new_dir != self.direction:
            self.reversals.append(here)
            if self.at_final_factor:
                self.final.append(here)
            elif len(self.reversals) >= self.d.factor_after[
                    min(self.fi, len(self.d.factor_after) - 1)]:
                self.fi = min(self.fi + 1, len(self.d.factors) - 1)
        self.direction = new_dir

        lo, hi = self.d.dt_min_ms, self.d.dt_max_ms
        if nxt < lo or nxt > hi:
            self.clamped += 1
        self.dt = float(min(max(nxt, lo), hi))

        if len(self.final) >= self.d.n_final_reversals:
            self.done, self.why = True, "ok"
        elif len(self.history) >= self.d.max_trials:
            self.done, self.why = True, "ran out of trials"

    # ------------------------------------------------------------------
    def threshold(self) -> float | None:
        """Geometric mean of dT at the last six reversals."""
        if len(self.final) < self.d.n_final_reversals:
            return None
        r = np.array(self.final[-self.d.n_final_reversals:])
        return float(np.exp(np.mean(np.log(r))))

    def report(self) -> dict:
        th = self.threshold()
        n = len(self.history)
        pc = (sum(c for _, c in self.history) / n) if n else float("nan")
        return dict(threshold_ms=th, n_trials=n, pc=pc,
                    n_reversals=len(self.reversals),
                    n_final=len(self.final), clamped=self.clamped,
                    at_floor=bool(th is not None
                                  and th <= self.d.dt_min_ms * 1.01),
                    at_ceiling=bool(th is not None
                                    and th >= self.d.dt_max_ms * 0.99),
                    why=self.why or "unfinished",
                    reversals=list(self.final))


# ---------------------------------------------------------------- pooling
def geomean(x) -> float:
    x = np.asarray([v for v in x if v is not None and v > 0], float)
    return float(np.exp(np.mean(np.log(x)))) if x.size else float("nan")


def geo_ci(x, conf: float = 0.95) -> tuple[float, float, float]:
    """Geometric mean and a t interval on it, computed in log space.

    Thresholds of this kind are log-normal, which is why the paper averages
    them geometrically; an arithmetic interval on a geometric mean would sit
    off-centre and can reach below zero.
    """
    from scipy import stats
    x = np.asarray([v for v in x if v is not None and v > 0], float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    lg = np.log(x)
    m = lg.mean()
    if x.size == 1:
        return float(np.exp(m)), float("nan"), float("nan")
    se = lg.std(ddof=1) / math.sqrt(x.size)
    t = stats.t.ppf(0.5 + conf / 2, x.size - 1)
    return float(np.exp(m)), float(np.exp(m - t * se)), float(np.exp(m + t * se))


# ------------------------------------------------------------- simulation
def simulate(d: Design, true_ms: float, slope: float = 1.4,
             lapse: float = 0.02, seed: int = 0,
             n_down: int = 3) -> Track:
    """Run the track against a listener whose threshold is known.

    A log-Weibull psychometric function on dT, pinned so that the 79.4 %
    point sits at `true_ms`.  This is how the rule is checked: a tracking
    procedure that does not recover a threshold it was given is not one to
    put a listener in front of.
    """
    rng = np.random.default_rng(seed)
    t = Track(d, n_down=n_down)
    # pin the function so that p(true_ms) is exactly the 79.4 % the rule
    # converges on, whatever the slope and the lapse rate are
    u = -math.log(1.0 - (0.794 - 0.5) / (0.5 - lapse))
    k = true_ms / u ** (1.0 / slope)
    while not t.done:
        x = t.level()
        p = 0.5 + (0.5 - lapse) * (1 - math.exp(-(x / k) ** slope))
        t.update(bool(rng.random() < p))
    return t
