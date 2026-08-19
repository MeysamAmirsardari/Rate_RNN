"""
tasks.interplay3.fast_layer2
============================

``Layer2MR`` with the same arithmetic and three fewer passes over the masks.

Why this exists
---------------
The multi-rate mask set is ``(n_units, N, N * R)``.  At fifty channels, six
rates and twenty-four units that is 1.8 million floats, and the published
``step`` touches all of them **three times** per time step:

    self.M *= (1 - lam)                     the global decay
    norms = norm(M_flat, axis=1)            for the cosine match
    M_flat @ D_flat                         for the output

At 1 ms resolution and a ten-minute stream that is memory bandwidth, not
arithmetic, and it dominates the run.  Each of the three can be removed
exactly -- not approximated:

* **Decay.**  Keep a scalar ``g`` with the true mask equal to ``g * M``.  The
  global decay is then ``g *= (1 - lam)``, which is O(1).  The instar update
  rearranges to ``M[k] = M[k] * (1 - eta) + (eta / g) * Dhat``, and clipping
  the stored mask at zero is equivalent to clipping the true one because
  ``g > 0``.
* **Norms.**  ``g`` cancels out of the cosine match, and only the winner's
  mask changes, so the norms can be cached and one recomputed per step.
* **The second matmul.**  ``Dhat = D / |D|``, so the match numerator is the
  output numerator divided by ``|D|``.  One matmul serves both.

The result is one pass over the masks per step instead of three, and the
outputs agree with the published class to floating-point noise -- which
``verify()`` checks rather than asserts.

``g`` shrinks as ``(1 - lam)**t`` and the stored masks grow to compensate, so
the pair is renormalised whenever ``g`` gets small.  Without that, a long
enough run would underflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from layer2_multirate.layer2 import Layer2MR

#: Renormalise once the global scale drops below this.
_G_FLOOR = 1e-60


class FastLayer2MR(Layer2MR):
    """Same model, same numbers, one pass over the masks per step."""

    def __init__(self, n_channels: int, cfg=None):
        super().__init__(n_channels, cfg)
        self.g = 1.0
        self._norms = np.linalg.norm(
            self.M.reshape(self.cfg.n_units, -1), axis=1)

    # ---- keep the inherited read-outs honest ------------------------
    @property
    def mask_norms(self) -> np.ndarray:
        return self.g * self._norms

    @property
    def M_true(self) -> np.ndarray:
        """The masks as the published class would hold them."""
        return self.g * self.M

    def _renormalise(self) -> None:
        self.M *= self.g
        self._norms *= self.g
        self.g = 1.0

    # ---- one integration step ---------------------------------------
    def step(self, E: np.ndarray, dt: float, learn: bool = True):
        c = self.cfg
        self.s += dt * (-self.s + E[:, None]) / self.tau[None, :]

        D = self.coincidence(E)
        D_flat = D.reshape(-1)
        M_flat = self.M.reshape(c.n_units, -1)

        nD = float(np.linalg.norm(D_flat))
        if nD > self.peak:
            self.peak = nD

        raw = M_flat @ D_flat                     # the one full pass
        y = np.maximum(self.g * raw, 0.0)

        if learn:
            if nD > c.gate_frac * self.peak:
                # cosine match: (M @ D / |D|) / |M|, with g cancelling
                k = int(np.argmax(raw / (nD * (self._norms + 1e-12))))
                Dh = D / nD
                Mk = self.M[k]
                Mk *= (1.0 - c.eta)
                Mk += (c.eta / self.g) * Dh
                np.clip(Mk, 0.0, None, out=Mk)
                self._norms[k] = np.linalg.norm(Mk)
                self.win_counts[k] += 1
            self.g *= (1.0 - c.lam)
            if self.g < _G_FLOOR:
                self._renormalise()
        return y, D


def verify(n_channels: int = 12, n_steps: int = 400, seed: int = 0,
           **cfg_kw) -> dict:
    """Run both classes on the same input and report the largest disagreement.

    Returns the numbers rather than asserting on them, so the caller decides
    what counts as agreement.
    """
    from layer2_multirate.config import MRConfig

    cfg = MRConfig(n_units=6, **cfg_kw)
    rng = np.random.default_rng(seed)
    E = np.abs(rng.standard_normal((n_channels, n_steps)))

    slow = Layer2MR(n_channels, cfg)
    fast = FastLayer2MR(n_channels, cfg)

    ys, yf = [], []
    for t in range(n_steps):
        ys.append(slow.step(E[:, t], 1e-3, learn=True)[0])
        yf.append(fast.step(E[:, t], 1e-3, learn=True)[0])
    ys, yf = np.array(ys), np.array(yf)

    scale = max(float(np.abs(ys).max()), 1e-30)
    m_slow, m_fast = slow.M, fast.M_true
    m_scale = max(float(np.abs(m_slow).max()), 1e-30)
    return dict(
        y_rel=float(np.abs(ys - yf).max() / scale),
        mask_rel=float(np.abs(m_slow - m_fast).max() / m_scale),
        wins_equal=bool(np.array_equal(slow.win_counts, fast.win_counts)),
        committed_equal=bool(np.array_equal(slow.committed, fast.committed)),
    )


if __name__ == "__main__":
    for n in (12, 50):
        r = verify(n_channels=n)
        print(f"N = {n:3d}  max relative y error {r['y_rel']:.2e}   "
              f"mask {r['mask_rel']:.2e}   "
              f"same winners {r['wins_equal']}   "
              f"same committed {r['committed_equal']}")
