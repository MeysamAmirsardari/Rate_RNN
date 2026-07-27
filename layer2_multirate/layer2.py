"""
layer2_multirate.layer2
=======================

Layer 2 with a **filterbank of rates** instead of a single slow trace.

What changed, and why
---------------------
With one trace, a unit can see one predecessor.  A three token word has two
transitions, so it fragments across units: the layer learns "B after A" and
"C after B" separately and nothing represents ABC.

A bank of low pass filters fixes that, because the **ratio of a channel's
trace across rates encodes how long ago it fired**.  A token that ended 30 ms
ago is strong at fast and slow rates alike; one that ended 160 ms ago survives
only at the slow rates.  So a single mask can hold

    C firing now   AND   B recent at a fast rate   AND   A recent at a slow rate

which is the whole word in one unit.  This is the standard multi timescale way
of representing elapsed time, and it needs no new learning rule: the four rules
are untouched and only the shape of the representation changes.

Shapes
------
    E      (N,)          layer 1 excitatory rate, unchanged
    s      (N, R)        one leaky integrator per channel per rate
    D      (N, N, R)     D[i,j,m] = E_i * s[j,m], diagonal in (i,j) zeroed
    M_k    (N, N, R)     one non negative mask per unit
    y      (K,)          y_k = relu(<M_k, D>)

Each rate is a single stage low pass.  The two stage cascade used in the single
rate version is dropped: it existed to suppress a channel coinciding with its
own trace, and zeroing the (i,j) diagonal already does that completely, so the
extra stage was measured to change nothing.
"""
from __future__ import annotations

import numpy as np

from .config import MRConfig


class Layer2MR:
    """Chunk selective units reading a multi rate coincidence tensor."""

    def __init__(self, n_channels: int, cfg: MRConfig | None = None):
        self.cfg = cfg or MRConfig()
        self.N = n_channels
        self.tau = np.asarray(self.cfg.rates, dtype=float)      # (R,)
        self.R = self.tau.size
        rng = np.random.default_rng(self.cfg.seed)
        self.M = rng.random((self.cfg.n_units, n_channels, n_channels,
                             self.R)) * self.cfg.w_init
        # a subunit pairs two DIFFERENT channels
        self.offdiag = ~np.eye(n_channels, dtype=bool)
        self.M *= self.offdiag[None, :, :, None]
        self.win_counts = np.zeros(self.cfg.n_units, dtype=int)
        self.reset_state()

    def reset_state(self):
        """Clear the traces; keep weights and learning statistics."""
        self.s = np.zeros((self.N, self.R))
        self.peak = 1e-9

    # ---- one integration step ---------------------------------------
    def step(self, E: np.ndarray, dt: float, learn: bool = True):
        c = self.cfg
        # filterbank: one leaky integrator per (channel, rate)
        self.s += dt * (-self.s + E[:, None]) / self.tau[None, :]

        D = E[:, None, None] * self.s[None, :, :]        # (N, N, R)
        D = D * self.offdiag[:, :, None]
        nD = float(np.sqrt((D * D).sum()))
        if nD > self.peak:
            self.peak = nD

        y = np.maximum(np.tensordot(self.M, D, axes=([1, 2, 3], [0, 1, 2])), 0.0)

        if learn:
            if nD > c.gate_frac * self.peak:
                Dh = D / nD
                flat = self.M.reshape(c.n_units, -1)
                norms = np.linalg.norm(flat, axis=1) + 1e-12
                match = np.tensordot(self.M, Dh,
                                     axes=([1, 2, 3], [0, 1, 2])) / norms
                k = int(np.argmax(match))
                self.M[k] += c.eta * (Dh - self.M[k])
                np.clip(self.M[k], 0.0, None, out=self.M[k])
                self.win_counts[k] += 1
            self.M *= (1.0 - c.lam)
        return y, D

    # ---- run over a layer 1 history ---------------------------------
    def run(self, E_hist: np.ndarray, dt: float, learn: bool = True,
            record_every: int = 0):
        N, T = E_hist.shape
        if N != self.N:
            raise ValueError(f"E has {N} channels, layer 2 expects {self.N}")
        y_hist = np.zeros((self.cfg.n_units, T))
        snaps, snap_t = [], []
        for t in range(T):
            y, _ = self.step(E_hist[:, t], dt, learn=learn)
            y_hist[:, t] = y
            if record_every and (t % record_every == 0):
                snaps.append(self.mask_norms.copy())
                snap_t.append(t * dt)
        return dict(y=y_hist,
                    norm_traj=np.array(snaps) if snaps
                    else np.empty((0, self.cfg.n_units)),
                    norm_t=np.array(snap_t))

    # ---- readout -----------------------------------------------------
    @property
    def mask_norms(self) -> np.ndarray:
        return np.linalg.norm(self.M.reshape(self.cfg.n_units, -1), axis=1)

    @property
    def committed(self) -> np.ndarray:
        n = self.mask_norms
        top = n.max()
        if top <= 0.0:
            return np.zeros(self.cfg.n_units, dtype=bool)
        return n >= self.cfg.commit_frac * top

    @property
    def n_components(self) -> int:
        return int(self.committed.sum())
