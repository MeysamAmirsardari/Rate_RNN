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

Representation
--------------
The filterbank is flattened in **rate-major order**:

    s_flat = [s[:, 0], s[:, 1], ..., s[:, R-1]]       (N R,)

so frequency runs from 0 to N-1 inside each contiguous rate block.  The
multi-rate coincidence map is then the same outer product as the single-rate
model:

    D = E[:, None] * s_flat[None, :]                    (N, N R)

Same-channel connections are zero in every rate block.  A unit has one fixed
mask with exactly D's shape.  It is a blind linear filter: no mask slice is
selected and the mask is not told whether E is one-hot.

Shapes
------
    E      (N,)          layer 1 excitatory rate, unchanged
    s      (N, R)        one leaky integrator per channel per rate
    s_flat (N R,)        rate blocks, frequency ordered within each block
    D      (N, N R)      outer(E, s_flat), repeated diagonals zeroed
    M_k    (N, N R)      one non-negative mask per unit
    y      (K,)          y = relu(M_flat @ D_flat)

Each rate is a single stage low pass.  The two stage cascade used in the single
rate version is dropped: it existed to suppress a channel coinciding with its
own trace, and zeroing the (i,j) diagonal already does that completely, so the
extra stage was measured to change nothing.
"""
from __future__ import annotations

import numpy as np

from .config import MRConfig


class Layer2MR:
    """Chunk-selective units applying fixed masks to a 2-D coincidence map."""

    def __init__(self, n_channels: int, cfg: MRConfig | None = None):
        self.cfg = cfg or MRConfig()
        self.N = n_channels
        self.tau = np.asarray(self.cfg.rates, dtype=float)      # (R,)
        self.R = self.tau.size
        rng = np.random.default_rng(self.cfg.seed)
        # Preserve the seeded weight mapping while storing every mask in the
        # same rate-major 2-D representation as D.
        seeded_weights = rng.random(
            (self.cfg.n_units, n_channels, n_channels, self.R)
        )
        self.M = (
            seeded_weights.transpose(0, 1, 3, 2)
            .reshape(self.cfg.n_units, n_channels, self.R * n_channels)
            * self.cfg.w_init
        )

        # Columns are [all frequencies at rate 0, all frequencies at rate 1,
        # ...].  Repeating the context-channel index constructs all diagonal
        # exclusions at once, without a loop.
        self.context_channels = np.tile(np.arange(self.N), self.R)
        self.context_rates = np.repeat(np.arange(self.R), self.N)
        self.valid_connections = (
            np.arange(self.N)[:, None] != self.context_channels[None, :]
        )
        self.M *= self.valid_connections[None, :, :]
        self.win_counts = np.zeros(self.cfg.n_units, dtype=int)
        self.reset_state()

    def reset_state(self):
        """Clear the traces; keep weights and learning statistics."""
        self.s = np.zeros((self.N, self.R))
        self.peak = 1e-9

    # ---- representation ---------------------------------------------
    def flatten_filterbank(self, s: np.ndarray | None = None) -> np.ndarray:
        """Flatten (frequency, rate) into rate-major feature order."""
        source = self.s if s is None else np.asarray(s)
        if source.shape != (self.N, self.R):
            raise ValueError(
                f"filterbank has shape {source.shape}, expected {(self.N, self.R)}"
            )
        return source.T.reshape(-1)

    def unflatten_filterbank(self, values: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`flatten_filterbank`, returning (frequency, rate)."""
        values = np.asarray(values)
        if values.size != self.N * self.R:
            raise ValueError(
                f"flat filterbank has {values.size} values, expected {self.N * self.R}"
            )
        return values.reshape(self.R, self.N).T

    def mask_context_rate(self, unit: int, input_channel: int) -> np.ndarray:
        """One analysis view of a mask row as (context frequency, rate)."""
        return self.unflatten_filterbank(self.M[unit, input_channel])

    def coincidence(self, E: np.ndarray) -> np.ndarray:
        """Form the 2-D multi-rate coincidence map in one outer product."""
        E = np.asarray(E)
        if E.shape != (self.N,):
            raise ValueError(f"E has shape {E.shape}, expected {(self.N,)}")
        return np.multiply.outer(E, self.flatten_filterbank()) * self.valid_connections

    # ---- one integration step ---------------------------------------
    def step(self, E: np.ndarray, dt: float, learn: bool = True):
        c = self.cfg
        # filterbank: one leaky integrator per (channel, rate)
        self.s += dt * (-self.s + E[:, None]) / self.tau[None, :]

        D = self.coincidence(E)                          # (N, N * R)
        D_flat = D.reshape(-1)
        M_flat = self.M.reshape(c.n_units, -1)
        nD = float(np.linalg.norm(D_flat))
        if nD > self.peak:
            self.peak = nD

        y = np.maximum(M_flat @ D_flat, 0.0)

        if learn:
            if nD > c.gate_frac * self.peak:
                Dh = D / nD
                Dh_flat = Dh.reshape(-1)
                norms = np.linalg.norm(M_flat, axis=1) + 1e-12
                match = (M_flat @ Dh_flat) / norms
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
