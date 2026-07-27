"""
layer2_syllable.layer2
======================

A second, purely feedforward layer that learns chunk selective units from the
activity of layer 1 (``model0``).  Layer 1 is not modified in any way; this
layer only reads its excitatory rates.

Representation
--------------
Each unit sees the instantaneous **directional coincidence map**

    D(t) = E(t) (x) s(t),        D[i, j] = E_i(t) * s_j(t)

where ``E`` is the layer 1 excitatory rate ("which channel is firing now") and
``s`` is a slow conductance driven by ``E`` ("which channel fired recently").
The two factors must have different kinetics.  If they do not, the time
integral of D is symmetric,

    integral E_i(t) E_j(t) dt  =  integral E_j(t) E_i(t) dt,

so the order information vanishes identically.  Direction selectivity here
comes from the same source as in a Reichardt correlator: an asymmetry of
delays between the two inputs being multiplied.

Unit k holds one non negative mask ``M_k`` of the same shape as D and responds
with the overlap

    y_k = relu(<M_k, D>)

which is a dendritic subunit reading a fast afferent from channel i together
with a slow afferent from channel j, and integrating them supralinearly
(Poirazi and Mel 2001; Branco, Clark and Hausser 2010 showed single pyramidal
dendrites discriminate input sequence order by exactly this mechanism).

Learning: four local rules
--------------------------
1. match     c_k = <M_k, D> / (|M_k| |D|)          direction only
2. compete   winner = argmax_k c_k                 strong lateral inhibition
3. learn     dM_win = eta * (D_hat - M_win)        instar Hebbian, clipped >= 0
4. forget    dM_k  -= lam * M_k   for every k      synaptic pruning

Rules 3 and 4 are a survival competition.  A unit that wins often has Hebbian
gain above its decay and sharpens onto one chunk type; a unit that never wins
receives only decay, its mask goes to zero and it falls silent.  The number of
committed units is therefore an outcome of the stream, not a setting.

Known limits (stated so they are not mistaken for results)
----------------------------------------------------------
* D is instantaneous, so a chunk with more than two tokens has more than one
  informative moment and can fragment across units.  This layer is correct for
  two token chunks; longer chunks need integration over a chunk, which needs
  boundaries.
* With silence between chunks the trace resets on its own, so chunk boundaries
  are supplied by the stimulus rather than discovered.  A gapless stream
  (``inter_gap = 0``) is the honest test and is supported by the stimulus
  module, but nothing here solves segmentation.
"""
from __future__ import annotations

import numpy as np

from .config import L2Config


class Layer2:
    """A population of chunk selective units reading layer 1 activity."""

    def __init__(self, n_channels: int, cfg: L2Config | None = None):
        self.N = n_channels
        self.cfg = cfg or L2Config()
        rng = np.random.default_rng(self.cfg.seed)
        # small random non negative masks; the asymmetry breaks the symmetry
        # between units, nothing else does
        self.M = rng.random((self.cfg.n_units, n_channels, n_channels)) * self.cfg.w_init
        if self.cfg.no_self_pairs:
            self._offdiag = ~np.eye(n_channels, dtype=bool)
            self.M *= self._offdiag[None, :, :]
        else:
            self._offdiag = np.ones((n_channels, n_channels), dtype=bool)
        self.reset_state()

    # ---- state (not weights, and not learning statistics) ------------
    def reset_state(self):
        """Clear the dynamical state between streams.

        Weights are preserved, and so are the cumulative learning statistics
        (``win_counts``, ``n_recruit``), which describe the training that has
        already happened and would otherwise be erased before they are read.
        """
        self.r = np.zeros(self.N)      # first stage of the slow conductance
        self.s = np.zeros(self.N)      # second stage, the trace the units read
        self.peak = 1e-9               # running peak drive, sets the plasticity gate
        if not hasattr(self, "win_counts"):
            self.win_counts = np.zeros(self.cfg.n_units, dtype=int)
            self.n_recruit = 0

    # ---- one integration step ---------------------------------------
    def step(self, E: np.ndarray, dt: float, learn: bool = True):
        """Advance by ``dt`` given the layer 1 rates ``E``; return (y, D)."""
        c = self.cfg
        # two stage slow conductance: rise then decay
        self.r += dt * (-self.r + E) / c.tau_rise
        self.s += dt * (-self.s + self.r) / c.tau_decay

        D = np.outer(E, self.s)
        if c.no_self_pairs:
            D = D * self._offdiag           # subunits pair distinct channels
        nD = float(np.linalg.norm(D))
        if nD > self.peak:
            self.peak = nD

        y = np.maximum(np.tensordot(self.M, D, axes=([1, 2], [0, 1])), 0.0)

        if learn:
            # plasticity is gated on drive: no learning in near silence
            if nD > c.gate_frac * self.peak:
                Dh = D / nD
                flat = self.M.reshape(c.n_units, -1)
                norms = np.linalg.norm(flat, axis=1) + 1e-12
                match = np.tensordot(self.M, Dh, axes=([1, 2], [0, 1])) / norms
                k = int(np.argmax(match))
                if c.rho > 0.0 and match[k] < c.rho:
                    k = int(np.argmin(norms))       # recruit the least committed
                    self.n_recruit += 1
                self.M[k] += c.eta * (Dh - self.M[k])
                np.clip(self.M[k], 0.0, None, out=self.M[k])
                self.win_counts[k] += 1
            # decay acts on every unit at every step, winner or not
            self.M *= (1.0 - c.lam)
        return y, D

    # ---- run over a layer 1 history ---------------------------------
    def run(self, E_hist: np.ndarray, dt: float, learn: bool = True,
            record_every: int = 0):
        """Drive the layer with an ``(N, T)`` layer 1 rate history.

        Returns ``y_hist`` of shape ``(n_units, T)``, the trace history and,
        if ``record_every`` is set, snapshots of the mask norms over time."""
        N, T = E_hist.shape
        if N != self.N:
            raise ValueError(f"E has {N} channels, layer 2 expects {self.N}")
        y_hist = np.zeros((self.cfg.n_units, T))
        s_hist = np.zeros((N, T))
        snaps, snap_t = [], []
        for t in range(T):
            y, _ = self.step(E_hist[:, t], dt, learn=learn)
            y_hist[:, t] = y
            s_hist[:, t] = self.s
            if record_every and (t % record_every == 0):
                snaps.append(self.mask_norms.copy())
                snap_t.append(t * dt)
        return dict(y=y_hist, s=s_hist,
                    norm_traj=np.array(snaps) if snaps else np.empty((0, self.cfg.n_units)),
                    norm_t=np.array(snap_t))

    # ---- readout -----------------------------------------------------
    @property
    def mask_norms(self) -> np.ndarray:
        return np.linalg.norm(self.M.reshape(self.cfg.n_units, -1), axis=1)

    @property
    def committed(self) -> np.ndarray:
        """Boolean mask of units that survived the decay competition."""
        n = self.mask_norms
        top = n.max()
        if top <= 0.0:
            return np.zeros(self.cfg.n_units, dtype=bool)
        return n >= self.cfg.commit_frac * top

    @property
    def n_components(self) -> int:
        """How many chunk types the layer settled on."""
        return int(self.committed.sum())
