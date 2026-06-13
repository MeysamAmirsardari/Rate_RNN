"""
live_demo_cortical.engine
================

A **streaming** stepper for the model0 A1 RNN.

``model0.simulate`` runs a whole pre-built stimulus array and resets all
state at call time -- unusable for a continuous microphone stream, where
the dynamical state (E, I, the multi-timescale depression variables, the
eligibility trace, the plastic weights) must persist across audio blocks.

``LiveEngine`` holds that state and advances it block by block.  Its
per-step update is a line-for-line mirror of the loop in
``model0/model.py`` -- it imports model0's own ``A1Config``,
``_build_inh_matrices`` and ``_relu`` and reproduces every branch
(``stp_enabled`` / single- vs multi-timescale STD / bounded vs additive
plasticity / self-plasticity clamp).  The model itself is unchanged; this
is purely an alternate driver.

    If ``model0/model.py``'s update is ever edited, mirror the change here.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from model0 import A1Config
from model0.model import _build_inh_matrices, _relu


class LiveEngine:
    """Persistent-state, block-wise driver for the model0 dynamics."""

    def __init__(self, cfg: A1Config, learn: bool = True,
                 W_init: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.learn = learn
        N = cfg.N
        rng = np.random.default_rng(seed)

        self.M_EI, self.M_IE = _build_inh_matrices(cfg)

        if W_init is None:
            self.W = cfg.W_init_scale * np.abs(rng.standard_normal((N, N)))
        else:
            self.W = np.asarray(W_init, dtype=float).copy()
        if not cfg.plastic_self:
            np.fill_diagonal(self.W, 0.0)

        # ---- dynamical state ----
        self.E = np.zeros(N)
        self.Iv = np.zeros(N)
        self.u = cfg.U * np.ones(N)
        self.x = np.ones(N)
        self.tr = np.zeros(N)

        # ---- multi-timescale STD state ----
        if cfg.stp_enabled and cfg.multiscale_std:
            self._tau_std = np.asarray(cfg.tau_std, dtype=float)
            self._decay_std = np.exp(-cfg.dt / self._tau_std)
            self._w_std = np.asarray(cfg.w_std, dtype=float)
            self.D_std = np.zeros((N, len(self._tau_std)))
        else:
            self.D_std = None

        self._eye = np.diag_indices(N)
        self.n_steps = 0

    # -----------------------------------------------------------------
    def step_block(self, drive: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance the model by ``drive.shape[1]`` steps.

        Parameters
        ----------
        drive : (N, k)
            Per-channel TC input for ``k`` consecutive 1-ms frames.

        Returns
        -------
        dict with histories over the block:
            E      (N, k)  excitatory rate
            I      (N, k)  inhibitory rate
            x_eff  (N, k)  available TC resource (depression gauge)
            inh    (N, k)  inhibitory current onto E (M_IE @ I)
            tr     (N, k)  eligibility trace (low-passed E) -- the *pre*-side of
                           the Hebbian rule, so ``<E_i(t) tr_j(t)>`` is a
                           directed (causal) coincidence detector j -> i.
        """
        cfg = self.cfg
        N, k = drive.shape
        if N != cfg.N:
            raise ValueError(f"drive has {N} rows but cfg.N = {cfg.N}")
        dt = cfg.dt

        E_h = np.empty((N, k))
        I_h = np.empty((N, k))
        x_h = np.empty((N, k))
        inh_h = np.empty((N, k))
        tr_h = np.empty((N, k))

        E, Iv, u, x, tr, W = (self.E, self.Iv, self.u, self.x, self.tr, self.W)

        for t in range(k):
            s = drive[:, t]

            # ---- TC drive (gated by short-term depression) ----
            if not cfg.stp_enabled:
                x_eff = np.ones(N)
                tm_in = cfg.A_TC * cfg.U * s
            elif cfg.multiscale_std:
                x_eff = np.maximum(0.0, 1.0 - self.D_std @ self._w_std)
                tm_in = cfg.A_TC * cfg.U * x_eff * s
            else:
                x_eff = x
                tm_in = cfg.A_TC * u * x * s

            rec_E = W @ E
            inh_to_E = self.M_IE @ Iv

            net_E = tm_in + rec_E - inh_to_E
            net_I = self.M_EI @ E

            dE = (-E + net_E) / cfg.tau_E
            dI = (-Iv + _relu(net_I)) / cfg.tau_I
            if cfg.stp_enabled and not cfg.multiscale_std:
                du = (cfg.U - u) / cfg.tau_F + cfg.U * (1.0 - u) * s
                dx = (1.0 - x) / cfg.tau_D - u * x * s
            dtr = (-tr + E) / cfg.tau_trace

            # ---- Hebbian rate-STDP on W (row=post, col=pre) ----
            if self.learn:
                En = E / cfg.W_norm
                trn = tr / cfg.W_norm
                if cfg.bounded_plasticity:
                    dW = (cfg.eta_LTP * (cfg.W_max - W) * np.outer(En, trn)
                          - cfg.eta_LTD * np.outer(trn, En)
                          - cfg.eta_het * W * En[None, :]
                          - cfg.W_decay * W)
                else:
                    dW = (cfg.eta_LTP * np.outer(En, trn)
                          - cfg.eta_LTD * np.outer(trn, En)
                          - cfg.W_decay * W)

            # ---- Euler step ----
            E = E + dt * dE
            Iv = Iv + dt * dI
            if cfg.stp_enabled:
                if cfg.multiscale_std:
                    self.D_std = self.D_std * self._decay_std + cfg.U_std * s[:, None]
                else:
                    u = u + dt * du
                    x = x + dt * dx
            tr = tr + dt * dtr
            if self.learn:
                W = W + dt * dW
                if not cfg.plastic_self:
                    W[self._eye] = 0.0
                np.clip(W, 0.0, cfg.W_max, out=W)
                if cfg.plastic_self:
                    W[self._eye] = np.minimum(W.diagonal(), cfg.W_max_self)

            if cfg.stp_enabled and not cfg.multiscale_std:
                np.clip(u, 0.0, 1.0, out=u)
                np.clip(x, 0.0, 1.0, out=x)
            np.clip(E, 0.0, None, out=E)

            E_h[:, t] = E
            I_h[:, t] = Iv
            x_h[:, t] = x_eff
            inh_h[:, t] = inh_to_E
            tr_h[:, t] = tr

        # write state back
        self.E, self.Iv, self.u, self.x, self.tr, self.W = E, Iv, u, x, tr, W
        self.n_steps += k

        return dict(E=E_h, I=I_h, x_eff=x_h, inh=inh_h, tr=tr_h)
