"""
model0.model
============

Rate-based recurrent model of A1 with tone-selective inhibition.

State variables (per integration step)
--------------------------------------
    E   (N,)   excitatory rates
    I   (N,)   inhibitory rates  -- one per channel (vs. scalar in model/)
    u   (N,)   TC release probability  (Tsodyks-Markram)
    x   (N,)   TC available resources  (Tsodyks-Markram)
    tr  (N,)   postsynaptic eligibility trace  (for E->E learning)
    W   (N,N)  plastic recurrent E->E weights, row = post, col = pre

Dynamics
--------
    tau_E dE/dt = -E + (tm_in + W @ E - M_IE @ I);   E := relu(E)
    tau_I dI/dt = -I + relu(M_EI @ E)

The rectifying nonlinearity is applied to the *rate* E (a post-step
clamp), not to the net input.  Inhibition therefore acts directly on
the firing rate: a cell driven net-negative is pushed toward zero at a
speed set by the inhibition magnitude, rather than merely left to decay
at the leak rate 1/tau_E.  Steady states are unchanged (E_ss = relu of
the net input); only the suppression transients are sharper.  (The I
equation keeps relu on the input, but this is a no-op: M_EI has only
non-negative entries and E >= 0, so M_EI @ E is never negative.)

where M_EI and M_IE are FIXED structured matrices (diag = self-strong,
off-diag = lateral-weak).

There is no spike-frequency adaptation -- the inhibitory unit I_i is the
adaptation, by construction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .config import A1Config


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _build_inh_matrices(cfg: A1Config) -> tuple[np.ndarray, np.ndarray]:
    """Construct E->I and I->E matrices: diag = self, off-diag = lateral."""
    N = cfg.N
    M_EI = cfg.w_EI_lat * np.ones((N, N)) + (cfg.w_EI_self - cfg.w_EI_lat) * np.eye(N)
    M_IE = cfg.w_IE_lat * np.ones((N, N)) + (cfg.w_IE_self - cfg.w_IE_lat) * np.eye(N)
    return M_EI, M_IE


def simulate(
    stim: np.ndarray,
    cfg: Optional[A1Config] = None,
    W_init: Optional[np.ndarray] = None,
    learn: bool = True,
    record_W_every: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the tone-selective-inhibition A1 model.

    Returns a dict with the same keys as ``model.simulate`` except:
        - ``out['I']`` has shape ``(N, T)`` (per-channel) instead of ``(T,)``.
        - There is no ``out['a']`` (no adaptation variable).
        - ``out['inh_to_E']`` (N, T) — the inhibitory current onto each E
          unit, ``M_IE @ I`` — useful for showing predictive suppression.
    """
    if cfg is None:
        cfg = A1Config()

    N, T = stim.shape
    if N != cfg.N:
        raise ValueError(f"stim has {N} channels but cfg.N = {cfg.N}")

    dt = cfg.dt
    rng = np.random.default_rng(seed)

    M_EI, M_IE = _build_inh_matrices(cfg)

    # ---- weight init ----
    if W_init is None:
        W = cfg.W_init_scale * np.abs(rng.standard_normal((N, N)))
    else:
        W = np.asarray(W_init, dtype=float).copy()
    if not cfg.plastic_self:
        np.fill_diagonal(W, 0.0)

    # ---- state ----
    E  = np.zeros(N)
    Iv = np.zeros(N)               # per-channel inhibitory rate
    u  = cfg.U * np.ones(N)
    x  = np.ones(N)
    tr = np.zeros(N)

    # ---- histories ----
    E_h     = np.zeros((N, T))
    I_h     = np.zeros((N, T))
    u_h     = np.zeros((N, T))
    x_h     = np.zeros((N, T))
    tm_h    = np.zeros((N, T))
    rec_h   = np.zeros((N, T))
    inh_h   = np.zeros((N, T))

    W_traj, W_t = [], []
    eye_diag_idx = np.diag_indices(N)

    for t in range(T):
        s = stim[:, t]

        tm_in    = cfg.A_TC * u * x * s
        rec_E    = W @ E
        inh_to_E = M_IE @ Iv            # (N,) per-channel inhibition

        net_E = tm_in + rec_E - inh_to_E
        net_I = M_EI @ E                 # each I_i driven mostly by E_i

        # relu is applied to the *rate* E (post-step clamp below), not to
        # net_E -- so inhibition acts directly on the firing rate.
        dE  = (-E  + net_E) / cfg.tau_E
        dI  = (-Iv + _relu(net_I)) / cfg.tau_I
        du  = (cfg.U - u) / cfg.tau_F + cfg.U * (1.0 - u) * s
        dx  = (1.0 - x) / cfg.tau_D            - u * x * s
        dtr = (-tr + E) / cfg.tau_trace

        # --- Hebbian rate-STDP on W (row = post, col = pre) ---
        # Two rules selectable via cfg.bounded_plasticity (see config.py
        # for the docstring on the equilibrium math):
        #   False (default) -- additive LTP + additive LTD + slow decay,
        #                      clipped at W_max (legacy behaviour).
        #   True            -- multiplicative bounded LTP +
        #                      directional anti-Hebbian LTD +
        #                      heterosynaptic LTD +
        #                      slow decay.  Each weight settles at its
        #                      own LTP/LTD-balance fraction of W_max,
        #                      approached exponentially.
        if learn:
            En  = E  / cfg.W_norm
            trn = tr / cfg.W_norm
            if cfg.bounded_plasticity:
                dW = (cfg.eta_LTP * (cfg.W_max - W) * np.outer(En, trn)
                      - cfg.eta_LTD * np.outer(trn, En)
                      - cfg.eta_het * W * En[None, :]
                      - cfg.W_decay * W)
            else:
                dW = (cfg.eta_LTP * np.outer(En,  trn)
                      - cfg.eta_LTD * np.outer(trn, En)
                      - cfg.W_decay * W)

        # --- Euler step ---
        E  += dt * dE
        Iv += dt * dI
        u  += dt * du
        x  += dt * dx
        tr += dt * dtr
        if learn:
            W += dt * dW
            if not cfg.plastic_self:
                W[eye_diag_idx] = 0.0
            np.clip(W, 0.0, cfg.W_max, out=W)
            if cfg.plastic_self:
                W[eye_diag_idx] = np.minimum(W.diagonal(), cfg.W_max_self)

        np.clip(u, 0.0, 1.0, out=u)
        np.clip(x, 0.0, 1.0, out=x)
        np.clip(E, 0.0, None, out=E)     # relu on the rate: E >= 0

        E_h[:,  t]  = E
        I_h[:,  t]  = Iv
        u_h[:,  t]  = u
        x_h[:,  t]  = x
        tm_h[:, t]  = tm_in
        rec_h[:, t] = rec_E
        inh_h[:, t] = inh_to_E

        if record_W_every and (t % record_W_every == 0):
            W_traj.append(W.copy())
            W_t.append(t * dt)

    return dict(
        E=E_h, I=I_h, u=u_h, x=x_h,
        tm_in=tm_h, rec_E=rec_h, inh_to_E=inh_h,
        W_final=W.copy(),
        W_traj=W_traj, W_t=np.array(W_t),
        t=np.arange(T) * dt,
        cfg=cfg,
    )
