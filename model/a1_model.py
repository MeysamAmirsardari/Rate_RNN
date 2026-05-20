"""
a1_model.py
===========

Rate-based recurrent model of primary auditory cortex (A1).

Architecture (matches the schematic)::

    Input (per tonotopic channel)
        |    short-term depression  (Tsodyks-Markram)
        v
        E  <-----------.   plastic, with LTP/LTD
        |              |   (lateral *and* self-recurrent)
        |  Exc->Inh    |
        v   (fixed)    |
    [ global I pool ]--'   Inh->Exc (fixed, divergent)

Only E->E is plastic.  Exc<->Inh weights are static (Isaacson & Scanziani,
Neuron 2011 — feedback inhibition in cortex).  Inhibition is global, which
captures the broadly tuned PV-driven inhibition that dominates A1 (Li,
Ibrahim et al., Nat. Neurosci. 2014; Moore & Wehr, J. Neurosci. 2013).

Time constants and STP parameters are chosen from in-vitro and in-vivo
A1 measurements (see A1Config docstring).  Numerical integration is
forward Euler at dt = 1 ms, well below every time constant in the model.

Public API
----------
    cfg = A1Config(...)
    out = simulate(stim, cfg)        # stim : (N, T) ndarray in [0, 1]
    out['E']          (N, T)         excitatory rates
    out['I']          (T,)           global inhibitory rate
    out['u'], ['x']   (N, T)         TM utilisation / available resources
    out['a']          (N, T)         spike-frequency adaptation variable
    out['tm_in']      (N, T)         TM-gated TC drive
    out['rec_E']      (N, T)         recurrent E->E current
    out['glob_I']     (T,)           inhibition seen by every E unit
    out['W_final']    (N, N)         trained recurrent weight matrix
    out['W_traj']     list[(N, N)]   snapshots (if record_W_every > 0)
    out['W_t']        (n_snap,)      time-points of snapshots
    out['t']          (T,)           time vector

Convention for W: row = post, column = pre.  rec_E = W @ E.

References
----------
- Markram, Wang & Tsodyks (1998) PNAS 95:5323.  STP model.
- Tsodyks & Markram (1997) PNAS 94:719.
- Wehr & Zador (2005) J. Neurosci. 25:7521. Synaptic depression in A1.
- Cruikshank, Lewis & Connors (2007) Neuron 56:1043.  Strong TC depression.
- Gentet et al. (2010) Neuron 65:422.  Cortical pyramidal membrane tau.
- Hu, Gan & Jonas (2014) Science 345:1255263.  Fast-spiking PV interneurons.
- Bi & Poo (1998) J. Neurosci. 18:10464.  STDP window ~ 20-40 ms.
- Song, Miller & Abbott (2000) Nat. Neurosci. 3:919.  LTP > LTD asymmetry.
- Pfister & Gerstner (2006) J. Neurosci. 26:9673.  Trace-based learning rules.
- Madison & Nicoll (1984) J. Physiol. 354:319.  mAHP / Ca-activated K current.
- Storm (1990) Prog. Brain Res. 83:161.  K conductances in pyramidal cells.
- Latham, Richmond, Nelson, Nirenberg (2000) J. Neurophysiol. 83:808.
  Rate-equation form of spike-frequency adaptation.
- Lu, Liang, Schreiner (2008) J. Neurosci. 28:8410.  Adaptation in A1.
- Mill, Coath, Wennekers, Denham (2011) PLoS Comp. Biol. 7:e1002265.
  Adaptation as an SSA mechanism in A1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


# =====================================================================
# Configuration
# =====================================================================
@dataclass
class A1Config:
    """Biologically grounded defaults for primary auditory cortex.

    All time constants are in **seconds**.  Rates are unitless but the
    parameters are scaled so that E ~ 0–60 corresponds loosely to Hz.
    """

    # ---- network ----
    N:  int   = 8                  # tonotopic E channels
    dt: float = 1e-3               # 1 ms — << every tau in the model

    # ---- rate dynamics ----
    tau_E: float = 20e-3           # pyramidal cell membrane tau (Gentet+2010)
    tau_I: float = 10e-3           # fast PV interneuron tau    (Hu+2014)

    # ---- E <-> I (FIXED, non-plastic) ----
    # Inhibitory loop gain ~ 0.6 keeps the network in the inhibition-stabilised
    # regime (Ozeki, Finn, Schaffer, Miller & Ferster, Neuron 2009) without
    # smothering recurrent prediction signals.
    W_EI: float = 0.6              # each E -> global I pool
    W_IE: float = 1.0              # global I -> every E (broad inhibition)

    # ---- Short-term depression on TC input (Tsodyks-Markram) ----
    # Wehr & Zador 2005; Cruikshank+2007; Markram+1998.
    tau_D: float = 0.30            # depression recovery (s)
    tau_F: float = 0.10            # facilitation decay (weak at TC)  (s)
    U:     float = 0.5             # baseline release prob — depressing regime
    A_TC:  float = 35.0            # absolute thalamocortical efficacy

    # ---- Hebbian plasticity on E->E (lateral + self) ----
    # Trace-based rule, rate-coded version of Bi-Poo / Pfister-Gerstner STDP.
    tau_trace:   float = 30e-3     # post trace ~ STDP window
    eta_LTP:     float = 0.8       # LTP rate
    eta_LTD:     float = 0.7       # LTD slightly weaker (Song+2000)
    W_decay:     float = 5e-4      # slow homeostatic forgetting (per s)
    # Cortico-cortical EPSPs are smaller than thalamocortical EPSPs in A1
    # (Stratford+1996, Cruikshank+2007), so the recurrent unitary strength
    # is bounded below the TC unitary strength.
    W_max:       float = 0.7       # soft upper bound on cross-weights
    # Self-recurrent (diagonal) cap.  The inhibition-stabilised regime
    # requires W_self < 1 + W_IE*W_EI; we sit comfortably below that bound.
    W_max_self:  float = 0.5
    W_norm:      float = 15.0      # rate scale that normalises the rule

    plastic_self: bool = True      # allow plastic self-recurrent E->E

    # ---- Spike-frequency / firing-rate adaptation (per E unit) ----
    # Models a Ca-activated K conductance (mAHP) — universal in cortical
    # pyramidal cells (Madison & Nicoll 1984 ; Storm 1990).  The adaptation
    # variable ``a`` accumulates with the cell's *own* firing rate
    # (Latham et al. 2000).  In A1 the measured kinetics are 100-500 ms
    # (Lu et al. 2008 ; Abolafia et al. 2011).
    #
    # **Divisive form** (Heeger 1992 ; Carandini & Heeger 2012):
    #     E_ss = relu(net) / (1 + g_a * a)
    # The shunting f-I gain divides whatever drive the neuron receives,
    # so it suppresses *both* the TC drive and any recurrent prediction
    # boost.  Subtractive adaptation cannot suppress the prediction-driven
    # boost (the boost lives in the numerator) — divisive can.
    #
    # This is the postsynaptic "memory" that lets predictive pre-activation
    # depress the subsequent driven response (Mill et al. 2011), i.e. the
    # mechanism that implements the
    #   "predictive pre-activation -> suppression via depression"
    # narrative for Model 1 (no iSTDP, blanket inhibition).
    tau_a: float = 0.05            # 50 ms — mAHP, matched to tone duration
    g_a:   float = 4.0             # divisive gain on relu(net_E)

    # ---- initial conditions ----
    W_init_scale: float = 0.0      # >0 to seed random initial E->E


# =====================================================================
# Nonlinearity
# =====================================================================
def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


# =====================================================================
# Main simulator
# =====================================================================
def simulate(
    stim: np.ndarray,
    cfg: Optional[A1Config] = None,
    W_init: Optional[np.ndarray] = None,
    learn: bool = True,
    record_W_every: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the A1 model for the duration encoded in ``stim``.

    Parameters
    ----------
    stim : ndarray, shape (N, T)
        Pre-synaptic feedforward (thalamocortical) drive per channel.
        Treat values as activation in [0, 1]; the postsynaptic current
        is ``A_TC * u * x * stim`` after short-term depression.
    cfg : A1Config, optional
        Model configuration.  Defaults are A1-biological (see class docstring).
    W_init : ndarray, shape (N, N), optional
        Initial recurrent E->E weights.  If ``None``, random non-negative
        weights with scale ``cfg.W_init_scale`` are used.
    learn : bool
        If False the recurrent weights are frozen (useful for test/probe runs).
    record_W_every : int
        If > 0, store a snapshot of W every this-many integration steps.
    seed : int, optional
        Seed for the (only) source of randomness — initial weights.

    Returns
    -------
    out : dict (see module docstring)
    """
    if cfg is None:
        cfg = A1Config()

    N, T = stim.shape
    if N != cfg.N:
        raise ValueError(f"stim has {N} channels but cfg.N = {cfg.N}")

    dt = cfg.dt
    rng = np.random.default_rng(seed)

    # ---- weight init ----
    if W_init is None:
        W = cfg.W_init_scale * np.abs(rng.standard_normal((N, N)))
    else:
        W = np.asarray(W_init, dtype=float).copy()
    if not cfg.plastic_self:
        np.fill_diagonal(W, 0.0)

    # ---- state ----
    E   = np.zeros(N)
    Iv  = 0.0                       # global inhibitory rate (scalar)
    u   = cfg.U * np.ones(N)
    x   = np.ones(N)
    tr  = np.zeros(N)               # post-synaptic eligibility trace
    a   = np.zeros(N)               # spike-frequency adaptation variable

    # ---- histories ----
    E_h     = np.zeros((N, T))
    I_h     = np.zeros(T)
    u_h     = np.zeros((N, T))
    x_h     = np.zeros((N, T))
    a_h     = np.zeros((N, T))
    tm_h    = np.zeros((N, T))
    rec_h   = np.zeros((N, T))
    gI_h    = np.zeros(T)

    W_traj, W_t = [], []

    eye_diag_idx = np.diag_indices(N)

    for t in range(T):
        s = stim[:, t]

        # --- inputs ---
        tm_in  = cfg.A_TC * u * x * s          # TM-gated TC drive
        rec_E  = W @ E                          # plastic E->E
        glob_I = cfg.W_IE * Iv                  # broadcast global inhibition

        # Divisive (shunting) adaptation — ``a`` reduces the postsynaptic
        # f-I gain rather than subtracting a fixed current.  Because the
        # divisor acts on *everything* the neuron receives, the same
        # recurrent prediction boost that loaded ``a`` is now itself
        # suppressed (Carandini & Heeger 2012).  ``a`` is still driven by
        # the cell's actual firing E (postsynaptic), so any source of
        # activity accumulates adaptation.
        net_E  = tm_in + rec_E - glob_I
        net_I  = cfg.W_EI * E.sum()             # all E drive the single I pool

        # --- derivatives ---
        dE  = (-E  + _relu(net_E) / (1.0 + cfg.g_a * a)) / cfg.tau_E
        dIv = (-Iv + _relu(net_I)) / cfg.tau_I
        # Tsodyks-Markram on each TC synapse
        du  = (cfg.U - u) / cfg.tau_F + cfg.U * (1.0 - u) * s
        dx  = (1.0 - x) / cfg.tau_D            - u * x * s
        # post-synaptic eligibility trace (Ca-trace, kept at tau_trace)
        dtr = (-tr + E) / cfg.tau_trace
        # spike-frequency adaptation (Latham et al. 2000 form)
        da  = (-a + E) / cfg.tau_a

        # --- Hebbian rate-STDP update on W (row = post, col = pre) ---
        #   dW_ij / dt =  eta_LTP * post_i * trace_pre_j
        #              - eta_LTD * trace_post_i * pre_j
        #              - W_decay * W_ij
        # The trace asymmetry encodes temporal order:
        #   pre-before-post  -> LTP (trace_pre is high when post fires)
        #   post-before-pre  -> LTD (trace_post is high when pre fires)
        if learn:
            En  = E  / cfg.W_norm
            trn = tr / cfg.W_norm
            dW  = (cfg.eta_LTP * np.outer(En,  trn)
                   - cfg.eta_LTD * np.outer(trn, En)
                   - cfg.W_decay * W)

        # --- Euler step ---
        E  += dt * dE
        Iv += dt * dIv
        u  += dt * du
        x  += dt * dx
        tr += dt * dtr
        a  += dt * da
        if learn:
            W += dt * dW
            if not cfg.plastic_self:
                W[eye_diag_idx] = 0.0
            np.clip(W, 0.0, cfg.W_max, out=W)
            # diagonal has a tighter bound to keep the network stable
            if cfg.plastic_self:
                W[eye_diag_idx] = np.minimum(W.diagonal(), cfg.W_max_self)

        # keep STP variables in their physiological range
        np.clip(u, 0.0, 1.0, out=u)
        np.clip(x, 0.0, 1.0, out=x)

        # --- record ---
        E_h[:,  t] = E
        I_h[t]     = Iv
        u_h[:,  t] = u
        x_h[:,  t] = x
        a_h[:,  t] = a
        tm_h[:, t] = tm_in
        rec_h[:, t] = rec_E
        gI_h[t]    = glob_I

        if record_W_every and (t % record_W_every == 0):
            W_traj.append(W.copy())
            W_t.append(t * dt)

    return dict(
        E=E_h, I=I_h, u=u_h, x=x_h, a=a_h,
        tm_in=tm_h, rec_E=rec_h, glob_I=gI_h,
        W_final=W.copy(),
        W_traj=W_traj, W_t=np.array(W_t),
        t=np.arange(T) * dt,
        cfg=cfg,
    )
