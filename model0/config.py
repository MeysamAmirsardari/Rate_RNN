"""
model0.config
=============

Configuration for the tone-selective inhibition variant of the A1 rate
model.

Architecture (vs. the original ``model/``)
------------------------------------------
Each tonotopic channel has its OWN excitatory unit *and* its OWN
inhibitory unit.  Inhibition is no longer a single broadband pool;
it inherits the channel's tonotopic selectivity.

    E_i  <-- TC input (gated by Tsodyks-Markram STD)
     |  ^
     |  |  plastic E->E (lateral + self), rate-STDP — same rule as model/
     v  |
   [ W_EE @ E ]
     |
     | <-- I_j, with strong self (i==j) and weak lateral (i!=j)
     v
    E_i  ----> I_i  (E -> I, strong self, weak lateral; FIXED)

No spike-frequency adaptation.  Its role is taken over by the slow,
tone-selective inhibitory unit I_i.

Why this works
--------------
After learning W[B<-A] is strong.  During tone A:
    1. E_A is driven by TC input.
    2. E_A pre-activates E_B via W[B<-A].
    3. E_B drives I_B (its OWN inhibitory unit).
    4. Because tau_I > tau_E, I_B builds up over the 50 ms of tone A
       and persists past the tone-A/tone-B boundary.
When tone B itself arrives, I_B is already elevated and subtractively
suppresses E_B.  The deviant case (W[B<-A] = 0) has no pre-activation
and therefore no residual I_B, so the tone-B response is larger.

The suppression is now carried by a real inhibitory interneuron with
its own state — not by a postsynaptic gain factor.  This maps onto
SST somatostatin interneurons, which are tone-tuned and slower than
PV (Kvitsiani et al., Nature 2013; Pi et al., Nature 2013).

References
----------
- Markram, Wang & Tsodyks (1998) PNAS 95:5323.  STP model.
- Wehr & Zador (2005) J. Neurosci. 25:7521.  TC depression in A1.
- Gentet et al. (2010) Neuron 65:422.  Pyramidal membrane tau.
- Kvitsiani et al. (2013) Nature 498:363.  SST vs PV dynamics in cortex.
- Pi et al. (2013) Nature 503:521.  SST interneurons: slow, tone-tuned.
- Song, Miller & Abbott (2000) Nat. Neurosci. 3:919.  LTP > LTD.
- Pfister & Gerstner (2006) J. Neurosci. 26:9673.  Trace-based STDP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class A1Config:
    """Tone-selective inhibition (one E + one I per channel)."""

    # ---- network ----
    # Two channels (A and B) only.  No tonotopic overlap — each tone
    # activates exactly one channel.  This is the minimal setup for
    # studying the pre-activation -> selective-inhibition -> suppression
    # cascade in isolation.
    N:  int   = 2                  # 2 tonotopic channels (A and B)
    dt: float = 1e-3               # 1 ms — << every tau in the model

    # ---- rate dynamics ----
    # tau_I > tau_E is the LOAD-BEARING piece of the architecture.  PV
    # (~10 ms) is the wrong cell type for predictive suppression; SST
    # (~30-100 ms effective IPSC kinetics) is right.  80 ms keeps I_B
    # alive across the intra-sequence gap so it can still suppress E_B
    # when tone B arrives.
    tau_E: float = 20e-3           # pyramidal membrane (Gentet+2010)
    tau_I: float = 80e-3           # SST-like (Kvitsiani+2013, Pi+2013)

    # ---- selective E <-> I (FIXED, non-plastic) ----
    # Each E drives mostly its OWN I; each I suppresses mostly its OWN E.
    # Lateral coupling is weak but non-zero -- real cortical inhibition
    # spans more than one column; the "selective" part is the *weighting*.
    #
    # w_IE_self = 4.0 represents SST cells combining strong dendritic
    # shunting with somatic inhibition (Kvitsiani+2013; Pala &
    # Petersen 2018).  This high value is *load-bearing*: it
    # asymmetrically amplifies suppression DELIVERY (I -> E) without
    # symmetrically increasing the loop gain that would dampen the
    # E_A drive that loads the predictive cascade in the first place.
    # Raising w_EI_self instead would damp both ends symmetrically
    # and produce no Pareto gain (verified empirically).
    w_EI_self: float = 0.20        # E_i -> I_i (strong)
    w_EI_lat:  float = 0.05        # E_i -> I_{j!=i} (weak)
    w_IE_self: float = 0.65        # I_i -> E_i (strong self-inhibition)
    w_IE_lat:  float = 0.20        # I_i -> E_{j!=i} (weak lateral)

    # ---- Short-term depression on TC input (Tsodyks-Markram) ----
    # Identical to model/.  Wehr & Zador 2005; Cruikshank+2007; Markram+1998.
    # Master switch: when False, ALL TC short-term plasticity is bypassed.
    # u, x (single-timescale) and D_k (multi-timescale) state evolution
    # are skipped entirely and the TC drive becomes A_TC * U * s -- the
    # resting value the TM machinery would deliver in the absence of
    # spikes.  Useful as a control to isolate STP contributions from
    # the rest of the cascade.
    stp_enabled: bool = True
    tau_D: float = 0.30
    tau_F: float = 0.10
    U:     float = 0.5
    A_TC:  float = 35.0

    # ---- Hebbian plasticity on E->E (lateral + self) ----
    # Same trace-based rate-STDP rule as model/.  W_norm is calibrated for
    # the new firing-rate regime under selective inhibition (see model.py
    # for the analytical scaling argument).
    tau_trace:   float = 30e-3
    eta_LTP:     float = 0.8
    # eta_LTD calibrated for the bounded rule.  At the diagonal the LTP
    # and LTD pre/post factors collapse to the same E*tr, so the net
    # drive is E*tr*(eta_LTP*(W_max - W) - eta_LTD).  Requiring this to
    # be positive at W=0 means eta_LTD < eta_LTP * W_max = 0.20; we sit
    # comfortably below that at 0.10 so self synapses settle at
    #   W*_self = W_max - eta_LTD/eta_LTP = 0.25 - 0.125 = 0.125
    # while cross synapses (anti-causal/causal STDP ratio r ~ 0.1)
    # settle close to W_max -- still set by the LTP/LTD balance, not
    # by the clip.
    eta_LTD:     float = 0.10
    W_decay:     float = 5e-4
    # Recurrent unitary strength bounded well below TC strength.  In
    # cortex, intracortical EPSPs are ~25-30% of thalamocortical EPSPs
    # (Stratford+1996; Cruikshank+2007).  W_max=0.25 sits at the lower
    # end of that range, which (together with w_IE_self=4.0) keeps the
    # pre-activation of E_B during tone A subtle (E_B^pre/E_A < 13%)
    # while preserving the ~5% MMN polarity that's empirically observed
    # in A1 (Ulanovsky+2003; Nieto-Diego & Malmierca 2016).
    # Single cap shared by self and lateral E->E synapses.  The bounded
    # rule sets each weight's fixed point internally (W_max - LTD/LTP·r),
    # so the cap no longer needs to differ between self and cross to
    # keep self synapses subordinate -- the dynamics do that on their
    # own.
    W_max:       float = 0.25
    W_max_self:  float = 0.25
    # W_norm calibrates the rate units in the Hebbian product
    # (E/W_norm)*(tr/W_norm).  Under the bounded rule the asymptote is
    # W_norm-independent (both LTP and LTD scale together), but the
    # convergence TIME scales as W_norm^2.  W_norm=20 keeps the cross
    # weight approach to W_max slow (~6000-trial timescale), which is
    # what most paradigms here want -- the network shouldn't have its
    # cross weights at the asymptote within a few hundred trials.
    # The AB/BA task is an exception: it specifically tests the
    # MMN-polarity signature, which requires W[B<-A] near the asymptote.
    # tasks/ab_ba_model0/ab_ba.py overrides W_norm=4 for that reason.
    W_norm:      float = 20.0

    plastic_self: bool = True

    # ---- Bounded plasticity (soft bound + heterosynaptic LTD) ----
    # When False (default), the trace-based STDP rule is additive:
    #   dW = eta_LTP * outer(E, tr) - eta_LTD * outer(tr, E) - W_decay * W
    # clipped at W_max.  Every plastic weight that receives sustained
    # net LTP saturates flat at W_max -- the cap visibly "sets" the
    # connection map rather than the dynamics doing so.
    #
    # When True, the rule is multiplicative with heterosynaptic LTD:
    #   dW = eta_LTP * (W_max - W) * outer(E, tr)    bounded Hebbian
    #       - eta_LTD * outer(tr, E)                  directional anti-Hebbian
    #       - eta_het * W * E[None, :]                heterosynaptic (pre-gated)
    #       - W_decay * W                             slow forgetting
    # Each weight settles at its own fraction of W_max set by the local
    # LTP / LTD / heterosynaptic balance; the approach is exponential
    # rather than linear; W_max becomes the biophysical asymptote
    # (finite AMPA-receptor capacity per synapse), not a clip the
    # dynamics hit.  References: van Rossum, Bi & Turrigiano (2000);
    # Gutig & Sompolinsky (2003); heterosynaptic LTD: Lynch et al.
    # (1977); Chistiakova & Volgushev (2009, review).
    bounded_plasticity: bool = True
    # Set to 0.  The pre-gated het-LTD term -eta_het*W*E_pre was added
    # as a per-pre stabilizer following Chistiakova & Volgushev (2009),
    # but in practice it pinned CROSS synapses below the W_max asymptote
    # (their causal partner fires often, making E_pre large).  Self
    # synapses are stabilized by the LTP/LTD balance alone, so removing
    # het-LTD selectively unblocks the cross-weight asymptote at W_max.
    # If het-LTD becomes necessary later (e.g. for an excitatory-pool
    # normalization story), reintroduce it at ~1e-4 to keep the
    # AB/BA cross-weight saturation intact.
    eta_het:            float = 0.0     # heterosynaptic LTD rate

    # ---- Multi-timescale short-term depression (optional) ----
    # When False (default), the TC synapse uses the single-timescale
    # Tsodyks-Markram resource x (recovery tau_D ~ 300 ms).  When True,
    # the TC depression is replaced by the three-timescale form from
    # STD_Playground_Emergent.m: three depression variables D_k per
    # channel, each driven by the tone input s and decaying at
    # tau_std[k].  The available resource is
    #     x_avail = max(0, 1 - sum_k w_std[k] * D_k)
    # and the TC drive becomes  A_TC * U * x_avail * s  (u-dynamics
    # are disabled; U serves as the baseline release probability).
    #
    # The slow (~5 s) component survives the 1 s inter-trial gap of the
    # roving paradigm and integrates across the 15 reps within a block,
    # producing the rep-1-vs-rep-15 repetition-suppression signature
    # observed in ECoG recordings of the local-global / roving task.
    # Reference: Mill, Coath, Wennekers, Denham (2011) PLoS Comp. Biol.
    # use exactly this multi-timescale STD as the SSA substrate.
    multiscale_std: bool = False
    tau_std: tuple = (0.100, 0.800, 5.000)   # s -- three timescales
    w_std:   tuple = (0.10, 0.35, 0.55)      # relative weight per scale
    # U_std calibrated for a binary-stimulus driver (model0 drives the
    # D_k by the tone input s rather than by the cell rate as in
    # STD_Playground_Emergent.m -- the binary drive is sharper, so U_std
    # is scaled down ~5x from the MATLAB default of 0.005 to land at
    # a realistic ~30% rep-1-vs-rep-15 suppression on the roving task).
    U_std:   float = 0.001                    # per-step utilisation

    # ---- initial conditions ----
    W_init_scale: float = 0.0


# =====================================================================
#  Inhibition-structure presets
# =====================================================================
# The four inhibition weights (w_EI_self, w_EI_lat, w_IE_self, w_IE_lat)
# control how tone-selective the E<->I loop is.  Two named presets:
#
#   selective_inh()  -- the load-bearing default: strong self, weak
#                       lateral on both sides.  Each I_i inherits the
#                       tone preference of its E_i, so predictive
#                       pre-activation of E_B (via W[B<-A]) drives I_B
#                       specifically, which then suppresses E_B when
#                       tone B itself arrives.
#
#   uniform_inh()    -- control: all four weights collapsed so every
#                       (i,j) entry of M_EI and M_IE is equal.  Matched
#                       on row-sum to selective_inh (at N=5) so the
#                       *total* inhibitory drive per cell is preserved
#                       and only the SELECTIVITY differs.  Predictive
#                       suppression should collapse.
#
# Row-sum match (N=5):
#   E->I:  selective row sum = 0.6 + 4*0.05 = 0.80;  uniform = 0.80/5 = 0.16
#   I->E:  selective row sum = 4.0 + 4*0.20 = 4.80;  uniform = 4.80/5 = 0.96
def selective_inh(**kw) -> "A1Config":
    """Tone-selective inhibition (default A1Config values)."""
    return A1Config(**kw)


def uniform_inh(**kw) -> "A1Config":
    """Uniform inhibition: every E->I and I->E weight identical.

    M_EI and M_IE collapse to rank-1 (all-ones) matrices -- each I_i
    pools E indiscriminately and inhibits E_i indiscriminately.  Total
    inhibitory drive per cell is preserved (row-sum matched to the
    selective preset at N=5)."""
    defaults = dict(
        w_EI_self=0.16, w_EI_lat=0.16,
        w_IE_self=0.16, w_IE_lat=0.16,
    )
    defaults.update(kw)
    return A1Config(**defaults)


INH_PRESETS = {
    "selective": selective_inh,
    "uniform":   uniform_inh,
}


# =====================================================================
#  Disynaptic inhibitory loop gain  (the cross-paradigm invariant)
# =====================================================================
def inhibitory_loop_gain(cfg: "A1Config") -> float:
    """Top eigenvalue of the symmetrised disynaptic loop ``M_IE @ M_EI``.

    This is the dynamically relevant measure of global (population-pooling)
    inhibition -- the gain of the E -> I -> E negative-feedback loop.  Its
    all-to-all term grows as ``N**2 * w_lat**2``, so a preset calibrated at
    one channel count sits in a completely different inhibitory regime at
    another N unless the LATERAL weights are rescaled to hold this quantity
    fixed.  ``sfg_config`` does exactly that.
    """
    N = cfg.N
    J, eye = np.ones((N, N)), np.eye(N)
    M_EI = cfg.w_EI_lat * J + (cfg.w_EI_self - cfg.w_EI_lat) * eye
    M_IE = cfg.w_IE_lat * J + (cfg.w_IE_self - cfg.w_IE_lat) * eye
    G = M_IE @ M_EI
    return float(np.max(np.linalg.eigvalsh((G + G.T) / 2)))


# =====================================================================
#  SFG / recurrent-binding regime  (finalised tasks/sfg2 config, promoted)
# =====================================================================
# The Stochastic-Figure-Ground recalibration (tasks/sfg2) is the first
# paradigm run at the full A1 channel count (N = 37) rather than the N = 2
# AB/BA minimal circuit the A1Config defaults were tuned for.  Three regime
# constants define it -- two flat, ONE that must scale with N:
#
#   * W_max = W_max_self = 0.17  (recurrent ceiling).  A coherent figure
#     assembly of n channels has recurrent gain W_FF * (n - 1); the
#     bounded-rule fixed point W_FF -> 0.045 at this cap keeps the gain
#     0.41 < 1 at n = 10 -- a stable amplifier, not a runaway loop.  Set by
#     assembly SIZE, not channel count, so it does NOT scale with N.
#
#   * W_decay = 2e-3  (forgetting).  FF/GG = 1 + W_decay/(eta_LTP*c_gnd):
#     larger decay sharpens the figure/ground weight contrast by pruning
#     weakly-correlated ground synapses.  Flat in N.
#
#   * inhibitory_loop_gain = 6.3  (global inhibition).  THIS is the term
#     that explodes as N**2 if the laterals are held fixed.  We hold the
#     loop gain itself fixed and SOLVE the lateral weights for it at the
#     requested N -- the cross-paradigm invariant.  Per-channel (self)
#     weights keep their biophysical A1Config values; only the population-
#     pooling laterals scale (w_lat ~ 1/N).
#
# At N = 37 this realises the committed tasks/sfg2 weights (selective
# w_EI_lat 0.0298 / w_IE_lat 0.1193; uniform u 0.0678 -- i.e. the task's
# rounded 0.030 / 0.120 / 0.068, all at loop gain 6.3).
#
# NOTE the selective structure cannot reach loop gain 6.3 below N = 8: the
# N**2 pooling term is too small, so the solve demands a lateral exceeding
# the self weight (unphysical) and raises (and the laterals are only
# genuinely *weak* -- << self -- for N >> 10).  This is a real finding for
# the invariance study, not a bug: the N = 2 AB/BA default sits at loop gain
# ~0.2, three regimes away from SFG's 6.3 -- so "hold the loop gain fixed
# across paradigms" is a Phase-1 decision with teeth, not a free lunch.
def sfg_config(inh: str = "selective", N: int = 37,
               loop_gain: float = 6.3, lat_ratio: float = 4.0,
               **kw) -> "A1Config":
    """Finalised SFG (recurrent-binding) regime at channel count ``N``.

    ``inh`` selects the inhibition STRUCTURE: ``"selective"`` keeps strong
    per-channel self inhibition with weak lateral; ``"uniform"`` collapses
    all four E<->I weights equal.  Both are solved to the SAME disynaptic
    ``loop_gain`` so the two structures differ only in selectivity, not in
    total inhibitory drive.  ``lat_ratio`` = w_IE_lat / w_EI_lat for the
    selective case (4.0 in the SFG calibration).  Extra ``kw`` override
    A1Config fields (e.g. ``stp_enabled=False``).
    """
    common = dict(N=N, W_max=0.17, W_max_self=0.17, W_decay=2e-3)
    common.update(kw)

    if inh == "uniform":
        # M_EI = M_IE = u*J  ->  loop gain = (u*N)**2.  Solve u = sqrt(g)/N.
        u = float(np.sqrt(loop_gain)) / N
        return A1Config(w_EI_self=u, w_EI_lat=u,
                        w_IE_self=u, w_IE_lat=u, **common)

    if inh == "selective":
        # Keep self at the A1Config biophysical values; lock
        # w_IE_lat = lat_ratio * w_EI_lat; solve the loop-gain quadratic
        # for w_EI_lat = x (derivation: G = M_IE@M_EI = c*J + alpha*beta*I,
        # top eigenvalue c*N + alpha*beta with alpha = a_self - x,
        # beta = b_self - rho*x):
        #   rho*(N-1)**2 x**2 + (rho*a_self + b_self)(N-1) x
        #       + (a_self*b_self - g) = 0
        a_self = A1Config.w_EI_self          # 0.20  (E_i -> I_i)
        b_self = A1Config.w_IE_self          # 0.65  (I_i -> E_i)
        rho = lat_ratio
        A = rho * (N - 1) ** 2
        B = (rho * a_self + b_self) * (N - 1)
        C = a_self * b_self - loop_gain
        x = (-B + float(np.sqrt(B * B - 4 * A * C))) / (2 * A)
        if not 0.0 < x < a_self or rho * x >= b_self:
            raise ValueError(
                f"selective loop gain {loop_gain} unreachable at N={N} with "
                f"physical laterals (solved w_EI_lat={x:.4f} vs self "
                f"{a_self}); the N**2 pooling term is too small -- N must be "
                f"large enough, or use inh='uniform'.")
        return A1Config(w_EI_lat=x, w_IE_lat=rho * x, **common)

    raise ValueError(f"unknown inhibition structure {inh!r}")


# Ready-to-use named presets (zero-arg factories, mirroring INH_PRESETS).
# Pass kwargs through to override N / loop_gain, e.g. SFG_PRESETS["selective"](N=37).
SFG_PRESETS = {
    "selective": lambda **kw: sfg_config("selective", **kw),
    "uniform":   lambda **kw: sfg_config("uniform", **kw),
}
