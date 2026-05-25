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
    w_EI_self: float = 0.6         # E_i -> I_i (strong)
    w_EI_lat:  float = 0.05        # E_i -> I_{j!=i} (weak)
    w_IE_self: float = 4.0         # I_i -> E_i (strong self-inhibition)
    w_IE_lat:  float = 0.20        # I_i -> E_{j!=i} (weak lateral)

    # ---- Short-term depression on TC input (Tsodyks-Markram) ----
    # Identical to model/.  Wehr & Zador 2005; Cruikshank+2007; Markram+1998.
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
    eta_LTD:     float = 0.7
    W_decay:     float = 5e-4
    # Recurrent unitary strength bounded well below TC strength.  In
    # cortex, intracortical EPSPs are ~25-30% of thalamocortical EPSPs
    # (Stratford+1996; Cruikshank+2007).  W_max=0.25 sits at the lower
    # end of that range, which (together with w_IE_self=4.0) keeps the
    # pre-activation of E_B during tone A subtle (E_B^pre/E_A < 13%)
    # while preserving the ~5% MMN polarity that's empirically observed
    # in A1 (Ulanovsky+2003; Nieto-Diego & Malmierca 2016).
    W_max:       float = 0.25
    W_max_self:  float = 0.17
    # W_norm calibrates the rate units in the Hebbian product
    # (E/W_norm)*(tr/W_norm).  In model0 E peaks at ~13 (vs ~4 in the
    # divisive model/), so the scale factor must rise accordingly.
    # W_norm=20 makes per-trial dW ~16x smaller than model/'s setting,
    # so W saturates gradually over ~130 trials rather than ~10 -- in
    # line with cortical plasticity timescales.
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
    bounded_plasticity: bool = False
    eta_het:            float = 0.01    # heterosynaptic LTD rate

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
        w_IE_self=0.96, w_IE_lat=0.96,
    )
    defaults.update(kw)
    return A1Config(**defaults)


INH_PRESETS = {
    "selective": selective_inh,
    "uniform":   uniform_inh,
}
