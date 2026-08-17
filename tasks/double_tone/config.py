"""
tasks.double_tone.config
========================

Paradigm and model configuration for the double-tone predictive-coding
experiment of Wacongne, Changeux & Dehaene (2012) J Neurosci 32:3665.

Paradigm
--------
Pairs of 50 ms tones, 200 ms onset-to-onset inside a pair.

    learning phase   10 pairs, all AB, 1 s between pairs
    test phase      120 pairs, 70% AB / 10% AA / 10% BB / 10% BA,
                    10-20 s between pairs

Two rare pairs are never consecutive, and a frequent pair immediately
following a rare one is dropped from the analysis -- both as in the paper.
Plasticity is on throughout BOTH phases.

What the paradigm is for
------------------------
The four conditions separate two accounts of the mismatch response.  The
second tone of a rare AA pair is a REPEAT of a tone heard 200 ms earlier,
so synaptic habituation predicts it should be small.  Predictive coding
predicts it should be large, because the network expected B and got A.
AA is therefore the condition where the two accounts point in opposite
directions, and the response has to beat its own depression to come out
large.  Wacongne et al. found it does.

The model's ordering, for the second tone, is

    BA  >  AA ~ BB  >  AB

AB is small because the prediction is fulfilled and the predicted channel
is sitting under the inhibition its own prediction raised.  AA and BB are
depressed by the 200 ms repeat but carry no predictive suppression.  BA is
neither depressed nor predicted, so it is largest.

Model configuration
-------------------
Six constants depart from A1Config.  They are not free fitting: each one
is forced by the 200 ms timescale of this paradigm, which is an order of
magnitude slower than the 30 ms trace and 80 ms interneuron the defaults
assume.

(1) tau_trace 30 -> 150 ms.  To associate A with B the eligibility trace
    has to still be running 200 ms after A.  At 30 ms it is down to
    e^-5 = 0.7% and NOTHING is learned.  The biophysical substrate for an
    eligibility window of this length is the NMDA receptor, whose decay
    time constant is ~100 ms; Wacongne et al. build their own STDP rule on
    exactly that current (their tau_NMDA,decay = 100 ms).

(2) W_max 0.25 -> 0.90.  W here is the AGGREGATE coupling between two
    channel populations, not a unitary EPSP.  The 25-30% of thalamocortical
    strength quoted for W_max = 0.25 (Stratford et al. 1996) is a unitary
    measurement, but recurrent synapses outnumber thalamic ones by roughly
    an order of magnitude -- ~80-90% of excitatory synapses even in layer 4
    are intracortical (Ahmed et al. 1994; Douglas & Martin 2007) -- so the
    aggregate recurrent gain can match thalamic drive without any unitary
    synapse being unusual.

(3) W_norm 20 -> 1.5.  Ten learning pairs is the whole training set, so
    the association has to be near asymptote within ten pairings.
    Convergence time scales as W_norm^2.

(4) W_decay 5e-4 -> 2e-2 (tau = 50 s).  Ten pairings is a weak induction
    protocol and leaves short-term potentiation, which decays over tens of
    seconds, not consolidated LTP (Volianskis & Jensen 2003; Zucker &
    Regehr 2002).  It also keeps the network genuinely plastic through the
    test phase: AB returns every ~21 s on average, comparable to tau.

(5) recurrent_from_trace = True.  This is the one that makes the paradigm
    work at all.  With the default fast recurrence the prediction is
    W @ E, which decays with tau_E = 20 ms and is gone long before the
    second tone lands 200 ms later -- a plain W[B<-A] asserts "B NOW", not
    "B in 200 ms".  That is exactly why Wacongne et al. needed explicit
    delay lines.  Here the recurrent drive is instead W @ tr, the same
    NMDA-timescale trace the learning rule already uses, so the prediction
    outlives the stimulus that raised it.  tau_I stays at the 80 ms
    default: a long interneuron would let each tone's OWN self-inhibition
    survive to the second tone and crush the repeats in AA and BB, which
    would turn the model back into a habituation model.

(6) w_EI_self 0.20 -> 1.0 and w_IE_self 0.65 -> 3.0.  The prediction is
    expressed as INHIBITION, not as phantom firing.  A network whose
    predicted channel fires at full rate before the sound arrives is
    hallucinating; the high E->I gain lets a SUBTLE pre-activation of E_B
    (a few per cent of a real tone response) drive I_B hard, and the high
    I->E gain turns that into strong suppression when the tone lands.

The lateral weights stay weak, and that is load-bearing rather than
cosmetic.  On a rare AA pair the inhibition standing on channel B is at
its maximum, and if inhibition were unselective it would suppress the
second A as well and destroy the effect.  Tone-selective inhibition is a
precondition for this paradigm, not a stylistic choice -- ``uniform_model_
config`` is the row-sum-matched control that shows it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple

from model0 import A1Config


# =====================================================================
#  Model constants (see module docstring for the justification of each)
# =====================================================================
TAU_TRACE    = 150e-3
TAU_I        = 80e-3
W_MAX        = 0.90
W_NORM       = 1.5
W_DECAY      = 2e-2
ETA_LTD      = 0.60    # anti-causal LTD; prunes the reverse association
A_REC        = 0.0     # no excitatory phantom
A_PRED       = 20.0     # prediction delivered to the inhibitory unit
A_CANCEL     = 30.0     # prediction subtracted at the target E unit (global preset)

W_EI_SELF    = 1.00
W_EI_LAT     = 0.05
W_IE_SELF    = 3.00
W_IE_LAT     = 0.20

N_CHANNELS   = 2
TONE_A, TONE_B = 0, 1


def model_config(**overrides) -> A1Config:
    """Selective-inhibition configuration for the double-tone task."""
    cfg = A1Config(
        N=N_CHANNELS,
        multiscale_std=True,
        tau_trace=TAU_TRACE,
        tau_I=TAU_I,
        recurrent_from_trace=True,
        A_rec=A_REC,
        A_pred=A_PRED,
        plastic_self=False,
        W_max=W_MAX,
        W_max_self=W_MAX,
        W_norm=W_NORM,
        W_decay=W_DECAY,
        eta_LTD=ETA_LTD,
        w_EI_self=W_EI_SELF,
        w_EI_lat=W_EI_LAT,
        w_IE_self=W_IE_SELF,
        w_IE_lat=W_IE_LAT,
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def global_model_config(**overrides) -> A1Config:
    """Blanket inhibition, with the prediction as a direct cancellation.

    General inhibition is fully unselective (row-sum matched to the
    selective preset, so every cell receives the same total).  The only
    channel-specific thing left is the predictive projection itself, which
    is subtracted at its target rather than routed through the shared
    interneuron pool.  This is the control showing that GENERAL inhibition
    need not be tone-tuned -- only the prediction does.
    """
    n = N_CHANNELS
    ei = (W_EI_SELF + (n - 1) * W_EI_LAT) / n
    ie = (W_IE_SELF + (n - 1) * W_IE_LAT) / n
    return model_config(w_EI_self=ei, w_EI_lat=ei,
                        w_IE_self=ie, w_IE_lat=ie,
                        A_pred=0.0, A_cancel=A_CANCEL, **overrides)


def uniform_model_config(**overrides) -> A1Config:
    """Row-sum-matched unselective control.

    Each cell receives the same TOTAL inhibition as under the selective
    preset; only the selectivity differs.  This is the control for the
    claim that the effect needs tone-tuned inhibition.
    """
    n = N_CHANNELS
    ei = (W_EI_SELF + (n - 1) * W_EI_LAT) / n
    ie = (W_IE_SELF + (n - 1) * W_IE_LAT) / n
    return model_config(w_EI_self=ei, w_EI_lat=ei,
                        w_IE_self=ie, w_IE_lat=ie, **overrides)


# =====================================================================
#  Paradigm
# =====================================================================
#: The four pair types, as (first tone, second tone) channel indices.
PAIRS: Dict[str, Tuple[int, int]] = {
    "AB": (TONE_A, TONE_B),
    "AA": (TONE_A, TONE_A),
    "BB": (TONE_B, TONE_B),
    "BA": (TONE_B, TONE_A),
}
FREQUENT = "AB"
RARE = ("AA", "BB", "BA")


@dataclass(frozen=True)
class DoubleToneConfig:
    """Wacongne et al. (2012) double-tone paradigm."""

    name: str = "wacongne2012"

    # ---- Tone timing (ms) ----
    tone_dur: int = 50
    soa_within: int = 200          # onset-to-onset inside a pair

    # ---- Learning phase ----
    n_learn: int = 10
    isi_learn: int = 1000          # onset-to-onset between pairs

    # ---- Test phase ----
    n_test: int = 120
    iti_test_min: int = 10_000
    iti_test_max: int = 20_000
    p_frequent: float = 0.70
    p_rare_each: float = 0.10

    # ---- Analysis ----
    response_window: int = 150     # ms after each tone onset
    pre_stim_ms: int = 100
    post_stim_ms: int = 250

    # ---- Exclusions (both from the paper) ----
    no_consecutive_rare: bool = True
    drop_frequent_after_rare: bool = True

    tone_amp: float = 1.0
    seed: int = 7

    # ============ Derived ============
    @property
    def pair_span(self) -> int:
        """Onset of tone 1 to offset of tone 2."""
        return self.soa_within + self.tone_dur

    @property
    def epoch_steps(self) -> int:
        return self.pre_stim_ms + self.pair_span + self.post_stim_ms

    @property
    def tone_onsets_in_epoch(self) -> Tuple[int, int]:
        return (self.pre_stim_ms, self.pre_stim_ms + self.soa_within)

    @property
    def n_each(self) -> Dict[str, int]:
        rare = int(round(self.p_rare_each * self.n_test))
        return {"AB": self.n_test - 3 * rare,
                "AA": rare, "BB": rare, "BA": rare}

    def replace(self, **kw) -> "DoubleToneConfig":
        return dataclasses.replace(self, **kw)


def default(**kw) -> DoubleToneConfig:
    return DoubleToneConfig(**kw)


def short(**kw) -> DoubleToneConfig:
    """Faster: 60 test pairs and a 4-8 s ITI, for iteration only."""
    return DoubleToneConfig(name="short", n_test=60, iti_test_min=4_000,
                            iti_test_max=8_000, **kw)


PRESETS = {"default": default, "short": short}


def get_preset(name: str, **overrides) -> DoubleToneConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. "
                         f"Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name](**overrides)
