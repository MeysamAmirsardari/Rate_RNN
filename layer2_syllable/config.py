"""
layer2_syllable.config
======================

Parameters for the second layer.  Layer 1 (``model0``) is untouched; everything
here belongs to the readout population that sits on top of it.

Only three biophysical quantities are introduced:

1. a slow synaptic conductance with a finite rise time (``tau_rise`` /
   ``tau_decay``), which is the layer's temporal memory,
2. a plasticity threshold (``gate_frac``), below which no learning happens,
3. a synaptic decay rate (``lam``), which prunes unused units.

Everything else (competition, the Hebbian rule) is parameter free apart from a
learning rate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class L2Config:
    """Configuration of the layer 2 population."""

    # ---- slow conjunctive trace -------------------------------------
    # A two stage (rise then decay) conductance, i.e. an alpha like kernel.
    # The finite RISE time is load bearing, not cosmetic: with an
    # instantaneous rise, a channel coincides strongly with its own trace and
    # the units learn single tones ("A is on") instead of ordered pairs
    # ("B follows A").  A conductance that is still rising while its own
    # channel fires suppresses that self coincidence by construction.
    # Values are in the range of NMDA receptor and dendritic plateau
    # kinetics (rise tens of ms, decay 100 to 200 ms).
    tau_rise:  float = 0.040       # s
    tau_decay: float = 0.150       # s, must span one chunk

    # ---- what a subunit is allowed to pair ---------------------------
    # Each coincidence subunit reads a fast afferent from one channel and a
    # slow afferent from another.  When True the two must be DIFFERENT
    # channels, so the layer codes transitions between channels rather than
    # sustained activity in one.
    #
    # This is a modelling commitment, so it is worth stating plainly.  The
    # first tone of a chunk has no predecessor, so at its onset the only
    # non zero part of D is the diagonal.  Units are perfectly happy to learn
    # that, and they do: with self pairs allowed the population splits into
    # two transition units plus two redundant "this tone is on" units, and the
    # latter duplicate information layer 1 already represents explicitly.
    # Pairing distinct afferents is also what a coincidence detector is for.
    no_self_pairs: bool = True

    # ---- population --------------------------------------------------
    # Deliberately more units than there are chunk types in the stream.  The
    # number that stay committed is a result, not a setting.
    n_units: int = 8

    # ---- plasticity --------------------------------------------------
    eta: float = 5e-3              # instar (Hebbian) rate, per time step
    lam: float = 1e-4              # synaptic decay, per time step
    gate_frac: float = 0.15        # learn only above this fraction of peak drive
    w_init: float = 0.05           # scale of the small random initial masks

    # Vigilance (ART style).  0 disables it.  If a single unit becomes a blend
    # of two chunk types it will match neither well; requiring the winner to
    # exceed ``rho`` forces a fresh unit to be recruited instead.  Left off by
    # default so that the minimal four rule model is tested first.
    rho: float = 0.0

    # A unit counts as committed when its mask norm exceeds this fraction of
    # the largest mask norm in the population.
    commit_frac: float = 0.20

    seed: int = 0
