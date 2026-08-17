"""Model 0: tone-selective inhibition (one E + one I per tonotopic channel).

Replaces the global PV-style blanket inhibition + divisive adaptation of
``model/`` with selective SST-style interneurons, one per channel.
Predictive suppression now arises from the *inhibitory* arm: pre-activation
of E_B during tone A drives I_B, and because tau_I > tau_E, residual I_B
suppresses the response of E_B when tone B itself arrives.
"""

from .config import (A1Config, selective_inh, uniform_inh, INH_PRESETS,
                     sfg_config, SFG_PRESETS, inhibitory_loop_gain,
                     shared_config, SHARED_W_MAX, SHARED_LOOP_GAIN_CAP,
                     ROVING_W_DECAY)
from .model import simulate

__all__ = ["A1Config", "selective_inh", "uniform_inh", "INH_PRESETS",
           "sfg_config", "SFG_PRESETS", "inhibitory_loop_gain",
           "shared_config", "SHARED_W_MAX", "SHARED_LOOP_GAIN_CAP",
           "ROVING_W_DECAY", "simulate"]
