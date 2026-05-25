"""Model 0: tone-selective inhibition (one E + one I per tonotopic channel).

Replaces the global PV-style blanket inhibition + divisive adaptation of
``model/`` with selective SST-style interneurons, one per channel.
Predictive suppression now arises from the *inhibitory* arm: pre-activation
of E_B during tone A drives I_B, and because tau_I > tau_E, residual I_B
suppresses the response of E_B when tone B itself arrives.
"""

from model0.config import A1Config
from model0.model import simulate

__all__ = ["A1Config", "simulate"]
