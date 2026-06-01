"""tasks.sfg2 -- faithful Stochastic Figure-Ground paradigm for model0.

Weight readout (W_FF / W_GG / W_FG) plus a per-channel excitation/
inhibition current decomposition (thalamic, recurrent, inhibitory, net)
for figure vs. ground channels.  Stimulus generator lives in
``stimulus.py``; the run + analysis + plotting driver in ``sfg2.py``.
"""

from .stimulus import build_session, build_presentation, FIG_CHANNELS_10, N_FIG_OPTS

__all__ = ["build_session", "build_presentation", "FIG_CHANNELS_10", "N_FIG_OPTS"]
