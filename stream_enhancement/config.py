"""
stream_enhancement.config
=========================

Separate configuration for the stream enhancement test.  This test uses layer 1
(``model0``) only; there is no second layer anywhere in it.

The hypothesis
--------------
A fast coherent stream sweeping through the tonotopic axis should be enhanced
relative to background noise, because each channel pre-activates the next one
through the learned recurrent weights, while the slow tone selective inhibition
cannot follow a stream this fast and therefore fails to cancel the gain.

The settings below are chosen to make that effect as large as the model allows,
which means they depart from the values used elsewhere in the project.  Every
departure is noted.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamConfig:
    """Everything the stream enhancement test needs."""

    # ---- tonotopic axis ------------------------------------------------
    # The stream and the noise are put on disjoint channels so that "signal"
    # and "noise" can be measured without ambiguity.  With them interleaved,
    # the pre-activation of the *next* stream channel gets counted as noise and
    # the measurement is meaningless.
    n_channels: int = 32
    n_stream: int = 16            # channels 0 .. n_stream-1 carry the stream

    # ---- the stream ----------------------------------------------------
    # A smooth sinusoidal sweep through channel space, one token at a time.
    token_ms: float = 20.0        # fast, as hypothesised
    period_tokens: int = 12       # tokens per cycle of the sweep
    sweep_centre: float = 7.5
    sweep_amp: float = 7.0

    # ---- the background ------------------------------------------------
    noise_amp: float = 0.55       # relative to a stream token at 1.0
    n_noise_per_token: int = 2    # how many noise channels are lit at a time

    # ---- layer 1 --------------------------------------------------------
    # tau_I is left at the model's default.  It is deliberately slow relative
    # to a 20 ms token, which is the whole point: inhibition cannot track the
    # stream but does accumulate enough to hold the background down.
    tau_I: float = 0.080

    # Recurrent weights.  Two departures from the rest of the project:
    #   plastic_self = False   self weights would add loop gain without adding
    #                          any prediction, and they destabilise the network
    #                          before the useful range is reached.
    #   W_max raised           the default 0.25 is calibrated so intracortical
    #                          EPSPs stay near 25 percent of thalamocortical.
    #                          The effect is real there but small; raising the
    #                          ceiling is what "maximise the effect" requires.
    W_max: float = 0.80
    W_norm: float = 8.0
    plastic_self: bool = False

    # Learning settles near 0.21 on this stimulus, well short of what the
    # effect needs.  w_scale multiplies the learned matrix so the mechanism can
    # be shown at full strength.  It is an IMPOSED amplification, not something
    # the model learns, and the figures label it as such.  Set it to 1.0 to see
    # the honest as-learned result.
    #
    # Before scaling, weights below w_threshold of the peak are dropped.  The
    # learned matrix carries a lot of small off-transition weight, and scaling
    # that up adds loop gain without adding any prediction, which sends the
    # network into runaway excitation well before the useful range.  Keeping
    # only the real transitions is what makes the strong regime reachable.
    w_threshold: float = 0.40
    w_scale: float = 3.5          # 4.0 diverges; see the characterisation figure

    # ---- protocol -------------------------------------------------------
    train_tokens: int = 1500      # clean stream, learning on
    test_tokens: int = 150        # stream in noise, weights frozen
    seed: int = 0
