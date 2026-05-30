"""
live_demo.config
================

Configuration for the **real-time microphone demo** of the model0 A1
rate RNN -- an "artificial auditory cortex listening through a cochlea".

Signal chain
------------
    microphone  ->  cochleo-thalamic mel front end (AGC)  ->  A1 rate RNN
                ->  scrolling display (input + cortical response)

The front end is the same log-mel transform used offline in
``tasks.auditory`` (180 mel bands, ``fmin..fmax``, ``power=2`` mel,
log/dB compression), with one difference required by streaming: the dB
reference is no longer the precomputed peak over a fixed syllable set but
a **running automatic gain control** (AGC) -- a slow peak tracker with
fast attack / slow release.  This is biologically the right move: the
auditory periphery and brainstem apply multi-stage automatic gain
control, so an adaptive reference is more faithful than a fixed one
(Robinson & McAlpine 2009; Rabinowitz et al. 2011, Neuron 70:1178 --
contrast gain control in A1).

One mel frame == one model integration step
-------------------------------------------
``hop_ms = 1`` and ``A1Config.dt = 1 ms`` so each spectrogram frame maps
1:1 onto one Euler step of the RNN (identical to the offline task).  At
``sr = 16 kHz`` the hop is ``16`` samples, exact.

Nothing here changes the model: ``to_a1_config`` simply instantiates the
committed ``model0`` inhibition preset at ``N = n_channels``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class LiveConfig:
    """Parameters for the live microphone -> A1 demo."""

    name: str = "default"

    # ---- spectral channels (== number of mel bands == model0 N) ----
    n_channels: int = 180

    # ---- audio capture / mel front end ----
    sr: int = 16000                 # capture + resample rate (Hz)
    n_fft: int = 512                # STFT window (samples) -> 32 ms @16 kHz
    hop_ms: float = 1.0             # frame hop (ms) -> 1 frame == 1 model step
    fmin: float = 50.0              # lowest mel edge (Hz)
    fmax: float = 8000.0            # highest mel edge (Hz)
    blocksize: int = 512            # samples per audio callback (~32 ms)
    in_channels: int = 1            # mono

    # ---- automatic gain control (the streaming dB reference) ----
    top_db: float = 60.0            # dynamic range below the running peak
    agc_attack: float = 0.50        # fraction toward a louder new peak / frame
    agc_release: float = 0.9990     # per-frame decay of the running peak
    ref_floor: float = 1.0          # absolute floor on the reference power
    drive_gain: float = 1.0         # scales the [0,1] drive (== syll_amp)

    # ---- adaptive noise gate (squelch) ----
    # A minimum-statistics noise-floor tracker (Martin 2001, IEEE TSAP
    # 9:504): the floor drops instantly to any new spectral minimum and
    # rises slowly, so loud speech never inflates it.  Frames whose peak
    # mel power sits less than ``gate_factor`` above the tracked floor are
    # treated as silence and emit zero drive -- otherwise the 60 dB log
    # compression would amplify the resting room noise into a full-channel
    # hallucination.  This is the cochlear-nucleus squelch / threshold.
    nf_rise: float = 1.0006         # per-frame rise of the noise-floor estimate
    gate_factor: float = 4.0        # peak must exceed gate_factor * floor (~6 dB)

    # ---- model ----
    inhibition: str = "selective"   # 'selective' | 'uniform' (INH_PRESETS)
    multiscale_std: bool = True     # multi-timescale TC depression (adapts live)
    learn: bool = True              # Hebbian plasticity on E->E during the demo

    # ---- display ----
    history_s: float = 4.0          # seconds of scrolling history shown
    target_fps: int = 60            # display refresh target
    input_cmap: str = "magma"       # thalamic-input heatmap colormap
    cortex_cmap: str = "viridis"    # cortical-response heatmap colormap

    # ============ derived ============
    @property
    def dt(self) -> float:
        """Model integration step (s) -- locked to one mel frame."""
        return self.hop_ms / 1000.0

    @property
    def hop_length(self) -> int:
        """STFT hop in samples (one frame per ``hop_ms`` of audio)."""
        return max(1, int(round(self.sr * self.hop_ms / 1000.0)))

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def history_frames(self) -> int:
        """Number of 1-ms frames in the scrolling display window."""
        return int(round(self.history_s * 1000.0 / self.hop_ms))

    @property
    def frame_ms(self) -> float:
        return self.hop_ms

    def replace(self, **kw) -> "LiveConfig":
        return dataclasses.replace(self, **kw)

    def to_a1_config(self):
        """Instantiate the committed model0 inhibition preset at N=n_channels.

        Imported lazily so the config module stays import-light (and usable
        for unit tests that never touch the model)."""
        from model0 import INH_PRESETS
        if self.inhibition not in INH_PRESETS:
            avail = ", ".join(sorted(INH_PRESETS))
            raise ValueError(
                f"Unknown inhibition preset '{self.inhibition}'. "
                f"Available: {avail}")
        return INH_PRESETS[self.inhibition](
            N=self.n_channels, dt=self.dt, multiscale_std=self.multiscale_std)

    def __post_init__(self):
        if self.n_channels < 2:
            raise ValueError(f"n_channels must be >= 2; got {self.n_channels}")
        if self.fmax <= self.fmin:
            raise ValueError(
                f"fmax ({self.fmax}) must exceed fmin ({self.fmin})")
        if self.fmax > self.sr / 2:
            raise ValueError(
                f"fmax ({self.fmax}) exceeds Nyquist (sr/2 = {self.sr / 2})")
        if self.inhibition not in ("selective", "uniform"):
            raise ValueError(
                f"inhibition must be 'selective' or 'uniform'; "
                f"got {self.inhibition!r}")


# =====================================================================
#  Presets
# =====================================================================
def default(**kw) -> LiveConfig:
    """Selective inhibition, plasticity + multi-timescale STD on."""
    return LiveConfig(**kw)


def uniform(**kw) -> LiveConfig:
    """Control: uniform (non-selective) inhibition."""
    return LiveConfig(name="uniform", inhibition="uniform", **kw)


def frozen(**kw) -> LiveConfig:
    """Plasticity off -- pure short-term dynamics (no Hebbian drift)."""
    return LiveConfig(name="frozen", learn=False, **kw)


PRESETS = {
    "default": default,
    "uniform": uniform,
    "frozen":  frozen,
}


def get_preset(name: str, **overrides) -> LiveConfig:
    if name not in PRESETS:
        avail = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available: {avail}")
    return PRESETS[name](**overrides)
