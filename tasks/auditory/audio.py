"""
tasks.auditory.audio
=====================

Speech -> mel-spectrogram -> model-input encoder for the auditory roving
task.  Turns each recorded syllable WAV into a per-channel thalamo-
cortical (TC) drive on ``cfg.n_channels`` mel bands, sampled at one frame
per model step (``hop_ms = 1 ms``).

Pipeline (per syllable)
-----------------------
1. Load the WAV, resample to ``cfg.sr``, force mono.
2. Power mel-spectrogram (``n_mels = cfg.n_channels``, ``fmin..fmax``).
3. Log-compress against a reference, with a ``top_db`` dynamic-range
   floor, and map onto [0, 1]:

       drive = clip((10*log10(S / ref) + top_db) / top_db, 0, 1) * syll_amp

   The reference is **shared across all five syllables** (the global
   spectrogram peak) unless ``cfg.per_syllable_norm`` is set, so natural
   loudness differences between syllables are preserved -- this is what
   distinguishes a deviant that is intrinsically louder/quieter.
4. Fit the frame count to exactly ``cfg.syll_dur`` (trim or zero-pad).

The log compression is the standard auditory-nerve / cochlear
nonlinearity; the mel axis is the tonotopic (log-frequency) map.  The
result is the artificial thalamic input the cortical model receives.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import librosa
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "tasks.auditory requires `librosa` (and `soundfile`/`soxr`). "
        "Install with `pip install librosa soundfile soxr`.") from exc

from .config import AuditoryConfig

_EPS = 1e-10


def _fit_frames(spec: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or zero-pad a (n_channels, T) spectrogram to exactly n_frames."""
    n, t = spec.shape
    if t == n_frames:
        return spec
    if t > n_frames:
        return spec[:, :n_frames]
    out = np.zeros((n, n_frames), dtype=spec.dtype)
    out[:, :t] = spec
    return out


def power_mel(y: np.ndarray, cfg: AuditoryConfig) -> np.ndarray:
    """Power mel-spectrogram of a waveform: shape (n_channels, n_frames)."""
    return librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_channels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0,
        center=True)


class MelEncoder:
    """Loads syllable WAVs and exposes their mel-spectrograms and the
    normalised TC drives used as model input.

    Attributes
    ----------
    power : dict[str, (n_channels, T_raw)]
        Raw power mel-spectrograms (before frame fitting), for plotting.
    db : dict[str, (n_channels, T_raw)]
        Log-mel spectrograms in dB relative to the shared reference.
    drive : dict[str, (n_channels, syll_dur)]
        Normalised [0, syll_amp] per-channel drive fed to the model.
    """

    def __init__(self, cfg: AuditoryConfig):
        self.cfg = cfg
        self.power: Dict[str, np.ndarray] = {}
        for s in cfg.syllables:
            y, _ = librosa.load(str(cfg.wav_path(s)), sr=cfg.sr, mono=True)
            self.power[s] = power_mel(y, cfg)

        # Shared (global) reference = peak power over all syllables, so the
        # dB mapping is comparable across the set.
        global_ref = max(float(P.max()) for P in self.power.values())
        global_ref = max(global_ref, _EPS)

        self.db: Dict[str, np.ndarray] = {}
        self.drive: Dict[str, np.ndarray] = {}
        for s, P in self.power.items():
            ref = max(float(P.max()), _EPS) if cfg.per_syllable_norm else global_ref
            db = 10.0 * np.log10(np.maximum(P, _EPS) / ref)     # <= 0 dB
            db = np.maximum(db, -cfg.top_db)                    # global floor
            self.db[s] = db
            drive = (db + cfg.top_db) / cfg.top_db              # [0, 1]
            drive = np.clip(drive, 0.0, 1.0) * cfg.syll_amp
            self.drive[s] = _fit_frames(drive, cfg.syll_dur)

    # ---- channel selection (for population read-outs) ----
    def active_channels(self, syllable: str) -> np.ndarray:
        """Mel channels this syllable drives above ``active_thresh`` * peak."""
        d = self.drive[syllable]
        peak = float(d.max())
        if peak <= 0.0:
            return np.empty(0, dtype=int)
        chan_peak = d.max(axis=1)
        return np.where(chan_peak >= self.cfg.active_thresh * peak)[0]

    def word_channels(self, word: str) -> np.ndarray:
        """Union of active channels across all syllables in ``word``."""
        chans = set()
        for s in word:
            chans.update(self.active_channels(s).tolist())
        return np.array(sorted(chans), dtype=int)

    def mel_frequencies(self) -> np.ndarray:
        """Centre frequency (Hz) of each mel channel, for axis labels."""
        return librosa.mel_frequencies(
            n_mels=self.cfg.n_channels, fmin=self.cfg.fmin, fmax=self.cfg.fmax)

    def n_empty_channels(self) -> int:
        """Mel channels that are silent for every syllable (degenerate
        filters / out-of-band) -- useful as a sanity check at high n_mels."""
        active = np.zeros(self.cfg.n_channels, dtype=bool)
        for s in self.cfg.syllables:
            active[self.active_channels(s)] = True
        return int((~active).sum())
