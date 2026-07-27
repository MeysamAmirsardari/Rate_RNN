"""
layer2_syllable.stimulus
========================

Token streams for the two layer model.

A *word* is a tuple of channel indices played one after another.  A *stream* is
a random concatenation of words drawn with given probabilities.  The AB versus
BA experiment is the two word case

    words = [(0, 1), (1, 0)],  weights = [0.5, 0.5]

The same function produces the Saffran style stream that will be needed later:
several three token words, equal probability, and ``inter_gap = 0`` so that
word boundaries carry no acoustic cue and have to be inferred from transition
statistics.  Nothing else in the module needs to change for that case.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def build_stream(words: Sequence[Sequence[int]],
                 weights: Sequence[float],
                 n_chunks: int,
                 n_channels: int,
                 dt: float,
                 *,
                 tone_dur: float = 0.050,
                 intra_gap: float = 0.030,
                 inter_gap: float = 0.500,
                 amp: float = 1.0,
                 seed: int = 0,
                 order=None):
    """Build an ``(n_channels, T)`` stimulus from a word sequence.

    Setting ``inter_gap`` equal to ``intra_gap`` makes the stream perfectly
    isochronous, so a word boundary is indistinguishable from a within word
    transition by timing alone and the only cue left is the transition
    statistics.  Words may differ in length; chunk onsets are returned
    explicitly so epoching never assumes otherwise.

    ``order`` optionally supplies the word sequence directly, for designs that
    constrain it (for example forbidding a word from following itself).
    """
    rng = np.random.default_rng(seed)
    if order is None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        order = rng.choice(len(words), size=n_chunks, p=w)
    else:
        order = np.asarray(order, dtype=int)

    n_tone = int(round(tone_dur / dt))
    n_intra = int(round(intra_gap / dt))
    n_inter = int(round(inter_gap / dt))

    def chunk_len(word):
        return len(word) * n_tone + (len(word) - 1) * n_intra + n_inter

    lengths = [chunk_len(words[i]) for i in order]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    T = int(sum(lengths))

    stim = np.zeros((n_channels, T))
    tone_on = []                      # absolute onset sample of every tone
    for k, wi in enumerate(order):
        word = words[wi]
        s0 = starts[k]
        ons = []
        for j, ch in enumerate(word):
            o = s0 + j * (n_tone + n_intra)
            stim[ch, o:o + n_tone] = amp
            ons.append(o)
        tone_on.append(ons)

    return dict(stim=stim, starts=starts, labels=order, lengths=np.array(lengths),
                tone_onsets=tone_on, n_tone=n_tone, n_intra=n_intra,
                n_inter=n_inter, words=list(words), dt=dt)


def chunk_windows(stream: dict, pad_s: float = 0.0):
    """Sample windows ``(start, stop)`` covering the tones of every chunk.

    The window ends at the offset of the last tone plus ``pad_s``, so the long
    silence between chunks is excluded from the response readout.
    """
    dt = stream["dt"]
    pad = int(round(pad_s / dt))
    out = []
    for ons in stream["tone_onsets"]:
        out.append((ons[0], ons[-1] + stream["n_tone"] + pad))
    return out
