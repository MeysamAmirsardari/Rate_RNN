"""Independent block split used for Figure 2 contact selection and ERP tests."""

from __future__ import annotations

import numpy as np


def context_stratified_discovery_mask(
    stimuli: np.ndarray,
    contexts: np.ndarray,
) -> np.ndarray:
    """Return a deterministic, approximately half-sized discovery split.

    Intact roving blocks are stratified by the ordered previous/current tone
    context.  Alternating blocks within every stratum enter the discovery set;
    the complementary blocks are reserved for ERP estimation and inference.
    """

    stimuli = np.asarray(stimuli)
    contexts = np.asarray(contexts)
    if stimuli.ndim != 1 or contexts.ndim != 1 or stimuli.shape != contexts.shape:
        raise ValueError("stimuli and contexts must be aligned one-dimensional arrays")

    discovery = np.zeros(stimuli.size, dtype=bool)
    pairs = sorted(set(zip(contexts.tolist(), stimuli.tolist())))
    for context, stimulus in pairs:
        indices = np.flatnonzero((contexts == context) & (stimuli == stimulus))
        if indices.size < 2:
            raise ValueError(
                f"context {context!r}->{stimulus!r} has fewer than two blocks"
            )
        discovery[indices[::2]] = True

    if not np.any(discovery) or np.all(discovery):
        raise AssertionError("discovery/inference split is degenerate")
    return discovery


__all__ = ["context_stratified_discovery_mask"]
