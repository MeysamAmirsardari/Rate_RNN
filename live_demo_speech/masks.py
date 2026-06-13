"""
live_demo_speech.masks
========================

Real-time stream separation from the coincidence matrix -- the segregation
read-out of the temporal-coherence framework (Krishnan, Elhilali & Shamma,
PLoS Comp Biol 2014).  Their nPCA factors the channel x channel coincidence
matrix into K non-negative masks (one per stream).

We implement the robust, fast version: **normalized spectral clustering of the
coincidence matrix**.  Two temporally coherent sources make C ~ a aᵀ + b bᵀ;
the top-K eigenvectors of C span the source subspace, and clustering the
channels in that (unit-normalised) eigen-embedding recovers the streams even
when the sources share a common mode (which trips up a bare SNMF / a single
eigenvector split).  Each channel's membership is weighted by how strongly it
participates in the coherent subspace, so the incoherent background -- which
lives near the origin -- gets ~zero mask.

Fast: one eigendecomposition of the (small) N x N matrix plus a few k-means
iterations on the active channels, warm-orderable across frames.  Microseconds
for N ~ 60-180, run at ~10 Hz off the live coincidence matrix.
"""
from __future__ import annotations

import numpy as np


class StreamSeparator:
    """Factor the coincidence matrix C into K soft non-negative stream masks
    by normalized spectral clustering of its top-K eigenvectors."""

    def __init__(self, n_channels: int, n_streams: int = 2,
                 iters: int = 15, seed: int = 0):
        self.K = n_streams
        self.iters = iters
        self.H = np.zeros((n_channels, n_streams))

    def update(self, C: np.ndarray) -> np.ndarray:
        """``C`` (N,N) symmetric coincidence -> ``H`` (N,K) non-negative masks,
        by degree-normalized (Ng-Jordan-Weiss) spectral clustering.  Degree
        normalisation suppresses the shared common mode that otherwise dominates
        the top eigenvectors and hides the stream-vs-stream contrast."""
        N, K = C.shape[0], self.K
        C = np.maximum(C, 0.0)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        deg = C.sum(1)
        H = np.zeros((N, K))
        act = np.where(deg > 0.05 * deg.max() + 1e-12)[0]
        if act.size >= K:
            Cs = C[np.ix_(act, act)]
            Dm = 1.0 / np.sqrt(Cs.sum(1) + 1e-9)
            L = (Dm[:, None] * Cs) * Dm[None, :]        # D^-1/2 C D^-1/2
            w, V = np.linalg.eigh(L)
            Y = V[:, np.argsort(w)[::-1][:K]]
            Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)   # row-normalise
            centers = [Y[0]]                            # k-means++ seed
            for _ in range(K - 1):
                d = np.min([((Y - c) ** 2).sum(1) for c in centers], axis=0)
                centers.append(Y[np.argmax(d)])
            centers = np.array(centers)
            for _ in range(self.iters):
                lab = ((Y[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
                for k in range(K):
                    m = lab == k
                    if m.any():
                        c = Y[m].mean(0)
                        centers[k] = c / (np.linalg.norm(c) + 1e-9)
            d = ((Y[:, None, :] - centers[None]) ** 2).sum(2)
            soft = np.exp(-5.0 * d)
            soft /= soft.sum(1, keepdims=True) + 1e-9   # soft membership
            H[act] = soft * deg[act, None]              # weight by participation
        if K == 2 and self.H.any():                     # stable column order
            keep = H[:, 0] @ self.H[:, 0] + H[:, 1] @ self.H[:, 1]
            swap = H[:, 0] @ self.H[:, 1] + H[:, 1] @ self.H[:, 0]
            if swap > keep:
                H = H[:, ::-1].copy()
        self.H = H
        return H

    def masks(self) -> np.ndarray:
        """Per-column max-normalised masks (channel membership in [0,1])."""
        return self.H / (self.H.max(0, keepdims=True) + 1e-9)

    def reset(self):
        self.H[:] = 0.0
