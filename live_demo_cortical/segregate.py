"""
live_demo_cortical.segregate
============================

General, continuous, **gap-insensitive** unsupervised stream segregation by
clustering per-event **directed-coincidence signatures**.

The idea (works for ANY standard/deviant pair, not just dev-vs-standard):

1.  **Segment** the continuous stream into events with an activity-gap detector.
    Intra-token gaps are bridged (the front-end window overlap keeps activity
    up across them); only the longer inter-event gaps split events.  The result
    does not depend on the inter-event gap length (it just has to be longer than
    the intra-token gaps) -- gap-insensitive.

2.  **Signature.** For each event, the directed coincidence
    ``D_e[i,j] = <x_i(t)·tr_j(t)>`` over the event (the model's Hebbian post-rate
    x pre-trace operator; the trace is *reset per event*, so the signature is
    decoupled from the gap and from the previous event).  ``D_e = S_e + Δ_e``:
    the **symmetric** part encodes which channels co-occur (separates *feature*
    changes, AB vs AC), the **antisymmetric** part encodes temporal order
    (separates *order* changes, AB vs BA, ABC vs ACB, CAB, CBA).

3.  **Center + cluster.** Subtract the running **mean** signature (the shared
    structure -- which, at 50-50, is exactly the part that cancels and carries
    no class information) and cluster the residuals (PCA + k-means).  The class
    contrast -- whether it lives in ``S`` or in ``Δ`` -- is the dominant
    *variance* direction across events, so one clusterer handles every case.

The directed operator runs on the front-end **drive** (the cochleo-thalamic
representation): it is crisp and gap-insensitive.  The recurrent cortical ``E``
can be substituted (``signal='E'`` in :func:`from_audio`) but its recurrent /
adaptation dynamics bleed across short gaps and blur subtle within-event order,
so it is not the default.  This is the piece intended to grow toward speech
(syllable-rate events, k talkers) later.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------
def directed_signature(X_seg: np.ndarray, dt: float, tau: float = 0.10):
    """Per-event directed coincidence ``D[i,j]=<X_i(t)·tr_j(t)>`` (flattened).

    ``tr`` is a causal eligibility trace of ``X`` (time constant ``tau``),
    initialised to zero at the start of the event, so the signature reflects
    only within-event structure."""
    N, L = X_seg.shape
    if L < 2:
        return np.zeros(N * N)
    c = dt / tau
    tr = lfilter([0.0, c], [1.0, -(1.0 - c)], X_seg, axis=1)   # tr[m]=(1-c)tr[m-1]+c X[m-1]
    return ((X_seg @ tr.T) / L).ravel()


def _kmeans(Z, k, iters=60, restarts=10, seed=0):
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    best, best_cost = None, np.inf
    for _ in range(restarts):
        c = Z[rng.choice(n, k, replace=False)].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(iters):
            lab = ((Z[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
            nc = np.array([Z[lab == j].mean(0) if (lab == j).any() else c[j]
                           for j in range(k)])
            if np.allclose(nc, c):
                break
            c = nc
        cost = float(((Z - c[lab]) ** 2).sum())
        if cost < best_cost:
            best_cost, best = cost, lab.copy()
    return best


# ---------------------------------------------------------------------
class EventSegregator:
    """Streaming segmenter + per-event directed-coincidence signatures, with an
    unsupervised k-stream clusterer over the centered signatures."""

    def __init__(self, n_channels: int, dt: float, n_streams: int = 2,
                 sig_tau: float = 0.10, merge_gap_s: float = 0.07,
                 on_frac: float = 0.02, min_event_s: float = 0.03,
                 n_pca: int | None = None):
        self.N = n_channels
        self.dt = dt
        self.k = n_streams
        self.sig_tau = sig_tau
        self.merge = max(1, int(round(merge_gap_s / dt)))
        self.on_frac = on_frac
        self.min_ev = max(1, int(round(min_event_s / dt)))
        self.n_pca = n_pca
        self.reset()

    def reset(self):
        self.onsets, self.offsets, self.sigs = [], [], []
        self.labels = None
        self._buf = []
        self._in = False
        self._start = 0
        self._last_active = -1
        self._since = 0
        self._t = 0
        self._peak = 1e-9

    # ---- streaming ----
    def push(self, X: np.ndarray):
        """Feed one block ``X`` (n_channels, k) of the representation (drive)."""
        if X.shape[1] == 0:
            return
        p = X.sum(0)
        self._peak = max(self._peak, float(p.max()))
        thr = self.on_frac * self._peak
        for j in range(X.shape[1]):
            on = p[j] > thr
            if on:
                if not self._in:
                    self._in = True
                    self._start = self._t
                    self._buf = []
                    self._last_active = -1
                self._buf.append(X[:, j])
                self._last_active = len(self._buf) - 1
                self._since = 0
            elif self._in:
                self._since += 1
                self._buf.append(X[:, j])
                if self._since > self.merge:
                    self._finish()
            self._t += 1

    def finalize(self):
        if self._in:
            self._finish()

    def _finish(self):
        buf = self._buf
        last = self._last_active
        self._in, self._buf, self._since, self._last_active = False, [], 0, -1
        if last + 1 < self.min_ev:
            return
        seg = np.asarray(buf[:last + 1]).T               # drop the trailing gap
        self.onsets.append(self._start * self.dt)
        self.offsets.append((self._start + last + 1) * self.dt)
        self.sigs.append(directed_signature(seg, self.dt, self.sig_tau))

    # ---- clustering (call anytime; cheap, re-clusters all events) ----
    def cluster(self, n_streams: int | None = None):
        k = n_streams or self.k
        S = np.asarray(self.sigs)
        if S.shape[0] < max(2, k):
            return None
        Sc = S - S.mean(0)                               # remove shared / 50-50-cancelling part
        U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
        npc = self.n_pca or min(2 * k, Vt.shape[0], S.shape[0] - 1)
        Z = Sc @ Vt[:npc].T
        self.labels = _kmeans(Z, k)
        return self.labels


# ---------------------------------------------------------------------
def from_audio(audio, cfg, n_streams=2, signal="drive", sig_tau=0.10, **kw):
    """Run the front end on ``audio`` and segregate.  ``signal='drive'`` (the
    cochleo-thalamic input, gap-insensitive) or ``'E'`` (the cortical RNN
    activations, via the engine).  Returns the fitted :class:`EventSegregator`."""
    from .audio import SpectroFrontEnd
    fe = SpectroFrontEnd(cfg)
    seg = EventSegregator(cfg.n_channels, cfg.dt, n_streams=n_streams,
                          sig_tau=sig_tau, **kw)
    eng = None
    if signal == "E":
        from .engine import LiveEngine
        eng = LiveEngine(cfg.to_a1_config(), learn=False, seed=0)
    bs = cfg.blocksize
    for lo in range(0, audio.size, bs):
        drive, _ = fe.push(audio[lo:lo + bs])
        if drive.shape[1] == 0:
            continue
        X = eng.step_block(drive)["E"] if eng is not None else drive
        seg.push(X.astype(np.float64))
    seg.finalize()
    return seg


# ---------------------------------------------------------------------
#  Demonstration / self-test: segregate the directional paradigms at 50-50
# ---------------------------------------------------------------------
def main(argv=None):
    import argparse
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from live_demo_cortical.config import get_preset
    from live_demo_cortical.audio import synth_sequence

    PARADIGMS = {
        "abba":    ((0, 1), (1, 0)),
        "abcacb":  ((0, 1, 2), (0, 2, 1)),
        "abc_cab": ((0, 1, 2), (2, 0, 1)),
        "abc_cba": ((0, 1, 2), (2, 1, 0)),
        "ab_ac":   ((0, 1), (0, 2)),
        "ac_bc":   ((0, 2), (1, 2)),
    }
    ap = argparse.ArgumentParser(
        description="Unsupervised segregation of the directional paradigms (50-50).")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--gaps", type=float, nargs="+", default=[150.0, 300.0, 500.0])
    ap.add_argument("--signal", choices=["drive", "E"], default="drive")
    args = ap.parse_args(argv)

    cfg = get_preset("directional")
    fr = (cfg.fmin * 1.6, cfg.fmin * 3.0, cfg.fmax * 0.7)

    def accuracy(seg, info):
        gt_t = np.array([t for _, t in info["events"]])
        gt_l = np.array([int(d) for d, _ in info["events"]])
        lab = seg.cluster()
        if lab is None:
            return 0.0
        true = np.array([gt_l[np.argmin(np.abs(gt_t - o))] for o in seg.onsets])
        return max(np.mean(lab == true), np.mean(lab != true))

    print(f"[ segregate self-test ]  signal={args.signal}  "
          f"(2-stream clustering, 50-50, no labels)")
    header = "paradigm  std/dev      " + "  ".join(f"gap={int(g)}ms" for g in args.gaps)
    print("  " + header)
    for name, (std, dev) in PARADIGMS.items():
        accs = []
        for g in args.gaps:
            y, _, info = synth_sequence(cfg.sr, fr, std, dev, p_dev=0.5,
                                        inter_ms=g, seconds=args.seconds, seed=0)
            seg = from_audio(y, cfg, n_streams=2, signal=args.signal)
            accs.append(accuracy(seg, info))
        tag = "".join("ABC"[i] for i in std) + "/" + "".join("ABC"[i] for i in dev)
        print("  %-9s %-10s  %s" % (
            name, tag, "  ".join("  %4.0f%%" % (100 * a) for a in accs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
