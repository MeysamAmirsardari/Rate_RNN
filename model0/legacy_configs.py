"""model0.legacy_configs
=======================
Snapshot of the per-task A1 configs as they were BEFORE the single
``shared_config`` (model0/config.py) was adopted -- the repository state at
git tag ``pre-shared-config``.

Kept so any paradigm can be returned to its original parameter set for
comparison or debugging.  To revert ONE task, swap its ``shared_config(...)``
call for the matching factory here; to revert EVERYTHING, check out the tag:

    git checkout pre-shared-config -- tasks model0

Each factory reproduces exactly what that task's ``main()`` built.  The
``live_demo`` package is intentionally absent: it was never changed and keeps
its own ``LiveConfig`` -> ``INH_PRESETS`` path untouched.

Summary of what every task used (relative to A1Config defaults):
  - mechanism constants (tau_E/I, tau_trace, eta_LTP/LTD, U, A_TC, tau_D,
    W_norm, bounded plasticity) were SHARED by all -- never overridden.
  - only N (structural), the multiscale_std enable, and the two overrides
    below differed.
"""
from __future__ import annotations

import dataclasses as _dc

from .config import A1Config, INH_PRESETS


def ab_ba(inh: str) -> A1Config:
    """AB-BA: the selective preset got a strong-inhibition + fast-learning
    override (w_IE_self 0.65->3.0, w_EI_self 0.20->0.40, W_norm 20->4); the
    uniform preset ran on the plain factory.  W_norm=4 only sped the
    (W_max-W)-gated approach 25x -- the asymptote is unchanged."""
    cfg = INH_PRESETS[inh]()
    if inh == "selective":
        cfg = _dc.replace(cfg, w_IE_self=3.0, w_EI_self=0.40, W_norm=4.0)
    return cfg


def oddball(inh: str, N: int = 12) -> A1Config:
    """Streaming oddball / SSA (the ABC-ABX paradigm): plain preset at N=12,
    single-timescale STD."""
    return INH_PRESETS[inh](N=N)


def local_global(inh: str) -> A1Config:
    """Local-global: plain preset at N=2, single-timescale STD."""
    return INH_PRESETS[inh]()


def roving(inh: str, N: int) -> A1Config:
    """Roving oddball: plain preset, multi-timescale STD on (slow 5 s
    component survives the ITI and integrates across reps)."""
    return INH_PRESETS[inh](N=N, multiscale_std=True)


def syllable(inh: str, N: int = 40) -> A1Config:
    """Syllable-figure roving: plain preset at N=40, multiscale STD on."""
    return INH_PRESETS[inh](N=N, multiscale_std=True)


def auditory(inh: str, N: int) -> A1Config:
    """Speech-mel-spectrogram roving: plain preset, multiscale STD on."""
    return INH_PRESETS[inh](N=N, multiscale_std=True)


def sfg_old(inh: str, N: int = 28) -> A1Config:
    """Original SFG (tasks/sfg_model0): plain preset at N=28, defaults
    otherwise (W_max=0.25, W_decay=5e-4)."""
    return INH_PRESETS[inh](N=N)


def sfg2(inh: str) -> A1Config:
    """SFG2 (tasks/sfg2): bespoke N=37 recalibration -- W_max=0.17,
    W_decay=2e-3, inhibition rescaled to disynaptic loop gain 6.3."""
    common = dict(N=37, W_max=0.17, W_max_self=0.17, W_decay=2e-3)
    if inh == "uniform":
        u = 0.068
        return A1Config(w_EI_self=u, w_EI_lat=u, w_IE_self=u, w_IE_lat=u, **common)
    return A1Config(w_EI_lat=0.030, w_IE_lat=0.120, **common)


LEGACY = {
    "ab_ba":        ab_ba,
    "oddball":      oddball,
    "local_global": local_global,
    "roving":       roving,
    "syllable":     syllable,
    "auditory":     auditory,
    "sfg_old":      sfg_old,
    "sfg2":         sfg2,
}
