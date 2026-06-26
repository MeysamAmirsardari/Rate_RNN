"""
erp_sequences.config
====================

Independent, editable configuration for the sequence-deviance ERP analysis
(:mod:`erp_sequences.make_erps`).  Everything you would want to tune lives here
in one dataclass -- the model0 inhibition / learning knobs, the stimulus timing,
the oddball protocol, the post-learning analysis window, and the paradigm set --
so the analysis can be re-run with different parameters without touching
``make_erps.py``::

    from erp_sequences.config import ERPConfig
    cfg = ERPConfig(p_dev=0.10, tone_dur=0.06, w_IE_self=4.0)
    a1  = cfg.a1config()        # -> the model0 A1Config it implies

The model0 biophysics (``A1Config``) stay where they belong
(``model0/config.py``); ``ERPConfig`` only collects the subset this experiment
varies, plus the experiment-level parameters that ``A1Config`` has no concept of
(p_dev, n_trials, timing, analysis window, paradigms).
"""
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from model0 import A1Config, INH_PRESETS

# standard / deviant token orders (channel indices; A=0, B=1, C=2, ...)
DEFAULT_PARADIGMS: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
    "abc_acb": ((0, 1, 2), (0, 2, 1)),   # ABC / ACB -- order (swap tones 2 & 3)
    "abc_cba": ((0, 1, 2), (2, 1, 0)),   # ABC / CBA -- order (full reversal)
    "ab_ac":   ((0, 1),    (0, 2)),      # AB  / AC  -- feature (2nd tone B->C)
    "ac_bc":   ((0, 2),    (1, 2)),      # AC  / BC  -- feature (1st tone A->B)
}

# The AB/BA MMN regime: applied on top of the *selective* preset only, and only
# where the corresponding ERPConfig field is left as None.  A tone-selective
# I->E arm strong enough to express predictive suppression (w_IE_self), the E->I
# drive that loads it (w_EI_self), and W_norm small enough that the recurrent
# E->E weights approach their asymptote within the trial budget.
_SELECTIVE_MMN = dict(w_IE_self=3.0, w_EI_self=0.40, W_norm=4.0)


@dataclass
class ERPConfig:
    """All tunable parameters for the model0 sequence-deviance ERP analysis."""

    # ---- model0 inhibition / learning knobs -----------------------------
    # `inhibition` picks the model0 preset.  The weight knobs below take the
    # AB/BA MMN regime (_SELECTIVE_MMN: w_IE_self/w_EI_self/W_norm) when left
    # None AND the preset is "selective"; under "uniform" a None leaves the
    # preset's own matched value, so the control stays genuinely uniform.  Set
    # any to a number to force it; use `extra_model` for any other A1Config
    # field (tau_I, tau_trace, multiscale_std, U_std, eta_LTP, ...).
    #
    # I->E is the inhibition gain:  M_IE = w_IE_lat*ones + (w_IE_self-w_IE_lat)*I.
    # For a UNIFORM config, set w_IE_self AND w_IE_lat to the SAME value to
    # change the gain "for all" -- then every entry of M_IE equals that value.
    inhibition: str = "uniform"            # "selective" | "uniform"  (model0 preset)
    n_channels: int = 3                    # tonotopic channels (A,B,C need >= 3)
    w_IE_self: Optional[float] = None      # I_i -> E_i      (None -> 3.0 selective / 0.16 uniform)
    w_IE_lat:  Optional[float] = None      # I_i -> E_{j!=i} (None -> 0.20 selective / 0.16 uniform)
    w_EI_self: Optional[float] = None      # E_i -> I_i      (None -> 0.40 selective / 0.16 uniform)
    w_EI_lat:  Optional[float] = None      # E_i -> I_{j!=i} (None -> 0.05 selective / 0.16 uniform)
    W_norm:    Optional[float] = None      # Hebbian rate scale / learning speed (None -> 4.0 selective)
    extra_model: dict = field(default_factory=dict)   # any other A1Config overrides

    # ---- stimulus timing (seconds) --------------------------------------
    tone_dur:  float = 50e-3           # per-tone duration
    intra_gap: float = 30e-3           # gap between tones within a sequence (load-bearing)
    inter_gap: float = 500e-3          # silence between sequences
    tone_amp:  float = 1.0             # tone amplitude into the TC drive

    # ---- oddball protocol -----------------------------------------------
    p_dev:    float = 0.10             # deviant probability (0.15 = 85/15; 0.10 = 90/10)
    n_trials: int   = 600              # sequences per paradigm
    seed:     int   = 1

    # ---- analysis / plotting --------------------------------------------
    learn_frac:       float = 0.5      # analyse trials AFTER this fraction (post-learning)
    post_tone_window: float = 0.10     # s after the last tone kept for the deviance-peak search
    plot_pad:         float = 0.20     # s of post-stimulus tail shown in the ERP panels
    dpi:              int   = 150

    # ---- which paradigms to run -----------------------------------------
    paradigms: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = field(
        default_factory=lambda: dict(DEFAULT_PARADIGMS))

    # ---------------------------------------------------------------------
    def a1config(self) -> A1Config:
        """Build the model0 ``A1Config`` implied by this ERP config."""
        if self.inhibition not in INH_PRESETS:
            raise ValueError(f"inhibition must be one of {list(INH_PRESETS)}, "
                             f"got {self.inhibition!r}")
        base = INH_PRESETS[self.inhibition]()
        regime = _SELECTIVE_MMN if self.inhibition == "selective" else {}
        over = {"N": self.n_channels}
        for k, v in (("w_IE_self", self.w_IE_self),
                     ("w_IE_lat", self.w_IE_lat),
                     ("w_EI_self", self.w_EI_self),
                     ("w_EI_lat", self.w_EI_lat),
                     ("W_norm", self.W_norm)):
            chosen = v if v is not None else regime.get(k)   # explicit > regime > preset
            if chosen is not None:
                over[k] = chosen
        over.update(self.extra_model)        # extra_model wins on conflict
        return dc.replace(base, **over)
