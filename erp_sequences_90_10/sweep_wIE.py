"""
erp_sequences_90_10.sweep_wIE
=============================

Sweep the **uniform I->E inhibition gain** on the 90/10 sequence-deviance ERPs.
Each gain ``g`` sets every entry of ``M_IE`` to ``g`` (w_IE_self = w_IE_lat = g)
and writes its four ERP figures to

    erp_sequences_90_10/sweep_wIE/wIE_<g>/erp_<paradigm>.png

so you can compare how the deviance (E and I) scales with inhibition strength.

Run
---
    python -m erp_sequences_90_10.sweep_wIE
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_sequences.config import ERPConfig
from erp_sequences.make_erps import run

# --- edit to taste ------------------------------------------------------
GAINS = [0.1, 0.16, 0.3, 0.6, 1.0]   # uniform I->E gain (applied to ALL pairs)
PARADIGMS = None                      # None = all four; or e.g. ["ab_ac"] to go faster
# ------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(__file__).resolve().parent / "sweep_wIE"
    for g in GAINS:
        out = base / f"wIE_{g}"
        print(f"\n##### uniform w_IE = {g}  ->  {out} #####")
        cfg = ERPConfig(p_dev=0.10, inhibition="uniform", w_IE_self=g, w_IE_lat=g)
        run(cfg, out, PARADIGMS)
    print(f"\nAll {len(GAINS)} gains done -> {base}")
