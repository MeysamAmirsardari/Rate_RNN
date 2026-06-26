"""
erp_sequences_90_10.make_erps
=============================

Strong-oddball variant of the sequence-deviance ERPs: **90% standard / 10%
deviant**.  It reuses the ERP engine in :mod:`erp_sequences.make_erps` and the
shared :class:`erp_sequences.config.ERPConfig`; edit the knobs below to play
with this case.

Inhibition gain
---------------
``W_IE_GAIN`` sets the I->E (inhibition) gain.  Under uniform inhibition every
entry of ``M_IE`` is equal, so this single number is "the inhibition gain for
all" -- it is written to both ``w_IE_self`` (diagonal) and ``w_IE_lat``
(off-diagonal).  ``None`` keeps the model0 preset default (0.16 for uniform).
To sweep, loop over a list of gains and give each its own output folder.

Run
---
    python -m erp_sequences_90_10.make_erps
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_sequences.config import ERPConfig
from erp_sequences.make_erps import run

# --- edit to play with the 90/10 case -----------------------------------
INHIBITION = "uniform"     # "selective" | "uniform"  (overrides config default)
W_IE_GAIN  = None          # uniform I->E gain for ALL channel pairs; None = preset (0.16)
# ------------------------------------------------------------------------

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    cfg = ERPConfig(p_dev=0.10, inhibition=INHIBITION,
                    w_IE_self=W_IE_GAIN, w_IE_lat=W_IE_GAIN)
    raise SystemExit(run(cfg, here))
