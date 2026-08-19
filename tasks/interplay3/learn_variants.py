"""
tasks.interplay3.learn_variants
===============================

Two changes to what layer 2 *learns*, aimed at the pedestal.

The problem, measured
---------------------
Over half of every mask ends up on cloud-by-cloud entries, against about 2% on
the block that codes its word.  Not because any single cloud entry is large --
they sit at 0.13 of a row peak against 1.00 for the immediately preceding tone
-- but because there are 38 x 6 cloud columns and 38 cloud rows, so the small
entries win on sheer number.  Five cloud tones sound at every instant, so that
half of the mask is driven continuously and contributes a floor.

Subtracting the population mean at read-out hides this.  It does not fix it:
the masks are still mostly cloud, the competition still runs on the cloud, and
the units still pile onto one or two words.  Both variants here change the
learning instead.

`prune` -- synaptic elimination
-------------------------------
After each instar update, zero every mask entry below a fraction of that unit's
largest.  The existing ``lam`` decays all synapses uniformly and prunes whole
*units*; nothing in the published rule prunes *synapses*, so a mask can carry
an arbitrarily large number of arbitrarily weak ones and there is no pressure
against it.  Hebbian competition in cortex does eliminate weak synapses, and
that is the missing term.

The threshold has to clear the cloud band (0.13 of a row peak) without cutting
the deepest predecessor (0.33), so it belongs between them.

`lateral` -- competition under inhibition
-----------------------------------------
The winner is currently the unit with the highest cosine match to the current
map.  Because every mask is mostly cloud and the cloud is the same for all of
them, that quantity is dominated by a term the units share, and the competition
partly measures who tracks the background best.

Under lateral inhibition a unit wins when it stands out *from the population*,
not when it is absolutely large.  Subtracting the population mean response
before the comparison is the cheap form of that, and it is the same operation
the read-out patch applied afterwards -- but applied where it can change what
gets learned rather than only how it is drawn.

Neither variant touches the stimulus, layer 1, the filterbank, or the instar
rule itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if __package__:
    from .fast_layer2 import FastLayer2MR, _G_FLOOR
else:
    from tasks.interplay3.fast_layer2 import FastLayer2MR, _G_FLOOR  # type: ignore


class VariantLayer2MR(FastLayer2MR):
    """``FastLayer2MR`` with optional synaptic pruning and lateral inhibition.

    With both options off it is exactly ``FastLayer2MR``, which is exactly
    ``Layer2MR``; the defaults are therefore the published model.
    """

    def __init__(self, n_channels: int, cfg=None, *, prune_frac: float = 0.0,
                 lateral: bool = False):
        super().__init__(n_channels, cfg)
        self.prune_frac = float(prune_frac)
        self.lateral = bool(lateral)

    def step(self, E: np.ndarray, dt: float, learn: bool = True):
        c = self.cfg
        self.s += dt * (-self.s + E[:, None]) / self.tau[None, :]

        D = self.coincidence(E)
        D_flat = D.reshape(-1)
        M_flat = self.M.reshape(c.n_units, -1)

        nD = float(np.linalg.norm(D_flat))
        if nD > self.peak:
            self.peak = nD

        raw = M_flat @ D_flat
        y = np.maximum(self.g * raw, 0.0)

        if learn:
            if nD > c.gate_frac * self.peak:
                score = raw - raw.mean() if self.lateral else raw
                k = int(np.argmax(score / (nD * (self._norms + 1e-12))))
                Mk = self.M[k]
                Mk *= (1.0 - c.eta)
                Mk += (c.eta / self.g) * (D / nD)
                np.clip(Mk, 0.0, None, out=Mk)
                if self.prune_frac > 0.0:
                    # Relative to the unit's own peak, so the global scale
                    # cancels and the threshold means the same thing at every
                    # point in the run.
                    Mk[Mk < self.prune_frac * Mk.max()] = 0.0
                self._norms[k] = np.linalg.norm(Mk)
                self.win_counts[k] += 1
            self.g *= (1.0 - c.lam)
            if self.g < _G_FLOOR:
                self._renormalise()
        return y, D
