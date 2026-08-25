"""Measure the controls on the built stimulus rather than trusting them.

Every number here is a confound if it differs across the conditions being
compared.  Construction arguments are not evidence: each of these has caught
a real defect at some point.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def verify(d: dict, quarters: int = 4) -> dict:
    cfg = d["cfg"]
    k, dens = cfg.k, d["density"]

    starts = np.bincount(d["slot"], minlength=d["n_slots"])
    conc = np.zeros(d["n_slots"] + k)
    for s in d["slot"]:
        conc[s:s + k] += 1
    inner = conc[k:d["n_slots"]]

    use = np.bincount(d["chan"], minlength=d["pool"]["n"])
    is_fig = np.zeros(d["pool"]["n"], bool)
    is_fig[d["fig_ch"]] = True
    dur = d["mix"].size / cfg.fs

    q = np.zeros((d["pool"]["n"], quarters), int)
    for c, s in zip(d["chan"], d["slot"]):
        q[c, min(quarters - 1, s * quarters // d["n_slots"])] += 1
    qbg = q[~is_fig]

    # envelope modulation, in dB, at the figure rate and anywhere else
    e = np.abs(hilbert(d["mix"]))[cfg.fs // 2:-cfg.fs // 2]
    sp = np.abs(np.fft.rfft(e - e.mean())) / e.size
    fr = np.fft.rfftfreq(e.size, 1 / cfg.fs)

    def mod_db(lo, hi):
        m = (fr >= lo) & (fr <= hi)
        return float(20 * np.log10(1 + 2 * sp[m].max() / e.mean()))

    # how long a background channel accidentally repeats at the figure's rate
    period = max(1, int(round(1000.0 / (cfg.rate_hz * cfg.hop))))
    run = 0
    for c in np.flatnonzero(~is_fig):
        s = np.sort(d["slot"][d["chan"] == c])
        if s.size < 2:
            continue
        gap = np.diff(s)
        hit = np.abs(gap - period) <= 1
        cur = best = 0
        for h in hit:
            cur = cur + 1 if h else 0
            best = max(best, cur)
        run = max(run, best)

    return dict(
        starts_per_slot=(int(starts[:-k].min()), int(starts.max())),
        sounding=(int(inner.min()), int(inner.max())),
        expected_sounding=k * dens,
        env_at_figure_rate=mod_db(cfg.rate_hz - .4, cfg.rate_hz + .4),
        env_1_to_50=mod_db(1, 50),
        env_50_to_400=mod_db(50, 400),
        fig_rate=float(use[is_fig].mean() / dur) if is_fig.any() else 0.0,
        bg_rate=float(use[~is_fig].mean() / dur),
        use_all=(int(use.min()), int(use.max())),
        use_bg=(int(use[~is_fig].min()), int(use[~is_fig].max())),
        per_quarter_bg=(int(qbg.min()), int(qbg.max())),
        accidental_run=int(run),
        peak_dbfs=float(20 * np.log10(np.max(np.abs(d["mix"])))),
        rms_dbfs=float(20 * np.log10(np.sqrt(np.mean(d["mix"] ** 2)))),
        seconds=float(dur),
    )


def table(rows: dict) -> str:
    """One condition per column, one control per row."""
    keys = list(next(iter(rows.values())).keys())
    w = max(len(k) for k in keys)
    out = [" " * w + "  " + "".join(f"{n:>18}" for n in rows)]
    for k in keys:
        cells = []
        for r in rows.values():
            v = r[k]
            cells.append(f"{v[0]}-{v[1]}" if isinstance(v, tuple)
                         else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        out.append(f"{k:<{w}}  " + "".join(f"{c:>18}" for c in cells))
    return "\n".join(out)
