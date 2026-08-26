"""Sensitivity against step, and the asynchrony at which the figure goes.

2IFC is scored straight from proportion correct -- no criterion to estimate,
which is why it is the default.  Yes/no is scored as z(hit) - z(false alarm)
per step, with the log-linear correction so a perfect cell is not infinite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def load(root: Path, sid: str, task: str = "sfg", session: int | None = None,
         block: str = "main") -> pd.DataFrame:
    """Every answered trial of one task, pooled over sessions by default."""
    pat = f"sub-{sid}/ses-*/sub-{sid}_ses-*_task-{task}_beh.tsv"
    files = sorted(root.glob(pat))
    if session is not None:
        files = [f for f in files if f"ses-{session:02d}" in f.name]
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f, sep="\t") for f in files],
                  ignore_index=True)
    d = d[d.block == block].dropna(subset=["response"])
    return d[d.response >= 0]


def score(d: pd.DataFrame, task: str) -> pd.DataFrame:
    """Proportion correct and d' for every step of every variant."""
    rows = []
    for (v, s), g in d.groupby(["variant", "step_ms"]):
        n, c = len(g), int(g.correct.sum())
        pc = np.clip(c / n, 1 / (2 * n), 1 - 1 / (2 * n))
        if task == "2ifc":
            dp = np.sqrt(2) * norm.ppf(pc)
        else:
            p, a = g[g.target == 1], g[g.target == 0]
            h = (p.correct.sum() + .5) / (len(p) + 1)
            fa = (len(a) - a.correct.sum() + .5) / (len(a) + 1)
            dp = norm.ppf(h) - norm.ppf(fa)
        rows.append(dict(variant=v, step_ms=s, n=n, correct=c, pc=c / n,
                         se=np.sqrt(pc * (1 - pc) / n), dprime=dp,
                         rt=g.rt.median()))
    return pd.DataFrame(rows).sort_values(["variant", "step_ms"])


def _nll(p, x, k, n, chance):
    s50, w, lam = p
    q = chance + (1 - chance - lam) / (1 + np.exp((x - s50) / w))
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(q) + (n - k) * np.log(1 - q))


def fit(d: pd.DataFrame, task: str, n_boot: int = 2000) -> dict | None:
    """A descending logistic in step, by maximum likelihood.

    The reported threshold is the step at which d' falls to 1 -- a level the
    task can actually resolve, unlike a half-way point that may sit off the
    end of the tested range.
    """
    g = d[d.variant == "rise"]
    if len(g) < 4:
        return None
    chance = 0.5 if task == "2ifc" else 0.5
    x, k, n = g.step_ms.values, g.correct.values, g.n.values
    pc1 = norm.cdf(1 / np.sqrt(2)) if task == "2ifc" else norm.cdf(0.5)

    def solve(k_):
        best, p0 = None, None
        for s0 in (10, 25, 40, 60):
            for w0 in (5, 15, 40):
                r = minimize(_nll, [s0, w0, 0.02], args=(x, k_, n, chance),
                             bounds=[(-50, 400), (1, 300), (0, .06)],
                             method="L-BFGS-B")
                if best is None or r.fun < best:
                    best, p0 = r.fun, r.x
        return p0

    p = solve(k)

    def step_at(p_):
        s50, w, lam = p_
        r = (1 - chance - lam) / max(pc1 - chance, 1e-9) - 1
        return s50 + w * np.log(r) if r > 0 else np.nan

    rng = np.random.default_rng(0)
    boot = np.array([step_at(solve(rng.binomial(n, k / n)))
                     for _ in range(n_boot // 20)])
    boot = boot[np.isfinite(boot)]

    s50, w, lam = p
    return dict(s50=s50, width=w, lapse=lam, pc_at_d1=pc1,
                step_at_d1=step_at(p),
                ci=(np.percentile(boot, 2.5), np.percentile(boot, 97.5))
                if boot.size else (np.nan, np.nan),
                curve=lambda z: chance + (1 - chance - lam)
                / (1 + np.exp((z - s50) / w)))


def report(summary: pd.DataFrame, f: dict | None, task: str) -> str:
    out = [summary.drop(columns=["se"]).to_string(index=False,
                                                  float_format="%.3f")]
    if f:
        out += ["",
                f"  d' = 1 at {f['step_at_d1']:.1f} ms "
                f"[{f['ci'][0]:.1f}, {f['ci'][1]:.1f}] 95% CI",
                f"  half-way at {f['s50']:.1f} ms, width {f['width']:.1f} ms, "
                f"lapse {f['lapse']:.3f}"]
    return "\n".join(out)
