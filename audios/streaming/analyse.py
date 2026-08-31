"""Thresholds, and the tests the claim rests on.

Everything here is done in log space.  Asynchrony-detection thresholds are
log-normal -- a listener who is twice as good is a factor, not a difference
-- which is why the paper takes geometric means, and it is also why the
error bars, the t tests and the model are all fitted to log dT.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import Design
from .session import TASK, paths, read_tsv, sessions
from .track import geo_ci


# ------------------------------------------------------------------ load
def load(root: Path, sid: str, session: int | None = None,
         task: str = TASK) -> list[dict]:
    ns = sessions(root, sid) if session is None else [session]
    out = []
    for n in ns:
        for r in read_tsv(paths(root, sid, n, task)["runs"]):
            if r.get("block") != "main" or not r.get("threshold_ms"):
                continue
            out.append(dict(
                session=int(r["session"]), run=int(r["run"]),
                repeat=int(r["repeat"]), cell=r["cell"], kind=r["kind"],
                df_st=float(r["df_st"]), gap_a_ms=float(r["gap_a_ms"]),
                lag_ms=float(r["lag_ms"]),
                pct=float(r["pct"]) if r.get("pct") else None,
                b_only=r["b_only"] in ("1", "True", "true"),
                threshold_ms=float(r["threshold_ms"]),
                n_trials=int(r["n_trials"]), pc=float(r["pc"]),
                clamped=int(r["clamped"]),
                at_floor=r["at_floor"] == "1",
                at_ceiling=r["at_ceiling"] == "1", why=r["why"]))
    return out


def load_trials(root: Path, sid: str, session: int | None = None,
                task: str = TASK) -> list[dict]:
    ns = sessions(root, sid) if session is None else [session]
    return [r for n in ns for r in read_tsv(paths(root, sid, n, task)["beh"])
            if r.get("block") == "main"]


# -------------------------------------------------------------- per cell
def by_cell(runs: list[dict]) -> list[dict]:
    """One row per condition: the geometric mean of that listener's runs."""
    cells: dict[str, list[dict]] = {}
    for r in runs:
        cells.setdefault(r["cell"], []).append(r)
    out = []
    for cell, rs in cells.items():
        th = [r["threshold_ms"] for r in rs]
        g, lo, hi = geo_ci(th)
        out.append(dict(
            cell=cell, kind=rs[0]["kind"], df_st=rs[0]["df_st"],
            gap_a_ms=rs[0]["gap_a_ms"], lag_ms=rs[0]["lag_ms"],
            pct=rs[0]["pct"], b_only=rs[0]["b_only"],
            n_runs=len(rs), threshold_ms=g, lo=lo, hi=hi,
            sd_log2=float(np.std(np.log2(th), ddof=1)) if len(th) > 1
            else float("nan"),
            n_trials=int(np.sum([r["n_trials"] for r in rs])),
            pc=float(np.mean([r["pc"] for r in rs])),
            ceiling=sum(r["at_ceiling"] for r in rs),
            floor=sum(r["at_floor"] for r in rs)))
    key = (lambda r: (r["df_st"], 99.0 if r["b_only"] else
                      (r["pct"] if r["pct"] is not None else r["gap_a_ms"])))
    return sorted(out, key=key)


# ----------------------------------------------------------------- tests
def compare(runs: list[dict], a: str, b: str) -> dict:
    """Welch t on log thresholds: a ratio, not a difference."""
    from scipy import stats
    xa = np.log([r["threshold_ms"] for r in runs if r["cell"] == a])
    xb = np.log([r["threshold_ms"] for r in runs if r["cell"] == b])
    if xa.size < 2 or xb.size < 2:
        return dict(a=a, b=b, ratio=float("nan"), p=float("nan"), n=0)
    t, p = stats.ttest_ind(xa, xb, equal_var=False)
    return dict(a=a, b=b, ratio=float(np.exp(xa.mean() - xb.mean())),
                t=float(t), p=float(p), n=int(xa.size + xb.size))


def fdr(p: list[float]) -> list[float]:
    """Benjamini-Hochberg."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return list(q)
    v = p[ok]
    o = np.argsort(v)
    n = v.size
    adj = np.minimum.accumulate((v[o] * n / np.arange(n, 0, -1))[::-1])[::-1]
    r = np.empty(n)
    r[o] = np.minimum(adj, 1.0)
    q[ok] = r
    return list(q)


def key_tests(d: Design, runs: list[dict]) -> list[dict]:
    """The comparisons the paper's claim actually rests on.

    In `replicate`, one per separation: synchronous against each of the
    three conditions that are not.  The claim is not that the synchronous
    threshold is small, it is that it is *smaller than* the asynchronous
    ones and that those are no better than having no A tones at all.
    """
    out = []
    if d.mode == "replicate":
        for df in d.df_st:
            sync = f"{df:.0f}st_gap{d.gap_b_ms:.0f}"
            for other in ([f"{df:.0f}st_gap{g:.0f}" for g in d.gap_a_ms
                           if g != d.gap_b_ms] + [f"{df:.0f}st_Bonly"]):
                c = compare(runs, sync, other)
                c["what"] = f"{df:.0f} st: synchronous vs {other.split('_')[1]}"
                out.append(c)
            for g in d.gap_a_ms:
                if g == d.gap_b_ms:
                    continue
                c = compare(runs, f"{df:.0f}st_gap{g:.0f}",
                            f"{df:.0f}st_Bonly")
                c["what"] = (f"{df:.0f} st: gap {g:.0f} vs B only "
                             f"(should be the same)")
                out.append(c)
    else:
        for df in d.sweep_df_st:
            base = f"{df:.0f}st_lag0pct"
            for pct in d.sweep_pct[1:]:
                c = compare(runs, base, f"{df:.0f}st_lag{pct:.0f}pct")
                c["what"] = f"{df:.0f} st: synchronous vs {pct:.0f} %"
                out.append(c)
    for c, q in zip(out, fdr([c.get("p", float("nan")) for c in out])):
        c["q"] = q
    return out


# ------------------------------------------------------------- the number
def boundary(cells: list[dict], df_st: float,
             criterion: float = 2.0) -> dict:
    """Where the sweep crosses `criterion` times the synchronous threshold.

    The transition from one stream to two is not a step, so a threshold on
    it has to be named rather than discovered.  A factor of two above the
    listener's own synchronous threshold is a defensible place to put it:
    it is well outside the run-to-run spread of a single cell (typically
    0.5 in log2, so a factor of 1.4) and well below the asynchronous
    plateau, which the paper measures at four to six times synchronous.
    Interpolation is linear in log threshold against lag, which is the
    space the data live in.
    """
    rows = sorted([c for c in cells if c["df_st"] == df_st
                   and c["pct"] is not None], key=lambda r: r["pct"])
    if len(rows) < 2:
        return dict(df_st=df_st, lag_ms=float("nan"), pct=float("nan"),
                    why="not enough lags")
    base = rows[0]["threshold_ms"]
    y = np.log(np.array([r["threshold_ms"] for r in rows]) / base)
    x = np.array([r["lag_ms"] for r in rows])
    tgt = math.log(criterion)
    if y.max() < tgt:
        return dict(df_st=df_st, lag_ms=float("nan"), pct=float("nan"),
                    why=f"never reaches {criterion:g}x synchronous")
    i = int(np.argmax(y >= tgt))
    if i == 0:
        return dict(df_st=df_st, lag_ms=float(x[0]), pct=float(rows[0]["pct"]),
                    why="already above at the first lag")
    f = (tgt - y[i - 1]) / (y[i] - y[i - 1])
    lag = float(x[i - 1] + f * (x[i] - x[i - 1]))
    return dict(df_st=df_st, lag_ms=lag,
                pct=100.0 * lag / (x[-1] if x[-1] else 1.0),
                base_ms=float(base), why="ok")


# ------------------------------------------------------------------ table
PAPER = {   # Fig. 2, read off the published axes, geometric means, N = 9
    (6.0, 50.0): 2.6, (9.0, 50.0): 2.7, (15.0, 50.0): 3.2,
    (6.0, 30.0): 11.5, (9.0, 30.0): 14.0, (15.0, 30.0): 18.0,
    (6.0, 70.0): 13.5, (9.0, 70.0): 14.5, (15.0, 70.0): 21.0,
    (6.0, None): 14.0, (9.0, None): 14.5, (15.0, None): 16.5,
}


def table(d: Design, cells: list[dict], tests: list[dict]) -> str:
    L = []
    if d.mode == "replicate":
        L += [f"{'condition':<18}{'threshold':>10}{'95% CI':>16}"
              f"{'runs':>6}{'sd log2':>9}{'% corr':>8}{'paper':>8}{'ratio':>8}",
              "-" * 83]
    else:
        L += [f"{'condition':<18}{'threshold':>10}{'95% CI':>16}"
              f"{'runs':>6}{'sd log2':>9}{'% corr':>8}{'x sync':>8}",
              "-" * 75]
    base = {}
    for c in cells:
        if c["pct"] == 0.0:
            base[c["df_st"]] = c["threshold_ms"]
    for c in cells:
        ci = (f"{c['lo']:.1f} - {c['hi']:.1f}"
              if np.isfinite(c["lo"]) else "-")
        row = (f"{c['cell']:<18}{c['threshold_ms']:>9.1f} "
               f"{ci:>15}{c['n_runs']:>6}{c['sd_log2']:>9.2f}"
               f"{100*c['pc']:>7.0f}%")
        if d.mode == "replicate":
            p = PAPER.get((c["df_st"], None if c["b_only"] else c["gap_a_ms"]))
            row += (f"{p:>8.1f}{c['threshold_ms']/p:>8.2f}" if p
                    else f"{'-':>8}{'-':>8}")
        else:
            b = base.get(c["df_st"])
            row += f"{c['threshold_ms']/b:>8.2f}" if b else f"{'-':>8}"
        if c["ceiling"]:
            row += f"  {c['ceiling']} run(s) hit the ceiling"
        if c["floor"]:
            row += f"  {c['floor']} run(s) hit the floor"
        L.append(row)

    L += ["", f"{'comparison':<48}{'ratio':>8}{'p':>10}{'q':>10}", "-" * 76]
    for t in tests:
        if not np.isfinite(t.get("p", np.nan)):
            L.append(f"{t['what']:<48}{'too few runs':>28}")
            continue
        L.append(f"{t['what']:<48}{t['ratio']:>8.2f}"
                 f"{t['p']:>10.4f}{t['q']:>10.4f}")
    return "\n".join(L)
