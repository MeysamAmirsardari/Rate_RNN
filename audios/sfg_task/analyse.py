"""Sensitivity against step, and the asynchrony at which the figure goes.

2IFC is scored straight from proportion correct: there is no criterion to
estimate, which is why it is the default.  Yes/no is scored as
z(hit) - z(false alarm) per step, with the log-linear correction so that a
perfect cell is not infinite.

Everything here is per subject.  `group` pools the thresholds afterwards,
which is the level at which a claim about listeners belongs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest, chi2, fisher_exact, norm, t as student

CHANCE = 0.5


# ------------------------------------------------------------------- load
def load(root: Path, sid: str, task: str = "sfg", session: int | None = None,
         block: str = "main") -> pd.DataFrame:
    """Every answered trial of one task, pooled over sessions by default."""
    pat = f"sub-{sid}/ses-*/sub-{sid}_ses-*_task-{task}_beh.tsv"
    files = sorted(root.glob(pat))
    if session is not None:
        files = [f for f in files if f"ses-{session:02d}" in f.name]
    if not files:
        return pd.DataFrame()
    parts = [x for x in (pd.read_csv(f, sep="\t") for f in files) if len(x)]
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d = d[d.block == block].dropna(subset=["response"])
    d = d[d.response >= 0].copy()
    # position in the experiment, counting sessions end to end
    d = d.sort_values(["session", "trial"]).reset_index(drop=True)
    d["index"] = np.arange(1, len(d) + 1)
    return d


# ----------------------------------------------------------------- scores
def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson interval: behaves at 0 and 1, where the normal one does not."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - (1 - conf) / 2)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _dprime(k: int, n: int, task: str, hits=None, fas=None):
    """d' and its standard error, by the delta method."""
    if task == "2ifc":
        pc = np.clip(k / n, 1 / (2 * n), 1 - 1 / (2 * n))
        z = norm.ppf(pc)
        return np.sqrt(2) * z, np.sqrt(2) * np.sqrt(pc * (1 - pc) / n) / norm.pdf(z)
    (kh, nh), (kf, nf) = hits, fas
    h = (kh + .5) / (nh + 1)
    f = (kf + .5) / (nf + 1)
    zh, zf = norm.ppf(h), norm.ppf(f)
    se = np.sqrt(h * (1 - h) / (nh * norm.pdf(zh) ** 2)
                 + f * (1 - f) / (nf * norm.pdf(zf) ** 2))
    return zh - zf, se


def fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate(
        (p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def score(d: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per cell: accuracy, d', and whether it beats chance."""
    rows = []
    for (v, s), g in d.groupby(["variant", "step_ms"]):
        n, k = len(g), int(g.correct.sum())
        lo, hi = wilson(k, n)
        if task == "2ifc":
            dp, sdp = _dprime(k, n, task)
        else:
            p, a = g[g.target == 1], g[g.target == 0]
            dp, sdp = _dprime(k, n, task,
                              hits=(int(p.correct.sum()), len(p)),
                              fas=(len(a) - int(a.correct.sum()), len(a)))
        rows.append(dict(
            variant=v, step_ms=s, n=n, correct=k, pc=k / n, lo=lo, hi=hi,
            dprime=dp, se_dprime=sdp, rt=g.rt.median(),
            p_chance=binomtest(k, n, CHANCE, "greater").pvalue))
    out = pd.DataFrame(rows).sort_values(["variant", "step_ms"])
    out["q_chance"] = fdr(out.p_chance.values)

    # every step against the easiest one measured, which is the question:
    # where does performance first fall away from the coherent chord?
    ref = out[out.variant == "rise"].iloc[0] if (out.variant == "rise").any() \
        else out.iloc[0]
    p = [fisher_exact([[r.correct, r.n - r.correct],
                       [ref.correct, ref.n - ref.correct]],
                      "less").pvalue if r.step_ms != ref.step_ms else np.nan
         for r in out.itertuples()]
    out["p_vs_best"] = p
    m = ~out.p_vs_best.isna()
    out.loc[m, "q_vs_best"] = fdr(out.loc[m, "p_vs_best"].values)
    return out.reset_index(drop=True)


# -------------------------------------------------------- the step effect
def _ll(beta, x, y):
    z = beta[0] + (beta[1] * x if len(beta) > 1 else 0.0)
    q = CHANCE + (1 - CHANCE) / (1 + np.exp(-z))
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))


def trend(d: pd.DataFrame) -> dict:
    """Does accuracy fall with step at all?

    A logistic on single trials, correct against step, tested against the
    same model with the slope removed.  Guessing is built into the link, so
    the intercept is sensitivity and not a baseline.
    """
    g = d[d.variant == "rise"]
    x = (g.step_ms.values - g.step_ms.mean()) / max(g.step_ms.std(), 1e-9)
    y = g.correct.values.astype(float)
    f1 = minimize(_ll, [1.0, -0.5], args=(x, y), method="Nelder-Mead")
    f0 = minimize(_ll, [1.0], args=(x, y), method="Nelder-Mead")
    stat = 2 * (f0.fun - f1.fun)
    return dict(slope=float(f1.x[1]), chi2=float(stat), df=1,
                p=float(chi2.sf(stat, 1)))


# --------------------------------------------------- psychometric fitting
def _nll(p, x, k, n):
    s50, w, lam = p
    q = CHANCE + (1 - CHANCE - lam) / (1 + np.exp((x - s50) / w))
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(q) + (n - k) * np.log(1 - q))


def fit(summary: pd.DataFrame, task: str, n_boot: int = 400) -> dict | None:
    """A descending logistic in step, by maximum likelihood.

    The threshold reported is the step at which d' falls to 1: a level the
    task can resolve, unlike a half-way point that may sit off the end of
    the range that was tested.
    """
    g = summary[summary.variant == "rise"]
    if len(g) < 4:
        return None
    x, k, n = g.step_ms.values, g.correct.values, g.n.values
    pc1 = norm.cdf(1 / np.sqrt(2)) if task == "2ifc" else norm.cdf(0.5)

    def solve(k_):
        best = None
        for s0 in (10, 25, 40, 60):
            for w0 in (5, 15, 40):
                r = minimize(_nll, [s0, w0, 0.02], args=(x, k_, n),
                             bounds=[(-50, 400), (1, 300), (0, .06)],
                             method="L-BFGS-B")
                if best is None or r.fun < best[0]:
                    best = (r.fun, r.x)
        return best[1]

    p = solve(k)

    def step_at(p_):
        s50, w, lam = p_
        r = (1 - CHANCE - lam) / max(pc1 - CHANCE, 1e-9) - 1
        return s50 + w * np.log(r) if r > 0 else np.nan

    rng = np.random.default_rng(0)
    boot = np.array([step_at(solve(rng.binomial(n, k / n)))
                     for _ in range(n_boot)])
    boot = boot[np.isfinite(boot)]
    s50, w, lam = p
    return dict(s50=s50, width=w, lapse=lam, pc_at_d1=pc1,
                step_at_d1=step_at(p),
                ci=(float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5))) if boot.size
                else (np.nan, np.nan),
                curve=lambda z: CHANCE + (1 - CHANCE - lam)
                / (1 + np.exp((z - s50) / w)))


# ------------------------------------------------------------ time course
def timecourse(d: pd.DataFrame, bw: float | None = None) -> dict:
    """Accuracy against position in the experiment, smoothed.

    A trial of any one step happens once every few trials, so a plain
    sliding window over trial index would hold four or five of them and be
    unreadable.  Each step's trials are weighted by a Gaussian in trial
    index instead, which gives a smooth curve and, from the sum of the
    weights, an honest effective N to put an error band on.
    """
    x = np.arange(1, len(d) + 1, dtype=float)
    bw = bw or max(8.0, len(d) / 7)
    out = {}
    for s, g in d.groupby("step_ms"):
        i = g["index"].values.astype(float)
        c = g.correct.values.astype(float)
        wts = np.exp(-0.5 * ((x[:, None] - i[None, :]) / bw) ** 2)
        neff = wts.sum(axis=1)
        pc = (wts * c[None, :]).sum(axis=1) / np.maximum(neff, 1e-9)
        out[s] = dict(x=x, pc=pc, neff=neff,
                      se=np.sqrt(np.clip(pc * (1 - pc), 0, None)
                                 / np.maximum(neff, 1e-9)))
    c = d.correct.values.astype(float)
    wts = np.exp(-0.5 * ((x[:, None] - x[None, :]) / (bw / 2)) ** 2)
    neff = wts.sum(axis=1)
    pc = (wts * c[None, :]).sum(axis=1) / neff
    out["all"] = dict(x=x, pc=pc, neff=neff,
                      se=np.sqrt(np.clip(pc * (1 - pc), 0, None) / neff))
    return dict(curves=out, bw=bw)


# ------------------------------------------------------------ the sanity checks
def checks(d: pd.DataFrame, task: str):
    """The things that would make the headline number mean something else.

    Only the response bias is a question about a single proportion.  The
    rest are questions about whether two proportions differ, and testing
    those against 0.5 would flag every subject who could do the task at all.
    """
    rows, comps = [], []

    def add(name, k, n, note=""):
        lo, hi = wilson(k, n)
        rows.append(dict(check=name, k=int(k), n=int(n),
                         value=k / n if n else np.nan, lo=lo, hi=hi, note=note))
        return len(rows) - 1

    def against_half(i, note):
        r = rows[i]
        comps.append(dict(comparison=rows[i]["check"], test="vs 0.5",
                          p=binomtest(r["k"], r["n"], .5).pvalue, note=note))

    def pair(i, j, note):
        x, y = rows[i], rows[j]
        p = fisher_exact([[x["k"], x["n"] - x["k"]],
                          [y["k"], y["n"] - y["k"]]]).pvalue
        comps.append(dict(comparison=f"{x['check']} vs {y['check']}",
                          test="Fisher", p=p, note=note))

    if task == "2ifc":
        i = add("said interval 1", (d.response == 1).sum(), len(d))
        against_half(i, "an interval preference")
        i = add("correct, figure in 1", d[d.target == 1].correct.sum(),
                (d.target == 1).sum())
        j = add("correct, figure in 2", d[d.target == 2].correct.sum(),
                (d.target == 2).sum())
        pair(i, j, "the two intervals should be equally good")
    else:
        i = add("hits", d[d.target == 1].correct.sum(), (d.target == 1).sum())
        j = add("correct rejections", d[d.target == 0].correct.sum(),
                (d.target == 0).sum())
        pair(i, j, "a criterion away from neutral")

    h = len(d) // 2
    i = add("first half", d.correct[:h].sum(), h)
    j = add("second half", d.correct[h:].sum(), len(d) - h)
    pair(i, j, "practice or fatigue")

    prev = d.correct.shift(1)
    i = add("after a correct", d[prev == 1].correct.sum(), (prev == 1).sum())
    j = add("after an error", d[prev == 0].correct.sum(), (prev == 0).sum())
    pair(i, j, "sequential effects")

    easy = d[d.step_ms == d.step_ms.min()]
    he = len(easy) // 2
    i = add("anchor first half", easy.correct[:he].sum(), he,
            f"the {d.step_ms.min():g} ms anchor")
    j = add("anchor second half", easy.correct[he:].sum(), len(easy) - he)
    pair(i, j, "is the subject still trying")

    # drift across the session, on single trials rather than by halves
    x = (d["index"].values - d["index"].mean()) / max(d["index"].std(), 1e-9)
    y = d.correct.values.astype(float)
    f1 = minimize(_ll, [1.0, 0.0], args=(x, y), method="Nelder-Mead")
    f0 = minimize(_ll, [1.0], args=(x, y), method="Nelder-Mead")
    st = 2 * (f0.fun - f1.fun)
    comps.append(dict(comparison="drift over the session", test="logistic",
                      p=float(chi2.sf(st, 1)),
                      note=f"slope {f1.x[1]:+.2f} per SD of trial index"))
    return pd.DataFrame(rows), pd.DataFrame(comps)


# ---------------------------------------------------------------- reports
def stars(q: float) -> str:
    return ("***" if q < .001 else "**" if q < .01 else
            "*" if q < .05 else "ns")


def report(summary: pd.DataFrame, f: dict | None, tr: dict,
           ck: pd.DataFrame, cmp: pd.DataFrame, task: str) -> str:
    s = summary.copy()
    s["95% CI"] = [f"{a:.2f}-{b:.2f}" for a, b in zip(s.lo, s.hi)]
    s["d'"] = [f"{a:.2f}+-{b:.2f}" for a, b in zip(s.dprime, s.se_dprime)]
    s["vs chance"] = [stars(q) for q in s.q_chance]
    s["vs best"] = [stars(q) if np.isfinite(q) else "-"
                    for q in s.get("q_vs_best", pd.Series(np.nan, s.index))]
    cols = ["variant", "step_ms", "n", "pc", "95% CI", "d'", "vs chance",
            "vs best", "rt"]
    out = [s[cols].to_string(index=False, float_format="%.3f"), ""]
    out.append(f"  step effect: chi2({tr['df']}) = {tr['chi2']:.1f}, "
               f"p = {tr['p']:.2g}  (single-trial logistic, slope "
               f"{tr['slope']:+.2f})")
    if f:
        g = summary[summary.variant == "rise"]
        span = g.step_ms.max() - g.step_ms.min()
        wide = not np.isfinite(f["ci"][0]) or (f["ci"][1] - f["ci"][0]) > span
        best = g.pc.idxmax() != g.index[0]
        out.append(f"  d' = 1 at {f['step_at_d1']:.1f} ms "
                   f"[{f['ci'][0]:.1f}, {f['ci'][1]:.1f}] 95% CI")
        out.append(f"  half-way at {f['s50']:.1f} ms, width {f['width']:.1f} ms,"
                   f" lapse {f['lapse']:.3f}")
        if wide or best:
            out.append("")
            out.append("  DO NOT USE THAT THRESHOLD:")
            if wide:
                out.append("    its interval is wider than the range of delays "
                           "that were tested, so the fit is unconstrained")
            if best:
                out.append(f"    accuracy peaks at "
                           f"{g.loc[g.pc.idxmax()].step_ms:g} ms rather than at "
                           f"the smallest delay, so the function is not "
                           f"monotonic and a threshold has no meaning")
    out += ["", "  checks"]
    for r in ck.itertuples():
        out.append(f"    {r.check:<22} {r.value:5.2f}  "
                   f"[{r.lo:.2f}-{r.hi:.2f}]  n={r.n}"
                   f"{('   ' + r.note) if r.note else ''}")
    out += ["", "  and whether any of them is real"]
    for r in cmp.itertuples():
        flag = "  <- look" if r.p < .05 else ""
        out.append(f"    {r.comparison:<46} p = {r.p:7.3f}  {r.note}{flag}")
    return "\n".join(out)


# ------------------------------------------------------------------ group
def group(rows: list[dict]) -> str:
    """Thresholds across subjects, which is the level a claim lives at."""
    th = np.array([r["threshold"] for r in rows], float)
    th = th[np.isfinite(th)]
    if th.size < 2:
        return "  need at least two subjects with a threshold"
    m, sd = th.mean(), th.std(ddof=1)
    sem = sd / np.sqrt(th.size)
    ci = student.ppf(.975, th.size - 1) * sem
    return (f"  n = {th.size} subjects\n"
            f"  d' = 1 threshold {m:.1f} +- {sem:.1f} ms (SEM), "
            f"95% CI [{m - ci:.1f}, {m + ci:.1f}], SD {sd:.1f}\n"
            f"  range {th.min():.1f} to {th.max():.1f} ms")
