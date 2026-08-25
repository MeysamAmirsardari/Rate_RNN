"""Measure the controls rather than trusting them.

Every number here is a cue if it differs between the figure-present and the
figure-absent interval.  Construction arguments are not evidence: run this
before running a subject, and put the table in the supplement.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch

from .config import Design
from .stimulus import make_pool, trial


def _envelope(y: np.ndarray, fs: int, smooth_ms: float = 10.0) -> np.ndarray:
    e = np.abs(hilbert(y))
    w = int(round(smooth_ms * fs / 1000.0))
    return np.convolve(e, np.ones(w) / w, mode="same")


def _bands(f: np.ndarray, p: np.ndarray, lo=150.0, hi=8000.0) -> np.ndarray:
    """Third-octave band powers, in dB."""
    edges = lo * 2.0 ** (np.arange(0, np.log2(hi / lo) * 3 + 1) / 3)
    return np.array([10 * np.log10(p[(f >= a) & (f < b)].sum() + 1e-30)
                     for a, b in zip(edges[:-1], edges[1:])])


def _sounding(d: Design, sch: dict) -> np.ndarray:
    n = np.zeros(d.n_slots + d.k)
    for s in sch["slot"]:
        n[s:s + d.k] += 1
    return n[d.k:d.n_slots - d.k]


def verify(d: Design, step_ms: float, n: int = 12, variant: str = "rise",
           win_ms: tuple[float, float] = (-150.0, 500.0)) -> dict:
    """Build `n` trials at one step and compare the two intervals."""
    pl = make_pool(d)
    fs = d.fs
    a0, a1 = (int(round(w * fs / 1000.0)) for w in win_ms)

    acc: dict[str, list] = {k: [[], []] for k in
                            ("tones", "snd_lo", "snd_hi", "rms", "fig", "bg",
                             "epoch", "band", "elem_cv", "repeat")}
    for i in range(n):
        for j, sch in enumerate(trial(d, pl, step_ms=step_ms, seed=9000 + i,
                                      variant=variant, rove=False)):
            y = sch["y"]
            snd = _sounding(d, sch)
            use = np.bincount(sch["chan"], minlength=pl["n"]) / d.interval_s
            fig = np.zeros(pl["n"], bool)
            fig[sch["fig_ch"]] = True

            e = _envelope(y, fs)
            e = e / e.mean()
            ep = []
            for o in sch["onsets"]:
                c0 = int(round(o * d.hop_ms * fs / 1000.0))
                if 0 <= c0 + a0 and c0 + a1 <= e.size:
                    ep.append(e[c0 + a0:c0 + a1])

            f, p = welch(y, fs, nperseg=8192)
            ch = sch["chan"][sch["is_fig"]]
            el = ch.reshape(-1, d.coherence) if variant != "scatter" else None
            rep = 0 if el is None else max(
                np.intersect1d(el[u], el[v]).size
                for u in range(len(el)) for v in range(u))

            acc["tones"][j].append(sch["chan"].size)
            acc["snd_lo"][j].append(snd.min())
            acc["snd_hi"][j].append(snd.max())
            acc["rms"][j].append(20 * np.log10(np.sqrt(np.mean(y ** 2))))
            acc["fig"][j].append(use[fig].mean())
            acc["bg"][j].append(use[~fig].mean())
            acc["epoch"][j].extend(ep)
            acc["band"][j].append(_bands(f, p))
            pw = (sch["gain"][sch["is_fig"]] ** 2
                  * pl["amp"][sch["chan"][sch["is_fig"]]] ** 2)
            pw = pw.reshape(-1, d.coherence).sum(axis=1)
            acc["elem_cv"][j].append(pw.std() / pw.mean())
            acc["repeat"][j].append(rep)

    m = {k: [np.mean(v[0], axis=0), np.mean(v[1], axis=0)]
         for k, v in acc.items()}
    ep_db = [20 * np.log10(x) for x in m["epoch"]]
    band_d = m["band"][0] - m["band"][1]

    # Half of the figure-present epochs against the other half: what the
    # same measurement returns when there is nothing to find.  The
    # between-condition difference means nothing on its own.
    h = len(acc["epoch"][0]) // 2
    floor = np.abs(20 * np.log10(np.mean(acc["epoch"][0][:h], axis=0))
                   - 20 * np.log10(np.mean(acc["epoch"][0][h:], axis=0))).max()

    return dict(
        step_ms=step_ms, variant=variant, n=n,
        tones=(m["tones"][0], m["tones"][1]),
        sounding=(f"{m['snd_lo'][0]:.0f}-{m['snd_hi'][0]:.0f}",
                  f"{m['snd_lo'][1]:.0f}-{m['snd_hi'][1]:.0f}"),
        rms_dbfs=(m["rms"][0], m["rms"][1]),
        fig_rate=(m["fig"][0], m["fig"][1]),
        bg_rate=(m["bg"][0], m["bg"][1]),
        contrast=(m["fig"][0] / m["bg"][0], m["fig"][1] / m["bg"][1]),
        elem_peak_db=(ep_db[0].max(), ep_db[1].max()),
        d_elem_peak_db=float(np.abs(ep_db[0] - ep_db[1]).max()),
        noise_floor_db=float(floor),
        d_rms_db=float(m["rms"][0] - m["rms"][1]),
        d_band_db=float(np.abs(band_d).max()),
        elem_gain_cv=(m["elem_cv"][0], m["elem_cv"][1]),
        shared_channels=(m["repeat"][0], m["repeat"][1]),
        epoch=ep_db, bands=m["band"],
    )


ROWS = [
    ("tones in the interval", "tones", "{:.0f}"),
    ("tones sounding", "sounding", "{}"),
    ("long-term level, dBFS", "rms_dbfs", "{:.2f}"),
    ("figure channel, tones/s", "fig_rate", "{:.2f}"),
    ("other channel, tones/s", "bg_rate", "{:.2f}"),
    ("contrast", "contrast", "{:.2f}"),
    ("element loudness peak, dB", "elem_peak_db", "{:.2f}"),
    ("element power spread, CV", "elem_gain_cv", "{:.4f}"),
    ("channels shared by 2 elements", "shared_channels", "{:.1f}"),
]
DIFFS = [
    ("|present - absent| level, dB", "d_rms_db"),
    ("|present - absent| envelope, dB", "d_elem_peak_db"),
    ("|present - absent| 1/3-oct, dB", "d_band_db"),
    ("  same measure, present only", "noise_floor_db"),
]


def table(res: list[dict]) -> str:
    """One step per column pair, one control per row."""
    w = max(len(r[0]) for r in ROWS + [(n, "") for n, _ in DIFFS]) + 2
    head = "".join(f"{s['step_ms']:>15.0f} ms" for s in res)
    out = [" " * w + head,
           " " * w + "".join(f"{'fig / no fig':>18}" for _ in res)]
    for name, key, fmt in ROWS:
        cells = "".join(f"{fmt.format(s[key][0]) + ' / ' + fmt.format(s[key][1]):>18}"
                        for s in res)
        out.append(f"{name:<{w}}" + cells)
    out.append("")
    for name, key in DIFFS:
        out.append(f"{name:<{w}}"
                   + "".join(f"{abs(s[key]):>18.3f}" for s in res))
    return "\n".join(out)
