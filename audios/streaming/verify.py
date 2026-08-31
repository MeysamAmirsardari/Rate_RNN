"""Measure the two intervals rather than trusting them.

The listener is asked which interval ended asynchronously.  Anything else
that separates the two intervals is a way of being right without hearing
what the experiment is about, so every one of them is measured here on
freshly built stimuli.  Construction arguments are not evidence.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .config import Design
from .stimulus import Balance, overlap_ms, trial


# ------------------------------------------------------------------ tools
def erb(f: float) -> float:
    """Glasberg & Moore (1990) equivalent rectangular bandwidth, in Hz."""
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def erb_rate(f: np.ndarray | float) -> np.ndarray | float:
    return 21.4 * np.log10(4.37 * np.asarray(f) / 1000.0 + 1.0)


def dbrms(y: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-30)


def third_octave(y: np.ndarray, fs: int, lo=100.0, hi=8000.0) -> np.ndarray:
    f = np.fft.rfftfreq(y.size, 1 / fs)
    p = np.abs(np.fft.rfft(y)) ** 2
    edges = lo * 2.0 ** (np.arange(0, np.log2(hi / lo) * 3 + 1) / 3)
    return np.array([10 * np.log10(p[(f >= a) & (f < b)].sum() + 1e-30)
                     for a, b in zip(edges[:-1], edges[1:])])


def envelope(y: np.ndarray, fs: int, smooth_ms: float = 2.0) -> np.ndarray:
    e = np.abs(hilbert(y))
    w = max(1, int(round(smooth_ms * fs / 1000.0)))
    return np.convolve(e, np.ones(w) / w, mode="same")


def last_offset_ms(d: Design, iv: dict) -> float:
    """When the interval actually stops making sound."""
    return float(max(iv["a"].max() if not iv["b_only"] else -np.inf,
                     iv["b"].max()) + d.tone)


def splatter_db(d: Design, f: float) -> float:
    """Energy of one gated tone outside +-3 ERB of its own frequency.

    A 10 ms raised cosine is gentle, but the number belongs in the table:
    a gate that splatters would put a click at every onset, and clicks are
    exactly what an asynchrony task is about.
    """
    from .stimulus import _tone
    n = int(round(d.tone * d.fs / 1000.0))
    y = _tone(d, f, n)
    ff = np.fft.rfftfreq(n, 1 / d.fs)
    p = np.abs(np.fft.rfft(y)) ** 2
    near = np.abs(erb_rate(np.maximum(ff, 1.0)) - erb_rate(f)) <= 3.0
    return float(10 * np.log10((p[~near].sum() + 1e-30) / (p.sum() + 1e-30)))


# ------------------------------------------------------------------ battery
def verify(d: Design, *, df_st: float, gap_a_ms: float, lag_ms: float,
           b_only: bool, dt_ms: float, n: int = 8) -> dict:
    """Build `n` trials of one cell at one dT and compare the intervals."""
    fs = d.fs
    f_b = d.f_b(df_st)
    acc: dict[str, list] = {k: [[], []] for k in
                            ("rms", "peak", "energy", "band", "env",
                             "on_ms", "offset_ms", "ab_overlap", "bb_gap",
                             "sign")}
    dt_real = []
    bal = Balance(4242)
    for i in range(n):
        tgt, sgn = bal.next()
        ivs, target = trial(d, df_st=df_st, gap_a_ms=gap_a_ms, lag_ms=lag_ms,
                            dt_ms=dt_ms, b_only=b_only, seed=5000 + i,
                            target=tgt, sign=sgn)
        std, sig = (ivs[1], ivs[0]) if target == 1 else (ivs[0], ivs[1])
        for j, iv in enumerate((std, sig)):
            y = iv["y"]
            acc["rms"][j].append(dbrms(y))
            acc["peak"][j].append(20 * np.log10(np.abs(y).max() + 1e-30))
            acc["energy"][j].append(float(np.sum(y ** 2)))
            acc["band"][j].append(third_octave(y, fs))
            acc["env"][j].append(envelope(y, fs))
            acc["offset_ms"][j].append(last_offset_ms(d, iv))
            n_tones = iv["b"].size * (1 if b_only else 2)
            acc["on_ms"][j].append(n_tones * d.tone)
            acc["ab_overlap"][j].append(
                0.0 if b_only else float(overlap_ms(d, iv["a"], iv["b"])[-1]))
            acc["bb_gap"][j].append(float(iv["b"][-1] - iv["b"][-2] - d.tone))
            acc["sign"][j].append(iv["sign"])
        # what the shift really came out as, after the onsets hit the grid
        ib = np.round(sig["b"] * fs / 1000.0)
        ib0 = np.round(std["b"] * fs / 1000.0)
        dt_real.append(abs(ib[-1] - ib0[-1]) / fs * 1000.0)

    out = dict(df_st=df_st, gap_a_ms=gap_a_ms, lag_ms=lag_ms, b_only=b_only,
               dt_ms=dt_ms, f_b=f_b)
    for k in ("rms", "peak", "on_ms"):
        out[k] = (float(np.mean(acc[k][0])), float(np.mean(acc[k][1])))
    # the shift goes both ways, so report the extremes rather than a mean
    # that would average a forward trial against a backward one and look
    # like nothing had moved
    for k in ("offset_ms", "ab_overlap", "bb_gap"):
        out[k] = (float(np.mean(acc[k][0])),
                  (float(np.min(acc[k][1])), float(np.max(acc[k][1]))))
    out["sign_balance"] = float(np.mean(acc["sign"][1]))
    out["energy_db"] = 10 * np.log10(np.mean(acc["energy"][1])
                                     / np.mean(acc["energy"][0]))
    b0, b1 = np.mean(acc["band"][0], 0), np.mean(acc["band"][1], 0)
    # only bands that carry something.  A third-octave band 100 dB below the
    # peak holds the numerical skirt of the gate and nothing else, and a
    # 1 dB difference down there is not a cue, it is round-off
    audible = b0 >= b0.max() - 40.0
    dd = np.where(audible, np.abs(b0 - b1), -np.inf)
    k = int(np.argmax(dd))
    out["band_db"] = float(np.abs(b0 - b1)[k])
    out["n_bands"] = int(audible.sum())
    edges = 100.0 * 2.0 ** (np.arange(0, np.log2(80.0) * 3 + 1) / 3)
    out["band_hz"] = float(np.sqrt(edges[k] * edges[k + 1]))
    # how far that band sits from either tone, in critical bands
    out["band_erb_from_tone"] = float(min(
        abs(erb_rate(out["band_hz"]) - erb_rate(d.f_a)),
        abs(erb_rate(out["band_hz"]) - erb_rate(f_b))))
    # and how much is in that band at all, so the difference can be read
    out["band_level_db"] = float(b0[k] - b0.max())
    e0, e1 = np.mean(acc["env"][0], 0), np.mean(acc["env"][1], 0)
    out["env_db"] = float(20 * np.log10(
        (np.mean(np.abs(e1 - e0)) + 1e-30) / (np.mean(e0) + 1e-30) + 1.0))
    out["env_first_ms"] = float(
        np.argmax(np.abs(e1 - e0) > 0.05 * e0.max()) / fs * 1000.0)
    out["dt_real_ms"] = float(np.mean(dt_real))
    out["dt_err_ms"] = float(np.max(np.abs(np.array(dt_real) - dt_ms)))
    out["clip"] = bool(max(acc["peak"][0] + acc["peak"][1]) >= 0.0)
    # the peak of two summed tones depends on their relative phase where they
    # overlap, so it moves with the shift.  It also moves from trial to trial
    # now that the phase is drawn per trial, and the comparison that matters
    # is the difference against that spread
    out["peak_sd"] = float(np.std(acc["peak"][0] + acc["peak"][1], ddof=1))
    return out


def spectral(d: Design, df_st: float) -> dict:
    """What the two tones do to each other, independent of any timing."""
    f_a, f_b = d.f_a, d.f_b(df_st)
    gm = np.sqrt(f_a * f_b)
    return dict(
        df_st=df_st, f_b=f_b,
        sep_erb=float(erb_rate(f_b) - erb_rate(f_a)),
        sep_over_erb=float((f_b - f_a) / erb(gm)),
        cdt_hz=float(2 * f_a - f_b),            # cubic difference tone
        cdt_in_band=bool(20.0 < 2 * f_a - f_b < f_a),
        cdt_erb_below_a=float(erb_rate(f_a) - erb_rate(max(2 * f_a - f_b, 20))),
        diff_hz=float(f_b - f_a),
        splatter_a_db=splatter_db(d, f_a),
        splatter_b_db=splatter_db(d, f_b),
    )


# -------------------------------------------------------------------- table
def table(rows: list[dict], spec: list[dict]) -> str:
    """Rows read standard / signal.  Anything that differs is a cue."""
    def head(r):
        if r["b_only"]:
            return f"{r['df_st']:.0f} st  B only"
        if r["lag_ms"] or r["gap_a_ms"] not in (30.0, 50.0, 70.0):
            return f"{r['df_st']:.0f} st  lag {r['lag_ms']:.0f}"
        return f"{r['df_st']:.0f} st  gap {r['gap_a_ms']:.0f}"

    w = 18
    L = [" " * 34 + "".join(f"{head(r):>{w}}" for r in rows),
         " " * 34 + "".join(f"{'std / sig':>{w}}" for r in rows)]

    def line(name, fmt, get):
        L.append(f"{name:<34}" + "".join(f"{get(r):>{w}}" for r in rows))

    line("tones sounding, total ms", None,
         lambda r: f"{r['on_ms'][0]:.0f} / {r['on_ms'][1]:.0f}")
    line("long-term level, dBFS", None,
         lambda r: f"{r['rms'][0]:.2f} / {r['rms'][1]:.2f}")
    line("peak, dBFS", None,
         lambda r: f"{r['peak'][0]:.2f} / {r['peak'][1]:.2f}")
    line("  peak spread over trials, dB", None,
         lambda r: f"{r['peak_sd']:.2f}")
    line("A-B overlap at the target, ms", None,
         lambda r: f"{r['ab_overlap'][0]:.0f} / "
                   f"{r['ab_overlap'][1][0]:.0f}-{r['ab_overlap'][1][1]:.0f}")
    line("gap before the target B, ms", None,
         lambda r: f"{r['bb_gap'][0]:.0f} / "
                   f"{r['bb_gap'][1][0]:.0f}-{r['bb_gap'][1][1]:.0f}")
    line("last offset, ms", None,
         lambda r: f"{r['offset_ms'][0]:.0f} / "
                   f"{r['offset_ms'][1][0]:.0f}-{r['offset_ms'][1][1]:.0f}")
    L.append("")
    line("|sig - std| energy, dB", None, lambda r: f"{r['energy_db']:+.4f}")
    line("|sig - std| 1/3-oct, dB", None, lambda r: f"{r['band_db']:.3f}")
    line("  over ... audible bands", None, lambda r: f"{r['n_bands']}")
    line("  in the band at, Hz", None,
         lambda r: f"{r['band_hz']:.0f}, {r['band_erb_from_tone']:.0f} ERB off")
    line("  that band sits, dB re peak", None,
         lambda r: f"{r['band_level_db']:.0f}")
    line("shift forward on ... of trials", None,
         lambda r: f"{50 * (1 + r['sign_balance']):.0f}%")
    line("first envelope difference, ms", None,
         lambda r: f"{r['env_first_ms']:.0f}")
    line("dT asked / rendered, ms", None,
         lambda r: f"{r['dt_ms']:.2f} / {r['dt_real_ms']:.2f}")
    line("worst dT rounding, ms", None, lambda r: f"{r['dt_err_ms']:.4f}")
    line("clipped", None, lambda r: "YES" if r["clip"] else "no")

    L += ["", "the two tones, before any timing:",
          f"{'':<14}{'f_B Hz':>9}{'sep ERB':>9}{'sep/ERB':>9}"
          f"{'2fA-fB Hz':>11}{'audible':>9}{'splatter dB':>13}"]
    for s in spec:
        L.append(f"{s['df_st']:>10.0f} st{s['f_b']:>9.0f}{s['sep_erb']:>9.1f}"
                 f"{s['sep_over_erb']:>9.1f}{s['cdt_hz']:>11.0f}"
                 f"{'yes' if s['cdt_in_band'] else 'no':>9}"
                 f"{max(s['splatter_a_db'], s['splatter_b_db']):>13.1f}")
    return "\n".join(L)
