"""Generate, measure, plot and save a set of SFG stimuli.

    python -m audios.sfg                      # the default set
    python -m audios.sfg --step-ms 10 --play
    python -m audios.sfg --coherence 4 --n-sounding 12
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from .config import SFGConfig
from .build import build
from .verify import verify, table

OUT = Path(__file__).resolve().parent / "out"


def write_wav(path: Path, x: np.ndarray, fs: int) -> None:
    d = np.clip(np.round(x * (2 ** 31)), -2 ** 31, 2 ** 31 - 1).astype("<i4")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(fs)
        w.writeframes(d.view("uint8").reshape(-1, 4)[:, 1:].tobytes())


def plot(sets: list, path: Path, seconds: float = 1.4) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.6),
                             sharey=True, constrained_layout=True,
                             squeeze=False)
    for ax, (name, d) in zip(axes[0], sets):
        cfg, pool = d["cfg"], d["pool"]
        n = int(seconds * 1000 / cfg.hop)
        sel = d["slot"] < n
        t = d["slot"][sel] * cfg.hop / 1000.0
        f = pool["st"][d["chan"][sel]]
        g = d["is_fig"][sel]
        ax.plot(t[~g], f[~g], "s", ms=3.2, color="k", mec="none")
        ax.plot(t[g], f[g], "s", ms=3.2, color="#E8121A", mec="none")
        ax.set_title(name, fontsize=10.5)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, seconds)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel(f"Semitones re {sets[0][1]['cfg'].f_lo:.0f} Hz")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for f in ("coherence", "n_sounding", "seed"):
        p.add_argument(f"--{f.replace('_', '-')}", type=int, default=None)
    for f in ("step_ms", "tone_ms", "rate_hz", "jitter_ms", "wobble_ms",
              "duration_s", "hop_ms", "contrast", "phon", "peak_dbfs"):
        p.add_argument(f"--{f.replace('_', '-')}", type=float, default=None)
    p.add_argument("--order", choices=("rise", "fall", "perm"), default=None)
    p.add_argument("--no-equal-loudness", action="store_true")
    p.add_argument("--play", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args(argv)

    kw = {k: v for k, v in vars(a).items()
          if v is not None and k not in ("play", "no_plot",
                                         "no_equal_loudness")}
    if a.no_equal_loudness:
        kw["equal_loudness"] = False
    base = SFGConfig(**kw)

    OUT.mkdir(exist_ok=True)
    conds = [("figure", base),
             ("no figure", base.replace(coherent=False)),
             ("lags redrawn", base.replace(redraw_lags=True))]

    rows, sets = {}, []
    print(f"{base.fs} Hz | {base.coherence} of {base.n_sounding} tones "
          f"coherent, {base.tone_ms:g} ms tones, figure every "
          f"{1000/base.rate_hz:g} ms +-{base.jitter_ms:g}")
    print(f"  shear {base.step_ms:g} ms ({base.order}) -> {base.hop:g} ms "
          f"grid, {base.k} slots per tone, {base.density} starting per slot")

    for name, cfg in conds:
        d = build(cfg)
        rows[name] = verify(d)
        sets.append((name, d))
        stem = name.replace(" ", "_")
        write_wav(OUT / f"sfg_{stem}.wav", d["mix"], cfg.fs)
        write_wav(OUT / f"sfg_{stem}_figure.wav", d["figure"], cfg.fs)

    pool = sets[0][1]["pool"]
    print(f"  pool {pool['n']} channels {pool['f'][0]:.0f}-{pool['f'][-1]:.0f}"
          f" Hz on a {pool['grid_st']:g} st grid"
          f"{', equal-loudness weighted' if base.equal_loudness else ''}\n")
    print(table(rows))
    print(f"\n  -> {len(conds) * 2} wavs in {OUT.name}/")

    if not a.no_plot:
        print(f"  -> {plot(sets, OUT / 'sfg_check.png').name}")
    if a.play:
        try:
            import sounddevice as sd
            sd.play(sets[0][1]["mix"], base.fs, blocking=True)
        except Exception as e:                    # noqa: BLE001
            print(f"  (cannot play: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
