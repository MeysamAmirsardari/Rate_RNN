"""Figure-ground detection as a function of onset asynchrony within the figure.

    python -m audios.sfg_task check                 # the confound battery
    python -m audios.sfg_task demo --step-ms 20 --play
    python -m audios.sfg_task calibrate
    python -m audios.sfg_task run S01              # the sweep, ~30 min
    python -m audios.sfg_task run S01 --controls   # the control session
    python -m audios.sfg_task analyse S01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Design
from .plot import envelopes, psychometric, raster

HERE = Path(__file__).resolve().parent
OUT, DATA = HERE / "out", HERE / "data"


def design(a) -> Design:
    kw = {k: v for k, v in vars(a).items()
          if v is not None and k in Design.__dataclass_fields__}
    if a.steps_ms:
        kw["steps_ms"] = tuple(a.steps_ms)
    d = Design(**kw)
    d.validate()
    return d


def cmd_check(a) -> None:
    from .verify import table, verify
    d = design(a)
    print(d.summary(), "\n")
    res = [verify(d, s, n=a.n) for s in d.steps_ms]
    print(table(res))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n  -> {raster(d, OUT / 'stimulus.png').name}"
          f", {envelopes(res, OUT / 'envelopes.png').name}  in {OUT}")


def cmd_demo(a) -> None:
    from .stimulus import make_pool, trial
    d = design(a)
    OUT.mkdir(parents=True, exist_ok=True)
    pl = make_pool(d)
    present, absent = trial(d, pl, step_ms=a.step_ms, seed=a.seed or d.seed,
                            variant=a.variant, rove=False)
    stem = f"step{a.step_ms:03.0f}_{a.variant}"
    files = {f"{stem}_figure": present["y"],
             f"{stem}_no_figure": absent["y"],
             f"{stem}_figure_alone": present["y_fig"]}
    for name, y in files.items():
        sf.write(OUT / f"{name}.wav", y, d.fs, subtype="PCM_24")
    print(f"  {d.coherence} tones, {a.step_ms:g} ms apart "
          f"({d.extent_ms(a.step_ms):.0f} ms per element), {a.variant}")
    print(f"  -> {', '.join(files)} in {OUT}")
    print(f"  -> {raster(d, OUT / f'{stem}.png', steps=[a.step_ms], seed=a.seed or d.seed, variant=a.variant).name}")
    if a.play:
        import sounddevice as sd
        for name, y in files.items():
            print(f"  playing {name}")
            sd.play(np.repeat(y[:, None], 2, 1), d.fs, blocking=True)


def cmd_calibrate(a) -> None:
    from .run import calibrate
    calibrate(design(a), device=a.device)


def cmd_run(a) -> None:
    from .run import run
    d = design(a)
    out = Path(a.out) if a.out else DATA / a.subject
    if a.controls:
        run(d, a.subject, out, device=a.device, steps=d.control_steps_ms,
            variants=d.control_variants, n_per=d.control_trials,
            tag="controls", practice=not a.no_practice)
    else:
        run(d, a.subject, out, device=a.device, practice=not a.no_practice)


def cmd_analyse(a) -> None:
    from .analyse import fit, load, report, score
    d = design(a)
    out = Path(a.out) if a.out else DATA / a.subject
    raw = load(out, a.block)
    summary = score(raw, d.task)
    f = fit(summary, d.task)
    print(report(summary, f, d.task))
    summary.to_csv(out / f"summary_{a.block}.csv", index=False)
    p = psychometric(summary, f, out / f"psychometric_{a.block}.png", 0.5)
    print(f"\n  -> {p.name}, summary_{a.block}.csv  in {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--task", choices=("2ifc", "yesno"))
        q.add_argument("--steps-ms", type=float, nargs="+")
        q.add_argument("--coherence", type=int)
        q.add_argument("--bg-sounding", type=int)
        q.add_argument("--events", type=int)
        q.add_argument("--interval-s", type=float)
        q.add_argument("--trials-per-step", type=int)
        q.add_argument("--order", choices=("rise", "fall"))
        q.add_argument("--seed", type=int)
        return q

    q = common(sub.add_parser("check", help="measure every control"))
    q.add_argument("--n", type=int, default=24, help="trials per step")
    q.set_defaults(fn=cmd_check)

    q = common(sub.add_parser("demo", help="write and play one trial"))
    q.add_argument("--step-ms", type=float, default=20.0)
    q.add_argument("--variant", default="rise",
                   choices=("rise", "perm", "redraw", "scatter"))
    q.add_argument("--play", action="store_true")
    q.set_defaults(fn=cmd_demo)

    q = common(sub.add_parser("calibrate", help="1 kHz at the stimulus level"))
    q.add_argument("--device")
    q.set_defaults(fn=cmd_calibrate)

    q = common(sub.add_parser("run", help="present the experiment"))
    q.add_argument("subject")
    q.add_argument("--controls", action="store_true")
    q.add_argument("--no-practice", action="store_true")
    q.add_argument("--device")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_run)

    q = common(sub.add_parser("analyse", help="d' and the threshold"))
    q.add_argument("subject")
    q.add_argument("--block", default="main")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_analyse)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
