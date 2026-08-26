"""Figure-ground detection as a function of onset asynchrony within the figure.

    python -m audios.sfg_task check                 # the confound battery
    python -m audios.sfg_task demo --step-ms 20 --play
    python -m audios.sfg_task calibrate
    python -m audios.sfg_task run                  # asks who is sitting down
    python -m audios.sfg_task run S01 --resume     # finish an interrupted one
    python -m audios.sfg_task run S01 --controls   # the control session
    python -m audios.sfg_task subjects             # what has been recorded
    python -m audios.sfg_task analyse S01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")                     # the CLI is the headless one; the
                                          # notebook wants its own backend
from .config import Design                # noqa: E402
from .plot import envelopes, psychometric, raster   # noqa: E402

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
    root = Path(a.root) if a.root else DATA
    if a.controls:
        run(d, a.subject, root, device=a.device, task="sfgcontrols",
            steps=d.control_steps_ms, variants=d.control_variants,
            n_per=d.control_trials, resume=a.resume,
            practice=not a.no_practice)
    else:
        run(d, a.subject, root, device=a.device, resume=a.resume,
            practice=not a.no_practice)


def cmd_subjects(a) -> None:
    from .session import paths, read_beh, sessions
    root = Path(a.root) if a.root else DATA
    subs = sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())
    if not subs:
        return print(f"  nothing recorded in {root}")
    print(f"  {'subject':<10}{'session':<9}{'task':<15}{'trials':>7}   answered")
    for sid in subs:
        for n in sessions(root, sid):
            for f in sorted(paths(root, sid, n, "x")["dir"].glob("*_beh.tsv")):
                rows = read_beh(f)
                main = [r for r in rows if r["block"] == "main"]
                got = sum(1 for r in main if r["response"])
                task = f.name.split("task-")[1].split("_")[0]
                print(f"  {sid:<10}{n:<9}{task:<15}{len(main):>7}   {got}")


def cmd_analyse(a) -> None:
    from .analyse import fit, load, report, score
    from .session import subject_dir
    d = design(a)
    root = Path(a.root) if a.root else DATA
    label = "sfgcontrols" if a.controls else "sfg"
    raw = load(root, a.subject, task=label, session=a.session)
    if raw.empty:
        return print(f"  no answered trials for sub-{a.subject}, task {label}")
    summary = score(raw, d.task)
    f = fit(summary, d.task)
    ses = sorted(raw.session.unique())
    print(f"  sub-{a.subject}  task-{label}  "
          f"session{'s' if len(ses) > 1 else ''} "
          f"{', '.join(str(int(x)) for x in ses)}  {len(raw)} trials\n")
    print(report(summary, f, d.task))
    out = subject_dir(root, a.subject)
    tag = label if a.session is None else f"{label}_ses-{a.session:02d}"
    summary.to_csv(out / f"sub-{a.subject}_task-{tag}_summary.csv", index=False)
    p = psychometric(summary, f,
                     out / f"sub-{a.subject}_task-{tag}_psychometric.png", 0.5)
    print(f"\n  -> {p.name} and the summary csv, in {out}")


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
    q.add_argument("subject", nargs="?",
                   help="participant id; the panel asks for it if omitted")
    q.add_argument("--resume", action="store_true",
                   help="reopen the last unfinished session of this task "
                        "instead of starting a new one")
    q.add_argument("--controls", action="store_true")
    q.add_argument("--no-practice", action="store_true")
    q.add_argument("--show-step", action="store_true",
                   help="print the condition on the trial line (it is a cue; "
                        "for testing the runner, not for subjects)")
    q.add_argument("--device")
    q.add_argument("--root", help=f"data root (default {DATA})")
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("subjects", help="what has been recorded")
    q.add_argument("--root")
    q.set_defaults(fn=cmd_subjects)

    q = common(sub.add_parser("analyse", help="d' and the threshold"))
    q.add_argument("subject")
    q.add_argument("--controls", action="store_true",
                   help="the control session rather than the sweep")
    q.add_argument("--session", type=int,
                   help="one session; the default pools every session")
    q.add_argument("--root")
    q.set_defaults(fn=cmd_analyse)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
