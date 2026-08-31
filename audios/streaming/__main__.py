"""python -m audios.streaming <command>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE / "data"
G, R, OFF, DIM = "\033[32m", "\033[31m", "\033[0m", "\033[2m"


def design(a):
    from .config import Design
    kw = {}
    if getattr(a, "mode", None):
        kw["mode"] = a.mode
    if getattr(a, "ear", None):
        kw["ear"] = a.ear
    if getattr(a, "runs_per_cell", None):
        kw["runs_per_cell"] = a.runs_per_cell
    d = Design(**kw)
    d.validate()
    return d


# ------------------------------------------------------------------ check
def cmd_check(a):
    from .verify import spectral, table, verify
    from .plot import stimulus
    d = design(a)
    print(d.summary(), "\n")
    OUT.mkdir(exist_ok=True)
    rows = []
    if d.mode == "replicate":
        for g in d.gap_a_ms:
            rows.append(verify(d, df_st=a.df_st, gap_a_ms=g, lag_ms=0.0,
                               b_only=False, dt_ms=a.dt_ms, n=a.n))
        rows.append(verify(d, df_st=a.df_st, gap_a_ms=d.gap_b_ms, lag_ms=0.0,
                           b_only=True, dt_ms=a.dt_ms, n=a.n))
    else:
        for p in d.sweep_pct:
            rows.append(verify(d, df_st=a.df_st, gap_a_ms=d.sweep_gap_ms,
                               lag_ms=d.lag_ms(p), b_only=False,
                               dt_ms=a.dt_ms, n=a.n))
    print(table(rows, [spectral(d, x) for x in
                       (d.df_st if d.mode == "replicate" else d.sweep_df_st)]))
    stimulus(d, OUT / f"stimulus_{d.mode}.png", df_st=a.df_st)
    print(f"\n  -> stimulus_{d.mode}.png  in {OUT}")


# ------------------------------------------------------------------- demo
def cmd_demo(a):
    import soundfile as sf
    from .stimulus import to_ear, trial
    d = design(a)
    OUT.mkdir(exist_ok=True)
    lag = d.lag_ms(a.pct) if a.pct is not None else 0.0
    gap = d.sweep_gap_ms if d.mode == "sweep" else a.gap_a_ms
    ivs, target = trial(d, df_st=a.df_st, gap_a_ms=gap, lag_ms=lag,
                        dt_ms=a.dt_ms, b_only=a.b_only, seed=a.seed,
                        target=1, sign=1)
    tag = (f"{d.mode}_{a.df_st:.0f}st_"
           + (f"lag{a.pct:.0f}pct" if a.pct is not None
              else f"gap{gap:.0f}")
           + ("_Bonly" if a.b_only else ""))
    for name, iv in (("signal", ivs[0]), ("standard", ivs[1])):
        p = OUT / f"{tag}_{name}.wav"
        sf.write(p, to_ear(d, iv["y"]), d.fs, subtype="PCM_16")
        print(f"  -> {p.name}")
    print(f"  {a.dt_ms:g} ms shift, {d.tone:.0f} ms tones, "
          f"A {d.f_a:.0f} Hz, B {d.f_b(a.df_st):.0f} Hz, lag {lag:.0f} ms")
    if a.play:
        from .run import play
        for name, iv in (("standard", ivs[1]), ("signal", ivs[0])):
            print(f"  {name}")
            play(d, (iv["y"] * 32767).astype("<i2"))


# -------------------------------------------------------------- selftest
def cmd_selftest(a):
    import time
    from .stimulus import trial
    from .track import geomean, simulate
    d = design(a)

    def ok(name, good, msg):
        print(f"  {name:<12}{G + 'ok' + OFF if good else R + 'FAILED' + OFF:>16}"
              f"  {msg}")
        return good

    print(d.summary(), "\n")
    ok("design", True, f"{len(d.conditions)} cells, {d.n_runs} runs, "
                       f"{d.est_minutes:.0f} min")

    t0 = time.perf_counter()
    n = 0
    for c in d.conditions:
        for _ in range(3):
            trial(d, df_st=c["df_st"], gap_a_ms=c["gap_a_ms"],
                  lag_ms=c["lag_ms"], dt_ms=3.0, b_only=c["b_only"],
                  seed=n, target=1, sign=1)
            n += 1
    ms = (time.perf_counter() - t0) / n * 1000
    ok("stimulus", ms < 60, f"{n} trials, {ms:.0f} ms each "
                            f"(rendered between trials, so this has to be "
                            f"well under the {d.iti_ms:.0f} ms gap)")

    print(f"  {DIM}track      recovering known thresholds, 300 runs each{OFF}")
    worst = 0.0
    for true in (2.6, 3.2, 11.5, 15.0, 21.0):
        th = [simulate(d, true, seed=s).report()["threshold_ms"]
              for s in range(300)]
        g = geomean(th)
        bias = 20 * np.log10(g / true)
        worst = max(worst, abs(bias))
        print(f"    {true:>5.1f} ms -> {g:>5.2f}  ({bias:+.2f} dB, "
              f"sd {np.std(np.log2([x for x in th if x])):.2f} log2)")
    ok("track", worst < 1.5, f"worst bias {worst:.2f} dB; the ceiling at "
                             f"{d.dt_max_ms:g} ms compresses large thresholds")

    try:
        import sounddevice as sd
        outs = [f"{i}: {x['name']}" for i, x in enumerate(sd.query_devices())
                if x["max_output_channels"] > 0]
        ok("devices", bool(outs), f"{len(outs)} output device(s)")
    except Exception as e:                                     # noqa: BLE001
        ok("devices", False, str(e))

    ok("terminal", sys.stdin.isatty(),
       "stdin is a terminal" if sys.stdin.isatty()
       else "stdin is not a terminal. Run from a shell")


# --------------------------------------------------------------- simulate
def cmd_simulate(a):
    """Write a whole session from a listener whose thresholds are known.

    The analysis has to be checked before a person sits down for three
    hours, and the only way to check it is to feed it data whose answer is
    already known.  In `replicate` the truth is the paper's Fig. 2; in
    `sweep` it is a logistic in log threshold running from the listener's
    synchronous value up to their B-only value, with the midpoint at
    `--boundary` ms of lag.  Nothing here is data.
    """
    import csv
    import math
    from .analyse import PAPER
    from .session import cell_name, paths, run_list, write_meta
    from .run import RUNS
    from .track import simulate

    d = design(a)
    root = Path(a.root or DATA)
    sid = a.subject
    rows = run_list(d, sid, 1)

    def truth(r):
        if d.mode == "replicate":
            v = PAPER.get((r["df_st"], None if r["b_only"] else r["gap_a_ms"]))
            return v if v else 14.0
        lo = PAPER[(r["df_st"], 50.0)]
        hi = PAPER[(r["df_st"], None)]
        z = (r["lag_ms"] - a.boundary) / max(a.width, 1e-6)
        return float(lo * (hi / lo) ** (1 / (1 + math.exp(-z))))

    p = paths(root, sid, 1)
    p["dir"].mkdir(parents=True, exist_ok=True)
    write_meta(p["meta"], d, dict(participant_id=sid, note="SIMULATED"), 1,
               blocks=dict(main=len(rows)))
    with p["runs"].open("w", newline="") as f:
        w = csv.DictWriter(f, RUNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            t = simulate(d, truth(r), seed=1000 + i)
            rep = t.report()
            w.writerow({**{k: r.get(k, "") for k in
                           ("block", "run", "repeat", "kind", "df_st",
                            "gap_a_ms", "lag_ms", "pct", "b_only", "seed")},
                        "session": 1, "task": "streaming",
                        "cell": cell_name(r),
                        "threshold_ms": ("" if rep["threshold_ms"] is None
                                         else round(rep["threshold_ms"], 4)),
                        "n_trials": rep["n_trials"],
                        "pc": round(rep["pc"], 4),
                        "n_reversals": rep["n_reversals"],
                        "clamped": rep["clamped"],
                        "at_floor": int(rep["at_floor"]),
                        "at_ceiling": int(rep["at_ceiling"]),
                        "why": rep["why"],
                        "reversals": ",".join(f"{x:.4f}"
                                              for x in rep["reversals"])})
    print(f"  wrote {len(rows)} simulated runs to {p['runs']}")
    print(f"  now: python -m audios.streaming analyse {sid} "
          f"--mode {d.mode} --root {root}")


# ------------------------------------------------------------------- run
def cmd_run(a):
    from .run import run
    run(design(a), a.subject, Path(a.root or DATA), device=a.device,
        resume=a.resume, practice=not a.no_practice)


def cmd_calibrate(a):
    from .run import calibrate
    calibrate(design(a), device=a.device)


def cmd_subjects(a):
    from .session import paths, sessions
    root = Path(a.root or DATA)
    subs = sorted(p.name[4:] for p in root.glob("sub-*") if p.is_dir())
    if not subs:
        return print(f"  nothing in {root}")
    for s in subs:
        ns = sessions(root, s)
        print(f"  sub-{s:<12}{len(ns)} session(s)")
        for n in ns:
            from .session import read_tsv
            r = read_tsv(paths(root, s, n)["runs"])
            done = [x for x in r if x.get("threshold_ms")]
            print(f"    ses-{n:02d}  {len(done)} runs with a threshold")


def cmd_analyse(a):
    from .analyse import by_cell, key_tests, load, load_trials, table, boundary
    from .plot import figure2, sweep, tracks
    root = Path(a.root or DATA)
    d = design(a)
    runs = load(root, a.subject, a.session)
    if not runs:
        return print(f"  no finished runs for sub-{a.subject}")
    cells = by_cell(runs)
    tests = key_tests(d, runs)
    print(f"\n  sub-{a.subject}   {len(runs)} runs, "
          f"{sum(c['n_trials'] for c in cells)} trials\n")
    print(table(d, cells, tests))
    if d.mode == "sweep":
        print()
        for df in sorted({c["df_st"] for c in cells}):
            b = boundary(cells, df)
            if np.isfinite(b["lag_ms"]):
                print(f"  {df:.0f} st: the threshold passes twice the "
                      f"synchronous value at a lag of {b['lag_ms']:.0f} ms "
                      f"({b['pct']:.0f} % of the way to alternation)")
            else:
                print(f"  {df:.0f} st: {b['why']}")

    from .session import subject_dir
    out = subject_dir(root, a.subject)
    stem = f"sub-{a.subject}_task-streaming"
    trials = load_trials(root, a.subject, a.session)
    if d.mode == "replicate":
        figure2(d, cells, out / f"{stem}_figure2.png")
    else:
        sweep(d, cells, out / f"{stem}_sweep.png")
    if trials:
        tracks(d, runs, trials, out / f"{stem}_tracks.png")
    import csv
    with (out / f"{stem}_cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    print(f"\n  -> {out}")


# ------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m audios.streaming")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, mode=True):
        if mode:
            q.add_argument("--mode", choices=("replicate", "sweep"))
        q.add_argument("--ear", choices=("left", "right", "both"))
        q.add_argument("--root")
        return q

    q = common(sub.add_parser("check", help="the acoustic battery"))
    q.add_argument("--df-st", type=float, default=6.0)
    q.add_argument("--dt-ms", type=float, default=3.0)
    q.add_argument("-n", type=int, default=12)
    q.set_defaults(f=cmd_check)

    q = common(sub.add_parser("demo", help="write one trial as wavs"))
    q.add_argument("--df-st", type=float, default=6.0)
    q.add_argument("--gap-a-ms", type=float, default=50.0)
    q.add_argument("--pct", type=float)
    q.add_argument("--dt-ms", type=float, default=20.0)
    q.add_argument("--b-only", action="store_true")
    q.add_argument("--seed", type=int, default=7)
    q.add_argument("--play", action="store_true")
    q.set_defaults(f=cmd_demo)

    q = common(sub.add_parser("selftest", help="before a subject sits down"))
    q.set_defaults(f=cmd_selftest)

    q = common(sub.add_parser("calibrate", help="set the level once"))
    q.add_argument("--device")
    q.set_defaults(f=cmd_calibrate)

    q = common(sub.add_parser("run", help="the experiment"))
    q.add_argument("subject", nargs="?")
    q.add_argument("--device")
    q.add_argument("--resume", action="store_true")
    q.add_argument("--no-practice", action="store_true")
    q.add_argument("--runs-per-cell", type=int)
    q.set_defaults(f=cmd_run)

    q = common(sub.add_parser(
        "simulate", help="a session from a listener whose answer is known"))
    q.add_argument("subject")
    q.add_argument("--boundary", type=float, default=25.0,
                   help="sweep: lag at which the streams come apart, ms")
    q.add_argument("--width", type=float, default=6.0)
    q.add_argument("--runs-per-cell", type=int)
    q.set_defaults(f=cmd_simulate)

    q = common(sub.add_parser("subjects", help="what has been recorded"),
               mode=False)
    q.set_defaults(f=cmd_subjects)

    q = common(sub.add_parser("analyse", help="thresholds, tests, figures"))
    q.add_argument("subject")
    q.add_argument("--session", type=int)
    q.set_defaults(f=cmd_analyse)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
