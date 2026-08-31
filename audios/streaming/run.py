"""Present the experiment.

An adaptive run cannot be pre-rendered: the next level depends on the last
answer.  Both intervals of a trial are two pure tones apiece and take a
couple of milliseconds to build, so they are made between trials and the
time it takes is measured in `selftest` rather than assumed.

Every session is a new directory with its own randomisation and its own
record of who sat down and on what machine.  `--resume` reopens the last
unfinished session and skips the runs it already finished; it refuses if the
design has changed underneath it.
"""

from __future__ import annotations

import csv
import json
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import participant
from .config import Design
from .session import (RUNS, TASK, TRIALS, audio, cell_name, next_session,
                      paths, resumable, run_list, trial_seed, write_meta)
from .stimulus import Balance
from .track import Track

BOLD, DIM, GREEN, RED, OFF = ("\033[1m", "\033[2m", "\033[32m", "\033[31m",
                              "\033[0m")
ANY = "".join(chr(c) for c in range(32, 127)) + "\r\n "


# ------------------------------------------------------------------ input
@contextmanager
def keyboard():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wait_key(fd, valid: str, timeout: float) -> tuple[str | None, float]:
    termios.tcflush(fd, termios.TCIFLUSH)      # ignore anything hit early
    t0 = time.perf_counter()
    while (dt := time.perf_counter() - t0) < timeout:
        if select.select([sys.stdin], [], [], 0.01)[0]:
            c = sys.stdin.read(1).lower()
            if c in valid:
                return c, dt
    return None, timeout


# --------------------------------------------------------------- playback
def play(d: Design, x: np.ndarray, device=None) -> None:
    """Monaural, into the ear the design names."""
    y = np.zeros((x.size, 2), x.dtype)
    if d.ear == "both":
        y[:, 0] = y[:, 1] = x
    else:
        y[:, 0 if d.ear == "left" else 1] = x
    sd.play(y, d.fs, device=device, blocking=True)


def calibrate(d: Design, seconds: float = 5.0, device=None) -> None:
    """A tone at the level of one stimulus tone, in the ear being used."""
    n = int(seconds * d.fs)
    t = np.arange(n) / d.fs
    y = np.sqrt(2.0) * 10 ** (d.rms_dbfs / 20) * np.sin(2 * np.pi * d.f_a * t)
    r = int(0.05 * d.fs)
    y[:r] *= np.linspace(0, 1, r)
    y[-r:] *= np.linspace(1, 0, r)
    print(f"  {d.f_a:.0f} Hz at {d.rms_dbfs:g} dBFS in the {d.ear} ear.")
    print(f"  Set the system so a meter at the headphone reads "
          f"{d.spl_db:g} dB SPL, then leave it alone.")
    play(d, (y * 32767).astype("<i2"), device)


BRIEF = """
{b}  Two short sequences of beeps, one after the other.{o}

  Each sequence has a low beep and a high beep, over and over. In one of the
  two sequences the very last high beep comes {b}slightly early or slightly
  late{o} -- it does not land where the others did.

  Your job is to say {b}which sequence{o} that was, the first or the second.

  It gets harder as you go, on purpose. Guess when you are not sure.

  {b}1{o} or {b}2{o}.   Press {b}q{o} to stop; everything answered is saved.

  Headphones on. Press any key to start."""


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a")
        self.t0 = time.perf_counter()

    def __call__(self, msg: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self.fh.write(f"{stamp}\t{time.perf_counter()-self.t0:8.2f}\t{msg}\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# -------------------------------------------------------------- a session
def run(d: Design, subject: str | None, root: Path, *, device=None,
        resume: bool = False, practice: bool = True) -> None:
    d.validate()
    if not sys.stdin.isatty():
        raise SystemExit(
            "  this needs a real terminal for the keypresses. Run it from a "
            "shell, not from an IDE console, a pipe or a notebook.")

    rows_all = run_list(d, subject or "?", 1)
    n_runs = len(rows_all)
    info = None
    session, done = None, set()
    if resume and subject:
        got = resumable(root, subject, n_runs)
        if not got:
            raise SystemExit(f"  no unfinished session for sub-{subject}")
        session, done = got
        meta = json.loads(paths(root, subject, session)["meta"].read_text())
        if json.loads(json.dumps(asdict(d), default=list)) != meta["design"]:
            raise SystemExit(
                "  the design has changed since that session started, so its "
                "run order is no longer the one that was interrupted. Start a "
                "new session instead.")
        info = meta["participant"]
        print(f"  resuming sub-{subject} ses-{session:02d}: "
              f"{len(done)} of {n_runs} runs already done")
    else:
        info = participant.panel(root, subject)
        subject = info["participant_id"]
        session = next_session(root, subject)

    rows = run_list(d, subject, session)
    p = paths(root, subject, session)
    p["dir"].mkdir(parents=True, exist_ok=True)
    if not resume:
        write_meta(p["meta"], d, info, session, blocks=dict(main=n_runs))
    log = Log(p["log"])
    log(f"start sub-{subject} ses-{session:02d} mode={d.mode} "
        f"{n_runs} runs, resume={resume}")

    new = not p["beh"].exists()
    fh = p["beh"].open("a", newline="")
    w = csv.DictWriter(fh, TRIALS, delimiter="\t", extrasaction="ignore")
    if new:
        w.writeheader()
    new_r = not p["runs"].exists()
    fr = p["runs"].open("a", newline="")
    wr = csv.DictWriter(fr, RUNS, delimiter="\t", extrasaction="ignore")
    if new_r:
        wr.writeheader()

    ctx = dict(d=d, w=w, fh=fh, wr=wr, fr=fr, log=log, device=device,
               session=session, task=TASK, t0=time.perf_counter(), trial=0)

    with keyboard() as fd:
        ctx["fd"] = fd
        print(BRIEF.format(b=BOLD, o=OFF))
        wait_key(fd, ANY, 900)
        print()

        if practice and not resume:
            print(f"{BOLD}  Practice{OFF}  one run, not recorded.\n")
            pr = dict(block="practice", kind="gap", df_st=d.practice_df_st,
                      gap_a_ms=d.gap_b_ms, lag_ms=0.0, pct=None,
                      b_only=False, run=0, repeat=0,
                      seed=trial_seed(d, f"{subject}|practice", 0))
            t = _track(ctx, pr)
            if t is None:
                return _bye(ctx, "quit during practice")
            th = t.threshold()
            print(f"\n  practice threshold "
                  f"{'%.1f ms' % th if th else 'not reached'}\n")

        for i, r in enumerate(rows, 1):
            if r["run"] in done:
                continue
            if i > 1 and (i - 1) % d.break_every == 0:
                _break(ctx, i, n_runs)
            print(f"{BOLD}  run {i}/{n_runs}{OFF}"
                  + (f"   {DIM}{cell_name(r)}{OFF}" if d.show_condition
                     else "") + "\n")
            t = _track(ctx, r)
            if t is None:
                return _bye(ctx, "quit")
            rep = t.report()
            wr.writerow({**{k: r.get(k, "") for k in
                            ("block", "run", "repeat", "kind", "df_st",
                             "gap_a_ms", "lag_ms", "pct", "b_only", "seed")},
                         "session": session, "task": TASK,
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
            fr.flush()
            th = rep["threshold_ms"]
            print(f"\n  {'%.1f ms' % th if th else 'no threshold'}"
                  f"   {DIM}{rep['n_trials']} trials, "
                  f"{100*rep['pc']:.0f}% correct{OFF}\n")

    log("finished")
    print(f"\n{BOLD}  Done.{OFF}  {n_runs} runs.")
    print(f"  trials     {p['beh']}")
    print(f"  runs       {p['runs']}")
    print(f"  details    {p['meta']}")
    fh.close()
    fr.close()
    log.close()


def _track(ctx, r) -> Track | None:
    """One adaptive run.  None if the subject quit."""
    d = ctx["d"]
    t = Track(d)
    bal = Balance(int(r["seed"]) % (2 ** 32))
    k = 0
    while not t.done:
        k += 1
        ctx["trial"] += 1
        dt = t.level()
        target, sign = bal.next()
        seed = trial_seed(d, f"run{r['run']}", k)
        a, target = audio(d, r, dt, seed, target, sign)

        head = (f"\r  {DIM}{k:>3}" +
                (f"  dT {dt:6.2f} ms" if d.show_condition else "") + OFF)
        print(f"{head}   {DIM}listening{OFF}            ", end="", flush=True)
        onset = time.perf_counter() - ctx["t0"]
        for j in range(2):
            if j:
                time.sleep(d.isi_ms / 1000)
            play(d, a[j], ctx["device"])

        print(f"{head}   which one ended late or early?  "
              f"{BOLD}1{OFF} or {BOLD}2{OFF}   ", end="", flush=True)
        c, rt = wait_key(ctx["fd"], "12q", d.response_s)
        if c == "q":
            return None
        resp = int(c) if c else 0
        ok = bool(c) and resp == target

        ctx["w"].writerow({
            "session": ctx["session"], "task": ctx["task"],
            "block": r["block"], "run": r["run"], "cell": cell_name(r),
            "trial": k, "dt_ms": round(dt, 4), "target": target,
            "response": resp if c else "", "correct": int(ok),
            "rt": round(rt, 4), "onset": round(onset, 3), "seed": seed})
        ctx["fh"].flush()

        if d.feedback:
            mark = (f"{GREEN}correct{OFF}" if ok else
                    (f"{RED}wrong{OFF}" if c else f"{RED}too slow{OFF}"))
            print(f"{head}   {mark}                                   ",
                  end="", flush=True)
            time.sleep(d.feedback_ms / 1000)
        time.sleep(d.iti_ms / 1000)
        # an unanswered trial is not scored as wrong: it is not evidence
        # about the listener, and feeding it to the track would drive the
        # level up for a reason that has nothing to do with hearing
        if c:
            t.update(ok)
    return t


def _break(ctx, i, n):
    d = ctx["d"]
    ctx["log"](f"break at run {i}/{n}")
    print(f"\n\n{BOLD}  Break{OFF}: {i - 1} of {n} runs done.")
    print(f"  Rest your ears. Press any key after {d.break_min_s:.0f} s.")
    time.sleep(d.break_min_s)
    wait_key(ctx["fd"], ANY, 900)
    ctx["log"]("break over")
    print()


def _bye(ctx, why: str):
    ctx["log"](why)
    ctx["fh"].close()
    ctx["fr"].close()
    ctx["log"].close()
    print(f"\n  stopped, {why}. Finished runs are saved; "
          f"--resume picks this session up.")
