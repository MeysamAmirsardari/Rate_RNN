"""Present the experiment.

Stimuli are pre-rendered, so the only thing happening during a trial is
playback and a keypress.  Responses are appended to disk as they arrive.

Every run is a new session with its own directory, its own randomisation and
its own record of who sat down and on what machine.  `resume` is the
exception, not the default: it reopens the last unfinished session of the
same task, and refuses if the design has changed since.
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
from .session import (next_session, paths, prerender, resumable, trial_list,
                      write_meta)

BOLD, DIM, GREEN, RED, OFF = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
FIELDS = ["session", "task", "block", "trial", "variant", "step_ms", "target",
          "response", "correct", "rt", "onset", "seed"]


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
    termios.tcflush(fd, termios.TCIFLUSH)     # ignore anything hit early
    t0 = time.perf_counter()
    while (dt := time.perf_counter() - t0) < timeout:
        if select.select([sys.stdin], [], [], 0.01)[0]:
            c = sys.stdin.read(1).lower()
            if c in valid:
                return c, dt
    return None, timeout


ANY = "".join(chr(c) for c in range(32, 127))


def play(x: np.ndarray, fs: int, device=None) -> None:
    sd.play(np.repeat(x[:, None], 2, axis=1), fs, device=device, blocking=True)


def calibrate(d: Design, seconds: float = 5.0, device=None) -> None:
    """A 1 kHz tone at the level of the stimuli.  Set the dial once, with a
    meter on the headphone, and every stimulus is then at that level."""
    t = np.arange(int(seconds * d.fs)) / d.fs
    y = np.sin(2 * np.pi * 1000 * t) * np.sqrt(2) * 10 ** (d.rms_dbfs / 20)
    e = int(0.05 * d.fs)
    y[:e] *= np.linspace(0, 1, e)
    y[-e:] *= np.linspace(1, 0, e)
    print(f"  1 kHz at {d.rms_dbfs:g} dBFS. Set the system to read 65 dB SPL, "
          f"then leave it alone.")
    play(y.astype(np.float32), d.fs, device)


PROMPT = {"2ifc": ("  which interval had the repeating figure?   "
                   f"{BOLD}1{OFF} or {BOLD}2{OFF}", "12"),
          "yesno": ("  was a repeating figure there?   "
                    f"{BOLD}y{OFF} or {BOLD}n{OFF}", "yn")}

BRIEF = """
{b}  A cloud of short tones.{o}  Sometimes a small group of them keeps
  coming back at the same pitches, over and over. That is the figure.
  It may arrive all at once, or one pitch after another like a run up
  a keyboard. Either way, listen for {b}the pitches that repeat{o}.

{ask}.   Guess if you are not sure. You are meant to be unsure.
  Press {b}q{o} to stop. Everything answered so far is saved.

  Headphones on, both ears. Press any key to start."""


class Log:
    """A plain text record of what happened and when."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a")
        self.t0 = time.perf_counter()

    def __call__(self, msg: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self.fh.write(f"{stamp}\t{time.perf_counter() - self.t0:8.2f}\t{msg}\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# -------------------------------------------------------------- a session
def run(d: Design, subject: str | None, root: Path, *, device=None,
        task: str = "sfg", steps=None, variants=("rise",), n_per=None,
        resume: bool = False, practice: bool = True) -> None:
    d.validate()
    if not sys.stdin.isatty():
        raise SystemExit(
            "  this needs a real terminal for the keypresses: run it from a "
            "shell, not from an IDE console, a pipe or a notebook")
    info = participant.enrol(root, subject)
    sid = info["participant_id"]

    rows = None
    if resume:
        found = resumable(root, sid, task, d.n_trials)
        if not found:
            return print(f"  nothing to resume for sub-{sid}, task {task}")
        n, done = found
        p = paths(root, sid, n, task)
        # through JSON first: asdict gives tuples, a round trip gives lists
        was = json.loads(p["meta"].read_text())["design"]
        now = json.loads(json.dumps(asdict(d), default=list))
        if was != now:
            return print(f"  ses-{n:02d} was recorded with a different design; "
                         f"finish it with the parameters it was started with, "
                         f"or drop --resume to start a new session")
        rows = [r for r in trial_list(d, sid, n, steps=steps, variants=variants,
                                      n_per=n_per, tag="main")
                if r["trial"] not in done]
        practice = False
        print(f"\n  resuming sub-{sid} ses-{n:02d}: {len(done)} answered, "
              f"{len(rows)} left")
    else:
        n = next_session(root, sid)
        p = paths(root, sid, n, task)

    p["dir"].mkdir(parents=True, exist_ok=True)
    log = Log(p["log"])
    if rows is None:
        rows = trial_list(d, sid, n, steps=steps, variants=variants,
                          n_per=n_per, tag="main")
        write_meta(p["meta"], d, info, n, task,
                   blocks=dict(main=len(rows), practice=d.practice_trials
                               if practice else 0))
        log(f"session start  sub-{sid} ses-{n:02d} task-{task}")
    else:
        log(f"session resumed  {len(rows)} trials left")

    print(f"\n{BOLD}  sub-{sid}  ses-{n:02d}  task-{task}{OFF}")
    print(d.summary())
    print(f"  -> {p['dir']}")
    audio = prerender(d, rows)

    new = not p["beh"].exists()
    fh = p["beh"].open("a", newline="")
    w = csv.DictWriter(fh, FIELDS, delimiter="\t")
    if new:
        w.writeheader()

    ask, keys = PROMPT[d.task]
    ctx = dict(d=d, ask=ask, keys=keys, n_int=2 if d.task == "2ifc" else 1,
               device=device, w=w, fh=fh, log=log, session=n, task=task,
               t0=time.perf_counter())
    print(BRIEF.format(b=BOLD, o=OFF, ask=ask.strip()))

    with keyboard() as fd:
        ctx["fd"] = fd
        wait_key(fd, ANY, 600)

        if practice:
            print(f"\n{BOLD}  Practice{OFF}: the easiest version, with feedback.")
            log("practice start")
            prow = trial_list(d, sid, n, steps=[0.0], n_per=d.practice_trials,
                              tag="practice")
            hits = []
            for r, a in zip(prow, prerender(d, prow)):
                ok = _one(ctx, r, a, feedback=True)
                if ok is None:
                    return _bye(ctx, "quit in practice")
                if ok is not ...:
                    hits.append(ok)
                if len(hits) >= 10 and sum(hits[-10:]) >= d.practice_criterion:
                    break
            met = len(hits) >= 10 and sum(hits[-10:]) >= d.practice_criterion
            print(f"  practice: {sum(hits)}/{len(hits)} correct"
                  + ("" if met else
                     f"  {RED}(never reached {d.practice_criterion}/10; "
                     f"the main block will be hard){OFF}") + "\n")
            log(f"practice done  {sum(hits)}/{len(hits)}  "
                f"criterion {'met' if met else 'NOT met'}")

        done = []
        for i, (r, a) in enumerate(zip(rows, audio)):
            if i and i % d.break_every == 0:
                _break(ctx, i, len(rows), done)
            ok = _one(ctx, r, a, feedback=d.feedback)
            if ok is None:
                return _bye(ctx, f"quit after {len(done)} trials")
            if ok is not ...:
                done.append(ok)

    pc = 100 * float(np.mean(done)) if done else float("nan")
    log(f"session complete  {sum(done)}/{len(done)} correct")
    print(f"\n  done: {sum(done)}/{len(done)} correct ({pc:.0f}%)")
    print(f"  responses  {p['beh']}")
    print(f"  details    {p['meta']}")
    print(f"  log        {p['log']}")
    fh.close()
    log.close()


def _one(ctx, r, a, *, feedback):
    """One trial.  True/False, `...` if unanswered, None if the subject quit."""
    d = ctx["d"]
    head = f"\r  {DIM}trial {r['trial']:>4}" + (
        f"  step {r['step_ms']:>3.0f} ms" if d.show_step else "") + OFF
    print(f"{head}   {DIM}listening{OFF}          ", end="", flush=True)
    onset = time.perf_counter() - ctx["t0"]
    time.sleep(d.ready_ms / 1000)
    for j in range(ctx["n_int"]):
        if j:
            time.sleep(d.isi_ms / 1000)
        play(a[j], d.fs, ctx["device"])

    print(f"{head}{ctx['ask']}   ", end="", flush=True)
    c, rt = wait_key(ctx["fd"], ctx["keys"] + "q", d.response_s)
    if c == "q":
        return None
    resp = ctx["keys"].index(c) + 1 if c else 0
    if d.task == "yesno":
        resp = {1: 1, 2: 0, 0: -1}[resp]
    ok = bool(c) and resp == r["target"]

    ctx["w"].writerow({
        **{k: r.get(k, "") for k in
           ("block", "trial", "variant", "step_ms", "target", "seed")},
        "session": ctx["session"], "task": ctx["task"],
        "response": resp if c else "", "correct": int(ok),
        "rt": round(rt, 4), "onset": round(onset, 3)})
    ctx["fh"].flush()

    if feedback:
        mark = f"{GREEN}correct{OFF}" if ok else (
            f"{RED}wrong{OFF}" if c else f"{RED}too slow{OFF}")
        print(f"{head}   {mark}                              ",
              end="", flush=True)
        time.sleep(d.feedback_ms / 1000)
    time.sleep(d.iti_ms / 1000)
    return ok if c else ...


def _break(ctx, i, n, done):
    d = ctx["d"]
    recent = done[-d.break_every:] or [0]
    pc = 100 * float(np.mean(recent))
    ctx["log"](f"break at trial {i}/{n}  last block {pc:.0f}%")
    print(f"\n\n{BOLD}  Break{OFF}: {i} of {n} trials, {pc:.0f}% correct in "
          f"the last block.")
    print(f"  Rest your ears. Press any key after {d.break_min_s:.0f} s.")
    time.sleep(d.break_min_s)
    wait_key(ctx["fd"], ANY, 900)
    ctx["log"]("break over")
    print()


def _bye(ctx, why: str):
    ctx["log"](why)
    ctx["fh"].close()
    ctx["log"].close()
    print(f"\n  stopped, {why}. Answers are saved; "
          f"--resume picks this session up.")
