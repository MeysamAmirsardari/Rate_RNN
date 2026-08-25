"""Present the experiment.

Stimuli are pre-rendered, so the only thing happening during a trial is
playback and a keypress.  Responses are appended to disk as they arrive.
"""

from __future__ import annotations

import csv
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import Design
from .session import prerender, save_design, trial_list

BOLD, DIM, GREEN, RED, OFF = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
FIELDS = ["trial", "block", "variant", "step_ms", "target", "response",
          "correct", "rt", "seed"]


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
    print(f"  1 kHz at {d.rms_dbfs:g} dBFS -- set the system to read "
          f"65 dB SPL, then leave it alone.")
    play(y.astype(np.float32), d.fs, device)


PROMPT = {"2ifc": ("  which interval had the repeating figure?   "
                   f"{BOLD}1{OFF} or {BOLD}2{OFF}", "12"),
          "yesno": ("  was a repeating figure there?   "
                    f"{BOLD}y{OFF} or {BOLD}n{OFF}", "yn")}


def run(d: Design, subject: str, out: Path, *, device=None,
        steps=None, variants=("rise",), n_per=None, tag="main",
        practice=True) -> None:
    d.validate()
    out.mkdir(parents=True, exist_ok=True)
    save_design(d, out / "design.json")

    rows = trial_list(d, subject, steps=steps, variants=variants,
                      n_per=n_per, tag=tag)
    print(d.summary())
    print(f"\n  subject {subject} -> {out}")

    # The list is regenerated identically from the subject and the seed, so
    # a session can be stopped and picked up later -- which it has to be,
    # at this length.
    csv_path = out / f"responses_{tag}.csv"
    new = not csv_path.exists()
    if not new:
        with csv_path.open() as f:
            done = {int(r["trial"]) for r in csv.DictReader(f) if r["trial"]}
        rows = [r for r in rows if r["trial"] not in done]
        print(f"  resuming: {len(done)} trials already done, "
              f"{len(rows)} left")
        if not rows:
            return print("  nothing left to run")
        practice = False
    audio = prerender(d, rows)
    fh = csv_path.open("a", newline="")
    w = csv.DictWriter(fh, FIELDS)
    if new:
        w.writeheader()

    ask, keys = PROMPT[d.task]
    n_int = 2 if d.task == "2ifc" else 1
    print(f"""
{BOLD}  A cloud of short tones.{OFF}  Sometimes a small group of them keeps
  coming back at the same pitches, over and over -- that is the figure.
  It may arrive all at once, or one pitch after another like a run up
  a keyboard.  Either way, listen for {BOLD}the pitches that repeat{OFF}.

{ask.strip()}.   Guess if you are not sure -- you are meant to be unsure.
  Press {BOLD}q{OFF} to stop; what you have done is saved.

  Headphones on, both ears.  Press any key to start.""")

    with keyboard() as fd:
        wait_key(fd, "".join(chr(c) for c in range(32, 127)), 600)

        if practice:
            print(f"\n{BOLD}  Practice{OFF} -- the easiest version, with feedback.")
            prow = trial_list(d, subject + "/practice", steps=[0.0],
                              n_per=d.practice_trials, tag="practice")
            paud = prerender(d, prow)
            hits = []
            for r, a in zip(prow, paud):
                ok = _one(d, r, a, fd, ask, keys, n_int, device, True, w, fh)
                if ok is None:
                    return _bye(fh)
                if ok is not ...:            # a trial with no answer at all
                    hits.append(ok)
                if len(hits) >= 10 and sum(hits[-10:]) >= d.practice_criterion:
                    break
            print(f"  practice: {sum(hits)}/{len(hits)} correct\n")

        done = []
        for i, (r, a) in enumerate(zip(rows, audio)):
            if i and i % d.break_every == 0:
                _break(fd, d, i, len(rows), done)
            ok = _one(d, r, a, fd, ask, keys, n_int, device, d.feedback, w, fh)
            if ok is None:
                return _bye(fh)
            if ok is not ...:
                done.append(ok)

    print(f"\n  done: {sum(done)}/{len(done)} correct "
          f"({100 * np.mean(done):.0f}%)  ->  {csv_path}")
    fh.close()


def _one(d, r, a, fd, ask, keys, n_int, device, feedback, w, fh):
    """One trial.  Returns True/False, or None if the subject quit."""
    print(f"\r  {DIM}trial {r['trial']:>4}  step {r['step_ms']:>3.0f} ms{OFF}"
          f"   {DIM}listening{OFF}          ", end="", flush=True)
    time.sleep(d.ready_ms / 1000)
    for j in range(n_int):
        if j:
            time.sleep(d.isi_ms / 1000)
        play(a[j], d.fs, device)

    print(f"\r  {DIM}trial {r['trial']:>4}  step {r['step_ms']:>3.0f} ms{OFF}"
          f"{ask}   ", end="", flush=True)
    c, rt = wait_key(fd, keys + "q", d.response_s)
    if c == "q":
        return None
    resp = keys.index(c) + 1 if c else 0
    if d.task == "yesno":
        resp = {1: 1, 2: 0, 0: -1}[resp]
    ok = bool(c) and resp == r["target"]


    w.writerow({**{k: r.get(k, "") for k in
                   ("trial", "block", "variant", "step_ms", "target", "seed")},
                "response": resp if c else "", "correct": int(ok),
                "rt": round(rt, 4)})
    fh.flush()
    if feedback:
        mark = f"{GREEN}correct{OFF}" if ok else (
            f"{RED}wrong{OFF}" if c else f"{RED}too slow{OFF}")
        print(f"\r  {DIM}trial {r['trial']:>4}  step {r['step_ms']:>3.0f} ms"
              f"{OFF}   {mark}                              ", end="",
              flush=True)
        time.sleep(d.feedback_ms / 1000)
    time.sleep(d.iti_ms / 1000)
    return ok if c else ...


def _break(fd, d, i, n, done):
    recent = done[-d.break_every:]
    print(f"\n\n{BOLD}  Break{OFF} -- {i}/{n} trials, "
          f"{100 * np.mean(recent):.0f}% correct in the last block.")
    print(f"  Rest your ears.  Press any key after {d.break_min_s:.0f} s.")
    time.sleep(d.break_min_s)
    wait_key(fd, "".join(chr(c) for c in range(32, 127)), 900)
    print()


def _bye(fh):
    fh.close()
    print("\n  stopped; responses saved.")
