"""Sessions on disk, the trial list, and turning it into sound.

One directory per session, laid out the way BIDS lays out behaviour, so the
data is readable by someone who has never seen this code.  A session is
written before a single trial is presented, and every trial is appended as
it is answered: a crash costs the trials that were run, not the design.

Each run is a *new* session.  The randomisation is seeded on the subject and
the session number together, so the same subject run twice gets two
different orders and two different sets of stimuli -- which is the point of
running them twice.  `resume` reopens an unfinished one instead, and can
only do so if the design has not changed underneath it.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

from .config import Design
from .stimulus import make_pool, trial


def trial_seed(d: Design, subject: str, i: int) -> int:
    ss = np.random.SeedSequence([d.seed, zlib.crc32(subject.encode()), i])
    return int(ss.generate_state(1)[0])


# ------------------------------------------------------------- provenance
def _git() -> dict:
    here = Path(__file__).resolve().parent
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(here), *a], text=True,
                                  capture_output=True, timeout=5).stdout.strip()
        except Exception:                                     # noqa: BLE001
            return ""
    return dict(commit=run("rev-parse", "HEAD")[:12],
                dirty=bool(run("status", "--porcelain", "--", str(here))))


def provenance() -> dict:
    """Enough to rebuild any stimulus in this session from scratch."""
    return dict(started=datetime.now().astimezone().isoformat(timespec="seconds"),
                host=socket.gethostname(), platform=platform.platform(),
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, **_git())


# ---------------------------------------------------------------- on disk
def subject_dir(root: Path, sid: str) -> Path:
    return root / f"sub-{sid}"


def session_dir(root: Path, sid: str, n: int) -> Path:
    return subject_dir(root, sid) / f"ses-{n:02d}"


def stem(sid: str, n: int, task: str) -> str:
    return f"sub-{sid}_ses-{n:02d}_task-{task}"


def paths(root: Path, sid: str, n: int, task: str) -> dict:
    d, st = session_dir(root, sid, n), stem(sid, n, task)
    return dict(dir=d, beh=d / f"{st}_beh.tsv", meta=d / f"{st}_beh.json",
                log=d / f"{st}_events.log")


def sessions(root: Path, sid: str) -> list[int]:
    d = subject_dir(root, sid)
    return sorted(int(p.name[4:]) for p in d.glob("ses-*") if p.is_dir()) \
        if d.is_dir() else []


def next_session(root: Path, sid: str) -> int:
    return (max(sessions(root, sid)) + 1) if sessions(root, sid) else 1


def read_beh(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resumable(root: Path, sid: str, task: str, n_expected: int):
    """The newest session of this task that is not finished, if any.

    Returns (session number, trial numbers already answered) or None.
    """
    for n in reversed(sessions(root, sid)):
        p = paths(root, sid, n, task)
        rows = [r for r in read_beh(p["beh"]) if r.get("block") == "main"]
        if p["beh"].exists() and len(rows) < n_expected:
            return n, {int(r["trial"]) for r in rows if r["trial"]}
    return None


def _shuffle(rng: np.random.Generator, rows: list[dict],
             max_run: int) -> list[dict]:
    """Randomise, but never more than `max_run` trials in a row in the same
    cell -- a streak lets the listener settle into one condition."""
    cell = [(r["variant"], r["step_ms"]) for r in rows]
    if len(set(cell)) <= max_run:
        return [rows[i] for i in rng.permutation(len(rows))]
    for _ in range(2000):
        order = rng.permutation(len(rows))
        c = [cell[i] for i in order]
        run = 1
        for a, b in zip(c, c[1:]):
            run = run + 1 if a == b else 1
            if run > max_run:
                break
        else:
            return [rows[i] for i in order]
    raise RuntimeError("cannot satisfy max_run; raise it")


def trial_list(d: Design, subject: str, session: int = 1, *, steps=None,
               variants=("rise",), n_per=None, tag: str = "main") -> list[dict]:
    subject = f"{subject}|ses-{session:02d}|{tag}"
    rng = np.random.default_rng(trial_seed(d, subject, 0))
    steps = list(d.steps_ms if steps is None else steps)
    n = d.n_per_step if n_per is None else n_per

    rows = []
    for v in variants:
        # scatter never groups the tones into elements, so the step has
        # nothing to act on: one cell, not one per step.
        for step in ([0.0] if v == "scatter" else steps):
            # half the trials with the figure in the first interval (2ifc)
            # or present (yesno), balanced exactly rather than in expectation
            targets = np.repeat([1, 2] if d.task == "2ifc" else [1, 0],
                                n // 2)
            for t in rng.permutation(targets):
                rows.append(dict(block=tag, variant=v, step_ms=float(step),
                                 target=int(t)))
    rows = _shuffle(rng, rows, d.max_run)
    for i, r in enumerate(rows):
        r["trial"] = i + 1
        r["seed"] = trial_seed(d, subject, i + 1)
    return rows


def audio(d: Design, pl: dict, row: dict) -> list[np.ndarray]:
    """The intervals of one trial, in presentation order, as int16."""
    present, absent = trial(d, pl, step_ms=row["step_ms"], seed=row["seed"],
                            variant=row["variant"])
    if d.task == "2ifc":
        order = [present, absent] if row["target"] == 1 else [absent, present]
    else:
        order = [present if row["target"] else absent]
    return [np.clip(s["y"] * 32767, -32768, 32767).astype("<i2")
            for s in order]


def prerender(d: Design, rows: list[dict], say=print) -> list:
    """Everything up front, so nothing is synthesised while a subject waits."""
    pl = make_pool(d)
    out, step = [], max(1, len(rows) // 20)
    for i, r in enumerate(rows):
        out.append(audio(d, pl, r))
        if i % step == 0:
            say(f"\r  rendering {100 * i // len(rows):3d}%", end="")
    mb = sum(x.nbytes for t in out for x in t) / 2 ** 20
    say(f"\r  rendered {len(rows)} trials, {mb:.0f} MB      ")
    return out


def write_meta(path: Path, d: Design, info: dict, session: int,
               task: str, **extra) -> None:
    """Design, provenance and a snapshot of the participant record, beside
    the responses, so the file is self-describing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        dict(task=task, session=session, participant=info,
             design=asdict(d), provenance=provenance(), **extra),
        indent=2, default=list))
