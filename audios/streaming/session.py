"""Sessions on disk, the order of the adaptive runs, and the sound.

One directory per session, laid out the way BIDS lays out behaviour.  The
unit here is a *track*, not a trial: a run of the adaptive procedure in one
condition, which ends when it ends.  Every trial is appended as it is
answered and every finished track is written with its reversals, so a crash
costs the track that was in progress and nothing else.
"""

from __future__ import annotations

import csv
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
from .stimulus import trial

TASK = "streaming"


def trial_seed(d: Design, subject: str, i: int) -> int:
    ss = np.random.SeedSequence([d.seed, zlib.crc32(subject.encode()), i])
    return int(ss.generate_state(1)[0])


# ------------------------------------------------------------- provenance
def _git() -> dict:
    here = Path(__file__).resolve().parent

    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(here), *a], text=True,
                                  capture_output=True,
                                  timeout=5).stdout.strip()
        except Exception:                                      # noqa: BLE001
            return ""
    return dict(commit=run("rev-parse", "HEAD")[:12],
                dirty=bool(run("status", "--porcelain", "--", str(here))))


def provenance() -> dict:
    return dict(
        started=datetime.now().astimezone().isoformat(timespec="seconds"),
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


def paths(root: Path, sid: str, n: int, task: str = TASK) -> dict:
    d, st = session_dir(root, sid, n), stem(sid, n, task)
    return dict(dir=d, beh=d / f"{st}_beh.tsv", runs=d / f"{st}_runs.tsv",
                meta=d / f"{st}_beh.json", log=d / f"{st}_events.log")


def sessions(root: Path, sid: str) -> list[int]:
    d = subject_dir(root, sid)
    return sorted(int(p.name[4:]) for p in d.glob("ses-*") if p.is_dir()) \
        if d.is_dir() else []


def next_session(root: Path, sid: str) -> int:
    s = sessions(root, sid)
    return (max(s) + 1) if s else 1


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resumable(root: Path, sid: str, n_expected: int, task: str = TASK):
    """The newest unfinished session of this task, and the runs it finished."""
    for n in reversed(sessions(root, sid)):
        p = paths(root, sid, n, task)
        done = {int(r["run"]) for r in read_tsv(p["runs"])
                if r.get("block") == "main" and r.get("run")}
        if p["meta"].exists() and len(done) < n_expected:
            return n, done
    return None


# --------------------------------------------------------------- the order
def run_list(d: Design, subject: str, session: int = 1,
             tag: str = "main") -> list[dict]:
    """Every adaptive run of the session, in the order they are presented.

    One repeat of every condition before any condition gets its second, and
    the order inside a repeat reshuffled each time.  A listener drifts over
    two hours -- practice early, fatigue late -- and blocking this way puts
    the same amount of each on every cell instead of on whichever cell
    happened to be run last.
    """
    key = f"{subject}|ses-{session:02d}|{tag}"
    rng = np.random.default_rng(trial_seed(d, key, 0))
    cells = d.conditions
    rows = []
    for rep in range(d.runs_per_cell):
        for i in rng.permutation(len(cells)):
            rows.append(dict(block=tag, repeat=rep + 1, **cells[int(i)]))
    for i, r in enumerate(rows):
        r["run"] = i + 1
        r["seed"] = trial_seed(d, key, i + 1)
    return rows


def cell_name(r: dict) -> str:
    if r["b_only"]:
        return f"{r['df_st']:.0f}st_Bonly"
    if r["kind"] == "sweep":
        return f"{r['df_st']:.0f}st_lag{r['pct']:.0f}pct"
    return f"{r['df_st']:.0f}st_gap{r['gap_a_ms']:.0f}"


# ------------------------------------------------------------------ sound
def audio(d: Design, row: dict, dt_ms: float, seed: int,
          target: int, sign: int) -> tuple[list[np.ndarray], int]:
    ivs, target = trial(d, df_st=row["df_st"], gap_a_ms=row["gap_a_ms"],
                        lag_ms=row["lag_ms"], dt_ms=dt_ms,
                        b_only=row["b_only"], seed=seed,
                        target=target, sign=sign)
    return [np.clip(iv["y"] * 32767, -32768, 32767).astype("<i2")
            for iv in ivs], target


def write_meta(path: Path, d: Design, info: dict, session: int,
               task: str = TASK, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        dict(task=task, session=session, participant=info,
             design=asdict(d), provenance=provenance(), **extra),
        indent=2, default=list))
