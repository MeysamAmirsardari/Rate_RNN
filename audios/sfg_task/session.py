"""The trial list, and turning it into sound.

Balanced before it is shuffled, seeded per trial, and written to disk before
a single trial is presented -- so the session is reproducible from the CSV
alone, and a crash costs the trials that were run, not the design.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Design
from .stimulus import make_pool, trial


def trial_seed(d: Design, subject: str, i: int) -> int:
    ss = np.random.SeedSequence([d.seed, zlib.crc32(subject.encode()), i])
    return int(ss.generate_state(1)[0])


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


def trial_list(d: Design, subject: str, *, steps=None, variants=("rise",),
               n_per=None, tag: str = "main") -> list[dict]:
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


def save_design(d: Design, path: Path) -> None:
    path.write_text(json.dumps(asdict(d), indent=2, default=list))
