"""Who the subject is, asked once and kept.

A terminal form rather than a dialog box.  The task runs from a terminal on
whatever machine the booth has, and a GUI toolkit is one more dependency to
install and one more thing to fail on the day.

Nothing here is invented: the consent line records that the experimenter
obtained consent under their own protocol, it does not stand in for it.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# name, prompt, accepted pattern (None = free text), default (None = required)
FIELDS = [
    ("participant_id", "participant id", r"^[A-Za-z0-9_-]{1,16}$", None),
    ("age", "age in years", r"^\d{1,3}$", None),
    ("sex", "sex (f / m / other / na)", r"^(f|m|other|na)$", "na"),
    ("handedness", "handedness (r / l / ambi)", r"^(r|l|ambi)$", "r"),
    ("hearing", "normal hearing, self-reported (y/n)", r"^(y|n)$", "y"),
    ("hearing_notes", "hearing notes, if any", None, ""),
    ("music_years", "years of musical training", r"^\d{1,2}$", "0"),
    ("headphones", "headphone model", None, None),
    ("experimenter", "experimenter initials", r"^[A-Za-z]{1,4}$", None),
    ("consent", "consent obtained under your protocol (y/n)", r"^y$", None),
    ("notes", "anything else worth recording", None, ""),
]
RULE = "  " + "-" * 56


def _prompt(label: str, pattern: str | None, default: str | None) -> str:
    tail = "" if default is None else (
        " (optional)" if default == "" else f" [{default}]")
    while True:
        got = input(f"  {label}{tail}: ").strip()
        if not got:
            if default is None:
                print("    required")
                continue
            got = default            # including an empty one: some fields
        if got and pattern and not re.match(pattern, got, re.I):
            print(f"    expected {pattern}")
            continue
        return got


def ask(subject: str | None = None) -> dict:
    print(f"\n  Participant details\n{RULE}")
    info = {}
    for name, label, pattern, default in FIELDS:
        if name == "participant_id" and subject:
            info[name] = subject
            print(f"  {label}: {subject}")
            continue
        info[name] = _prompt(label, pattern, default)
    return info


def path(root: Path, sid: str) -> Path:
    return root / f"sub-{sid}" / f"sub-{sid}_participant.json"


def load(root: Path, sid: str) -> dict | None:
    p = path(root, sid)
    return json.loads(p.read_text()) if p.exists() else None


def enrol(root: Path, subject: str | None = None) -> dict:
    """The panel.  A returning subject is shown what we already hold and can
    confirm it rather than typing it again."""
    sid = subject or _prompt("participant id", r"^[A-Za-z0-9_-]{1,16}$", None)
    known = load(root, sid)
    if known:
        print(f"\n  Already enrolled: sub-{sid}\n{RULE}")
        for k, v in known.items():
            if k not in ("participant_id", "consent"):
                print(f"  {k:<14} {v}")
        print(RULE)
        if input("  still correct? [y]: ").strip().lower() in ("", "y"):
            known["consent"] = _prompt(
                "consent obtained under your protocol (y/n)", r"^y$", None)
            known["experimenter"] = _prompt(
                "experimenter initials", r"^[A-Za-z]{1,4}$",
                known.get("experimenter"))
            return known
    info = ask(sid)
    save(root, info)
    return info


def save(root: Path, info: dict) -> Path:
    p = path(root, info["participant_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2))
    register(root, info)
    return p


def register(root: Path, info: dict) -> None:
    """One row per subject in participants.tsv, as BIDS expects."""
    cols = ["participant_id"] + [n for n, *_ in FIELDS if n != "participant_id"]
    tsv = root / "participants.tsv"
    rows = {}
    if tsv.exists():
        with tsv.open() as f:
            rows = {r["participant_id"]: r
                    for r in csv.DictReader(f, delimiter="\t")}
    rows[f"sub-{info['participant_id']}"] = {
        **{c: info.get(c, "") for c in cols},
        "participant_id": f"sub-{info['participant_id']}"}
    root.mkdir(parents=True, exist_ok=True)
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
