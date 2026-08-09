"""Read the Baphy playback log for the 2026 AB/BA recordings.

This is the table `ft_oe_list` builds as ``outp.stimat``. Baphy writes it to
``<site>/tmp/<run>_p_SEQ.mat`` as ``exptevents``: one row per stimulus epoch
with a ``Note`` naming the sequence, a ``StartTime``/``StopTime`` in seconds
relative to the start of its trial, and the trial number.

Recovering it means the six ``allM2`` tag columns are all available:
``[repLength, prevRepLength, repNum, stim, prevStim, trial]``. Column 4, the
stimulus index, is the one that could not be reconstructed from the Open Ephys
archive alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


BAPHY_ROOT = Path("/Users/eminent/Downloads/nmg038_039")

#: Baphy run -> Open Ephys recording folder. `a01` is Experiment 1 (tones,
#: 180 ms notes), `a03` is Experiment 3 (50 ms notes, 100 ms gap). Experiment 2
#: (speech) has no matching Open Ephys folder in the archive.
RUN_TO_RECORDING = {
    "nmg038a01": "Nutmeg_2026-04-30_15-30-21_SEQ1",
    "nmg038a03": "Nutmeg_2026-04-30_16-05-18_SEQ3",
    "nmg039a01": "Nutmeg_2026-05-01_17-25-02_SEQ1",
    "nmg039a03": "Nutmeg_2026-05-01_17-54-34_SEQ3",
}

_STIM = re.compile(r"^Stim\s*,\s*Note\s+(\S+)\s*,")


@dataclass(frozen=True)
class Playback:
    run: str
    date: str
    trial: np.ndarray          # 1-based acquisition block
    onset_s: np.ndarray        # seconds from the start of that trial
    offset_s: np.ndarray
    name: np.ndarray           # sequence name, e.g. '9400-5300'
    standard: str
    deviant: str
    note_ms: float
    note_gap_ms: float
    sequence_gap_ms: float
    sequences_per_trial: int
    deviant_pct: float

    @property
    def is_deviant(self) -> np.ndarray:
        return self.name == self.deviant

    @property
    def soa_ms(self) -> float:
        return 2 * self.note_ms + self.note_gap_ms + self.sequence_gap_ms


def read_playback(run: str, root: Path | None = None) -> Playback:
    site = run[:-3]
    path = (Path(root or BAPHY_ROOT) / site / "tmp" / f"{run}_p_SEQ.mat")
    loaded = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    events = loaded["exptevents"]
    reference = loaded["exptparams"].TrialObject.ReferenceHandle

    trial, onset, offset, name = [], [], [], []
    for event in events:
        match = _STIM.match(str(event.Note))
        if match is None:
            continue
        trial.append(int(event.Trial))
        onset.append(float(event.StartTime))
        offset.append(float(event.StopTime))
        name.append(match.group(1))

    name = np.array(name)
    unique, counts = np.unique(name, return_counts=True)
    if unique.size != 2:
        raise ValueError(f"{run}: expected two sequences, found {unique.tolist()}")
    order = np.argsort(counts)
    return Playback(
        run=run, date=str(loaded["globalparams"].date),
        trial=np.array(trial, dtype=np.int64),
        onset_s=np.array(onset, dtype=float),
        offset_s=np.array(offset, dtype=float),
        name=name,
        deviant=str(unique[order[0]]), standard=str(unique[order[-1]]),
        note_ms=float(reference.NoteDur) * 1000.0,
        note_gap_ms=float(reference.NoteGap) * 1000.0,
        sequence_gap_ms=float(reference.SeqGap) * 1000.0,
        sequences_per_trial=int(reference.SeqNumber),
        deviant_pct=float(reference.Deviant_pct),
    )


def describe(playback: Playback) -> str:
    n_deviant = int(playback.is_deviant.sum())
    total = playback.name.size
    return (f"{playback.run}  {playback.date}  "
            f"standard {playback.standard} ({total - n_deviant}) / "
            f"deviant {playback.deviant} ({n_deviant}, "
            f"{100 * n_deviant / total:.1f}%)  "
            f"SOA {playback.soa_ms:.0f} ms  "
            f"{playback.sequences_per_trial}/trial x "
            f"{playback.trial.max()} trials")


if __name__ == "__main__":
    for run in sorted(RUN_TO_RECORDING):
        print(describe(read_playback(run)))
