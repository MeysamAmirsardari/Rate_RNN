"""
tasks.export_responses
=======================

Headless export of the **full model0 response** for the deviance experiments,
as CSVs for MATLAB -- so the model can be compared, statistically, with ECoG
recordings of the same paradigms.

For every experiment (and every run / condition separately, where the paradigm
swaps the standard/deviant role) it runs the task's own ``run_experiment`` /
``run_condition`` and dumps **all signals** ``model0.simulate`` produces, each as
a ``(time x channel)`` CSV:

    stim.csv      the stimulus fed to the model (the cochleo-thalamic INPUT)
    tm_in.csv     the thalamo-cortical drive into E (input after short-term plasticity)
    E.csv         excitatory rate            I.csv      inhibitory rate
    u.csv         STP facilitation           x.csv      STP depression (available resource)
    rec_E.csv     recurrent excitation W·E   inh_to_E.csv  inhibitory current M_IE·I
    W_final.csv   the learned E->E weight matrix (channel x channel)

plus, per run:

    events.csv    onset_s, is_deviant, label (+ paradigm extras, e.g. roving block/rep)
    meta.csv      sr, dt, n_channels, decimate, signal list, channel mapping, run params

Load in MATLAB:
    E   = readmatrix('ab_ba/run1_BA_deviant/E.csv');   % col 1 = time_s, cols 2..N+1 = channels
    ev  = readtable ('ab_ba/run1_BA_deviant/events.csv');
    % epoch each signal around ev.onset_s, average is_deviant==1 vs ==0, compare to ECoG

Run
    python -m tasks.export_responses                       # all 4 -> ./tasks_responses
    python -m tasks.export_responses --experiments ab_ba roving
    python -m tasks.export_responses --decimate 5          # signals at 200 Hz (smaller CSVs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# per-channel (N, T) time-series that simulate() emits, in a natural order
_SIGNALS = ["stim", "tm_in", "E", "I", "u", "x", "rec_E", "inh_to_E"]
_SCALAR = (int, float, str, bool, np.integer, np.floating)


def _dump(res: dict, outdir: Path, events: dict, decimate: int = 1):
    """Write every per-channel signal + W_final + events + meta for one run."""
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = res["cfg"]
    dt = float(cfg.dt)
    t = np.asarray(res["t"])
    N, T = res["E"].shape
    td = t[::decimate]
    chan_hdr = "time_s," + ",".join(f"ch{i:02d}" for i in range(N))

    written = []
    for key in _SIGNALS:
        a = res.get(key)
        if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape == (N, T):
            mat = np.column_stack([td, a[:, ::decimate].T])
            np.savetxt(outdir / f"{key}.csv", mat, delimiter=",",
                       header=chan_hdr, comments="", fmt="%.6g")
            written.append(key)

    if isinstance(res.get("W_final"), np.ndarray):
        np.savetxt(outdir / "W_final.csv", res["W_final"], delimiter=",",
                   header=",".join(f"ch{i:02d}" for i in range(N)),
                   comments="", fmt="%.6g")

    cols = list(events.keys())
    ne = len(events[cols[0]])
    with open(outdir / "events.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(ne):
            f.write(",".join(str(events[c][k]) for c in cols) + "\n")

    with open(outdir / "meta.csv", "w") as f:
        f.write("key,value\n")
        f.write(f"sr_hz,{1.0 / (dt * decimate):g}\n")
        f.write(f"dt_s,{dt * decimate:g}\n")
        f.write(f"n_channels,{N}\n")
        f.write(f"n_frames,{td.size}\n")
        f.write(f"decimate,{decimate}\n")
        f.write(f"signals,{'|'.join(written)}\n")
        for k, v in res.items():
            if isinstance(v, _SCALAR):
                f.write(f"{k},{v}\n")
    return written, td.size, N


# ---------------------------------------------------------------------
#  Per-experiment adapters: yield (run_tag, res, events)
# ---------------------------------------------------------------------
def _ab_ba(seed):
    from tasks.ab_ba_model0.ab_ba import run_experiment   # model0 (per-channel I)
    for tag, p_AB in [("run1_BA_deviant", 0.90), ("run2_AB_deviant", 0.10)]:
        res = run_experiment(p_AB=p_AB, seed=seed)
        dt = res["cfg"].dt
        codes = res["codes"]
        minority = "BA" if p_AB > 0.5 else "AB"          # the rare = deviant
        ev = dict(onset_s=res["seq_starts"] * dt,
                  is_deviant=(codes == minority).astype(int),
                  label=codes)
        yield tag, res, ev


def _oddball(seed):
    from tasks.oddball.oddball import run_condition
    from tasks.oddball.config import OddballConfig, ALL_CONDITIONS
    cfg = OddballConfig()
    for cond in ALL_CONDITIONS:
        res = run_condition(cond, cfg=cfg)
        dt = res["cfg"].dt
        ev = dict(onset_s=res["trial_starts"] * dt,
                  is_deviant=res["trial_is_deviant"].astype(int),
                  label=res["trial_channel"])
        yield cond, res, ev


def _local_global(seed):
    from tasks.local_global_model0.local_global import run_experiment
    for tag, std in [("run1_xxxxy_std", "xxxxy"), ("run2_xxxxx_std", "xxxxx")]:
        res = run_experiment(std_type=std, seed=seed)
        dt = res["cfg"].dt
        codes = res["codes"]
        ev = dict(onset_s=res["seq_starts"] * dt,
                  is_deviant=(codes == res["dev_type"]).astype(int),
                  label=codes)
        yield tag, res, ev


def _roving(seed):
    from tasks.roving.roving import run_experiment
    res = run_experiment()
    dt = res["cfg"].dt
    word = np.asarray(res["seq_word"])
    block = np.asarray(res["seq_block"])
    rep = np.zeros(word.size, dtype=int)                 # repetition index within block
    c = 0
    for k in range(word.size):
        c = 0 if (k == 0 or block[k] != block[k - 1]) else c + 1
        rep[k] = c
    ev = dict(onset_s=res["seq_starts"] * dt,
              is_deviant=(rep == 0).astype(int),          # first rep after a rove
              label=word, block=block, rep_in_block=rep)
    yield "session", res, ev


_EXPERIMENTS = {
    "ab_ba": _ab_ba,
    "oddball": _oddball,
    "local_global": _local_global,
    "roving": _roving,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Export full model0 responses (CSV) for the deviance experiments.")
    ap.add_argument("--experiments", nargs="+", default=list(_EXPERIMENTS),
                    choices=list(_EXPERIMENTS))
    ap.add_argument("--outdir", default="tasks_responses")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep every k-th time frame (1 = full 1 kHz; 5 = 200 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    print(f"[ tasks.export_responses ] -> {out}  (decimate={args.decimate})")
    for exp in args.experiments:
        print(f"  == {exp} ==")
        for tag, res, ev in _EXPERIMENTS[exp](args.seed):
            sigs, T, N = _dump(res, out / exp / tag, ev, args.decimate)
            n_dev = int(np.sum(ev["is_deviant"]))
            print(f"    {tag:18s} {N} ch x {T} frames · {len(ev['onset_s'])} "
                  f"events ({n_dev} deviant) · signals: {', '.join(sigs)}")
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
