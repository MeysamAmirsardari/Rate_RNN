"""
live_demo_cortical.export_responses
===================================

Headless (no-GUI) export of the **model's responses** for the directional
experiments, as CSVs for MATLAB -- so the model can be compared, statistically,
against ECoG recordings of the same standard/deviant paradigms.

For each experiment it runs the exact directional pipeline (cochlear front end →
A1 RNN → directed coincidence ``D[i,j]=⟨E_i·tr_j⟩``) and writes, into
``<outdir>/<experiment>/``:

    response_E.csv   time_s + per-channel excitatory rate E(t)  (the cortical
                     response; the analogue of per-electrode ECoG activity)
    signals.csv      time_s, pop_E (Σ E), fwd_energy (consistent order),
                     rev_energy (order VIOLATION = the model's deviance signal),
                     net_flow (fwd − rev)
    events.csv       onset_s, is_deviant (0/1), sequence (e.g. ABC / CAB)
    meta.csv         key,value: sr, dt, n_channels, tau_trace, forget_s, tone
                     frequencies, lead/active/period, paradigm definition

Load in MATLAB, e.g.:
    R   = readmatrix('abc_cab/response_E.csv');   % col 1 = time_s, cols 2..N+1 = channels
    ev  = readtable('abc_cab/events.csv');        % epoch around ev.onset_s by ev.is_deviant
    sig = readtable('abc_cab/signals.csv');

Run
    python -m live_demo_cortical.export_responses                 # all 4 -> ./responses
    python -m live_demo_cortical.export_responses --seconds 120 --tau-trace 0.15
    python -m live_demo_cortical.export_responses --paradigms abc_cab abc_cba
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_demo_cortical.config import get_preset
from live_demo_cortical.audio import SpectroFrontEnd, SequenceSource
from live_demo_cortical.engine import LiveEngine
from live_demo_cortical.__main__ import _SEQ_PARADIGMS

_LETTERS = "ABCDEFG"


def _seq_str(seq):
    return "".join(_LETTERS[i] for i in seq)


def _run(cfg, standard, deviant, seconds, seed):
    """Drive the directional pipeline; return (E (N,T), fwd (T,), rev (T,),
    info, centers).  Mirrors app._update_directed / the selftest exactly."""
    src = SequenceSource(cfg, standard, deviant, seconds=seconds, seed=seed)
    y, info = src._y, src.info
    fe = SpectroFrontEnd(cfg)
    eng = LiveEngine(cfg.to_a1_config(), learn=cfg.learn, seed=0)
    g = float(np.exp(-cfg.dt / max(cfg.forget_s, 1e-3)))
    N = cfg.n_channels
    D = np.zeros((N, N))
    idx = np.arange(N)
    E_cols, fwd, rev = [], [], []
    bs = cfg.blocksize
    for lo in range(0, y.size, bs):
        d, _ = fe.push(y[lo:lo + bs])
        if not d.shape[1]:
            continue
        out = eng.step_block(d)
        E, tr, k = out["E"], out["tr"], out["E"].shape[1]
        E_cols.append(E.astype(np.float32))
        w = (1.0 - g) * g ** (k - 1 - np.arange(k))
        D = g ** k * D + (E * w) @ tr.T
        Dz = D.copy(); np.fill_diagonal(Dz, 0.0)
        Dz[np.abs(idx[:, None] - idx[None, :]) <= 3] = 0.0   # cross-tone only
        Dl = Dz - Dz.T
        Dhat = Dl / (np.linalg.norm(Dl) + 1e-9)
        fwd.extend((E * (np.maximum(Dhat, 0.0) @ tr)).sum(0).tolist())
        rev.extend((E * (np.maximum(-Dhat, 0.0) @ tr)).sum(0).tolist())
    E = np.concatenate(E_cols, axis=1)
    return E, np.asarray(fwd), np.asarray(rev), info, fe.center_freqs()


def export(name, standard, deviant, cfg, outdir, seconds, seed, decimate):
    E, fwd, rev, info, centers = _run(cfg, standard, deviant, seconds, seed)
    N, T = E.shape
    if decimate > 1:                                  # optional downsample
        E = E[:, ::decimate]; fwd = fwd[::decimate]; rev = rev[::decimate]
        T = E.shape[1]
    t = np.arange(T) * cfg.dt * decimate

    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)

    hdr = "time_s," + ",".join(f"ch{i + 1:02d}_{centers[i]:.0f}Hz" for i in range(N))
    np.savetxt(d / "response_E.csv", np.column_stack([t, E.T]),
               delimiter=",", header=hdr, comments="", fmt="%.5g")

    np.savetxt(d / "signals.csv",
               np.column_stack([t, E.sum(0), fwd, rev, fwd - rev]),
               delimiter=",", header="time_s,pop_E,fwd_energy,rev_energy,net_flow",
               comments="", fmt="%.5g")

    with open(d / "events.csv", "w") as f:
        f.write("onset_s,is_deviant,sequence\n")
        for is_dev, t0 in info["events"]:
            seq = _seq_str(deviant if is_dev else standard)
            f.write(f"{t0:.4f},{int(is_dev)},{seq}\n")

    with open(d / "meta.csv", "w") as f:
        f.write("key,value\n")
        rows = [("paradigm", name), ("standard", _seq_str(standard)),
                ("deviant", _seq_str(deviant)), ("sr_hz", cfg.sr),
                ("dt_s", cfg.dt * decimate), ("n_channels", N),
                ("tau_trace_s", cfg.tau_trace), ("forget_s", cfg.forget_s),
                ("seconds", seconds), ("seed", seed),
                ("tone_freqs_hz", "|".join(f"{x:.1f}" for x in info["freqs"])),
                ("lead_s", info["lead_s"]), ("active_s", info["active_s"]),
                ("period_s", info["period_s"])]
        for k, v in rows:
            f.write(f"{k},{v}\n")
    return T, len(info["events"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Export model responses (CSV) for the directional experiments.")
    ap.add_argument("--paradigms", nargs="+", default=list(_SEQ_PARADIGMS),
                    choices=list(_SEQ_PARADIGMS), help="which experiments")
    ap.add_argument("--outdir", default="responses")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="duration per experiment (more = more events for stats)")
    ap.add_argument("--tau-trace", type=float, default=0.15, dest="tau_trace",
                    help="eligibility-trace time constant (s); 0.15 = the 150 ms "
                         "dynamic")
    ap.add_argument("--forget", type=float, default=None,
                    help="leaky directed-coincidence horizon (s)")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep every k-th frame in response_E/signals "
                         "(1 = full 1 kHz; 5 = 200 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    overrides = {"tau_trace": args.tau_trace}
    if args.forget:
        overrides["forget_s"] = args.forget
    cfg = get_preset("directional", **overrides)

    out = Path(args.outdir)
    print(f"[ export_responses ] N={cfg.n_channels} ch · sr={cfg.sr} · "
          f"tau_trace={cfg.tau_trace}s · forget={cfg.forget_s}s · {args.seconds:.0f}s each")
    for name in args.paradigms:
        std, dev = _SEQ_PARADIGMS[name]
        T, ne = export(name, std, dev, cfg, out, args.seconds, args.seed,
                       args.decimate)
        print(f"  {name:8s} {_seq_str(std)} vs {_seq_str(dev)}: "
              f"{T} frames x {cfg.n_channels} ch, {ne} events -> {out}/{name}/")
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
