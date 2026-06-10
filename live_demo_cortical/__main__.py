"""
live_demo_cortical.__main__
==================

Entry point for the live model0 demo.

    python -m live_demo_cortical                              live microphone
    python -m live_demo_cortical --preset frozen              plasticity off
    python -m live_demo_cortical --preset dynamic             faster learning/forgetting
    python -m live_demo_cortical --preset dynamic2            ~10 s forgetting horizon
    python -m live_demo_cortical --inhibition uniform         control inhibition
    python -m live_demo_cortical --source synthetic           mic-free tone bursts
    python -m live_demo_cortical --source wav --wav a.wav     play a recording
    python -m live_demo_cortical --selftest                   headless validation
    python -m live_demo_cortical --snapshot out.png --source wav --wav a.wav
                                                     headless GUI screenshot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# project root on sys.path for `python live_demo_cortical/__main__.py` too
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_demo_cortical.config import LiveConfig, get_preset
from live_demo_cortical.audio import MelFrontEnd, MicSource, WavSource, SyntheticSource
from live_demo_cortical.engine import LiveEngine


# ---------------------------------------------------------------------
def _build_source(cfg: LiveConfig, args):
    if args.source == "mic":
        return MicSource(cfg, device=args.device)
    if args.source == "wav":
        if not args.wav:
            raise SystemExit("--source wav requires --wav PATH")
        return WavSource(cfg, args.wav, loop=True)
    if args.source == "synthetic":
        return SyntheticSource(cfg)
    raise SystemExit(f"unknown source {args.source!r}")


def _pseudo_speech(cfg: LiveConfig) -> np.ndarray:
    """Concatenate the five auditory-task syllables with gaps (offline)."""
    import librosa
    wav_dir = _ROOT / "tasks" / "syllables_wav"
    stems = ["boo", "pee", "tah", "bay", "see"]
    gap = np.zeros(int(0.25 * cfg.sr))
    parts = []
    for _ in range(2):
        for st in stems:
            p = wav_dir / f"{st}.wav"
            if p.exists():
                y, _sr = librosa.load(str(p), sr=cfg.sr, mono=True)
                parts += [y.astype(np.float64), gap.copy()]
    if not parts:
        # fall back to synthetic if the wavs are unavailable
        return SyntheticSource(cfg).read(int(4.0 * cfg.sr))
    return np.concatenate(parts)


# ---------------------------------------------------------------------
def run_selftest(cfg: LiveConfig, preview_path: str) -> int:
    """Drive the full pipeline headlessly on pseudo-speech and validate."""
    print("[ live_demo_cortical self-test ]")
    print(f"  {cfg.n_channels} channels · sr={cfg.sr} · hop={cfg.hop_length} "
          f"samp ({cfg.hop_ms} ms/frame) · inhibition={cfg.inhibition} · "
          f"learn={cfg.learn}")
    fe = MelFrontEnd(cfg)
    eng = LiveEngine(cfg.to_a1_config(), learn=cfg.learn, seed=0)
    audio = _pseudo_speech(cfg)

    import time
    bs = cfg.blocksize
    db_cols, E_cols = [], []
    n_open = n_tot = 0
    t0 = time.perf_counter()
    for lo in range(0, audio.size, bs):
        drive, db = fe.push(audio[lo:lo + bs])
        if drive.shape[1] == 0:
            continue
        out = eng.step_block(drive)
        db_cols.append(db)
        E_cols.append(out["E"])
        n_tot += drive.shape[1]
        n_open += int((drive.max(axis=0) > 0).sum())
    rt = time.perf_counter() - t0
    db = np.concatenate(db_cols, axis=1)
    E = np.concatenate(E_cols, axis=1)

    speed = (n_tot / 1000.0) / rt if rt > 0 else float("inf")
    act = E[:, E.sum(0) > 0]
    spars = 100.0 * (act > 0.05 * E.max()).mean() if act.size else 0.0

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    check("finite outputs", bool(np.isfinite(E).all() and np.isfinite(db).all()))
    check("real-time capable", speed > 1.0, f"({speed:.1f}x real time)")
    check("noise gate engaged", 0 < n_open < n_tot,
          f"({100 * n_open / n_tot:.0f}% frames open)")
    check("model bounded", 0 < E.max() < 50.0, f"(peak E={E.max():.2f})")
    check("sparse cortical code", spars < 40.0,
          f"({spars:.1f}% channels active)")

    from live_demo_cortical.preview import render_static
    render_static(cfg, db, E, fe, preview_path,
                  title=f"Artificial auditory cortex — model0 "
                        f"[{cfg.inhibition} inhibition, learn={cfg.learn}]")
    print(f"  preview saved -> {preview_path}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_snapshot(cfg: LiveConfig, args) -> int:
    """Render a screenshot of the real GUI headlessly (offscreen Qt)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from live_demo_cortical.app import LiveDemoApp

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = SyntheticSource(cfg)            # source unused; we feed offline
    win = LiveDemoApp(cfg, src)
    win.show()
    audio = (_pseudo_speech(cfg) if args.source != "synthetic"
             else SyntheticSource(cfg).read(int(cfg.history_s * cfg.sr)))
    # keep only the most recent history window so the frame is full
    keep = int(cfg.history_s * cfg.sr)
    win.feed_offline(audio[-keep:] if audio.size > keep else audio)
    app.processEvents()
    win.grab_image(args.snapshot)
    print(f"  GUI snapshot saved -> {args.snapshot}")
    return 0


def run_live(cfg: LiveConfig, args) -> int:
    from pyqtgraph.Qt import QtWidgets
    from live_demo_cortical.app import LiveDemoApp

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    source = _build_source(cfg, args)
    win = LiveDemoApp(cfg, source)
    win.show()
    win.start()
    print("[ live_demo_cortical ] running — keys: C cortex · L learn · "
          "I inhibition · Space pause · R reset · Q quit")
    return app.exec()


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live model0 A1 demo.")
    ap.add_argument("--preset", default="default",
                    help="config preset: default | uniform | frozen | dynamic | dynamic2")
    ap.add_argument("--source", default="mic",
                    choices=["mic", "wav", "synthetic"])
    ap.add_argument("--wav", default=None, help="WAV path for --source wav")
    ap.add_argument("--device", type=int, default=None,
                    help="sounddevice input device index")
    ap.add_argument("--inhibition", default=None,
                    choices=["selective", "uniform"])
    ap.add_argument("--no-learn", action="store_true",
                    help="disable Hebbian plasticity")
    ap.add_argument("--history", type=float, default=None,
                    help="seconds of scrolling history")
    ap.add_argument("--selftest", action="store_true",
                    help="headless validation + preview PNG")
    ap.add_argument("--snapshot", default=None,
                    help="render a headless GUI screenshot to this path")
    ap.add_argument("--preview", default="live_demo_cortical_preview.png",
                    help="preview PNG path for --selftest")
    args = ap.parse_args(argv)

    overrides = {}
    if args.inhibition:
        overrides["inhibition"] = args.inhibition
    if args.no_learn:
        overrides["learn"] = False
    if args.history:
        overrides["history_s"] = args.history
    cfg = get_preset(args.preset, **overrides)

    if args.selftest:
        return run_selftest(cfg, args.preview)
    if args.snapshot:
        return run_snapshot(cfg, args)
    return run_live(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
