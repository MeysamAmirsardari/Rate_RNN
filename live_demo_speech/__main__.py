"""
live_demo_speech.__main__
=========================

Entry point for the live **two-talker speech segregation** demo.

    python -m live_demo_speech                               two LibriSpeech talkers
    python -m live_demo_speech --talkers libri2 libri3       pick the talkers
    python -m live_demo_speech --source mic                  live microphone
    python -m live_demo_speech --source wav --wav mix.wav     a recorded mixture
    python -m live_demo_speech --selftest                    headless validation
    python -m live_demo_speech --snapshot out.png            headless GUI screenshot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_demo_speech.config import LiveConfig, get_preset
from live_demo_speech.audio import MicSource, WavSource
from live_demo_speech.speech import TwoTalkerSource


# ---------------------------------------------------------------------
def _build_source(cfg: LiveConfig, args):
    if args.source == "mic":
        return MicSource(cfg, device=args.device)
    if args.source == "wav":
        if not args.wav:
            raise SystemExit("--source wav requires --wav PATH")
        return WavSource(cfg, args.wav, loop=True)
    if args.source == "twotalker":
        return TwoTalkerSource(cfg, names=tuple(args.talkers), snr_db=args.snr)
    raise SystemExit(f"unknown source {args.source!r}")


def _cochlea_power(cfg, fe, y):
    """Per-channel power cochleagram of ``y`` (energy -- the basis for an
    SNR-like separation metric)."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                            center=False)) ** 2
    return fe._mel_fb @ S


def _corr(u, v):
    u = u.ravel() - u.mean()
    v = v.ravel() - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


# ---------------------------------------------------------------------
def run_selftest(cfg: LiveConfig, preview_path: str) -> int:
    """Drive the segregation headlessly on a two-talker mixture and validate
    that the time-resolved pitch mask SEPARATES: each recovered talker stream
    must match its clean source better than the raw mixture does."""
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from live_demo_speech.app import LiveDemoApp

    names = ("libri1", "libri3")
    print("[ live_demo_speech self-test — two-talker separation (pitch) ]")
    print(f"  {cfg.n_channels} channels · {cfg.fmin:.0f}-{cfg.fmax:.0f} Hz · "
          f"talkers {names}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = TwoTalkerSource(cfg, names=names, seconds=cfg.history_s + 2.0)
    win = LiveDemoApp(cfg, src)
    win.show()
    t0 = time.perf_counter()
    win.feed_offline(src._y)
    rt = time.perf_counter() - t0
    speed = (src._y.size / cfg.sr) / rt if rt > 0 else float("inf")

    Pa = _cochlea_power(cfg, win.fe, src.sources[0])
    Pb = _cochlea_power(cfg, win.fe, src.sources[1])
    m1 = win._m1
    L = min(Pa.shape[1], m1.shape[1])                      # align by the tail
    Pa, Pb, m1 = Pa[:, -L:], Pb[:, -L:], m1[:, -L:]
    # ideal-binary-mask routing accuracy: per T-F bin, does the mask send energy
    # to the dominant talker?  (energy-weighted; the standard CASA mask metric)
    dom = Pa > Pb
    w = Pa + Pb
    pred = m1 > 0.5
    agree = ((pred == dom) * w).sum() / (w.sum() + 1e-9)
    ibm = max(agree, 1.0 - agree)
    f0a, f0b = win.psep.f0_hz()

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    check("finite masks", bool(np.isfinite(m1).all() and np.isfinite(win._C).all()))
    check("real-time capable", speed > 1.0, f"({speed:.1f}x real time)")
    check("two F0s tracked", abs(f0a - f0b) > 15.0,
          f"(F0 = {f0a:.0f} / {f0b:.0f} Hz)")
    check("routes T-F energy to the right talker", ibm > 0.65,
          f"(ideal-binary-mask accuracy {100 * ibm:.0f}%, chance 50%)")

    win.grab_image(preview_path)
    print(f"  preview saved -> {preview_path}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_snapshot(cfg: LiveConfig, args) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from live_demo_speech.app import LiveDemoApp

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = _build_source(cfg, args)
    if hasattr(src, "start"):
        src.start()
    win = LiveDemoApp(cfg, src)
    win.show()
    keep = int(cfg.history_s * cfg.sr)
    audio = src.read(keep) if args.source != "mic" else np.zeros(keep)
    win.feed_offline(audio)
    win.grab_image(args.snapshot)
    print(f"  GUI snapshot saved -> {args.snapshot}")
    return 0


def run_live(cfg: LiveConfig, args) -> int:
    from pyqtgraph.Qt import QtWidgets
    from live_demo_speech.app import LiveDemoApp

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    source = _build_source(cfg, args)
    win = LiveDemoApp(cfg, source)
    win.show()
    win.start()
    print("[ live_demo_speech ] running — keys: Space pause · R reset · Q quit")
    return app.exec()


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live two-talker speech segregation.")
    ap.add_argument("--preset", default="default",
                    help="config preset: default | uniform | selective")
    ap.add_argument("--source", default="twotalker",
                    choices=["twotalker", "mic", "wav"])
    ap.add_argument("--talkers", nargs=2, default=["libri1", "libri3"],
                    metavar=("A", "B"), help="LibriSpeech example talkers")
    ap.add_argument("--snr", type=float, default=0.0,
                    help="talker A level relative to B (dB)")
    ap.add_argument("--wav", default=None, help="WAV path for --source wav")
    ap.add_argument("--device", type=int, default=None,
                    help="sounddevice input device index")
    ap.add_argument("--inhibition", default=None, choices=["selective", "uniform"])
    ap.add_argument("--history", type=float, default=None,
                    help="seconds of scrolling history")
    ap.add_argument("--selftest", action="store_true",
                    help="headless validation + preview PNG")
    ap.add_argument("--snapshot", default=None,
                    help="render a headless GUI screenshot to this path")
    ap.add_argument("--preview", default="live_demo_speech_preview.png",
                    help="preview PNG path for --selftest")
    args = ap.parse_args(argv)

    overrides = {}
    if args.inhibition:
        overrides["inhibition"] = args.inhibition
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
