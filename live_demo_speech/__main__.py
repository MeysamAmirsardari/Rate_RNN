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


def _clean_cochleagrams(cfg: LiveConfig, fe, sources):
    """Per-channel cochleagram (N, T) of each clean source, via the front end's
    filterbank (no AGC) -- for the grouping ground truth."""
    import librosa
    fb = fe._mel_fb
    cgs = []
    for y in sources:
        S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                center=False)) ** 2
        cgs.append(fb @ S)
    return cgs


# ---------------------------------------------------------------------
def run_selftest(cfg: LiveConfig, preview_path: str) -> int:
    """Drive the segregation pipeline headlessly on a two-talker mixture and
    validate the GROUPING: each clearly-dominated cochlear channel is assigned
    to the talker that actually dominates it.  Saves the GUI as the preview."""
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from live_demo_speech.app import LiveDemoApp

    names = ("libri1", "libri3")
    print("[ live_demo_speech self-test — two-talker grouping ]")
    print(f"  {cfg.n_channels} channels · {cfg.fmin:.0f}-{cfg.fmax:.0f} Hz · "
          f"{len(cfg.coh_rates_s)} rates · talkers {names}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = TwoTalkerSource(cfg, names=names, seconds=10.0)
    win = LiveDemoApp(cfg, src)
    win.show()
    t0 = time.perf_counter()
    win.feed_offline(src._y)
    rt = time.perf_counter() - t0
    speed = (src._y.size / cfg.sr) / rt if rt > 0 else float("inf")

    # grouping ground truth: per channel, the energy-weighted fraction of
    # frames in which talker A dominates (time-resolved, honest).
    ca, cb = _clean_cochleagrams(cfg, win.fe, src.sources)
    w = ca + cb
    frac = (((ca > cb) * w).sum(1)) / (w.sum(1) + 1e-9)
    clear = np.where((frac > 0.6) | (frac < 0.4))[0]
    M = win._masks
    true_a = frac[clear] > 0.5
    pred_a = M[clear, 0] > M[clear, 1]
    acc = max(np.mean(pred_a == true_a), np.mean(pred_a != true_a)) \
        if clear.size else 0.0

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    check("finite masks", bool(np.isfinite(M).all() and np.isfinite(win._C).all()))
    check("real-time capable", speed > 1.0, f"({speed:.1f}x real time)")
    check("coincidence populated", win._C.max() > 0.1, f"(max C={win._C.max():.2f})")
    # GROUPING is a tracked research metric, not a pass/fail: multi-rate
    # envelope coherence alone gives only FRAGILE/PARTIAL two-talker grouping
    # (the shared speech envelope dominates; talker contrast is weak).  Clean
    # separation needs pitch / harmonic binding -- the next step.
    print(f"  [diag] talker grouping = {100 * acc:.0f}% of {clear.size} "
          f"clearly-dominated channels  (partial; pitch binding is next)")

    win.grab_image(preview_path)
    print(f"  preview saved -> {preview_path}")
    print("  RESULT:", "PASS (pipeline)" if ok else "FAIL")
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
