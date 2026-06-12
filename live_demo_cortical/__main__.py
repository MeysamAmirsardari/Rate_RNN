"""
live_demo_cortical.__main__
==================

Entry point for the live stream-segregation demo.

    python -m live_demo_cortical                              live microphone
    python -m live_demo_cortical --source twostream           two coherent tone streams
    python -m live_demo_cortical --source sfg                 stochastic figure-ground
    python -m live_demo_cortical --source synthetic           mic-free tone bursts
    python -m live_demo_cortical --source wav --wav a.wav     play a recording
    python -m live_demo_cortical --selftest                   headless validation
    python -m live_demo_cortical --snapshot out.png --source twostream
                                                     headless GUI screenshot
    python -m live_demo_cortical.export                       render two-stream MP3s
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
from live_demo_cortical.audio import (MicSource, WavSource, SyntheticSource,
                                      SFGSource, TwoStreamSource)


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
    if args.source == "sfg":
        return SFGSource(cfg)
    if args.source == "twostream":
        return TwoStreamSource(cfg)
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
    """Drive the segregation pipeline headlessly on a two-stream scene and
    validate that the two coherent streams are recovered (and save the GUI)."""
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from live_demo_cortical.app import LiveDemoApp

    print("[ live_demo_cortical self-test — stream segregation ]")
    print(f"  {cfg.n_channels} channels · sr={cfg.sr} · "
          f"{cfg.fmin:.0f}–{cfg.fmax:.0f} Hz · {cfg.coh_window_s:.0f}s "
          f"coincidence window · {cfg.n_streams} streams")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    secs = max(cfg.history_s, cfg.coh_window_s + 4.0)
    src = TwoStreamSource(cfg, seconds=secs + 2.0, seed=0)
    win = LiveDemoApp(cfg, src)
    win.show()
    src.start()
    audio = src.read(int(secs * cfg.sr))
    t0 = time.perf_counter()
    win.feed_offline(audio)
    rt = time.perf_counter() - t0
    speed = secs / rt if rt > 0 else float("inf")

    # tone-level separation score vs the known groups: each tone's main channel
    # should be claimed by one mask, with group A in one mask and B in the other.
    M = win._masks
    pool = np.geomspace(cfg.fmin, cfg.fmax, cfg.n_channels)
    wfft, fb = np.hanning(cfg.n_fft), win.fe._mel_fb
    def tone_chan(f):
        x = np.sin(2 * np.pi * f * np.arange(cfg.n_fft) / cfg.sr) * wfft
        return int((fb @ (np.abs(np.fft.rfft(x)) ** 2)).argmax())
    def label(c):
        k = int(M[c].argmax())
        return k if M[c, k] > 0.5 else -1
    gA = [int(i) for i in src.groups[0]]
    gB = [int(i) for i in src.groups[1]]
    lA = [label(tone_chan(pool[i])) for i in gA]
    lB = [label(tone_chan(pool[i])) for i in gB]
    best = max(sum(l == a for l in lA) + sum(l == b for l in lB)
               for a, b in ((0, 1), (1, 0)))
    score = best / (len(gA) + len(gB))

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    check("finite outputs",
          bool(np.isfinite(M).all() and np.isfinite(win._C).all()))
    check("real-time capable", speed > 1.0, f"({speed:.1f}x real time)")
    check("coincidence populated", win._C.max() > 0.1,
          f"(max C={win._C.max():.2f})")
    check("streams separated", score >= 0.8,
          f"({score * 100:.0f}% tones correctly grouped)")

    win.grab_image(preview_path)
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
    keep = int(cfg.history_s * cfg.sr)
    # feed the chosen source (mic falls back to pseudo-speech for a static frame)
    if args.source in ("twostream", "sfg", "synthetic", "wav"):
        feed_src = _build_source(cfg, args)
        if hasattr(feed_src, "start"):
            feed_src.start()
        audio = feed_src.read(keep)
    else:
        audio = _pseudo_speech(cfg)
    # keep only the most recent history window so the frame is full
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
    print("[ live_demo_cortical ] running — keys: "
          "Space pause · R reset · Q quit")
    return app.exec()


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live model0 A1 demo.")
    ap.add_argument("--preset", default="default",
                    help="config preset: default | uniform | frozen | dynamic | dynamic2")
    ap.add_argument("--source", default="mic",
                    choices=["mic", "wav", "synthetic", "sfg", "twostream"])
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
