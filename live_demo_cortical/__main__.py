"""
live_demo_cortical.__main__
==================

Entry point for the live stream-segregation demo.

    python -m live_demo_cortical                              live microphone
    python -m live_demo_cortical --source twostream           two coherent tone streams
    python -m live_demo_cortical --source abba                AB-BA directional (order)
    python -m live_demo_cortical --source abcacb              ABC-ACB directional (order)
    python -m live_demo_cortical --source abcacb --tau-trace 0.15   longer-timescale trace
    python -m live_demo_cortical --source abcacb --preset segregate  unsupervised k-stream
                                                     clustering (balanced, any paradigm)
    python -m live_demo_cortical --source abc_cab            ABC vs CAB  (rotation)
    python -m live_demo_cortical --source abc_cba            ABC vs CBA  (reversal)
    python -m live_demo_cortical --source ab_ac              AB vs AC    (feature swap)
    python -m live_demo_cortical --source ac_bc              AC vs BC    (feature swap)
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
                                      SFGSource, TwoStreamSource, ABBASource,
                                      ABCACBSource, SequenceSource)

# Directional ordered-sequence paradigms: source name -> (standard, deviant)
# index tuples into the (3-tone) pool.  ORDER manipulations (CAB/CBA rotations
# and reversals) are flagged by the order-violation read-out; the last two swap
# a tone for a DIFFERENT one (a feature change, not an order change).
_SEQ_PARADIGMS = {
    "abc_cab": ((0, 1, 2), (2, 0, 1)),   # ABC vs CAB (cyclic rotation)
    "abc_cba": ((0, 1, 2), (2, 1, 0)),   # ABC vs CBA (full reversal)
    "ab_ac":   ((0, 1), (0, 2)),         # AB vs AC  (2nd tone swapped: feature)
    "ac_bc":   ((0, 2), (1, 2)),         # AC vs BC  (1st tone swapped: feature)
}
# every directional paradigm auto-selects the 'directional' preset
_DIRECTIONAL_SOURCES = ("abba", "abcacb", *_SEQ_PARADIGMS)
# all ordered-sequence paradigms as (standard, deviant) -- for the 'segregate'
# mode every paradigm is one balanced (50-50) two-class SequenceSource
_ALL_SEQ = {"abba": ((0, 1), (1, 0)), "abcacb": ((0, 1, 2), (0, 2, 1)),
            **_SEQ_PARADIGMS}


def _directional_source(cfg: LiveConfig, name: str, **kw):
    """Build a directional stimulus source by name."""
    if name == "abba":
        return ABBASource(cfg, **kw)
    if name == "abcacb":
        return ABCACBSource(cfg, **kw)
    std, dev = _SEQ_PARADIGMS[name]
    return SequenceSource(cfg, std, dev, **kw)


# ---------------------------------------------------------------------
def _build_source(cfg: LiveConfig, args):
    # 'segregate' mode: any ordered-sequence paradigm becomes a BALANCED 50-50
    # two-class stream (no standard/deviant) for unsupervised clustering.
    if cfg.mode == "segregate" and args.source in _ALL_SEQ:
        std, dev = _ALL_SEQ[args.source]
        return SequenceSource(cfg, std, dev, p_dev=cfg.dev_prob)
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
    if args.source in _DIRECTIONAL_SOURCES:
        return _directional_source(cfg, args.source)
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
def run_selftest_directional(cfg: LiveConfig, preview_path: str,
                             source: str = "abba") -> int:
    """Validate the DIRECTIONAL mode: the order-VIOLATION energy must flag each
    deviant -- B→A among AB-BA pairs, or A→C→B among ABC-ACB triplets.  The
    discriminator is the reverse (against-template) coincidence energy, not the
    net flow (which the common leads dominate)."""
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from live_demo_cortical.app import LiveDemoApp
    from live_demo_cortical.audio import SpectroFrontEnd
    from live_demo_cortical.engine import LiveEngine

    tag = source.upper().replace("_", "-")
    print(f"[ live_demo_cortical self-test — directional ({tag}) ]")
    print(f"  {cfg.n_channels} channels · sr={cfg.sr} · forget={cfg.forget_s:.1f}s"
          f" · order-violation energy of D=⟨E·tr⟩ on activations")
    secs = max(cfg.history_s + 8.0, 30.0)
    src = _directional_source(cfg, source, seconds=secs, seed=0)
    info, y = src.info, src._y

    # standalone pass -> per-event forward / violation energy (mirrors the app)
    fe = SpectroFrontEnd(cfg)
    eng = LiveEngine(cfg.to_a1_config(), learn=cfg.learn, seed=0)
    g, N = float(np.exp(-cfg.dt / max(cfg.forget_s, 1e-3))), cfg.n_channels
    D = np.zeros((N, N)); rev = []; bs = cfg.blocksize
    t0 = time.perf_counter()
    for lo in range(0, y.size, bs):
        d, _ = fe.push(y[lo:lo + bs])
        if not d.shape[1]:
            continue
        out = eng.step_block(d); E, tr, k = out["E"], out["tr"], out["E"].shape[1]
        w = (1.0 - g) * g ** (k - 1 - np.arange(k))
        D = g ** k * D + (E * w) @ tr.T
        Dz = D.copy(); np.fill_diagonal(Dz, 0.0)
        idx = np.arange(N)
        Dz[np.abs(idx[:, None] - idx[None, :]) <= 3] = 0.0   # cross-tone only
        Dl = Dz - Dz.T; Dhat = Dl / (np.linalg.norm(Dl) + 1e-9)
        Dm = np.maximum(-Dhat, 0.0)
        rev.extend((E * (Dm @ tr)).sum(0).tolist())
    rt = time.perf_counter() - t0
    speed = (y.size / cfg.sr) / rt if rt > 0 else float("inf")
    rev = np.asarray(rev)
    # peak violation energy in each event's NON-first-tone region (where an
    # order swap shows); deviants are the outliers above the standard baseline.
    lead_s = info.get("lead_s", 0.07)
    active_s = info.get("active_s", info["period_s"])
    peaks, labs = [], []
    for is_dev, ts in info["events"]:
        a = int((ts + lead_s) * 1000); b = int((ts + active_s) * 1000)
        if b >= rev.size:
            continue
        peaks.append(float(rev[a:b].max())); labs.append(bool(is_dev))
    peaks = np.asarray(peaks); labs = np.asarray(labs, dtype=bool)
    med = float(np.median(peaks))
    mad = float(np.median(np.abs(peaks - med))) + 1e-9
    pred = peaks > med + 3.0 * 1.4826 * mad         # deviant = violation outlier
    tp = float(np.mean(pred[labs])) if labs.any() else 0.0        # deviants caught
    tn = float(np.mean(~pred[~labs])) if (~labs).any() else 0.0   # standards kept
    bal = 0.5 * (tp + tn)                            # balanced accuracy (honest)
    tot = int(peaks.size)

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")

    check("finite energies", bool(np.isfinite(rev).all() and np.isfinite(peaks).all()))
    check("real-time capable", speed > 1.0, f"({speed:.1f}x real time)")
    check("deviant detected (order violation)", bal >= 0.8,
          f"(balanced {100 * bal:.0f}%: {100 * tp:.0f}% deviants caught, "
          f"{100 * (1 - tn):.0f}% false alarms; N={tot})")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = LiveDemoApp(cfg, src)
    win.show()
    keep = int((cfg.history_s + 8.0) * cfg.sr)
    win.feed_offline(y[-keep:] if y.size > keep else y)
    win.grab_image(preview_path)
    print(f"  preview saved -> {preview_path}")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_selftest(cfg: LiveConfig, preview_path: str, source: str = "abba") -> int:
    """Drive the segregation pipeline headlessly and validate it: the two
    coherent streams are recovered (coherence mode), or the deviant order is
    flagged (directional mode).  Saves the GUI as the preview PNG."""
    if cfg.mode == "directional":
        return run_selftest_directional(cfg, preview_path, source)
    if cfg.mode == "segregate":
        from live_demo_cortical.segregate import main as _seg_selftest
        return _seg_selftest([])           # the all-paradigm clustering table
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
    # feed the chosen source (mic falls back to pseudo-speech for a static frame).
    # In directional mode, feed extra lead-in so the directed map D is warm
    # before the displayed window.
    if args.source != "mic":
        feed_src = _build_source(cfg, args)
        if hasattr(feed_src, "start"):
            feed_src.start()
        n = keep + (int(8.0 * cfg.sr)
                    if cfg.mode in ("directional", "segregate") else 0)
        audio = feed_src.read(n)
    else:
        audio = _pseudo_speech(cfg)
    win.feed_offline(audio)
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
                    help="config preset: default | uniform | frozen | dynamic | "
                         "dynamic2 | directional")
    ap.add_argument("--source", default="mic",
                    choices=["mic", "wav", "synthetic", "sfg", "twostream",
                             "abba", "abcacb", "abc_cab", "abc_cba",
                             "ab_ac", "ac_bc"])
    ap.add_argument("--wav", default=None, help="WAV path for --source wav")
    ap.add_argument("--device", type=int, default=None,
                    help="sounddevice input device index")
    ap.add_argument("--inhibition", default=None,
                    choices=["selective", "uniform"])
    ap.add_argument("--no-learn", action="store_true",
                    help="disable Hebbian plasticity")
    ap.add_argument("--history", type=float, default=None,
                    help="seconds of scrolling history")
    ap.add_argument("--tau-trace", type=float, default=None, dest="tau_trace",
                    help="eligibility-trace time constant (s) -- the directional "
                         "read-out's lead→lag window (abba uses 0.05; try 0.15 "
                         "for a longer-timescale trace)")
    ap.add_argument("--forget", type=float, default=None,
                    help="leaky directed-coincidence forgetting horizon (s) "
                         "-- the flow trace's scene memory (abba uses 3.0)")
    ap.add_argument("--selftest", action="store_true",
                    help="headless validation + preview PNG")
    ap.add_argument("--snapshot", default=None,
                    help="render a headless GUI screenshot to this path")
    ap.add_argument("--preview", default="live_demo_cortical_preview.png",
                    help="preview PNG path for --selftest")
    args = ap.parse_args(argv)

    # the AB-BA / ABC-ACB stimuli are directional paradigms -> default preset
    if args.source in _DIRECTIONAL_SOURCES and args.preset == "default":
        args.preset = "directional"

    overrides = {}
    if args.inhibition:
        overrides["inhibition"] = args.inhibition
    if args.no_learn:
        overrides["learn"] = False
    if args.history:
        overrides["history_s"] = args.history
    if args.tau_trace:
        overrides["tau_trace"] = args.tau_trace
    if args.forget:
        overrides["forget_s"] = args.forget
    cfg = get_preset(args.preset, **overrides)

    if args.selftest:
        return run_selftest(cfg, args.preview, args.source)
    if args.snapshot:
        return run_snapshot(cfg, args)
    return run_live(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
