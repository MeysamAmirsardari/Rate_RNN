"""
live_demo_cortical.export
=========================

Render long **two-stream tone mixtures** to MP3 (or WAV) -- the inputs the live
segregation demo pulls apart.  Each scene is two temporally-coherent tone
groups (plus an incoherent background) that fire on independent schedules, so
they can be separated ONLY by temporal coherence -- not by frequency band or by
pitch (the tones are pure).  The scenes differ in how the two streams sit in
frequency and how fast they fire:

    interleave   the two streams alternate across the SAME frequency range --
                 frequency alone cannot separate them (the compelling case)
    bands        a low stream and a high stream (classic two-band streaming)
    galloping    interleaved, but one stream fast and one slow (a "gallop")
    sparse       two slow, widely-spaced streams over a quiet background

Play a file to hear the two streams, or watch one segregate in real time:

    python -m live_demo_cortical --source wav --wav demo_audio/two_stream_interleave.mp3

Run
    python -m live_demo_cortical.export                 # all scenes -> ./demo_audio
    python -m live_demo_cortical.export --seconds 60 --format wav
    python -m live_demo_cortical.export --only interleave bands
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live_demo_cortical.config import get_preset
from live_demo_cortical.audio import synth_two_stream
from audio.generate import write_audio

# (scene -> synth_two_stream kwargs).  Distinct seeds so the scenes don't share
# the same tone layout.
SCENES = {
    "interleave": dict(layout="interleave", rate_a=5.0, rate_b=5.0, seed=1),
    "bands":      dict(layout="bands",      rate_a=5.0, rate_b=5.0, seed=2),
    "galloping":  dict(layout="interleave", rate_a=8.0, rate_b=3.0, seed=3),
    "sparse":     dict(layout="bands",      rate_a=3.0, rate_b=3.0, p_bg=0.04,
                       seed=4),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render two-stream demo audio.")
    ap.add_argument("--seconds", type=float, default=45.0, help="clip length (s)")
    ap.add_argument("--format", choices=["mp3", "wav"], default="mp3")
    ap.add_argument("--only", nargs="+", choices=list(SCENES),
                    default=list(SCENES), help="render only these scenes")
    ap.add_argument("--outdir",
                    default=str(Path(__file__).resolve().parent / "demo_audio"))
    args = ap.parse_args(argv)

    cfg = get_preset("default")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        y, sr, (gA, gB) = synth_two_stream(
            cfg.sr, seconds=args.seconds, fmin=cfg.fmin, fmax=cfg.fmax,
            n_pool=cfg.n_channels, **SCENES[name])
        path = write_audio(out / f"two_stream_{name}.{args.format}", y, sr)
        print(f"  wrote {path.name:32s} {y.size / sr:5.1f} s  "
              f"|A|={len(gA)} |B|={len(gB)} @ {sr} Hz")
    print(f"Done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
