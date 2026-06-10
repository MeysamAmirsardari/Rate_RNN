"""live_demo_cortical -- real-time microphone demo of the model0 A1 rate RNN.

An "artificial auditory cortex listening through a cochlea": microphone ->
streaming cochleo-mel front end (AGC + noise gate) -> model0 A1 RNN ->
scrolling input/cortex/population display.

    python -m live_demo_cortical                 # live, microphone
    python -m live_demo_cortical --source synthetic
    python -m live_demo_cortical --source wav --wav path/to/file.wav
    python -m live_demo_cortical --selftest      # headless validation + preview PNG
"""

from .config import LiveConfig, get_preset, PRESETS

__all__ = ["LiveConfig", "get_preset", "PRESETS"]
