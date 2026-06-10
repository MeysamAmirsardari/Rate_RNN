"""
live_demo_cortical.preview
=================

Static (matplotlib) render of a stretch of the live pipeline -- the
thalamic input spectrogram, the A1 cortical response, and the population
traces, time-aligned.  Used by the headless self-test (``python -m
live_demo_cortical --selftest``) as a validation artifact, and as a no-Qt
fallback visualisation of a recorded/looped stream.

The live, interactive view is ``live_demo_cortical.app`` (pyqtgraph).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .audio import MelFrontEnd
from .config import LiveConfig

# Dark, talk-ready palette (matches the pyqtgraph app).
_BG = "#0e1117"
_PANEL = "#161b22"
_FG = "#c9d1d9"
_GRID = "#30363d"
_ACCENT_E = "#58d6ff"
_ACCENT_I = "#ff8c42"


def render_static(cfg: LiveConfig, db: np.ndarray, E: np.ndarray,
                  fe: Optional[MelFrontEnd], path: str,
                  title: str = "Artificial auditory cortex — model0"):
    """Render input dB / cortical E / population traces to ``path``.

    Parameters
    ----------
    cfg : LiveConfig
    db  : (n_channels, T)  thalamic input in dB (<= 0).
    E   : (n_channels, T)  cortical excitatory rate.
    fe  : MelFrontEnd or None  used only for kHz axis labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = db.shape[1]
    t = np.arange(T) * (cfg.hop_ms / 1000.0)
    extent = [t[0] if T else 0.0, t[-1] if T else 1.0, 0, cfg.n_channels]

    fig = plt.figure(figsize=(13, 8.6), facecolor=_BG, constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 3.0, 1.5])
    fig.suptitle(title, color=_FG, fontsize=15, fontweight="bold")

    # ---- Row 1: thalamic input spectrogram ----
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(db, aspect="auto", origin="lower", cmap=cfg.input_cmap,
                     vmin=-cfg.top_db, vmax=0.0, extent=extent,
                     interpolation="nearest")
    _style_heat(ax0, "Thalamic input  (cochleo-mel, dB)", "mel channel")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.018, pad=0.01)
    _style_cbar(cb0, "dB")
    if fe is not None:
        _add_khz_axis(ax0, cfg, fe)

    # ---- Row 2: cortical response ----
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    vmax = float(np.percentile(E, 99.5)) if E.size and E.max() > 0 else 1.0
    im1 = ax1.imshow(E, aspect="auto", origin="lower", cmap=cfg.cortex_cmap,
                     vmin=0.0, vmax=max(vmax, 1e-6), extent=extent,
                     interpolation="nearest")
    _style_heat(ax1, "A1 cortical response  (excitatory rate E)", "channel")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.018, pad=0.01)
    _style_cbar(cb1, "rate")

    # ---- Row 3: population traces ----
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    popE = E.sum(axis=0)
    active = (E > 0.05 * (E.max() if E.max() > 0 else 1.0)).sum(axis=0)
    ax2.plot(t, popE, color=_ACCENT_E, lw=1.4, label="population E (sum)")
    ax2b = ax2.twinx()
    ax2b.plot(t, active, color=_ACCENT_I, lw=1.1, alpha=0.9,
              label="active channels")
    _style_line(ax2, "Population activity", "time (s)", "summed E")
    ax2b.set_ylabel("active ch", color=_ACCENT_I, fontsize=10)
    ax2b.tick_params(colors=_ACCENT_I, labelsize=9)
    for sp in ax2b.spines.values():
        sp.set_color(_GRID)
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8,
               frameon=False, loc="upper right", labelcolor=_FG)

    fig.savefig(path, dpi=150, facecolor=_BG)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- helpers
def _style_heat(ax, title, ylabel):
    ax.set_title(title, color=_FG, fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel, color=_FG, fontsize=10)
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(_GRID)


def _style_line(ax, title, xlabel, ylabel):
    ax.set_title(title, color=_FG, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, color=_FG, fontsize=10)
    ax.set_ylabel(ylabel, color=_ACCENT_E, fontsize=10)
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_FG, labelsize=9)
    ax.tick_params(axis="y", colors=_ACCENT_E)
    ax.grid(True, color=_GRID, lw=0.5, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_color(_GRID)


def _style_cbar(cb, label):
    cb.set_label(label, color=_FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=_FG, labelsize=8)
    cb.outline.set_edgecolor(_GRID)
    import matplotlib.pyplot as plt
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=_FG)


def _add_khz_axis(ax, cfg: LiveConfig, fe: MelFrontEnd):
    freqs = fe.mel_frequencies()
    ax2 = ax.twinx()
    ax2.set_ylim(0, cfg.n_channels)
    idx = np.linspace(0, cfg.n_channels - 1, 6).astype(int)
    ax2.set_yticks(idx)
    ax2.set_yticklabels([f"{freqs[i] / 1000:.1f}" for i in idx])
    ax2.set_ylabel("freq (kHz)", color=_FG, fontsize=9)
    ax2.tick_params(colors=_FG, labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color(_GRID)
