"""
live_demo_cortical.app
=============

Real-time **stream segregation** visualiser (temporal-coherence framework;
Krishnan, Elhilali & Shamma, PLoS Comp Biol 2014; Teki et al., eLife 2013).

Two modes (``cfg.mode``):

``coherence``  -- the **symmetric** coincidence of the INPUT drive factored by
    nPCA into frequency streams (simultaneous coherent groups).  Panels: input
    log-spectrogram, pitch, stream 1, stream 2, coincidence C, nPCA masks.

``directional`` -- the **directed** coincidence of the model's ACTIVATIONS,
    ``D[i,j] = <E_i(t)·tr_j(t)>`` (the model's own Hebbian post-rate x pre-trace
    operator, leak-integrated).  D is asymmetric, so it resolves temporal ORDER:
    it tells A→B (standard) from B→A (deviant) -- which a symmetric coincidence
    cannot.  Panels: input, temporal-flow trace (forward AB / reverse BA),
    Stream AB, Stream BA, the directed connection map D, and per-channel
    lead→lag.  Streams are time-gated by the sign of the local flow.

Keyboard
--------
    Space  pause / resume
    R      reset
    Q/Esc  quit
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .audio import SpectroFrontEnd, PitchGram
from .config import LiveConfig
from .masks import StreamSeparator
from .engine import LiveEngine

# QShortcut moved between Qt modules across bindings; resolve once.
try:
    _QShortcut = QtGui.QShortcut
except AttributeError:                      # PyQt5 keeps it in QtWidgets
    _QShortcut = QtWidgets.QShortcut

# ---- palette ----
_BG = "#0e1117"
_PANEL = "#161b22"
_FG = "#c9d1d9"
_MUTED = "#8b949e"
_GRID = "#30363d"
_S1 = "#ff8c42"          # stream 1 / lead accent (orange)
_S2 = "#58d6ff"          # stream 2 accent (cyan)
_FWD = "#3fb950"         # forward / standard (green)
_REV = "#f85149"         # reverse / deviant (red)
_STREAM_COLORS = [_S1, _S2, "#b072ff", "#3fb950"]


class LiveDemoApp(QtWidgets.QMainWindow):
    """Scrolling segregation view; ``coherence`` (frequency) or ``directional``
    (temporal order) per ``cfg.mode``."""

    def __init__(self, cfg: LiveConfig, source, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.source = source
        self.paused = False
        self.directional = (cfg.mode == "directional")
        self.segregate = (cfg.mode == "segregate")

        self.fe = SpectroFrontEnd(cfg)

        F = cfg.history_frames
        N = cfg.n_channels
        self._F = F
        self._t = np.linspace(-cfg.history_s, 0.0, F)
        self._in_db = np.full((N, F), -cfg.top_db, dtype=np.float32)
        self._last_read = None
        self._fps_ema = float(cfg.target_fps)
        self._last_tick = time.perf_counter()

        if self.directional:
            self.engine = LiveEngine(cfg.to_a1_config(), learn=cfg.learn, seed=0)
            self._D = np.zeros((N, N))                 # leaky directed coincidence
            self._gamma = float(np.exp(-cfg.dt / max(cfg.forget_s, 1e-3)))
            self._fwd = np.zeros(F, dtype=np.float32)   # consistent-order energy
            self._rev = np.zeros(F, dtype=np.float32)   # order-VIOLATION energy (deviant)
            self._lead = np.zeros(N, dtype=np.float32)  # per-channel lead score
            self._D_vmax = 1e-4
            self._flow_scale = 1e-6
        elif self.segregate:
            from .segregate import EventSegregator
            self.seg = EventSegregator(
                N, cfg.dt, n_streams=cfg.n_streams, sig_tau=cfg.seg_tau,
                merge_gap_s=cfg.seg_merge_gap_s, max_events=cfg.seg_max_events)
            self._evlab = np.full(F, -1, dtype=np.int16)   # per-frame stream label
            self._segD = np.zeros((N, N), dtype=np.float32)  # last event's D
            self._D_vmax = 1e-4
            self._n_events = 0
        else:
            self.pitch = PitchGram(self.fe.center_freqs(), n_pitch=cfg.n_pitch,
                                   fmin=cfg.pitch_fmin, fmax=cfg.pitch_fmax,
                                   n_harm=cfg.pitch_harmonics, decay=cfg.pitch_decay)
            self.sep = StreamSeparator(cfg.n_channels, n_streams=cfg.n_streams,
                                       iters=cfg.sep_iters)
            self._drive = np.zeros((N, F), dtype=np.float32)
            self._pitch = np.zeros((cfg.n_pitch, F), dtype=np.float32)
            self._C = np.zeros((N, N), dtype=np.float32)
            self._masks = np.zeros((N, cfg.n_streams), dtype=np.float32)
            self._C_vmax = 1e-2
            self._pitch_vmax = 1e-2
            self._coh_tick = 0

        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(int(1000 / max(1, cfg.target_fps)))

    # -----------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        cfg = self.cfg
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        self.setWindowTitle("Stream segregation (live)")
        self.resize(1360, 900)
        self.setStyleSheet(f"background-color:{_BG};")

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(_BG)
        self.setCentralWidget(self.glw)

        self._img_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                       cfg.n_channels)

        if self.directional:
            title = ("Directional Segregation")
            sub = ("model activations E -> directed coincidence "
                   "D[i,j]=⟨E_i·tr_j⟩")
        elif self.segregate:
            title = ("Unsupervised Stream Segregation")
            sub = ("per-event directed-coincidence signatures -> centre -> "
                   "PCA + k-means · balanced, gap-insensitive, any paradigm")
        else:
            title = ("Stream Segregation; temporal coherence (nPCA of the "
                     "coincidence matrix)")
            sub = ("log-spectrogram -> coincidence C -> nPCA masks -> segregated "
                   "streams . real time")
        self.title = self.glw.addLabel(title, row=0, col=0, colspan=4,
                                       color=_FG, size="20pt", bold=True)
        self.subtitle = self.glw.addLabel(sub, row=1, col=0, colspan=4,
                                          color=_MUTED, size="10pt")
        self.stats = self.glw.addLabel("", row=2, col=0, colspan=4,
                                       color=_FG, size="10.5pt")

        spec_cmap = pg.colormap.get(cfg.input_cmap, source="matplotlib")
        centers = self.fe.center_freqs()
        dbl = (-cfg.top_db, 0.0)

        # ---- panel 1: input log-spectrogram (both modes) ----
        self.p_in = self.glw.addPlot(row=3, col=0)
        self.img_in = pg.ImageItem(); self.img_in.setColorMap(spec_cmap)
        in_title = ("Input sequence" if (self.directional or self.segregate)
                    else "Input log-spectrogram")
        self._heat(self.p_in, self.img_in, in_title, centers, cfg.n_channels)
        self.img_in.setLevels(dbl)
        self._add_cbar(self.img_in, dbl, spec_cmap, "dB", row=3)

        if self.directional:
            self._build_directional(centers, spec_cmap, dbl)
        elif self.segregate:
            self._build_segregate(centers, spec_cmap, dbl)
        else:
            self._build_coherence(centers, spec_cmap, dbl)

        for r in (3, 4, 5, 6):
            self.glw.ci.layout.setRowStretchFactor(r, 4)
        self.glw.ci.layout.setColumnStretchFactor(0, 10)
        self.glw.ci.layout.setColumnStretchFactor(2, 7)

        self._refresh_images()
        self._install_shortcuts()

    # ---- coherence-mode panels (frequency streams) ----
    def _build_coherence(self, centers, spec_cmap, dbl):
        cfg = self.cfg
        pitch_cmap = pg.colormap.get(cfg.pitch_cmap, source="matplotlib")
        pfreqs = self.pitch.pitch_freqs()
        self._pitch_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                         cfg.n_pitch)

        self.p_pitch = self.glw.addPlot(row=4, col=0)
        self.img_pitch = pg.ImageItem(); self.img_pitch.setColorMap(pitch_cmap)
        self._heat(self.p_pitch, self.img_pitch, "Pitch  (subharmonic summation)",
                   pfreqs, cfg.n_pitch, left="pitch F0")
        self.bar_pitch = self._add_cbar(self.img_pitch, (0.0, self._pitch_vmax),
                                        pitch_cmap, "salience", row=4)

        self.p_str1 = self.glw.addPlot(row=5, col=0)
        self.img_str1 = pg.ImageItem(); self.img_str1.setColorMap(spec_cmap)
        self._heat(self.p_str1, self.img_str1, "Stream 1", centers,
                   cfg.n_channels, title_color=_S1)
        self.img_str1.setLevels(dbl)

        self.p_str2 = self.glw.addPlot(row=6, col=0)
        self.img_str2 = pg.ImageItem(); self.img_str2.setColorMap(spec_cmap)
        self._heat(self.p_str2, self.img_str2, "Stream 2", centers,
                   cfg.n_channels, title_color=_S2, xlabel="time (s)")
        self.img_str2.setLevels(dbl)

        for p in (self.p_pitch, self.p_str1, self.p_str2):
            p.setXLink(self.p_in)

        self.p_C, self.img_C, self.bar_C = self._mk_matrix(
            3, "C", "magma", "corr")

        self.p_masks = self.glw.addPlot(row=5, col=2, rowspan=2)
        self.p_masks.setTitle("nPCA stream masks", color=_FG, size="11pt",
                              bold=True)
        self.p_masks.setLabel("bottom", "channel", color=_MUTED)
        self.p_masks.setLabel("left", "mask", color=_MUTED)
        self._style_curve_plot(self.p_masks)
        self.p_masks.setRange(xRange=(0, cfg.n_channels), yRange=(0, 1.05),
                              padding=0)
        self.curve_masks = [
            self.p_masks.plot(
                pen=pg.mkPen(_STREAM_COLORS[i % len(_STREAM_COLORS)], width=2),
                fillLevel=0.0,
                brush=pg.mkBrush(_STREAM_COLORS[i % len(_STREAM_COLORS)] + "40"))
            for i in range(cfg.n_streams)]

    # ---- directional-mode panels (temporal order) ----
    def _build_directional(self, centers, spec_cmap, dbl):
        cfg = self.cfg

        # temporal-flow trace: forward (AB, green) above 0, reverse (BA, red) below
        self.p_flow = self.glw.addPlot(row=4, col=0)
        self.p_flow.setTitle("Temporal flow", color=_FG, size="11pt", bold=True)
        self.p_flow.setLabel("left", "flow", color=_MUTED)
        self._style_curve_plot(self.p_flow)
        self.p_flow.setRange(xRange=(-cfg.history_s, 0.0), yRange=(-1.05, 1.05),
                             padding=0)
        self.p_flow.addLine(y=0.0, pen=pg.mkPen(_GRID, width=1))
        self.curve_flow_pos = self.p_flow.plot(
            pen=pg.mkPen(_FWD, width=1.2), fillLevel=0.0,
            brush=pg.mkBrush(_FWD + "55"))
        self.curve_flow_neg = self.p_flow.plot(
            pen=pg.mkPen(_REV, width=1.2), fillLevel=0.0,
            brush=pg.mkBrush(_REV + "55"))

        self.p_str1 = self.glw.addPlot(row=5, col=0)
        self.img_str1 = pg.ImageItem(); self.img_str1.setColorMap(spec_cmap)
        self._heat(self.p_str1, self.img_str1, "Stream 2",
                   centers, cfg.n_channels, title_color=_FWD)
        self.img_str1.setLevels(dbl)

        self.p_str2 = self.glw.addPlot(row=6, col=0)
        self.img_str2 = pg.ImageItem(); self.img_str2.setColorMap(spec_cmap)
        self._heat(self.p_str2, self.img_str2, "Stream 1",
                   centers, cfg.n_channels, title_color=_REV, xlabel="time (s)")
        self.img_str2.setLevels(dbl)

        for p in (self.p_flow, self.p_str1, self.p_str2):
            p.setXLink(self.p_in)

        # directed connection map D (asymmetric): row=post (follower), col=pre (leader)
        self.p_C, self.img_C, self.bar_C = self._mk_matrix(
            3, "Connections: directed  ⟨E·tr⟩", "magma", "⟨E·tr⟩",
            xlabel="channel (leads)", ylabel="channel (follows)")

        # per-channel lead→lag score (col-sum − row-sum of D): + leads, − follows
        self.p_lead = self.glw.addPlot(row=5, col=2, rowspan=2)
        self.p_lead.setTitle("lead → lag   (per channel)", color=_FG,
                             size="11pt", bold=True)
        self.p_lead.setLabel("bottom", "channel", color=_MUTED)
        self.p_lead.setLabel("left", "leads (+) / follows (−)", color=_MUTED)
        self._style_curve_plot(self.p_lead)
        self.p_lead.setRange(xRange=(0, cfg.n_channels), yRange=(-1.05, 1.05),
                             padding=0)
        self.p_lead.addLine(y=0.0, pen=pg.mkPen(_GRID, width=1))
        self.curve_lead = self.p_lead.plot(
            pen=pg.mkPen(_S1, width=2), fillLevel=0.0,
            brush=pg.mkBrush(_S1 + "40"))

    # ---- segregate-mode panels (unsupervised event clustering) ----
    def _build_segregate(self, centers, spec_cmap, dbl):
        cfg = self.cfg
        # Stream 1 / Stream 2 = the cochleagram masked by each event's cluster
        self.p_t1 = self.glw.addPlot(row=4, col=0)
        self.img_t1 = pg.ImageItem(); self.img_t1.setColorMap(spec_cmap)
        self._heat(self.p_t1, self.img_t1, "Stream 1  (cluster A)", centers,
                   cfg.n_channels, title_color=_S1)
        self.img_t1.setLevels(dbl)
        self.p_t2 = self.glw.addPlot(row=5, col=0)
        self.img_t2 = pg.ImageItem(); self.img_t2.setColorMap(spec_cmap)
        self._heat(self.p_t2, self.img_t2, "Stream 2  (cluster B)", centers,
                   cfg.n_channels, title_color=_S2, xlabel="time (s)")
        self.img_t2.setLevels(dbl)
        # event tape: per-frame cluster colour, aligned with the streams
        self.p_tape = self.glw.addPlot(row=6, col=0)
        self.p_tape.setTitle("events (auto-clustered)", color=_FG, size="11pt",
                             bold=True)
        self.p_tape.setMouseEnabled(x=False, y=False)
        self.p_tape.hideButtons(); self.p_tape.setMenuEnabled(False)
        self.p_tape.setDefaultPadding(0.0)
        self.p_tape.getViewBox().setBackgroundColor(_PANEL)
        self.p_tape.getAxis("left").hide()
        self.p_tape.getAxis("bottom").setPen(_GRID)
        self.p_tape.getAxis("bottom").setTextPen(_MUTED)
        self.p_tape.setRange(xRange=(-cfg.history_s, 0.0), yRange=(0, 1), padding=0)
        self.p_tape.setMaximumHeight(60)
        self.img_tape = pg.ImageItem(); self.p_tape.addItem(self.img_tape)
        self._tape_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s, 1.0)
        for p in (self.p_t1, self.p_t2, self.p_tape):
            p.setXLink(self.p_in)
        # right: signature scatter (PCA of per-event D) + last-event D
        self.p_scatter = self.glw.addPlot(row=3, col=2, rowspan=2)
        self.p_scatter.setTitle("signature space  (PCA of per-event D)",
                                color=_FG, size="11pt", bold=True)
        self.p_scatter.setLabel("bottom", "PC1", color=_MUTED)
        self.p_scatter.setLabel("left", "PC2", color=_MUTED)
        self.p_scatter.setMouseEnabled(x=False, y=False)
        self.p_scatter.hideButtons(); self.p_scatter.setMenuEnabled(False)
        self.p_scatter.getViewBox().setBackgroundColor(_PANEL)
        for a in ("left", "bottom"):
            self.p_scatter.getAxis(a).setPen(_GRID)
            self.p_scatter.getAxis(a).setTextPen(_MUTED)
        self.scatter = pg.ScatterPlotItem(size=9, pen=None)
        self.p_scatter.addItem(self.scatter)
        self.p_C, self.img_C, self.bar_C = self._mk_matrix(
            5, "last event   D=⟨E·tr⟩", "magma", "⟨·⟩")

    def _tape_rgb(self):
        F = self._F
        rgb = np.empty((1, F, 3), dtype=np.ubyte)
        rgb[0, :] = (22, 27, 34)                         # background (panel)
        for lab in range(self.cfg.n_streams):
            m = self._evlab == lab
            if m.any():
                c = pg.mkColor(_STREAM_COLORS[lab % len(_STREAM_COLORS)])
                rgb[0, m] = c.getRgb()[:3]
        return rgb

    # ---- small UI helpers (shared) ----
    def _heat(self, plot, img, title, right_freqs, ny, *, left="channel",
              xlabel="", title_color=_FG):
        cfg = self.cfg
        plot.addItem(img)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons(); plot.setMenuEnabled(False); plot.setDefaultPadding(0.0)
        vb = plot.getViewBox()
        vb.setBackgroundColor(_PANEL)
        vb.enableAutoRange(x=False, y=False)
        plot.setRange(xRange=(-cfg.history_s, 0.0), yRange=(0, ny), padding=0)
        plot.setTitle(title, color=title_color, size="11pt", bold=True)
        plot.setLabel("left", left, color=_MUTED)
        plot.setLabel("bottom", xlabel, color=_MUTED)
        for a in ("left", "bottom"):
            plot.getAxis(a).setPen(_GRID)
            plot.getAxis(a).setTextPen(_MUTED)
        fmt = "%.2f" if float(np.max(right_freqs)) < 1000.0 else "%.1f"
        idx = np.linspace(0, ny - 1, 6).astype(int)
        rax = plot.getAxis("right")
        rax.setTicks([[(int(i), fmt % (right_freqs[i] / 1000.0)) for i in idx]])
        rax.setLabel("kHz", color=_MUTED)
        rax.setPen(_GRID); rax.setTextPen(_MUTED)
        plot.showAxis("right")

    def _style_curve_plot(self, plot):
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons(); plot.setMenuEnabled(False)
        plot.getViewBox().setBackgroundColor(_PANEL)
        for a in ("left", "bottom"):
            plot.getAxis(a).setPen(_GRID)
            plot.getAxis(a).setTextPen(_MUTED)

    def _add_cbar(self, img, values, cmap, label, row):
        bar = pg.ColorBarItem(values=values, colorMap=cmap, label=label,
                              width=14)
        bar.setImageItem(img)
        self.glw.addItem(bar, row=row, col=1)
        self._style_cbar(bar)
        return bar

    def _mk_matrix(self, row, title, cmap_name, label, *, xlabel="channel",
                   ylabel="channel"):
        cfg = self.cfg
        cmap = pg.colormap.get(cmap_name, source="matplotlib")
        p = self.glw.addPlot(row=row, col=2, rowspan=2)
        img = pg.ImageItem(); img.setColorMap(cmap); p.addItem(img)
        p.setTitle(title, color=_FG, size="11pt", bold=True)
        p.setLabel("left", ylabel, color=_MUTED)
        p.setLabel("bottom", xlabel, color=_MUTED)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons(); p.setMenuEnabled(False); p.setDefaultPadding(0.0)
        vb = p.getViewBox()
        vb.setBackgroundColor(_BG)
        vb.setAspectLocked(True)
        for _a in ("left", "bottom"):
            p.getAxis(_a).setPen(_GRID)
            p.getAxis(_a).setTextPen(_MUTED)
            p.getAxis(_a).setTicks(
                [[(v, str(v)) for v in range(0, cfg.n_channels + 1, 20)]])
        p.setRange(xRange=(0, cfg.n_channels), yRange=(0, cfg.n_channels),
                   padding=0)
        bar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap, label=label,
                              width=14)
        bar.setImageItem(img)
        self.glw.addItem(bar, row=row, col=3, rowspan=2)
        self._style_cbar(bar)
        return p, img, bar

    def _style_cbar(self, bar):
        try:
            bar.getAxis("right").setTextPen(_MUTED)
            bar.getAxis("right").setPen(_GRID)
        except Exception:
            pass

    # -----------------------------------------------------------------
    #  Streaming loop
    # -----------------------------------------------------------------
    def start(self):
        if hasattr(self.source, "start"):
            self.source.start()
        self._last_read = time.perf_counter()
        self.timer.start()

    def _read_samples(self) -> np.ndarray:
        now = time.perf_counter()
        if getattr(self.source, "paced", False):
            elapsed = now - (self._last_read or now)
            n = int(self.cfg.sr * elapsed)
            self._last_read = now
            if n <= 0:
                return np.empty(0)
            return self.source.read(n)
        self._last_read = now
        return self.source.read()

    def _tick(self):
        if self.paused:
            return
        samples = self._read_samples()
        drive, db = self.fe.push(samples)
        if drive.shape[1]:
            self._push_cols(self._in_db, db.astype(np.float32))
            if self.directional:
                out = self.engine.step_block(drive)
                self._update_directed(out["E"], out["tr"])
            elif self.segregate:
                self.seg.push(drive.astype(np.float64))
            else:
                self._push_cols(self._drive, drive.astype(np.float32))
                self._push_cols(self._pitch,
                                self.pitch.push(drive).astype(np.float32))
            self._refresh_images()

        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * (1.0 / dt)
        self._update_stats()

    def _push_cols(self, buf, new):
        k = new.shape[1]
        if k >= buf.shape[1]:
            buf[:, :] = new[:, -buf.shape[1]:]
        else:
            buf[:, :-k] = buf[:, k:]
            buf[:, -k:] = new

    def _push_1d(self, buf, new):
        k = new.shape[0]
        if k >= buf.shape[0]:
            buf[:] = new[-buf.shape[0]:]
        else:
            buf[:-k] = buf[k:]
            buf[-k:] = new

    # -----------------------------------------------------------------
    #  coherence compute (frequency streams)
    # -----------------------------------------------------------------
    def _update_coincidence(self):
        cfg = self.cfg
        cs = max(1, int(round(cfg.coh_bin_ms / cfg.frame_ms)))
        nb = min(self._F // cs,
                 int(cfg.coh_window_s * 1000.0 / cfg.coh_bin_ms))
        if nb < 8:
            return
        rec = self._drive[:, -nb * cs:]
        binned = rec.reshape(rec.shape[0], nb, cs).mean(2)
        binned = binned - binned.mean(0, keepdims=True)
        x = binned - binned.mean(1, keepdims=True)
        nrm = np.sqrt((x * x).sum(1))
        denom = np.outer(nrm, nrm)
        C = np.where(denom > 1e-9, (x @ x.T) / (denom + 1e-12), 0.0)
        np.fill_diagonal(C, 0.0)
        iu = np.triu_indices(C.shape[0], 1)
        off = C[iu]
        med = float(np.median(off))
        mad = float(np.median(np.abs(off - med))) + 1e-9
        theta = med + cfg.coh_floor_z * 1.4826 * mad
        C = np.clip(C - theta, 0.0, 1.0)
        self.sep.update(C)
        self._masks = self.sep.masks().astype(np.float32)
        self._C = C.astype(np.float32)
        p = float(np.percentile(self._C, 99.7))
        self._C_vmax = max(0.85 * self._C_vmax + 0.15 * p, 0.05)

    def _stream_db(self, k):
        m = self._masks[:, k][:, None]
        return (self._in_db * m - self.cfg.top_db * (1.0 - m)).astype(np.float32)

    # -----------------------------------------------------------------
    #  directional compute (temporal order)
    # -----------------------------------------------------------------
    def _update_directed(self, E, tr):
        """Leak-integrate the directed coincidence D[i,j]=<E_i(t) tr_j(t)> (the
        model's Hebbian post x pre-trace), then read off the directional flow
        (a forward/reverse signal over time) and a per-channel lead→lag score
        from its antisymmetric part."""
        g, k = self._gamma, E.shape[1]
        w = (1.0 - g) * g ** (k - 1 - np.arange(k))     # leaky-integration weights
        self._D = (g ** k) * self._D + (E * w) @ tr.T
        Dz = self._D.copy()
        np.fill_diagonal(Dz, 0.0)
        # keep only CROSS-tone coincidences: zero the near-diagonal band so a
        # tone's own spectral spread (within-cluster leakage) can't register as
        # a self-violation at every onset -- only DISTINCT tones define an order.
        N = Dz.shape[0]
        idx = np.arange(N)
        Dz[np.abs(idx[:, None] - idx[None, :]) <= 3] = 0.0
        Delta = Dz - Dz.T                               # skew part = direction
        Dhat = Delta / (np.linalg.norm(Delta) + 1e-9)
        # Split the directional template into forward (consistent) and reverse
        # (order-violating) edges.  Then per frame:
        #   fwd(t) = activity that flows WITH the established order,
        #   rev(t) = activity that flows AGAINST it.
        # A deviant (BA, or the B↔C swap in ACB) shows as a rev spike, while the
        # net flow fwd-rev stays positive (the common A-leads dominate) -- so it
        # is the VIOLATION energy rev, not the net flow, that flags a deviant.
        Dp = np.maximum(Dhat, 0.0)
        Dm = np.maximum(-Dhat, 0.0)
        fwd = (E * (Dp @ tr)).sum(0)
        rev = (E * (Dm @ tr)).sum(0)
        self._push_1d(self._fwd, fwd.astype(np.float32))
        self._push_1d(self._rev, rev.astype(np.float32))
        # lead score: col-sum − row-sum of D.  D[i,j] large ⟺ i follows j, so
        # col j (others follow j) − row j (j follows others) = net "j leads".
        self._lead = (Dz.sum(0) - Dz.sum(1)).astype(np.float32)
        self._D_vmax = max(0.85 * self._D_vmax
                           + 0.15 * float(np.percentile(Dz, 99.7)), 1e-4)
        self._flow_scale = max(                         # consistent-energy scale
            0.85 * self._flow_scale
            + 0.15 * float(np.percentile(self._fwd, 95)), 1e-6)

    # -----------------------------------------------------------------
    #  refresh
    # -----------------------------------------------------------------
    def _refresh_images(self):
        if self.directional:
            self._refresh_directional()
        elif self.segregate:
            self._refresh_segregate()
        else:
            self._refresh_coherence()

    def _refresh_coherence(self):
        dbl = (-self.cfg.top_db, 0.0)
        self.img_in.setImage(self._in_db, autoLevels=False, levels=dbl)
        self._coh_tick += 1
        if self._coh_tick % 6 == 0:
            self._update_coincidence()
        disp = np.maximum(self._pitch - np.median(self._pitch, axis=0,
                                                   keepdims=True), 0.0)
        pmax = float(np.percentile(disp, 99.5)) if disp.any() else 0.0
        self._pitch_vmax = max(0.8 * self._pitch_vmax + 0.2 * pmax, 1e-2)
        self.img_pitch.setImage(disp, autoLevels=False,
                                levels=(0.0, self._pitch_vmax))
        self.bar_pitch.setLevels((0.0, self._pitch_vmax))
        self.img_str1.setImage(self._stream_db(0), autoLevels=False, levels=dbl)
        self.img_str2.setImage(self._stream_db(1), autoLevels=False, levels=dbl)
        self.img_C.setImage(self._C, autoLevels=False, levels=(0.0, self._C_vmax))
        self.bar_C.setLevels((0.0, self._C_vmax))
        _x = np.arange(self.cfg.n_channels)
        for i, cv in enumerate(self.curve_masks):
            cv.setData(_x, self._masks[:, i])
        for im in (self.img_in, self.img_str1, self.img_str2):
            im.setRect(self._img_rect)
        self.img_pitch.setRect(self._pitch_rect)

    def _refresh_directional(self):
        cfg = self.cfg
        dbl = (-cfg.top_db, 0.0)
        self.img_in.setImage(self._in_db, autoLevels=False, levels=dbl)
        # gates from the forward/violation energies, both referenced to the
        # consistent-energy scale: gp ~ how much standard-order flow now, gn ~
        # how much order-violation (deviant) now.
        sc = self._flow_scale + 1e-9
        gp = np.clip(self._fwd / sc, 0.0, 1.0)        # forward / consistent gate
        gn = np.clip(self._rev / sc, 0.0, 1.0)        # reverse / violation gate
        floor = -cfg.top_db
        ab = self._in_db * gp[None, :] + floor * (1.0 - gp[None, :])
        ba = self._in_db * gn[None, :] + floor * (1.0 - gn[None, :])
        self.img_str1.setImage(ab.astype(np.float32), autoLevels=False, levels=dbl)
        self.img_str2.setImage(ba.astype(np.float32), autoLevels=False, levels=dbl)
        # directed connection map (asymmetric)
        Dz = self._D.copy(); np.fill_diagonal(Dz, 0.0)
        self.img_C.setImage(Dz.astype(np.float32), autoLevels=False,
                            levels=(0.0, self._D_vmax))
        self.bar_C.setLevels((0.0, self._D_vmax))
        # flow trace (green forward / red reverse) and per-channel lead score
        self.curve_flow_pos.setData(self._t, gp)
        self.curve_flow_neg.setData(self._t, -gn)
        lead = self._lead / (np.abs(self._lead).max() + 1e-9)
        self.curve_lead.setData(np.arange(cfg.n_channels), lead)
        for im in (self.img_in, self.img_str1, self.img_str2):
            im.setRect(self._img_rect)

    def _refresh_segregate(self):
        cfg = self.cfg
        N, F = cfg.n_channels, self._F
        dbl = (-cfg.top_db, 0.0)
        floor = -cfg.top_db
        self.img_in.setImage(self._in_db, autoLevels=False, levels=dbl)
        # re-cluster only when a new event has completed (cheap, label-stable)
        if len(self.seg.sigs) != self._n_events and len(self.seg.sigs) >= 2:
            self.seg.cluster()
            self._n_events = len(self.seg.sigs)
        # rebuild the per-frame stream-label buffer from event spans + labels
        self._evlab[:] = -1
        now = self.seg.frame
        if self.seg.labels is not None:
            for (s, e), lab in zip(self.seg.spans, self.seg.labels):
                c0 = max(0, s - (now - F))
                c1 = min(F, e - (now - F))
                if c1 > c0:
                    self._evlab[c0:c1] = lab
            self._segD = self.seg.sigs[-1].reshape(N, N)
        # streams = cochleagram gated by each event's cluster
        t1 = np.where(self._evlab[None, :] == 0, self._in_db, floor)
        t2 = np.where(self._evlab[None, :] == 1, self._in_db, floor)
        self.img_t1.setImage(t1.astype(np.float32), autoLevels=False, levels=dbl)
        self.img_t2.setImage(t2.astype(np.float32), autoLevels=False, levels=dbl)
        self.img_tape.setImage(self._tape_rgb()); self.img_tape.setRect(self._tape_rect)
        # last-event directed coincidence
        Dz = self._segD.copy(); np.fill_diagonal(Dz, 0.0)
        if Dz.any():
            self._D_vmax = max(0.85 * self._D_vmax
                               + 0.15 * float(np.percentile(Dz, 99.5)), 1e-4)
        self.img_C.setImage(Dz.astype(np.float32), autoLevels=False,
                            levels=(0.0, self._D_vmax))
        self.bar_C.setLevels((0.0, self._D_vmax))
        # signature scatter (PCA), coloured by cluster
        P, L = self.seg.proj, self.seg.labels
        if L is not None and P.shape[0] == len(L) and P.shape[0]:
            brushes = [pg.mkBrush(_STREAM_COLORS[int(l) % len(_STREAM_COLORS)])
                       for l in L]
            self.scatter.setData(P[:, 0], P[:, 1], brush=brushes, pen=None, size=9)
            xr = (float(P[:, 0].min()), float(P[:, 0].max()))
            yr = (float(P[:, 1].min()), float(P[:, 1].max()))
            if xr[1] > xr[0] and yr[1] > yr[0]:
                self.p_scatter.setRange(xRange=xr, yRange=yr, padding=0.2)
        for im in (self.img_in, self.img_t1, self.img_t2):
            im.setRect(self._img_rect)

    def _update_stats(self):
        cfg = self.cfg
        lvl = self.fe.level_db
        gate = self.fe.gate_open
        gate_c = _FWD if gate else _MUTED
        paused = (f"<b style='color:{_S1}'>⏸ PAUSED</b> &nbsp;|&nbsp; "
                  if self.paused else "")
        if self.directional:
            sc = self._flow_scale + 1e-9
            rnow = float(self._rev[-1]) if self._F else 0.0
            rev_frac = 100.0 * float(np.mean(self._rev > 0.4 * sc))
            dirn = (f"<b style='color:{_REV}'>deviant</b>" if rnow > 0.4 * sc
                    else f"<b style='color:{_FWD}'>standard</b>")
            self.stats.setText(
                paused +
                f"<span style='color:{gate_c}'>●</span> "
                f"<span style='color:{_MUTED}'>input</span> <b>{lvl:+5.1f} dB</b>"
                f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>flow now</span> "
                f"{dirn} &nbsp;|&nbsp; "
                f"<span style='color:{_MUTED}'>reverse</span> "
                f"<b style='color:{_REV}'>{rev_frac:3.0f}%</b> &nbsp;|&nbsp; "
                f"<span style='color:{_MUTED}'>forget</span> "
                f"<b>{cfg.forget_s:.1f} s</b> &nbsp;|&nbsp; "
                f"<span style='color:{_MUTED}'>|D| peak</span> "
                f"<b>{self._D_vmax:.3f}</b> &nbsp;|&nbsp; "
                f"<span style='color:{_MUTED}'>{self._fps_ema:.0f} fps</span>")
            return
        if self.segregate:
            ne = len(self.seg.sigs)
            sizes = ""
            if self.seg.labels is not None:
                cnt = [int((self.seg.labels == c).sum()) for c in range(cfg.n_streams)]
                sizes = " / ".join(
                    f"<b style='color:{_STREAM_COLORS[c % len(_STREAM_COLORS)]}'>"
                    f"{cnt[c]}</b>" for c in range(cfg.n_streams))
            self.stats.setText(
                paused +
                f"<span style='color:{gate_c}'>●</span> "
                f"<span style='color:{_MUTED}'>input</span> <b>{lvl:+5.1f} dB</b>"
                f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>streams</span> "
                f"<b>{cfg.n_streams}</b> &nbsp;|&nbsp; "
                f"<span style='color:{_MUTED}'>events</span> <b>{ne}</b>"
                + (f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>cluster sizes</span> "
                   f"{sizes}" if sizes else "")
                + f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>window</span> "
                  f"<b>{cfg.seg_max_events}</b> &nbsp;|&nbsp; "
                  f"<span style='color:{_MUTED}'>{self._fps_ema:.0f} fps</span>")
            return
        nact = int((self._drive[:, -1] > 0.02).sum()) if self._F else 0
        s1 = int((self._masks[:, 0] > 0.5).sum())
        s2 = int((self._masks[:, 1] > 0.5).sum()) if cfg.n_streams > 1 else 0
        m1, m2 = self._masks[:, 0], self._masks[:, min(1, cfg.n_streams - 1)]
        denom = float(np.linalg.norm(m1) * np.linalg.norm(m2))
        overlap = float(m1 @ m2) / denom if denom > 1e-9 else 0.0
        sep = 100.0 * (1.0 - overlap)
        self.stats.setText(
            paused +
            f"<span style='color:{gate_c}'>●</span> "
            f"<span style='color:{_MUTED}'>input</span> <b>{lvl:+5.1f} dB</b>"
            f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>active</span> "
            f"<b>{nact}/{cfg.n_channels}</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>stream sizes</span> "
            f"<b style='color:{_S1}'>{s1}</b> / "
            f"<b style='color:{_S2}'>{s2}</b> ch &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>separation</span> "
            f"<b style='color:{_FWD}'>{sep:3.0f}%</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>{self._fps_ema:.0f} fps</span>")

    # -----------------------------------------------------------------
    #  Controls
    # -----------------------------------------------------------------
    def _install_shortcuts(self):
        binds = [("Space", self.toggle_pause), ("R", self.reset),
                 ("Q", self.close)]
        self._shortcuts = []
        for seq, fn in binds:
            sc = _QShortcut(QtGui.QKeySequence(seq), self)
            try:
                sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            except AttributeError:
                sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self._last_read = time.perf_counter()
            try:
                if not getattr(self.source, "paced", False):
                    self.source.read()
            except Exception:
                pass
        self._update_stats()

    def reset(self):
        self.fe.reset()
        self._in_db[:] = -self.cfg.top_db
        if self.directional:
            self.engine = LiveEngine(self.cfg.to_a1_config(),
                                     learn=self.cfg.learn, seed=0)
            self._D[:] = 0.0
            self._fwd[:] = 0.0
            self._rev[:] = 0.0
            self._lead[:] = 0.0
            self._D_vmax = 1e-4
            self._flow_scale = 1e-6
        elif self.segregate:
            self.seg.reset()
            self._evlab[:] = -1
            self._segD[:] = 0.0
            self._D_vmax = 1e-4
            self._n_events = 0
        else:
            self.sep.reset()
            self._drive[:] = 0.0
            self._pitch[:] = 0.0
            self._C[:] = 0.0
            self._masks[:] = 0.0
            self._pitch_vmax = 1e-2
        self._refresh_images()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self.close()
        elif k == QtCore.Qt.Key.Key_R:
            self.reset()
        elif k == QtCore.Qt.Key.Key_Space:
            self.toggle_pause()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        try:
            self.timer.stop()
            if hasattr(self.source, "stop"):
                self.source.stop()
        finally:
            super().closeEvent(ev)

    # -----------------------------------------------------------------
    #  Offline driving (for headless snapshot / validation)
    # -----------------------------------------------------------------
    def feed_offline(self, audio: np.ndarray):
        bs = self.cfg.blocksize
        for lo in range(0, audio.size, bs):
            drive, db = self.fe.push(audio[lo:lo + bs])
            if drive.shape[1] == 0:
                continue
            self._push_cols(self._in_db, db.astype(np.float32))
            if self.directional:
                out = self.engine.step_block(drive)
                self._update_directed(out["E"], out["tr"])
            elif self.segregate:
                self.seg.push(drive.astype(np.float64))
            else:
                self._push_cols(self._drive, drive.astype(np.float32))
                self._push_cols(self._pitch,
                                self.pitch.push(drive).astype(np.float32))
        if self.segregate:
            self.seg.finalize()
        elif not self.directional:
            self._update_coincidence()
        self._refresh_images()
        self._update_stats()

    def grab_image(self, path: str):
        QtWidgets.QApplication.processEvents()
        self.glw.grab().save(path)
        return path
