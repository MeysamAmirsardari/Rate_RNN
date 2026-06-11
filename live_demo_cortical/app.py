"""
live_demo_cortical.app
=============

Real-time, talk-ready visualisation of the model0 A1 RNN listening to a
live audio stream (microphone, WAV, or synthetic).

Two scrolling heatmaps share a time axis:

    top     cochleo-thalamic mel input (dB)   -- what the thalamus sends
    middle  A1 excitatory rate E              -- the cortical response
    bottom  population traces (summed E + active-channel count)

plus a header with the title and a live stats strip.  The visual contrast
between the dense input (top) and the sparse cortical code (middle) *is*
the model's claim -- lateral inhibition as a global normaliser turning a
dense thalamic drive into a sparse, contrast-enhanced cortical
representation, with multi-timescale depression making repeated sounds
visibly adapt.

Built on pyqtgraph for smooth scrolling; the model still steps at 1 ms
internally while the display refreshes at ``target_fps``.

Keyboard
--------
    C      toggle the cortical spectral front end on/off
    L      toggle Hebbian learning on/off
    I      toggle selective <-> uniform inhibition (state carried over)
    Space  pause / resume
    R      reset model + display
    Q/Esc  quit
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .audio import SpectroFrontEnd
from .config import LiveConfig
from .masks import StreamSeparator
from .engine import LiveEngine

# QShortcut moved between Qt modules across bindings; resolve once.
try:
    _QShortcut = QtGui.QShortcut
except AttributeError:                      # PyQt5 keeps it in QtWidgets
    _QShortcut = QtWidgets.QShortcut

# ---- palette (matches preview.py) ----
_BG = "#0e1117"
_PANEL = "#161b22"
_FG = "#c9d1d9"
_MUTED = "#8b949e"
_GRID = "#30363d"
_E = "#58d6ff"
_ACT = "#ff8c42"
_OK = "#3fb950"
_OFF = "#f85149"


class LiveDemoApp(QtWidgets.QMainWindow):
    """Main window: scrolling input/cortex heatmaps + population traces."""

    def __init__(self, cfg: LiveConfig, source, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.source = source
        self.paused = False

        self.fe = SpectroFrontEnd(cfg)
        self.engine = LiveEngine(cfg.to_a1_config(), learn=cfg.learn, seed=0)
        self.sep = StreamSeparator(cfg.n_channels, n_streams=cfg.n_streams,
                                   iters=cfg.sep_iters)
        self._inhibition = cfg.inhibition
        self._learn = cfg.learn

        # ---- scrolling ring buffers ----
        F = cfg.history_frames
        N = cfg.n_channels
        self._F = F
        self._in_db = np.full((N, F), -cfg.top_db, dtype=np.float32)
        self._E = np.zeros((N, F), dtype=np.float32)
        self._drive = np.zeros((N, F), dtype=np.float32)   # input drive (for C)
        self._popE = np.zeros(F, dtype=np.float32)
        self._active = np.zeros(F, dtype=np.float32)
        self._E_vmax = 1.0
        self._W_vmax = 1e-3
        self._C = np.zeros((N, N), dtype=np.float32)   # live coincidence matrix
        self._C_vmax = 1e-2
        self._masks = np.zeros((N, cfg.n_streams), dtype=np.float32)
        self._coh_tick = 0                              # throttle C re-computation
        self._last_read = None
        self._fps_ema = float(cfg.target_fps)
        self._last_tick = time.perf_counter()

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
        self.setWindowTitle("model0 (live)")
        self.resize(1280, 860)
        self.setStyleSheet(f"background-color:{_BG};")

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(_BG)
        self.setCentralWidget(self.glw)

        # image -> plot coordinate rect: x in [-history, 0] s, y in [0, N] ch.
        self._img_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                       cfg.n_channels)

        # ---- header ----
        self.title = self.glw.addLabel(
            "Stream Segregation — nPCA of the coincidence matrix", row=0, col=0,
            colspan=4, color=_FG, size="20pt", bold=True)
        self.subtitle = self.glw.addLabel(
            "log-spectrogram → coincidence C → nPCA stream masks "
            "(normalized spectral clustering) · real time",
            row=1, col=0, colspan=4, color=_MUTED, size="10pt")
        self.stats = self.glw.addLabel("", row=2, col=0, colspan=4,
                                       color=_FG, size="10.5pt")

        in_cmap = pg.colormap.get(cfg.input_cmap, source="matplotlib")
        ctx_cmap = pg.colormap.get(cfg.cortex_cmap, source="matplotlib")
        freqs = self.fe.mel_frequencies()

        # ---- input spectrogram ----
        self.p_in = self.glw.addPlot(row=3, col=0)
        self.img_in = pg.ImageItem()
        self.img_in.setColorMap(in_cmap)
        self._setup_heat(self.p_in, self.img_in,
                         "Thalamic input; cochleo-mel (dB)", freqs)
        self.img_in.setLevels((-cfg.top_db, 0.0))
        bar_in = pg.ColorBarItem(values=(-cfg.top_db, 0.0), colorMap=in_cmap,
                                 label="dB", width=14)
        bar_in.setImageItem(self.img_in)
        self.glw.addItem(bar_in, row=3, col=1)
        self._style_cbar(bar_in)

        # ---- cortical response ----
        self.p_ctx = self.glw.addPlot(row=4, col=0)
        self.img_ctx = pg.ImageItem()
        self.img_ctx.setColorMap(ctx_cmap)
        self._setup_heat(self.p_ctx, self.img_ctx,
                         "A1 Cortical Response; excitatory rate E", freqs)
        self.img_ctx.setLevels((0.0, self._E_vmax))
        self.bar_ctx = pg.ColorBarItem(values=(0.0, self._E_vmax),
                                       colorMap=ctx_cmap, label="rate", width=14)
        self.bar_ctx.setImageItem(self.img_ctx)
        self.glw.addItem(self.bar_ctx, row=4, col=1)
        self._style_cbar(self.bar_ctx)
        self.p_ctx.setXLink(self.p_in)

        # ---- right column ----
        # TOP    coincidence C[i,j] = windowed correlation of the response E.
        # BOTTOM the nPCA stream masks (rank-2 SNMF of C): each coherent source
        #        is one curve over channels -- the real-time segregation.
        def _mk_matrix(row, title, cmap_name, label):
            cmap = pg.colormap.get(cmap_name, source="matplotlib")
            p = self.glw.addPlot(row=row, col=2)
            img = pg.ImageItem(); img.setColorMap(cmap); p.addItem(img)
            p.setTitle(title, color=_FG, size="11pt", bold=True)
            p.setLabel("left", "channel", color=_MUTED)
            p.setLabel("bottom", "channel", color=_MUTED)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons(); p.setMenuEnabled(False); p.setDefaultPadding(0.0)
            vb = p.getViewBox()
            vb.setBackgroundColor(_BG)        # letterbox blends into canvas
            vb.setAspectLocked(True)          # 1:1 pixels -> true square
            for _a in ("left", "bottom"):
                p.getAxis(_a).setPen(_GRID)
                p.getAxis(_a).setTextPen(_MUTED)
                p.getAxis(_a).setTicks(
                    [[(v, str(v)) for v in range(0, cfg.n_channels + 1, 50)]])
            p.setRange(xRange=(0, cfg.n_channels),
                       yRange=(0, cfg.n_channels), padding=0)
            bar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap,
                                  label=label, width=14)
            bar.setImageItem(img)
            self.glw.addItem(bar, row=row, col=3)
            self._style_cbar(bar)
            return p, img, bar

        self.p_C, self.img_C, self.bar_C = _mk_matrix(
            3, "Coincidence  C  (corr of input)", "magma", "corr")

        # nPCA stream masks (one curve per stream, over channel = log-frequency)
        self.p_masks = self.glw.addPlot(row=4, col=2)
        self.p_masks.setTitle("nPCA stream masks", color=_FG, size="11pt", bold=True)
        self.p_masks.setLabel("bottom", "channel", color=_MUTED)
        self.p_masks.setLabel("left", "mask", color=_MUTED)
        self.p_masks.setMouseEnabled(x=False, y=False)
        self.p_masks.hideButtons(); self.p_masks.setMenuEnabled(False)
        self.p_masks.getViewBox().setBackgroundColor(_PANEL)
        for _a in ("left", "bottom"):
            self.p_masks.getAxis(_a).setPen(_GRID)
            self.p_masks.getAxis(_a).setTextPen(_MUTED)
        self.p_masks.setRange(xRange=(0, cfg.n_channels), yRange=(0, 1.05),
                              padding=0)
        _scols = ["#ff8c42", "#58d6ff", "#b072ff", "#3fb950"]
        self.curve_masks = [
            self.p_masks.plot(pen=pg.mkPen(_scols[i % len(_scols)], width=2))
            for i in range(cfg.n_streams)]

        # ---- population traces ----
        self.p_pop = self.glw.addPlot(row=5, col=0)
        self.p_pop.setMaximumHeight(150)
        self._style_plot(self.p_pop, "Population Activity", "time (s)",
                         "summed E")
        self.p_pop.setXLink(self.p_in)
        t = np.linspace(-cfg.history_s, 0.0, self._F)
        self._t = t
        self.curve_E = self.p_pop.plot(t, self._popE, pen=pg.mkPen(_E, width=2))
        self.vb_act = pg.ViewBox()
        self.p_pop.scene().addItem(self.vb_act)
        self.p_pop.getAxis("right").linkToView(self.vb_act)
        self.p_pop.showAxis("right")
        self.p_pop.getAxis("right").setLabel("active ch", color=_ACT)
        self.p_pop.getAxis("right").setPen(_GRID)
        self.p_pop.getAxis("right").setTextPen(_ACT)
        self.vb_act.setXLink(self.p_in)
        self.vb_act.enableAutoRange(x=False, y=False)
        self.vb_act.setXRange(-cfg.history_s, 0.0, padding=0)
        self.vb_act.setYRange(0.0, 100.0, padding=0)
        self.curve_act = pg.PlotDataItem(t, self._active,
                                         pen=pg.mkPen(_ACT, width=1.5))
        self.vb_act.addItem(self.curve_act)
        self.vb_act.setMaximumHeight(150)
        self.p_pop.getViewBox().sigResized.connect(
            lambda: self.vb_act.setGeometry(
                self.p_pop.getViewBox().sceneBoundingRect()))

        # layout proportions: heatmaps (col 0) wide; W panel (col 2) sized so
        # its rowspan-2 cell is ~square (it spans the two heatmap rows).
        self.glw.ci.layout.setRowStretchFactor(3, 6)
        self.glw.ci.layout.setRowStretchFactor(4, 6)
        self.glw.ci.layout.setRowStretchFactor(5, 2)
        self.glw.ci.layout.setColumnStretchFactor(0, 10)
        self.glw.ci.layout.setColumnStretchFactor(2, 6)
        self._refresh_images()
        self._install_shortcuts()

    def _setup_heat(self, plot, img, title, freqs):
        cfg = self.cfg
        plot.addItem(img)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.setDefaultPadding(0.0)
        vb = plot.getViewBox()
        vb.setBackgroundColor(_PANEL)
        vb.enableAutoRange(x=False, y=False)
        plot.setRange(xRange=(-cfg.history_s, 0.0),
                      yRange=(0, cfg.n_channels), padding=0)
        plot.setTitle(title, color=_FG, size="11pt", bold=True)
        # left axis: mel channel; right axis: tonotopic centre freq (kHz)
        plot.setLabel("left", "mel channel", color=_MUTED)
        plot.getAxis("left").setPen(_GRID)
        plot.getAxis("left").setTextPen(_MUTED)
        plot.getAxis("bottom").setPen(_GRID)
        plot.getAxis("bottom").setTextPen(_MUTED)
        idx = np.linspace(0, cfg.n_channels - 1, 6).astype(int)
        rax = plot.getAxis("right")
        rax.setTicks([[(int(i), f"{freqs[i] / 1000:.1f}") for i in idx]])
        rax.setLabel("kHz", color=_MUTED)
        rax.setPen(_GRID)
        rax.setTextPen(_MUTED)
        plot.showAxis("right")

    def _style_plot(self, plot, title, xlabel, ylabel):
        plot.setTitle(title, color=_FG, size="11pt", bold=True)
        plot.setLabel("bottom", xlabel, color=_MUTED)
        plot.setLabel("left", ylabel, color=_E)
        plot.getViewBox().setBackgroundColor(_PANEL)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        for a in ("left", "bottom"):
            plot.getAxis(a).setPen(_GRID)
        plot.getAxis("bottom").setTextPen(_MUTED)
        plot.getAxis("left").setTextPen(_E)
        plot.setXRange(-self.cfg.history_s, 0.0, padding=0)

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
        # unpaced (mic): drain whatever the callback queued
        self._last_read = now
        return self.source.read()

    def _tick(self):
        if self.paused:
            return
        samples = self._read_samples()
        drive, db = self.fe.push(samples)
        k = drive.shape[1]
        if k:
            out = self.engine.step_block(drive)
            E = out["E"]
            self._push_cols(self._in_db, db.astype(np.float32))
            self._push_cols(self._E, E.astype(np.float32))
            self._push_cols(self._drive, drive.astype(np.float32))
            thr = 0.05 * max(self._E_vmax, 1e-6)
            popE = E.sum(axis=0)
            active = (E > thr).sum(axis=0).astype(np.float32)
            self._push_1d(self._popE, popE.astype(np.float32))
            self._push_1d(self._active, active)
            self._refresh_images()

        # fps estimate
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

    def _update_cortex_scale(self):
        """Track the colour ceiling toward the 90th percentile of the active
        cortical code so the sparse response blooms vividly (the strongest
        few channels saturate) instead of rendering uniformly dim."""
        pos = self._E[self._E > 1e-3]
        if pos.size > 16:
            target = float(np.percentile(pos, 90))
            self._E_vmax = max(0.85 * self._E_vmax + 0.15 * target, 0.05)
        else:
            self._E_vmax = max(0.97 * self._E_vmax, 0.05)

    def _update_W_scale(self):
        """Track the W colour ceiling toward the 99.5th percentile so the
        forming assemblies bloom vividly as the weights grow from ~0."""
        w = self.engine.W
        p = float(np.percentile(w, 99.5)) if w.size else 0.0
        self._W_vmax = max(0.9 * self._W_vmax + 0.1 * p, 1e-3)

    def _update_coincidence(self):
        """C[i,j] = correlation of the chord-binned cortical response E over the
        recent window -- the live temporal-coherence matrix.  E is binned at the
        chord timescale first, so C measures co-occupancy of chords (the figure
        cue) rather than the shared within-chord onset (which would make
        everything correlate).  Throttled: the only O(N^2 * bins) step."""
        cs = max(1, int(round(self.cfg.coh_bin_ms / self.cfg.frame_ms)))   # frames/chord
        nb = min(self._F // cs, int(self.cfg.coh_window_s * 1000.0 / self.cfg.coh_bin_ms))
        if nb < 8:
            return
        rec = self._drive[:, -nb * cs:]
        binned = rec.reshape(rec.shape[0], nb, cs).mean(2)     # (N, nb) chord energies
        binned = binned - binned.mean(0, keepdims=True)        # remove common-mode
        x = binned - binned.mean(1, keepdims=True)             # center each channel
        nrm = np.sqrt((x * x).sum(1))
        denom = np.outer(nrm, nrm)
        C = np.where(denom > 1e-9, (x @ x.T) / (denom + 1e-12), 0.0)
        # nPCA: factor the coincidence matrix (diagonal ~1) into stream masks
        self.sep.update(np.clip(C, 0.0, 1.0))
        self._masks = self.sep.masks().astype(np.float32)
        np.fill_diagonal(C, 0.0)                    # drop trivial self-correlation
        self._C = np.clip(C, 0.0, 1.0).astype(np.float32)
        p = float(np.percentile(self._C, 99.7))
        self._C_vmax = max(0.85 * self._C_vmax + 0.15 * p, 0.05)

    def _refresh_images(self):
        self._update_cortex_scale()
        self.img_in.setImage(self._in_db, autoLevels=False,
                             levels=(-self.cfg.top_db, 0.0))
        self.img_ctx.setImage(self._E, autoLevels=False,
                              levels=(0.0, self._E_vmax))
        self._coh_tick += 1
        if self._coh_tick % 6 == 0:                 # ~10 Hz, not every frame
            self._update_coincidence()
        self.img_C.setImage(self._C, autoLevels=False, levels=(0.0, self._C_vmax))
        self.bar_C.setLevels((0.0, self._C_vmax))
        _x = np.arange(self.cfg.n_channels)
        for i, cv in enumerate(self.curve_masks):
            cv.setData(_x, self._masks[:, i])
        # (re)anchor the images into plot coordinates -- setImage on an
        # empty item leaves an identity transform, so apply the rect here.
        self.img_in.setRect(self._img_rect)
        self.img_ctx.setRect(self._img_rect)
        self.bar_ctx.setLevels((0.0, self._E_vmax))
        self.curve_E.setData(self._t, self._popE)
        self.curve_act.setData(self._t, self._active)
        amax = float(self._active.max())
        if amax > 0:
            self.vb_act.setYRange(0.0, max(5.0, amax * 1.1), padding=0)

    def _update_stats(self):
        lvl = self.fe.level_db
        gate = self.fe.gate_open
        nact = int(self._active[-1]) if self._F else 0
        spars = 100.0 * nact / self.cfg.n_channels
        learn_c = _OK if self._learn else _OFF
        gate_c = _OK if gate else _MUTED
        paused_badge = (f"<b style='color:{_ACT}'>⏸ PAUSED</b> &nbsp;|&nbsp; "
                        if self.paused else "")
        self.stats.setText(
            paused_badge +
            f"<span style='color:{gate_c}'>●</span> "
            f"<span style='color:{_MUTED}'>input</span> "
            f"<b>{lvl:+5.1f} dB</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>inhibition</span> "
            f"<b style='color:{_E}'>{self._inhibition}</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>plasticity</span> "
            f"<b style='color:{learn_c}'>{'ON' if self._learn else 'OFF'}</b> "
            f"&nbsp;|&nbsp; <span style='color:{_MUTED}'>active</span> "
            f"<b style='color:{_ACT}'>{nact}/{self.cfg.n_channels}</b> "
            f"({spars:.0f}%) &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>peak E</span> "
            f"<b>{self._E_vmax:.2f}</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>{self._fps_ema:.0f} fps</span>")

    # -----------------------------------------------------------------
    #  Controls
    # -----------------------------------------------------------------
    def _carry_state(self, new_engine, old):
        new_engine.E[:] = old.E
        new_engine.Iv[:] = old.Iv
        new_engine.tr[:] = old.tr
        new_engine.W[:] = old.W
        if new_engine.D_std is not None and old.D_std is not None:
            new_engine.D_std[:] = old.D_std

    def _install_shortcuts(self):
        """App-level shortcuts so the keys work no matter which child widget
        (e.g. the pyqtgraph view) holds focus -- a plain keyPressEvent on the
        window is easily swallowed by the focused GraphicsView."""
        binds = [("Space", self.toggle_pause), ("L", self.toggle_learning),
                 ("I", self.toggle_inhibition),
                 ("R", self.reset), ("Q", self.close)]
        self._shortcuts = []
        for seq, fn in binds:
            sc = _QShortcut(QtGui.QKeySequence(seq), self)
            try:
                sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            except AttributeError:               # PyQt5 enum location
                sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            # resume cleanly: don't replay the audio that piled up while paused
            self._last_read = time.perf_counter()
            try:
                if not getattr(self.source, "paced", False):
                    self.source.read()           # drain the mic backlog
            except Exception:
                pass
        self._update_stats()

    def toggle_learning(self):
        self._learn = not self._learn
        self.engine.learn = self._learn

    def toggle_inhibition(self):
        self._inhibition = ("uniform" if self._inhibition == "selective"
                            else "selective")
        new_cfg = self.cfg.replace(inhibition=self._inhibition)
        new_engine = LiveEngine(new_cfg.to_a1_config(), learn=self._learn,
                                W_init=self.engine.W, seed=0)
        self._carry_state(new_engine, self.engine)
        self.engine = new_engine
        self.cfg = new_cfg

    def reset(self):
        self.engine = LiveEngine(self.cfg.to_a1_config(), learn=self._learn,
                                 seed=0)
        self.fe.reset()
        self.sep.reset()
        self._in_db[:] = -self.cfg.top_db
        self._E[:] = 0.0
        self._drive[:] = 0.0
        self._popE[:] = 0.0
        self._active[:] = 0.0
        self._E_vmax = 1.0
        self._C[:] = 0.0
        self._masks[:] = 0.0
        self._refresh_images()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self.close()
        elif k == QtCore.Qt.Key.Key_L:
            self.toggle_learning()
        elif k == QtCore.Qt.Key.Key_I:
            self.toggle_inhibition()
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
        """Synchronously push a whole audio array through the pipeline and
        refresh the view once (no timer / no real-time pacing).  Used to
        produce a screenshot of the real GUI headlessly."""
        bs = self.cfg.blocksize
        for lo in range(0, audio.size, bs):
            drive, db = self.fe.push(audio[lo:lo + bs])
            if drive.shape[1] == 0:
                continue
            out = self.engine.step_block(drive)
            E = out["E"]
            self._push_cols(self._in_db, db.astype(np.float32))
            self._push_cols(self._E, E.astype(np.float32))
            self._push_cols(self._drive, drive.astype(np.float32))
            thr = 0.05 * max(self._E_vmax, 1e-3)
            self._push_1d(self._popE, E.sum(axis=0).astype(np.float32))
            self._push_1d(self._active, (E > thr).sum(axis=0).astype(np.float32))
        for _ in range(30):          # converge the smoothed colour ceiling
            self._update_cortex_scale()
        self._update_coincidence()   # not throttled here (single offline refresh)
        self._refresh_images()
        self._update_stats()

    def grab_image(self, path: str):
        """Save a screenshot of the window (works with the offscreen
        Qt platform for headless rendering)."""
        QtWidgets.QApplication.processEvents()
        self.glw.grab().save(path)
        return path
