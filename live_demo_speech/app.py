"""
live_demo_speech.app
====================

Real-time **two-talker speech segregation** by pitch / periodicity (Licklider
1951; Meddis & Hewitt 1992; Wang & Brown CASA) -- the cue that actually
separates concurrent voices.

Pipeline
--------
    mixture → log-frequency cochleagram                (display)
            → cochlear band-pass + rectify + low-pass
            → per-channel autocorrelation              (brainstem periodicity)
            → summary autocorrelation → two F0 tracks
            → route each channel, each moment, to the F0 it is periodic at
            → time-frequency talker masks

The shared slow envelope of two talkers is a common mode that defeats
envelope-coherence grouping; **periodicity** (harmonicity / common F0) is what
binds each voice.  The mask is time-resolved: a channel is routed to whichever
of the two fundamentals its *local* autocorrelation supports right now.

Six panels: input cochleagram, pitch (summary autocorrelation, two F0 tracks),
Talker 1, Talker 2, periodicity coincidence (channels sharing F0), per-channel
talker assignment.

Keyboard:  Space pause · R reset · Q/Esc quit
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .audio import SpectroFrontEnd
from .config import LiveConfig
from .periodicity import PeriodicitySeparator

try:
    _QShortcut = QtGui.QShortcut
except AttributeError:
    _QShortcut = QtWidgets.QShortcut

_BG = "#0e1117"
_PANEL = "#161b22"
_FG = "#c9d1d9"
_MUTED = "#8b949e"
_GRID = "#30363d"
_T1 = "#ff8c42"          # talker 1 (orange, lower F0)
_T2 = "#58d6ff"          # talker 2 (cyan, higher F0)
_OK = "#3fb950"


class LiveDemoApp(QtWidgets.QMainWindow):
    """Scrolling two-talker speech segregation by periodicity."""

    def __init__(self, cfg: LiveConfig, source, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.source = source
        self.paused = False

        self.fe = SpectroFrontEnd(cfg)
        self.psep = PeriodicitySeparator(cfg, n_streams=cfg.n_streams)

        F = cfg.history_frames
        N = cfg.n_channels
        self._F = F
        self._nlags = self.psep._lags.size
        self._f0_axis = self.psep.fs2 / self.psep._lags        # Hz per lag bin
        self._in_db = np.full((N, F), -cfg.top_db, dtype=np.float32)
        self._m1 = np.full((N, F), 0.5, dtype=np.float32)      # talker-1 soft mask
        self._sacf = np.zeros((self._nlags, F), dtype=np.float32)
        self._C = np.zeros((N, N), dtype=np.float32)
        self._cur_m1 = np.full(N, 0.5, dtype=np.float32)
        self._cur_sacf = np.zeros(self._nlags, dtype=np.float32)
        self._C_vmax = 1e-2
        self._sacf_vmax = 1e-2
        self._tick_i = 0
        self._last_read = None
        self._fps_ema = float(cfg.target_fps)
        self._last_tick = time.perf_counter()

        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(int(1000 / max(1, cfg.target_fps)))

    # -----------------------------------------------------------------
    #  UI
    # -----------------------------------------------------------------
    def _build_ui(self):
        cfg = self.cfg
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
        self.setWindowTitle("Speech segregation (live)")
        self.resize(1360, 900)
        self.setStyleSheet(f"background-color:{_BG};")
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(_BG)
        self.setCentralWidget(self.glw)

        self._img_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                       cfg.n_channels)
        self._sacf_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                        self._nlags)

        self.title = self.glw.addLabel(
            "Speech Stream Segregation — pitch / periodicity (concurrent voices)",
            row=0, col=0, colspan=4, color=_FG, size="19pt", bold=True)
        self.subtitle = self.glw.addLabel(
            "cochlea → per-channel autocorrelation → two F0 tracks → route each "
            "channel to its fundamental · real time",
            row=1, col=0, colspan=4, color=_MUTED, size="10pt")
        self.stats = self.glw.addLabel("", row=2, col=0, colspan=4, color=_FG,
                                       size="10.5pt")

        spec_cmap = pg.colormap.get(cfg.input_cmap, source="matplotlib")
        pitch_cmap = pg.colormap.get(cfg.pitch_cmap, source="matplotlib")
        centers = self.fe.center_freqs()
        dbl = (-cfg.top_db, 0.0)

        # left column (shared time axis)
        self.p_in = self.glw.addPlot(row=3, col=0)
        self.img_in = pg.ImageItem(); self.img_in.setColorMap(spec_cmap)
        self._heat(self.p_in, self.img_in,
                   "Input cochleagram  (two-talker mixture)", centers,
                   cfg.n_channels)
        self.img_in.setLevels(dbl)
        self._add_cbar(self.img_in, dbl, spec_cmap, "dB", row=3)

        self.p_pitch = self.glw.addPlot(row=4, col=0)
        self.img_pitch = pg.ImageItem(); self.img_pitch.setColorMap(pitch_cmap)
        self._heat(self.p_pitch, self.img_pitch,
                   "Pitch  (summary autocorrelation: two F0 tracks)",
                   self._f0_axis, self._nlags, left="F0")
        self.bar_pitch = self._add_cbar(self.img_pitch, (0.0, self._sacf_vmax),
                                        pitch_cmap, "salience", row=4)

        self.p_t1 = self.glw.addPlot(row=5, col=0)
        self.img_t1 = pg.ImageItem(); self.img_t1.setColorMap(spec_cmap)
        self._heat(self.p_t1, self.img_t1, "Talker 1  (lower F0)", centers,
                   cfg.n_channels, title_color=_T1)
        self.img_t1.setLevels(dbl)

        self.p_t2 = self.glw.addPlot(row=6, col=0)
        self.img_t2 = pg.ImageItem(); self.img_t2.setColorMap(spec_cmap)
        self._heat(self.p_t2, self.img_t2, "Talker 2  (higher F0)", centers,
                   cfg.n_channels, title_color=_T2, xlabel="time (s)")
        self.img_t2.setLevels(dbl)

        for p in (self.p_pitch, self.p_t1, self.p_t2):
            p.setXLink(self.p_in)

        # right column (channel axis)
        self.p_C, self.img_C, self.bar_C = self._mk_matrix(
            3, "Periodicity coincidence  (channels sharing F0)", "magma", "corr")

        self.p_masks = self.glw.addPlot(row=5, col=2, rowspan=2)
        self.p_masks.setTitle("talker assignment  (per channel, now)",
                              color=_FG, size="11pt", bold=True)
        self.p_masks.setLabel("bottom", "channel", color=_MUTED)
        self.p_masks.setLabel("left", "P(talker)", color=_MUTED)
        self.p_masks.setMouseEnabled(x=False, y=False)
        self.p_masks.hideButtons(); self.p_masks.setMenuEnabled(False)
        self.p_masks.getViewBox().setBackgroundColor(_PANEL)
        for _a in ("left", "bottom"):
            self.p_masks.getAxis(_a).setPen(_GRID)
            self.p_masks.getAxis(_a).setTextPen(_MUTED)
        self.p_masks.setRange(xRange=(0, cfg.n_channels), yRange=(0, 1.05),
                              padding=0)
        self.curve_m1 = self.p_masks.plot(pen=pg.mkPen(_T1, width=2),
                                          fillLevel=0.0,
                                          brush=pg.mkBrush(_T1 + "40"))
        self.curve_m2 = self.p_masks.plot(pen=pg.mkPen(_T2, width=2))

        for r in (3, 4, 5, 6):
            self.glw.ci.layout.setRowStretchFactor(r, 4)
        self.glw.ci.layout.setColumnStretchFactor(0, 10)
        self.glw.ci.layout.setColumnStretchFactor(2, 7)
        self._refresh_images()
        self._install_shortcuts()

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

    def _add_cbar(self, img, values, cmap, label, row):
        bar = pg.ColorBarItem(values=values, colorMap=cmap, label=label, width=14)
        bar.setImageItem(img)
        self.glw.addItem(bar, row=row, col=1)
        self._style_cbar(bar)
        return bar

    def _mk_matrix(self, row, title, cmap_name, label):
        cfg = self.cfg
        cmap = pg.colormap.get(cmap_name, source="matplotlib")
        p = self.glw.addPlot(row=row, col=2, rowspan=2)
        img = pg.ImageItem(); img.setColorMap(cmap); p.addItem(img)
        p.setTitle(title, color=_FG, size="11pt", bold=True)
        p.setLabel("left", "channel", color=_MUTED)
        p.setLabel("bottom", "channel", color=_MUTED)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons(); p.setMenuEnabled(False); p.setDefaultPadding(0.0)
        vb = p.getViewBox(); vb.setBackgroundColor(_BG); vb.setAspectLocked(True)
        for _a in ("left", "bottom"):
            p.getAxis(_a).setPen(_GRID)
            p.getAxis(_a).setTextPen(_MUTED)
            p.getAxis(_a).setTicks(
                [[(v, str(v)) for v in range(0, cfg.n_channels + 1, 24)]])
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

    def _ingest(self, samples):
        """Push one audio chunk through the cochleagram + periodicity analyzer
        and append the (time-resolved) result to the scrolling buffers."""
        drive, db = self.fe.push(samples)
        self.psep.push(samples)
        k = db.shape[1]
        if k == 0:
            return
        self._tick_i += 1
        if self._tick_i % 2 == 0:                       # re-solve periodicity often
            _, sacf, C = self.psep.compute()
            self._cur_m1 = self.psep.masks[:, 0].copy()
            self._cur_sacf = sacf
            self._C = C
            p = float(np.percentile(C, 99.5))
            self._C_vmax = max(0.85 * self._C_vmax + 0.15 * p, 0.05)
            ps = float(np.percentile(sacf, 99.0)) if sacf.any() else 0.0
            self._sacf_vmax = max(0.8 * self._sacf_vmax + 0.2 * ps, 1e-2)
        self._push_cols(self._in_db, db.astype(np.float32))
        self._push_cols(self._m1, np.repeat(self._cur_m1[:, None], k, axis=1))
        self._push_cols(self._sacf,
                        np.repeat(self._cur_sacf[:, None], k, axis=1))

    def _tick(self):
        if self.paused:
            return
        self._ingest(self._read_samples())
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

    # -----------------------------------------------------------------
    #  Refresh
    # -----------------------------------------------------------------
    def _talker_db(self, m):
        """Cochleagram soft-gated by the time-resolved mask m (N, F)."""
        return (self._in_db * m - self.cfg.top_db * (1.0 - m)).astype(np.float32)

    def _refresh_images(self):
        dbl = (-self.cfg.top_db, 0.0)
        self.img_in.setImage(self._in_db, autoLevels=False, levels=dbl)
        # SACF pitchgram: subtract per-frame floor so the F0 tracks stand out
        disp = np.maximum(self._sacf - np.median(self._sacf, axis=0,
                                                 keepdims=True), 0.0)
        self.img_pitch.setImage(disp, autoLevels=False,
                                levels=(0.0, self._sacf_vmax))
        self.bar_pitch.setLevels((0.0, self._sacf_vmax))
        self.img_t1.setImage(self._talker_db(self._m1), autoLevels=False,
                             levels=dbl)
        self.img_t2.setImage(self._talker_db(1.0 - self._m1), autoLevels=False,
                             levels=dbl)
        self.img_C.setImage(self._C, autoLevels=False, levels=(0.0, self._C_vmax))
        self.bar_C.setLevels((0.0, self._C_vmax))
        _x = np.arange(self.cfg.n_channels)
        self.curve_m1.setData(_x, self._cur_m1)
        self.curve_m2.setData(_x, 1.0 - self._cur_m1)
        for im in (self.img_in, self.img_t1, self.img_t2):
            im.setRect(self._img_rect)
        self.img_pitch.setRect(self._sacf_rect)

    def _update_stats(self):
        cfg = self.cfg
        lvl = self.fe.level_db
        gate = self.fe.gate_open
        gate_c = _OK if gate else _MUTED
        f0a, f0b = self.psep.f0_hz()
        t1 = int((self._cur_m1 > 0.5).sum())
        paused = (f"<b style='color:{_T1}'>⏸ PAUSED</b> &nbsp;|&nbsp; "
                  if self.paused else "")
        self.stats.setText(
            paused +
            f"<span style='color:{gate_c}'>●</span> "
            f"<span style='color:{_MUTED}'>input</span> <b>{lvl:+5.1f} dB</b>"
            f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>F0</span> "
            f"<b style='color:{_T1}'>{f0a:3.0f}</b> / "
            f"<b style='color:{_T2}'>{f0b:3.0f}</b> Hz &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>talker-1 channels</span> "
            f"<b style='color:{_T1}'>{t1}</b>/{cfg.n_channels} &nbsp;|&nbsp; "
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
        self.psep.reset()
        self._in_db[:] = -self.cfg.top_db
        self._m1[:] = 0.5
        self._sacf[:] = 0.0
        self._C[:] = 0.0
        self._cur_m1[:] = 0.5
        self._cur_sacf[:] = 0.0
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
    #  Offline driving (headless snapshot / validation)
    # -----------------------------------------------------------------
    def feed_offline(self, audio: np.ndarray):
        bs = self.cfg.blocksize
        for lo in range(0, audio.size, bs):
            self._ingest(audio[lo:lo + bs])
        self._refresh_images()
        self._update_stats()

    def grab_image(self, path: str):
        QtWidgets.QApplication.processEvents()
        self.glw.grab().save(path)
        return path
