"""
live_demo_cortical.app
=============

Real-time **stream segregation** visualiser (temporal-coherence framework;
Krishnan, Elhilali & Shamma, PLoS Comp Biol 2014; Teki et al., eLife 2013).

A mixture of overlapping sounds enters as a log-frequency spectrogram.  A
channel x channel **coincidence matrix** C accumulates which channels are
co-active over a short window -- the temporal-coherence cue.  Factoring C
(nPCA, here normalized spectral clustering) yields one **mask** per stream;
applying each mask back to the spectrogram recovers the **segregated streams**.

Six panels (no model / no "surprise" -- this view is segregation only):

    input log-spectrogram   the mixture (dB)
    pitch                   subharmonic-summation pitch-gram
    stream 1                mixture x mask_1   (orange)
    stream 2                mixture x mask_2   (cyan)
    connections  (C)        the coincidence matrix
    masks                   the nPCA per-channel stream masks

Built on pyqtgraph for smooth scrolling; the coincidence matrix and masks
re-solve at ~10 Hz while the heatmaps scroll every frame.

Keyboard
--------
    Space  pause / resume
    R      reset coincidence + display
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
_S1 = "#ff8c42"          # stream 1 accent (orange)
_S2 = "#58d6ff"          # stream 2 accent (cyan)
_OK = "#3fb950"
_OFF = "#f85149"
_STREAM_COLORS = [_S1, _S2, "#b072ff", "#3fb950"]


class LiveDemoApp(QtWidgets.QMainWindow):
    """Main window: input + pitch + two segregated streams, beside the live
    coincidence matrix and its nPCA stream masks."""

    def __init__(self, cfg: LiveConfig, source, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.source = source
        self.paused = False

        self.fe = SpectroFrontEnd(cfg)
        self.pitch = PitchGram(self.fe.center_freqs(), n_pitch=cfg.n_pitch,
                               fmin=cfg.pitch_fmin, fmax=cfg.pitch_fmax,
                               n_harm=cfg.pitch_harmonics, decay=cfg.pitch_decay)
        self.sep = StreamSeparator(cfg.n_channels, n_streams=cfg.n_streams,
                                   iters=cfg.sep_iters)

        # ---- scrolling ring buffers ----
        F = cfg.history_frames
        N = cfg.n_channels
        P = cfg.n_pitch
        self._F = F
        self._in_db = np.full((N, F), -cfg.top_db, dtype=np.float32)
        self._drive = np.zeros((N, F), dtype=np.float32)   # gated input (for C)
        self._pitch = np.zeros((P, F), dtype=np.float32)   # pitch-gram
        self._C = np.zeros((N, N), dtype=np.float32)       # coincidence matrix
        self._masks = np.zeros((N, cfg.n_streams), dtype=np.float32)
        self._C_vmax = 1e-2
        self._pitch_vmax = 1e-2
        self._coh_tick = 0                                  # throttle C re-solve
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
        self.setWindowTitle("Stream segregation (live)")
        self.resize(1360, 900)
        self.setStyleSheet(f"background-color:{_BG};")

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(_BG)
        self.setCentralWidget(self.glw)

        # image -> plot rects: x in [-history, 0] s; y in [0, N] or [0, n_pitch]
        self._img_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                       cfg.n_channels)
        self._pitch_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                         cfg.n_pitch)

        # ---- header ----
        self.title = self.glw.addLabel(
            "Stream Segregation — temporal coherence (nPCA of the "
            "coincidence matrix)", row=0, col=0, colspan=4, color=_FG,
            size="20pt", bold=True)
        self.subtitle = self.glw.addLabel(
            "log-spectrogram → coincidence C → nPCA masks → segregated "
            "streams · real time", row=1, col=0, colspan=4, color=_MUTED,
            size="10pt")
        self.stats = self.glw.addLabel("", row=2, col=0, colspan=4,
                                       color=_FG, size="10.5pt")

        spec_cmap = pg.colormap.get(cfg.input_cmap, source="matplotlib")
        pitch_cmap = pg.colormap.get(cfg.pitch_cmap, source="matplotlib")
        centers = self.fe.center_freqs()
        pfreqs = self.pitch.pitch_freqs()
        dbl = (-cfg.top_db, 0.0)

        # ---- LEFT COLUMN: time-domain heatmaps (shared time axis) ----
        # input log-spectrogram
        self.p_in = self.glw.addPlot(row=3, col=0)
        self.img_in = pg.ImageItem(); self.img_in.setColorMap(spec_cmap)
        self._heat(self.p_in, self.img_in, "Input log-spectrogram  (mixture)",
                   centers, cfg.n_channels)
        self.img_in.setLevels(dbl)
        self._add_cbar(self.img_in, dbl, spec_cmap, "dB", row=3)

        # pitch-gram
        self.p_pitch = self.glw.addPlot(row=4, col=0)
        self.img_pitch = pg.ImageItem(); self.img_pitch.setColorMap(pitch_cmap)
        self._heat(self.p_pitch, self.img_pitch, "Pitch  (subharmonic summation)",
                   pfreqs, cfg.n_pitch, left="pitch F0")
        self.bar_pitch = self._add_cbar(self.img_pitch, (0.0, self._pitch_vmax),
                                        pitch_cmap, "salience", row=4)

        # segregated stream 1
        self.p_str1 = self.glw.addPlot(row=5, col=0)
        self.img_str1 = pg.ImageItem(); self.img_str1.setColorMap(spec_cmap)
        self._heat(self.p_str1, self.img_str1, "Stream 1", centers,
                   cfg.n_channels, title_color=_S1)
        self.img_str1.setLevels(dbl)

        # segregated stream 2
        self.p_str2 = self.glw.addPlot(row=6, col=0)
        self.img_str2 = pg.ImageItem(); self.img_str2.setColorMap(spec_cmap)
        self._heat(self.p_str2, self.img_str2, "Stream 2", centers,
                   cfg.n_channels, title_color=_S2, xlabel="time (s)")
        self.img_str2.setLevels(dbl)

        for p in (self.p_pitch, self.p_str1, self.p_str2):
            p.setXLink(self.p_in)

        # ---- RIGHT COLUMN: channel-domain panels ----
        # coincidence matrix C (square), spanning the input+pitch rows
        self.p_C, self.img_C, self.bar_C = self._mk_matrix(
            3, "Connections   C   (coincidence)", "magma", "corr")

        # nPCA stream masks (one curve per stream, over channel = log-frequency),
        # spanning the two stream rows
        self.p_masks = self.glw.addPlot(row=5, col=2, rowspan=2)
        self.p_masks.setTitle("nPCA stream masks", color=_FG, size="11pt",
                              bold=True)
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
        self.curve_masks = [
            self.p_masks.plot(
                pen=pg.mkPen(_STREAM_COLORS[i % len(_STREAM_COLORS)], width=2),
                fillLevel=0.0,
                brush=pg.mkBrush(_STREAM_COLORS[i % len(_STREAM_COLORS)] + "40"))
            for i in range(cfg.n_streams)]

        # layout proportions: the four time heatmaps (col 0) take most width;
        # the square matrix + masks (col 2) a narrower right column.
        for r in (3, 4, 5, 6):
            self.glw.ci.layout.setRowStretchFactor(r, 4)
        self.glw.ci.layout.setColumnStretchFactor(0, 10)
        self.glw.ci.layout.setColumnStretchFactor(2, 7)

        self._refresh_images()
        self._install_shortcuts()

    # ---- small UI helpers ----
    def _heat(self, plot, img, title, right_freqs, ny, *, left="channel",
              xlabel="", title_color=_FG):
        """Configure a scrolling time x channel heatmap with a tonotopic
        (kHz) right axis."""
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
        # right axis: tonotopic centre frequency in kHz
        fmt = "%.2f" if float(np.max(right_freqs)) < 1000.0 else "%.1f"
        idx = np.linspace(0, ny - 1, 6).astype(int)
        rax = plot.getAxis("right")
        rax.setTicks([[(int(i), fmt % (right_freqs[i] / 1000.0)) for i in idx]])
        rax.setLabel("kHz", color=_MUTED)
        rax.setPen(_GRID); rax.setTextPen(_MUTED)
        plot.showAxis("right")

    def _add_cbar(self, img, values, cmap, label, row):
        bar = pg.ColorBarItem(values=values, colorMap=cmap, label=label,
                              width=14)
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
        vb = p.getViewBox()
        vb.setBackgroundColor(_BG)          # letterbox blends into canvas
        vb.setAspectLocked(True)            # 1:1 pixels -> true square
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
        # unpaced (mic): drain whatever the callback queued
        self._last_read = now
        return self.source.read()

    def _tick(self):
        if self.paused:
            return
        samples = self._read_samples()
        drive, db = self.fe.push(samples)
        if drive.shape[1]:
            self._push_cols(self._in_db, db.astype(np.float32))
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

    def _update_coincidence(self):
        """C[i,j] = correlation of the chord-binned input over the recent
        window -- the live temporal-coherence matrix.  The drive is binned at
        the chord timescale first (so C measures co-occupancy of chords, the
        figure cue, not the shared within-chord onset), the common mode is
        removed, then nPCA factors C into stream masks.  The only O(N^2*bins)
        step -- throttled to ~10 Hz."""
        cfg = self.cfg
        cs = max(1, int(round(cfg.coh_bin_ms / cfg.frame_ms)))       # frames/chord
        nb = min(self._F // cs,
                 int(cfg.coh_window_s * 1000.0 / cfg.coh_bin_ms))
        if nb < 8:
            return
        rec = self._drive[:, -nb * cs:]
        binned = rec.reshape(rec.shape[0], nb, cs).mean(2)           # (N, nb)
        binned = binned - binned.mean(0, keepdims=True)              # common-mode
        x = binned - binned.mean(1, keepdims=True)                  # center channels
        nrm = np.sqrt((x * x).sum(1))
        denom = np.outer(nrm, nrm)
        C = np.where(denom > 1e-9, (x @ x.T) / (denom + 1e-12), 0.0)
        np.fill_diagonal(C, 0.0)                    # drop trivial self-correlation
        # denoise: the incoherent background fills C with a floor of rectified
        # sampling noise (~1/sqrt(n_chords)) that otherwise gets absorbed into a
        # stream.  Cut every edge at/below a robust floor (median + z*MAD of the
        # off-diagonal); only the coherent within-stream blocks survive, so nPCA
        # splits A vs B instead of coherent-vs-background.
        iu = np.triu_indices(C.shape[0], 1)
        off = C[iu]
        med = float(np.median(off))
        mad = float(np.median(np.abs(off - med))) + 1e-9
        theta = med + cfg.coh_floor_z * 1.4826 * mad
        C = np.clip(C - theta, 0.0, 1.0)        # soft floor-subtraction
        # nPCA: factor the cleaned coincidence matrix into non-negative masks
        self.sep.update(C)
        self._masks = self.sep.masks().astype(np.float32)
        self._C = C.astype(np.float32)
        p = float(np.percentile(self._C, 99.7))
        self._C_vmax = max(0.85 * self._C_vmax + 0.15 * p, 0.05)

    def _stream_db(self, k):
        """Stream k = the spectrogram soft-gated by mask k: member channels keep
        their dB, non-members fall to the floor (so each stream looks like the
        mixture restricted to its own channels)."""
        m = self._masks[:, k][:, None]
        return (self._in_db * m - self.cfg.top_db * (1.0 - m)).astype(np.float32)

    def _refresh_images(self):
        dbl = (-self.cfg.top_db, 0.0)
        self.img_in.setImage(self._in_db, autoLevels=False, levels=dbl)

        self._coh_tick += 1
        if self._coh_tick % 6 == 0:                 # ~10 Hz, not every frame
            self._update_coincidence()

        # pitch-gram: subtract the per-frame broadband floor (the median over
        # F0) so a dense tone cloud doesn't wash the panel out -- only F0s that
        # stand ABOVE the background periodicity survive -- then scale to a
        # running ceiling.
        disp = np.maximum(self._pitch - np.median(self._pitch, axis=0,
                                                   keepdims=True), 0.0)
        pmax = float(np.percentile(disp, 99.5)) if disp.any() else 0.0
        self._pitch_vmax = max(0.8 * self._pitch_vmax + 0.2 * pmax, 1e-2)
        self.img_pitch.setImage(disp, autoLevels=False,
                                levels=(0.0, self._pitch_vmax))
        self.bar_pitch.setLevels((0.0, self._pitch_vmax))

        # segregated streams = masked spectrogram
        self.img_str1.setImage(self._stream_db(0), autoLevels=False, levels=dbl)
        self.img_str2.setImage(self._stream_db(1), autoLevels=False, levels=dbl)

        # coincidence matrix + masks
        self.img_C.setImage(self._C, autoLevels=False, levels=(0.0, self._C_vmax))
        self.bar_C.setLevels((0.0, self._C_vmax))
        _x = np.arange(self.cfg.n_channels)
        for i, cv in enumerate(self.curve_masks):
            cv.setData(_x, self._masks[:, i])

        # (re)anchor images into plot coordinates
        for im in (self.img_in, self.img_str1, self.img_str2):
            im.setRect(self._img_rect)
        self.img_pitch.setRect(self._pitch_rect)

    def _update_stats(self):
        cfg = self.cfg
        lvl = self.fe.level_db
        gate = self.fe.gate_open
        gate_c = _OK if gate else _MUTED
        nact = int((self._drive[:, -1] > 0.02).sum()) if self._F else 0
        s1 = int((self._masks[:, 0] > 0.5).sum())
        s2 = int((self._masks[:, 1] > 0.5).sum()) if cfg.n_streams > 1 else 0
        # separation index: 1 - cosine overlap of the two masks (1 = disjoint)
        m1, m2 = self._masks[:, 0], self._masks[:, min(1, cfg.n_streams - 1)]
        denom = float(np.linalg.norm(m1) * np.linalg.norm(m2))
        overlap = float(m1 @ m2) / denom if denom > 1e-9 else 0.0
        sep = 100.0 * (1.0 - overlap)
        paused_badge = (f"<b style='color:{_S1}'>⏸ PAUSED</b> &nbsp;|&nbsp; "
                        if self.paused else "")
        self.stats.setText(
            paused_badge +
            f"<span style='color:{gate_c}'>●</span> "
            f"<span style='color:{_MUTED}'>input</span> "
            f"<b>{lvl:+5.1f} dB</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>active</span> "
            f"<b>{nact}/{cfg.n_channels}</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>stream sizes</span> "
            f"<b style='color:{_S1}'>{s1}</b> / "
            f"<b style='color:{_S2}'>{s2}</b> ch &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>separation</span> "
            f"<b style='color:{_OK}'>{sep:3.0f}%</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>{self._fps_ema:.0f} fps</span>")

    # -----------------------------------------------------------------
    #  Controls
    # -----------------------------------------------------------------
    def _install_shortcuts(self):
        """App-level shortcuts so the keys work no matter which child widget
        holds focus."""
        binds = [("Space", self.toggle_pause), ("R", self.reset),
                 ("Q", self.close)]
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
            self._last_read = time.perf_counter()
            try:
                if not getattr(self.source, "paced", False):
                    self.source.read()           # drain the mic backlog
            except Exception:
                pass
        self._update_stats()

    def reset(self):
        self.fe.reset()
        self.sep.reset()
        self._in_db[:] = -self.cfg.top_db
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
        """Synchronously push a whole audio array through the pipeline and
        refresh once (no timer / no real-time pacing) -- for a headless GUI
        screenshot."""
        bs = self.cfg.blocksize
        for lo in range(0, audio.size, bs):
            drive, db = self.fe.push(audio[lo:lo + bs])
            if drive.shape[1] == 0:
                continue
            self._push_cols(self._in_db, db.astype(np.float32))
            self._push_cols(self._drive, drive.astype(np.float32))
            self._push_cols(self._pitch,
                            self.pitch.push(drive).astype(np.float32))
        self._update_coincidence()   # not throttled here (single offline refresh)
        self._refresh_images()
        self._update_stats()

    def grab_image(self, path: str):
        """Save a screenshot of the window (works with the offscreen Qt
        platform for headless rendering)."""
        QtWidgets.QApplication.processEvents()
        self.glw.grab().save(path)
        return path
