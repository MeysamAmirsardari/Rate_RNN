"""
live_demo_speech.app
====================

Real-time **two-talker speech segregation** with the model0 A1 rate RNN, by
temporal coherence (Shamma, Elhilali & Micheyl 2011; Krishnan, Elhilali &
Shamma 2014).

Pipeline
--------
    mixture → log-frequency cochleagram → A1 RNN (E activations)
            → multi-rate envelope coincidence  C[i,j]
            → nPCA (normalized spectral clustering) → talker masks
            → masked cochleagram = each talker's channel group

The binding cue for speech is **common amplitude modulation** of the channel
envelopes (the syllabic/voicing rhythm).  ``C`` is the correlation of the A1
**activation** envelopes over a sliding window, summed across several temporal
**rates** (the cortical modulation-rate filterbank) — the "multiple replications
with different dynamics".  Factoring ``C`` groups channels by talker.

This is temporal-coherence **grouping** (which channels belong to which talker),
the regime the model is built for — not reconstruction-grade source separation
(two talkers share channels over time; only an oracle per-bin mask could do
that).  The masks track each talker's spectral group.

Six panels: input cochleagram, pitch (F0 salience), Talker 1, Talker 2,
coincidence C, nPCA masks.

Keyboard
--------
    Space  pause / resume      R  reset      Q/Esc  quit
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from scipy.signal import lfilter
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .audio import SpectroFrontEnd, PitchGram
from .config import LiveConfig
from .masks import StreamSeparator
from .engine import LiveEngine

try:
    _QShortcut = QtGui.QShortcut
except AttributeError:
    _QShortcut = QtWidgets.QShortcut

_BG = "#0e1117"
_PANEL = "#161b22"
_FG = "#c9d1d9"
_MUTED = "#8b949e"
_GRID = "#30363d"
_T1 = "#ff8c42"          # talker 1 (orange)
_T2 = "#58d6ff"          # talker 2 (cyan)
_OK = "#3fb950"
_TALKER_COLORS = [_T1, _T2, "#b072ff", "#3fb950"]


class LiveDemoApp(QtWidgets.QMainWindow):
    """Scrolling two-talker speech segregation view."""

    def __init__(self, cfg: LiveConfig, source, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.source = source
        self.paused = False

        self.fe = SpectroFrontEnd(cfg)
        self.engine = LiveEngine(cfg.to_a1_config(), learn=cfg.learn, seed=0)
        self.pitch = PitchGram(self.fe.center_freqs(), n_pitch=cfg.n_pitch,
                               fmin=cfg.pitch_fmin, fmax=cfg.pitch_fmax,
                               n_harm=cfg.pitch_harmonics, decay=cfg.pitch_decay)
        self.sep = StreamSeparator(cfg.n_channels, n_streams=cfg.n_streams,
                                   iters=cfg.sep_iters)

        F = cfg.history_frames
        N = cfg.n_channels
        self._F = F
        self._in_db = np.full((N, F), -cfg.top_db, dtype=np.float32)
        self._E = np.zeros((N, F), dtype=np.float32)          # cortex activations
        self._pitch = np.zeros((cfg.n_pitch, F), dtype=np.float32)
        self._C = np.zeros((N, N), dtype=np.float32)
        self._masks = np.zeros((N, cfg.n_streams), dtype=np.float32)
        self._C_vmax = 1e-2
        self._pitch_vmax = 1e-2
        self._coh_tick = 0
        self._last_read = None
        self._fps_ema = float(cfg.target_fps)
        self._last_tick = time.perf_counter()
        # per-frame leak coefficients for the modulation-rate filterbank
        dt = cfg.dt
        self._rate_alphas = [float(np.exp(-dt / max(t, 1e-3)))
                             for t in cfg.coh_rates_s]

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
        self._pitch_rect = QtCore.QRectF(-cfg.history_s, 0.0, cfg.history_s,
                                         cfg.n_pitch)

        self.title = self.glw.addLabel(
            "Speech Stream Segregation — temporal coherence in the A1 model",
            row=0, col=0, colspan=4, color=_FG, size="19pt", bold=True)
        self.subtitle = self.glw.addLabel(
            "two-talker mixture → cochlea → A1 RNN → multi-rate envelope "
            "coincidence → nPCA talker masks · real time",
            row=1, col=0, colspan=4, color=_MUTED, size="10pt")
        self.stats = self.glw.addLabel("", row=2, col=0, colspan=4, color=_FG,
                                       size="10.5pt")

        spec_cmap = pg.colormap.get(cfg.input_cmap, source="matplotlib")
        pitch_cmap = pg.colormap.get(cfg.pitch_cmap, source="matplotlib")
        centers = self.fe.center_freqs()
        pfreqs = self.pitch.pitch_freqs()
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
        self._heat(self.p_pitch, self.img_pitch, "Pitch  (F0 salience)",
                   pfreqs, cfg.n_pitch, left="pitch F0")
        self.bar_pitch = self._add_cbar(self.img_pitch, (0.0, self._pitch_vmax),
                                        pitch_cmap, "salience", row=4)

        self.p_t1 = self.glw.addPlot(row=5, col=0)
        self.img_t1 = pg.ImageItem(); self.img_t1.setColorMap(spec_cmap)
        self._heat(self.p_t1, self.img_t1, "Talker 1", centers, cfg.n_channels,
                   title_color=_T1)
        self.img_t1.setLevels(dbl)

        self.p_t2 = self.glw.addPlot(row=6, col=0)
        self.img_t2 = pg.ImageItem(); self.img_t2.setColorMap(spec_cmap)
        self._heat(self.p_t2, self.img_t2, "Talker 2", centers, cfg.n_channels,
                   title_color=_T2, xlabel="time (s)")
        self.img_t2.setLevels(dbl)

        for p in (self.p_pitch, self.p_t1, self.p_t2):
            p.setXLink(self.p_in)

        # right column (channel axis)
        self.p_C, self.img_C, self.bar_C = self._mk_matrix(
            3, "Coincidence  C  (envelope coherence)", "magma", "corr")

        self.p_masks = self.glw.addPlot(row=5, col=2, rowspan=2)
        self.p_masks.setTitle("nPCA talker masks", color=_FG, size="11pt",
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
                pen=pg.mkPen(_TALKER_COLORS[i % len(_TALKER_COLORS)], width=2),
                fillLevel=0.0,
                brush=pg.mkBrush(_TALKER_COLORS[i % len(_TALKER_COLORS)] + "40"))
            for i in range(cfg.n_streams)]

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
        vb.setBackgroundColor(_BG)
        vb.setAspectLocked(True)
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

    def _tick(self):
        if self.paused:
            return
        samples = self._read_samples()
        drive, db = self.fe.push(samples)
        if drive.shape[1]:
            out = self.engine.step_block(drive)
            self._push_cols(self._in_db, db.astype(np.float32))
            self._push_cols(self._E, out["E"].astype(np.float32))
            inten = np.clip((db + self.cfg.top_db) / self.cfg.top_db, 0.0, 1.0)
            self._push_cols(self._pitch, self.pitch.push(inten).astype(np.float32))
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
    #  Multi-rate envelope coincidence on the A1 activations
    # -----------------------------------------------------------------
    def _update_coincidence(self):
        """C[i,j] = correlation of the A1 activation envelopes over the recent
        window, summed across temporal rates (the modulation-rate filterbank).
        A robust floor (median + z*MAD) is subtracted to drop the shared
        common mode, then nPCA factors C into talker masks."""
        cfg = self.cfg
        nwin = min(self._F, int(cfg.coh_window_s * 1000.0 / cfg.hop_ms))
        if nwin < 200:
            return
        rec = self._E[:, -nwin:]
        N = rec.shape[0]
        C = np.zeros((N, N))
        for al in self._rate_alphas:                 # each rate = one replication
            s = lfilter([1.0 - al], [1.0, -al], rec, axis=1)   # causal EMA
            x = s - s.mean(1, keepdims=True)
            nrm = np.sqrt((x * x).sum(1))
            C += (x @ x.T) / (np.outer(nrm, nrm) + 1e-9)
        C /= len(self._rate_alphas)
        C = np.maximum(C, 0.0)
        np.fill_diagonal(C, 0.0)
        # deflate the common mode: both talkers share the overall speech
        # on/off envelope, which is the (all-positive) leading eigenvector of C.
        # Removing it stops the clustering from splitting speech-vs-silence and
        # exposes the talker-vs-talker contrast underneath.
        w, V = np.linalg.eigh(C)
        C = C - w[-1] * np.outer(V[:, -1], V[:, -1])
        C = np.maximum(C, 0.0)
        np.fill_diagonal(C, 0.0)
        iu = np.triu_indices(N, 1)
        off = C[iu]
        med = float(np.median(off))
        mad = float(np.median(np.abs(off - med))) + 1e-9
        C = np.clip(C - (med + cfg.coh_floor_z * 1.4826 * mad), 0.0, 1.0)
        self.sep.update(C)
        self._masks = self.sep.masks().astype(np.float32)
        self._C = C.astype(np.float32)
        p = float(np.percentile(self._C, 99.5))
        self._C_vmax = max(0.85 * self._C_vmax + 0.15 * p, 0.05)

    def _talker_db(self, k):
        """Talker k = the cochleagram soft-gated by its channel mask."""
        m = self._masks[:, k][:, None]
        return (self._in_db * m - self.cfg.top_db * (1.0 - m)).astype(np.float32)

    def _refresh_images(self):
        dbl = (-self.cfg.top_db, 0.0)
        self.img_in.setImage(self._in_db, autoLevels=False, levels=dbl)
        self._coh_tick += 1
        if self._coh_tick % 6 == 0:
            self._update_coincidence()
        # pitch: subtract per-frame floor so the F0 tracks stand out
        disp = np.maximum(self._pitch - np.median(self._pitch, axis=0,
                                                  keepdims=True), 0.0)
        pmax = float(np.percentile(disp, 99.5)) if disp.any() else 0.0
        self._pitch_vmax = max(0.8 * self._pitch_vmax + 0.2 * pmax, 1e-2)
        self.img_pitch.setImage(disp, autoLevels=False,
                                levels=(0.0, self._pitch_vmax))
        self.bar_pitch.setLevels((0.0, self._pitch_vmax))
        self.img_t1.setImage(self._talker_db(0), autoLevels=False, levels=dbl)
        self.img_t2.setImage(self._talker_db(1), autoLevels=False, levels=dbl)
        self.img_C.setImage(self._C, autoLevels=False, levels=(0.0, self._C_vmax))
        self.bar_C.setLevels((0.0, self._C_vmax))
        _x = np.arange(self.cfg.n_channels)
        for i, cv in enumerate(self.curve_masks):
            cv.setData(_x, self._masks[:, i])
        for im in (self.img_in, self.img_t1, self.img_t2):
            im.setRect(self._img_rect)
        self.img_pitch.setRect(self._pitch_rect)

    def _update_stats(self):
        cfg = self.cfg
        lvl = self.fe.level_db
        gate = self.fe.gate_open
        gate_c = _OK if gate else _MUTED
        t1 = int((self._masks[:, 0] > 0.5).sum())
        t2 = int((self._masks[:, 1] > 0.5).sum()) if cfg.n_streams > 1 else 0
        m1, m2 = self._masks[:, 0], self._masks[:, min(1, cfg.n_streams - 1)]
        denom = float(np.linalg.norm(m1) * np.linalg.norm(m2))
        overlap = float(m1 @ m2) / denom if denom > 1e-9 else 0.0
        grp = 100.0 * (1.0 - overlap)
        paused = (f"<b style='color:{_T1}'>⏸ PAUSED</b> &nbsp;|&nbsp; "
                  if self.paused else "")
        self.stats.setText(
            paused +
            f"<span style='color:{gate_c}'>●</span> "
            f"<span style='color:{_MUTED}'>input</span> <b>{lvl:+5.1f} dB</b>"
            f" &nbsp;|&nbsp; <span style='color:{_MUTED}'>talker groups</span> "
            f"<b style='color:{_T1}'>{t1}</b> / "
            f"<b style='color:{_T2}'>{t2}</b> ch &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>distinctness</span> "
            f"<b style='color:{_OK}'>{grp:3.0f}%</b> &nbsp;|&nbsp; "
            f"<span style='color:{_MUTED}'>rates</span> "
            f"<b>{len(cfg.coh_rates_s)}</b> &nbsp;|&nbsp; "
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
        self.engine = LiveEngine(self.cfg.to_a1_config(), learn=self.cfg.learn,
                                 seed=0)
        self.sep.reset()
        self._in_db[:] = -self.cfg.top_db
        self._E[:] = 0.0
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
    #  Offline driving (headless snapshot / validation)
    # -----------------------------------------------------------------
    def feed_offline(self, audio: np.ndarray):
        bs = self.cfg.blocksize
        for lo in range(0, audio.size, bs):
            drive, db = self.fe.push(audio[lo:lo + bs])
            if drive.shape[1] == 0:
                continue
            out = self.engine.step_block(drive)
            self._push_cols(self._in_db, db.astype(np.float32))
            self._push_cols(self._E, out["E"].astype(np.float32))
            inten = np.clip((db + self.cfg.top_db) / self.cfg.top_db, 0.0, 1.0)
            self._push_cols(self._pitch, self.pitch.push(inten).astype(np.float32))
        self._update_coincidence()
        self._refresh_images()
        self._update_stats()

    def grab_image(self, path: str):
        QtWidgets.QApplication.processEvents()
        self.glw.grab().save(path)
        return path
