"""
stream_enhancement.run
======================

Does a fast coherent stream survive background noise better than the noise
does, purely because of layer 1's learned recurrent weights?

Layer 1 only.  No second layer is involved anywhere in this test.

Protocol
--------
1. Expose the model to a clean sweeping stream and let the recurrent weights
   learn its transitions.
2. Freeze the weights.  Present the same stream buried in background noise.
3. Compare the response with those weights against the response with the
   recurrent pathway removed entirely (W = 0), on the identical stimulus.

Everything is measured on channels that are unambiguously stream or
unambiguously noise; see config.py for why that matters.

Run
---
    python -m stream_enhancement.run
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model0 import selective_inh, simulate
from stream_enhancement.config import StreamConfig

INK, MUTED, GRID = "#22252a", "#8a9099", "#dfe3e8"
C_STREAM, C_NOISE, C_ACC = "#1f6b4a", "#a83f28", "#6b5b95"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "bold", "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "figure.dpi": 110,
})


def _tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


# ---------------------------------------------------------------------
def build(cfg, n_tokens, noisy, dt=1e-3, seed=0, token_ms=None):
    """Sweeping stream, optionally with background noise on the other half."""
    rng = np.random.default_rng(seed)
    nt = int(round((token_ms or cfg.token_ms)))
    T = n_tokens * nt
    stim = np.zeros((cfg.n_channels, T))
    chan = []
    noise_ch = np.arange(cfg.n_stream, cfg.n_channels)
    for k in range(n_tokens):
        c = int(round(cfg.sweep_centre +
                      cfg.sweep_amp * np.sin(2 * np.pi * k / cfg.period_tokens)))
        stim[c, k * nt:(k + 1) * nt] = 1.0
        chan.append(c)
        if noisy:
            for _ in range(cfg.n_noise_per_token):
                stim[rng.choice(noise_ch), k * nt:(k + 1) * nt] += cfg.noise_amp
    return stim, np.array(chan), nt


def ideal_W(cfg, w):
    """The trajectory's transitions at a fixed strength.

    Used for the timing sweeps.  Re-learning at each token duration would
    confound the comparison, because longer tokens learn stronger weights and
    the stronger weights alone change the answer.  The channel sequence does
    not depend on token duration, so the same transition set applies
    throughout and only the timing varies."""
    W = np.zeros((cfg.n_channels, cfg.n_channels))
    ch = [int(round(cfg.sweep_centre +
                    cfg.sweep_amp * np.sin(2 * np.pi * k / cfg.period_tokens)))
          for k in range(cfg.period_tokens + 1)]
    for a, b in zip(ch[:-1], ch[1:]):
        W[b, a] = w
    return W


def a1(cfg, tau_I=None, W_max=None):
    return selective_inh(N=cfg.n_channels, tau_I=tau_I or cfg.tau_I,
                         W_max=W_max or cfg.W_max, W_max_self=W_max or cfg.W_max,
                         W_norm=cfg.W_norm, plastic_self=cfg.plastic_self)


def learn_weights(cfg, token_ms=None, tau_I=None, threshold=True):
    """Learn on a clean stream, then keep only the real transitions.

    The raw learned matrix carries a lot of small off-transition weight.  It is
    harmless at unit strength but, once scaled, contributes loop gain without
    contributing any prediction and drives the network unstable, so it is
    dropped here rather than amplified."""
    clean, chan, _ = build(cfg, cfg.train_tokens, noisy=False, seed=cfg.seed + 1,
                           token_ms=token_ms)
    out = simulate(clean, cfg=a1(cfg, tau_I), learn=True, seed=cfg.seed)
    W = out["W_final"]
    if threshold and W.max() > 0:
        W = W * (W > cfg.w_threshold * W.max())
    return W, chan


def measure(X, chan, nt, cfg):
    """Peak response on the stream channel vs on the noise channels."""
    noise_ch = np.arange(cfg.n_stream, cfg.n_channels)
    sig = np.mean([X[c, k * nt:(k + 1) * nt].max() for k, c in enumerate(chan)])
    noi = np.mean([X[noise_ch, k * nt:(k + 1) * nt].max()
                   for k in range(len(chan))])
    return float(sig), float(noi)


def trial(cfg, W, token_ms=None, tau_I=None, seed=None):
    """One frozen weight comparison on identical noisy input."""
    stim, chan, nt = build(cfg, cfg.test_tokens, noisy=True,
                           seed=cfg.seed + 7 if seed is None else seed,
                           token_ms=token_ms)
    c = a1(cfg, tau_I)
    Z = np.zeros((cfg.n_channels, cfg.n_channels))
    off = simulate(stim, cfg=c, W_init=Z, learn=False, seed=cfg.seed)
    on = simulate(stim, cfg=c, W_init=W, learn=False, seed=cfg.seed)
    s0, n0 = measure(off["E"], chan, nt, cfg)
    s1, n1 = measure(on["E"], chan, nt, cfg)
    stable = np.isfinite(on["E"]).all() and on["E"].max() < 1e3
    return dict(stim=stim, chan=chan, nt=nt, off=off, on=on,
                s0=s0, n0=n0, s1=s1, n1=n1,
                gain_s=s1 / max(s0, 1e-9), gain_n=n1 / max(n0, 1e-9),
                snr=(s1 / max(n1, 1e-9)) / max(s0 / max(n0, 1e-9), 1e-9),
                stable=bool(stable))


# ---------------------------------------------------------------------
def fig_effect(cfg, res, W, w_learn, fname):
    stim, chan, nt = res["stim"], res["chan"], res["nt"]
    T = min(int(60 * nt), stim.shape[1])
    ts = np.arange(T) * 1e-3
    ext = [0, ts[-1], cfg.n_channels - 0.5, -0.5]
    # a high percentile rather than the outright max, so the weaker panel is
    # still legible while both panels keep the same scale
    vmax = float(np.percentile(
        np.concatenate([res["off"]["E"][:, :T].ravel(),
                        res["on"]["E"][:, :T].ravel()]), 99.5))

    fig = plt.figure(figsize=(15.5, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 0.42])
    fig.suptitle("A fast sweeping stream in background noise, with and without "
                 "the learned recurrent pathway", fontsize=14, fontweight="bold")

    def band(ax, col="w"):
        ax.axhline(cfg.n_stream - 0.5, color=col, lw=1.6)
        ax.text(ts[-1] * 0.995, 1.2, "stream channels", color=col, fontsize=9,
                ha="right", fontweight="bold")
        ax.text(ts[-1] * 0.995, cfg.n_stream + 1.6, "noise channels", color=col,
                fontsize=9, ha="right", fontweight="bold")

    ax = fig.add_subplot(gs[0, :3])
    ax.imshow(stim[:, :T], aspect="auto", cmap="bone_r", extent=ext,
              interpolation="nearest")
    band(ax, col=C_ACC)          # dark text: this panel has a light background
    ax.set_ylabel("channel"); ax.set_xticklabels([])
    ax.set_title("a.  The stimulus: a coherent sweep, plus noise on the other half",
                 fontsize=11)

    ax = fig.add_subplot(gs[1, :3])
    ax.imshow(res["off"]["E"][:, :T], aspect="auto", cmap="magma", vmin=0,
              vmax=vmax, extent=ext, interpolation="nearest")
    band(ax)
    ax.set_ylabel("channel"); ax.set_xticklabels([])
    ax.set_title("b.  Response with the recurrent pathway removed (W = 0)",
                 fontsize=11)

    ax = fig.add_subplot(gs[2, :3])
    im = ax.imshow(res["on"]["E"][:, :T], aspect="auto", cmap="magma", vmin=0,
                   vmax=vmax, extent=ext, interpolation="nearest")
    band(ax)
    ax.set_ylabel("channel"); ax.set_xlabel("time (s)")
    ax.set_title("c.  Response with the learned recurrent weights, same stimulus, "
                 "same colour scale", fontsize=11)

    # right column: the numbers
    ax = _tidy(fig.add_subplot(gs[0, 3]))
    ax.bar([0, 1], [res["s0"], res["s1"]], color=[GRID, C_STREAM], width=0.55)
    for x, v in ((0, res["s0"]), (1, res["s1"])):
        ax.text(x, v * 1.02, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["W = 0", "learned"], fontsize=9)
    ax.set_ylabel("peak response")
    ax.set_title(f"d.  Stream  {res['gain_s']:.2f}x", fontsize=10.5,
                 color=C_STREAM)

    ax = _tidy(fig.add_subplot(gs[1, 3]))
    ax.bar([0, 1], [res["n0"], res["n1"]], color=[GRID, C_NOISE], width=0.55)
    for x, v in ((0, res["n0"]), (1, res["n1"])):
        ax.text(x, v * 1.02, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["W = 0", "learned"], fontsize=9)
    ax.set_ylabel("peak response")
    ax.set_title(f"e.  Background  {res['gain_n']:.2f}x", fontsize=10.5,
                 color=C_NOISE)

    ax = _tidy(fig.add_subplot(gs[2, 3]))
    r0, r1 = res["s0"] / res["n0"], res["s1"] / res["n1"]
    ax.bar([0, 1], [r0, r1], color=[GRID, C_ACC], width=0.55)
    for x, v in ((0, r0), (1, r1)):
        ax.text(x, v * 1.02, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["W = 0", "learned"], fontsize=9)
    ax.set_ylabel("stream / background")
    ax.set_title(f"f.  Separation  {res['snr']:.2f}x", fontsize=10.5, color=C_ACC)

    _caption(fig, gs[3, :],
             f"Panels b and c show the same stimulus on the same colour scale; only the recurrent weights differ.\n"
             f"The stream gains {res['gain_s']:.2f}x while the background falls to {res['gain_n']:.2f}x, so the separation between them "
             f"improves {res['snr']:.2f}x. The background is not merely relatively worse, it is absolutely suppressed:\n"
             f"the stream, amplified by prediction, drives the inhibitory units harder, and that inhibition reaches the rest of the axis.\n"
             f"Weights here are the learned matrix scaled by {cfg.w_scale:.1f} (learned {w_learn:.2f}, used {w_learn*cfg.w_scale:.2f}). "
             f"That scaling is imposed, not learned; the next figure shows what it costs.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")


def _caption(fig, cell, text):
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=8.7, color=MUTED,
            va="top", ha="left", linespacing=1.5)


def fig_characterisation(cfg, W_unit, w_learn, fname):
    """Where the effect comes from and what it depends on."""
    fig = plt.figure(figsize=(15.5, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.42])
    fig.suptitle("What the effect depends on", fontsize=14, fontweight="bold")

    # (a) recurrent strength, taken past the point where the loop runs away
    scales = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    snr, gs_, gn_, stab = [], [], [], []
    for sc in scales:
        r = trial(cfg, W_unit * w_learn * sc)
        snr.append(r["snr"] if r["stable"] else np.nan)
        gs_.append(r["gain_s"] if r["stable"] else np.nan)
        gn_.append(r["gain_n"] if r["stable"] else np.nan)
        stab.append(r["stable"])
    wv = np.array(scales) * w_learn
    ax = _tidy(fig.add_subplot(gs[0, 0]))
    ax.plot(wv, snr, "o-", color=C_ACC, lw=2, ms=6)
    ax.axvline(w_learn, color=C_STREAM, ls="--", lw=1.5)
    ax.text(w_learn * 1.06, np.nanmax(snr) * 0.55, "what the model\nactually learns",
            fontsize=8.4, color=C_STREAM)
    ax.axvline(w_learn * cfg.w_scale, color=INK, ls=":", lw=1.4)
    ax.text(w_learn * cfg.w_scale * 1.03, np.nanmax(snr) * 0.9, "used in\nfigure 1",
            fontsize=8.4, color=INK)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0)
    bad = [w for w, ok in zip(wv, stab) if not ok]
    if bad:
        ax.axvspan(min(bad), wv[-1] * 1.02, color=C_NOISE, alpha=0.12, lw=0)
        ax.text(min(bad) * 1.01, np.nanmax(snr) * 0.35,
                "runaway\nexcitation", fontsize=8.4, color=C_NOISE)
    ax.set_xlabel("recurrent transition weight"); ax.set_ylabel("separation gain")
    ax.set_title("a.  Effect size against recurrent strength", fontsize=10.5)

    ax = _tidy(fig.add_subplot(gs[0, 1]))
    ax.plot(wv, gs_, "o-", color=C_STREAM, lw=2, ms=6, label="stream")
    ax.plot(wv, gn_, "s-", color=C_NOISE, lw=2, ms=6, label="background")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0)
    ax.set_xlabel("recurrent transition weight"); ax.set_ylabel("gain")
    ax.legend(fontsize=8.5)
    ax.set_title("b.  The two move in opposite directions", fontsize=10.5)

    # (c) token duration, at a FIXED recurrent weight so only timing varies
    w_fix = w_learn * cfg.w_scale
    W_fix = ideal_W(cfg, w_fix)
    toks = [10, 20, 40, 80, 160, 320]
    snr_t = []
    for tm in toks:
        r = trial(cfg, W_fix, token_ms=tm)
        snr_t.append(r["snr"] if r["stable"] else np.nan)
    ax = _tidy(fig.add_subplot(gs[0, 2]))
    ax.plot(toks, snr_t, "o-", color=C_ACC, lw=2, ms=6)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0)
    ax.axvline(cfg.token_ms, color=INK, ls=":", lw=1.3)
    ax.set_xscale("log"); ax.set_xticks(toks)
    ax.set_xticklabels([str(t) for t in toks])
    ax.set_xlabel("token duration (ms)"); ax.set_ylabel("separation gain")
    ax.set_title("c.  Fast streams benefit most", fontsize=10.5)

    # (d) inhibition timescale, same fixed weight
    taus = [0.010, 0.020, 0.040, 0.080, 0.160, 0.320]
    snr_i = []
    for ti in taus:
        r = trial(cfg, W_fix, tau_I=ti)
        snr_i.append(r["snr"] if r["stable"] else np.nan)
    ax = _tidy(fig.add_subplot(gs[1, 0]))
    ax.plot(np.array(taus) * 1e3, snr_i, "o-", color=C_ACC, lw=2, ms=6)
    ax.axvline(cfg.token_ms, color=C_NOISE, ls="--", lw=1.4)
    ax.text(cfg.token_ms * 1.1, np.nanmin(snr_i), "one token", fontsize=8.4,
            color=C_NOISE)
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("inhibitory time constant (ms)"); ax.set_ylabel("separation gain")
    ax.set_title("d.  Inhibition must be slower than the stream", fontsize=10.5)

    # (e, f) the mechanism, in currents
    r = trial(cfg, W_unit * w_learn * cfg.w_scale)
    nt, chan = r["nt"], r["chan"]
    k0 = 30
    a, b = k0 * nt, (k0 + 6) * nt
    ts = (np.arange(a, b) - a) * 1e-3
    ax = _tidy(fig.add_subplot(gs[1, 1]))
    cs = chan[k0:k0 + 6]
    rec = np.array([r["on"]["rec_E"][c, a + i * nt:a + (i + 1) * nt].max()
                    for i, c in enumerate(cs)])
    tm = np.array([r["on"]["tm_in"][c, a + i * nt:a + (i + 1) * nt].max()
                   for i, c in enumerate(cs)])
    inh = np.array([r["on"]["inh_to_E"][c, a + i * nt:a + (i + 1) * nt].max()
                    for i, c in enumerate(cs)])
    x = np.arange(len(cs))
    ax.bar(x - 0.25, tm, 0.25, color="#8fa8c8", label="thalamic drive")
    ax.bar(x, rec, 0.25, color=C_STREAM, label="recurrent prediction")
    ax.bar(x + 0.25, -inh, 0.25, color=C_NOISE, label="inhibition")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"ch {c}" for c in cs], fontsize=8)
    ax.set_ylabel("current")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("e.  Currents on six consecutive stream tokens", fontsize=10.5)

    noise_ch = np.arange(cfg.n_stream, cfg.n_channels)
    ax = _tidy(fig.add_subplot(gs[1, 2]))
    ax.plot(ts, r["off"]["inh_to_E"][noise_ch, a:b].mean(0), lw=2, color=GRID,
            label="W = 0")
    ax.plot(ts, r["on"]["inh_to_E"][noise_ch, a:b].mean(0), lw=2, color=C_NOISE,
            label="learned W")
    ax.set_xlabel("time (s)"); ax.set_ylabel("inhibition onto noise channels")
    ax.legend(fontsize=8.5)
    ax.set_title("f.  Why the background falls", fontsize=10.5)

    _caption(fig, gs[2, :],
             f"a, b.  The effect grows steeply with recurrent strength, and the two curves move apart: the stream is amplified while the background is pushed down. "
             f"The dashed line marks what the model\nactually learns on this stimulus ({w_learn:.2f}), where the effect is real but modest. Reaching the large effect needs weights "
             f"several times stronger, and past about 1.0 the recurrent loop runs away,\nbecause a sweeping stream that returns to its starting channel makes the learned weights a closed cycle. "
             f"That gap between what is learned and what is needed is the honest result of this test.\n"
             f"c, d.  The hypothesis holds on timing: separation falls steadily as tokens lengthen, and improves as inhibition is made slower. "
             f"These two panels hold the recurrent weight fixed at an idealised\ntransition matrix rather than re-learning at each setting, because longer tokens learn stronger weights and that alone "
             f"would drive the answer. The absolute gains are therefore smaller than\nin panel a, which uses the full learned matrix; only the shape of these two curves is meant to be read.\n"
             f"e, f.  The mechanism, in currents: the recurrent prediction adds to the thalamic drive on stream channels, and the resulting extra "
             f"activity raises inhibition across the whole axis, which is what removes the background.")

    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  saved {Path(fname).name}")
    return dict(scales=wv, snr=snr, toks=toks, snr_t=snr_t, taus=taus, snr_i=snr_i)


# ---------------------------------------------------------------------
def main(argv=None):
    out = Path(__file__).resolve().parent
    cfg = StreamConfig()
    print("[ stream_enhancement ] layer 1 only, no second layer")
    print(f"  {cfg.n_channels} channels, stream on 0-{cfg.n_stream-1}, "
          f"noise on {cfg.n_stream}-{cfg.n_channels-1}")
    print(f"  token {cfg.token_ms:.0f} ms, tau_I {cfg.tau_I*1e3:.0f} ms, "
          f"noise amplitude {cfg.noise_amp}\n")

    W, chan = learn_weights(cfg)
    tr = [W[b, a] for a, b in zip(chan[:-1], chan[1:])]
    w_learn = float(np.mean(tr))
    W_unit = W / max(w_learn, 1e-9)          # unit strength template
    print(f"  learned transition weight: {w_learn:.3f}")
    print(f"  used in figure 1: {w_learn * cfg.w_scale:.3f} "
          f"(learned x {cfg.w_scale})")

    res = trial(cfg, W * cfg.w_scale)
    print(f"  stream {res['s0']:.2f} -> {res['s1']:.2f}   ({res['gain_s']:.2f}x)")
    print(f"  noise  {res['n0']:.2f} -> {res['n1']:.2f}   ({res['gain_n']:.2f}x)")
    print(f"  separation improves {res['snr']:.2f}x\n")

    res_asis = trial(cfg, W)
    print(f"  as learned, without scaling: separation {res_asis['snr']:.2f}x\n")

    fig_effect(cfg, res, W * cfg.w_scale, w_learn, str(out / "se_effect.png"))
    fig_characterisation(cfg, W_unit, w_learn, str(out / "se_characterisation.png"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
