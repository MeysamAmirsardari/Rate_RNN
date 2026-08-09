# Model figure — caption draft

**Claim title:** A cortical circuit carrying three memories, read out by a
multiscale stage that composes relations.

Panel **a** is a schematic; its annotated time constants are configuration
values. Panel **b** contains no drawings — every array is the object the
implementation forms, computed in `model_data.py` and traceable to
`data/model_figure_data.npz` and `data/model_provenance.json`.

---

## Caption

**Figure S1 | A cortical circuit carrying three memories, read out by a
multiscale stage that composes relations.**

**a**, Layer 1. Each tonotopic channel receives a thalamocortical input through
a depressing synapse and drives one excitatory cell; the excitatory cells are
coupled by plastic lateral connections, drawn solid where trained and dotted
where untrained. Each excitatory cell drives its own inhibitory cell, which
inhibits mostly that same channel with weak lateral spread. Inhibitory terminals
are flat, excitatory ones are arrowheads. The circuit carries **three
memories**: short-term synaptic depression, which is unlearned and specific to a
channel (τ = 100 ms, 800 ms and 5 s); the recurrent weight matrix `W`, which is
learned across channels by rate-STDP; and the inhibitory state, which is fixed
and slow (τ_I = 80 ms against τ_E = 20 ms). `M_EI` and `M_IE` are fixed
structured matrices, strong on the diagonal and weak off it; **only `W` is
plastic**. Four channels are drawn — the model's channel count is set by the
paradigm (2 for AB/BA, 5 for roving, 14 for the Saffran stream, 37 for
figure-ground).

**Interface.** Layer 2 receives the excitatory rate vector `E`, one value per
channel, and nothing else. It does not read `W`. This is a property of the
implementation, not a simplification for the figure: `Layer2MR.step` takes `E`
as its only argument from layer 1, and it is the reason composition survives the
layer-1 controls (4.00 words held in order with layer 1 plastic, 3.88 with it
frozen, 3.75 with no layer 1 at all).

**b**, Layer 2, shown as measured arrays at one instant of a real Saffran
stream — the final token of the four-token word W3 = 6 → 7 → 8 → 9, with channel
9 firing. Left to right: the rate vector `E`; the filterbank state `s`, one leaky
integrator per channel per timescale (6 rates, τ = 30–500 ms, log-spaced); and
the coincidence map `D = outer(E, s_flat)`, 14 × 84, with same-channel entries
zero and the six rate blocks separated. Because `E` is effectively one-hot for
an isochronous token stream, **only the row of the channel firing now is
non-zero** — that row is outlined. Below, that row is enlarged as context token
× filter timescale and set beside the mask of one committed unit in the same
layout. The mask is a template for the pattern: outlined cells mark the strongest
filter carrying each earlier token of the word, and they move monotonically to
slower filters with age — token 8 on the 92 ms filter, token 7 on 162 ms,
token 6 on 285 ms. That ordering is what makes the mask represent a word rather
than a set of tokens. Colour is scaled within each map. 24 units are available;
how many commit is a result, reported in Figure 5.

---

## Values quoted in the caption

| Quantity | Value | Source |
|---|---|---|
| Excitatory membrane constant | τ_E = 20 ms | `A1Config.tau_E` |
| Inhibitory constant | τ_I = 80 ms | `A1Config.tau_I` |
| Depression timescales | 100 ms, 800 ms, 5 s | `A1Config.tau_std` |
| Filterbank | 6 rates, τ = 30–500 ms log-spaced | `MRConfig.rates` |
| Coincidence map shape | 14 × 84 | `Layer2MR.coincidence` |
| Unit budget | 24 | `MRConfig.n_units` |
| Snapshot word | W3 = 6 → 7 → 8 → 9, final token channel 9 | `model_provenance.json` |
| Mask filter for token 8 | τ = 92 ms | measured from `M` |
| Mask filter for token 7 | τ = 162 ms | measured from `M` |
| Mask filter for token 6 | τ = 285 ms | measured from `M` |
| Layer-1 dependency control | 4.00 / 3.88 / 3.75 words in order | `figure_5/data/saffran_figure5_data.npz` |

---

## Statements this figure must not make

- **It must not draw layer 2 as depending on `W`.** The interface band exists to
  prevent exactly that reading. An undifferentiated arrow between the layers —
  as in the source grant figure — would imply a hierarchy the manuscript has
  tested and not found.
- **It must not present panel b as showing a trained layer 1.** The cached
  excerpt is replayed through a fresh layer 1, so `W` is near its initial value
  during the replay. Nothing shown depends on `W`, but the panel is not evidence
  about layer 1's learned state.
- **It must not claim the mask ordering is typical from this panel alone.** One
  unit, one seed, one instant. The across-seed result is Figure 5.
- **It must not imply the readout is biologically localised.** The manuscript
  does not identify layer 2 with a cortical area, and the figure names it a
  downstream readout, not a region.
- **It must not describe the model as an account of perception, streaming or
  attention.** None is modelled.
- **It must not be read as four memories.** Three memories live in layer 1; the
  readout's masks are a fourth store but belong to a different stage, and the
  figure separates them by band.

## Known limitations

- **Panel a is a schematic and carries no simulated trace.** The measured
  layer-1 dynamics, including the E/I timescale separation, are Figure 1
  panel d.
- **`y = relu(M · D)` is stated but not plotted.** One scalar per unit at one
  instant carries no visual information; unit activity over time is Figure 5
  panel b.
- **The mask shown is the driving row of a 14 × 84 mask**, reshaped without
  averaging. The unit's full mask has 13 further current-channel rows, which are
  empty for this unit; the same reshape and the same justification are used in
  Figure 5 panel c.
- **The learned-mask colour scale is per map.** `D` and `M` differ by orders of
  magnitude in absolute value — `D` is a product of rates, `M` a normalised
  template — so a shared scale would render one of them blank. The caption says
  colour is scaled within each map, and no quantitative comparison between the
  two colour scales is intended or supported.
