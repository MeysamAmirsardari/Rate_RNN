# Model figure — the two layers, and the interface between them

## Scientific role

This is the model-definition figure. `RESULTS_FIGURE_STRATEGY.md` assigns it to
Supplementary Figure S1 ("equations and full architecture"); it is drawn at main
figure quality so it can be promoted if the model needs to be defined in the
main text.

It exists because the compact architecture panel of Figure 1 cannot carry the
detail, and because one structural fact needs to be stated where a reader can
check it:

> **Layer 2 reads layer 1's rate vector `E`, and never its weight matrix `W`.**

`Layer2MR.step(E, dt)` takes `E` as its only argument from layer 1. That is the
mechanical reason composition survives the layer-1 controls — 4.00 words held in
order with layer 1 plastic, 3.88 frozen, 3.75 with no layer 1 at all — and it is
why the readout must be drawn as a stage reading the coincidence map, not as a
hierarchy resting on learned recurrent weights.

The source material was two grant figures: a layer-1 circuit diagram and a
DREX-style pipeline, both labelled with Aim numbers. Both are redrawn here in
the manuscript's own visual system, and the Aim vocabulary is gone.

## Build

From the repository root:

```bash
python -m final_figures.figure_model.make_model_figure
```

`--force-data` recomputes the layer-1 and layer-2 arrays. **Figure 5 must be
built first**, because this figure reuses its cached exemplar session
(`final_figures/figure_5/data/saffran_exemplar.npz`) rather than re-running an
exposure; the data module raises a clear error with the command to run if that
file is missing.

## Panel b is measured, not drawn

An architecture figure could defensibly illustrate every stage. This one does
not. Each object in the layer-2 pipeline is the actual array the implementation
forms, computed in `model_data.py`:

| Object | How it is obtained |
|---|---|
| `E` | `model0.simulate` driven by the genuine cached Saffran stream excerpt |
| `s` (14 × 6) | `Layer2MR`'s own filterbank update, stepped forward over that rate history |
| `D` (14 × 84) | `Layer2MR.coincidence(E)` — the class's own outer product |
| `M` (14 × 84) | a learned mask, taken **unmodified** from the Figure 5 exemplar session |

The snapshot instant is deliberately chosen: the final token of a four-token
word, so the coincidence map on display actually holds a whole word of context.
It lands on word **W3 = 6 → 7 → 8 → 9**, with channel 9 firing.

### What the real mask shows

The learned mask for unit 13 places each earlier token of the word on a strictly
slower filter:

| Token | Filter τ carrying it |
|---|---|
| 8 (most recent) | 92 ms |
| 7 | 162 ms |
| 6 (oldest) | 285 ms |
| 9 | firing now |

That monotone ordering is the composition claim, in the measured weights. A mask
that held the same three tokens without the ordering would represent a *set*;
the ordering is what makes it a word. This is the same quantity Figure 5 scores
as "words held in order", shown here as the raw object rather than as a count.

### D is mostly empty, and that is the point

At any instant only the channel firing now has a non-zero row in `D`, because
`D = outer(E, s_flat)` and `E` is effectively one-hot for an isochronous token
stream. The full 14 × 84 map is drawn at size, with that one live row outlined,
so the reader sees both the true shape of the representation and how sparse it
is. The row is then enlarged as (context token × filter timescale) beside the
learned mask in the same layout — the figure-5 reshape, applied without
averaging or weight selection.

## Non-negotiable safeguards

- **The interface is stated, not implied.** The band between the panels carries
  the claim that layer 2 reads `E` alone. Drawing an undifferentiated arrow from
  layer 1 to layer 2 — as the source grant figure does — would suggest a
  dependency the manuscript has measured and not found.
- **Panel a is labelled a schematic; panel b is labelled measured.** Panel b's
  subtitle says "measured arrays, not drawings" in the figure itself.
- **No grant vocabulary.** No Aim labels anywhere.
- **The colour key is Figure 1's, unchanged**: depression `terracotta`,
  plastic recurrent excitation `model`, inhibition `teal`, readout `decoder`.
  Excitatory and inhibitory cells are additionally distinguished by fill and by
  terminal shape (arrowhead versus flat bar).
- **Nothing below 5 pt**; the smallest type is 5.2 pt. Mathtext is avoided
  entirely — it renders subscripts at 0.7 of the base size and substitutes
  DejaVu for italic variables. For the same reason the outer product is written
  `outer(E, s_flat)` rather than with a CIRCLED TIMES sign, **which Arial does
  not contain**; using it silently pulled DejaVu into the PDF. The exported PDF
  embeds Arial Regular and Bold and nothing else.
- **Four channels are drawn in panel a**, and the caption says the channel count
  is set by the paradigm (2 for AB/BA, 5 for roving, 14 for Saffran, 37 for
  figure-ground). The drawing is generic over N.

## Known limitations

- **Panel a is a schematic.** Its annotated time constants are real
  configuration values, but no trace in panel a is simulated output. Figure 1
  panel d is where the layer-1 dynamics are shown as measured traces.
- **One instant, one seed, one word.** Panel b is a single snapshot of a single
  exemplar session. It shows what the representation *is*, not how reliably it
  forms; Figure 5 carries the across-seed statistics.
- **The layer-1 run behind panel b starts from naive weights.** The excerpt is a
  late-exposure stream, but it is replayed through a fresh layer 1, so `W` is
  near its initial value during the replay. This does not affect what is shown:
  `E` is dominated by the thalamic drive, and layer 2 does not read `W` at all.
  It does mean panel b must not be read as showing a *trained* layer 1.
- **`y` is not drawn.** The output `y = relu(M · D)` is stated as an equation
  rather than plotted, because a single scalar per unit at one instant carries
  no visual information. Unit activity over time is Figure 5 panel b.

## Outputs

- `outputs/figure_model_architecture.{pdf,svg,png}` — 183 × 138 mm, fixed
  export bounds, 600 dpi review image.
- `data/model_figure_data.npz` — `E`, `s`, `D`, `M`, the rate window, and the
  snapshot descriptors.
- `data/model_provenance.json` — the full layer-1 and layer-2 configurations,
  the settings hash, the stimulus source, the snapshot identity and software
  versions.
- `data/provenance.json` — panel sources, colour key and output hashes.
