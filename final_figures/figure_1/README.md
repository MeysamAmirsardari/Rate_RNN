# Figure 1 — The claim, the circuit, and the matrix

## Scientific role

Figure 1 is the conceptual opener. It states the paper's claim, defines its
terms, draws the circuit, teaches the reader to read the model's traces, maps
the Results, and sets out the matrix the rest of the paper fills in.

**Claim:** an auditory object is a learned predictive dependency among sound
elements. Grouping, prediction and deviance detection are not separate
faculties; they are manifestations of one circuit.

The figure is organised as:

1. **A** the problem — one scene, two organisations, one operation;
2. **B** the definition — coincidence → token, repetition → link, link →
   prediction;
3. **C** the circuit — novelty, dependency, balance, composition;
4. **D** the reading key — **real `model0` output**, not a cartoon;
5. **E** the four paradigms along `novelty → dependency → selection →
   composition`;
6. **F** the prediction matrix, revisited with measured effect sizes in the
   synthesis figure.

## Build

From the repository root:

```bash
python -m final_figures.figure_1.make_figure_1
```

`--force-data` repeats the panel-D simulation. The cache key is a hash of the
complete protocol including the full `A1Config`, so changing any parameter
forces a rerun rather than silently reusing a stale run. The simulation takes a
few seconds; everything else is drawn.

The standalone Layer-2 introduction panel is built separately:

```bash
python -m final_figures.figure_1.make_layer2_panel
```

It is a vector-native schematic of the implemented `Layer2MR` computation,
with the forward pass visually separated from the competitive local-learning
loop. This prevents the learning winner from being misread as a winner-take-all
output: every unit response remains visible.

## Figure 1 establishes the colour key

Whatever colour a mechanism gets here it keeps for Figures 2–6.

| Meaning | `style.py` key | Hex |
|---|---|---|
| Short-term synaptic depression | `terracotta` | `#BD6B6B` |
| Plastic recurrent excitation | `model` | `#2C6E5A` |
| Inhibition | `teal` | `#457B7A` |
| Downstream multiscale readout | `decoder` | `#685994` |

Colour is never the only channel. Excitatory and inhibitory cells also differ in
fill (solid versus open) and their synapses in terminal shape (arrowhead versus
flat bar), so panel C survives greyscale reproduction and colour-blind readers.
Trained and untrained plastic connections differ in stroke (solid versus
dotted), not in colour.

## Panel D is measured, and says so

Panel D is the only panel with data in it, and the requirement was that it be
real simulator output rather than a hand-drawn illustration.

`panel_d_data.py` runs the AB/BA paradigm from `tasks/ab_ba_model0/ab_ba.py` —
400 sequences, 90% AB, 50 ms tones, 30 ms intra-sequence gap, 500 ms
inter-sequence gap, two delta-tuned channels — and reduces it to the five traces
the panel plots. The task's own configuration overrides are reused verbatim
(`w_IE_self=3.0`, `w_EI_self=0.40`, `W_norm=4.0`), because the panel must
portray the network the AB/BA results are computed in rather than a differently
tuned one.

**The only reduction is averaging over sequences of one type.** No trace is
smoothed, rescaled, detrended or idealised. What is plotted is the mean of the
174 AB sequences in the second half of the run, by which point `W[B←A]` has
converged.

### The time constants are measured, not restated

The panel's point is the E/I timescale separation, and the figure does not take
that on trust from the configuration. `_one_over_e_fall_ms` reads each time
constant back out of the plotted trace as the 1/e fall from its own peak:

| Signal | Measured 1/e fall | Configured |
|---|---|---|
| Excitatory rate, channel A | **22 ms** | `tau_E` = 20 ms |
| Inhibitory current, channel B | **86 ms** | `tau_I` = 80 ms |

The inhibitory current is driven twice within a sequence — once by the
predictive pre-activation during tone A and again by tone B itself — so its free
decay is only interpretable after tone-B offset, and the search window starts
there. Both the measured and the configured value are printed, so neither can be
mistaken for the other.

The consequence, which is the whole reason the panel exists: at tone-B onset the
inhibitory current on channel B is still **1.328, or 24% of its peak of 5.538**,
while the excitation that produced it has returned to **0.000**. A prediction is
still suppressing its target at the moment the target arrives.

The learned link is real and directional: **W[B←A] = 0.2117, W[A←B] = 0.0000**.
Depression could not produce that asymmetry, which is the argument for
introducing the second memory.

## The matrix states predictions, and marks what is settled

Panel F renders each prediction as a visual state — filled (required), open (not
needed), hatched (not yet tested), dash (not engaged by that paradigm) — rather
than as ticks and crosses. A small square marks the cells already settled by
simulation elsewhere in this manuscript.

| Paradigm | Depression | Recurrent plasticity | Inhibitory structure | Multiscale readout |
|---|---|---|---|---|
| Roving | **required** ▪ | not needed ▪ | not needed ▪ | not engaged |
| AB/BA | not needed | **required** | **required** | not engaged |
| SFG | not tested | **required** ▪ | not needed ▪ | not engaged |
| Saffran | not tested | not needed ▪ | not tested | **required** ▪ |

▪ = already measured. The numbers are quoted in `FIGURE_1_CAPTION_DRAFT.md` and
come from the committed data of Figures 2, 4 and 5.

### Two cells contradict the intuitive prediction, and the matrix lets them

Inhibitory *structure* is irrelevant in both roving (uniform inhibition +0.298
against intact +0.298) and figure-ground (uniform assembly drive within about 1%
of selective at every figure size). Drawing those cells filled because
inhibition "ought" to matter would have set up a promise the Results withdraw.
They are drawn open and marked as measured.

### One deliberate departure from the commissioning brief

The brief's draft matrix listed Saffran × recurrent plasticity as *"required for
layer 2"*. **This figure draws it open — predicted not needed — and marks it as
measured.** The committed Figure 5 data show composition surviving the layer-1
controls:

| Layer 1 | Words held in order (n = 8 seeds) |
|---|---|
| Plastic | 4.00 |
| Frozen | 3.88 |
| Absent entirely | 3.75 |

Layer 2 does learn; its own masks are plastic. But the column in this matrix is
**layer-1** recurrent plasticity, and the composition result does not depend on
it. Marking the cell "required" would have contradicted Figure 5 in the same
manuscript. The same measurement is why panel C draws the readout as a stage
that reads the coincidence map rather than as a hierarchy resting on layer 1's
learned weights.

## Non-negotiable safeguards

- **Panel D is simulator output and is labelled as such** in its own subtitle,
  in the caption and in `provenance.json`. The four schematic panels are labelled
  as schematics in the caption and carry no numbers.
- **No grant vocabulary.** The three "AIM" regions of the source architecture
  draft are relabelled *novelty*, *dependency* and *balance*, and the fourth
  stage, *composition*, is added.
- **Plastic and fixed connections are distinguished structurally**, and the
  statement "Only E→E connections are plastic. E→I and I→E are fixed." is
  printed inside panel C rather than left to the caption.
- **The matrix is labelled as predictions** in its title, its subtitle and its
  key, and untested cells are visibly hatched.
- **Every claim-style panel title is checked against the figure that owns it.**
  Panel F's settled cells are cross-checked against
  `figure_2/data/model_perturbation_summary.csv`,
  `figure_4/data/figure_4_summary.csv` and
  `figure_5/data/figure_5_summary.csv`, not against the prose of the brief —
  which quoted an earlier roving run (−0.018 / +0.273 / +0.284 / +0.274) that no
  longer matches the committed Figure 2 data (−0.013 / +0.298 / +0.306 /
  +0.298).
- **Nothing is set below 5 pt.** The smallest type in the figure is 5.2 pt.
  Mathtext is avoided entirely: a mathtext subscript renders at 0.7 of the base
  size, which would have put 3.6 pt type on the page, and it substitutes DejaVu
  for the italic variable, breaking the Arial-only embedding. The exported PDF
  embeds Arial Regular, Bold and Italic and nothing else.

## Layout notes

The schematic panels draw in **true millimetres with a locked aspect ratio**
(`_mm_axes`), so a circle is a circle, a stated gap is that distance on the
printed page, and geometry can be reasoned about at final size. Headings and
panel letters are also offset in millimetres (`_heading`), which is what keeps
their spacing visually identical across a 45 mm panel and a 7 mm sub-panel.

Panel bands are positioned in millimetres from the page top, and each gap
between bands holds exactly one heading block.

## Outputs

- `outputs/figure_1_architecture.{pdf,svg,png}` — vector submission, editable
  vector, and 600 dpi review image. 183 × 247 mm, fixed export bounds.
- `outputs/figure_1_layer2_panel.{pdf,svg,png}` — standalone Layer-2
  introduction panel. 183 × 76 mm, fixed export bounds.
- `data/figure_1_layer2_panel_provenance.json` — algorithm summary, source
  hashes and output hashes for the standalone Layer-2 panel.
- `data/panel_d_traces.npz` — the panel-D traces and measured scalars.
- `data/panel_d_provenance.json` — protocol, full `A1Config`, settings hash,
  software versions, measured values.
- `data/provenance.json` — panel sources, the colour key, the matrix as drawn,
  and output hashes.
