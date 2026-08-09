# Prompt — build Figure 1

Paste everything below the line into a fresh session, working in
`/Users/eminent/Projects/Rate_RNN_py`.

---

You are building **Figure 1** of a Nature-family manuscript on predictive
auditory scene analysis. Figures 2, 4 and 5 are already built and submitted-ready
in `final_figures/`. Figure 1 is the conceptual opener: it states the claim,
defines the terms, draws the circuit, and sets up a matrix the rest of the paper
fills in. Read `final_figures/figure_2/`, `figure_4/` and `figure_5/` before
writing anything — you are joining an established visual system, not inventing
one.

## 1. What the paper argues

An auditory object is a **learned predictive dependency among sound elements**.
Grouping, prediction and deviance detection are not separate faculties; they are
manifestations of one circuit. The spine of the Results is

> novelty → dependency → selection → composition

and it is carried by three memories in a rate model of A1 (`model0/`), plus a
downstream readout:

1. **Short-term synaptic depression** — channel-specific, unlearned memory of
   recent drive. Answers *is this new?* Redundancy reduction before any scene
   model exists.
2. **Rate-STDP on lateral E→E connections** — cross-channel, learnable memory of
   coincidence and order. Answers *what goes with what, what comes next?* This is
   the scene model.
3. **Excitatory–inhibitory balance** — the E/I timescale difference decides
   whether a prediction meeting its target appears as *suppression* of the
   expected element or *enhancement* of a coherent group against competitors.
4. **A multiscale downstream readout** — a bank of timescales applied to
   *relations* rather than magnitudes, which composes pairwise links into
   multi-element objects.

Each is introduced where the previous one provably cannot go further. Depression
is per-channel, so it cannot represent order — AB and BA leave identical
depression states. A learned link states what should follow but not which of
several simultaneously predictable organisations is heard. Pairwise links are not
multi-element objects.

## 2. Repository orientation

| Path | What it is |
|---|---|
| `final_figures/style.py` | The binding visual grammar. Import from it; hard-code nothing. |
| `final_figures/figure_2/` | Roving. Read `make_figure_2.py` for the layout idiom. |
| `final_figures/figure_4/` | SFG. Read it for the reserved-panel idiom. |
| `final_figures/figure_5/` | Saffran. Read it for the stacked A/B idiom and the three-measure panel. |
| `model0/` | The rate model: `A1Config`, `simulate`. |
| `tasks/` | Paradigm generators (`roving/`, `ab_ba_model0/`, `sfg2/`). |
| `layer2_multirate/` | The multiscale downstream readout. |
| `docs_rate_rnn/RESULTS_FIGURE_STRATEGY.md` | The manuscript-level figure plan. |
| `docs_rate_rnn/FIGURE_PLAN.md` | The earlier, more detailed panel plan. |

Create `final_figures/figure_1/` with `make_figure_1.py`, `README.md`,
`FIGURE_1_CAPTION_DRAFT.md`, `data/`, `outputs/`.

## 3. The style contract — non-negotiable

```python
from final_figures.style import (
    COLORS, clean_axis, export_figure, manuscript_style, mm, panel_label)

with manuscript_style():
    fig = plt.figure(figsize=(mm(183), mm(210)))   # height is yours to choose
    ...
    export_figure(fig, OUTPUT_STEM, fixed_bounds=True)
```

- **183 mm wide.** `fixed_bounds=True` — never `bbox_inches="tight"`, it silently
  rescales a hand-tuned layout.
- **Arial/Helvetica.** Base 7 pt, panel titles 8.2 pt semibold, axis labels
  7.2 pt, ticks 6.7 pt, panel letters 10.5 pt bold, in-panel annotation
  5.2–6.0 pt. Nothing below 5 pt.
- **Every panel gets a group heading**: an 8.2 pt semibold claim line and a
  5.9 pt `COLORS["ash"]` subtitle beneath it. Copy `_group_heading` from any
  built figure.
- **Titles state claims, not methods.** "Depression asks whether the model
  failed", not "Circuit schematic".
- Unboxed axes (`clean_axis`), frameless legends, no gridlines.
- Exports PDF + SVG + PNG at 600 dpi with editable type (`pdf.fonttype = 42`).

### Colour is semantic and already fixed

| Meaning | Key | Hex |
|---|---|---|
| novel · deviant · repetition 1 · surprise | `rep1` | `#7C102A` |
| adapted · standard · repetition 15 · predicted | `rep15` | `#2166AC` |
| decoder / posterior / downstream readout | `decoder` | `#685994` |
| intact model · figure · bound set | `model` | `#2C6E5A` |
| deviant tone window | `peach` | `#FDDBC7` |
| shared tone window | `sage` | `#CCEBC5` |
| text and axes | `charcoal` | `#2D3748` |
| secondary structure, chance, null | `ash` | `#999999` |
| lesioned / control condition | — | `#7E8792` |
| alternative structure control | `terracotta` | `#BD6B6B` |

Figure 1 **establishes** this key. Whatever colour a mechanism gets here, it
keeps for the rest of the paper. Suggested mechanism assignment, which you may
refine but must then apply consistently: depression = `terracotta`, plastic
recurrent excitation = `model`, inhibition = `teal`, downstream readout =
`decoder`.

## 4. Panels

Six panels. A–C and E are drawn schematics; **D must be real simulator output**;
F is a matrix.

### A — The problem
One acoustic scene as a **tonotopy × time lattice** of tone tokens. Draw two
link families over the *same* tokens: **vertical** links (what goes with what,
now — concurrent coincidence) and **horizontal** links (what does this continue —
sequential dependency). Same elements, two organisations, one operation. This
panel must make a reader who knows nothing about the model understand why
grouping and prediction cannot be separate faculties.

### B — The definition
A three-step cartoon: **coincidence → token**, **repetition → token-to-token
link**, **link → prediction**. This is the paper's definition of an auditory
object, made visual. Keep it to three beats and no more.

### C — The circuit
Redraw the architecture. The user's own draft (attached to this task, and the
better of the two source images) is the right starting point and has the right
anatomy:

- a **tonotopic axis** down the left,
- an **SD gate** on each thalamocortical input (draw as a depressing synapse),
- one **excitatory cell per channel**,
- **plastic lateral E→E** connections, drawn as *trained* (solid) and *untrained*
  (dotted) so plasticity is visible as a state, not just a label,
- one **inhibitory cell per channel**, with **fixed** E→I and I→E connections,
- a **downstream multiscale readout** stage above or right of layer 1, reading
  the *coincidence map* (post-rate × pre-trace) through a bank of timescales.

**Strip the "AIM I / II / III" labels.** They are grant vocabulary and read as
unfinished in a paper. Relabel the three regions **novelty**, **dependency**,
**balance**, and add the fourth stage, **composition**. Note in the panel that
only excitatory connections are plastic; inhibitory connections are fixed.

### D — The reading key *(real data, not a cartoon)*
Run `model0` on one short tone sequence and plot, on a single shared time axis:

1. the input pulse train,
2. the thalamic drive with short-term depression (`out["tm_in"]`),
3. the excitatory rate (`out["E"]`),
4. the recurrent predictive current (`out["rec_E"]`),
5. the inhibitory current (`out["inh_to_E"]`).

The point of the panel is the **E/I timescale separation** (`tau_E = 20 ms`,
`tau_I = 80 ms`) that is the paper's control lever: inhibition outlives the
excitation that drove it, so a prediction can still be suppressing its target
when the target arrives. Make that visible — annotate the two time constants and
shade the interval where inhibition persists past excitation.

Use a sequence in which a learned link exists, so the recurrent current actually
pre-activates the next channel. `tasks/ab_ba_model0/ab_ba.py` builds a suitable
stimulus; `model0.simulate` returns every trace named above.

### E — The four paradigms
Roving → AB/BA → SFG → Saffran, arranged along the computational progression
`novelty → dependency → selection → composition`, each with a two-line
thumbnail of its stimulus and the question it answers. This is the reader's map
of the Results.

### F — The hypothesis matrix
Rows = the four paradigms. Columns = depression, recurrent plasticity,
inhibitory structure, multiscale readout. **These are predictions, and the panel
must say so** — the same matrix is revisited in the synthesis figure with
*observed* effect sizes, and the paper's argument is that the two match.

Render predictions as an explicit visual state (filled = predicted required,
open = predicted not needed, hatched/grey = untested), not as ticks and crosses.

**Facts you must respect.** Some cells are already measured, and Figure 1 must
not promise something the paper later contradicts without flagging it:

| Paradigm | Depression | Plasticity | Inhibitory structure | Readout |
|---|---|---|---|---|
| Roving | **required** (measured: −0.018 without it, +0.273 intact) | **not needed** (measured: +0.284 frozen) | **not needed** (measured: uniform +0.274) | — |
| AB/BA | predicted not needed | predicted required | predicted required | — |
| SFG | untested | **required** (measured: assembly drive 0 when frozen) | **not needed** (measured: uniform ≡ selective within 1%) | — |
| Saffran | untested | required for layer 2 | untested | **required** (measured: 4/4 words multiscale, 0/4 single-timescale) |

Two of those measured cells contradict the intuitive prediction — inhibitory
structure is irrelevant in both roving and SFG. That is a genuine result and the
matrix should let it show rather than quietly asserting inhibition matters
everywhere.

## 5. Hard rules

1. **No grant vocabulary.** No "Aim", no "Specific Aim", no proposal framing.
2. **Do not overstate the architecture.** The downstream readout does *not*
   currently depend on layer 1's learned weights (measured: 3.75 words covered
   with no layer 1 at all versus 4.00 with it). Draw it as a stage that reads the
   coincidence map, not as a hierarchy the paper has demonstrated.
3. **Panel D is data.** If any trace is idealised, say so in the caption. Do not
   hand-draw something that the simulator can produce.
4. **Every claim in a panel title must be one this paper actually supports.**
   Check against `figure_2/README.md`, `figure_4/README.md`,
   `figure_5/README.md` — each lists what its figure may and may not claim.
5. **The matrix is predictions, labelled as predictions.**
6. Write `README.md` and `FIGURE_1_CAPTION_DRAFT.md` alongside, in the style of
   the existing three: caption, quoted numbers, "statements the figure must not
   make", and "known limitations".

## 6. Deliverables

- `final_figures/figure_1/make_figure_1.py`, importing only from
  `final_figures.style`.
- `outputs/figure_1_architecture.{pdf,svg,png}`.
- `data/` with the panel-D simulation cached and a `provenance.json` recording
  the configuration, a settings hash and output hashes.
- `README.md` and `FIGURE_1_CAPTION_DRAFT.md`.
- Rebuildable in one command: `python -m final_figures.figure_1.make_figure_1`.

## 7. Acceptance checklist

- [ ] 183 mm wide, fixed export bounds, nothing under 5 pt, no clipped text.
- [ ] Every panel has a claim-style heading plus grey subtitle.
- [ ] Panel letters bold lowercase-of-record (A–F), none clipped off-page.
- [ ] Panel D is real `model0` output and the E/I timescale separation is
      visible and annotated.
- [ ] The circuit panel carries no Aim labels and marks plastic versus fixed
      connections distinctly.
- [ ] The matrix is labelled as predictions and its untested cells are visibly
      untested.
- [ ] Colour assignments match the table in §3 and will hold for Figures 2–6.
- [ ] Render the PDF back to an image and inspect it for overlap, crowding and
      legibility before declaring it done.

Work iteratively: render, look at the PNG, fix, re-render. The existing figures
took four to six rounds each; expect the same.
