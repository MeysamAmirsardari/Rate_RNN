# Figure 1 — caption draft

**Claim title:** Distinct circuit timescales support novelty, learned
prediction, selection and sequence composition.

Panels **a**, **b**, **c** and **e** are schematics and contain no data. Panel
**d** is measured `model0` output; every value quoted from it is traceable to
`data/panel_d_traces.npz` and `data/panel_d_provenance.json`. Panel **f**
states predictions, and the values quoted for the cells already settled come
from the figures that settle them (Figures 2, 4 and 5).

---

## Caption

**Figure 1 | Distinct circuit timescales support novelty, learned prediction,
selection and sequence composition.**

**a**, The problem. One acoustic scene, drawn as a tonotopy × time lattice of
tone tokens. Two families of dependency are drawn over the *same* tokens. A
vertical link joins elements that always sound together (lag 0); a directed arc
joins an element to the one that always follows it (lag > 0). Both are drawn in
the same colour because the figure's claim is that they are the same operation:
a learned dependency between elements, evaluated at zero lag or at a non-zero
lag. Grey tokens are unrelated scene elements. Grouping and prediction are
therefore not separate faculties that a circuit must implement twice.

**b**, The definition, in three steps. Elements that co-occur become one token;
tokens that recur in a fixed order become a link (faded copies mark earlier
occurrences); a link is a prediction of what comes next (dashed outline marks
the expected but not-yet-present token). An auditory object, in this paper, is a
learned predictive dependency among sound elements — nothing more is assumed.

**c**, The circuit, and the four questions it answers. A tonotopic array of
channels; each thalamocortical input passes a depressing synapse (**novelty**),
drives one excitatory cell, and the excitatory cells are coupled by **plastic
lateral E→E connections** (**dependency**). Trained connections are drawn solid
and untrained ones dotted, so plasticity is visible as a state of the circuit
rather than as a label on it. Each excitatory cell drives its own inhibitory
cell, which inhibits mostly that same channel with weak lateral spread
(**balance**); inhibitory terminals are drawn flat, excitatory ones as
arrowheads. **Only E→E connections are plastic; E→I and I→E are fixed.** A
downstream stage reads the coincidence map (post-rate × pre-trace) through a
bank of timescales and forms multi-element units (**composition**). Four
channels are drawn; the model's channel count is set by the paradigm (2 for
AB/BA, 5 for roving, 14 for Saffran, 37 for figure-ground).

**d**, How to read the model, on real output. `model0` was run on the AB/BA
paradigm — 400 sequences, 90% AB, 50 ms tones separated by a 30 ms gap — and the
panel shows the mean of the 174 AB sequences in the second half of the run, by
which point the network has learned the dependency in one direction only
(W[B←A] = 0.21, W[A←B] = 0.00). Five quantities share one clock: the input
pulses, the thalamic drive after short-term depression, the excitatory rate, the
recurrent current, and the inhibitory current. Pale traces are channel A, full
traces channel B. **No trace is smoothed, rescaled or idealised.** During tone A
the learned link drives channel B before channel B sounds. The grey band is the
30 ms gap. The two time constants are read back out of these traces as the 1/e
fall from each signal's own peak: excitation falls in 22 ms and inhibition in
86 ms, against configured values of 20 ms and 80 ms. Because inhibition is the
slower of the two, at tone-B onset the inhibitory current on channel B is still
24% of its peak (1.33 of 5.54) while the excitation that produced it has
returned to zero. That gap between the two timescales is the lever the rest of
the paper uses.

**e**, The four paradigms, in the order the Results take them, along the
progression novelty → dependency → selection → composition. Each is introduced
where the previous one provably cannot go further: depression is per-channel and
so cannot represent order, because AB and BA leave identical depression states; a
learned link states what should follow but not which of several simultaneously
available organisations is heard; and pairwise links are not multi-element
objects. Thumbnails are schematic.

**f**, The hypothesis matrix. Rows are the four paradigms, columns the four
mechanisms. **These are predictions entering the Results, not results**; the
synthesis figure revisits the same matrix with measured effect sizes, and the
argument of the paper is that the two agree. Filled = predicted required, open =
predicted not needed, hatched = not yet tested, dash = not engaged by that
paradigm. A small square marks the six cells already settled by simulation in
this manuscript, whose values are given below. Two of those settled cells run
against the intuitive prediction — inhibitory *structure* turns out to be
irrelevant to both roving and figure-ground — and the matrix shows them as
measured rather than promising an effect the Results later withdraw.

---

## Values quoted in the caption

### Panel d — measured, this figure

| Quantity | Value | Source |
|---|---|---|
| Sequences averaged | 174 late AB sequences of 400 | `panel_d_provenance.json` |
| Learned forward weight | W[B←A] = 0.2117 | `panel_d_traces.npz` |
| Learned reverse weight | W[A←B] = 0.0000 | `panel_d_traces.npz` |
| Excitation 1/e fall | 22 ms (configured `tau_E` = 20 ms) | measured from the trace |
| Inhibition 1/e fall | 86 ms (configured `tau_I` = 80 ms) | measured from the trace |
| Excitation at tone-B onset | 0.000 | measured from the trace |
| Inhibition at tone-B onset | 1.328, = 24% of its peak 5.538 | measured from the trace |

### Panel f — the six cells already settled

| Cell | Measured | Source |
|---|---|---|
| Roving × depression — **required** | suppression index +0.298 intact vs **−0.013** with depression removed | `figure_2/data/model_perturbation_summary.csv` |
| Roving × plasticity — **not needed** | +0.306 with plasticity frozen vs +0.298 intact | same |
| Roving × inhibitory structure — **not needed** | +0.298 with uniform inhibition vs +0.298 intact | same |
| SFG × plasticity — **required** | assembly drive 0.035→0.101 across figure sizes, **exactly 0 at every size when frozen** | `figure_4/data/figure_4_summary.csv` |
| SFG × inhibitory structure — **not needed** | uniform 0.0359→0.1020 vs selective 0.0355→0.1010, within ~1% | same |
| Saffran × multiscale readout — **required** | 4.00/4 words held in order with the multiscale readout, **0.00/4** with a single timescale | `figure_5/data/figure_5_summary.csv` |

Roving values are the mean suppression index over eight order seeds. SFG values
are means over eight session seeds. Saffran values are means over eight exposure
seeds.

### The seventh settled cell, and why it reads "not needed"

Saffran × recurrent plasticity is drawn **open (predicted not needed)** and
marked as measured. Composition survives the layer-1 controls: words held in
order are 4.00 with layer 1 plastic, 3.88 with it frozen and 3.75 with no layer 1
at all (`figure_5/data/saffran_figure5_data.npz`). Layer 2 does learn — its own
masks are plastic — but the *column* in this matrix is layer-1 recurrent
plasticity, and the composition result does not depend on it.

---

## Statements this figure must not make

- **It must not claim the downstream readout is a hierarchy built on layer 1.**
  Panel c draws it as a stage reading the coincidence map, because that is what
  the measurement supports: 3.75 of 4 words are held in order with no layer 1 at
  all, against 4.00 with it. Any wording that makes composition *depend* on the
  learned recurrent weights is contradicted by Figure 5, panel g.
- **It must not present the matrix as results.** The panel title, the subtitle
  and the key all say prediction. Only the marked cells are settled, and they are
  settled elsewhere in the paper, not here.
- **It must not assert that inhibitory structure matters everywhere.** It is
  measured as irrelevant in both roving and figure-ground. Panel f says so.
- **It must not claim that panel d generalises across paradigms.** It is one
  sequence type, in one paradigm, at one parameter setting — the AB/BA
  configuration, which raises `w_IE_self` to 3.0, `w_EI_self` to 0.40 and lowers
  `W_norm` to 4.0 relative to the shared defaults. It is a reading key for the
  traces, not evidence for a result.
- **It must not describe the model as an account of perception, streaming or
  attention.** None of the three is modelled.
- **It must not be read as asserting five distinct memories.** There are three
  memories in layer 1 (depression, recurrent weights, inhibitory state) plus a
  downstream readout. "Balance" names the interaction of two of them, not a
  fourth store.

## Known limitations

- **Panels a, b, c and e are drawings.** They carry no measurement and should
  never be cited as evidence. Panel e's stimulus thumbnails are schematic and do
  not reproduce the exact channel counts, durations or densities of the
  paradigms; those are specified in Figures 2, 4 and 5.
- **The AB/BA row of the matrix is entirely prediction.** No ablation of the
  AB/BA paradigm is reported in this manuscript. The row is drawn from the
  argument in the text, not from simulation, and carries no measured markers.
- **The depression column for SFG and Saffran is untested**, and drawn hatched.
  Depression is active in those simulations; what is untested is whether the
  reported effects survive its removal.
- **Panel d's time constants are single-exponential 1/e readouts.** The traces
  are not pure exponentials — the inhibitory current is driven twice within the
  sequence — so the inhibitory fall is measured after tone-B offset, where the
  decay is free. The measured values (22 ms, 86 ms) are close to but not
  identical with the configured membrane constants (20 ms, 80 ms), and the
  caption gives both rather than either alone.
- **The coincidence map drawn in panel c is illustrative.** The cell values are
  drawn from a fixed random seed for texture; they are not a measured map.
