# Figure 5 — Statistical learning and multi-event composition

## Scientific role

Figure 5 is the last step of the manuscript's spine, `novelty → dependency →
selection → composition`. It asks whether pairwise transition knowledge is
already a representation of a multi-event object, and answers no.

The figure is organized so the reader meets the stream, then the units, then
the evidence:

1. **A** the continuous isochronous stream, with no acoustic boundary cue;
2. **B** whole-word unit activity on the same clock, directly beneath;
3. **C** the kernels those units learned — masks over the coincidence map;
4. **D** behaviour, function and structure on one shared set of conditions;
5. **E** the pairwise structure that is in fact learned;
6. **F** what the multiscale bank adds — an elapsed-time code;
7. **G** the layer-1 dependency control;
8. **H** discrimination against composition.

## Vocabulary

**Two three-token and two four-token words** over fourteen channels. Mixed
lengths are load-bearing: with a single word length a readout can succeed with a
fixed-width template, whereas a mixed vocabulary forces the units to discover
where each word starts and stops. It also forces the test to be scored **within
length**, since otherwise item duration is confounded with word status.

## Build

```bash
python -m final_figures.figure_5.make_figure_5
```

`--force-data` repeats every simulated exposure session. The cache key is a hash
of the complete protocol, so changing any parameter forces a rerun rather than
silently reusing stale sessions.

## Three measures, and why all three are shown

Panel D reports the same eight seeds under three questions. They do not agree,
and the disagreement is the result.

| Measure | Question | Dissociates the readouts? |
|---|---|---|
| **Behaviour** — population sum over committed units, word vs part-word AUC, scored within item length | Does the network treat this item as familiar? | No, and it *favours* the single-timescale readout |
| **Function** — best unit chosen on half the presentations, scored on the other half | Is any single unit selective for the whole item? | No — both reach 4/4, **and so does scrambled exposure** |
| **Structure** — ordered span: the mask holds the complete word in order, each older token on a strictly slower filter | Is the word represented *as a word*? | Yes, completely |

The behavioural measure is a familiarity readout and is dominated by how many
familiar transitions an item contains — it *is* transition counting. A word of
length *n* contains *n−1* within-word transitions and a part-word at most *n−2*,
so a population of transition detectors passes the infant test without anything
representing a word. That is the strategy document's own point, made
quantitative rather than asserted.

The functional measure is included because it is the obvious objection to the
structural one, and it does not survive. It reaches 4/4 even for a network given
**scrambled** exposure, which learned nothing: words and part-words of equal
length contain different tokens, so token-level tuning alone passes it. Panel D
draws that scrambled reference on the same axes, because a metric that cannot
tell learning from no learning cannot be evidence of composition. The claim
therefore rests on the structural measure alone, and the figure says so.

## Unit budget

24 units are available to both readouts. How many commit is a result, and the
budget is not what produces the effect:

| Units available | Committed | Words held in order (4 seeds) |
|---|---|---|
| 14 | 14.0 (saturated) | 3.75 |
| 16 | 14.5 | 3.50 |
| 20 | 16.0 | **4.00** |
| 24 | 17.0 | **4.00** |
| 32 | 17.2 | **4.00** |

Below 20 the population saturates and composition is capacity-limited. At and
above 20 the committed count plateaus near 17 whatever the budget, and all four
words are held in every seed. 24 is therefore a modest margin over the minimum
sufficient budget, not a generous allocation, and the network leaves ~7 units
uncommitted. Discrimination keeps creeping up with more units (0.887 at 24,
0.946 at 32) while composition does not move at all — one more sign that the
population-sum measure indexes transition count rather than composition.

## Kernels are masks on the coincidence map

A unit's complete kernel has shape `(channel firing now, filter x context
channel)` — `14 x 84` here. **Neither axis is time.** Each exemplar is driven
from one current-token row. Panel C reshapes that measured row without
averaging into `context channel x filter` (`14 x 6`), which preserves every
weight used by the unit while removing thirteen visually empty current-channel
rows. The current token is stated above each map and the strongest cells
carrying the preceding word tokens are outlined.

## Non-negotiable analysis safeguards

- **One protocol for both readouts.** `L2Config` and `MRConfig` already share
  `eta`, `lam`, `gate_frac`, `w_init` and `commit_frac`; this module forces the
  same `n_units` on both. They differ **only** in the number of timescales in
  the trace bank (1 versus 6). Word set, exposure length, test items, test
  procedure and seed structure are identical.
- **Words and part-words are compared only at equal length**, then averaged.
  Part-words are built by splicing the tail of one word onto the head of
  another, and any splice that coincides with a real word is discarded.
- **`span_chain` is one function applied to both readouts.** Predecessors are
  taken greedily, each on a different channel and a strictly slower filter. With
  one timescale no slower filter exists, so the chain cannot exceed length two.
- **A unit spans a word** when the token driving it is that word's final token
  and the tail of its chain is the complete word. A chain may carry an extra
  older token beyond the word — the boundary context that habitually preceded
  it — and that still counts, because the whole word is present in order.
- **Predecessor weights are floored at 2% of a unit's mask mass**, so numerical
  dust cannot complete a chain by accident.
- **The detector metric is cross-validated.** The unit is selected on half the
  presentations and scored on the other half, so selection cannot inflate it.
- **The scrambled-exposure floor uses identical timing.** Same fourteen tokens,
  same tone and gap, same isochrony, same test items — only the word structure
  is removed.
- **The layer-1 dependency control is reported whatever it shows.**
- **Model activation and infant listening behaviour are never placed on a shared
  signed axis.** The behavioural result establishes a criterion — discrimination
  above chance — not a magnitude the model can be plotted beside.
- **The replication unit is the exposure order seed** (n = 8). Layer 2 masks
  start from the same small random values in every seed, so these are
  independent exposure sessions, not independent networks.

## Moved to the supplement

The per-seed reproducibility panel (whole-word units and span depth for every
seed) is no longer in the main figure. Its data are in
`data/saffran_figure5_runs.csv`, one row per simulated session.

## Outputs

- `outputs/figure_5_composition.{pdf,svg,png}` — vector submission, editable
  vector, and 600 dpi review image.
- `data/saffran_figure5_runs.csv` — every simulated session.
- `data/figure_5_summary.csv` — the panel-H condition summary.
- `data/*provenance.json` — protocol, settings hash, software versions, output
  hashes.
