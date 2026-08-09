# Figure 5 — caption draft

**Claim title:** A multiscale readout composes learned transitions into
multi-event sequence representations.

Every value is traceable to `data/saffran_figure5_runs.csv` (one row per
simulated session). All uncertainties are SEM over the eight exposure order
seeds.

---

## Caption

**Figure 5 | A multiscale readout composes learned transitions into multi-event
sequence representations.**

**a**, The exposure stream. Two three-token and two four-token words, built from
fourteen channels, are concatenated in random order with no word immediately
repeating. The stream is perfectly isochronous — 50 ms tokens separated by 30 ms
both inside a word and across a boundary — so nothing in the timing or the level
marks a boundary. The only cue is statistical: within a word
P(next | current) = 1, across a boundary P(next | current) = 1/3. Colour and
label mark word identity. Dashed lines mark word onsets and are drawn for the
reader, not present in the stimulus.

**b**, Whole-word unit activity on the same clock, late in exposure. Each trace
is one unit whose mask holds a complete word, normalized to its own peak and
coloured by the word it represents; the background tint marks each word
occurrence in that word's colour. Each unit is active inside bands of its own
colour and silent elsewhere.

**c**, The templates those units learned. A complete unit mask has shape
(channel firing now, filter × context channel) = 14 × 84. Each exemplar is
driven from one current-token row; the panel reshapes that measured row, without
averaging or selecting weights, into context token × filter timescale (14 × 6).
Colour is normalized within unit. Outlines mark the strongest filter carrying
each earlier token of the corresponding word. **Neither axis is time**: filter
identity makes relative age recoverable. Earlier word tokens occupy
progressively slower filters, and the four-token words (W3, W4) recruit one more
context token than the three-token words.

**d**, Three questions of the same eight seeds, on one shared set of conditions.
*Behaviour*: word versus part-word discrimination from the summed response of
the committed population, scored within item length and averaged, since item
duration would otherwise be confounded with word status. Both readouts are far
above chance and the single-timescale readout is if anything higher
(0.915 ± 0.021 versus 0.881 ± 0.028); scrambled exposure falls to chance
(0.445 ± 0.032).
*Function*: the number of items for which some single unit is selective, the
unit chosen on half the presentations and scored on the other half. All three
conditions reach 4.00 ± 0.00 — **including scrambled exposure, which learned
nothing.** Words and part-words of equal length contain different tokens, so
token-level tuning alone passes this test; functional selectivity is therefore
not evidence of composition.
*Structure*: the number of words a unit's mask holds in order, each older token
on a strictly slower filter. 0.00 ± 0.00 for the single-timescale readout,
4.00 ± 0.00 for the multiscale readout in every seed, and 0.00 ± 0.00 after
scrambled exposure.

Only the structural measure separates the two architectures. A word of length
*n* contains *n* − 1 within-word transitions and a part-word at most *n* − 2, so
a population of transition detectors passes a familiarity test without anything
representing a word.

**e**, The pairwise structure that is learned. Left, the transition matrix the
paradigm defines. Right, layer-1 recurrent weights after exposure for one seed.
Self-connections are masked in both: they are not transitions between tokens,
are absent from the design by construction, and are strong enough in the model
to set the colour scale and bury the structure the panel is about. Within-word
transitions are learned more strongly than boundary transitions (within/boundary
weight 1.28).

**f**, What the multiscale bank adds. Left, the response of six logarithmically
spaced leaky integrators (τ = 30–500 ms) to a single token. Right, the profile
across rates at three elapsed times, each normalized to its own peak — a
monotone function of how long ago the token fired, so elapsed time is
recoverable. This is the only new ingredient; the learning rules are unchanged.

**g**, The layer-1 dependency control. Words held in order and discrimination
with layer 1 plastic (4.00 ± 0.00 words, AUC 0.881 ± 0.028), with layer 1
present but its recurrent weights frozen at zero (3.88 ± 0.12, 0.875 ± 0.022),
and with no layer 1 at all (3.75 ± 0.16, 0.910 ± 0.026). The readout barely
depends on the learned relational representation of the recurrent layer, and
discrimination is if anything highest without it. It is therefore described as a
generic multiscale downstream sequence readout, not a demonstrated hierarchy.

**h**, Familiarity-composition state space. Ordered composition is plotted
horizontally and familiarity vertically for every exposure seed. The labelled
regions distinguish a familiar-only representation, a familiar and composed
representation, and neither. The arrow is the matched structured-exposure
comparison: adding multiscale context moves the readout from familiar-only to
complete-word composition with little change in familiarity. Scrambled
exposure remains at zero composition and below chance familiarity. The peach
region is below chance discrimination — the criterion the behavioural result
establishes.
Saffran et al. report a direction of preference rather than a magnitude a model
activation can be plotted beside, so the comparison is to that criterion and no
signed behavioural axis is shared with the model.

---

## Quoted numbers

### Structured exposure, layer 1 plastic

| Readout | AUC (all) | 3-token | 4-token | Selective units | Words held in order | Max span |
|---|---|---|---|---|---|---|
| Single timescale | 0.915 ± 0.021 | 0.871 | 0.960 | 4.00 ± 0.00 | **0.00 ± 0.00** | 2 |
| Multiscale | 0.881 ± 0.028 | 0.809 | 0.953 | 4.00 ± 0.00 | **4.00 ± 0.00** | 4 |

### Controls

| Condition | Readout | AUC | Words held in order |
|---|---|---|---|
| Layer 1 plastic | Multiscale | 0.881 ± 0.028 | 4.00 ± 0.00 |
| Layer 1 frozen | Multiscale | 0.875 ± 0.022 | 3.88 ± 0.12 |
| No layer 1 | Multiscale | 0.910 ± 0.026 | 3.75 ± 0.16 |
| Scrambled exposure | Multiscale | 0.445 ± 0.032 | 0.00 ± 0.00 |
| Scrambled exposure | Single timescale | 0.460 ± 0.034 | 0.00 ± 0.00 |

Within/boundary layer-1 weight ratio: 1.28.
Selective units under scrambled exposure: 4.00 ± 0.00 for both readouts.

---

## Statements the figure must not make

1. **Do not call the pairwise map an auditory object.** The object-level claim
   begins at panel c.
2. **Do not present the behavioural pass as evidence of composition.** The
   single-timescale readout passes it and is nominally higher.
3. **Do not use functional selectivity as evidence of composition.** It is at
   ceiling after scrambled exposure, where nothing was learned.
4. **Do not overlay infant listening time on the model axis.** The behavioural
   result is a criterion, not a magnitude.
5. **Do not claim a hierarchy.** Composition survives with no first layer at
   all; the correct wording is *generic multiscale downstream sequence readout*.
6. **Do not call the seeds independent networks.** Layer 2 masks start
   identically in every seed; the seed varies the exposure order.

---

## Known limitations, stated here rather than discovered by a reviewer

1. **The replication unit is the exposure session, not the network.**
2. **Layer 1 is re-run on the isolated test stream** in the mode used for
   exposure, following the committed protocol in
   `layer2_multirate.run_saffran`, so layer 1 continues to adapt during testing
   in the plastic condition.
3. **Isolated presentation is mildly out of distribution for a context-sensitive
   readout.** A three-token word never occurs without a predecessor in the
   stream, so multiscale units tuned to that predecessor are tested slightly off
   their learned context. This is why the multiscale readout's three-token
   discrimination (0.809) sits below its four-token discrimination (0.953), and
why panel d reports item length separately rather than averaging it away.
4. **The population-sum readout is a familiarity measure, not a word readout.**
   It counts familiar transitions, and the figure uses it only because it is the
   closest analogue of the behavioural measure.
5. **The span criterion is conservative in one direction and permissive in
   another.** It requires the chain's tail to match a real word ending at the
   driving token; it reads the single strongest cell per predecessor, so
   evidence spread across several rates may be undercounted.
6. **No token is shared between words**, so this does not test composition when
   words overlap in their elements.
