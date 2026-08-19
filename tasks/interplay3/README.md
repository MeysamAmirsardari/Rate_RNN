# interplay3 — three four-tone words in a fifty-channel cloud

The multi-rate version of [interplay2](../interplay2/README.md), which stays as
it is. There a token was two tones and one slow trace sufficed. A four-tone
word has three transitions, and a single-rate layer fragments across them, so
this task uses `layer2_multirate`: a bank of filters, because the ratio of a
channel's trace across rates encodes how long ago it fired.

**Does that still work when the word is buried in a cloud that never falls
silent?** `layer2_multirate` was never tested in that condition — its Saffran
stream had nothing in it but words.

## The stimulus

Fifty channels: three four-tone words (A–D, E–H, I–L) over twelve of them, and
thirty-eight carrying the cloud. Tones are 50 ms with a 10 ms gap on every
channel alike. Three clocks a third of a slot apart; in every block the three
words sit on three *different* ones, so no two words can ever start together,
and which clock a word gets rotates by block, so word identity carries no phase
either. Every one of the fifty channels is on for **exactly** the same total
time — one tone per 19-slot block.

Nineteen slots is forced, not chosen: exact duty matching needs
`n_voices × block_slots = 38`, and 38 factorises only as 2 × 19. The two cloud
voices are then pinned to an *adjacent* clock pair, because a 50 ms tone in a
60 ms slot leaves a hole only an adjacent clock fills — rotating the voices
instead opens a 10 ms silence at every block boundary, which would hand the
model a segmentation cue the design withholds.

| condition | what it removes |
|---|---|
| `paired` | nothing — the test case |
| `shuffled` | the order: each word's four channels land on four independently drawn slots instead of four consecutive ones |
| `sync` | the asynchrony between words: all three move onto clock 0 and start together, so the twelve channels are one object |

## Running

```bash
python -m tasks.interplay3.run_parallel --seeds 3
```

Three processes, one per condition, merged afterwards — about four minutes.
`python -m tasks.interplay3.interplay3 --figures-only` redraws from the cache.

## Result (5 units, 3 seeds each)

| condition | words with a spanning unit | word decoding | its own shuffled null |
|---|---|---|---|
| `paired` | **1.67 ± 0.47** of 3 | 0.399 ± 0.087 | 0.348 |
| `shuffled` | 0.33 ± 0.47 | 0.425 ± 0.107 | 0.349 |
| `sync` | 1.00 ± 0.00 | 0.333 ± 0.000 | 0.333 |

Coverage separates the conditions and the mechanism is intact. Three things
have to be said against it, though, and none of them should be dressed up.
Note that five units is a real constraint: see the capacity table below, where
twenty-four units on this same stimulus reach 3/3.

**The read-out is dead.** `shuffled` decodes nominally *above* `paired`, and
both sit at their own fitted-assignment nulls. With five units and three
classes there is no headroom. An earlier version of this stimulus, with a
sparser cloud and twenty-four units, gave 0.844 against a 0.405 null; at five
units the decoding panel should be dropped rather than shown as a null result.

**The span criterion has a false-positive mode.** `shuffled` produced a
spurious spanning unit in one seed of three. A word's four channels still fall
inside the same 1.14 s block when their order is destroyed, so "at least 0.25
of the row peak, at increasing rates" can be satisfied by block-level
co-occurrence alone. Requiring each predecessor to beat the **cloud band**
rather than a fraction of the row peak would close it, and can be computed
from the stored masks without re-running.

**Five units is the limit here, and the learning rate has to be tuned to the
population size.** Words spanned, on the dense cloud, seed 0:

| units \ eta | 1e-4 | 2e-4 | 5e-4 | 1e-3 | 2e-3 |
|---|---|---|---|---|---|
| 5 | 1 | **2** | 1 | 0 | -- |
| 12 | -- | 1 | 1 | **2** | collapse |
| 24 | -- | 2 | 2 | **3** | collapse |

Two things fall out of this table and neither is optional to state.

The optimum **moves with the population size** -- 2e-4 at five units, 1e-3 at
twelve and twenty-four -- because ``eta`` is per winning step and a unit in a
population of five wins five times as often as one in a population of
twenty-four. A units sweep at fixed ``eta`` therefore measures the tuning, not
the capacity. An earlier draft of this file reported exactly that mistake, and
concluded that the denser cloud had cost the third word; it had not.

Above the optimum the failure is not graceful. At 2e-3 the whole population
collapses onto a single committed unit at every size tested: with the cloud
dominating the map every unit chases the same average, and whichever pulls
ahead first keeps winning.

So the dense cloud is fine, and twenty-four units properly tuned still cover
all three words (17 commit, 10 on a word channel, 3/3 spanned). Five units
reach two of three at their own optimum. That is a capacity result, measured
with the rate tuned at each size rather than held fixed.

## Why the responses looked weak, and what actually fixes it

The raw traces sat on a large floor -- baseline 10, peak 25, 45% modulation --
and the cause is measurable. **Over half of every mask ends up on
cloud-by-cloud entries**, against about 2% on the block that codes its word.
Not because any single cloud entry is large -- they sit at 0.13 of a row peak
against 1.00 for the immediately preceding tone -- but because there are
38 x 6 cloud columns and 38 cloud rows, so the small entries win on count.
Five cloud tones sound at every instant, so that half of the mask is driven
continuously and contributes a constant floor.

**The published rule has no pressure against weak synapses.** ``lam`` decays
every synapse equally and prunes whole *units*; nothing prunes *synapses*, so
a mask can carry unlimited numbers of arbitrarily weak ones. Hebbian
competition in cortex does eliminate weak synapses, and that term was simply
missing. Adding it -- zero every entry below a fraction of the unit's largest,
after each update -- fixes the masks at source rather than at read-out. Five
units, paired, seed 0:

| threshold | 0 | 0.05 | 0.10 | 0.15 | **0.20** | 0.25 |
|---|---|---|---|---|---|---|
| cloud mass | 0.57 | 0.56 | 0.25 | 0.24 | **0.12** | 0.10 |
| mass on own row | 0.09 | 0.11 | 0.52 | 0.59 | **0.80** | 0.83 |
| raw modulation | 45% | 46% | 96% | 81% | **91%** | 55% |

The transition sits exactly at the cloud band: 0.05 is below it and changes
nothing, 0.25 is above the deepest predecessor (0.33 of a row peak) and starts
cutting the staircase itself. `PRUNE_FRAC = 0.20`.

Lateral inhibition in the competition -- letting a unit win when it stands out
from the population rather than when its absolute match is largest -- was tried
in the same sweep and **does nothing** (cloud mass 0.572, modulation 42%).
Changing who wins does not change what the winner learns, because the instar
target is still the raw coincidence map.

### What pruning bought, across 3 seeds

| | words covered | word decoding | its own null |
|---|---|---|---|
| baseline `paired` | 1.67 ± 0.47 | 0.399 ± 0.087 | 0.348 |
| **pruned `paired`** | 1.00 ± 0.00 | **0.584 ± 0.034** | 0.369 |
| pruned `shuffled` | 0.00 ± 0.00 | 0.487 ± 0.076 | 0.371 |
| pruned `sync` | 1.00 ± 0.00 | 0.333 ± 0.000 | 0.333 |

It is a genuine trade, and both halves have to be reported.

**Decoding recovers.** Baseline sat at its own null (0.399 vs 0.348) and was
not worth showing; pruned it clears it by a wide margin (0.584 vs 0.369), and
the variance drops threefold. The units that do form are far more selective.

**Coverage falls**, from 1.67 to exactly 1.00 in every seed. Sharper masks
mean a unit commits harder to what it already has, so five units spread even
less. Pruning does not solve the capacity limit -- it deepens it.

**The control tightened.** Baseline `shuffled` produced a spurious spanning
unit in one seed of three; pruned it produces none in any, and `sync` still
gives exactly one. The false positive came from cloud-driven mass, which is
what pruning removes.

## The parameter that had to move

Layer 2's instar rate, 5e-3 → 5e-4. Measured on 300 blocks with layer 1 in the
loop:

| eta | 5e-3 | 2e-3 | 1e-3 | 5e-4 |
|---|---|---|---|---|
| committed units | 1 | 1 | 13 | 24 |
| spanning units | 0 | 0 | 0 | **3 — all words** |

The fast rates do not merely fail to span, they collapse the population onto a
single unit: with the cloud dominating the map every unit chases the same
average and whichever pulls ahead first keeps winning.

## `fast_layer2.py`

An exact refactor of `Layer2MR`, kept here so `layer2_multirate/` is untouched:
a global decay scale, cached mask norms, and one matmul serving both the output
and the cosine match — one pass over the masks per step instead of three. Same
winners, same committed set, 3e-15 relative error, 3.2× faster.
`python -m tasks.interplay3.fast_layer2` prints the check.
