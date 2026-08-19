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
have to be said against it, though, and none of them should be dressed up:

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

**The cloud density is the binding constraint, not the population size.** At a
fixed eta of 2e-4 on this stimulus, seed 0:

| units | 5 | 12 | 24 |
|---|---|---|---|
| on a word channel | 5 | 5 | 7 |
| words spanned | 2 | 1 | 2 |

More units do not help. Full coverage (3/3, every seed) was obtained on the
*earlier* version of this stimulus, which had two simultaneous cloud tones
rather than five. Raising the cloud from 2 to 5 is what cost the third word.
One caveat on the table above: eta was tuned at five units, and because eta is
per winning step the optimum rises with population size, so the 12- and
24-unit points are probably under-tuned. They bound the effect rather than
measure it.

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
