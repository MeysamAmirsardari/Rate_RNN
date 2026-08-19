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

## Result (3 seeds each)

| condition | words with a spanning unit | word decoding | shuffled-label null |
|---|---|---|---|
| `paired` | **3.00 ± 0.00 of 3** | 0.844 ± 0.028 | 0.405 |
| `shuffled` | 0.00 ± 0.00 | 0.462 ± 0.010 | 0.475 |
| `sync` | 1.00 ± 0.00 | 0.333 ± 0.000 | 0.333 |

`paired` gives all three words a spanning unit in 3/3 runs. `shuffled` gives
none, and decodes *at* its own null. `sync` collapses to exactly one spanning
unit every time, at exactly chance — the twelve channels have become one
object, and a read-out cannot recover which word it is because there is no
longer a which.

### The mechanism, drawn

Panels a–c of `interplay3_allocation` are the whole claim: for each word's
spanning unit, the mask weight on each earlier tone against filter rate. The
staircase is exact and identical across all three words —

| tones back | 1 | 2 | 3 |
|---|---|---|---|
| peak filter rate | 53 ms | 92 ms | 162 ms |
| weight (of row peak) | 1.00 | 0.47 | 0.33 |

against a cloud band at ~0.13. This is the multi-rate mechanism doing exactly
what it is for: one mask holding "D now, C at a fast rate, B at a slower one,
A at the slowest".

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
