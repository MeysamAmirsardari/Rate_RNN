# interplay2 — two ordered tokens in a tone cloud

Does the layer-2 population give each recurring ordered pair a unit of its own?

## The stimulus

Twenty channels, no silence anywhere.

| channels | what they carry |
|---|---|
| A, B | token **AB** — B one slot after A, on the integer clock |
| C, D | token **CD** — D one slot after C, on a clock offset by half a slot |
| 4–19 | the cloud: two voices, one per clock, no ordered pair in them |

Every channel is on for **exactly** the same total time (one 100 ms tone per
800 ms block, 12.5% duty), so a token cannot be found by rate. The two tones
of a token never overlap, so it cannot be bound as a chord. Each token has one
background voice synchronous with it and one asynchronous, so "token" is not
confounded with "phase".

## Conditions

| condition | what it removes |
|---|---|
| `paired` | nothing — the test case |
| `shuffled` | the contingency: A, B, C, D keep their duty but land on independent slots, so the lag is random in sign and size (the +1-slot lag still occurs at its 1/8 base rate) |
| `sync` | the asynchrony *between* tokens: CD moves onto the integer clock and starts with AB, making the four channels one object |

Layer-1 modes `raw` / `frozen` / `full` separate what the cortical dynamics
contribute from what the readout contributes.

## Running

```bash
python -m tasks.interplay2.interplay2 --seeds 5 --layer1-sweep
```

`--figures-only` redraws from `results.pkl`. Figures are written next to this
file: `interplay2_stimulus`, `interplay2_tape` (unit activity along the input),
`interplay2_allocation`.

## The one parameter that had to move

Layer 2's instar learning rate, 5e-3 → 2.5e-4. Each mask is an exponential
moving average of the normalised coincidence map over the moments that unit
wins, with a memory of 1/eta winning steps. With two channels and silence
between chunks that map is nearly empty and a short memory suffices; here two
cloud tones are on at every sample, the map has of order ten comparable
entries at every moment, and a short memory leaves each mask tracking whichever
background pair happened to be on. The conductance kinetics also moved, to
36/50 ms — see `config.py` for the measurement behind both.

## Result (5 seeds each)

| condition | allocation | AB vs CD decoding | shuffled-label null |
|---|---|---|---|
| `paired` | **two units, one per token, 5/5** | 0.870 ± 0.020 | 0.514 |
| `shuffled` | neither token, 5/5 | 0.554 ± 0.038 | 0.525 |
| `sync` | one unit, both tokens, 5/5 | 0.500 ± 0.000 | 0.500 |

The mask entry for the ordered pair reaches **18.4 ± 0.8** times a flat mask in
`paired`, against **1.0 ± 0.3** in `shuffled`. In `sync` the within-token and
cross-token entries are equal to two decimal places (14.39 vs 14.39, ratio
1.00): the four channels are one object and the mask cannot tell AB from CD,
which is what gives "two units" its meaning in `paired`.

### What the units actually learned

A committed mask is close to separable — a "B is firing now" row times an
"A fired recently" column — which is what a coincidence subunit multiplying
two afferents *should* look like, not a defect. The load-bearing question is
whether the second factor picks out A, and it does: given that B is firing,
the owning unit puts **19.2%** of its row on A against **4.5 ± 0.9%** on each
of the other channels (flat = 5.3%), a ratio of 4.3. Same for C before D.

This also rules out the obvious alternative, that the layer simply assigns one
unit per salient channel: in `shuffled`, B and D fire exactly as often and for
exactly as long, and no unit forms on either. The contingency is required.

### Layer 1's contribution

| layer 1 | two units | decoding |
|---|---|---|
| `raw` (no cortex) | 3/5 | 0.690 ± 0.156 |
| `frozen` (adaptation + inhibition, no learning) | 5/5 | 0.872 ± 0.011 |
| `full` | 5/5 | 0.870 ± 0.020 |

Layer 1 helps, and the help is **adaptation and selective inhibition, not
recurrent plasticity** — `frozen` already recovers the whole effect and `full`
adds nothing on top of it.
