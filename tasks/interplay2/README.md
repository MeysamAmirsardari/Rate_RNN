# interplay2 — two ordered tokens in a tone cloud

Does the layer-2 population give each recurring ordered pair a unit of its own?

## The stimulus

Twenty channels, no silence anywhere.

| channels | what they carry |
|---|---|
| A, B | token **AB** — B one slot after A, on the integer clock |
| C, D | token **CD** — D one slot after C, on a clock offset by half a slot |
| 4–19 | the cloud: two voices, one per clock, no ordered pair in them |

Tones are 50 ms with a 10 ms gap after every one, on every channel alike, so
the stream stays isochronous and the gap marks nothing. Every channel is on
for **exactly** the same total time (one tone per 480 ms block, 10.4% duty),
so a token cannot be found by rate. The two tones of a token never overlap, so
it cannot be bound as a chord. Each token has one background voice synchronous
with it and one asynchronous, so "token" is not confounded with "phase". The
cloud never falls silent, which is asserted rather than assumed: a silent
moment would reset the layer-2 trace and hand the model a segmentation cue.

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
file: `interplay2_cloud` (the stimulus as a tone cloud, channel order shuffled),
`interplay2_stimulus` (the same stimulus with its constraints measured),
`interplay2_tape` (unit activity along the input), `interplay2_allocation`.

## The two constants that had to move

**Instar learning rate, 5e-3 → 5e-4.** Each mask is an exponential moving
average of the normalised coincidence map over the moments that unit wins,
with a memory of 1/eta winning steps. With two channels and silence between
chunks that map is nearly empty and a short memory suffices; here the cloud is
on at every sample, the map has of order ten comparable entries at every
moment, and a short memory leaves each mask tracking whichever background pair
happened to be on.

**Conductance kinetics, 40/150 ms → 22/30 ms.** What matters is the kernel's
width *in slots* — how many past channels still carry a trace when the second
tone arrives — and that ratio is scale-free, so the timing is set by the
paradigm and the kinetics follow. At the 60 ms slot the plateau of best
contrast is tau_rise 20–30 ms crossed with tau_decay 18–30 ms. **These are
faster than NMDA**: the GluN2A reading available at a 100 ms slot (36/50 ms) is
not available here. The contrast at the optimum is the same either way (3.45),
so nothing is lost except that interpretation. Widening the kernel is not free:
at tau_decay = 120 ms the token pair is only 1.9× a flat mask entry and the
population never differentiates at all.

See `config.py` for the sweeps behind both.

## Result (5 seeds each)

| condition | allocation | AB vs CD decoding | shuffled-label null |
|---|---|---|---|
| `paired` | **two units, one per token, 5/5** | 0.853 ± 0.021 | 0.510 |
| `shuffled` | neither token, 5/5 | 0.550 ± 0.021 | 0.518 |
| `sync` | one unit, both tokens, 5/5 | 0.500 ± 0.000 | 0.500 |

The mask entry for the ordered pair reaches **16.3 ± 1.1** times a flat mask in
`paired`, against **0.9 ± 0.1** in `shuffled`. In `sync` the within-token and
cross-token entries are equal to two decimal places (14.25 vs 14.25, ratio
1.00): the four channels are one object and the mask cannot tell AB from CD,
which is what gives "two units" its meaning in `paired`.

### What the units actually learned

A committed mask is close to separable — a "B is firing now" row times an
"A fired recently" column — which is what a coincidence subunit multiplying
two afferents *should* look like, not a defect. The load-bearing question is
whether the second factor picks out A, and it does: given that B is firing,
the owning unit puts **19.4%** of its row on A against **4.5 ± 1.2%** on each
of the other channels (flat = 5.3%), a ratio of 4.4. C before D is the same
(18.8% vs 4.5 ± 1.0%).

This also rules out the obvious alternative, that the layer simply assigns one
unit per salient channel: in `shuffled`, B and D fire exactly as often and for
exactly as long, and no unit forms on either. The contingency is required.

### Layer 1's contribution

| layer 1 | two units | decoding |
|---|---|---|
| `raw` (no cortex) | 1/5 | 0.686 ± 0.152 |
| `frozen` (adaptation + inhibition, no learning) | 5/5 | 0.840 ± 0.032 |
| `full` | 5/5 | 0.853 ± 0.021 |

Layer 1 helps, and the help is **adaptation and selective inhibition, not
recurrent plasticity** — `frozen` already recovers the whole effect and `full`
adds nothing on top of it. The dependence is stronger at this timing than at
the 100 ms slot, where `raw` still managed 3/5.
