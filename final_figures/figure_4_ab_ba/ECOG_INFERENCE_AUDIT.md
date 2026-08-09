# Figure 4 ECoG inference audit

## RESOLVED 2026-08-09 - the playback table was found

`/Users/eminent/Downloads/nmg038_039.zip` holds the Baphy session folders for
both 2026 recordings. `nmg038` is 2026-04-30, `nmg039` is 2026-05-01, and each
`tmp/<run>_p_SEQ.mat` contains `exptevents`: one row per sequence with its
name, its onset relative to the trial, and its trial number. That **is**
`outp.stimat`, so `allM2` column 4 - the only piece that could not be rebuilt
from the Open Ephys archive - is available after all. Earlier searches missed
it because they used the animal name ("Nutmeg"/"Nugmeg") rather than the Baphy
prefix `nmg`, and looked for `.mat` data files rather than a session zip.

Recovered parameters, which match the experiment sheet exactly:

| Run | Date | Standard | Deviant | Note / gap / seq gap | Blocks x seq |
|---|---|---|---|---|---|
| nmg038a01 | 2026-04-30 | 9400-5300 (344) | 5300-9400 (56) | 0.18 / 0 / 1.5 s | 16 x 25 |
| nmg039a01 | 2026-05-01 | 5300-9400 (340) | 9400-5300 (60) | 0.18 / 0 / 1.5 s | 16 x 25 |
| nmg038a03 | 2026-04-30 | 1500-4000 (340) | 4000-1500 (60) | 0.05 / 0.1 / 1.5 s | 16 x 25 |
| nmg039a03 | 2026-05-01 | 4000-1500 (347) | 1500-4000 (53) | 0.05 / 0.1 / 1.5 s | 16 x 25 |

`Deviant_pct` is 15 in every run: the probability is the same in both, and the
counterbalancing is in *which* physical sequence carries it.

## Pipeline

`ECoG/ab_ba/baphy_events.py` reads the playback table.
`ECoG/ab_ba/reconstruct_2026.py` locates the acquisition blocks - the TTL HIGH
interval, 46 525 ms, not the rise-to-rise interval which includes the pause -
and applies the 1-250 Hz band requested by `scripts_AB_BA.m`; without it
the epochs are pure drift.
`ECoG/ab_ba/ab_ba_inference.py` epochs all 400 sequences per recording and runs
the identity-controlled test.

The publication Panel B now uses
`ECoG/ab_ba/ab_ba_channel_erp.py`: odd blocks select a contact, even blocks
estimate and test it, and exact block-label permutations control the joint
AB/BA/time cluster family. Neural samples are never indexed until after the
six reconstructed `allM2` tag fields.

## Result

Each physical sequence is 15% rare in one recording and 85% regular in the
other, so rare-minus-regular holds acoustics fixed. Odd blocks form an
independent contact-discovery split. Within each even held-out block, rare
presentations are matched to the nearest-in-position unused regular
presentations before the block mean is formed, preventing the more numerous
regular trials from receiving artificially smaller measurement noise. The
two-sided cluster test uses unsmoothed block means, exact enumeration of all
12,870 assignments and one maximum family over both sequences and all
0–600-ms samples.

| Contrast | Held-out contact | smallest P_FWER |
|---|---|---|
| 5300-9400 rare vs regular | contact 10 | 0.547 |
| 9400-5300 rare vs regular | contact 29 | 0.975 |

**No cluster survives correction.** Panel B therefore draws no significance
rail and marks both facets `cluster-FWER n.s.`. Exact non-significant clusters
remain in `ECoG/ab_ba/results/ab_ba_channel_erp/clusters.csv` rather than being
discarded.

## Standing caveat

Probability is manipulated between recordings, one per assignment, and the two
are on different days. They are treated as one session at the experimenter's
instruction, pending re-recording. Reversing the probability assignment across
interleaved blocks *within* one recording would remove the confound entirely
and make the block the unit for the manipulation as well as for the resampling.

## Historical record

### What was actually available


- Four raw Open Ephys recordings: SEQ1 and SEQ3 on each of two acquisition
  days. The speech-token SEQ2 recordings are absent.
- Each raw event file contains one setup pulse and 16 whole-block pulses. It
  does not contain the identities or timings of the 25 sequences within a
  block.
- Six MATLAB figure files. Each retains three difference means, three rare
  means, three regular means, one 32-contact activation vector and one
  20-sample-smoothed accuracy curve. No trial epochs, fold predictions,
  bootstrap draws, null distribution, variance estimate or interval survives.

Only the two Experiment 1 figure extracts are used in the current manuscript
panel. The remaining speech and gapped-tone extracts are retained for source
provenance and are not pooled into Figure 4.

## Missing provenance

The original MATLAB analysis obtained sequence number, identity, onset and
offset from `ft_oe_list`'s `outp.stimat`. Neither `ft_oe_list.m` nor the
`stimat`/`xx1`/`xx2`/`allMatrices` structures are present. The expected bridge
file is also absent:

`/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat`

Neural activity and undocumented ADC fluctuations must not be used to infer
these missing labels, because doing so would make the outcome define its own
target.

## Exact Experiment 1 export invocation

The repository bridge accepts an experiment selector. With `ft_oe_list.m`,
`Gen_M2Mat.m` and their Open Ephys dependencies on the MATLAB path, the
lossless Experiment 1-only invocation is:

```matlab
addpath('/Users/eminent/Projects/ECoG');
addpath('/Users/eminent/Projects/Rate_RNN_py/ECoG/ab_ba/matlab');
export_ab_ba_preprocessed( ...
    '/Users/eminent/Projects/ECoG/AB_BA', ...
    '/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat', ...
    1);
```

The third argument `1` is required here so the recovery scope matches the
displayed zero-gap experiment. The export should contain both 32 x trial x
time condition arrays, original source-row identifiers, trial/group labels,
the physical sequence identity and the zero-based time axis. The present
machine cannot create this file because `ft_oe_list.m` and its playback table
are absent.

## Problems in the legacy MATLAB decoder

- all observations were standardized before cross-validation;
- acquisition-block membership was not protected across folds;
- channels and latency were selected on the same full data;
- folds were repartitioned across time with an unseeded random generator;
- only one accuracy value per time sample was saved, followed by a 20-sample
  moving mean;
- the only ERP significance code is commented out and subtracts
  `X01_all - X01_all`.

These curves are retained only as source provenance and are not styled as a
new leakage-safe refit.

## Prespecified recovery analysis

If the original playback table and trial epochs are recovered, use a
counterbalanced probability-reversal interaction rather than treating the two
Experiment 1 cross-day curves as independent tests:

1. train within day 1 on the two sequence roles and test day 2, where acoustic
   identity reverses role;
2. symmetrize day 2 to day 1;
3. use the corresponding difference-of-differences for ERPs;
4. keep normalization, feature selection and channel selection inside
   training folds;
5. keep every epoch from an acquisition block in the same fold;
6. resample acquisition blocks, preserve day/block/physical identity in
   permutations, refit the complete decoder for every permutation and control
   the full time family with a cluster maximum; and
7. report observation counts, blocks, fold construction, corrected
   probabilities and the conditional scope of the result.

With one animal and two acquisition days, resulting intervals and tests would
still be conditional on those recordings rather than animal-population
inference.
