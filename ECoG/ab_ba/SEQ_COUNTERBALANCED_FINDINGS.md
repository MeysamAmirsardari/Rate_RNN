# Counterbalanced SEQ analysis — what the recoverable data can and cannot show

Run with `python -m ECoG.ab_ba.seq_counterbalanced`. Outputs land in
`results/seq_counterbalanced/`.

## 1. The AB/BA recordings themselves: still not analysable, now proven

`scripts_AB_BA.m` builds its panels from `allMatrices{expnum,ch}.allM2`, which
`Gen_M2Mat.m` builds from `xx.outp{expnum}`. `Gen_M2Mat.m` **is** on this
machine; `ft_oe_list.m`, which produces `xx`, is not. What `Gen_M2Mat` needs
from it is `outp.stimat` — a per-sequence table of `[trial, stimulus index,
onset ms, offset ms]` — plus `outp.dataMat`, `outp.stims`, `outp.dur`,
`outp.pre`. Everything downstream is deterministic once `stimat` exists.

Checked directly for the 2026-04-30/05-01 recordings:

| Candidate source | Result |
|---|---|
| `all_channels.events` | 34 records: 17 rising, 17 falling on one TTL line. One setup pulse plus 16 block pulses ~46.5 s apart. No per-sequence mark. |
| `messages.events` | Clock text only (`Software time: … @100000Hz`). |
| `Continuous_Data.openephys.xml`, `settings.xml` | Acquisition configuration only. |
| ADC channels | **4 kHz sample rate**; ADC2–ADC8 only, no ADC1. The 5.3 and 9.4 kHz tones are above Nyquist, so no ADC channel can carry tone identity even in principle. |
| `.fig` payloads | MATLAB `Line` and `Bar` objects only — plotted means, no trial dimension. |

Sequence **timing** is recoverable (block onset plus a fixed cadence). Sequence
**identity** is not, and it lives only in the Baphy parameter file that
`ft_oe_list` reads — `mfile` in a sibling `xx` points at
`D:\ECoG_Data\Nugmeg\nmg002\nmg002a07_p_SEQ.m`, a Windows path not present here.

## 2. Two complete recordings do exist, and they are counterbalanced

`ECoG_data/` holds two full `ft_oe_list` outputs for the SEQ runclass:

| File | stim 1 | stim 2 |
|---|---|---|
| `Nugmeg_2025-03-03_12-31-33.mat` | 20% (rare) | 80% (frequent) |
| `Nugmeg_2025-03-03_13-35-18_SEQ3.mat` | 80% (frequent) | 20% (rare) |

Same animal, same day, about one hour apart. 10 acquisition trials × 25
sequences, 32 channels at 1 kHz, complete `stimat`. The stimuli are
`800--800--800--800-6000` and `800-800-800-800-800`; they differ in the final
note, and their probabilities are reversed between recordings, so the **same
physical sequence is rare in one recording and frequent in the other**.

This is the identity-controlled probability reversal the manuscript argues for,
with trial-level data — so the analysis was run to completion.

## 3. Results

Per acquisition trial, `d = response(stim 1) − response(stim 2)` holds acoustics
fixed. The interaction `d(rec_a) − d(rec_b)` equals
`(stim1 rare − stim1 frequent) + (stim2 rare − stim2 frequent)`, cancelling the
acoustic difference and any additive recording offset.

| Test | Result |
|---|---|
| GFP, cluster mass, full 0–1500 ms | best cluster 655–673 ms, *P*<sub>FWER</sub> = 0.24 — **null** |
| GFP, cluster mass, a-priori deviant window 550–900 ms | 655–673 ms, *P*<sub>FWER</sub> = 0.084 — **null** |
| Per channel, max-\|t\| over 32 channels × 1500 ms | ch 16 at 669 ms, *t* = −7.05, *P*<sub>FWER</sub> = **0.0198**; ch 12 at 669 ms, *P* = 0.043 |

The surviving term is perfectly separated across trials — every rec_a trial is
more negative than every rec_b trial — and an exact enumeration of all 184,756
splits at that channel and latency gives *P* = 1.1 × 10⁻⁵.

## 4. Why the significant term must not be reported as a probability effect

**The gain control.** Both sequences open with the same 800 Hz note, so 0–130 ms
is matched across stimulus *and* recording. Recording A is nonetheless larger
there: **+9.9%** at channel 16 and **+5.4%** on population GFP. The recordings
differ in gain, and the additive interaction does not remove a gain difference
when the acoustic contrast is large.

**The magnitude.** At the surviving latency the interaction is **0.74 ×** the
acoustic difference between the two sequences. A mismatch response that is
three-quarters the size of a 6 kHz-versus-800 Hz contrast is not credible.

**The design, which is decisive.** Probability was manipulated *between*
recordings. The 20 acquisition trials are nested within two recordings, so they
are pseudo-replicates: the permutation is exact for the null "trials are
exchangeable between recordings", but the scientific null is "probability had no
effect", and for that the effective replication is **one recording against one**.
Perfect between-recording separation with zero within-recording overlap is the
signature of a session difference, not of a trial-level effect. Any
session × stimulus difference — drift, impedance, depth of anaesthesia — is
aliased onto the term being estimated and cannot be separated from it.

The counterbalanced interaction removes an *additive* session effect. It cannot
remove a session-by-stimulus one, and that is exactly what is left.

## 5. Consequence for Figure 4 panels B and C

Panels B and C compare the rare sequence from one day against the post-rare
sequence from the other day (`scripts_AB_BA.m` lines 79–94: `tmp_01` from
`allMatrices2`, `tmp_15` from `allMatrices1`). That is the same between-session
comparison analysed above, without even the counterbalancing.

The analysis here shows a comparison of that form produces a ~7% difference in
the expected direction that survives correction and is nonetheless attributable
to the session. So the panels stay descriptive, and now for a demonstrated
reason rather than a missing-file one.

## 6. What would make this testable

Replication at the level at which probability is manipulated. Either

1. **at least three or four recordings per probability assignment**, so that
   recording becomes a replicated random factor and the interaction can be
   tested against between-recording variance; or, much better,
2. **reverse the probabilities within a single recording**, in interleaved
   blocks. Then probability varies within session, block becomes the resampling
   unit, and session drift is orthogonal to the contrast by construction.

Option 2 also fixes the gain confound outright and needs no more animals. With
16 blocks per recording, alternating the probability assignment across blocks
would give 8 blocks per condition per recording — ample for a block-level
permutation test of exactly the kind now running on the Figure 2 ERPs.

The prespecified pipeline in `ECOG_INFERENCE_AUDIT.md` §"Prespecified recovery
analysis" remains correct and is implemented here; it only needs a design with
replication to become interpretable.
