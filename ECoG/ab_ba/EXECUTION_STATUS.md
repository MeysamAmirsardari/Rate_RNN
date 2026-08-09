# Execution status

## Completed locally

- The supplied MATLAB script and its exercised `Gen_M2Mat` branch were audited.
- All four legacy Open Ephys recordings passed structural checks: 4 kHz,
  32 neural channels, matched per-channel lengths/timestamps, paired TTL edges,
  and 16 complete long trials per recording.
- All six supplied MATLAB reference figures were identified, hashed, and their
  saved legacy peak times/top-three channels extracted.
- The compact MATLAB export bridge, Python importer, source-faithful decoder,
  leakage-safe grouped decoder, outputs, provenance, and tests are complete.
- Unit tests and static checks pass.

## Required before the six decoders can run

`/Users/eminent/Projects/ECoG/AB_BA/ab_ba_preprocessed_export.mat` does not yet
exist. It must be produced in the original MATLAB environment with
`ft_oe_list.m` and its Open Ephys/playback-log dependencies on the MATLAB path.
The exact command is in `README.md`.

This is not replaced by response-derived label inference. The raw
`all_channels.events` files provide whole-trial boundaries, not the identity of
each of the 25 sequences. Deriving those labels from ECoG responses and then
decoding the same responses would be circular and scientifically invalid.
