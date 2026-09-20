# Reproducibility and interpretation

## Frozen result hierarchy

`frozen/predictions_recording_route_a_v3.csv` is the sole canonical prediction
file for the v7 manuscript. Its SHA-256 is recorded in
`frozen/neutral_preprocessing_protocol.json` and `MANIFEST.sha256`. It contains
recording-level rows only; windows and seeds are not inferential replicates.

## Neutral preprocessing contract

The public feature entry point is `1111.py`, which forwards to the maintained
`eeg_feature_pipeline.py`. Both use the eight-channel selector, full-recording
50-Hz Q=30 notch, 0.5--55 Hz fourth-order zero-phase filtering, no window
z-score, 2,125-sample windows, 125-sample stride, 4,096-point FFT and 272-D
features. Deep baselines use the same filtering and raw window scale through
`route_a/route_a_lib/data.py`.

## Statistical unit and timing

The participant is the inferential and bootstrap cluster. Cross-task directions
and recordings are not independent inferential replicates. DASF/CB-SF gate
auditing, the matched three-split audit, label-mapping analysis, seed/gate
stability, held-out-gain audit and preprocessing sensitivity analyses are
post-hoc exploratory. Non-significance is not interpreted as equivalence.

## Target-batch inference

One gate weight is estimated for a held-out participant/task batch using all
unlabeled target recordings in that batch: three target recordings per
cross-task direction and four recordings per LOSO task. This is transductive
target-batch evaluation, not a causal streaming or single-recording online
system.

## Data and exclusions

Raw EEG is downloaded from the cited public data repository and is not
redistributed. Feature caches, checkpoints, drafts, correspondence and local
logs are excluded. No independent external cohort is included.
