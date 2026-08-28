# Reproducibility and interpretation

## Frozen result hierarchy

`frozen/predictions_recording_route_a_v3.csv` is the only canonical prediction file for the Route A v3 manuscript. It contains no window or seed pseudo-replicates. The identical source file is retained inside the numbered audit folder solely so that script 93 can be executed without path changes.

## Statistical unit

The participant is the inferential and bootstrap cluster. Cross-task direction and recording are not treated as independent inferential replicates. Five random seeds are ensembled before each recording decision.

## Analysis timing

DASF/CB-SF gate auditing, robust CB-SF handling, the matched three-split audit, Stroop LOSO extension, label-mapping analysis, margin analysis and preprocessing sensitivity analysis are post-hoc exploratory. The paper reports failure to establish superiority; non-significance is not interpreted as equivalence.

## Target-batch inference

One gate weight is estimated for a held-out participant/task cell using all unlabeled target recordings in that cell: three target recordings in each cross-task direction and four in each LOSO task. This is transductive target-batch evaluation, not a causal streaming or single-recording online system.

## Exclusions

- Raw EEG is downloaded from the public data repository and is not redistributed.
- Feature caches, raw window predictions and model checkpoints are excluded because they are regenerable and large.
- Manuscript drafts, correspondence and drafting material are excluded.
- No independently frozen external cohort is included.

## Source portability

The released `1111.py` and `eeg_feature_pipeline.py` are identical sanitized copies of the frozen feature implementation. Only the subject path table is made portable through `CBSF_DATA_ROOT`; feature equations and frozen preprocessing constants are unchanged. Numbered public runner copies use the repository root rather than the author's parent workspace when locating this file.
