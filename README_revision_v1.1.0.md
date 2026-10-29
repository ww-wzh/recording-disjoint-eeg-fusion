# Revision evidence v1.1.0

This release adds the machine-readable evidence for the revised Sensors analysis:

Recording-Disjoint Validation Reveals Large Split-Dependent Differences in EEG Mental-Workload Classification: An Exploratory Audit of Selective Fusion

The package contains fixed-learner split controls, endpoint-aligned balanced-gain sensitivity results, and the participant-feasibility record used to explain the different sample sizes in Supplementary Table S25. The earlier v1.0.0 held-out-gain release remains available as a historical version and is not overwritten.

## Contents

- `split_control/`: fixed Logistic learner temporal, local-filter, and non-overlap controls.
- `balanced_gain/`: balanced-accuracy-gain sensitivity and stability files.
- `participant_feasibility.csv`: protocol-based feasibility record for the S25 panels.
- `MANIFEST_revision_v1.1.0.sha256`: SHA-256 values for the files in this revision evidence release.

All files are derived from the frozen revision analysis. The manuscript is not included in this evidence-only release.
