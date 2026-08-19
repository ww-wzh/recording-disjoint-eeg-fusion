# Reproducibility scope

## Canonical artifact

`frozen/predictions_recording_jmbe_final.csv` is the single source for all numbers in the final manuscript. It contains 3,360 recording-method rows covering 14 methods, 15 participants, bidirectional cross-task transfer, Arithmetic LOSO, and Stroop LOSO. It contains no seed or window field used as an inferential replicate.

The 2,520-row file under `route_a/frozen/` and the Stroop file under `supervisor_requested_audits/33_输出_Stroop_LOSO_14方法冻结统计表图/` are source components of the combined freeze. They must not be reported as separate alternative result sets.

## Artifact levels

- Level 1: the canonical recording predictions and `manuscript_artifacts/` reproduce the submitted numerical results.
- Level 2: the supervisor-requested output folders retain recording, participant, diagnostic, and selected seed/window probabilities needed to audit the combined freeze.
- Level 3: the public raw EEG dataset and released runners permit an independent refit. GPU nondeterminism and dependency versions may produce small probability differences.

## Analysis timing

The robust S4 correction, matched split audit, fixed-blend comparisons, Stroop LOSO extension, margin sensitivity analysis, and preprocessing sensitivity analysis followed inspection of earlier results. They are post-hoc exploratory. Historical protocol files are retained as development provenance, not offered as proof of public preregistration or of a prospective confirmatory non-inferiority design.

## Exclusions

- Raw EEG is obtained from Mendeley Data and is not redistributed.
- Model checkpoints and feature caches are excluded because they are regenerable and substantially larger than the audit artifacts.
- Smoke-test outputs and superseded manuscript tables are excluded.
- No independently frozen external cohort is included.

## Statistical unit

The held-out participant is the inferential unit. Five seeds are ensembled before recording decisions. Seeds, directions, recordings, and windows are not treated as independent inferential replicates.
