# Held-out-gain gate audit

This directory contains the post-hoc held-out-gain gate audit reported in the
revised manuscript. It is supplementary evidence and does not replace the
canonical v7 neutral prediction file in `../frozen/`.

## Scope

The audit reconstructs the gain response in the frozen order: each seed is
first aggregated to recording-level probabilities, the ordinary recording
accuracy difference is computed for that seed, and the five seed-specific
differences are then median-aggregated. The fixed gate target is ordinary
recording accuracy; the reported primary endpoint remains participant-macro
balanced accuracy. The audit is therefore exploratory and must not be read as
a formal risk guarantee or as direct optimization of balanced accuracy.

The held-out-gain analysis is transductive at the target-batch level. It uses
the frozen prediction outputs and the frozen protocol; it is not a
single-recording online procedure.

## Key files

- `heldout_gate_method_comparisons_BH14.csv`: all 14 participant-level
  comparisons against the Always-neural reference, with raw and BH-adjusted
  p-values.
- `heldout_gate_recording_and_participant_metrics.csv`: recording-level and
  participant-level metrics used for the audit.
- `heldout_gate_exact_sign_flip_wilcoxon_BH.csv`: the complete sensitivity
  comparison table with exact sign-flip and Wilcoxon results.
- `heldout_gate_recording_predictions.csv`: corrected held-out-gain recording
  predictions used by the participant-level summaries.
- `heldout_gate_window_predictions.csv`: supporting window-level predictions.
- `heldout_gate_protocol.json`: frozen aggregation and exclusion rules.
- `MANIFEST.sha256`: SHA-256 checksums for every file in this directory.

The primary v7 neutral recording predictions remain
`../frozen/predictions_recording_route_a_v3.csv`; its SHA-256 is verified by
the repository-level `verify_release.py`.

## Verification

Run `verify_heldout_gate.py` from the repository root. This checks the local
manifest, required columns, row coverage, and the corrected aggregation
provenance without retraining any model.
