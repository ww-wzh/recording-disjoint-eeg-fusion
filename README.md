# Recording-Disjoint EEG Fusion Evaluation

This repository accompanies the manuscript **Recording-Disjoint Evaluation of
Selective Decision-Level Fusion for EEG Mental-Workload Classification: An
Exploratory Methodological Audit**. The canonical package is the neutral
preprocessing analysis released as version 1.0.0.

## Evidence scope

This is an exploratory audit on a small public EEG dataset. The primary
methodological result is the large gap produced by an intentionally leaky
random-window comparator: relative to the matched recording-disjoint split,
the participant-macro balanced-accuracy gap is **46.50 percentage points for
Arithmetic** and **47.69 percentage points for Stroop**. These values describe
split-induced evaluation differences; they are not a causal estimate of window
overlap alone.

DASF and CB-SF did not establish superiority over the always-neural comparator
under participant-level inference and multiplicity correction. The selective
fusion analyses are exploratory and do not claim formal risk control, a risk
guarantee, equivalence, external validation, safety certification, online
deployment, or performance leadership.

## Canonical prediction file

The only primary prediction file is:

```text
frozen/predictions_recording_route_a_v3.csv
```

- SHA-256: `cba069dd3837e9b3894dc2649aa6c8a9ac80812de996b2dc51cc0aa056f5c5cd`
- Rows: 2,660 recording-method rows
- Methods: 14
- Rows per method: 190
- Cross-task participants: 13 (S01--S13)
- Arithmetic LOSO participants: 15 (S01--S15)
- Stroop LOSO participants: 13 (S01--S13)
- Evaluation unit: complete recording
- Inferential unit: participant
- Five seeds are ensemble members, not inferential replicates

The exact preprocessing and aggregation contract is in
`frozen/neutral_preprocessing_protocol.json`. No raw EEG is redistributed.

## Neutral preprocessing

The canonical pipeline selects eight EEG channels after removing the OpenBCI
packet counter, applies a complete-recording 50-Hz Q=30 notch and a zero-phase
0.5--55 Hz fourth-order Butterworth filter, and does not apply window or
recording z-score normalization. It uses 8.5-second windows with a 0.5-second
stride, 2,125 samples per window, a 4,096-point FFT, neutral feature
multipliers, 272 features, and a 36-dimensional eight-channel Riemannian
tangent space.

## Held-out-gain audit

The `heldout_gate/` directory contains the corrected post-hoc held-out-gain
audit. It uses the frozen per-seed recording aggregation order and includes the
complete 14-comparison table, recording/participant metrics, event counts, and
the 79-comparison sensitivity family. Run
`heldout_gate/verify_heldout_gate.py` to verify it without retraining. This
audit is target-batch transductive inference, not single-recording online
prediction; the gain target remains ordinary recording accuracy while the
reported primary endpoint is balanced accuracy.

## Matched three-split audit

The reproducible protocol is in
`manuscript_artifacts/v7_neutral/v7_neutral_matched_three_split_protocol.md`.
All three split strategies use the same features, StandardScaler, balanced
logistic regression and recording-level geometric probability aggregation; only
the split unit changes. The random-window comparator is deliberately retained
as a leakage audit control.

## Verify without retraining

Run `python verify_release.py` to check the canonical SHA-256, schema, method
coverage, probability normalization, neutral protocol, and every file listed
in `MANIFEST.sha256`. Run `python heldout_gate/verify_heldout_gate.py` for the
held-out-gain supplement. `pytest -q` runs the lightweight channel-selection
tests.

## Repository layout

```text
frozen/                         canonical predictions and neutral protocol
manuscript_artifacts/v7_neutral/ frozen tables, figures and audit outputs
heldout_gate/                   held-out-gain results and verification
revision_pipeline/              split, model, aggregation and gate code
route_a/                        raw/deep/Riemannian adapters
tests/                          lightweight unit tests
verify_release.py               standalone integrity verification
```

Raw EEG must be downloaded independently from the public dataset cited in the
manuscript and supplied through `CBSF_DATA_ROOT`. Do not add raw EEG,
checkpoints, caches, Word drafts, or private local paths to this repository.

## License and citation

Code is released under the MIT License. Repository:
https://github.com/ww-wzh/recording-disjoint-eeg-fusion
