# Recording-Disjoint EEG Fusion Evaluation

This repository contains the code and frozen audit artifacts associated with the manuscript **Recording-Disjoint Evaluation Reveals Limited Benefit of Selective Decision-Level Fusion for EEG Mental-Workload Classification**.

## Scope of the evidence

This is a post-hoc methodological and negative-result study. Under recording-disjoint cross-task validation and fully nested participant-level LOSO, neither dynamic adaptive selection fusion (DASF) nor conditional benefit-aware soft fusion (CB-SF) established superiority over the neural comparator or simple fusion alternatives. DASF returned the neural branch in every evaluated gate cell. CB-SF matched Always-NN in bidirectional cross-task transfer and differed by one of 60 recordings in each LOSO task. The release does **not** claim confirmatory non-inferiority, safety, no-regret behavior, formal tail-risk control, external validation, or performance leadership.

## Canonical manuscript prediction file

The only prediction file from which the final manuscript tables and figures should be generated is:

```text
frozen/predictions_recording_jmbe_final.csv
```

- SHA-256: `1db8ae4b67520cf6c4ae07d4002971a3debe0bab395225c10f6265f1fab6d6d1`
- Rows: 3,360 recording-method rows
- Methods: 14
- Participants: 15
- Settings: bidirectional cross-task, Arithmetic LOSO, and Stroop LOSO
- Evaluation unit: recording
- Statistical unit: participant
- Seeds and windows are not inferential rows

The source component at `route_a/frozen/predictions_recording_route_a.csv` is retained only to regenerate the combined freeze. It is not a second manuscript result set.

## Main numerical findings

- Random-window validation was 49.00 percentage points above matched recording-disjoint validation for Arithmetic and 47.00 points above it for Stroop.
- Bidirectional cross-task accuracy was 68.33% for Always-NN, DASF, and CB-SF; EEG-Conformer was numerically highest at 74.17%.
- Arithmetic LOSO accuracy was 61.67% for Always-NN and DASF, 63.33% for CB-SF, and 68.33% for unconditional stacked fusion.
- Stroop LOSO accuracy was 61.67% for Always-NN and DASF, 63.33% for CB-SF, and 70.00% for EEGNet.
- None of 39 complete-setting or 26 direction-specific comparisons remained significant after Benjamini-Hochberg correction.

## Data

Raw EEG files are not redistributed. Download Version 1 of the public dataset:

Nirabi et al. (2024), *Cognitive Load Assessment Through EEG: A Dataset from Arithmetic and Stroop Tasks*. Mendeley Data. https://doi.org/10.17632/kt38js3jv7.1

Set `CBSF_DATA_ROOT` to the directory containing `Arithmetic_Data` and `Stroop_Data`, or place them under `data/raw_data/`. The released loader uses only eight EEG channels. Marker, status, counter, accelerometer, packet, and other auxiliary columns are excluded.

## Quick verification

Run `verify_release.py` directly in PyCharm or with Python. It verifies the final SHA-256, row schema, method/participant counts, unique recording keys, required audit artifacts, and the complete release manifest. Verification does not train a model.

To regenerate the final statistical tables and figures without model training, run:

```text
supervisor_requested_audits/40_生成_JMBE论文最终表图.py
```

The complete supervisor-requested sequence and the English purpose of every numbered entry point are documented in `SUPERVISOR_REQUESTED_AUDITS.md`.

## Repository layout

```text
revision_pipeline/              split, model, aggregation, and gate implementation
route_a/                        corrected Route A runners, protocols, tests, and source freeze
supervisor_requested_audits/    matched split audit, Stroop LOSO, sensitivity code and outputs
frozen/                         canonical 3,360-row manuscript prediction file
manuscript_artifacts/            final tables, figures, paired tests, and manifests
tests/                           protocol and feature-configuration tests
eeg_feature_pipeline.py         portable eight-channel data and feature loader
```

## Reproducibility status

All additions requested after inspection of earlier results are explicitly labelled post-hoc exploratory. The robust S4 correction, the -2.5/-5.0/-7.5 pp margin sensitivity analysis, the matched split audit, fixed blends, Stroop LOSO, and preprocessing sensitivity analysis were not prospectively registered. The target-batch gates use all unlabeled recordings from the held-out participant cell and are transductive; they are not single-recording online rules.

## License and citation

Code is released under the MIT License. The intended public repository is `https://github.com/ww-wzh/recording-disjoint-eeg-fusion`. Create a tagged GitHub release, archive that release with Zenodo, and then add the Zenodo DOI to the manuscript and repository metadata.
