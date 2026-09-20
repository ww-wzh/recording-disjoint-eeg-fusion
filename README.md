# Recording-Disjoint EEG Fusion Evaluation

This repository accompanies the manuscript **Recording-Disjoint Evaluation of Selective Decision-Level Fusion for EEG Mental-Workload Classification: An Exploratory Methodological Audit**.

## Evidence scope

This is an exploratory methodological audit on a small public EEG dataset. Its main positive finding is methodological: an overlapping random-window split produced 45.33 percentage points higher balanced accuracy for Arithmetic and 48.46 points higher balanced accuracy for Stroop than the matched recording-disjoint split. DASF and CB-SF did not establish superiority over the always-neural comparator after participant-level inference and multiplicity correction. All 65 reported method comparisons were non-significant after Benjamini-Hochberg correction.

CB-SF uses all unlabeled recordings in a held-out participant/task cell to compute one target-batch weight. It is therefore transductive target-batch inference, not single-recording online prediction. Gate corrections and reviewer-requested extensions are explicitly treated as post-hoc exploratory analyses. This release does not claim formal risk control, a risk guarantee, safety, confirmatory non-inferiority, online deployment, external validation, or performance leadership.

## Canonical prediction file

The only prediction file used for the Route A v3 manuscript tables is:

```text
frozen/predictions_recording_route_a_v3.csv
```

- SHA-256: `1c0fd57fda8042293f23303499638b0f76d17c38c6a0e346b86613b5aa03aea0`
- Rows: 2,660 recording-method rows
- Methods: 14
- Rows per method: 190
- Cross-task participants: 13 (S01--S13)
- Arithmetic LOSO participants: 15 (S01--S15)
- Stroop LOSO participants: 13 (S01--S13)
- Evaluation unit: complete recording
- Inferential unit: participant
- Five seeds are ensembled before recording-level evaluation and are never inferential replicates

## Held-out-gain audit supplement

The `heldout_gate/` directory contains the post-hoc held-out-gain audit used
in the revised manuscript. It is supplementary evidence and does not replace
the canonical Route A v3 prediction file. The corrected files use the frozen
per-seed recording aggregation order and include the complete 14-comparison
table, recording/participant metrics, event counts, and exact sign-flip/
Wilcoxon sensitivity results. Run `heldout_gate/verify_heldout_gate.py` to
check this directory without retraining. The audit remains exploratory and
uses target-batch transductive inference; it is not a formal risk guarantee or
a single-recording online method.

## Main numerical context

- Bidirectional cross-task balanced accuracy: Always neural 52.88%, DASF 46.15%, CB-SF 50.00%.
- Arithmetic LOSO balanced accuracy: Always neural 58.33%, DASF 58.33%, CB-SF 60.00%.
- Stroop LOSO balanced accuracy: Always neural 61.54%, DASF 61.54%, CB-SF 59.62%.
- Random-window minus recording-disjoint: Arithmetic +45.33 pp; Stroop +48.46 pp.
- No method comparison remained significant after BH correction.

These are descriptive results under the frozen Route A v3 protocol, not evidence that the methods are equivalent.

## Public data

Raw EEG is not redistributed. Download Version 1 of the public dataset:

Nirabi et al. (2024), *Cognitive Load Assessment Through EEG: A Dataset from Arithmetic and Stroop Tasks*. Mendeley Data. https://doi.org/10.17632/kt38js3jv7.1

Set the environment variable `CBSF_DATA_ROOT` to the folder containing `Arithmetic_Data` and `Stroop_Data`, or place those two folders under `data/raw_data/` in the repository.

## Verify without retraining

Run `verify_release.py` directly in PyCharm. It verifies the canonical SHA-256, row schema, method and participant coverage, probability normalization, required artifacts, and every file listed in `MANIFEST.sha256`.

To regenerate only the final tables and figures from included audited results, run:

```text
supervisor_requested_audits/route_a_v3/93_生成_RouteA_v3最终论文主表补充表与无代码变量名图片.py
```

No model training is performed by either of those two verification/reporting steps.

## Repository layout

```text
frozen/                         canonical 2,660-row recording predictions
manuscript_artifacts/           final five main tables, seven supplements and three figures
revision_pipeline/              split, model, aggregation and gate implementation
route_a/                        raw/deep/Riemannian adapters used by the v3 runners
supervisor_requested_audits/    numbered v3 protocols, runners and audit outputs
tests/                          channel-selection unit test
verify_release.py               standalone integrity verification
```

See `RUN_ORDER.md`, `REPRODUCIBILITY.md`, and `ENVIRONMENT.md` before attempting a full refit.

## License and citation

Code is released under the MIT License. Repository: https://github.com/ww-wzh/recording-disjoint-eeg-fusion. After the repository version is approved, create a tagged GitHub release, archive that release with Zenodo, and then add the issued DOI to the manuscript and `CITATION.cff`.
