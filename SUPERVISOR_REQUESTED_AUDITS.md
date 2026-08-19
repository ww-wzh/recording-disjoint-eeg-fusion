# Supervisor-requested audits and direct-run entry points

All items in this folder were added after inspection of earlier results and are therefore post-hoc exploratory unless the file itself states otherwise. The original numbered Chinese filenames are retained so that the released code is identical in organization to the audited run. Every Python file can be launched directly in PyCharm without command-line parameters.

## Order

1. `03_运行_三种数据划分对照.py`: matched random-window, recording-disjoint, and participant-disjoint comparison using one fixed learner and identical hyperparameters.
2. `10_运行_非劣效和双向CrossTask统计.py`: exploratory margin sensitivity, both cross-task directions, all 14 methods, participant-level paired tests, and BH correction.
3. `17_运行_Stroop_LOSO核心模型.py`: fully nested Stroop LOSO for the neural, random-forest, ExtraTrees, and OOF stacked branches.
4. `19_检查并汇总_Stroop_LOSO核心模型.py`: integrity and hash audit of the core Stroop outputs.
5. `22_运行_Stroop_LOSO融合和对照.py`: DASF, CB-SF, fixed blends, and recording-level OOF stacking for Stroop LOSO.
6. `25_运行_Stroop_LOSO_Riemannian基线.py`: Riemannian MDM and tangent-space logistic-regression baselines.
7. `27_运行_Stroop_LOSO_EEGNet基线.py`: fully nested Stroop LOSO EEGNet baseline.
8. `30_运行_Stroop_LOSO_EEGConformer基线.py`: fully nested Stroop LOSO EEG-Conformer baseline.
9. `32_运行_冻结Stroop_LOSO_14方法并生成统计表图.py`: freezes and checks the complete 14-method Stroop LOSO recording predictions.
10. `36_运行_预处理敏感性分析.py`: frozen fixed-learner sensitivity analysis for window standardization, recording standardization, 50-Hz attenuation, and legacy spectral multipliers.
11. `40_生成_JMBE论文最终表图.py`: combines the Route A and Stroop freezes into the canonical 3,360-row file and regenerates every final statistical table and figure.

## Important interpretation rules

- `frozen/predictions_recording_jmbe_final.csv` is the only canonical manuscript prediction file.
- Recording is the evaluation unit and participant is the inferential unit.
- Five seeds are ensembled before recording decisions; seeds are not independent samples.
- DASF/CB-SF gate results, S4 correction, margin thresholds, split audit, Stroop extension, and preprocessing sensitivity analysis are exploratory.
- No result establishes superiority, confirmatory non-inferiority, online deployment, safety, no-regret behavior, or formal risk control.
- Raw EEG, feature caches, and model checkpoints are intentionally excluded. They can be regenerated from the public dataset.
