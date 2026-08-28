# Route A v3 run order

All entry points are located in `supervisor_requested_audits/route_a_v3/` and can be run directly in PyCharm without command-line arguments.

## Full refit from public raw EEG

1. `65_运行_RouteA_v3数据策略与哈希预检.py`: verify pure eight-channel parsing, cohort rules, physical-file hashes and split boundaries.
2. `67_运行_RouteA_v3核心模型_双向CrossTask.py`: run both cross-task directions.
3. `68_运行_RouteA_v3核心模型_Arithmetic_LOSO.py`: run fully nested Arithmetic LOSO.
4. `69_运行_RouteA_v3核心模型_Stroop_LOSO.py`: run fully nested Stroop LOSO.
5. `71_汇总并校验_RouteA_v3核心模型输出.py`: audit the core outputs.
6. `72_冻结_RouteA_v3核心预测.py`: freeze core five-seed predictions.
7. `73_运行_RouteA_v3_DASF硬门控.py`: run the exploratory DASF hard gate.
8. `74_运行_RouteA_v3_CB-SF软门控.py`: run the exploratory CB-SF target-batch soft gate.
9. `75_运行_RouteA_v3固定权重与recording级stacking基线.py`: fixed blends and LOSO recording-level stacking.
10. `76_运行_RouteA_v3_CrossTask_sourceOOF与recording级stacking.py`: source-OOF cross-task recording stacking.
11. `77_运行_RouteA_v3_Riemannian基线.py`: Riemannian MDM and tangent-space logistic regression.
12. `79_运行_RouteA_v3_EEGNet.py`: EEGNet for all frozen settings.
13. `80_运行_RouteA_v3_EEGConformer.py`: EEG-Conformer for all frozen settings.
14. `81_汇总并冻结_RouteA_v3全部14方法预测.py`: create the canonical 14-method recording-level freeze.

Files 66 and 78 are shared utilities and are not run directly. Files 64, 85 and 91 are frozen machine-readable protocols.

## Reporting and reviewer-requested audits

Run 82, 83, 84, 86, 87, 88, 89, 90, 92 and 93 in numerical order. They generate full classification/calibration metrics, participant bootstrap and BH statistics, sample/seed/gate stability, matched split auditing, label-mapping sensitivity, feature and algorithm definitions, baseline fairness, preprocessing sensitivity, and final tables/figures.

The included audited outputs let reviewers run `verify_release.py` and number 93 without repeating model training.
