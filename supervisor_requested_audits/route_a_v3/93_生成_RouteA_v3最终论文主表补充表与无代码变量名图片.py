"""编号93：从 Route A v3 冻结结果生成最终论文表格和图片。

本文件不训练模型、不修改预测，只读取81号主冻结预测及82--92号已通过审计的
结果。所有公开表图使用论文显示名称，不在图中出现内部代码变量名。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：20_final_reporting_v3
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FROZEN_ROOT = HERE / "10_all_methods_frozen"
METRICS_ROOT = HERE / "11_complete_metrics"
STATS_ROOT = HERE / "12_participant_statistics"
STABILITY_ROOT = HERE / "13_stability_audit"
SPLIT_ROOT = HERE / "14_three_split_audit"
LABEL_ROOT = HERE / "15_label_mapping_sensitivity"
PREPROCESS_ROOT = HERE / "19_preprocessing_sensitivity_v3"
OUTPUT_ROOT = HERE / "20_final_reporting_v3"

FROZEN_PREDICTION = FROZEN_ROOT / "81_冻结_RouteA_v3全部14方法_recording级预测.csv"
FROZEN_MANIFEST = FROZEN_ROOT / "81_冻结_RouteA_v3全部方法_manifest.json"
METRICS_MANIFEST = METRICS_ROOT / "82_完整指标_manifest.json"
STATS_MANIFEST = STATS_ROOT / "83_参与者Bootstrap与配对统计_manifest.json"
SPLIT_MANIFEST = SPLIT_ROOT / "86_三种数据划分审计_manifest.json"
PREPROCESS_MANIFEST = PREPROCESS_ROOT / "92_预处理敏感性_manifest.json"

INTERVAL_CSV = STATS_ROOT / "83_全部方法_参与者Bootstrap百分位CI.csv"
COMPARISON_CSV = STATS_ROOT / "83_65项相对AlwaysNeural配对比较与BH.csv"
PARTICIPANT_CSV = STATS_ROOT / "83_五类分析设置_参与者级完整指标.csv"
POOLED_METRICS_CSV = METRICS_ROOT / "82_总体recording级完整指标.csv"
CLASS_BALANCE_CSV = METRICS_ROOT / "82_类别数量与平衡.csv"
RECORDING_STRUCTURE_CSV = STABILITY_ROOT / "84_recording窗口数分布汇总.csv"
INNER_SPLIT_CSV = STABILITY_ROOT / "84_内层划分样本量汇总.csv"
SEED_STABILITY_CSV = STABILITY_ROOT / "84_五种子性能波动汇总.csv"
GATE_STABILITY_CSV = STABILITY_ROOT / "84_CB-SF权重稳定性汇总.csv"
SPLIT_SUMMARY_CSV = SPLIT_ROOT / "86_三种划分全部指标BootstrapCI.csv"
SPLIT_PAIRED_CSV = SPLIT_ROOT / "86_三种划分配对差值与BH.csv"
LABEL_SUMMARY_CSV = LABEL_ROOT / "87_标签映射敏感性全部指标BootstrapCI.csv"
PREPROCESS_SUMMARY_CSV = PREPROCESS_ROOT / "92_预处理敏感性总体Bootstrap汇总.csv"
PREPROCESS_PAIRED_CSV = PREPROCESS_ROOT / "92_相对参考流程配对差值与BH校正.csv"

TABLE1 = OUTPUT_ROOT / "93_Table1_Cohort_and_recording_structure.csv"
TABLE2 = OUTPUT_ROOT / "93_Table2_Primary_14_method_balanced_accuracy.csv"
TABLE2_NUMERIC = OUTPUT_ROOT / "93_Table2_numeric_long_format.csv"
TABLE3 = OUTPUT_ROOT / "93_Table3_DASF_and_CB-SF_vs_Always_neural.csv"
TABLE4 = OUTPUT_ROOT / "93_Table4_Three_split_leakage_audit.csv"
TABLE5 = OUTPUT_ROOT / "93_Table5_Preprocessing_sensitivity.csv"
TABLE_S1 = OUTPUT_ROOT / "93_TableS1_All_14_methods_five_settings.csv"
TABLE_S2 = OUTPUT_ROOT / "93_TableS2_All_recording_metrics.csv"
TABLE_S3 = OUTPUT_ROOT / "93_TableS3_All_65_paired_comparisons.csv"
TABLE_S4 = OUTPUT_ROOT / "93_TableS4_All_participant_level_results.csv"
TABLE_S5 = OUTPUT_ROOT / "93_TableS5_Class_balance.csv"
TABLE_S6 = OUTPUT_ROOT / "93_TableS6_Inner_split_and_seed_stability.csv"
TABLE_S7 = OUTPUT_ROOT / "93_TableS7_Label_mapping_sensitivity.csv"
FIGURE1 = OUTPUT_ROOT / "93_Figure1_Overlapping_window_split_audit.png"
FIGURE2 = OUTPUT_ROOT / "93_Figure2_Primary_14_method_results.png"
FIGURE_S1 = OUTPUT_ROOT / "93_FigureS1_Directional_cross_task_results.png"
CAPTIONS_MD = OUTPUT_ROOT / "93_Final_table_and_figure_captions.md"
REPORT_GUIDE_MD = OUTPUT_ROOT / "93_论文结果引用指南_防止数字混用.md"
MANIFEST_JSON = OUTPUT_ROOT / "93_最终表图_manifest.json"

METHOD_ORDER = [
    "Always neural",
    "Always fused",
    "Random forest",
    "Extra Trees",
    "Fixed blend (10%)",
    "Fixed blend (25%)",
    "Equal blend (50%)",
    "OOF recording-level stacking",
    "DASF",
    "CB-SF",
    "Riemannian TS-LR",
    "Riemannian MDM",
    "EEGNet",
    "EEG-Conformer",
]
PRIMARY_SETTINGS = ["Bidirectional cross-task", "Arithmetic LOSO", "Stroop LOSO"]
DIRECTION_SETTINGS = ["Arithmetic to Stroop", "Stroop to Arithmetic"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs() -> tuple[dict, dict, dict, dict, dict]:
    required = [
        FROZEN_PREDICTION, FROZEN_MANIFEST, METRICS_MANIFEST, STATS_MANIFEST,
        SPLIT_MANIFEST, PREPROCESS_MANIFEST, INTERVAL_CSV, COMPARISON_CSV,
        PARTICIPANT_CSV, POOLED_METRICS_CSV, CLASS_BALANCE_CSV,
        RECORDING_STRUCTURE_CSV, INNER_SPLIT_CSV, SEED_STABILITY_CSV,
        GATE_STABILITY_CSV, SPLIT_SUMMARY_CSV, SPLIT_PAIRED_CSV,
        LABEL_SUMMARY_CSV, PREPROCESS_SUMMARY_CSV, PREPROCESS_PAIRED_CSV,
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"93号缺少已完成审计输出：{missing}")
    frozen = load_json(FROZEN_MANIFEST)
    metrics = load_json(METRICS_MANIFEST)
    stats = load_json(STATS_MANIFEST)
    split = load_json(SPLIT_MANIFEST)
    preprocess = load_json(PREPROCESS_MANIFEST)
    expected_hash = "1c0fd57fda8042293f23303499638b0f76d17c38c6a0e346b86613b5aa03aea0"
    if sha256(FROZEN_PREDICTION) != expected_hash or frozen["prediction_sha256"] != expected_hash:
        raise AssertionError("93号读取的不是81号确认过的主冻结预测")
    for name, manifest in {
        "81 freeze": frozen,
        "82 metrics": metrics,
        "83 statistics": stats,
        "86 split audit": split,
        "92 preprocessing": preprocess,
    }.items():
        if manifest.get("status") != "passed":
            raise AssertionError(f"{name}没有通过校验")
    if metrics["input_prediction_sha256"] != expected_hash:
        raise AssertionError("82号指标不是从81号冻结预测生成")
    if stats["input_prediction_sha256"] != expected_hash:
        raise AssertionError("83号统计不是从81号冻结预测生成")
    if preprocess["parent_prediction_sha256"] != expected_hash:
        raise AssertionError("92号敏感性分析没有正确引用81号冻结")
    return frozen, metrics, stats, split, preprocess


def public_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"公开表缺少字段：{missing}")
    return frame[columns].copy()


def format_ci(mean: float, low: float, high: float) -> str:
    return f"{100.0 * float(mean):.2f} ({100.0 * float(low):.2f}-{100.0 * float(high):.2f})"


def build_primary_tables(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balanced = intervals[intervals["metric"] == "balanced_accuracy"].copy()
    balanced["method_display"] = pd.Categorical(
        balanced["method_display"], categories=METHOD_ORDER, ordered=True
    )
    balanced["setting_display"] = pd.Categorical(
        balanced["setting_display"], categories=[*DIRECTION_SETTINGS, *PRIMARY_SETTINGS], ordered=True
    )
    balanced = balanced.sort_values(["setting_display", "method_display"]).reset_index(drop=True)
    if len(balanced) != 70:
        raise AssertionError(f"五种设置应有14 x 5 = 70行balanced accuracy，实际{len(balanced)}")

    numeric = public_columns(
        balanced,
        [
            "setting_display", "method_display", "n_participants",
            "participant_macro_mean", "participant_standard_deviation", "participant_median",
            "ci95_percentile_low", "ci95_percentile_high",
        ],
    ).rename(
        columns={
            "setting_display": "Analysis setting",
            "method_display": "Method",
            "n_participants": "Participants",
            "participant_macro_mean": "Balanced accuracy",
            "participant_standard_deviation": "Participant SD",
            "participant_median": "Participant median",
            "ci95_percentile_low": "95% CI low",
            "ci95_percentile_high": "95% CI high",
        }
    )

    primary = balanced[balanced["setting_display"].astype(str).isin(PRIMARY_SETTINGS)].copy()
    primary["formatted"] = primary.apply(
        lambda row: format_ci(
            row["participant_macro_mean"], row["ci95_percentile_low"], row["ci95_percentile_high"]
        ),
        axis=1,
    )
    wide = primary.pivot(index="method_display", columns="setting_display", values="formatted").reset_index()
    wide = wide[["method_display", *PRIMARY_SETTINGS]]
    wide["method_display"] = pd.Categorical(wide["method_display"], categories=METHOD_ORDER, ordered=True)
    wide = wide.sort_values("method_display").rename(columns={"method_display": "Method"})
    wide.columns.name = None

    directional = balanced[balanced["setting_display"].astype(str).isin(DIRECTION_SETTINGS)].copy()
    return wide.reset_index(drop=True), numeric, directional


def build_gate_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    selected = comparisons[comparisons["method_display"].isin(["DASF", "CB-SF"])].copy()
    selected["setting_display"] = pd.Categorical(
        selected["setting_display"], categories=[*DIRECTION_SETTINGS, *PRIMARY_SETTINGS], ordered=True
    )
    selected["method_display"] = pd.Categorical(
        selected["method_display"], categories=["DASF", "CB-SF"], ordered=True
    )
    selected = selected.sort_values(["setting_display", "method_display"])
    if len(selected) != 10:
        raise AssertionError("DASF/CB-SF相对Always neural应有10项比较")
    output = public_columns(
        selected,
        [
            "setting_display", "method_display", "n_participants", "reference_mean", "method_mean",
            "mean_paired_difference_pp", "ci95_percentile_low_pp", "ci95_percentile_high_pp",
            "standardized_paired_effect_dz", "paired_rank_biserial", "wilcoxon_raw_p",
            "bh_adjusted_p_reporting_family", "bh_adjusted_p_single_family_65_sensitivity",
            "participants_method_better", "participants_equal", "participants_method_worse",
            "superiority_established",
        ],
    )
    return output.rename(
        columns={
            "setting_display": "Analysis setting",
            "method_display": "Method",
            "n_participants": "Participants",
            "reference_mean": "Always-neural mean",
            "method_mean": "Method mean",
            "mean_paired_difference_pp": "Paired difference (pp)",
            "ci95_percentile_low_pp": "Difference 95% CI low (pp)",
            "ci95_percentile_high_pp": "Difference 95% CI high (pp)",
            "standardized_paired_effect_dz": "Paired effect dz",
            "paired_rank_biserial": "Rank-biserial effect",
            "wilcoxon_raw_p": "Raw Wilcoxon p",
            "bh_adjusted_p_reporting_family": "Reporting-family BH p",
            "bh_adjusted_p_single_family_65_sensitivity": "All-65 BH p",
            "participants_method_better": "Participants better",
            "participants_equal": "Participants tied",
            "participants_method_worse": "Participants worse",
            "superiority_established": "Superiority established",
        }
    )


def build_structure_table(structure: pd.DataFrame) -> pd.DataFrame:
    scopes = [
        "cross_task_arithmetic_source", "cross_task_stroop_source",
        "cross_task_arithmetic_target", "cross_task_stroop_target",
        "arithmetic_loso", "stroop_loso",
    ]
    display = {
        "cross_task_arithmetic_source": "Cross-task Arithmetic source",
        "cross_task_stroop_source": "Cross-task Stroop source",
        "cross_task_arithmetic_target": "Cross-task Arithmetic target",
        "cross_task_stroop_target": "Cross-task Stroop target",
        "arithmetic_loso": "Arithmetic LOSO",
        "stroop_loso": "Stroop LOSO",
    }
    selected = structure[structure["scope"].isin(scopes)].copy()
    selected["Cohort"] = selected["scope"].map(display)
    return public_columns(
        selected,
        [
            "Cohort", "participants", "file_recordings", "unique_signal_hashes", "total_windows",
            "minimum_windows", "median_windows", "maximum_windows",
            "minimum_duration_seconds", "median_duration_seconds", "maximum_duration_seconds",
        ],
    ).rename(
        columns={
            "participants": "Participants",
            "file_recordings": "Recordings",
            "unique_signal_hashes": "Unique signal hashes",
            "total_windows": "Windows",
            "minimum_windows": "Minimum windows/recording",
            "median_windows": "Median windows/recording",
            "maximum_windows": "Maximum windows/recording",
            "minimum_duration_seconds": "Minimum duration (s)",
            "median_duration_seconds": "Median duration (s)",
            "maximum_duration_seconds": "Maximum duration (s)",
        }
    )


def build_split_table(split_summary: pd.DataFrame) -> pd.DataFrame:
    selected = split_summary[split_summary["metric"] == "recording_balanced_accuracy"].copy()
    task_display = {"arithmetic": "Arithmetic", "stroop": "Stroop"}
    selected["Task"] = selected["task"].map(task_display)
    selected["Result"] = selected.apply(
        lambda row: format_ci(
            row["participant_macro_mean"], row["ci95_percentile_low"], row["ci95_percentile_high"]
        ),
        axis=1,
    )
    order = ["Random-window split", "Recording-disjoint split", "Participant-disjoint split"]
    selected["strategy_display"] = pd.Categorical(selected["strategy_display"], categories=order, ordered=True)
    selected = selected.sort_values(["Task", "strategy_display"])
    return public_columns(selected, ["Task", "strategy_display", "n_participants", "Result"]).rename(
        columns={
            "strategy_display": "Split strategy",
            "n_participants": "Participants",
            "Result": "Balanced accuracy, % (95% CI)",
        }
    )


def build_preprocess_table(paired: pd.DataFrame) -> pd.DataFrame:
    selected = public_columns(
        paired,
        [
            "evaluation_display", "condition_display", "variant", "n_paired_participants",
            "mean_balanced_accuracy_difference", "paired_balanced_accuracy_ci95_low",
            "paired_balanced_accuracy_ci95_high", "wilcoxon_balanced_accuracy_p_raw",
            "wilcoxon_p_bh_24_comparison_family", "participants_improved", "participants_tied",
            "participants_worsened", "significant_after_bh_0_05",
        ],
    )
    for column in [
        "mean_balanced_accuracy_difference", "paired_balanced_accuracy_ci95_low",
        "paired_balanced_accuracy_ci95_high",
    ]:
        selected[column] = 100.0 * selected[column].astype(float)
    return selected.rename(
        columns={
            "evaluation_display": "Evaluation",
            "condition_display": "Condition",
            "variant": "Variant",
            "n_paired_participants": "Participants",
            "mean_balanced_accuracy_difference": "Balanced-accuracy difference (pp)",
            "paired_balanced_accuracy_ci95_low": "95% CI low (pp)",
            "paired_balanced_accuracy_ci95_high": "95% CI high (pp)",
            "wilcoxon_balanced_accuracy_p_raw": "Raw Wilcoxon p",
            "wilcoxon_p_bh_24_comparison_family": "BH-adjusted p",
            "participants_improved": "Participants improved",
            "participants_tied": "Participants tied",
            "participants_worsened": "Participants worsened",
            "significant_after_bh_0_05": "BH significant",
        }
    )


def public_all_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis_cell_display", "method_display", "n_participants", "n_recordings", "n_class_0", "n_class_1",
        "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "pr_auc",
        "average_precision", "log_loss", "brier_score", "ece_5_bins", "maximum_calibration_error_5_bins",
        "calibration_intercept", "calibration_slope", "calibration_fit_status",
    ]
    return public_columns(frame, columns).rename(
        columns={"analysis_cell_display": "Analysis setting", "method_display": "Method"}
    )


def public_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "setting_display", "comparison_family", "primary_metric", "reference_display", "method_display",
        "n_participants", "reference_mean", "method_mean", "mean_paired_difference_pp",
        "ci95_percentile_low_pp", "ci95_percentile_high_pp", "standardized_paired_effect_dz",
        "paired_rank_biserial", "participants_method_better", "participants_equal", "participants_method_worse",
        "wilcoxon_raw_p", "wilcoxon_status", "n_zero_differences", "n_nonzero_differences",
        "bh_adjusted_p_reporting_family", "bh_adjusted_p_single_family_65_sensitivity",
        "superiority_established",
    ]
    output = public_columns(frame, columns)
    output["comparison_family"] = output["comparison_family"].map(
        {
            "complete_settings_39_posthoc": "39 complete-setting comparisons",
            "direction_specific_26_posthoc": "26 direction-specific comparisons",
        }
    )
    return output.rename(
        columns={
            "setting_display": "Analysis setting",
            "comparison_family": "Comparison family",
            "primary_metric": "Primary metric",
            "reference_display": "Reference",
            "method_display": "Method",
        }
    )


def public_participants(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "setting_display", "subject", "method_display", "n_recordings", "n_class_0", "n_class_1",
        "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auroc",
        "pr_auc", "average_precision", "log_loss", "brier_score",
    ]
    return public_columns(frame, columns).rename(
        columns={"setting_display": "Analysis setting", "subject": "Participant", "method_display": "Method"}
    )


def public_class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis_cell_display", "n_participants", "n_recordings", "n_class_0", "n_class_1",
        "class_0_fraction", "class_1_fraction", "majority_class_accuracy",
    ]
    return public_columns(frame, columns).rename(
        columns={
            "analysis_cell_display": "Analysis setting",
            "n_participants": "Participants",
            "n_recordings": "Recordings",
            "n_class_0": "Class-0 recordings",
            "n_class_1": "Class-1 recordings",
            "class_0_fraction": "Class-0 fraction",
            "class_1_fraction": "Class-1 fraction",
            "majority_class_accuracy": "Majority-class accuracy",
        }
    )


def public_stability(inner: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    setting_map = {
        ("cross_task", "arithmetic_to_stroop"): "Arithmetic to Stroop",
        ("cross_task", "stroop_to_arithmetic"): "Stroop to Arithmetic",
        ("loso", "arithmetic"): "Arithmetic LOSO",
        ("loso", "stroop"): "Stroop LOSO",
    }
    inner_public = pd.DataFrame(
        {
            "Section": "Inner-split sample structure",
            "Analysis setting": [setting_map[(str(row.protocol), str(row.direction))] for row in inner.itertuples()],
            "Method": "All model-selection components",
            "Metric": "Sample structure",
            "Outer cells": inner["outer_cells"],
            "Seeds": inner["seed_scopes"],
            "Inner folds per outer seed, minimum": inner["inner_folds_per_outer_seed_min"],
            "Inner folds per outer seed, maximum": inner["inner_folds_per_outer_seed_max"],
            "Outer-training participants": inner["outer_training_participants"],
            "Outer-training recordings": inner["outer_training_recordings"],
            "Inner-training participants": inner["inner_training_participants"],
            "Inner-training recordings": inner["inner_training_recordings"],
            "Inner-validation participants": inner["inner_validation_participants"],
            "Inner-validation recordings": inner["inner_validation_recordings"],
            "Outer-test recordings": inner["outer_test_recordings"],
            "Minimum inner-training windows": inner["minimum_inner_training_windows"],
            "Median inner-training windows": inner["median_inner_training_windows"],
            "Maximum inner-training windows": inner["maximum_inner_training_windows"],
            "Seed mean": np.nan,
            "Seed standard deviation": np.nan,
            "Seed minimum": np.nan,
            "Seed maximum": np.nan,
            "Seed range": np.nan,
            "Interpretation": "Complete disjoint inner-fold structure; windows are not independent sample units.",
        }
    )
    metric_display = {
        "participant_macro_accuracy": "Participant-macro recording accuracy",
        "participant_macro_balanced_accuracy": "Participant-macro recording balanced accuracy",
    }
    seed_public = pd.DataFrame(
        {
            "Section": "Five-seed stability diagnostic",
            "Analysis setting": seeds["setting_display"],
            "Method": seeds["method_display"],
            "Metric": seeds["metric"].map(metric_display).fillna(seeds["metric"]),
            "Outer cells": np.nan,
            "Seeds": seeds["seeds"],
            "Inner folds per outer seed, minimum": np.nan,
            "Inner folds per outer seed, maximum": np.nan,
            "Outer-training participants": np.nan,
            "Outer-training recordings": np.nan,
            "Inner-training participants": np.nan,
            "Inner-training recordings": np.nan,
            "Inner-validation participants": np.nan,
            "Inner-validation recordings": np.nan,
            "Outer-test recordings": np.nan,
            "Minimum inner-training windows": np.nan,
            "Median inner-training windows": np.nan,
            "Maximum inner-training windows": np.nan,
            "Seed mean": seeds["seed_mean"],
            "Seed standard deviation": seeds["seed_standard_deviation"],
            "Seed minimum": seeds["seed_minimum"],
            "Seed maximum": seeds["seed_maximum"],
            "Seed range": seeds["seed_range"],
            "Interpretation": "Model-randomness diagnostic only; seeds are ensemble members.",
        }
    )
    return pd.concat([inner_public, seed_public], ignore_index=True)


def public_label_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    metric_display = {
        "accuracy": "Recording accuracy",
        "balanced_accuracy": "Recording balanced accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "f1": "F1 score",
        "auroc": "AUROC",
        "pr_auc": "PR-AUC",
        "log_loss": "Log loss",
        "brier_score": "Brier score",
    }
    output = public_columns(
        frame,
        [
            "task_display", "variant_display", "variant_role", "metric", "n_participants",
            "participant_macro_mean", "participant_standard_deviation", "ci95_percentile_low",
            "ci95_percentile_high",
        ],
    )
    output["metric"] = output["metric"].map(metric_display).fillna(output["metric"])
    return output.rename(
        columns={
            "task_display": "Task",
            "variant_display": "Label definition",
            "variant_role": "Analysis role",
            "metric": "Metric",
            "n_participants": "Participants",
            "participant_macro_mean": "Participant-macro mean",
            "participant_standard_deviation": "Participant SD",
            "ci95_percentile_low": "95% CI low",
            "ci95_percentile_high": "95% CI high",
        }
    )


def method_color(method: str) -> str:
    if method == "Always neural":
        return "#1f2937"
    if method == "DASF":
        return "#c2413b"
    if method == "CB-SF":
        return "#16836b"
    if method == "EEGNet":
        return "#3569a8"
    if method == "EEG-Conformer":
        return "#d58a26"
    return "#7a8491"


def style_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#d9dde3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def build_split_figure(split_summary: pd.DataFrame, split_paired: pd.DataFrame) -> None:
    selected = split_summary[split_summary["metric"] == "recording_balanced_accuracy"].copy()
    strategies = ["Random-window split", "Recording-disjoint split", "Participant-disjoint split"]
    colors = ["#c2413b", "#16836b", "#d58a26"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for axis, task, title in zip(axes, ["arithmetic", "stroop"], ["Arithmetic", "Stroop"]):
        cell = selected[selected["task"] == task].set_index("strategy_display").loc[strategies]
        means = 100.0 * cell["participant_macro_mean"].to_numpy(dtype=float)
        lows = 100.0 * cell["ci95_percentile_low"].to_numpy(dtype=float)
        highs = 100.0 * cell["ci95_percentile_high"].to_numpy(dtype=float)
        x = np.arange(len(strategies))
        for xi, value, low, high, color in zip(x, means, lows, highs, colors):
            axis.errorbar(
                [xi], [value], yerr=np.asarray([[value - low], [high - value]]), fmt="none",
                ecolor=color, elinewidth=2.0, capsize=4, capthick=1.4,
            )
            axis.scatter([xi], [value], s=80, color=color, edgecolors="white", linewidths=0.8, zorder=3)
            axis.text(xi, value + 3.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
        direct = split_paired[
            (split_paired["task"] == task)
            & (split_paired["left_strategy_display"] == "Random-window split")
            & (split_paired["right_strategy_display"] == "Recording-disjoint split")
        ].iloc[0]
        axis.text(
            0.5, 8.0,
            f"Random minus recording-disjoint: {float(direct['mean_paired_difference_pp']):.1f} pp\n"
            f"BH-adjusted p = {float(direct['bh_adjusted_p_six_comparison_family']):.4f}",
            ha="center", va="bottom", fontsize=9, color="#30343b",
        )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(["Random\nwindow", "Recording\ndisjoint", "Participant\ndisjoint"])
        axis.set_ylim(0, 108)
        style_axes(axis)
    axes[0].set_ylabel("Participant-macro recording balanced accuracy (%)")
    figure.tight_layout()
    figure.savefig(FIGURE1, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_method_figure(intervals: pd.DataFrame, settings: list[str], path: Path, figsize: tuple[float, float]) -> None:
    balanced = intervals[intervals["metric"] == "balanced_accuracy"].copy()
    figure, axes = plt.subplots(1, len(settings), figsize=figsize, sharey=True)
    if len(settings) == 1:
        axes = [axes]
    y = np.arange(len(METHOD_ORDER))[::-1]
    for panel, (axis, setting) in enumerate(zip(axes, settings)):
        cell = balanced[balanced["setting_display"] == setting].set_index("method_display").loc[METHOD_ORDER]
        means = 100.0 * cell["participant_macro_mean"].to_numpy(dtype=float)
        lows = 100.0 * cell["ci95_percentile_low"].to_numpy(dtype=float)
        highs = 100.0 * cell["ci95_percentile_high"].to_numpy(dtype=float)
        colors = [method_color(method) for method in METHOD_ORDER]
        for yi, mean, low, high, color in zip(y, means, lows, highs, colors):
            axis.plot([low, high], [yi, yi], color=color, linewidth=1.8)
            axis.scatter(mean, yi, s=34, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        axis.axvline(50.0, color="#8b929c", linestyle="--", linewidth=1.0)
        axis.set_xlim(30, 85)
        axis.set_title(setting, fontsize=11, fontweight="bold")
        axis.set_xlabel("Balanced accuracy (%)")
        axis.set_yticks(y)
        if panel == 0:
            axis.set_yticklabels(METHOD_ORDER)
        else:
            axis.tick_params(axis="y", labelleft=False)
        style_axes(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_captions() -> str:
    return """# Final table and figure captions

## Table 1

**Cohort and recording structure.** Window counts refer to 8.5-s windows advanced by 0.5 s. Cross-task target cells exclude the natural recording because the Arithmetic and Stroop natural files are exact within-participant duplicates. Stroop LOSO excludes S14 and S15 because their high-level files are byte-identical and provenance could not be resolved.

## Table 2

**Participant-macro recording-level balanced accuracy for all 14 methods in the three complete analysis settings.** Values are percentage means with two-sided 95% percentile participant-bootstrap intervals in parentheses. Five random seeds are ensemble members rather than statistical replicates. Descriptively highest values do not establish statistical superiority.

## Table 3

**Paired comparison of DASF and CB-SF with the prespecified Always-neural comparator.** Differences are method minus comparator in percentage points. Confidence intervals use 10,000 participant-cluster percentile resamples. Wilcoxon tests use Pratt zero handling and an approximate two-sided calculation because of zeros and ties. Both reporting-family and single-family 65-comparison Benjamini-Hochberg adjusted p values are provided. No row established superiority.

## Table 4

**Matched split audit using an identical fixed learner and hyperparameters.** Random-window splitting places highly overlapping windows from the same recording on both sides of the split. Recording-disjoint and participant-disjoint estimates use complete held-out groups.

## Table 5

**Post-hoc preprocessing sensitivity analysis.** Differences are relative to the reference preprocessing and use participant-level recording balanced accuracy. The 24 comparisons form one Benjamini-Hochberg family. These results cannot be used to select a replacement primary pipeline.

## Figure 1

**Effect of data partitioning on participant-macro recording balanced accuracy.** Points are participant-macro means and error bars are two-sided 95% percentile participant-bootstrap intervals. The annotated paired contrasts compare random-window with recording-disjoint splitting after averaging split repeats within each participant. The participant, rather than the window or split seed, is the inferential unit.

## Figure 2

**All-method results in the three complete analysis settings.** Points are participant-macro recording balanced accuracy and horizontal lines are two-sided 95% percentile participant-bootstrap intervals. The dashed vertical line marks 50% balanced accuracy. Colors identify the primary comparator, proposed exploratory gates, and raw-EEG deep baselines; gray points are other comparators. Numerical ordering is descriptive and does not imply superiority.

## Supplementary Figure S1

**Direction-specific cross-task results.** Points and intervals use participants as clusters. The two directions were added as post-hoc directional reporting and were adjusted separately from the three complete settings, with an additional all-65-family sensitivity adjustment.
"""


def build_report_guide(frozen_hash: str) -> str:
    return f"""# Route A v3 论文结果引用指南

## 唯一数字来源

最终论文所有主方法结果必须追溯到81号冻结预测，SHA-256为：

`{frozen_hash}`

不能再引用48号论文、旧route_a结果或旧53--57号输出中的任何准确率、区间、p值或图片。

## 正文推荐引用

1. 数据结构：Table 1。
2. 14方法三项完整设置：Table 2和Figure 2。
3. DASF/CB-SF相对Always neural：Table 3。
4. 窗口泄漏警示：Table 4和Figure 1，这是论文最强的阳性方法学结果。
5. 预处理敏感性：Table 5，仅放补充材料或正文限制段。
6. 两个cross-task方向：Table S1和Figure S1。
7. 全部65项比较：Table S3，不得只挑原始p值较小的结果。

## 必须使用的结果措辞

- “未能证明DASF或CB-SF优于Always neural。”
- “复杂融合方法在严格划分下没有表现出稳定优势。”
- “Random-window相对recording-disjoint高估约45--48个百分点。”
- “所有门控与预处理扩展均为post-hoc exploratory analyses。”

## 禁止措辞

- formal risk control
- risk guarantee
- safe deployment
- no-regret
- online prediction
- statistically equivalent
- universally effective fusion
"""


def main() -> None:
    print("[1/6] 核对81号冻结预测及82、83、86、92号结果血缘。")
    frozen, _, _, _, _ = verify_inputs()
    intervals = pd.read_csv(INTERVAL_CSV)
    comparisons = pd.read_csv(COMPARISON_CSV)
    participants = pd.read_csv(PARTICIPANT_CSV)
    pooled_metrics = pd.read_csv(POOLED_METRICS_CSV)
    class_balance = pd.read_csv(CLASS_BALANCE_CSV)
    structure = pd.read_csv(RECORDING_STRUCTURE_CSV)
    split_summary = pd.read_csv(SPLIT_SUMMARY_CSV)
    split_paired = pd.read_csv(SPLIT_PAIRED_CSV)
    preprocess_paired = pd.read_csv(PREPROCESS_PAIRED_CSV)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[2/6] 生成主文Table 1--5。")
    table1 = build_structure_table(structure)
    table2, table2_numeric, directional = build_primary_tables(intervals)
    table3 = build_gate_table(comparisons)
    table4 = build_split_table(split_summary)
    table5 = build_preprocess_table(preprocess_paired)
    table1.to_csv(TABLE1, index=False, lineterminator="\n")
    table2.to_csv(TABLE2, index=False, lineterminator="\n")
    table2_numeric.to_csv(TABLE2_NUMERIC, index=False, lineterminator="\n")
    table3.to_csv(TABLE3, index=False, lineterminator="\n")
    table4.to_csv(TABLE4, index=False, lineterminator="\n")
    table5.to_csv(TABLE5, index=False, lineterminator="\n")

    print("[3/6] 生成完整14方法、65项比较和参与者级补充表。")
    table2_numeric.to_csv(TABLE_S1, index=False, lineterminator="\n")
    public_all_metrics(pooled_metrics).to_csv(TABLE_S2, index=False, lineterminator="\n")
    public_comparisons(comparisons).to_csv(TABLE_S3, index=False, lineterminator="\n")
    public_participants(participants).to_csv(TABLE_S4, index=False, lineterminator="\n")
    public_class_balance(class_balance).to_csv(TABLE_S5, index=False, lineterminator="\n")
    inner = pd.read_csv(INNER_SPLIT_CSV)
    seeds = pd.read_csv(SEED_STABILITY_CSV)
    public_stability(inner, seeds).to_csv(TABLE_S6, index=False, lineterminator="\n")
    public_label_sensitivity(pd.read_csv(LABEL_SUMMARY_CSV)).to_csv(
        TABLE_S7, index=False, lineterminator="\n"
    )

    print("[4/6] 生成无内部代码变量名的主图和方向性补充图。")
    build_split_figure(split_summary, split_paired)
    build_method_figure(intervals, PRIMARY_SETTINGS, FIGURE2, (15.8, 7.2))
    build_method_figure(intervals, DIRECTION_SETTINGS, FIGURE_S1, (11.8, 7.2))

    print("[5/6] 生成最终图注、数字引用指南和SHA-256清单。")
    CAPTIONS_MD.write_text(build_captions(), encoding="utf-8")
    REPORT_GUIDE_MD.write_text(build_report_guide(frozen["prediction_sha256"]), encoding="utf-8")
    outputs = [
        TABLE1, TABLE2, TABLE2_NUMERIC, TABLE3, TABLE4, TABLE5,
        TABLE_S1, TABLE_S2, TABLE_S3, TABLE_S4, TABLE_S5, TABLE_S6, TABLE_S7,
        FIGURE1, FIGURE2, FIGURE_S1, CAPTIONS_MD, REPORT_GUIDE_MD,
    ]
    manifest = {
        "analysis_version": "route-a-v3-final-reporting-r1",
        "status": "passed",
        "frozen_prediction_sha256": frozen["prediction_sha256"],
        "single_source_rule": "All primary method results descend from the 81 frozen prediction file.",
        "main_tables": 5,
        "supplementary_tables": 7,
        "figures": 3,
        "method_count": 14,
        "paired_comparison_count": int(len(comparisons)),
        "bh_significant_method_comparisons": int(comparisons["bh_significant_single_family_65_0_05"].sum()),
        "source_sha256": {
            "81_frozen_predictions": sha256(FROZEN_PREDICTION),
            "83_method_intervals": sha256(INTERVAL_CSV),
            "83_method_comparisons": sha256(COMPARISON_CSV),
            "86_split_summary": sha256(SPLIT_SUMMARY_CSV),
            "86_split_comparisons": sha256(SPLIT_PAIRED_CSV),
            "92_preprocessing_comparisons": sha256(PREPROCESS_PAIRED_CSV),
            "reporting_runner": sha256(Path(__file__).resolve()),
        },
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "publication_position": "exploratory methodological audit; no algorithmic superiority or formal risk-control claim",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[6/6] Route A v3最终论文表图生成完成。")
    print(f"主表：5；补充表：7；图片：3；65项方法比较BH显著项：{manifest['bh_significant_method_comparisons']}")
    print(f"唯一主预测SHA-256：{frozen['prediction_sha256']}")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
