"""编号83：Route A v3 参与者级 bootstrap、配对统计与 BH 校正。

本脚本只读取 81 号冻结预测和 82 号指标 manifest，不训练模型。统计规则：
- recording 是评价单位，participant 是推断单位；
- Cross-task 总体分析先在同一 participant 内合并两个方向的 6 条 recording；
- 10,000 次 participant cluster bootstrap，报告 percentile 95% CI；
- 主比较指标为 participant-level balanced accuracy；
- Wilcoxon 使用 two-sided、Pratt zero handling、approx、无连续性校正；
- 39 个完整设置比较与 26 个方向性比较分别进行 BH 校正，同时提供
  全部 65 项放入单一 family 的敏感性校正；
- -2.5、-5.0、-7.5 pp 只作为 post-hoc 描述性界值，不作正式非劣效声明。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：12_participant_statistics
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from statsmodels.stats.power import TTestPower


HERE = Path(__file__).resolve().parent
FROZEN_PATH = (
    HERE
    / "10_all_methods_frozen"
    / "81_冻结_RouteA_v3全部14方法_recording级预测.csv"
)
FROZEN_MANIFEST_PATH = (
    HERE / "10_all_methods_frozen" / "81_冻结_RouteA_v3全部方法_manifest.json"
)
METRICS_MANIFEST_PATH = HERE / "11_complete_metrics" / "82_完整指标_manifest.json"
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
OUTPUT_ROOT = HERE / "12_participant_statistics"

BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_BASE_SEED = 20260826
PRIMARY_METRIC = "balanced_accuracy"
REFERENCE_METHOD = "always_nn"
EPSILON = 1e-12

METHODS = [
    "always_nn",
    "always_fuse",
    "rf",
    "extra_trees",
    "fixed_blend_010",
    "fixed_blend_025",
    "equal_blend_050",
    "stack_recording",
    "dasf_clean",
    "cbsf",
    "riemann_ts_logreg",
    "riemann_mdm",
    "eegnet",
    "eeg_conformer",
]

METHOD_DISPLAY = {
    "always_nn": "Always neural",
    "always_fuse": "Always fused",
    "rf": "Random forest",
    "extra_trees": "Extra Trees",
    "fixed_blend_010": "Fixed blend (10%)",
    "fixed_blend_025": "Fixed blend (25%)",
    "equal_blend_050": "Equal blend (50%)",
    "stack_recording": "OOF recording-level stacking",
    "dasf_clean": "DASF",
    "cbsf": "CB-SF",
    "riemann_ts_logreg": "Riemannian TS-LR",
    "riemann_mdm": "Riemannian MDM",
    "eegnet": "EEGNet",
    "eeg_conformer": "EEG-Conformer",
}

SETTING_SPECS = [
    {
        "setting": "cross_task_bidirectional",
        "setting_display": "Bidirectional cross-task",
        "comparison_family": "complete_settings_39_posthoc",
        "protocol": "cross_task",
        "direction": None,
        "expected_participants": 13,
        "expected_recordings_per_participant": 6,
    },
    {
        "setting": "loso_arithmetic",
        "setting_display": "Arithmetic LOSO",
        "comparison_family": "complete_settings_39_posthoc",
        "protocol": "loso",
        "direction": "arithmetic",
        "expected_participants": 15,
        "expected_recordings_per_participant": 4,
    },
    {
        "setting": "loso_stroop",
        "setting_display": "Stroop LOSO",
        "comparison_family": "complete_settings_39_posthoc",
        "protocol": "loso",
        "direction": "stroop",
        "expected_participants": 13,
        "expected_recordings_per_participant": 4,
    },
    {
        "setting": "cross_task_arithmetic_to_stroop",
        "setting_display": "Arithmetic to Stroop",
        "comparison_family": "direction_specific_26_posthoc",
        "protocol": "cross_task",
        "direction": "arithmetic_to_stroop",
        "expected_participants": 13,
        "expected_recordings_per_participant": 3,
    },
    {
        "setting": "cross_task_stroop_to_arithmetic",
        "setting_display": "Stroop to Arithmetic",
        "comparison_family": "direction_specific_26_posthoc",
        "protocol": "cross_task",
        "direction": "stroop_to_arithmetic",
        "expected_participants": 13,
        "expected_recordings_per_participant": 3,
    },
]

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auroc",
    "pr_auc",
    "average_precision",
    "log_loss",
    "brier_score",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in (BOOTSTRAP_BASE_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def calculate_metrics(group: pd.DataFrame) -> dict:
    y_true = group["true_label"].to_numpy(dtype=int)
    y_pred = group["pred_label"].to_numpy(dtype=int)
    p1_raw = group["p1"].to_numpy(dtype=float)
    p1 = np.clip(p1_raw, EPSILON, 1.0 - EPSILON)

    negative = y_true == 0
    positive = y_true == 1
    tn = int(np.sum(negative & (y_pred == 0)))
    fp = int(np.sum(negative & (y_pred == 1)))
    fn = int(np.sum(positive & (y_pred == 0)))
    tp = int(np.sum(positive & (y_pred == 1)))
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)

    if np.unique(y_true).size == 2:
        auroc = float(roc_auc_score(y_true, p1))
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, p1)
        pr_auc = float(auc(pr_recall, pr_precision))
        average_precision = float(average_precision_score(y_true, p1))
    else:
        auroc = float("nan")
        pr_auc = float("nan")
        average_precision = float("nan")

    probability_matrix = np.column_stack([1.0 - p1, p1])
    selected_probability = probability_matrix[np.arange(len(y_true)), y_true]
    return {
        "n_recordings": int(len(group)),
        "n_class_0": int(np.sum(y_true == 0)),
        "n_class_1": int(np.sum(y_true == 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "log_loss": float(-np.mean(np.log(selected_probability))),
        "brier_score": float(np.mean(np.square(p1_raw - y_true))),
    }


def validate_inputs() -> tuple[pd.DataFrame, dict, dict, dict]:
    required = [FROZEN_PATH, FROZEN_MANIFEST_PATH, METRICS_MANIFEST_PATH, POLICY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少输入文件，请先完成81和82号：{missing}")

    frozen_manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    metrics_manifest = json.loads(METRICS_MANIFEST_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    observed_hash = sha256(FROZEN_PATH)
    if observed_hash != frozen_manifest.get("prediction_sha256"):
        raise ValueError("81号预测文件与81号manifest的SHA-256不一致")
    if observed_hash != metrics_manifest.get("input_prediction_sha256"):
        raise ValueError("82号指标不是从当前81号冻结预测生成")
    if frozen_manifest.get("status") != "passed" or metrics_manifest.get("status") != "passed":
        raise ValueError("81或82号manifest未通过")

    predictions = pd.read_csv(FROZEN_PATH)
    if len(predictions) != 2660 or set(predictions["method"].unique()) != set(METHODS):
        raise ValueError("81号冻结文件不是14方法 × 190 recording")
    return predictions, frozen_manifest, metrics_manifest, policy


def build_participant_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in SETTING_SPECS:
        selected = predictions[predictions["protocol"] == spec["protocol"]].copy()
        if spec["direction"] is not None:
            selected = selected[selected["direction"] == spec["direction"]].copy()

        observed_participants = int(selected["subject"].nunique())
        if observed_participants != spec["expected_participants"]:
            raise AssertionError(
                f"{spec['setting']} 应有 {spec['expected_participants']} 名参与者，实际 {observed_participants}"
            )
        for (subject, method), group in selected.groupby(["subject", "method"], sort=True):
            if len(group) != spec["expected_recordings_per_participant"]:
                raise AssertionError(
                    f"{spec['setting']} S{int(subject):02d} {method} 应有 "
                    f"{spec['expected_recordings_per_participant']} 条recording，实际{len(group)}"
                )
            if group["true_label"].nunique() != 2:
                raise AssertionError(f"{spec['setting']} S{int(subject):02d} 缺少一个类别")
            rows.append(
                {
                    "dataset": str(group["dataset"].iloc[0]),
                    "setting": spec["setting"],
                    "setting_display": spec["setting_display"],
                    "comparison_family": spec["comparison_family"],
                    "subject": int(subject),
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    **calculate_metrics(group),
                }
            )
    result = pd.DataFrame(rows)
    expected_rows = sum(spec["expected_participants"] for spec in SETTING_SPECS) * len(METHODS)
    if len(result) != expected_rows:
        raise AssertionError(f"参与者分析单元应有{expected_rows}行，实际{len(result)}行")
    return result


def bootstrap_distribution(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Participant bootstrap要求至少两个有限的一维参与者值")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    return values[indices].mean(axis=1)


def percentile_interval(distribution: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(distribution, [0.025, 0.975], method="linear")
    return float(low), float(high)


def build_all_metric_intervals(participant: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (setting, method), group in participant.groupby(["setting", "method"], sort=True):
        metadata = group.iloc[0]
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            distribution = bootstrap_distribution(values, stable_seed("metric", setting, method, metric))
            low, high = percentile_interval(distribution)
            rows.append(
                {
                    "dataset": metadata["dataset"],
                    "setting": setting,
                    "setting_display": metadata["setting_display"],
                    "comparison_family": metadata["comparison_family"],
                    "method": method,
                    "method_display": metadata["method_display"],
                    "metric": metric,
                    "higher_is_better": metric not in {"log_loss", "brier_score"},
                    "n_participants": int(len(values)),
                    "participant_macro_mean": float(np.mean(values)),
                    "participant_standard_deviation": float(np.std(values, ddof=1)),
                    "participant_median": float(np.median(values)),
                    "ci95_percentile_low": low,
                    "ci95_percentile_high": high,
                    "ci95_width": high - low,
                }
            )
    return pd.DataFrame(rows)


def paired_rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    denominator = positive + negative
    if denominator == 0:
        return 0.0
    return (positive - negative) / denominator


def wilcoxon_pratt(differences: np.ndarray) -> tuple[float, float, str]:
    differences = np.asarray(differences, dtype=float)
    if np.allclose(differences, 0.0, atol=1e-15):
        return 0.0, 1.0, "all_zero_p_assigned_1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = wilcoxon(
            differences,
            zero_method="pratt",
            correction=False,
            alternative="two-sided",
            method="approx",
        )
    return float(result.statistic), float(result.pvalue), "approx_pratt_no_correction"


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if not np.isfinite(p_values).all():
        raise ValueError("BH输入含非有限p值")
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted_ranked = ranked * count / np.arange(1, count + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def build_primary_comparisons(
    participant: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict] = []
    difference_lookup: dict[tuple[str, str], np.ndarray] = {}
    for setting, setting_group in participant.groupby("setting", sort=True):
        reference = (
            setting_group[setting_group["method"] == REFERENCE_METHOD]
            .set_index("subject")[PRIMARY_METRIC]
            .sort_index()
        )
        metadata = setting_group.iloc[0]
        for method in METHODS:
            if method == REFERENCE_METHOD:
                continue
            current = (
                setting_group[setting_group["method"] == method]
                .set_index("subject")[PRIMARY_METRIC]
                .sort_index()
            )
            paired = pd.concat([reference.rename("reference"), current.rename("method")], axis=1, join="inner")
            expected_participants = next(
                int(spec["expected_participants"])
                for spec in SETTING_SPECS
                if spec["setting"] == setting
            )
            if len(paired) != expected_participants:
                raise AssertionError(f"{setting}/{method}配对参与者数量异常")
            differences = paired["method"].to_numpy(dtype=float) - paired["reference"].to_numpy(dtype=float)
            difference_lookup[(setting, method)] = differences
            distribution = bootstrap_distribution(differences, stable_seed("difference", setting, method))
            low, high = percentile_interval(distribution)
            statistic, raw_p, wilcoxon_status = wilcoxon_pratt(differences)
            difference_sd = float(np.std(differences, ddof=1))
            mean_difference = float(np.mean(differences))
            rows.append(
                {
                    "dataset": metadata["dataset"],
                    "setting": setting,
                    "setting_display": metadata["setting_display"],
                    "comparison_family": metadata["comparison_family"],
                    "primary_metric": PRIMARY_METRIC,
                    "reference_method": REFERENCE_METHOD,
                    "reference_display": METHOD_DISPLAY[REFERENCE_METHOD],
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "n_participants": int(len(differences)),
                    "reference_mean": float(paired["reference"].mean()),
                    "method_mean": float(paired["method"].mean()),
                    "mean_paired_difference": mean_difference,
                    "mean_paired_difference_pp": 100.0 * mean_difference,
                    "median_paired_difference": float(np.median(differences)),
                    "paired_difference_sd": difference_sd,
                    "standardized_paired_effect_dz": (
                        mean_difference / difference_sd if difference_sd > 0 else float("nan")
                    ),
                    "paired_rank_biserial": paired_rank_biserial(differences),
                    "participants_method_better": int(np.sum(differences > 0)),
                    "participants_equal": int(np.sum(np.isclose(differences, 0.0, atol=1e-15))),
                    "participants_method_worse": int(np.sum(differences < 0)),
                    "common_language_superiority": float(
                        (np.sum(differences > 0) + 0.5 * np.sum(np.isclose(differences, 0.0, atol=1e-15)))
                        / len(differences)
                    ),
                    "ci95_percentile_low": low,
                    "ci95_percentile_high": high,
                    "ci95_percentile_low_pp": 100.0 * low,
                    "ci95_percentile_high_pp": 100.0 * high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_raw_p": raw_p,
                    "wilcoxon_status": wilcoxon_status,
                    "wilcoxon_zero_method": "pratt",
                    "wilcoxon_calculation": "two-sided approximate, no continuity correction",
                    "n_zero_differences": int(np.sum(np.isclose(differences, 0.0, atol=1e-15))),
                    "n_nonzero_differences": int(np.sum(~np.isclose(differences, 0.0, atol=1e-15))),
                }
            )

    comparisons = pd.DataFrame(rows)
    if len(comparisons) != 65:
        raise AssertionError(f"应生成65项比较，实际{len(comparisons)}项")
    comparisons["bh_adjusted_p_reporting_family"] = np.nan
    for family, indices in comparisons.groupby("comparison_family").groups.items():
        expected = 39 if family == "complete_settings_39_posthoc" else 26
        if len(indices) != expected:
            raise AssertionError(f"{family}应有{expected}项比较，实际{len(indices)}项")
        comparisons.loc[indices, "bh_adjusted_p_reporting_family"] = benjamini_hochberg(
            comparisons.loc[indices, "wilcoxon_raw_p"].to_numpy(dtype=float)
        )
    comparisons["bh_adjusted_p_single_family_65_sensitivity"] = benjamini_hochberg(
        comparisons["wilcoxon_raw_p"].to_numpy(dtype=float)
    )
    comparisons["bh_significant_reporting_family_0_05"] = (
        comparisons["bh_adjusted_p_reporting_family"] < 0.05
    )
    comparisons["bh_significant_single_family_65_0_05"] = (
        comparisons["bh_adjusted_p_single_family_65_sensitivity"] < 0.05
    )
    comparisons["superiority_established"] = (
        (comparisons["ci95_percentile_low"] > 0.0)
        & comparisons["bh_significant_reporting_family_0_05"]
    )
    return comparisons, difference_lookup


def build_family_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in comparisons.groupby("comparison_family", sort=True):
        rows.append(
            {
                "comparison_family": family,
                "family_definition": (
                    "3 complete settings x 13 methods"
                    if family == "complete_settings_39_posthoc"
                    else "2 direction-specific cross-task settings x 13 methods"
                ),
                "family_status": "post-hoc exploratory reporting family; not preregistered",
                "comparisons": int(len(group)),
                "raw_p_below_0_05": int((group["wilcoxon_raw_p"] < 0.05).sum()),
                "bh_adjusted_p_below_0_05": int(
                    (group["bh_adjusted_p_reporting_family"] < 0.05).sum()
                ),
                "minimum_raw_p": float(group["wilcoxon_raw_p"].min()),
                "minimum_bh_adjusted_p": float(group["bh_adjusted_p_reporting_family"].min()),
            }
        )
    rows.append(
        {
            "comparison_family": "single_family_65_sensitivity",
            "family_definition": "all 5 settings x 13 methods",
            "family_status": "sensitivity correction across all primary comparisons",
            "comparisons": 65,
            "raw_p_below_0_05": int((comparisons["wilcoxon_raw_p"] < 0.05).sum()),
            "bh_adjusted_p_below_0_05": int(
                (comparisons["bh_adjusted_p_single_family_65_sensitivity"] < 0.05).sum()
            ),
            "minimum_raw_p": float(comparisons["wilcoxon_raw_p"].min()),
            "minimum_bh_adjusted_p": float(
                comparisons["bh_adjusted_p_single_family_65_sensitivity"].min()
            ),
        }
    )
    return pd.DataFrame(rows)


def build_margin_sensitivity(
    comparisons: pd.DataFrame,
    difference_lookup: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict] = []
    selected = comparisons[comparisons["method"].isin(["dasf_clean", "cbsf"])]
    for _, comparison in selected.iterrows():
        differences = difference_lookup[(comparison["setting"], comparison["method"])]
        for margin in [0.025, 0.05, 0.075]:
            rows.append(
                {
                    "setting": comparison["setting"],
                    "setting_display": comparison["setting_display"],
                    "method": comparison["method"],
                    "method_display": comparison["method_display"],
                    "reference_method": REFERENCE_METHOD,
                    "n_participants": int(len(differences)),
                    "exploratory_margin": margin,
                    "exploratory_margin_pp": 100.0 * margin,
                    "margin_status": "post-hoc sensitivity parameter; not externally validated",
                    "mean_paired_difference": float(np.mean(differences)),
                    "ci95_percentile_low": float(comparison["ci95_percentile_low"]),
                    "ci95_percentile_high": float(comparison["ci95_percentile_high"]),
                    "lower_ci_above_negative_margin": bool(
                        comparison["ci95_percentile_low"] > -margin
                    ),
                    "participants_below_negative_margin": int(np.sum(differences < -margin)),
                    "fraction_participants_below_negative_margin": float(
                        np.mean(differences < -margin)
                    ),
                    "allowed_interpretation": (
                        "descriptive compatibility check only; not a formal non-inferiority test or guarantee"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 30:
        raise AssertionError(f"界值敏感性应有30行，实际{len(result)}行")
    return result


def minimum_detectable_standardized_effect(n_participants: int, alpha: float) -> float:
    return float(
        TTestPower().solve_power(
            effect_size=None,
            nobs=n_participants,
            alpha=alpha,
            power=0.80,
            alternative="two-sided",
        )
    )


def build_sample_size_sensitivity() -> pd.DataFrame:
    scenarios = [
        ("nominal_two_sided", 0.05, 13),
        ("nominal_two_sided", 0.05, 15),
        ("bonferroni_bound_complete_39", 0.05 / 39.0, 13),
        ("bonferroni_bound_complete_39", 0.05 / 39.0, 15),
        ("bonferroni_bound_directional_26", 0.05 / 26.0, 13),
        ("bonferroni_bound_all_65", 0.05 / 65.0, 13),
        ("bonferroni_bound_all_65", 0.05 / 65.0, 15),
    ]
    rows = []
    for scenario, alpha, n_participants in scenarios:
        rows.append(
            {
                "scenario": scenario,
                "n_participants": n_participants,
                "two_sided_alpha": alpha,
                "target_power": 0.80,
                "minimum_detectable_standardized_paired_effect_dz": (
                    minimum_detectable_standardized_effect(n_participants, alpha)
                ),
                "analysis_type": "paired-t approximation for design sensitivity",
                "limitation": (
                    "not observed post-hoc power and not exact Wilcoxon/BH power; "
                    "Bonferroni rows are conservative bounds because BH thresholds are adaptive"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    predictions, frozen_manifest, metrics_manifest, policy = validate_inputs()
    participant = build_participant_analysis(predictions)
    metric_intervals = build_all_metric_intervals(participant)
    comparisons, difference_lookup = build_primary_comparisons(participant)
    family_summary = build_family_summary(comparisons)
    margin_sensitivity = build_margin_sensitivity(comparisons, difference_lookup)
    sample_size_sensitivity = build_sample_size_sensitivity()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "participant_analysis": OUTPUT_ROOT / "83_五类分析设置_参与者级完整指标.csv",
        "all_metric_intervals": OUTPUT_ROOT / "83_全部方法_参与者Bootstrap百分位CI.csv",
        "primary_comparisons": OUTPUT_ROOT / "83_65项相对AlwaysNeural配对比较与BH.csv",
        "family_summary": OUTPUT_ROOT / "83_BH比较家族汇总.csv",
        "margin_sensitivity": OUTPUT_ROOT / "83_CB-SF与DASF_非劣界值敏感性_仅探索性.csv",
        "sample_size_sensitivity": OUTPUT_ROOT / "83_样本量与检验敏感性.csv",
    }
    participant.to_csv(outputs["participant_analysis"], index=False, lineterminator="\n", float_format="%.17g")
    metric_intervals.to_csv(outputs["all_metric_intervals"], index=False, lineterminator="\n", float_format="%.17g")
    comparisons.to_csv(outputs["primary_comparisons"], index=False, lineterminator="\n", float_format="%.17g")
    family_summary.to_csv(outputs["family_summary"], index=False, lineterminator="\n", float_format="%.17g")
    margin_sensitivity.to_csv(outputs["margin_sensitivity"], index=False, lineterminator="\n", float_format="%.17g")
    sample_size_sensitivity.to_csv(outputs["sample_size_sensitivity"], index=False, lineterminator="\n", float_format="%.17g")

    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "analysis_role": policy["analysis_role"],
        "input_prediction_sha256": frozen_manifest["prediction_sha256"],
        "input_metrics_manifest_sha256": sha256(METRICS_MANIFEST_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "evaluation_unit": "recording",
        "inferential_unit": "participant",
        "primary_metric": PRIMARY_METRIC,
        "reference_method": REFERENCE_METHOD,
        "cross_task_overall_unit": (
            "one participant cluster containing both directions and six target recordings"
        ),
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "resampling_unit": "participant as a whole cluster",
            "interval": "two-sided 95% percentile interval",
            "quantile_interpolation": "linear",
        },
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "Pratt",
            "method": "approximate because zeros/ties preclude an exact signed-rank calculation",
            "continuity_correction": False,
            "all_zero_rule": "statistic=0 and p=1",
        },
        "multiple_comparisons": {
            "complete_settings": "39 comparisons: 3 settings x 13 methods",
            "direction_specific": "26 comparisons: 2 directions x 13 methods",
            "family_status": "post-hoc exploratory reporting families; not preregistered",
            "sensitivity": "all 65 comparisons additionally adjusted as one BH family",
        },
        "noninferiority": {
            "status": "no formal non-inferiority test or claim",
            "margins": [0.025, 0.05, 0.075],
            "role": "post-hoc descriptive sensitivity only",
        },
        "power_reporting": (
            "minimum detectable standardized paired effect at 80% power; "
            "paired-t approximation, not observed post-hoc power"
        ),
        "participant_analysis_rows": int(len(participant)),
        "metric_interval_rows": int(len(metric_intervals)),
        "primary_comparison_rows": int(len(comparisons)),
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "old_route_a_results_read": False,
    }
    manifest_path = OUTPUT_ROOT / "83_参与者Bootstrap与配对统计_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected = comparisons[comparisons["method"].isin(["dasf_clean", "cbsf"])][
        [
            "setting_display",
            "method_display",
            "n_participants",
            "reference_mean",
            "method_mean",
            "mean_paired_difference_pp",
            "ci95_percentile_low_pp",
            "ci95_percentile_high_pp",
            "wilcoxon_raw_p",
            "bh_adjusted_p_reporting_family",
        ]
    ].copy()
    for column in [
        "reference_mean",
        "method_mean",
        "mean_paired_difference_pp",
        "ci95_percentile_low_pp",
        "ci95_percentile_high_pp",
        "wilcoxon_raw_p",
        "bh_adjusted_p_reporting_family",
    ]:
        selected[column] = selected[column].round(4)

    print("Route A v3 参与者级统计已生成")
    print(f"输入预测 SHA-256：{frozen_manifest['prediction_sha256']}")
    print(f"Bootstrap：{BOOTSTRAP_REPEATS}次 participant cluster percentile 95% CI")
    print(f"完整设置比较：{int((comparisons['comparison_family'] == 'complete_settings_39_posthoc').sum())}")
    print(f"方向性比较：{int((comparisons['comparison_family'] == 'direction_specific_26_posthoc').sum())}")
    print(
        "分开BH校正后显著项："
        f"{int(comparisons['bh_significant_reporting_family_0_05'].sum())}"
    )
    print(
        "65项单一BH family校正后显著项："
        f"{int(comparisons['bh_significant_single_family_65_0_05'].sum())}"
    )
    print(selected.to_string(index=False))
    print("校验状态：passed")


if __name__ == "__main__":
    main()
