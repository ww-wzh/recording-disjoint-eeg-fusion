"""编号82：从 81 号唯一冻结文件生成完整分类、概率与校准指标。

本脚本不训练模型，也不读取旧 Route A 结果。它输出：
1. 四个实验单元、14 个方法的 pooled recording-level 描述性指标；
2. 每名参与者的 recording-level 指标，供 83 号参与者级统计使用；
3. 参与者指标的描述性汇总；
4. 固定五分箱的概率校准数据；
5. 每个实验单元的类别数量与类别平衡；
6. 指标定义、文件哈希和运行环境 manifest。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输入目录：10_all_methods_frozen
输出目录：11_complete_metrics
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE / "10_all_methods_frozen"
INPUT_PATH = INPUT_ROOT / "81_冻结_RouteA_v3全部14方法_recording级预测.csv"
INPUT_MANIFEST_PATH = INPUT_ROOT / "81_冻结_RouteA_v3全部方法_manifest.json"
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
OUTPUT_ROOT = HERE / "11_complete_metrics"

EPSILON = 1e-12
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 6)
GROUP_KEYS = ["dataset", "protocol", "direction", "method"]
PARTICIPANT_KEYS = ["dataset", "protocol", "direction", "subject", "method"]

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

CELL_DISPLAY = {
    ("cross_task", "arithmetic_to_stroop"): "Arithmetic to Stroop",
    ("cross_task", "stroop_to_arithmetic"): "Stroop to Arithmetic",
    ("loso", "arithmetic"): "Arithmetic LOSO",
    ("loso", "stroop"): "Stroop LOSO",
}

METRIC_COLUMNS = [
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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    negative = y_true == 0
    positive = y_true == 1
    tn = int(np.sum(negative & (y_pred == 0)))
    fp = int(np.sum(negative & (y_pred == 1)))
    fn = int(np.sum(positive & (y_pred == 0)))
    tp = int(np.sum(positive & (y_pred == 1)))
    return tn, fp, fn, tp


def calibration_bins(y_true: np.ndarray, p1: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    bin_index = np.searchsorted(CALIBRATION_EDGES, p1, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(CALIBRATION_EDGES) - 2)
    rows: list[dict] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    total = len(y_true)
    for index in range(len(CALIBRATION_EDGES) - 1):
        selected = bin_index == index
        count = int(np.sum(selected))
        lower = float(CALIBRATION_EDGES[index])
        upper = float(CALIBRATION_EDGES[index + 1])
        if count:
            mean_probability = float(np.mean(p1[selected]))
            observed_rate = float(np.mean(y_true[selected]))
            gap = abs(mean_probability - observed_rate)
            weighted_gap += count / total * gap
            maximum_gap = max(maximum_gap, gap)
        else:
            mean_probability = float("nan")
            observed_rate = float("nan")
            gap = float("nan")
        rows.append(
            {
                "bin_index": index + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "right_edge_included": bool(index == len(CALIBRATION_EDGES) - 2),
                "n_recordings": count,
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_gap": gap,
            }
        )
    return pd.DataFrame(rows), float(weighted_gap), float(maximum_gap)


def calibration_intercept_slope(y_true: np.ndarray, p1: np.ndarray) -> tuple[float, float, str]:
    clipped = np.clip(p1, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    if np.unique(y_true).size != 2:
        return float("nan"), float("nan"), "one_class"
    if float(np.std(logits)) < 1e-12:
        return float("nan"), float("nan"), "constant_probability"
    try:
        # C=1e6 gives a practically unpenalized descriptive calibration fit.
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=10000,
            fit_intercept=True,
        )
        model.fit(logits.reshape(-1, 1), y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0]), "estimated"
    except Exception as error:  # pragma: no cover - retained as an auditable status
        return float("nan"), float("nan"), f"failed:{type(error).__name__}"


def calculate_metrics(group: pd.DataFrame, include_calibration: bool) -> tuple[dict, pd.DataFrame | None]:
    y_true = group["true_label"].to_numpy(dtype=int)
    y_pred = group["pred_label"].to_numpy(dtype=int)
    p1_raw = group["p1"].to_numpy(dtype=float)
    p1 = np.clip(p1_raw, EPSILON, 1.0 - EPSILON)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    balanced = float(np.nanmean([sensitivity, specificity]))
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)

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
    natural_log_loss = float(-np.mean(np.log(selected_probability)))
    brier = float(np.mean(np.square(p1_raw - y_true)))

    metrics = {
        "n_recordings": int(len(group)),
        "n_class_0": int(np.sum(y_true == 0)),
        "n_class_1": int(np.sum(y_true == 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "log_loss": natural_log_loss,
        "brier_score": brier,
    }

    if not include_calibration:
        return metrics, None

    bins, ece, mce = calibration_bins(y_true, p1_raw)
    intercept, slope, status = calibration_intercept_slope(y_true, p1_raw)
    metrics.update(
        {
            "ece_5_bins": ece,
            "maximum_calibration_error_5_bins": mce,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "calibration_fit_status": status,
        }
    )
    return metrics, bins


def attach_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "method" in output.columns:
        output.insert(
            output.columns.get_loc("method") + 1,
            "method_display",
            output["method"].map(METHOD_DISPLAY),
        )
    output.insert(
        output.columns.get_loc("direction") + 1,
        "analysis_cell_display",
        [CELL_DISPLAY[(protocol, direction)] for protocol, direction in zip(output["protocol"], output["direction"])],
    )
    return output


def build_pooled_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    calibration_rows: list[pd.DataFrame] = []
    for keys, group in predictions.groupby(GROUP_KEYS, sort=True):
        metadata = dict(zip(GROUP_KEYS, keys))
        metrics, bins = calculate_metrics(group, include_calibration=True)
        metric_rows.append({**metadata, "n_participants": int(group["subject"].nunique()), **metrics})
        assert bins is not None
        for key, value in metadata.items():
            bins[key] = value
        calibration_rows.append(bins)
    pooled = attach_display_columns(pd.DataFrame(metric_rows))
    calibration = attach_display_columns(pd.concat(calibration_rows, ignore_index=True))
    return pooled, calibration


def build_participant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in predictions.groupby(PARTICIPANT_KEYS, sort=True):
        metadata = dict(zip(PARTICIPANT_KEYS, keys))
        metrics, _ = calculate_metrics(group, include_calibration=False)
        rows.append({**metadata, **metrics})
    return attach_display_columns(pd.DataFrame(rows))


def summarise_participants(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in participant_metrics.groupby(GROUP_KEYS, sort=True):
        metadata = dict(zip(GROUP_KEYS, keys))
        for metric in METRIC_COLUMNS:
            values = group[metric].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "n_participants_total": int(group["subject"].nunique()),
                    "n_participants_valid": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "q1": float(np.quantile(values, 0.25)) if values.size else float("nan"),
                    "q3": float(np.quantile(values, 0.75)) if values.size else float("nan"),
                    "minimum": float(np.min(values)) if values.size else float("nan"),
                    "maximum": float(np.max(values)) if values.size else float("nan"),
                }
            )
    return attach_display_columns(pd.DataFrame(rows))


def build_class_balance(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = predictions[predictions["method"] == "always_nn"]
    rows: list[dict] = []
    for (dataset, protocol, direction), group in reference.groupby(
        ["dataset", "protocol", "direction"], sort=True
    ):
        n0 = int((group["true_label"].astype(int) == 0).sum())
        n1 = int((group["true_label"].astype(int) == 1).sum())
        total = n0 + n1
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "direction": direction,
                "analysis_cell_display": CELL_DISPLAY[(protocol, direction)],
                "n_participants": int(group["subject"].nunique()),
                "n_recordings": total,
                "n_class_0": n0,
                "n_class_1": n1,
                "class_0_fraction": n0 / total,
                "class_1_fraction": n1 / total,
                "majority_class_accuracy": max(n0, n1) / total,
            }
        )
    return pd.DataFrame(rows)


def validate_input() -> tuple[pd.DataFrame, dict, dict]:
    for path in [INPUT_PATH, INPUT_MANIFEST_PATH, POLICY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件：{path}\n请先运行81号脚本。")
    input_manifest = json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    observed_hash = sha256(INPUT_PATH)
    expected_hash = input_manifest.get("prediction_sha256")
    if observed_hash != expected_hash:
        raise ValueError(
            "81号冻结预测的 SHA-256 与 manifest 不一致；请不要手工修改冻结预测文件。"
        )
    if input_manifest.get("status") != "passed" or input_manifest.get("rows") != 2660:
        raise ValueError("81号 manifest 未通过或行数不是2660")
    predictions = pd.read_csv(INPUT_PATH)
    if len(predictions) != 2660 or predictions["method"].nunique() != 14:
        raise ValueError("81号冻结预测不是14方法 × 190 recording")
    return predictions, input_manifest, policy


def main() -> None:
    predictions, input_manifest, policy = validate_input()
    pooled, calibration = build_pooled_metrics(predictions)
    participant = build_participant_metrics(predictions)
    participant_summary = summarise_participants(participant)
    class_balance = build_class_balance(predictions)

    expected_pooled_rows = 14 * 4
    expected_participant_rows = 14 * (13 + 13 + 15 + 13)
    if len(pooled) != expected_pooled_rows:
        raise AssertionError(f"总体指标应有{expected_pooled_rows}行，实际{len(pooled)}行")
    if len(participant) != expected_participant_rows:
        raise AssertionError(
            f"参与者指标应有{expected_participant_rows}行，实际{len(participant)}行"
        )
    if len(calibration) != expected_pooled_rows * 5:
        raise AssertionError("每个方法/实验单元必须有五个固定校准分箱")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pooled_recording_metrics": OUTPUT_ROOT / "82_总体recording级完整指标.csv",
        "participant_metrics": OUTPUT_ROOT / "82_每名参与者完整指标.csv",
        "participant_descriptive_summary": OUTPUT_ROOT / "82_参与者指标描述性汇总.csv",
        "calibration_bins": OUTPUT_ROOT / "82_概率校准五分箱数据.csv",
        "class_balance": OUTPUT_ROOT / "82_类别数量与平衡.csv",
    }
    pooled.to_csv(outputs["pooled_recording_metrics"], index=False, lineterminator="\n", float_format="%.17g")
    participant.to_csv(outputs["participant_metrics"], index=False, lineterminator="\n", float_format="%.17g")
    participant_summary.to_csv(
        outputs["participant_descriptive_summary"], index=False, lineterminator="\n", float_format="%.17g"
    )
    calibration.to_csv(outputs["calibration_bins"], index=False, lineterminator="\n", float_format="%.17g")
    class_balance.to_csv(outputs["class_balance"], index=False, lineterminator="\n", float_format="%.17g")

    output_hashes = {name: sha256(path) for name, path in outputs.items()}
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "analysis_role": policy["analysis_role"],
        "input_prediction_file": INPUT_PATH.name,
        "input_prediction_sha256": input_manifest["prediction_sha256"],
        "script_sha256": sha256(Path(__file__).resolve()),
        "evaluation_unit": "recording",
        "statistical_unit_for_inference": "participant; inference is deferred to script 83",
        "cross_task_combination_rule": "directions are reported separately in script 82",
        "positive_class": 1,
        "label_mapping": policy["binary_label"],
        "classification_threshold": "argmax(p0, p1), identical to the frozen prediction label",
        "metric_definitions": {
            "balanced_accuracy": "mean of sensitivity and specificity",
            "sensitivity": "TP/(TP+FN) for class 1",
            "specificity": "TN/(TN+FP) for class 0",
            "f1": "2TP/(2TP+FP+FN) for class 1",
            "auroc": "recording-level ROC area for p(class 1)",
            "pr_auc": "trapezoidal area under the precision-recall curve",
            "average_precision": "step-wise average precision; reported separately from trapezoidal PR-AUC",
            "log_loss": "mean negative natural log probability assigned to the true class; probabilities clipped to 1e-12",
            "brier_score": "mean squared error between p(class 1) and the binary label",
            "ece_5_bins": "recording-count-weighted absolute calibration gap in five fixed-width bins",
            "calibration_intercept_slope": "joint logistic calibration fit on clipped prediction logits with C=1e6",
        },
        "calibration_bin_edges": CALIBRATION_EDGES.tolist(),
        "pooled_metric_rows": int(len(pooled)),
        "participant_metric_rows": int(len(participant)),
        "calibration_rows": int(len(calibration)),
        "class_balance_rows": int(len(class_balance)),
        "output_sha256": output_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "old_route_a_results_read": False,
    }
    manifest_path = OUTPUT_ROOT / "82_完整指标_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = pooled[
        [
            "analysis_cell_display",
            "method_display",
            "n_participants",
            "n_recordings",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auroc",
            "brier_score",
        ]
    ].copy()
    for metric in ["accuracy", "balanced_accuracy", "f1", "auroc", "brier_score"]:
        display[metric] = display[metric].round(4)
    print("Route A v3 完整分类、概率与校准指标已生成")
    print(f"输入 SHA-256：{input_manifest['prediction_sha256']}")
    print(f"总体指标行数：{len(pooled)}")
    print(f"参与者指标行数：{len(participant)}")
    print(f"校准分箱行数：{len(calibration)}")
    print("统计推断：未在82号执行；83号将以参与者为单位进行配对统计")
    print(display.to_string(index=False))
    print("校验状态：passed")


if __name__ == "__main__":
    main()
