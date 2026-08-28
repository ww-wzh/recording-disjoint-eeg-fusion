"""编号87：运行 Route A v3 标签映射敏感性分析。

目的：回应审稿人对 natural/low 与 medium/high 二元合并依据的疑问。
本分析在 Arithmetic 和 Stroop 中分别使用 participant-disjoint LOSO，固定使用
训练折 StandardScaler + LogisticRegression，不进行超参数选择。

比较四种标签定义：
1. natural 与 low/medium/high；
2. natural/low 与 medium/high（论文主定义）；
3. natural/low/medium 与 high；
4. 仅保留 natural 与 high（极端等级分析）。

这是观察主结果后、按审稿意见增加的 post-hoc exploratory sensitivity analysis，
不会替换或改写 81 号冻结的 Route A v3 主预测。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：15_label_mapping_sensitivity
脚本支持 JSON checkpoint 续跑；中断后重新运行会跳过已完成折。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
PARENT_PROTOCOL_PATH = HERE / "64_冻结_RouteA_v3协议.json"
FREEZE_MANIFEST_PATH = HERE / "10_all_methods_frozen" / "81_冻结_RouteA_v3全部方法_manifest.json"
OUTPUT_ROOT = HERE / "15_label_mapping_sensitivity"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"

sys.path.insert(0, str(REVISION_ROOT))


BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_BASE_SEED = 20260827

ANALYSIS_CONFIG = {
    "analysis_version": "2026-08-27-route-a-v3-label-mapping-sensitivity-r1",
    "parent_protocol": "2026-08-22-route-a-v3",
    "status": "post_hoc_reviewer_requested_exploratory_sensitivity",
    "main_prediction_freeze_is_unchanged": True,
    "evaluation_protocol": "participant-disjoint leave-one-subject-out within each task",
    "learner": {
        "pipeline": "training-fold StandardScaler followed by LogisticRegression",
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 4000,
        "random_state": 0,
        "hyperparameter_selection": "none",
    },
    "aggregation": "geometric mean of window probabilities within each recording",
    "primary_metric": "participant-level recording balanced accuracy",
    "inferential_unit": "participant",
    "bootstrap": "10000 participant-level percentile resamples",
    "label_variants": {
        "threshold_1_natural_vs_rest": {
            "display": "Natural vs low/medium/high",
            "included_raw_levels": [0, 1, 2, 3],
            "class_0_raw_levels": [0],
            "class_1_raw_levels": [1, 2, 3],
            "role": "lower-threshold sensitivity",
        },
        "primary_threshold_2": {
            "display": "Natural/low vs medium/high",
            "included_raw_levels": [0, 1, 2, 3],
            "class_0_raw_levels": [0, 1],
            "class_1_raw_levels": [2, 3],
            "role": "manuscript primary binary mapping",
        },
        "threshold_3_high_vs_rest": {
            "display": "Natural/low/medium vs high",
            "included_raw_levels": [0, 1, 2, 3],
            "class_0_raw_levels": [0, 1, 2],
            "class_1_raw_levels": [3],
            "role": "upper-threshold sensitivity",
        },
        "extremes_natural_vs_high": {
            "display": "Natural vs high only",
            "included_raw_levels": [0, 3],
            "class_0_raw_levels": [0],
            "class_1_raw_levels": [3],
            "role": "extreme-level sensitivity with intermediate levels excluded",
        },
    },
    "interpretation_warning": (
        "The variants define different endpoints and class compositions. Differences describe "
        "label-definition sensitivity and do not establish that one mapping is uniquely correct."
    ),
}

TASK_DISPLAY = {"arithmetic": "Arithmetic", "stroop": "Stroop"}
PRIMARY_VARIANT = "primary_threshold_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_digest() -> str:
    payload = json.dumps(ANALYSIS_CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(value) for value in (BOOTSTRAP_BASE_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_core_engine():
    path = REVISION_ROOT / "run_corrected_experiments.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_label_mapping_engine", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载纯8通道数据引擎：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model():
    learner = ANALYSIS_CONFIG["learner"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(learner["C"]),
            class_weight=str(learner["class_weight"]),
            solver=str(learner["solver"]),
            max_iter=int(learner["max_iter"]),
            random_state=int(learner["random_state"]),
        ),
    )


def apply_label_variant(bundle, variant_name: str):
    variant = ANALYSIS_CONFIG["label_variants"][variant_name]
    included = np.asarray(variant["included_raw_levels"], dtype=int)
    indices = np.flatnonzero(np.isin(bundle.labels_raw, included))
    mapped = bundle.subset(indices)
    positive = set(int(value) for value in variant["class_1_raw_levels"])
    mapped.labels = np.asarray([int(int(value) in positive) for value in mapped.labels_raw], dtype=np.int64)
    if set(np.unique(mapped.labels).tolist()) != {0, 1}:
        raise AssertionError(f"{variant_name}过滤后没有同时包含两个类别")
    return mapped


def subset_participants(bundle, participants: list[int]):
    indices = np.flatnonzero(np.isin(bundle.subjects, participants))
    if len(indices) == 0:
        raise ValueError(f"{bundle.task}过滤后没有窗口")
    return bundle.subset(indices)


def safe_divide(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else float(numerator / denominator)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, p1: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    negative = y_true == 0
    positive = y_true == 1
    tn = int(np.sum(negative & (y_pred == 0)))
    fp = int(np.sum(negative & (y_pred == 1)))
    fn = int(np.sum(positive & (y_pred == 0)))
    tp = int(np.sum(positive & (y_pred == 1)))
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    clipped = np.clip(p1, 1e-12, 1.0 - 1e-12)
    selected = np.where(y_true == 1, clipped, 1.0 - clipped)
    if np.unique(y_true).size == 2:
        auroc = float(roc_auc_score(y_true, p1))
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, p1)
        pr_auc = float(auc(pr_recall, pr_precision))
        average_precision = float(average_precision_score(y_true, p1))
    else:
        auroc = float("nan")
        pr_auc = float("nan")
        average_precision = float("nan")
    return {
        "n_recordings": int(len(y_true)),
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
        "log_loss": float(-np.mean(np.log(selected))),
        "brier_score": float(np.mean(np.square(p1 - y_true))),
    }


def aggregate_recordings(bundle, validation: np.ndarray, probability: np.ndarray) -> list[dict]:
    labels = bundle.labels[validation]
    raw_labels = bundle.labels_raw[validation]
    recordings = bundle.recordings[validation]
    rows = []
    for recording in sorted(np.unique(recordings).tolist(), key=str):
        mask = recordings == recording
        binary_values = np.unique(labels[mask])
        raw_values = np.unique(raw_labels[mask])
        if binary_values.size != 1 or raw_values.size != 1:
            raise ValueError(f"recording {recording}标签不唯一")
        values = np.clip(probability[mask], 1e-12, 1.0)
        aggregate = np.exp(np.mean(np.log(values), axis=0))
        aggregate = aggregate / aggregate.sum()
        prediction = int(np.argmax(aggregate))
        rows.append(
            {
                "recording_id": str(recording),
                "raw_level": int(raw_values[0]),
                "true_label": int(binary_values[0]),
                "p0": float(aggregate[0]),
                "p1": float(aggregate[1]),
                "pred_label": prediction,
                "correct": int(prediction == int(binary_values[0])),
                "n_windows": int(np.sum(mask)),
            }
        )
    return rows


def checkpoint_path(task: str, variant_name: str, subject: int) -> Path:
    return CHECKPOINT_ROOT / task / variant_name / f"s{subject:02d}.json"


def run_or_load_fold(bundle, variant_name: str, subject: int, fingerprint: str) -> tuple[dict, list[dict]]:
    path = checkpoint_path(bundle.task, variant_name, subject)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("analysis_fingerprint") == fingerprint:
                return dict(stored["fold_row"]), list(stored["recording_rows"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    train = np.flatnonzero(bundle.subjects != subject)
    validation = np.flatnonzero(bundle.subjects == subject)
    train_subjects = sorted(np.unique(bundle.subjects[train]).astype(int).tolist())
    validation_subjects = sorted(np.unique(bundle.subjects[validation]).astype(int).tolist())
    if validation_subjects != [subject]:
        raise AssertionError("LOSO验证参与者不正确")
    if set(train_subjects) & set(validation_subjects):
        raise AssertionError("participant-disjoint划分出现参与者重叠")
    if set(bundle.recordings[train].tolist()) & set(bundle.recordings[validation].tolist()):
        raise AssertionError("participant-disjoint划分出现recording重叠")

    model = build_model()
    model.fit(bundle.features[train], bundle.labels[train])
    probability = model.predict_proba(bundle.features[validation])
    if list(model.classes_) != [0, 1]:
        raise AssertionError("LogisticRegression类别顺序不是[0,1]")
    recording_rows = aggregate_recordings(bundle, validation, probability)
    frame = pd.DataFrame(recording_rows)
    metrics = classification_metrics(
        frame["true_label"].to_numpy(dtype=int),
        frame["pred_label"].to_numpy(dtype=int),
        frame["p1"].to_numpy(dtype=float),
    )
    variant = ANALYSIS_CONFIG["label_variants"][variant_name]
    fold_row = {
        "task": bundle.task,
        "task_display": TASK_DISPLAY[bundle.task],
        "variant": variant_name,
        "variant_display": variant["display"],
        "variant_role": variant["role"],
        "held_out_subject": int(subject),
        "train_participants": int(len(train_subjects)),
        "validation_participants": 1,
        "train_recordings": int(len(np.unique(bundle.recordings[train]))),
        "validation_recordings": int(len(np.unique(bundle.recordings[validation]))),
        "train_windows": int(len(train)),
        "validation_windows": int(len(validation)),
        "participant_overlap": 0,
        "recording_overlap": 0,
        **metrics,
    }
    metadata = {
        "task": bundle.task,
        "task_display": TASK_DISPLAY[bundle.task],
        "variant": variant_name,
        "variant_display": variant["display"],
        "variant_role": variant["role"],
        "held_out_subject": int(subject),
    }
    recording_rows = [{**metadata, **row} for row in recording_rows]
    payload = {
        "analysis_fingerprint": fingerprint,
        "fold_row": fold_row,
        "recording_rows": recording_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return fold_row, recording_rows


def bootstrap_distribution(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("participant bootstrap要求至少两个有限值")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    return values[indices].mean(axis=1)


def percentile_interval(distribution: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(distribution, [0.025, 0.975], method="linear")
    return float(low), float(high)


def build_overall_summary(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auroc",
        "pr_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    ]
    rows = []
    for (task, variant_name), group in folds.groupby(["task", "variant"], sort=True):
        variant = ANALYSIS_CONFIG["label_variants"][variant_name]
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            distribution = bootstrap_distribution(values, stable_seed("summary", task, variant_name, metric))
            low, high = percentile_interval(distribution)
            rows.append(
                {
                    "task": task,
                    "task_display": TASK_DISPLAY[task],
                    "variant": variant_name,
                    "variant_display": variant["display"],
                    "variant_role": variant["role"],
                    "metric": metric,
                    "n_participants": int(len(values)),
                    "participant_macro_mean": float(np.mean(values)),
                    "participant_standard_deviation": float(np.std(values, ddof=1)),
                    "ci95_percentile_low": low,
                    "ci95_percentile_high": high,
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_pratt(differences: np.ndarray) -> tuple[float, float, str]:
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


def rank_biserial(differences: np.ndarray) -> float:
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    denominator = positive + negative
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(len(ranked), dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def build_paired_sensitivity(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alternatives = [name for name in ANALYSIS_CONFIG["label_variants"] if name != PRIMARY_VARIANT]
    for task, group in folds.groupby("task", sort=True):
        pivot = group.pivot(index="held_out_subject", columns="variant", values="balanced_accuracy")
        for alternative in alternatives:
            paired = pivot[[alternative, PRIMARY_VARIANT]].dropna()
            differences = (
                paired[alternative].to_numpy(dtype=float)
                - paired[PRIMARY_VARIANT].to_numpy(dtype=float)
            )
            distribution = bootstrap_distribution(
                differences, stable_seed("paired", task, alternative, PRIMARY_VARIANT)
            )
            low, high = percentile_interval(distribution)
            statistic, raw_p, method = wilcoxon_pratt(differences)
            rows.append(
                {
                    "task": task,
                    "task_display": TASK_DISPLAY[task],
                    "alternative_variant": alternative,
                    "alternative_display": ANALYSIS_CONFIG["label_variants"][alternative]["display"],
                    "reference_variant": PRIMARY_VARIANT,
                    "reference_display": ANALYSIS_CONFIG["label_variants"][PRIMARY_VARIANT]["display"],
                    "primary_metric": "participant-level recording balanced accuracy",
                    "difference_definition": "alternative minus primary mapping",
                    "n_paired_participants": int(len(differences)),
                    "alternative_mean": float(paired[alternative].mean()),
                    "primary_mapping_mean": float(paired[PRIMARY_VARIANT].mean()),
                    "mean_paired_difference": float(np.mean(differences)),
                    "mean_paired_difference_pp": float(100.0 * np.mean(differences)),
                    "ci95_percentile_low": low,
                    "ci95_percentile_high": high,
                    "ci95_percentile_low_pp": 100.0 * low,
                    "ci95_percentile_high_pp": 100.0 * high,
                    "participants_alternative_better": int(np.sum(differences > 0)),
                    "participants_equal": int(np.sum(np.isclose(differences, 0.0, atol=1e-15))),
                    "participants_alternative_worse": int(np.sum(differences < 0)),
                    "paired_rank_biserial": rank_biserial(differences),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_raw_p": raw_p,
                    "wilcoxon_status": method,
                    "analysis_status": ANALYSIS_CONFIG["status"],
                    "interpretation_warning": ANALYSIS_CONFIG["interpretation_warning"],
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 6:
        raise AssertionError(f"标签敏感性配对比较应有6行，实际{len(result)}行")
    result["bh_adjusted_p_six_comparison_family"] = bh_adjust(
        result["wilcoxon_raw_p"].to_numpy(dtype=float)
    )
    result["bh_significant_0_05"] = result["bh_adjusted_p_six_comparison_family"] < 0.05
    return result


def build_class_structure(folds: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task",
        "task_display",
        "variant",
        "variant_display",
        "variant_role",
        "validation_recordings",
        "n_class_0",
        "n_class_1",
    ]
    return (
        folds[columns]
        .groupby(["task", "task_display", "variant", "variant_display", "variant_role"], sort=True)
        .agg(
            n_participants=("validation_recordings", "size"),
            recordings_per_participant=("validation_recordings", "first"),
            class_0_recordings_per_participant=("n_class_0", "first"),
            class_1_recordings_per_participant=("n_class_1", "first"),
        )
        .reset_index()
    )


def main() -> None:
    if not PARENT_PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"缺少 Route A v3 冻结协议：{PARENT_PROTOCOL_PATH}")
    if not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"缺少 81 号主预测冻结清单：{FREEZE_MANIFEST_PATH}")
    parent = json.loads(PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if parent.get("protocol_version") != ANALYSIS_CONFIG["parent_protocol"]:
        raise ValueError("87号分析与64号父协议版本不匹配")
    if freeze.get("status") != "passed":
        raise ValueError("81号主预测冻结清单未通过校验")

    script_digest = sha256(Path(__file__).resolve())
    analysis_config_digest = config_digest()
    fingerprint = hashlib.sha256(
        "|".join(
            [
                script_digest,
                analysis_config_digest,
                sha256(PARENT_PROTOCOL_PATH),
                sha256(FREEZE_MANIFEST_PATH),
            ]
        ).encode("utf-8")
    ).hexdigest()

    print("[1/4] 读取纯8通道、已移除packet counter的272维特征。")
    core = load_core_engine()
    arithmetic, stroop, metadata, _ = core.load_openbci_eeg_only(PROJECT_ROOT, torch.device("cpu"))
    if metadata.get("feature_dimension") != 272 or metadata.get("eeg_channels") != 8:
        raise AssertionError(f"特征或通道数不符合冻结协议：{metadata}")
    if not metadata.get("packet_counter_removed"):
        raise AssertionError("packet counter未移除")
    arithmetic = subset_participants(arithmetic, list(range(1, 16)))
    stroop = subset_participants(stroop, list(range(1, 14)))

    all_fold_rows = []
    all_recording_rows = []
    print("[2/4] 运行或续跑两项任务的四种标签定义 participant-disjoint LOSO。")
    for original_bundle in [arithmetic, stroop]:
        subjects = sorted(np.unique(original_bundle.subjects).astype(int).tolist())
        for variant_name in ANALYSIS_CONFIG["label_variants"]:
            mapped_bundle = apply_label_variant(original_bundle, variant_name)
            for subject in subjects:
                fold_row, recording_rows = run_or_load_fold(
                    mapped_bundle, variant_name, subject, fingerprint
                )
                all_fold_rows.append(fold_row)
                all_recording_rows.extend(recording_rows)
            print(f"[done] {original_bundle.task} - {variant_name}")

    folds = pd.DataFrame(all_fold_rows)
    recording_predictions = pd.DataFrame(all_recording_rows)
    if len(folds) != (15 + 13) * 4:
        raise AssertionError(f"逐折结果应有112行，实际{len(folds)}行")
    if len(recording_predictions) != (15 + 13) * (4 + 4 + 4 + 2):
        raise AssertionError(f"recording预测应有392行，实际{len(recording_predictions)}行")

    print("[3/4] 生成participant bootstrap、配对敏感性和类别结构。")
    summary = build_overall_summary(folds)
    paired = build_paired_sensitivity(folds)
    class_structure = build_class_structure(folds)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "folds": OUTPUT_ROOT / "87_标签映射敏感性逐参与者结果.csv",
        "recording_predictions": OUTPUT_ROOT / "87_标签映射敏感性recording级预测.csv",
        "overall_summary": OUTPUT_ROOT / "87_标签映射敏感性全部指标BootstrapCI.csv",
        "paired_sensitivity": OUTPUT_ROOT / "87_标签映射相对主定义配对敏感性与BH.csv",
        "class_structure": OUTPUT_ROOT / "87_标签映射类别结构.csv",
        "protocol_snapshot": OUTPUT_ROOT / "87_标签映射敏感性协议快照.json",
        "manifest": OUTPUT_ROOT / "87_标签映射敏感性_manifest.json",
    }
    frames = {
        "folds": folds,
        "recording_predictions": recording_predictions,
        "overall_summary": summary,
        "paired_sensitivity": paired,
        "class_structure": class_structure,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False, lineterminator="\n", float_format="%.17g")
    outputs["protocol_snapshot"].write_text(
        json.dumps(ANALYSIS_CONFIG, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "analysis_version": ANALYSIS_CONFIG["analysis_version"],
        "status": "passed",
        "analysis_role": ANALYSIS_CONFIG["status"],
        "main_prediction_freeze_is_unchanged": True,
        "parent_protocol_sha256": sha256(PARENT_PROTOCOL_PATH),
        "main_prediction_freeze_manifest_sha256": sha256(FREEZE_MANIFEST_PATH),
        "script_sha256": script_digest,
        "analysis_config_sha256": analysis_config_digest,
        "analysis_fingerprint": fingerprint,
        "data": {
            "eeg_channels": int(metadata["eeg_channels"]),
            "feature_dimension": int(metadata["feature_dimension"]),
            "packet_counter_removed": bool(metadata["packet_counter_removed"]),
            "arithmetic_participants": 15,
            "stroop_participants": 13,
        },
        "counts": {
            "participant_folds": int(len(folds)),
            "recording_predictions": int(len(recording_predictions)),
            "label_variants": int(len(ANALYSIS_CONFIG["label_variants"])),
            "paired_comparisons": int(len(paired)),
            "bh_significant_comparisons": int(paired["bh_significant_0_05"].sum()),
        },
        "outputs": {
            name: {"file": path.name, "sha256": sha256(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "interpretation_warning": ANALYSIS_CONFIG["interpretation_warning"],
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[4/4] Route A v3 标签映射敏感性分析完成。")
    balanced = summary[summary["metric"] == "balanced_accuracy"].copy()
    print(
        balanced[
            [
                "task_display",
                "variant_display",
                "n_participants",
                "participant_macro_mean",
                "ci95_percentile_low",
                "ci95_percentile_high",
            ]
        ].to_string(index=False)
    )
    print(f"逐参与者结果：{len(folds)}；recording预测：{len(recording_predictions)}")
    print(f"六项标签敏感性比较BH校正后显著项：{int(paired['bh_significant_0_05'].sum())}")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
