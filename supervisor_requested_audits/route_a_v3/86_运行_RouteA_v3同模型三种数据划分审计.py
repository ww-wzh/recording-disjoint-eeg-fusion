"""编号86：运行 Route A v3 同模型、同超参数的三种数据划分审计。

比较对象：
- random-window：同一 recording 的重叠窗口可同时出现在训练和验证；
- recording-disjoint：同一 participant 内按完整 recording 划分；
- participant-disjoint：完整留一参与者验证。

三种划分均使用训练折 StandardScaler + 固定 LogisticRegression，不调参。
脚本支持 JSON checkpoint 续跑；中断后重新运行会跳过已完成折。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
冻结协议：85_冻结_RouteA_v3同模型三种数据划分审计协议.json
输出目录：14_three_split_audit
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
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
PROTOCOL_PATH = HERE / "85_冻结_RouteA_v3同模型三种数据划分审计协议.json"
PARENT_PROTOCOL_PATH = HERE / "64_冻结_RouteA_v3协议.json"
STABILITY_MANIFEST_PATH = HERE / "13_stability_audit" / "84_样本结构与稳定性审计_manifest.json"
OUTPUT_ROOT = HERE / "14_three_split_audit"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"

sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.splits import Split, balanced_recording_folds  # noqa: E402


BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_BASE_SEED = 20260826
WINDOW_SECONDS = 8.5
STRIDE_SECONDS = 0.5

STRATEGY_DISPLAY = {
    "random_window": "Random-window split",
    "recording_disjoint": "Recording-disjoint split",
    "participant_disjoint": "Participant-disjoint split",
}

COMPARISON_PAIRS = [
    ("random_window", "recording_disjoint"),
    ("recording_disjoint", "participant_disjoint"),
    ("random_window", "participant_disjoint"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(value) for value in (BOOTSTRAP_BASE_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_core_engine():
    path = REVISION_ROOT / "run_corrected_experiments.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_three_split_engine", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载纯8通道数据引擎：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model(protocol: dict):
    learner = protocol["learner"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(learner["C"]),
            class_weight=str(learner["class_weight"]),
            solver=str(learner["solver"]),
            max_iter=int(learner["max_iter"]),
            random_state=int(learner["model_random_state"]),
        ),
    )


def subset_bundle(bundle, participants: list[int]):
    indices = np.flatnonzero(np.isin(bundle.subjects, participants))
    if len(indices) == 0:
        raise ValueError(f"{bundle.task}过滤后没有窗口")
    return bundle.subset(indices)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


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
    probability = np.column_stack([1.0 - clipped, clipped])
    selected = probability[np.arange(len(y_true)), y_true]
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
        "n": int(len(y_true)),
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


def recording_probability_rows(bundle, validation: np.ndarray, probability: np.ndarray) -> list[dict]:
    labels = bundle.labels[validation]
    recordings = bundle.recordings[validation]
    rows: list[dict] = []
    for recording in sorted(np.unique(recordings).tolist(), key=str):
        mask = recordings == recording
        values = np.unique(labels[mask])
        if values.size != 1:
            raise ValueError(f"recording {recording}含多个标签")
        probabilities = np.clip(probability[mask], 1e-12, 1.0)
        aggregate = np.exp(np.mean(np.log(probabilities), axis=0))
        aggregate = aggregate / aggregate.sum()
        prediction = int(np.argmax(aggregate))
        rows.append(
            {
                "recording_id": str(recording),
                "true_label": int(values[0]),
                "p0": float(aggregate[0]),
                "p1": float(aggregate[1]),
                "pred_label": prediction,
                "correct": int(prediction == int(values[0])),
                "validation_windows_from_recording": int(np.sum(mask)),
            }
        )
    return rows


def parse_window_position(window_id: object) -> int:
    value = str(window_id)
    marker = value.rsplit(":w", maxsplit=1)
    if len(marker) != 2:
        raise ValueError(f"无法解析window ID：{value}")
    return int(marker[1])


def raw_overlap_diagnostics(bundle, train: np.ndarray, validation: np.ndarray) -> tuple[float, float, float]:
    train_positions: dict[str, np.ndarray] = {}
    for recording in np.unique(bundle.recordings[train]):
        mask = bundle.recordings[train] == recording
        positions = np.asarray(
            [parse_window_position(value) for value in bundle.window_ids[train][mask]],
            dtype=int,
        )
        train_positions[str(recording)] = np.sort(positions)

    overlaps = []
    for recording, window_id in zip(bundle.recordings[validation], bundle.window_ids[validation]):
        candidates = train_positions.get(str(recording))
        if candidates is None or len(candidates) == 0:
            overlaps.append(0.0)
            continue
        position = parse_window_position(window_id)
        insertion = int(np.searchsorted(candidates, position))
        distances = []
        if insertion < len(candidates):
            distances.append(abs(int(candidates[insertion]) - position))
        if insertion > 0:
            distances.append(abs(int(candidates[insertion - 1]) - position))
        nearest_distance = min(distances)
        temporal_separation = nearest_distance * STRIDE_SECONDS
        overlap = max(0.0, WINDOW_SECONDS - temporal_separation) / WINDOW_SECONDS
        overlaps.append(float(overlap))
    values = np.asarray(overlaps, dtype=float)
    return float(np.mean(values > 0.0)), float(np.mean(values)), float(np.max(values))


def evaluate_split(bundle, split: Split, protocol: dict) -> tuple[dict, list[dict]]:
    model = build_model(protocol)
    model.fit(bundle.features[split.train], bundle.labels[split.train])
    probability = model.predict_proba(bundle.features[split.validation])
    if list(model.classes_) != [0, 1]:
        raise AssertionError("LogisticRegression类别顺序不是[0,1]")
    prediction = np.argmax(probability, axis=1).astype(int)

    train_recordings = set(bundle.recordings[split.train].tolist())
    validation_recordings = set(bundle.recordings[split.validation].tolist())
    train_subjects = set(int(value) for value in bundle.subjects[split.train].tolist())
    validation_subjects = set(int(value) for value in bundle.subjects[split.validation].tolist())
    recording_overlap = train_recordings & validation_recordings
    participant_overlap = train_subjects & validation_subjects
    contaminated_fraction, mean_raw_overlap, maximum_raw_overlap = raw_overlap_diagnostics(
        bundle, split.train, split.validation
    )

    recording_rows = recording_probability_rows(bundle, split.validation, probability)
    recording_frame = pd.DataFrame(recording_rows)
    recording_metrics = classification_metrics(
        recording_frame["true_label"].to_numpy(dtype=int),
        recording_frame["pred_label"].to_numpy(dtype=int),
        recording_frame["p1"].to_numpy(dtype=float),
    )
    window_metrics = classification_metrics(
        bundle.labels[split.validation],
        prediction,
        probability[:, 1],
    )
    fold = {
        "train_windows": int(len(split.train)),
        "validation_windows": int(len(split.validation)),
        "train_recordings": int(len(train_recordings)),
        "validation_recordings": int(len(validation_recordings)),
        "train_participants": int(len(train_subjects)),
        "validation_participants": int(len(validation_subjects)),
        "overlapping_recordings": int(len(recording_overlap)),
        "recording_overlap_rate": float(len(recording_overlap) / max(1, len(validation_recordings))),
        "overlapping_participants": int(len(participant_overlap)),
        "participant_overlap_rate": float(len(participant_overlap) / max(1, len(validation_subjects))),
        "validation_windows_with_raw_overlap_fraction": contaminated_fraction,
        "mean_nearest_train_raw_overlap_fraction": mean_raw_overlap,
        "maximum_nearest_train_raw_overlap_fraction": maximum_raw_overlap,
        **{f"window_{key}": value for key, value in window_metrics.items()},
        **{f"recording_{key}": value for key, value in recording_metrics.items()},
    }
    return fold, recording_rows


def checkpoint_path(task: str, strategy: str, subject: int, split_seed: int | None, fold: int) -> Path:
    seed_text = "none" if split_seed is None else str(int(split_seed))
    return CHECKPOINT_ROOT / task / strategy / f"s{subject:02d}_seed{seed_text}_fold{fold:02d}.json"


def run_or_load_checkpoint(
    bundle,
    split: Split,
    protocol: dict,
    protocol_digest: str,
    *,
    strategy: str,
    held_out_subject: int,
    split_seed: int | None,
    fold: int,
) -> tuple[dict, list[dict]]:
    path = checkpoint_path(bundle.task, strategy, held_out_subject, split_seed, fold)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("protocol_sha256") == protocol_digest:
                return dict(stored["fold_row"]), list(stored["recording_rows"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    fold_row, recording_rows = evaluate_split(bundle, split, protocol)
    metadata = {
        "task": bundle.task,
        "strategy": strategy,
        "strategy_display": STRATEGY_DISPLAY[strategy],
        "held_out_subject": int(held_out_subject),
        "split_seed": None if split_seed is None else int(split_seed),
        "fold": int(fold),
    }
    fold_row = {**metadata, **fold_row}
    recording_rows = [{**metadata, **row} for row in recording_rows]
    payload = {
        "protocol_sha256": protocol_digest,
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


def run_within_participant(bundle, protocol: dict, protocol_digest: str) -> None:
    split_seeds = [int(value) for value in protocol["random_window"]["split_seeds"]]
    for subject in sorted(np.unique(bundle.subjects).astype(int).tolist()):
        subset = bundle.subset(np.flatnonzero(bundle.subjects == subject))
        for seed in split_seeds:
            splitter = StratifiedShuffleSplit(
                n_splits=int(protocol["random_window"]["splits_per_seed"]),
                test_size=float(protocol["random_window"]["validation_fraction"]),
                random_state=seed,
            )
            for fold, (train, validation) in enumerate(
                splitter.split(subset.features, subset.labels), start=1
            ):
                run_or_load_checkpoint(
                    subset,
                    Split(train=np.asarray(train, dtype=np.int64), validation=np.asarray(validation, dtype=np.int64)),
                    protocol,
                    protocol_digest,
                    strategy="random_window",
                    held_out_subject=subject,
                    split_seed=seed,
                    fold=fold,
                )
            for fold, split in enumerate(
                balanced_recording_folds(subset.labels_raw, subset.recordings, seed), start=1
            ):
                run_or_load_checkpoint(
                    subset,
                    split,
                    protocol,
                    protocol_digest,
                    strategy="recording_disjoint",
                    held_out_subject=subject,
                    split_seed=seed,
                    fold=fold,
                )
        print(f"[within done] {bundle.task} S{subject:02d}")


def run_participant_disjoint(bundle, protocol: dict, protocol_digest: str) -> None:
    subjects = sorted(np.unique(bundle.subjects).astype(int).tolist())
    for fold, subject in enumerate(subjects, start=1):
        validation = np.flatnonzero(bundle.subjects == subject).astype(np.int64)
        train = np.flatnonzero(bundle.subjects != subject).astype(np.int64)
        run_or_load_checkpoint(
            bundle,
            Split(train=train, validation=validation),
            protocol,
            protocol_digest,
            strategy="participant_disjoint",
            held_out_subject=subject,
            split_seed=None,
            fold=fold,
        )
        print(f"[participant done] {bundle.task} S{subject:02d}")


def collect_checkpoints(protocol_digest: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    recording_rows = []
    for path in sorted(CHECKPOINT_ROOT.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != protocol_digest:
            raise ValueError(f"checkpoint协议哈希不一致：{path}")
        fold_rows.append(payload["fold_row"])
        recording_rows.extend(payload["recording_rows"])
    folds = pd.DataFrame(fold_rows)
    recordings = pd.DataFrame(recording_rows)
    if len(folds) != 588:
        raise AssertionError(f"逐折结果应有588行，实际{len(folds)}行")
    if len(recordings) < 1700:
        raise AssertionError(f"recording预测数量异常：{len(recordings)}")
    return folds, recordings


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


def build_subject_summary(folds: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "recording_accuracy",
        "recording_balanced_accuracy",
        "recording_sensitivity",
        "recording_specificity",
        "recording_f1",
        "recording_auroc",
        "recording_log_loss",
        "recording_brier_score",
        "window_accuracy",
        "window_balanced_accuracy",
        "window_log_loss",
        "recording_overlap_rate",
        "participant_overlap_rate",
        "validation_windows_with_raw_overlap_fraction",
        "mean_nearest_train_raw_overlap_fraction",
    ]
    aggregations = {column: ["mean", "std", "min", "max"] for column in metric_columns}
    grouped = folds.groupby(
        ["task", "strategy", "strategy_display", "held_out_subject"], sort=True
    ).agg(aggregations)
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    grouped = grouped.reset_index()
    evaluation_counts = (
        folds.groupby(["task", "strategy", "held_out_subject"]).size().rename("n_evaluations").reset_index()
    )
    grouped = grouped.merge(
        evaluation_counts,
        on=["task", "strategy", "held_out_subject"],
        validate="one_to_one",
    )
    expected = 15 * 3 + 13 * 3
    if len(grouped) != expected:
        raise AssertionError(f"参与者级三划分结果应有{expected}行，实际{len(grouped)}")
    return grouped


def build_overall_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "recording_accuracy_mean",
        "recording_balanced_accuracy_mean",
        "recording_sensitivity_mean",
        "recording_specificity_mean",
        "recording_f1_mean",
        "recording_auroc_mean",
        "recording_log_loss_mean",
        "recording_brier_score_mean",
        "window_accuracy_mean",
        "window_balanced_accuracy_mean",
        "window_log_loss_mean",
        "recording_overlap_rate_mean",
        "participant_overlap_rate_mean",
        "validation_windows_with_raw_overlap_fraction_mean",
        "mean_nearest_train_raw_overlap_fraction_mean",
    ]
    rows = []
    for (task, strategy), group in subjects.groupby(["task", "strategy"], sort=True):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            distribution = bootstrap_distribution(values, stable_seed("summary", task, strategy, metric))
            low, high = percentile_interval(distribution)
            rows.append(
                {
                    "task": task,
                    "strategy": strategy,
                    "strategy_display": STRATEGY_DISPLAY[strategy],
                    "metric": metric.removesuffix("_mean"),
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


def build_paired_comparisons(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, task_group in subjects.groupby("task", sort=True):
        pivot = task_group.pivot(
            index="held_out_subject",
            columns="strategy",
            values="recording_balanced_accuracy_mean",
        )
        for left, right in COMPARISON_PAIRS:
            paired = pivot[[left, right]].dropna()
            differences = paired[left].to_numpy(dtype=float) - paired[right].to_numpy(dtype=float)
            distribution = bootstrap_distribution(differences, stable_seed("paired", task, left, right))
            low, high = percentile_interval(distribution)
            statistic, raw_p, status = wilcoxon_pratt(differences)
            rows.append(
                {
                    "task": task,
                    "primary_metric": "participant-level recording balanced accuracy",
                    "left_strategy": left,
                    "left_strategy_display": STRATEGY_DISPLAY[left],
                    "right_strategy": right,
                    "right_strategy_display": STRATEGY_DISPLAY[right],
                    "difference_definition": "left minus right",
                    "n_paired_participants": int(len(differences)),
                    "left_mean": float(paired[left].mean()),
                    "right_mean": float(paired[right].mean()),
                    "mean_paired_difference": float(np.mean(differences)),
                    "mean_paired_difference_pp": float(100.0 * np.mean(differences)),
                    "ci95_percentile_low": low,
                    "ci95_percentile_high": high,
                    "ci95_percentile_low_pp": 100.0 * low,
                    "ci95_percentile_high_pp": 100.0 * high,
                    "participants_left_better": int(np.sum(differences > 0)),
                    "participants_equal": int(np.sum(np.isclose(differences, 0.0, atol=1e-15))),
                    "participants_left_worse": int(np.sum(differences < 0)),
                    "paired_rank_biserial": rank_biserial(differences),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_raw_p": raw_p,
                    "wilcoxon_status": status,
                    "interpretation_warning": (
                        "Most direct matched estimate of same-recording near-duplicate contamination."
                        if {left, right} == {"random_window", "recording_disjoint"}
                        else "Participant-disjoint evaluation also changes training population and domain shift; do not attribute the full difference to leakage."
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 6:
        raise AssertionError("应有6项划分配对比较")
    result["bh_adjusted_p_six_comparison_family"] = bh_adjust(
        result["wilcoxon_raw_p"].to_numpy(dtype=float)
    )
    result["bh_significant_0_05"] = result["bh_adjusted_p_six_comparison_family"] < 0.05
    return result


def main() -> None:
    for path in [PROTOCOL_PATH, PARENT_PROTOCOL_PATH, STABILITY_MANIFEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    stability = json.loads(STABILITY_MANIFEST_PATH.read_text(encoding="utf-8"))
    if parent.get("protocol_version") != protocol.get("parent_protocol"):
        raise ValueError("85号审计协议的parent protocol不匹配")
    if stability.get("status") != "passed" or stability.get("protocol_version") != parent.get("protocol_version"):
        raise ValueError("84号稳定性审计未通过或协议版本不匹配")
    protocol_digest = sha256(PROTOCOL_PATH)

    print("[1/5] 读取纯8通道、已移除packet counter的272维特征。")
    core = load_core_engine()
    arithmetic, stroop, metadata, _ = core.load_openbci_eeg_only(PROJECT_ROOT, torch.device("cpu"))
    if metadata.get("feature_dimension") != 272 or metadata.get("eeg_channels") != 8:
        raise AssertionError(f"特征或通道数不符合冻结协议：{metadata}")
    if not metadata.get("packet_counter_removed"):
        raise AssertionError("packet counter未移除")
    arithmetic = subset_bundle(arithmetic, [int(value) for value in protocol["data"]["arithmetic_participants"]])
    stroop = subset_bundle(stroop, [int(value) for value in protocol["data"]["stroop_participants"]])

    print("[2/5] Arithmetic：运行或续跑三种划分。")
    run_within_participant(arithmetic, protocol, protocol_digest)
    run_participant_disjoint(arithmetic, protocol, protocol_digest)

    print("[3/5] Stroop：运行或续跑三种划分。")
    run_within_participant(stroop, protocol, protocol_digest)
    run_participant_disjoint(stroop, protocol, protocol_digest)

    print("[4/5] 汇总participant-level结果和配对统计。")
    folds, recording_predictions = collect_checkpoints(protocol_digest)
    subjects = build_subject_summary(folds)
    summary = build_overall_summary(subjects)
    paired = build_paired_comparisons(subjects)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "folds": OUTPUT_ROOT / "86_三种划分逐折审计.csv",
        "recording_predictions": OUTPUT_ROOT / "86_三种划分recording级预测.csv",
        "participant_summary": OUTPUT_ROOT / "86_三种划分参与者级汇总.csv",
        "overall_summary": OUTPUT_ROOT / "86_三种划分全部指标BootstrapCI.csv",
        "paired_comparisons": OUTPUT_ROOT / "86_三种划分配对差值与BH.csv",
    }
    frames = {
        "folds": folds,
        "recording_predictions": recording_predictions,
        "participant_summary": subjects,
        "overall_summary": summary,
        "paired_comparisons": paired,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False, lineterminator="\n", float_format="%.17g")

    source_paths = [
        PROTOCOL_PATH,
        PARENT_PROTOCOL_PATH,
        STABILITY_MANIFEST_PATH,
        REVISION_ROOT / "run_corrected_experiments.py",
        REVISION_ROOT / "eeg_channel_selection.py",
        REVISION_ROOT / "revision_pipeline" / "splits.py",
        PROJECT_ROOT / "1111.py",
    ]
    manifest = {
        "status": "passed",
        "protocol_version": protocol["protocol_version"],
        "parent_protocol": protocol["parent_protocol"],
        "analysis_role": protocol["analysis_role"],
        "protocol_sha256": protocol_digest,
        "script_sha256": sha256(Path(__file__).resolve()),
        "feature_metadata": metadata,
        "learner": protocol["learner"],
        "same_model_and_hyperparameters_across_splits": True,
        "arithmetic_participants": 15,
        "stroop_participants": 13,
        "fold_rows": int(len(folds)),
        "recording_prediction_rows": int(len(recording_predictions)),
        "participant_summary_rows": int(len(subjects)),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_unit": "participant; split repeats averaged within participant first",
        "bootstrap_interval": "percentile 95%",
        "paired_test": "two-sided approximate Wilcoxon, Pratt zeros, no continuity correction",
        "bh_family": "six post-hoc split comparisons",
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "checkpoint_count": int(len(list(CHECKPOINT_ROOT.rglob("*.json")))),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "old_route_a_results_read": False,
    }
    manifest_path = OUTPUT_ROOT / "86_三种数据划分审计_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[5/5] Route A v3三种数据划分审计完成。")
    primary = summary[summary["metric"] == "recording_balanced_accuracy"][[
        "task",
        "strategy_display",
        "n_participants",
        "participant_macro_mean",
        "ci95_percentile_low",
        "ci95_percentile_high",
    ]].copy()
    for column in ["participant_macro_mean", "ci95_percentile_low", "ci95_percentile_high"]:
        primary[column] = primary[column].round(4)
    print(primary.to_string(index=False))
    leakage = paired[
        (paired["left_strategy"] == "random_window")
        & (paired["right_strategy"] == "recording_disjoint")
    ][[
        "task",
        "mean_paired_difference_pp",
        "ci95_percentile_low_pp",
        "ci95_percentile_high_pp",
        "wilcoxon_raw_p",
        "bh_adjusted_p_six_comparison_family",
    ]].copy()
    print("Random-window minus recording-disjoint：")
    print(leakage.round(4).to_string(index=False))
    print(f"逐折结果：{len(folds)}；recording预测：{len(recording_predictions)}")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
