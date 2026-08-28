"""编号84：生成 Route A v3 样本结构与稳定性审计材料。

本脚本不训练模型，专门回应审稿人关于小样本和稳定性的质疑：
1. 每条 recording 的窗口数、时长、哈希与纳入状态；
2. 每个 inner fold 的实际 participant、recording 和 window 数量；
3. 八种有 seed-level 预测的方法在五个随机种子下的性能波动；
4. Residual MLP、EEGNet、EEG-Conformer 的 selected epoch 波动；
5. DASF 逐 seed 决策一致性和 CB-SF 最终 ensemble-level 权重分布。

seed-level 结果仅用于稳定性诊断，绝不作为统计重复。最终统计仍只使用81号
冻结的五种子集成预测，participant 仍是唯一推断单位。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：13_stability_audit
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
PREFLIGHT_PATH = HERE / "preflight" / "65_预检_recording清单.csv"
FROZEN_MANIFEST_PATH = (
    HERE / "10_all_methods_frozen" / "81_冻结_RouteA_v3全部方法_manifest.json"
)
METRICS_MANIFEST_PATH = HERE / "11_complete_metrics" / "82_完整指标_manifest.json"
STATS_MANIFEST_PATH = (
    HERE / "12_participant_statistics" / "83_参与者Bootstrap与配对统计_manifest.json"
)
OUTPUT_ROOT = HERE / "13_stability_audit"

sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.splits import balanced_recording_folds  # noqa: E402


SEEDS = [1335, 1388, 1441, 1494, 1547]

RAW_SEED_SOURCES = [
    (
        Path("01_core/cross_task/raw_seed_predictions.csv"),
        ["always_nn", "always_fuse", "rf", "extra_trees"],
    ),
    (
        Path("01_core/loso_arithmetic/raw_seed_predictions.csv"),
        ["always_nn", "always_fuse", "rf", "extra_trees"],
    ),
    (
        Path("01_core/loso_stroop/raw_seed_predictions.csv"),
        ["always_nn", "always_fuse", "rf", "extra_trees"],
    ),
    (Path("07_riemannian/raw_seed_predictions.csv"), ["riemann_mdm", "riemann_ts_logreg"]),
    (Path("08_eegnet/raw_seed_predictions.csv"), ["eegnet"]),
    (Path("09_eeg_conformer/raw_seed_predictions.csv"), ["eeg_conformer"]),
]

SEED_DIAGNOSTIC_METHODS = [
    "always_nn",
    "always_fuse",
    "rf",
    "extra_trees",
    "riemann_mdm",
    "riemann_ts_logreg",
    "eegnet",
    "eeg_conformer",
]

METHOD_DISPLAY = {
    "always_nn": "Always neural",
    "always_fuse": "Always fused",
    "rf": "Random forest",
    "extra_trees": "Extra Trees",
    "riemann_mdm": "Riemannian MDM",
    "riemann_ts_logreg": "Riemannian TS-LR",
    "eegnet": "EEGNet",
    "eeg_conformer": "EEG-Conformer",
}

SETTING_SPECS = [
    ("cross_task_bidirectional", "Bidirectional cross-task", "cross_task", None, 13, 6),
    ("cross_task_arithmetic_to_stroop", "Arithmetic to Stroop", "cross_task", "arithmetic_to_stroop", 13, 3),
    ("cross_task_stroop_to_arithmetic", "Stroop to Arithmetic", "cross_task", "stroop_to_arithmetic", 13, 3),
    ("loso_arithmetic", "Arithmetic LOSO", "loso", "arithmetic", 15, 4),
    ("loso_stroop", "Stroop LOSO", "loso", "stroop", 13, 4),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joined(values: list[object]) -> str:
    return "|".join(str(value) for value in values)


def validate_inputs() -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    required = [
        POLICY_PATH,
        PREFLIGHT_PATH,
        FROZEN_MANIFEST_PATH,
        METRICS_MANIFEST_PATH,
        STATS_MANIFEST_PATH,
        *[HERE / path for path, _ in RAW_SEED_SOURCES],
        HERE / "03_dasf" / "gate_diagnostics_seed.csv",
        HERE / "03_dasf" / "gate_decisions_final.csv",
        HERE / "04_cbsf" / "gate_weights.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少输入文件，请先完成65和81--83号：{missing}")

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    frozen_manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    metrics_manifest = json.loads(METRICS_MANIFEST_PATH.read_text(encoding="utf-8"))
    stats_manifest = json.loads(STATS_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_prediction_hash = frozen_manifest["prediction_sha256"]
    if metrics_manifest.get("input_prediction_sha256") != expected_prediction_hash:
        raise ValueError("82号不是从当前81号冻结预测生成")
    if stats_manifest.get("input_prediction_sha256") != expected_prediction_hash:
        raise ValueError("83号不是从当前81号冻结预测生成")
    if any(manifest.get("status") != "passed" for manifest in [frozen_manifest, metrics_manifest, stats_manifest]):
        raise ValueError("81--83号中至少一个manifest未通过")

    preflight = pd.read_csv(PREFLIGHT_PATH)
    if len(preflight) != 120 or preflight["recording_id"].nunique() != 120:
        raise ValueError("65号预检清单应包含120个原始文件记录")
    return preflight, policy, frozen_manifest, metrics_manifest, stats_manifest


def build_recording_audit(preflight: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = preflight.copy()
    output["subject"] = output["subject"].astype(int)
    output["raw_label"] = output["raw_label"].astype(int)
    output["binary_label"] = (output["raw_label"] >= 2).astype(int)
    output["source_filename"] = output["file"].map(lambda value: Path(str(value)).name)
    output = output.drop(columns=["file"])
    output["duplicate_hash_group_size"] = output.groupby("file_sha256")["recording_id"].transform("size")
    output["is_exact_duplicate_signal"] = output["duplicate_hash_group_size"] > 1

    in_cross_cohort = output["subject"].between(1, 13)
    output["included_cross_task_source"] = in_cross_cohort
    output["included_cross_task_target"] = in_cross_cohort & output["raw_label"].isin([1, 2, 3])
    output["included_arithmetic_loso"] = output["task"].eq("arithmetic")
    output["included_stroop_loso"] = output["task"].eq("stroop") & in_cross_cohort
    output["included_any_v3_analysis"] = output[
        [
            "included_cross_task_source",
            "included_cross_task_target",
            "included_arithmetic_loso",
            "included_stroop_loso",
        ]
    ].any(axis=1)

    reasons = []
    for row in output.itertuples(index=False):
        notes: list[str] = []
        if row.task == "stroop" and row.subject in {14, 15}:
            notes.append("excluded_from_stroop_v3_due_to_unresolved_S14_S15_high_file_duplicate")
        if row.subject in {14, 15}:
            notes.append("outside_cross_task_complete_case_cohort")
        if row.raw_label == 0:
            notes.append("excluded_from_cross_task_target_due_to_same_subject_cross_task_natural_duplicate")
        reasons.append(";".join(notes) if notes else "included_without_special_exclusion")
    output["audit_note"] = reasons

    columns = [
        "task",
        "subject",
        "raw_label",
        "binary_label",
        "recording_id",
        "source_filename",
        "file_sha256",
        "duplicate_hash_group_size",
        "is_exact_duplicate_signal",
        "samples",
        "duration_seconds",
        "windows",
        "input_columns",
        "output_eeg_columns",
        "packet_counter_removed",
        "parser_schema",
        "modulo_step_fraction",
        "included_cross_task_source",
        "included_cross_task_target",
        "included_arithmetic_loso",
        "included_stroop_loso",
        "included_any_v3_analysis",
        "audit_note",
    ]
    output = output[columns].sort_values(["task", "subject", "raw_label"]).reset_index(drop=True)

    subsets = [
        ("deposited_arithmetic_files", output["task"].eq("arithmetic")),
        ("deposited_stroop_files", output["task"].eq("stroop")),
        ("cross_task_arithmetic_source", output["task"].eq("arithmetic") & output["included_cross_task_source"]),
        ("cross_task_stroop_source", output["task"].eq("stroop") & output["included_cross_task_source"]),
        ("cross_task_arithmetic_target", output["task"].eq("arithmetic") & output["included_cross_task_target"]),
        ("cross_task_stroop_target", output["task"].eq("stroop") & output["included_cross_task_target"]),
        ("arithmetic_loso", output["included_arithmetic_loso"]),
        ("stroop_loso", output["included_stroop_loso"]),
    ]
    summary_rows = []
    for scope, mask in subsets:
        selected = output[mask]
        windows = selected["windows"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scope": scope,
                "file_recordings": int(len(selected)),
                "unique_signal_hashes": int(selected["file_sha256"].nunique()),
                "participants": int(selected["subject"].nunique()),
                "total_windows": int(np.sum(windows)),
                "minimum_windows": int(np.min(windows)),
                "q1_windows": float(np.quantile(windows, 0.25)),
                "median_windows": float(np.median(windows)),
                "mean_windows": float(np.mean(windows)),
                "q3_windows": float(np.quantile(windows, 0.75)),
                "maximum_windows": int(np.max(windows)),
                "minimum_duration_seconds": float(selected["duration_seconds"].min()),
                "median_duration_seconds": float(selected["duration_seconds"].median()),
                "maximum_duration_seconds": float(selected["duration_seconds"].max()),
            }
        )
    return output, pd.DataFrame(summary_rows)


def recording_windows_lookup(recording_audit: pd.DataFrame) -> dict[str, int]:
    return {
        str(row.recording_id): int(row.windows)
        for row in recording_audit.itertuples(index=False)
    }


def build_inner_split_audit(recording_audit: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_lookup = recording_windows_lookup(recording_audit)
    rows: list[dict] = []

    direction_tasks = [
        ("arithmetic_to_stroop", "arithmetic", "stroop"),
        ("stroop_to_arithmetic", "stroop", "arithmetic"),
    ]
    for subject in policy["cross_task"]["participants"]:
        for direction, source_task, target_task in direction_tasks:
            source = recording_audit[
                (recording_audit["task"] == source_task)
                & (recording_audit["subject"] == int(subject))
            ].sort_values("raw_label")
            target = recording_audit[
                (recording_audit["task"] == target_task)
                & (recording_audit["subject"] == int(subject))
                & (recording_audit["raw_label"].isin(policy["cross_task"]["target_recordings"]))
            ].sort_values("raw_label")
            for seed in SEEDS:
                folds = balanced_recording_folds(
                    source["raw_label"].to_numpy(),
                    source["recording_id"].to_numpy(),
                    seed,
                )
                for fold_index, fold in enumerate(folds, start=1):
                    train_ids = sorted(source.iloc[fold.train]["recording_id"].astype(str).tolist())
                    validation_ids = sorted(source.iloc[fold.validation]["recording_id"].astype(str).tolist())
                    target_ids = sorted(target["recording_id"].astype(str).tolist())
                    rows.append(
                        {
                            "protocol": "cross_task",
                            "direction": direction,
                            "outer_target_subject": int(subject),
                            "seed_scope": int(seed),
                            "inner_fold": fold_index,
                            "recording_disjoint": True,
                            "participant_disjoint": False,
                            "outer_training_participants": 1,
                            "outer_training_recordings": 4,
                            "inner_training_participants": 1,
                            "inner_training_recordings": len(train_ids),
                            "inner_validation_participants": 1,
                            "inner_validation_recordings": len(validation_ids),
                            "outer_test_participants": 1,
                            "outer_test_recordings": len(target_ids),
                            "inner_training_windows": sum(window_lookup[value] for value in train_ids),
                            "inner_validation_windows": sum(window_lookup[value] for value in validation_ids),
                            "outer_test_windows": sum(window_lookup[value] for value in target_ids),
                            "inner_training_subject_ids": str(subject),
                            "inner_validation_subject_ids": str(subject),
                            "inner_training_recording_ids": joined(train_ids),
                            "inner_validation_recording_ids": joined(validation_ids),
                            "outer_test_recording_ids": joined(target_ids),
                        }
                    )

    for task in ["arithmetic", "stroop"]:
        cohort = [int(value) for value in policy["loso"][task]["participants"]]
        task_rows = recording_audit[
            (recording_audit["task"] == task)
            & (recording_audit["subject"].isin(cohort))
        ]
        for outer_subject in cohort:
            outer_train_subjects = [value for value in cohort if value != outer_subject]
            outer_test = task_rows[task_rows["subject"] == outer_subject]
            for fold_index, validation_subject in enumerate(outer_train_subjects, start=1):
                inner_train_subjects = [
                    value for value in outer_train_subjects if value != validation_subject
                ]
                inner_train = task_rows[task_rows["subject"].isin(inner_train_subjects)]
                inner_validation = task_rows[task_rows["subject"] == validation_subject]
                rows.append(
                    {
                        "protocol": "loso",
                        "direction": task,
                        "outer_target_subject": outer_subject,
                        "seed_scope": "seed_invariant",
                        "inner_fold": fold_index,
                        "recording_disjoint": True,
                        "participant_disjoint": True,
                        "outer_training_participants": len(outer_train_subjects),
                        "outer_training_recordings": int(len(task_rows) - len(outer_test)),
                        "inner_training_participants": len(inner_train_subjects),
                        "inner_training_recordings": int(len(inner_train)),
                        "inner_validation_participants": 1,
                        "inner_validation_recordings": int(len(inner_validation)),
                        "outer_test_participants": 1,
                        "outer_test_recordings": int(len(outer_test)),
                        "inner_training_windows": int(inner_train["windows"].sum()),
                        "inner_validation_windows": int(inner_validation["windows"].sum()),
                        "outer_test_windows": int(outer_test["windows"].sum()),
                        "inner_training_subject_ids": joined(inner_train_subjects),
                        "inner_validation_subject_ids": str(validation_subject),
                        "inner_training_recording_ids": joined(sorted(inner_train["recording_id"].astype(str))),
                        "inner_validation_recording_ids": joined(sorted(inner_validation["recording_id"].astype(str))),
                        "outer_test_recording_ids": joined(sorted(outer_test["recording_id"].astype(str))),
                    }
                )

    audit = pd.DataFrame(rows)
    if len(audit) != 626:
        raise AssertionError(f"内层划分应有626行，实际{len(audit)}行")
    summary_rows = []
    for (protocol, direction), group in audit.groupby(["protocol", "direction"], sort=True):
        summary_rows.append(
            {
                "protocol": protocol,
                "direction": direction,
                "outer_cells": int(group["outer_target_subject"].nunique()),
                "audit_rows": int(len(group)),
                "seed_scopes": int(group["seed_scope"].nunique()),
                "inner_folds_per_outer_seed_min": int(
                    group.groupby(["outer_target_subject", "seed_scope"]).size().min()
                ),
                "inner_folds_per_outer_seed_max": int(
                    group.groupby(["outer_target_subject", "seed_scope"]).size().max()
                ),
                "outer_training_participants": joined(sorted(group["outer_training_participants"].unique())),
                "outer_training_recordings": joined(sorted(group["outer_training_recordings"].unique())),
                "inner_training_participants": joined(sorted(group["inner_training_participants"].unique())),
                "inner_training_recordings": joined(sorted(group["inner_training_recordings"].unique())),
                "inner_validation_participants": joined(sorted(group["inner_validation_participants"].unique())),
                "inner_validation_recordings": joined(sorted(group["inner_validation_recordings"].unique())),
                "outer_test_recordings": joined(sorted(group["outer_test_recordings"].unique())),
                "minimum_inner_training_windows": int(group["inner_training_windows"].min()),
                "median_inner_training_windows": float(group["inner_training_windows"].median()),
                "maximum_inner_training_windows": int(group["inner_training_windows"].max()),
            }
        )
    return audit, pd.DataFrame(summary_rows)


def aggregate_seed_recordings() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "dataset",
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "window_id",
        "method",
        "seed",
        "true_label",
        "p0",
        "p1",
    }
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "method", "seed"]
    for relative_path, methods in RAW_SEED_SOURCES:
        frame = pd.read_csv(HERE / relative_path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{relative_path}缺少列：{missing}")
        frame = frame[frame["method"].isin(methods)].copy()
        if set(frame["method"].unique()) != set(methods):
            raise ValueError(f"{relative_path}未包含预期方法{methods}")
        probabilities = np.clip(frame[["p0", "p1"]].to_numpy(dtype=float), 1e-12, 1.0)
        frame["log_p0"] = np.log(probabilities[:, 0])
        frame["log_p1"] = np.log(probabilities[:, 1])
        grouped = frame.groupby(keys, as_index=False, sort=True).agg(
            true_label=("true_label", "first"),
            label_count=("true_label", "nunique"),
            log_p0=("log_p0", "mean"),
            log_p1=("log_p1", "mean"),
            n_windows=("window_id", "size"),
        )
        if (grouped["label_count"] != 1).any():
            raise ValueError(f"{relative_path}同一recording含多个标签")
        geom = np.exp(grouped[["log_p0", "log_p1"]].to_numpy(dtype=float))
        geom = geom / geom.sum(axis=1, keepdims=True)
        grouped["p0"] = geom[:, 0]
        grouped["p1"] = geom[:, 1]
        grouped["pred_label"] = np.argmax(geom, axis=1).astype(int)
        grouped["correct"] = (
            grouped["pred_label"].to_numpy(dtype=int)
            == grouped["true_label"].to_numpy(dtype=int)
        ).astype(int)
        frames.append(
            grouped[
                keys
                + ["true_label", "p0", "p1", "pred_label", "correct", "n_windows"]
            ]
        )

    recording = pd.concat(frames, ignore_index=True)
    if len(recording) != 7600:
        raise AssertionError(f"五种子recording诊断应有7600行，实际{len(recording)}行")
    counts = recording.groupby("method").size().to_dict()
    if set(counts) != set(SEED_DIAGNOSTIC_METHODS) or any(value != 950 for value in counts.values()):
        raise AssertionError(f"seed诊断方法覆盖异常：{counts}")
    recording["method_display"] = recording["method"].map(METHOD_DISPLAY)
    recording["diagnostic_only_not_statistical_replicate"] = True
    return recording


def participant_accuracy(group: pd.DataFrame) -> tuple[float, float]:
    y_true = group["true_label"].to_numpy(dtype=int)
    y_pred = group["pred_label"].to_numpy(dtype=int)
    sensitivity = np.mean(y_pred[y_true == 1] == 1)
    specificity = np.mean(y_pred[y_true == 0] == 0)
    return float(np.mean(y_true == y_pred)), float((sensitivity + specificity) / 2.0)


def build_seed_performance(recording: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for setting, setting_display, protocol, direction, expected_subjects, recordings_per_subject in SETTING_SPECS:
        selected = recording[recording["protocol"] == protocol]
        if direction is not None:
            selected = selected[selected["direction"] == direction]
        for (method, seed), group in selected.groupby(["method", "seed"], sort=True):
            participant_rows = []
            for subject, subject_group in group.groupby("subject", sort=True):
                if len(subject_group) != recordings_per_subject:
                    raise AssertionError(f"{setting}/{method}/seed{seed}/S{subject} recording数异常")
                accuracy, balanced = participant_accuracy(subject_group)
                participant_rows.append((accuracy, balanced))
            if len(participant_rows) != expected_subjects:
                raise AssertionError(f"{setting}/{method}/seed{seed}参与者数异常")
            values = np.asarray(participant_rows, dtype=float)
            pooled_accuracy, pooled_balanced = participant_accuracy(group)
            rows.append(
                {
                    "setting": setting,
                    "setting_display": setting_display,
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "seed": int(seed),
                    "n_participants": expected_subjects,
                    "n_recordings": int(len(group)),
                    "participant_macro_accuracy": float(values[:, 0].mean()),
                    "participant_macro_balanced_accuracy": float(values[:, 1].mean()),
                    "pooled_recording_accuracy": pooled_accuracy,
                    "pooled_recording_balanced_accuracy": pooled_balanced,
                    "diagnostic_only_not_statistical_replicate": True,
                }
            )
    performance = pd.DataFrame(rows)
    if len(performance) != 200:
        raise AssertionError(f"种子性能应有200行，实际{len(performance)}行")

    summary_rows = []
    for (setting, method), group in performance.groupby(["setting", "method"], sort=True):
        for metric in ["participant_macro_accuracy", "participant_macro_balanced_accuracy"]:
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "setting": setting,
                    "setting_display": group["setting_display"].iloc[0],
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "metric": metric,
                    "seeds": len(values),
                    "seed_mean": float(np.mean(values)),
                    "seed_standard_deviation": float(np.std(values, ddof=1)),
                    "seed_minimum": float(np.min(values)),
                    "seed_maximum": float(np.max(values)),
                    "seed_range": float(np.max(values) - np.min(values)),
                    "interpretation": "model-randomness diagnostic only; seeds are ensemble members",
                }
            )
    return performance, pd.DataFrame(summary_rows)


def epoch_values(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item) for item in str(value).split("|") if str(item).strip()]


def epoch_row(
    model: str,
    protocol: str,
    direction: str,
    subject: int,
    seed: int,
    selected_epoch: int,
    inner_epochs: list[int],
    training_recordings: int,
    test_recordings: int,
    parameter_count: int | None,
) -> dict:
    values = np.asarray(inner_epochs, dtype=float)
    return {
        "model": model,
        "protocol": protocol,
        "direction": direction,
        "subject": int(subject),
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "inner_folds": int(len(values)),
        "inner_epoch_mean": float(np.mean(values)),
        "inner_epoch_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "inner_epoch_minimum": int(np.min(values)),
        "inner_epoch_maximum": int(np.max(values)),
        "inner_epoch_range": int(np.max(values) - np.min(values)),
        "outer_training_recordings": int(training_recordings),
        "outer_test_recordings": int(test_recordings),
        "parameter_count": parameter_count,
    }


def build_epoch_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    core_paths = [
        HERE / "01_core" / "cross_task" / "diagnostics_seed.csv",
        HERE / "01_core" / "loso_arithmetic" / "diagnostics_seed.csv",
        HERE / "01_core" / "loso_stroop" / "diagnostics_seed.csv",
    ]
    for path in core_paths:
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            if row.protocol == "cross_task":
                train_recordings, test_recordings = 4, 3
            elif row.direction == "arithmetic":
                train_recordings, test_recordings = 56, 4
            else:
                train_recordings, test_recordings = 48, 4
            rows.append(
                epoch_row(
                    "Residual MLP",
                    str(row.protocol),
                    str(row.direction),
                    int(row.outer_subject),
                    int(row.seed),
                    int(row.selected_epoch),
                    epoch_values(row.inner_best_epochs),
                    train_recordings,
                    test_recordings,
                    None,
                )
            )

    for model, folder in [("EEGNet", "08_eegnet"), ("EEG-Conformer", "09_eeg_conformer")]:
        paths = sorted((HERE / folder / "checkpoints").rglob("*.diagnostic.json"))
        if len(paths) != 270:
            raise AssertionError(f"{model}应有270个diagnostic JSON，实际{len(paths)}")
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                epoch_row(
                    model,
                    str(data["protocol"]),
                    str(data["direction"]),
                    int(data["subject"]),
                    int(data["seed"]),
                    int(data["selected_epoch"]),
                    [int(value) for value in data["inner_selected_epochs"]],
                    int(data["training_recordings"]),
                    int(data["test_recordings"]),
                    int(data["parameter_count"]),
                )
            )

    audit = pd.DataFrame(rows)
    if len(audit) != 810:
        raise AssertionError(f"epoch审计应有810行，实际{len(audit)}行")

    cell_rows = []
    for keys, group in audit.groupby(["model", "protocol", "direction", "subject"], sort=True):
        values = group["selected_epoch"].to_numpy(dtype=float)
        cell_rows.append(
            {
                "model": keys[0],
                "protocol": keys[1],
                "direction": keys[2],
                "subject": int(keys[3]),
                "seeds": int(len(values)),
                "selected_epoch_mean": float(np.mean(values)),
                "selected_epoch_standard_deviation": float(np.std(values, ddof=1)),
                "selected_epoch_minimum": int(np.min(values)),
                "selected_epoch_maximum": int(np.max(values)),
                "selected_epoch_range": int(np.max(values) - np.min(values)),
                "inner_epoch_range_mean_across_seeds": float(group["inner_epoch_range"].mean()),
            }
        )
    cell = pd.DataFrame(cell_rows)
    if len(cell) != 162:
        raise AssertionError(f"epoch cell稳定性应有162行，实际{len(cell)}")

    summary_rows = []
    for keys, group in cell.groupby(["model", "protocol", "direction"], sort=True):
        summary_rows.append(
            {
                "model": keys[0],
                "protocol": keys[1],
                "direction": keys[2],
                "cells": int(len(group)),
                "median_selected_epoch_range_across_seeds": float(group["selected_epoch_range"].median()),
                "mean_selected_epoch_range_across_seeds": float(group["selected_epoch_range"].mean()),
                "maximum_selected_epoch_range_across_seeds": int(group["selected_epoch_range"].max()),
                "median_within_cell_epoch_sd": float(group["selected_epoch_standard_deviation"].median()),
                "maximum_inner_fold_epoch_range_mean": float(group["inner_epoch_range_mean_across_seeds"].max()),
            }
        )
    return audit, cell, pd.DataFrame(summary_rows)


def build_dasf_stability() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostic = pd.read_csv(HERE / "03_dasf" / "gate_diagnostics_seed.csv")
    final = pd.read_csv(HERE / "03_dasf" / "gate_decisions_final.csv")
    keys = ["protocol", "subject", "direction"]
    final_columns = keys + ["used_fusion", "selected_input_method"]
    merged = diagnostic.merge(
        final[final_columns],
        on=keys,
        suffixes=("_seed", "_final"),
        validate="many_to_one",
    )
    if len(merged) != 270:
        raise AssertionError("DASF逐seed诊断应有270行")
    merged["seed_decision_agrees_with_final"] = (
        merged["used_fusion_seed"].astype(bool) == merged["used_fusion_final"].astype(bool)
    )
    merged["diagnostic_only_not_statistical_replicate"] = True

    cell_rows = []
    for key_values, group in merged.groupby(keys, sort=True):
        fusion_count = int(group["used_fusion_seed"].astype(bool).sum())
        cell_rows.append(
            {
                "protocol": key_values[0],
                "subject": int(key_values[1]),
                "direction": key_values[2],
                "final_used_fusion": bool(group["used_fusion_final"].iloc[0]),
                "final_selected_input_method": group["selected_input_method_final"].iloc[0],
                "seed_fusion_decisions": fusion_count,
                "seed_neural_decisions": 5 - fusion_count,
                "seed_fusion_fraction": fusion_count / 5.0,
                "unanimous_across_five_seeds": fusion_count in {0, 5},
                "seed_agreement_with_final_fraction": float(group["seed_decision_agrees_with_final"].mean()),
                "validation_gain_seed_mean": float(group["validation_gain"].mean()),
                "validation_gain_seed_standard_deviation": float(group["validation_gain"].std(ddof=1)),
                "validation_gain_seed_minimum": float(group["validation_gain"].min()),
                "validation_gain_seed_maximum": float(group["validation_gain"].max()),
                "validation_gain_seed_range": float(group["validation_gain"].max() - group["validation_gain"].min()),
                "effective_margin": float(group["effective_margin"].median()),
            }
        )
    cell = pd.DataFrame(cell_rows)
    if len(cell) != 54:
        raise AssertionError("DASF cell稳定性应有54行")

    summary_rows = []
    for (protocol, direction), group in cell.groupby(["protocol", "direction"], sort=True):
        summary_rows.append(
            {
                "protocol": protocol,
                "direction": direction,
                "cells": int(len(group)),
                "final_fusion_cells": int(group["final_used_fusion"].sum()),
                "unanimous_seed_decision_cells": int(group["unanimous_across_five_seeds"].sum()),
                "unanimous_seed_decision_fraction": float(group["unanimous_across_five_seeds"].mean()),
                "mean_seed_fusion_fraction": float(group["seed_fusion_fraction"].mean()),
                "minimum_seed_agreement_with_final": float(group["seed_agreement_with_final_fraction"].min()),
                "median_validation_gain_seed_range": float(group["validation_gain_seed_range"].median()),
                "maximum_validation_gain_seed_range": float(group["validation_gain_seed_range"].max()),
            }
        )
    return merged, cell, pd.DataFrame(summary_rows)


def build_cbsf_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.read_csv(HERE / "04_cbsf" / "gate_weights.csv")
    if len(weights) != 54:
        raise AssertionError("CB-SF应有54个最终gate cell")
    weights["weight_at_zero"] = np.isclose(weights["weight"], 0.0, atol=1e-12)
    weights["weight_at_frozen_cap_0_8"] = np.isclose(weights["weight"], 0.8, atol=1e-8)
    weights["target_batch_recordings"] = np.where(weights["protocol"] == "cross_task", 3, 4)
    weights["gate_level"] = "one weight after five-seed ensemble per participant-direction cell"
    weights["seed_level_gate_weight_available"] = False

    summary_rows = []
    for (protocol, direction), group in weights.groupby(["protocol", "direction"], sort=True):
        summary_rows.append(
            {
                "protocol": protocol,
                "direction": direction,
                "cells": int(len(group)),
                "weight_mean": float(group["weight"].mean()),
                "weight_standard_deviation": float(group["weight"].std(ddof=1)),
                "weight_minimum": float(group["weight"].min()),
                "weight_median": float(group["weight"].median()),
                "weight_maximum": float(group["weight"].max()),
                "weight_range": float(group["weight"].max() - group["weight"].min()),
                "weights_at_zero": int(group["weight_at_zero"].sum()),
                "weights_at_cap_0_8": int(group["weight_at_frozen_cap_0_8"].sum()),
                "ood_fallback_cells": int(group["ood_fallback"].astype(bool).sum()),
                "predicted_benefit_mean": float(group["predicted_benefit"].mean()),
                "predicted_catastrophe_mean": float(group["predicted_catastrophe"].mean()),
                "meta_training_rows_minimum": int(group["meta_training_rows"].min()),
                "meta_training_rows_maximum": int(group["meta_training_rows"].max()),
                "target_batch_recordings": int(group["target_batch_recordings"].iloc[0]),
            }
        )
    return weights, pd.DataFrame(summary_rows)


def main() -> None:
    preflight, policy, frozen_manifest, metrics_manifest, stats_manifest = validate_inputs()
    recording_audit, window_summary = build_recording_audit(preflight)
    inner_splits, inner_split_summary = build_inner_split_audit(recording_audit, policy)
    seed_recordings = aggregate_seed_recordings()
    seed_performance, seed_variability = build_seed_performance(seed_recordings)
    epoch_audit, epoch_cell, epoch_summary = build_epoch_audit()
    dasf_seed, dasf_cell, dasf_summary = build_dasf_stability()
    cbsf_cell, cbsf_summary = build_cbsf_stability()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "recording_audit": OUTPUT_ROOT / "84_每条recording窗口数哈希与纳入状态.csv",
        "window_summary": OUTPUT_ROOT / "84_recording窗口数分布汇总.csv",
        "inner_splits": OUTPUT_ROOT / "84_每个内层折实际样本量与ID.csv",
        "inner_split_summary": OUTPUT_ROOT / "84_内层划分样本量汇总.csv",
        "seed_recordings": OUTPUT_ROOT / "84_五种子recording级预测_仅稳定性诊断.csv",
        "seed_performance": OUTPUT_ROOT / "84_五种子逐方法性能_仅稳定性诊断.csv",
        "seed_variability": OUTPUT_ROOT / "84_五种子性能波动汇总.csv",
        "epoch_audit": OUTPUT_ROOT / "84_模型选择epoch逐cell与seed.csv",
        "epoch_cell": OUTPUT_ROOT / "84_模型选择epoch逐cell稳定性.csv",
        "epoch_summary": OUTPUT_ROOT / "84_模型选择epoch稳定性汇总.csv",
        "dasf_seed": OUTPUT_ROOT / "84_DASF逐seed门控诊断.csv",
        "dasf_cell": OUTPUT_ROOT / "84_DASF逐cell门控稳定性.csv",
        "dasf_summary": OUTPUT_ROOT / "84_DASF门控稳定性汇总.csv",
        "cbsf_cell": OUTPUT_ROOT / "84_CB-SF逐cell最终权重.csv",
        "cbsf_summary": OUTPUT_ROOT / "84_CB-SF权重稳定性汇总.csv",
    }
    frames = {
        "recording_audit": recording_audit,
        "window_summary": window_summary,
        "inner_splits": inner_splits,
        "inner_split_summary": inner_split_summary,
        "seed_recordings": seed_recordings,
        "seed_performance": seed_performance,
        "seed_variability": seed_variability,
        "epoch_audit": epoch_audit,
        "epoch_cell": epoch_cell,
        "epoch_summary": epoch_summary,
        "dasf_seed": dasf_seed,
        "dasf_cell": dasf_cell,
        "dasf_summary": dasf_summary,
        "cbsf_cell": cbsf_cell,
        "cbsf_summary": cbsf_summary,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False, lineterminator="\n", float_format="%.17g")

    source_paths = [
        POLICY_PATH,
        PREFLIGHT_PATH,
        FROZEN_MANIFEST_PATH,
        METRICS_MANIFEST_PATH,
        STATS_MANIFEST_PATH,
        *[HERE / path for path, _ in RAW_SEED_SOURCES],
        HERE / "03_dasf" / "gate_diagnostics_seed.csv",
        HERE / "03_dasf" / "gate_decisions_final.csv",
        HERE / "04_cbsf" / "gate_weights.csv",
    ]
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "analysis_role": policy["analysis_role"],
        "input_prediction_sha256": frozen_manifest["prediction_sha256"],
        "input_metrics_manifest_sha256": sha256(METRICS_MANIFEST_PATH),
        "input_statistics_manifest_sha256": sha256(STATS_MANIFEST_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": {str(path.relative_to(HERE)).replace("\\", "/"): sha256(path) for path in source_paths},
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "recording_file_rows": int(len(recording_audit)),
        "unique_deposited_signal_hashes": int(recording_audit["file_sha256"].nunique()),
        "inner_split_rows": int(len(inner_splits)),
        "seed_recording_rows": int(len(seed_recordings)),
        "seed_performance_rows": int(len(seed_performance)),
        "epoch_rows": int(len(epoch_audit)),
        "dasf_seed_rows": int(len(dasf_seed)),
        "cbsf_gate_cells": int(len(cbsf_cell)),
        "seed_diagnostic_scope": (
            "seed-level outputs quantify model randomness only; they are not statistical replicates"
        ),
        "final_inference_source": (
            "81 frozen five-seed ensemble; participant is the inferential unit"
        ),
        "cross_task_inner_split_limitation": (
            "each fold has only two source recordings for training and two for validation, "
            "all from the same participant; recording IDs are disjoint but estimates are unstable"
        ),
        "arithmetic_nested_loso_inner_split": (
            "13 training participants/52 recordings and 1 validation participant/4 recordings "
            "inside each 14-participant outer training set"
        ),
        "stroop_nested_loso_inner_split": (
            "11 training participants/44 recordings and 1 validation participant/4 recordings "
            "inside each 12-participant outer training set"
        ),
        "cbsf_seed_order": (
            "five seed probabilities are ensembled first; one gate weight is then estimated per target batch"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "old_route_a_results_read": False,
    }
    manifest_path = OUTPUT_ROOT / "84_样本结构与稳定性审计_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Route A v3 样本结构与稳定性审计已生成")
    print(f"输入预测 SHA-256：{frozen_manifest['prediction_sha256']}")
    print(f"原始文件记录：{len(recording_audit)}；唯一信号哈希：{recording_audit['file_sha256'].nunique()}")
    print(f"内层划分审计行数：{len(inner_splits)}")
    print(f"五种子recording诊断行数：{len(seed_recordings)}")
    print(f"五种子性能单元：{len(seed_performance)}")
    print(f"模型选择epoch诊断行数：{len(epoch_audit)}")
    print(f"DASF逐seed诊断：{len(dasf_seed)}；CB-SF最终gate cell：{len(cbsf_cell)}")
    print("Cross-task内层每折：2条训练recording，2条验证recording，同一参与者但recording-disjoint")
    print("Arithmetic nested LOSO内层：13人/52条训练，1人/4条验证")
    print("Stroop nested LOSO内层：11人/44条训练，1人/4条验证")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
