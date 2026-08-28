"""编号92：运行 Route A v3 预处理常数与标准化敏感性分析。

本分析固定使用91号协议和同一个诊断学习器，比较参考预处理与四个变体。
它不会修改64号协议、81号冻结预测或主论文结果，结果只能作为观察主结果后
增加的探索性敏感性分析。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：19_preprocessing_sensitivity_v3
断点续跑：逐recording特征缓存，并为每个完整变体保存结果checkpoint。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = HERE.parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
PARENT_PROTOCOL = HERE / "64_冻结_RouteA_v3协议.json"
FROZEN_MANIFEST = HERE / "10_all_methods_frozen" / "81_冻结_RouteA_v3全部方法_manifest.json"
PROTOCOL_PATH = HERE / "91_冻结_RouteA_v3预处理敏感性分析协议.json"
BASE_RUNNER_PATH = SUPPLEMENT_ROOT / "36_运行_预处理敏感性分析.py"

OUTPUT_ROOT = HERE / "19_preprocessing_sensitivity_v3"
CACHE_ROOT = OUTPUT_ROOT / "cache_by_variant"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
FOLD_CSV = OUTPUT_ROOT / "92_预处理敏感性逐折结果.csv"
PARTICIPANT_CSV = OUTPUT_ROOT / "92_预处理敏感性逐参与者结果.csv"
SUMMARY_CSV = OUTPUT_ROOT / "92_预处理敏感性总体Bootstrap汇总.csv"
PAIRED_CSV = OUTPUT_ROOT / "92_相对参考流程配对差值与BH校正.csv"
FIGURE_PNG = OUTPUT_ROOT / "92_预处理变体相对参考balanced_accuracy差值.png"
SUPPLEMENT_MD = OUTPUT_ROOT / "92_Supplementary_Preprocessing_Sensitivity.md"
CHINESE_MD = OUTPUT_ROOT / "92_中文解读_预处理敏感性分析.md"
MANIFEST_JSON = OUTPUT_ROOT / "92_预处理敏感性_manifest.json"

ANALYSIS_VERSION = "route-a-v3-preprocessing-sensitivity-r1"
BOOTSTRAP_REPEATS = 10_000
FEATURE_ATOL = 2e-4
FEATURE_RTOL = 2e-4

VARIANT_DISPLAY = {
    "reference": "Reference preprocessing",
    "no_window_zscore": "No within-window z-score",
    "recording_zscore": "Recording-level z-score",
    "no_50hz_attenuation": "No 50-Hz attenuation",
    "neutral_legacy_multipliers": "Neutral legacy multipliers",
}
EVALUATION_DISPLAY = {
    "recording_disjoint": "Within-participant recording-disjoint",
    "participant_disjoint_loso": "Participant-disjoint LOSO",
    "cross_task": "Within-participant cross-task transfer",
}
CONDITION_DISPLAY = {
    "arithmetic": "Arithmetic",
    "stroop": "Stroop",
    "arithmetic_to_stroop": "Arithmetic to Stroop",
    "stroop_to_arithmetic": "Stroop to Arithmetic",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    token = "|".join(str(value) for value in (ANALYSIS_VERSION, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def load_base_runner():
    spec = importlib.util.spec_from_file_location("route_a_v3_preprocessing_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载经过审计的特征变体实现：{BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_lineage() -> tuple[dict, dict, dict]:
    required = [PROTOCOL_PATH, PARENT_PROTOCOL, FROZEN_MANIFEST, BASE_RUNNER_PATH]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少92号输入：{missing}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != ANALYSIS_VERSION:
        raise AssertionError("91号敏感性协议版本不匹配")
    if sha256(PARENT_PROTOCOL) != protocol["parent_protocol_sha256"]:
        raise AssertionError("64号协议SHA-256与91号记录不一致")
    if frozen["prediction_sha256"] != protocol["parent_prediction_sha256"]:
        raise AssertionError("81号冻结预测SHA-256与91号记录不一致")
    if parent["protocol_version"] != protocol["parent_protocol"]:
        raise AssertionError("91号没有正确继承64号Route A v3协议")
    if protocol["main_prediction_freeze_is_unchanged"] is not True:
        raise AssertionError("敏感性分析不得修改主预测冻结")
    feature_source = PROJECT_ROOT / "1111.py"
    if sha256(feature_source) != protocol["shared_feature_pipeline"]["reference_feature_source_sha256"]:
        raise AssertionError("272维参考特征源代码SHA-256与91号记录不一致")
    return protocol, parent, frozen


def build_variant_configs(base, protocol: dict) -> list[Any]:
    variants = []
    for row in protocol["variants"]:
        variants.append(
            base.VariantConfig(
                variant_id=str(row["id"]),
                normalization=str(row["normalization"]),
                line_attenuation_50hz=float(row["line_attenuation_50hz"]),
                frontal_multiplier=float(row["frontal_relative_power_multiplier"]),
                theta_alpha_multiplier=float(row["theta_alpha_relative_power_multiplier"]),
            )
        )
    observed = [config.variant_id for config in variants]
    if observed != list(VARIANT_DISPLAY):
        raise AssertionError(f"91号变体顺序不匹配：{observed}")
    return variants


def config_digest(protocol: dict, config: Any) -> str:
    row = next(row for row in protocol["variants"] if row["id"] == config.variant_id)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "variant": row,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def subset_participants(bundle: Any, participants: list[int]) -> Any:
    indices = np.flatnonzero(np.isin(bundle.subjects, np.asarray(participants, dtype=int)))
    if len(indices) == 0:
        raise ValueError(f"{bundle.task}过滤参与者后没有窗口")
    return bundle.subset(indices)


def cache_path(config: Any, task: str, subject: int, raw_label: int) -> Path:
    return CACHE_ROOT / config.variant_id / f"{task}_s{subject:02d}_r{raw_label}.npz"


def cache_metadata(source: Path, protocol: dict, config: Any, sfreq: float) -> str:
    try:
        relative_source = source.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        relative_source = source.name
    stat = source.stat()
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "variant_digest": config_digest(protocol, config),
        "source_relative_path": relative_source,
        "source_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "sampling_rate_hz": float(sfreq),
        "window_seconds": float(protocol["data"]["window_seconds"]),
        "stride_seconds": float(protocol["data"]["stride_seconds"]),
        "bands_hz": protocol["shared_feature_pipeline"]["bands_hz"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_or_extract_recording(
    base,
    *,
    source: Path,
    eeg: np.ndarray,
    sfreq: float,
    task: str,
    subject: int,
    raw_label: int,
    config: Any,
    protocol: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(config, task, subject, raw_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata(source, protocol, config, sfreq)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["metadata_json"].item()) == metadata:
                    features = cached["features"].astype(np.float32)
                    covariances = cached["covariances"].astype(np.float64)
                    if features.ndim == 2 and features.shape[1] == 272:
                        return features, covariances
        except (KeyError, OSError, ValueError):
            pass
    bands = [tuple(map(float, pair)) for pair in protocol["shared_feature_pipeline"]["bands_hz"]]
    features, covariances = base.extract_features(
        eeg,
        sfreq=float(sfreq),
        win_sec=float(protocol["data"]["window_seconds"]),
        stride_sec=float(protocol["data"]["stride_seconds"]),
        bands=bands,
        config=config,
        device=device,
    )
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        covariances=covariances.astype(np.float64),
        metadata_json=np.asarray(metadata),
    )
    return features.astype(np.float32), covariances.astype(np.float64)


def build_variant_bundles(base, legacy, protocol: dict, config: Any, device: torch.device) -> dict[str, Any]:
    normalized_subjects = legacy.normalize_subjects(legacy.SUBJECTS)
    task_participants = {
        "arithmetic": [int(value) for value in protocol["data"]["arithmetic_participants"]],
        "stroop": [int(value) for value in protocol["data"]["stroop_participants"]],
    }
    bundles: dict[str, Any] = {}
    for task in ["arithmetic", "stroop"]:
        print(f"    {VARIANT_DISPLAY[config.variant_id]}：提取{CONDITION_DISPLAY[task]}特征")
        permitted = set(task_participants[task])
        records = [
            record
            for record in legacy.build_records(normalized_subjects, task=task)
            if int(record.subject_id) in permitted
        ]
        expected_recordings = 60 if task == "arithmetic" else 52
        if len(records) != expected_recordings:
            raise AssertionError(f"{task}应有{expected_recordings}条recording，实际{len(records)}")
        feature_parts = []
        covariance_parts = []
        raw_label_parts = []
        subject_parts = []
        recording_parts = []
        window_id_parts = []
        for record in records:
            source = Path(record.file_path).resolve()
            eeg, sfreq = legacy.load_txt_eeg(
                str(source), default_sfreq=float(protocol["data"]["sampling_rate_hz"])
            )
            features, covariances = load_or_extract_recording(
                base,
                source=source,
                eeg=eeg,
                sfreq=float(sfreq),
                task=task,
                subject=int(record.subject_id),
                raw_label=int(record.label),
                config=config,
                protocol=protocol,
                device=device,
            )
            if not np.isfinite(features).all() or not np.isfinite(covariances).all():
                raise RuntimeError(f"{config.variant_id}/{task}/S{int(record.subject_id):02d}含非有限特征")
            count = len(features)
            rec_id = f"{task}:s{int(record.subject_id):02d}:r{int(record.label)}"
            feature_parts.append(features)
            covariance_parts.append(covariances)
            raw_label_parts.append(np.full(count, int(record.label), dtype=np.int64))
            subject_parts.append(np.full(count, int(record.subject_id), dtype=np.int64))
            recording_parts.append(np.full(count, rec_id, dtype=object))
            window_id_parts.append(np.asarray([f"{rec_id}:w{i:05d}" for i in range(count)], dtype=object))
        labels_raw = np.concatenate(raw_label_parts)
        bundles[task] = base.FeatureBundle(
            features=np.concatenate(feature_parts).astype(np.float32),
            covariances=np.concatenate(covariance_parts).astype(np.float64),
            labels_raw=labels_raw,
            labels=(labels_raw >= 2).astype(np.int64),
            subjects=np.concatenate(subject_parts),
            recordings=np.concatenate(recording_parts),
            window_ids=np.concatenate(window_id_parts),
            task=task,
        )
    return bundles


def build_model(protocol: dict):
    learner = protocol["learner"]
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


def evaluate_indices(base, train_bundle: Any, train: np.ndarray, test_bundle: Any, test: np.ndarray, protocol: dict) -> dict:
    model = build_model(protocol)
    model.fit(train_bundle.features[train], train_bundle.labels[train])
    probability = model.predict_proba(test_bundle.features[test])
    y_true, p_recording, _ = base.aggregate_recordings(
        probability, test_bundle.labels[test], test_bundle.recordings[test]
    )
    predicted = np.argmax(p_recording, axis=1)
    return {
        "train_windows": int(len(train)),
        "train_recordings": int(np.unique(train_bundle.recordings[train]).size),
        "validation_windows": int(len(test)),
        "validation_recordings": int(len(y_true)),
        "recording_accuracy": float(accuracy_score(y_true, predicted)),
        "recording_balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "recording_log_loss": float(log_loss(y_true, p_recording, labels=[0, 1])),
    }


def evaluate_recording_disjoint(base, bundle: Any, config: Any, protocol: dict) -> list[dict]:
    rows = []
    seeds = [int(value) for value in protocol["evaluations"]["recording_disjoint"]["split_seeds"]]
    for subject in sorted(np.unique(bundle.subjects).astype(int).tolist()):
        subset = bundle.subset(np.flatnonzero(bundle.subjects == subject))
        for split_seed in seeds:
            splits = base.balanced_recording_folds(subset.labels_raw, subset.recordings, seed=split_seed)
            for fold, split in enumerate(splits, start=1):
                train = np.asarray(split.train, dtype=np.int64)
                test = np.asarray(split.validation, dtype=np.int64)
                if set(subset.recordings[train]) & set(subset.recordings[test]):
                    raise AssertionError("recording-disjoint内层发生recording重叠")
                rows.append(
                    {
                        "variant_id": config.variant_id,
                        "variant": VARIANT_DISPLAY[config.variant_id],
                        "evaluation": "recording_disjoint",
                        "evaluation_display": EVALUATION_DISPLAY["recording_disjoint"],
                        "condition": bundle.task,
                        "condition_display": CONDITION_DISPLAY[bundle.task],
                        "participant": subject,
                        "split_seed": split_seed,
                        "fold": fold,
                        **evaluate_indices(base, subset, train, subset, test, protocol),
                    }
                )
    return rows


def evaluate_loso(base, bundle: Any, config: Any, protocol: dict) -> list[dict]:
    rows = []
    for fold, subject in enumerate(sorted(np.unique(bundle.subjects).astype(int).tolist()), start=1):
        train = np.flatnonzero(bundle.subjects != subject)
        test = np.flatnonzero(bundle.subjects == subject)
        rows.append(
            {
                "variant_id": config.variant_id,
                "variant": VARIANT_DISPLAY[config.variant_id],
                "evaluation": "participant_disjoint_loso",
                "evaluation_display": EVALUATION_DISPLAY["participant_disjoint_loso"],
                "condition": bundle.task,
                "condition_display": CONDITION_DISPLAY[bundle.task],
                "participant": subject,
                "split_seed": "",
                "fold": fold,
                **evaluate_indices(base, bundle, train, bundle, test, protocol),
            }
        )
    return rows


def evaluate_cross_task(base, source: Any, target: Any, config: Any, condition: str, protocol: dict) -> list[dict]:
    rows = []
    participants = [int(value) for value in protocol["data"]["cross_task_participants"]]
    source_levels = np.asarray(protocol["data"]["cross_task_source_raw_levels"], dtype=int)
    target_levels = np.asarray(protocol["data"]["cross_task_target_raw_levels"], dtype=int)
    for fold, subject in enumerate(participants, start=1):
        train = np.flatnonzero((source.subjects == subject) & np.isin(source.labels_raw, source_levels))
        test = np.flatnonzero((target.subjects == subject) & np.isin(target.labels_raw, target_levels))
        if np.unique(source.recordings[train]).size != 4 or np.unique(target.recordings[test]).size != 3:
            raise AssertionError(f"{condition}/S{subject:02d}不是4条source和3条target recording")
        rows.append(
            {
                "variant_id": config.variant_id,
                "variant": VARIANT_DISPLAY[config.variant_id],
                "evaluation": "cross_task",
                "evaluation_display": EVALUATION_DISPLAY["cross_task"],
                "condition": condition,
                "condition_display": CONDITION_DISPLAY[condition],
                "participant": subject,
                "split_seed": "",
                "fold": fold,
                **evaluate_indices(base, source, train, target, test, protocol),
            }
        )
    return rows


def checkpoint_path(variant_id: str) -> Path:
    return CHECKPOINT_ROOT / f"{variant_id}_fold_results.csv"


def load_valid_checkpoint(config: Any, protocol: dict) -> pd.DataFrame | None:
    path = checkpoint_path(config.variant_id)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError):
        return None
    expected_digest = config_digest(protocol, config)
    valid = bool(
        len(frame) == int(protocol["reproducibility"]["expected_fold_rows_per_variant"])
        and set(frame["variant_id"].astype(str)) == {config.variant_id}
        and set(frame["analysis_digest"].astype(str)) == {expected_digest}
        and set(frame["evaluation"].astype(str)) == set(EVALUATION_DISPLAY)
        and np.isfinite(
            frame[["recording_accuracy", "recording_balanced_accuracy", "recording_log_loss"]].to_numpy(dtype=float)
        ).all()
    )
    return frame if valid else None


def save_checkpoint(rows: list[dict], config: Any, protocol: dict) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["analysis_digest"] = config_digest(protocol, config)
    expected = int(protocol["reproducibility"]["expected_fold_rows_per_variant"])
    if len(frame) != expected:
        raise AssertionError(f"{config.variant_id}应有{expected}行逐折结果，实际{len(frame)}")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checkpoint_path(config.variant_id), index=False, lineterminator="\n")
    return frame


def bootstrap_ci(values: np.ndarray, *seed_parts: object) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(stable_seed(*seed_parts))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def build_participant_results(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby(
            [
                "variant_id", "variant", "evaluation", "evaluation_display",
                "condition", "condition_display", "participant",
            ],
            as_index=False,
        )
        .agg(
            n_evaluations=("fold", "size"),
            mean_recording_balanced_accuracy=("recording_balanced_accuracy", "mean"),
            mean_recording_accuracy=("recording_accuracy", "mean"),
            mean_recording_log_loss=("recording_log_loss", "mean"),
        )
        .sort_values(["evaluation", "condition", "variant_id", "participant"])
        .reset_index(drop=True)
    )


def build_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["variant_id", "variant", "evaluation", "evaluation_display", "condition", "condition_display"]
    for group_keys, group in participants.groupby(keys, sort=True):
        balanced = group["mean_recording_balanced_accuracy"].to_numpy(dtype=float)
        accuracy = group["mean_recording_accuracy"].to_numpy(dtype=float)
        logloss = group["mean_recording_log_loss"].to_numpy(dtype=float)
        bal_low, bal_high = bootstrap_ci(balanced, *group_keys, "balanced")
        acc_low, acc_high = bootstrap_ci(accuracy, *group_keys, "accuracy")
        loss_low, loss_high = bootstrap_ci(logloss, *group_keys, "logloss")
        rows.append(
            {
                **dict(zip(keys, group_keys)),
                "n_participants": int(group["participant"].nunique()),
                "mean_recording_balanced_accuracy": float(balanced.mean()),
                "balanced_accuracy_ci95_low": bal_low,
                "balanced_accuracy_ci95_high": bal_high,
                "mean_recording_accuracy": float(accuracy.mean()),
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "mean_recording_log_loss": float(logloss.mean()),
                "log_loss_ci95_low": loss_low,
                "log_loss_ci95_high": loss_high,
            }
        )
    return pd.DataFrame(rows).sort_values(["evaluation", "condition", "variant_id"]).reset_index(drop=True)


def wilcoxon_p(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=float)
    if np.allclose(difference, 0.0):
        return 1.0
    return float(
        wilcoxon(
            difference,
            zero_method="pratt",
            correction=False,
            alternative="two-sided",
            method="approx",
        ).pvalue
    )


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def build_paired(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (evaluation, condition), group in participants.groupby(["evaluation", "condition"], sort=True):
        reference = group[group["variant_id"] == "reference"].set_index("participant")
        for variant_id in [value for value in VARIANT_DISPLAY if value != "reference"]:
            comparison = group[group["variant_id"] == variant_id].set_index("participant")
            common = reference.index.intersection(comparison.index).sort_values()
            balanced_difference = (
                comparison.loc[common, "mean_recording_balanced_accuracy"].to_numpy(dtype=float)
                - reference.loc[common, "mean_recording_balanced_accuracy"].to_numpy(dtype=float)
            )
            accuracy_difference = (
                comparison.loc[common, "mean_recording_accuracy"].to_numpy(dtype=float)
                - reference.loc[common, "mean_recording_accuracy"].to_numpy(dtype=float)
            )
            logloss_difference = (
                comparison.loc[common, "mean_recording_log_loss"].to_numpy(dtype=float)
                - reference.loc[common, "mean_recording_log_loss"].to_numpy(dtype=float)
            )
            low, high = bootstrap_ci(balanced_difference, evaluation, condition, variant_id, "paired")
            rows.append(
                {
                    "evaluation": evaluation,
                    "evaluation_display": EVALUATION_DISPLAY[evaluation],
                    "condition": condition,
                    "condition_display": CONDITION_DISPLAY[condition],
                    "variant_id": variant_id,
                    "variant": VARIANT_DISPLAY[variant_id],
                    "comparator": VARIANT_DISPLAY["reference"],
                    "n_paired_participants": int(len(common)),
                    "mean_balanced_accuracy_difference": float(balanced_difference.mean()),
                    "paired_balanced_accuracy_ci95_low": low,
                    "paired_balanced_accuracy_ci95_high": high,
                    "participants_improved": int(np.sum(balanced_difference > 0)),
                    "participants_tied": int(np.sum(np.isclose(balanced_difference, 0))),
                    "participants_worsened": int(np.sum(balanced_difference < 0)),
                    "mean_accuracy_difference": float(accuracy_difference.mean()),
                    "mean_log_loss_difference": float(logloss_difference.mean()),
                    "wilcoxon_balanced_accuracy_p_raw": wilcoxon_p(balanced_difference),
                    "analysis_status": "post-hoc reviewer-requested exploratory sensitivity",
                }
            )
    output = pd.DataFrame(rows)
    output["wilcoxon_p_bh_24_comparison_family"] = bh_adjust(
        output["wilcoxon_balanced_accuracy_p_raw"].to_numpy(dtype=float)
    )
    output["significant_after_bh_0_05"] = output["wilcoxon_p_bh_24_comparison_family"] < 0.05
    output["bh_family"] = "all 24 non-reference preprocessing comparisons"
    return output.sort_values(["evaluation", "condition", "variant_id"]).reset_index(drop=True)


def build_figure(paired: pd.DataFrame) -> None:
    variant_ids = [value for value in VARIANT_DISPLAY if value != "reference"]
    conditions = [
        ("recording_disjoint", "arithmetic"),
        ("recording_disjoint", "stroop"),
        ("participant_disjoint_loso", "arithmetic"),
        ("participant_disjoint_loso", "stroop"),
        ("cross_task", "arithmetic_to_stroop"),
        ("cross_task", "stroop_to_arithmetic"),
    ]
    matrix = np.zeros((len(conditions), len(variant_ids)), dtype=float)
    for row_index, (evaluation, condition) in enumerate(conditions):
        for column_index, variant_id in enumerate(variant_ids):
            selected = paired[
                (paired["evaluation"] == evaluation)
                & (paired["condition"] == condition)
                & (paired["variant_id"] == variant_id)
            ]
            if len(selected) != 1:
                raise AssertionError("敏感性图片缺少唯一配对结果")
            matrix[row_index, column_index] = 100.0 * float(
                selected.iloc[0]["mean_balanced_accuracy_difference"]
            )
    limit = max(5.0, float(np.max(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(12.5, 7.2))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) > 0.55 * limit else "black"
            axis.text(column, row, f"{value:+.1f}", ha="center", va="center", color=color)
    axis.set_xticks(range(len(variant_ids)))
    axis.set_xticklabels([VARIANT_DISPLAY[value] for value in variant_ids], rotation=20, ha="right")
    axis.set_yticks(range(len(conditions)))
    axis.set_yticklabels(
        [f"{EVALUATION_DISPLAY[evaluation]}: {CONDITION_DISPLAY[condition]}" for evaluation, condition in conditions]
    )
    axis.set_xlabel("Preprocessing sensitivity variant")
    axis.set_ylabel("Evaluation setting")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Balanced-accuracy difference from reference (pp)")
    figure.tight_layout()
    figure.savefig(FIGURE_PNG, dpi=240, bbox_inches="tight")
    plt.close(figure)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def build_supplement(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    summary_table = summary[
        [
            "evaluation_display", "condition_display", "variant", "n_participants",
            "mean_recording_balanced_accuracy", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high",
        ]
    ].copy()
    for column in ["mean_recording_balanced_accuracy", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high"]:
        summary_table[column] = summary_table[column].map(lambda value: f"{100.0 * float(value):.2f}%")
    paired_table = paired[
        [
            "evaluation_display", "condition_display", "variant", "n_paired_participants",
            "mean_balanced_accuracy_difference", "paired_balanced_accuracy_ci95_low",
            "paired_balanced_accuracy_ci95_high", "wilcoxon_balanced_accuracy_p_raw",
            "wilcoxon_p_bh_24_comparison_family",
        ]
    ].copy()
    for column in ["mean_balanced_accuracy_difference", "paired_balanced_accuracy_ci95_low", "paired_balanced_accuracy_ci95_high"]:
        paired_table[column] = paired_table[column].map(lambda value: f"{100.0 * float(value):+.2f} pp")
    return f"""# Post-hoc preprocessing sensitivity analysis

## Scope

This reviewer-requested analysis was specified after inspection of the primary Route A v3 results and is therefore exploratory. It does not modify the frozen main predictions and no preprocessing variant may replace the reference pipeline because it yields a higher descriptive result.

Five preprocessing definitions were compared with an identical fixed StandardScaler and class-balanced logistic-regression learner. The corrected cohorts were used: Arithmetic S01-S15, Stroop S01-S13, and within-participant bidirectional cross-task transfer for S01-S13. Cross-task source cells contained raw levels 0-3 and target cells contained levels 1-3; the duplicated target natural recording was excluded. Recording-level balanced accuracy was primary. Percentile intervals used 10,000 participant-level bootstrap resamples, and 24 paired Wilcoxon tests were adjusted as one Benjamini-Hochberg family.

The reference pipeline used complete-window channel-wise z-scoring, a 0.03 multiplier for spectral power from 48.5 to 51.5 Hz, a 1.12 multiplier for the first four channel positions, and a joint 1.10 theta/alpha relative-power multiplier. The four variants removed window z-scoring, replaced it with complete-recording z-scoring, removed the 50-Hz spectral multiplier, or jointly neutralized the two legacy emphasis multipliers. These are implementation sensitivities rather than optimized alternatives.

## Descriptive results

{frame_to_markdown(summary_table)}

## Paired differences from the reference pipeline

{frame_to_markdown(paired_table)}

Complete participant-level and fold-level outputs accompany this supplement. Neither zero-phase filtering nor complete-window/complete-recording standardization is a causal sample-by-sample operation; the analysis supports only offline or delayed completed-window use.
"""


def build_chinese(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    significant = int(paired["significant_after_bh_0_05"].sum())
    largest = paired.iloc[np.argmax(np.abs(paired["mean_balanced_accuracy_difference"].to_numpy(dtype=float)))]
    return f"""# Route A v3 预处理敏感性分析中文解读

## 这一步做了什么

固定同一个LogisticRegression诊断模型，只改变预处理中的一个因素，检查窗口z-score、整段recording z-score、50 Hz功率乘数以及1.12/1.10遗留乘数是否明显改变结果。该分析使用当前v3人群：Arithmetic 15人、Stroop 13人、Cross-task 13人且目标只保留r1-r3。

## 怎样理解结果

一共进行了24项“变体相对参考流程”的参与者配对比较，BH校正后显著项为{significant}。绝对平均差值最大的项目是“{largest['evaluation_display']} / {largest['condition_display']} / {largest['variant']}”，balanced accuracy差值为{100.0 * float(largest['mean_balanced_accuracy_difference']):+.2f}个百分点。

即使某个变体数值更高，也不能据此替换主流程，因为这些分析是在主结果出来以后增加的。正确用途是说明结果对若干遗留预处理选择有多敏感，并把不稳定性作为限制公开。

## 论文必须保留的限制

1. 0.03、1.12和1.10都是遗留启发式常数，不是理论推导或v3调参结果。
2. 零相位滤波、完整8.5秒窗口z-score和整段recording z-score都不是逐采样因果处理。
3. 该敏感性分析使用固定诊断学习器，不是重新训练14种主方法，因此不能拿它替代81号主结果。
4. 统计单位始终是参与者；5个划分seed只在参与者内部平均，不能增加样本量。

总体设置数：{len(summary)}；配对比较数：{len(paired)}。
"""


def write_outputs(
    protocol: dict,
    frozen: dict,
    folds: pd.DataFrame,
    participants: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    parity: dict,
    device: torch.device,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLD_CSV, index=False, lineterminator="\n")
    participants.to_csv(PARTICIPANT_CSV, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_CSV, index=False, lineterminator="\n")
    paired.to_csv(PAIRED_CSV, index=False, lineterminator="\n")
    build_figure(paired)
    SUPPLEMENT_MD.write_text(build_supplement(summary, paired), encoding="utf-8")
    CHINESE_MD.write_text(build_chinese(summary, paired), encoding="utf-8")

    expected_cache_count = int(protocol["reproducibility"]["variant_cache_count"])
    cache_counts = {
        variant_id: len(list((CACHE_ROOT / variant_id).glob("*.npz")))
        for variant_id in VARIANT_DISPLAY
    }
    if any(count != expected_cache_count for count in cache_counts.values()):
        raise AssertionError(f"每个变体必须有{expected_cache_count}个缓存：{cache_counts}")
    source_files = {
        "runner": Path(__file__).resolve(),
        "protocol": PROTOCOL_PATH,
        "parent_protocol": PARENT_PROTOCOL,
        "base_feature_variant_implementation": BASE_RUNNER_PATH,
        "legacy_feature_source": PROJECT_ROOT / "1111.py",
        "route_a_data_adapter": ROUTE_A_ROOT / "route_a_lib" / "data.py",
        "split_source": REVISION_ROOT / "revision_pipeline" / "splits.py",
    }
    output_files = [FOLD_CSV, PARTICIPANT_CSV, SUMMARY_CSV, PAIRED_CSV, FIGURE_PNG, SUPPLEMENT_MD, CHINESE_MD]
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "passed",
        "analysis_role": "post_hoc_reviewer_requested_exploratory_sensitivity",
        "main_prediction_freeze_unchanged": True,
        "parent_prediction_sha256": frozen["prediction_sha256"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "primary_metric": "recording-level balanced accuracy",
        "bootstrap": "10000 participant-level percentile resamples",
        "bh_family": "all 24 non-reference comparisons",
        "fold_rows": int(len(folds)),
        "participant_rows": int(len(participants)),
        "summary_rows": int(len(summary)),
        "paired_rows": int(len(paired)),
        "bh_significant_rows": int(paired["significant_after_bh_0_05"].sum()),
        "cache_file_counts": cache_counts,
        "reference_feature_parity": parity,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only",
        },
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256(path) for path in output_files},
        "claim_scope": "descriptive post-hoc sensitivity only; no superiority, non-inferiority, safety, causal or online claim",
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    protocol, _, frozen = verify_lineage()
    base = load_base_runner()
    variants = build_variant_configs(base, protocol)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[1/7] 使用设备：{device}；核对91号协议和81号主预测冻结。")

    legacy = base._load_legacy(PROJECT_ROOT)
    frozen_arithmetic, frozen_stroop = base.load_feature_bundles(PROJECT_ROOT, device)
    frozen_bundles = {
        "arithmetic": subset_participants(
            frozen_arithmetic, [int(value) for value in protocol["data"]["arithmetic_participants"]]
        ),
        "stroop": subset_participants(
            frozen_stroop, [int(value) for value in protocol["data"]["stroop_participants"]]
        ),
    }
    all_frames = []
    parity = None
    for index, config in enumerate(variants, start=1):
        print(f"[2/7] 变体 {index}/{len(variants)}：{VARIANT_DISPLAY[config.variant_id]}")
        bundles = build_variant_bundles(base, legacy, protocol, config, device)
        if config.variant_id == "reference":
            parity = base.check_reference_parity(bundles, frozen_bundles)
            print("      纯8通道参考特征一致性校验通过。")
        checkpoint = load_valid_checkpoint(config, protocol)
        if checkpoint is not None:
            print("      已有完整同协议checkpoint，跳过模型重算。")
            all_frames.append(checkpoint)
            continue
        rows = []
        print("      运行同一模型的recording-disjoint、LOSO和双向cross-task诊断。")
        for task in ["arithmetic", "stroop"]:
            rows.extend(evaluate_recording_disjoint(base, bundles[task], config, protocol))
            rows.extend(evaluate_loso(base, bundles[task], config, protocol))
        rows.extend(
            evaluate_cross_task(
                base, bundles["arithmetic"], bundles["stroop"], config,
                "arithmetic_to_stroop", protocol,
            )
        )
        rows.extend(
            evaluate_cross_task(
                base, bundles["stroop"], bundles["arithmetic"], config,
                "stroop_to_arithmetic", protocol,
            )
        )
        all_frames.append(save_checkpoint(rows, config, protocol))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if parity is None:
        raise AssertionError("没有完成reference纯8通道特征一致性检查")
    print("[3/7] 合并逐折checkpoint并核对样本结构。")
    folds = pd.concat(all_frames, ignore_index=True)
    expected_folds = int(protocol["reproducibility"]["expected_total_fold_rows"])
    if len(folds) != expected_folds:
        raise AssertionError(f"应有{expected_folds}行逐折结果，实际{len(folds)}")

    print("[4/7] 生成参与者级结果和10,000次percentile bootstrap区间。")
    participants = build_participant_results(folds)
    summary = build_summary(participants)
    if len(participants) != int(protocol["reproducibility"]["expected_participant_rows"]):
        raise AssertionError("参与者级敏感性结果不完整")
    if len(summary) != int(protocol["reproducibility"]["expected_summary_rows"]):
        raise AssertionError("敏感性总体汇总不完整")

    print("[5/7] 计算24项配对差值、Wilcoxon检验和单一BH校正。")
    paired = build_paired(participants)
    if len(paired) != int(protocol["reproducibility"]["expected_paired_comparisons"]):
        raise AssertionError("敏感性配对比较数量不正确")

    print("[6/7] 写入表格、图片、补充材料和SHA-256清单。")
    write_outputs(protocol, frozen, folds, participants, summary, paired, parity, device)

    print("[7/7] Route A v3预处理敏感性分析完成。")
    display = summary[
        [
            "evaluation_display", "condition_display", "variant", "n_participants",
            "mean_recording_balanced_accuracy", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high",
        ]
    ].copy()
    for column in ["mean_recording_balanced_accuracy", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high"]:
        display[column] = display[column].map(lambda value: f"{100.0 * float(value):.2f}%")
    print(display.to_string(index=False))
    print(f"24项BH校正后显著项：{int(paired['significant_after_bh_0_05'].sum())}")
    print("主预测冻结未改变；本结果仅为post-hoc exploratory sensitivity。")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
