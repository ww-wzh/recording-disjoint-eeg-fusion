"""直接运行：比较 random-window、recording-disjoint 和 participant-disjoint。

用途：回应导师关于“相同模型和超参数下直接比较三种划分”的意见。
本脚本不需要命令行参数，不修改 Route A 冻结结果。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
PROJECT_ROOT = REVISION_ROOT
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
PROTOCOL_PATH = HERE / "02_冻结方案_导师意见补充实验.json"

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(ROUTE_A_ROOT))

from revision_pipeline.splits import Split, balanced_recording_folds  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402


FOLD_OUTPUT = HERE / "04_结果_三种划分逐折.csv"
SUBJECT_OUTPUT = HERE / "05_结果_三种划分参与者级.csv"
SUMMARY_OUTPUT = HERE / "06_结果_三种划分汇总.csv"
PAIRED_OUTPUT = HERE / "07_结果_三种划分配对差值.csv"
MANIFEST_OUTPUT = HERE / "08_记录_三种划分运行清单.json"

STRATEGY_LABELS = {
    "random_window": "Random-window split",
    "recording_disjoint": "Recording-disjoint split",
    "participant_disjoint": "Participant-disjoint split",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=4000,
            random_state=0,
        ),
    )


def recording_accuracy(
    probability: np.ndarray,
    labels: np.ndarray,
    recordings: np.ndarray,
) -> float:
    outcomes: list[int] = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        recording_labels = np.unique(labels[mask])
        if recording_labels.size != 1:
            raise ValueError(f"Recording {recording!r} contains inconsistent labels")
        aggregate = np.exp(np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0))
        aggregate = aggregate / aggregate.sum()
        outcomes.append(int(int(np.argmax(aggregate)) == int(recording_labels[0])))
    return float(np.mean(outcomes))


def evaluate_split(
    bundle: FeatureBundle,
    split: Split,
    *,
    strategy: str,
    held_out_subject: int,
    split_seed: int | None,
    fold: int,
) -> dict[str, object]:
    model = build_model()
    model.fit(bundle.features[split.train], bundle.labels[split.train])
    probability = model.predict_proba(bundle.features[split.validation])
    prediction = np.argmax(probability, axis=1)

    train_recordings = set(bundle.recordings[split.train].tolist())
    validation_recordings = set(bundle.recordings[split.validation].tolist())
    train_subjects = set(bundle.subjects[split.train].tolist())
    validation_subjects = set(bundle.subjects[split.validation].tolist())
    recording_overlap = train_recordings & validation_recordings
    participant_overlap = train_subjects & validation_subjects

    if strategy != "random_window" and recording_overlap:
        raise AssertionError(f"Unexpected recording overlap: {sorted(recording_overlap)}")
    if strategy == "participant_disjoint" and participant_overlap:
        raise AssertionError(f"Unexpected participant overlap: {sorted(participant_overlap)}")

    return {
        "task": bundle.task,
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "held_out_subject": int(held_out_subject),
        "split_seed": "" if split_seed is None else int(split_seed),
        "fold": int(fold),
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
        "nominal_adjacent_window_raw_overlap": float((8.5 - 0.5) / 8.5),
        "validation_window_accuracy": float(
            accuracy_score(bundle.labels[split.validation], prediction)
        ),
        "validation_window_log_loss": float(
            log_loss(bundle.labels[split.validation], probability, labels=[0, 1])
        ),
        "validation_recording_accuracy": recording_accuracy(
            probability,
            bundle.labels[split.validation],
            bundle.recordings[split.validation],
        ),
    }


def audit_within_participant(
    bundle: FeatureBundle,
    split_seeds: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in sorted(np.unique(bundle.subjects).astype(int).tolist()):
        subset = bundle.subset(np.flatnonzero(bundle.subjects == subject))
        print(f"  {bundle.task}: participant S{subject:02d} random/recording splits")
        for seed in split_seeds:
            random_splitter = StratifiedShuffleSplit(
                n_splits=2,
                test_size=0.12,
                random_state=int(seed),
            )
            for fold, (train, validation) in enumerate(
                random_splitter.split(subset.features, subset.labels),
                start=1,
            ):
                rows.append(
                    evaluate_split(
                        subset,
                        Split(train=np.asarray(train), validation=np.asarray(validation)),
                        strategy="random_window",
                        held_out_subject=subject,
                        split_seed=seed,
                        fold=fold,
                    )
                )

            for fold, split in enumerate(
                balanced_recording_folds(subset.labels_raw, subset.recordings, seed=seed),
                start=1,
            ):
                rows.append(
                    evaluate_split(
                        subset,
                        split,
                        strategy="recording_disjoint",
                        held_out_subject=subject,
                        split_seed=seed,
                        fold=fold,
                    )
                )
    return rows


def audit_participant_disjoint(bundle: FeatureBundle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold, subject in enumerate(sorted(np.unique(bundle.subjects).astype(int).tolist()), start=1):
        print(f"  {bundle.task}: participant S{subject:02d} participant-disjoint LOSO")
        validation = np.flatnonzero(bundle.subjects == subject).astype(np.int64)
        train = np.flatnonzero(bundle.subjects != subject).astype(np.int64)
        rows.append(
            evaluate_split(
                bundle,
                Split(train=train, validation=validation),
                strategy="participant_disjoint",
                held_out_subject=subject,
                split_seed=None,
                fold=fold,
            )
        )
    return rows


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_resamples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Participant bootstrap requires at least two one-dimensional values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(n_resamples), len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def build_subject_results(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby(["task", "strategy", "strategy_label", "held_out_subject"], as_index=False)
        .agg(
            n_evaluations=("fold", "size"),
            mean_recording_overlap_rate=("recording_overlap_rate", "mean"),
            mean_participant_overlap_rate=("participant_overlap_rate", "mean"),
            mean_window_accuracy=("validation_window_accuracy", "mean"),
            mean_recording_accuracy=("validation_recording_accuracy", "mean"),
            mean_window_log_loss=("validation_window_log_loss", "mean"),
        )
        .sort_values(["task", "strategy", "held_out_subject"])
        .reset_index(drop=True)
    )


def build_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, ((task, strategy), group) in enumerate(subjects.groupby(["task", "strategy"])):
        values = group["mean_recording_accuracy"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(values, seed=20260727 + index)
        rows.append(
            {
                "task": task,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "n_participants": int(group["held_out_subject"].nunique()),
                "mean_recording_overlap_rate": float(group["mean_recording_overlap_rate"].mean()),
                "mean_participant_overlap_rate": float(group["mean_participant_overlap_rate"].mean()),
                "mean_window_accuracy": float(group["mean_window_accuracy"].mean()),
                "mean_recording_accuracy": float(values.mean()),
                "recording_accuracy_ci95_low": low,
                "recording_accuracy_ci95_high": high,
                "mean_window_log_loss": float(group["mean_window_log_loss"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "strategy"]).reset_index(drop=True)


def build_paired_results(subjects: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("random_window", "recording_disjoint"),
        ("recording_disjoint", "participant_disjoint"),
        ("random_window", "participant_disjoint"),
    ]
    rows: list[dict[str, object]] = []
    for task_index, (task, task_rows) in enumerate(subjects.groupby("task")):
        pivot = task_rows.pivot(
            index="held_out_subject",
            columns="strategy",
            values="mean_recording_accuracy",
        )
        for comparison_index, (left, right) in enumerate(comparisons):
            paired = pivot[[left, right]].dropna()
            difference = paired[left].to_numpy(dtype=np.float64) - paired[right].to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(
                difference,
                seed=20261727 + task_index * 10 + comparison_index,
            )
            rows.append(
                {
                    "task": task,
                    "left_strategy": left,
                    "right_strategy": right,
                    "difference_definition": "left minus right",
                    "n_paired_participants": int(len(paired)),
                    "mean_paired_difference": float(difference.mean()),
                    "paired_ci95_low": low,
                    "paired_ci95_high": high,
                    "interpretation_warning": (
                        "Participant-disjoint comparisons include participant domain shift and different "
                        "training-set scope; do not attribute the full difference to window leakage."
                        if "participant_disjoint" in {left, right}
                        else "Random-window versus recording-disjoint isolates same-recording contamination most directly."
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    folds: pd.DataFrame,
    subjects: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    folds.to_csv(FOLD_OUTPUT, index=False, lineterminator="\n")
    subjects.to_csv(SUBJECT_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")
    paired.to_csv(PAIRED_OUTPUT, index=False, lineterminator="\n")

    source_files = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "feature_source": PROJECT_ROOT / "eeg_feature_pipeline.py",
        "route_a_protocol": ROUTE_A_ROOT / "protocol_route_a.json",
    }
    output_files = [FOLD_OUTPUT, SUBJECT_OUTPUT, SUMMARY_OUTPUT, PAIRED_OUTPUT]
    manifest = {
        "purpose": "Matched three-split methodological audit requested by the supervisor",
        "analysis_status": "post_hoc_exploratory",
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "fold_rows": int(len(folds)),
        "participant_rows": int(len(subjects)),
        "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256_file(path) for path in output_files},
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    split_seeds = [int(value) for value in protocol["three_split_audit"]["random_window"]["split_seeds"]]

    print("[1/4] 正在读取冻结的 8 通道 EEG 特征。")
    arithmetic, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))

    print("[2/4] 正在运行 Arithmetic 三种划分。")
    rows = audit_within_participant(arithmetic, split_seeds)
    rows.extend(audit_participant_disjoint(arithmetic))

    print("[3/4] 正在运行 Stroop 三种划分。")
    rows.extend(audit_within_participant(stroop, split_seeds))
    rows.extend(audit_participant_disjoint(stroop))

    folds = pd.DataFrame(rows)
    subjects = build_subject_results(folds)
    summary = build_summary(subjects)
    paired = build_paired_results(subjects)
    write_outputs(folds, subjects, summary, paired)

    print("[4/4] 三种数据划分对照完成。")
    display = summary[
        [
            "task",
            "strategy_label",
            "n_participants",
            "mean_recording_overlap_rate",
            "mean_recording_accuracy",
            "recording_accuracy_ci95_low",
            "recording_accuracy_ci95_high",
        ]
    ].copy()
    for column in [
        "mean_recording_overlap_rate",
        "mean_recording_accuracy",
        "recording_accuracy_ci95_low",
        "recording_accuracy_ci95_high",
    ]:
        display[column] = display[column].map(lambda value: f"{100.0 * float(value):.2f}%")
    print(display.to_string(index=False))
    print(f"\n结果文件已写入：{HERE}")
    print("下一步：把控制台输出发给我，并保留 04-08 全部文件。")


if __name__ == "__main__":
    main()

