"""编号76：v3 cross-task source-OOF recording-level stacker。

67 的 cross-task checkpoint 只保存了 target-task 预测，没有保存 source-task
的 OOF 概率。审稿人要求 recording-level OOF stacker，因此本脚本只补算
source OOF 并保存为 recording-level 概率；它不会改写 01_core 或 02_frozen。

脚本按 cell/seed 保存 checkpoint，支持中断后从上次进度继续。运行时间会明显
长于 74、75；请直接在 PyCharm 中运行本文件。
输出目录：06_cross_task_stacker
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
CORE_ROOT = HERE / "01_core"
FROZEN_ROOT = HERE / "02_frozen"
OUTPUT_ROOT = HERE / "06_cross_task_stacker"
CHECKPOINT_ROOT = OUTPUT_ROOT / "source_oof_recordings"
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "route_a"))

from revision_pipeline.aggregation import aggregate_recordings, validate_frozen_predictions  # noqa: E402
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.splits import balanced_recording_folds  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402


METHODS = ("always_nn", "rf", "extra_trees")
SEEDS = (1335, 1388, 1441, 1494, 1547)


def _load_core_module():
    path = REVISION_ROOT / "run_corrected_experiments.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_cross_stack_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _subset(bundle: FeatureBundle, subjects: set[int], raw_labels: set[int]) -> FeatureBundle:
    mask = np.isin(bundle.subjects, sorted(subjects)) & np.isin(bundle.labels_raw, sorted(raw_labels))
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"Empty bundle subset {bundle.task}")
    return bundle.subset(indices)


def _recording_rows(source: FeatureBundle, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        probability = probabilities[method]
        for recording in np.unique(source.recordings):
            mask = source.recordings == recording
            labels = np.unique(source.labels[mask])
            if labels.size != 1:
                raise ValueError(f"Recording {recording} has inconsistent labels")
            p = np.exp(np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0))
            p = p / p.sum()
            rows.append(
                {
                    "recording_id": str(recording),
                    "method": method,
                    "true_label": int(labels[0]),
                    "p0": float(p[0]),
                    "p1": float(p[1]),
                }
            )
    return pd.DataFrame(rows)


def _checkpoint_path(subject: int, direction: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / f"s{int(subject):02d}_{direction}_seed{int(seed)}.csv"


def _flatten_pivot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas' two-level pivot columns to stable plain names."""
    output = frame.copy()
    names = []
    for column in output.columns:
        if isinstance(column, tuple):
            left, right = column
            names.append(str(left) if str(right) == "" else f"{left}__{right}")
        else:
            names.append(str(column))
    output.columns = names
    return output


def _run_one(
    core,
    source: FeatureBundle,
    target: FeatureBundle,
    subject: int,
    direction: str,
    seed: int,
    protocol: Protocol,
    device: torch.device,
) -> pd.DataFrame:
    path = _checkpoint_path(subject, direction, seed)
    if path.exists():
        frame = pd.read_csv(path)
        if len(frame) == 12 and set(frame["method"]) == set(METHODS):
            print(f"[skip] source OOF {path.name}")
            return frame
    folds = balanced_recording_folds(source.labels_raw, source.recordings, seed=int(seed))
    result = core.nested_fit_predict(
        source.features,
        source.labels,
        source.recordings,
        target.features,
        folds,
        core.model_config(protocol),
        int(seed),
        device,
    )
    frame = _recording_rows(source, {method: result.oof[method] for method in METHODS})
    frame["subject"] = int(subject)
    frame["direction"] = direction
    frame["seed"] = int(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"[done] source OOF S{subject:02d} {direction} seed={seed}")
    return frame


def _fit_stack(source_recordings: pd.DataFrame, target_recordings: pd.DataFrame) -> pd.DataFrame:
    median_source = (
        source_recordings.groupby(["recording_id", "method"], as_index=False)[["p0", "p1", "true_label"]]
        .median()
    )
    # Labels are identical across seeds, but median is used only as an integrity check.
    labels = source_recordings.groupby("recording_id", as_index=False)["true_label"].agg(
        lambda x: int(np.unique(x)[0])
    )
    pivot = _flatten_pivot_columns(
        median_source.pivot_table(index="recording_id", columns="method", values=["p0", "p1"], aggfunc="first").reset_index()
    )
    pivot = pivot.merge(labels, on="recording_id", validate="one_to_one")
    required = {f"{name}__{method}" for name in ("p0", "p1") for method in METHODS}
    if not required.issubset(set(pivot.columns)):
        raise AssertionError("Cross-task stacker source OOF is missing a base method")
    x_train = np.column_stack([
        np.log(np.clip(pivot[f"p1__{method}"].to_numpy(), 1e-6, 1.0 - 1e-6) / np.clip(1.0 - pivot[f"p1__{method}"].to_numpy(), 1e-6, 1.0))
        for method in METHODS
    ])
    y_train = pivot["true_label"].to_numpy(dtype=np.int64)
    if np.unique(y_train).size != 2:
        raise AssertionError("Cross-task stacker source OOF has only one class")
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=1335)
    model.fit(x_train, y_train)

    target = target_recordings[target_recordings["method"].isin(METHODS)].copy()
    target_pivot = _flatten_pivot_columns(
        target.pivot_table(index=["dataset", "protocol", "subject", "direction", "recording_id", "true_label"], columns="method", values=["p0", "p1"], aggfunc="first").reset_index()
    )
    if not required.issubset(set(target_pivot.columns)):
        raise AssertionError("Cross-task target frozen predictions are missing a base method")
    x_test = np.column_stack([
        np.log(np.clip(target_pivot[f"p1__{method}"].to_numpy(), 1e-6, 1.0 - 1e-6) / np.clip(1.0 - target_pivot[f"p1__{method}"].to_numpy(), 1e-6, 1.0))
        for method in METHODS
    ])
    probability = model.predict_proba(x_test)
    output = target_pivot[["dataset", "protocol", "subject", "direction", "recording_id", "true_label"]].copy()
    output["method"] = "stack_recording"
    output["p0"] = probability[:, 0]
    output["p1"] = probability[:, 1]
    output["pred_label"] = np.argmax(probability, axis=1).astype(int)
    output["correct"] = (output["pred_label"].to_numpy() == output["true_label"].to_numpy()).astype(int)
    count_keys = ["dataset", "protocol", "subject", "direction", "recording_id"]
    counts = target[target["method"] == "always_nn"][count_keys + ["n_windows"]]
    output = output.merge(counts, on=count_keys, how="left", validate="one_to_one")
    output["ensemble_size"] = 5
    return output


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    protocol = Protocol.load(POLICY_PATH)
    core = _load_core_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arithmetic, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))
    cohort = set(int(value) for value in policy["cross_task"]["participants"])
    source_labels = set(int(value) for value in policy["cross_task"]["source_recordings"])
    target_labels = set(int(value) for value in policy["cross_task"]["target_recordings"])
    pairs = [
        (arithmetic, stroop, "arithmetic_to_stroop"),
        (stroop, arithmetic, "stroop_to_arithmetic"),
    ]
    all_source = []
    for source_all, target_all, direction in pairs:
        source = _subset(source_all, cohort, source_labels)
        target_all_subset = _subset(target_all, cohort, target_labels)
        for subject in sorted(cohort):
            source_cell = source.subset(np.flatnonzero(source.subjects == subject))
            target_cell = target_all_subset.subset(np.flatnonzero(target_all_subset.subjects == subject))
            for seed in SEEDS:
                all_source.append(_run_one(core, source_cell, target_cell, subject, direction, seed, protocol, device))

    source_oof = pd.concat(all_source, ignore_index=True)
    core_recordings = aggregate_recordings(pd.read_csv(FROZEN_ROOT / "core_window_ensemble.csv"))
    generated = []
    for subject in sorted(cohort):
        for direction in ("arithmetic_to_stroop", "stroop_to_arithmetic"):
            source_cell = source_oof[(source_oof["subject"].astype(int) == subject) & (source_oof["direction"] == direction)]
            target_cell = core_recordings[
                (core_recordings["protocol"] == "cross_task")
                & (core_recordings["subject"].astype(int) == subject)
                & (core_recordings["direction"] == direction)
            ]
            generated.append(_fit_stack(source_cell, target_cell))
    stack = pd.concat(generated, ignore_index=True)
    validate_frozen_predictions(stack)
    combined = pd.concat([core_recordings, stack], ignore_index=True)
    validate_frozen_predictions(combined)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_oof.to_csv(OUTPUT_ROOT / "source_oof_recording_seed.csv", index=False, lineterminator="\n")
    stack.to_csv(OUTPUT_ROOT / "cross_task_stack_recording_predictions.csv", index=False, lineterminator="\n")
    combined.to_csv(OUTPUT_ROOT / "cross_task_recording_predictions_with_core.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "passed",
        "method": "stack_recording",
        "protocol": "cross_task",
        "base_methods": list(METHODS),
        "source_recording_oof_rows": int(len(source_oof)),
        "stack_recording_rows": int(len(stack)),
        "cells": int(len(generated)),
        "seeds": list(SEEDS),
        "stacker_fit_unit": "source-task recording-level OOF; four source recordings per participant-direction cell",
        "target_labels_used_for_fit": False,
        "old_route_a_results_read": False,
        "protocol_version": policy["protocol_version"],
    }
    (OUTPUT_ROOT / "76_cross_task_stacker_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Cross-task source OOF and recording-level stacker completed")
    print(stack.groupby("direction", as_index=False)["correct"].mean().to_string(index=False))


if __name__ == "__main__":
    main()
