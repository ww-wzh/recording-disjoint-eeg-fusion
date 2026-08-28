"""编号73：Route A v3 DASF hard gate。

DASF 在这里是次要、探索性的被审计门控基线，不是确认性算法结果。

Cross-task：source 保留四条 recording 完成 balanced recording-disjoint OOF，
但 gate benefit 和 covariance shift 只用 source r1--r3，与三条 target recording 对齐。
LOSO：使用核心模型已经保存的 participant-level inner OOF recording predictions。
最终输出只从 v3 五 seed 中位数 ensemble 的 always_nn/always_fuse 中二选一，
不会读取目标标签来决定 gate。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "route_a"))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    validate_frozen_predictions,
)
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.splits import balanced_recording_folds  # noqa: E402
from route_a_lib.data import load_feature_bundles, normalized_frobenius_shift  # noqa: E402
from route_a_lib.probability import clean_dasf_decision  # noqa: E402


POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
CORE_ROOT = HERE / "01_core"
FROZEN_ROOT = HERE / "02_frozen"
OUTPUT_ROOT = HERE / "03_dasf"
SEEDS = [1335, 1388, 1441, 1494, 1547]
BASE_METHODS = {"always_nn", "always_fuse"}


def _load_core_module():
    path = REVISION_ROOT / "run_corrected_experiments.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_dasf_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bundle_subset(bundle, *, subjects: set[int] | None = None, raw_labels: set[int] | None = None):
    mask = np.ones(len(bundle.labels), dtype=bool)
    if subjects is not None:
        mask &= np.isin(bundle.subjects, sorted(subjects))
    if raw_labels is not None:
        mask &= np.isin(bundle.labels_raw, sorted(raw_labels))
    index = np.flatnonzero(mask)
    if len(index) == 0:
        raise ValueError(f"Filtering removed all rows from {bundle.task}")
    return bundle.subset(index)


def _recording_accuracy(probability: np.ndarray, labels: np.ndarray, recordings: np.ndarray) -> float:
    values = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        label = np.unique(labels[mask])
        if label.size != 1:
            raise ValueError(f"Inconsistent label in {recording}")
        p = np.exp(np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0))
        values.append(int(np.argmax(p) == int(label[0])))
    return float(np.mean(values))


def _checkpoint_path(protocol_name: str, subject: int, direction: str, seed: int) -> Path:
    folder = OUTPUT_ROOT / "checkpoints" / protocol_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"s{int(subject):02d}_{direction}_seed{int(seed)}.json"


def _cross_task_seed_diagnostic(core, source, target, *, subject: int, direction: str, seed: int, protocol: Protocol) -> dict:
    path = _checkpoint_path("cross_task", subject, direction, seed)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if int(saved["seed"]) == int(seed) and int(saved["subject"]) == int(subject) and saved["direction"] == direction:
            return saved
    folds = balanced_recording_folds(source.labels_raw, source.recordings, seed=int(seed))
    result = core.nested_fit_predict(
        source.features,
        source.labels,
        source.recordings,
        target.features,
        folds,
        core.model_config(protocol),
        int(seed),
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    gate_source = source.labels_raw >= 1
    nn_accuracy = _recording_accuracy(result.oof["always_nn"][gate_source], source.labels[gate_source], source.recordings[gate_source])
    fuse_accuracy = _recording_accuracy(result.oof["always_fuse"][gate_source], source.labels[gate_source], source.recordings[gate_source])
    shift = normalized_frobenius_shift(source.covariances[gate_source], target.covariances)
    decision = clean_dasf_decision(nn_accuracy, fuse_accuracy, shift, base_margin=0.02, alpha=0.2, max_margin=0.15)
    saved = {
        "protocol": "cross_task",
        "subject": int(subject),
        "direction": direction,
        "seed": int(seed),
        "validation_accuracy_nn": float(nn_accuracy),
        "validation_accuracy_fuse": float(fuse_accuracy),
        "validation_gain": float(decision.validation_gain),
        "covariance_shift": float(shift),
        "effective_margin": float(decision.effective_margin),
        "used_fusion": bool(decision.use_fusion),
        "selected_input_method": "always_fuse" if decision.use_fusion else "always_nn",
        "gate_source_recordings": 3,
        "inner_folds": len(folds),
        "selected_epoch": int(result.selected_epoch),
    }
    path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return saved


def _loso_seed_diagnostics(feature_bundle, meta_path: Path, *, task: str, cohort: list[int]) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    rows = []
    for outer_subject in cohort:
        for seed in SEEDS:
            part = meta[(meta["outer_subject"].astype(int) == int(outer_subject)) & (meta["seed"].astype(int) == int(seed))]
            if set(part["method"].astype(str)) != BASE_METHODS:
                raise AssertionError(f"Missing LOSO OOF methods for {task} S{outer_subject} seed {seed}")
            accuracies = {}
            for method in ("always_nn", "always_fuse"):
                one = part[part["method"] == method]
                accuracies[method] = float(
                    np.mean((one["p1"].to_numpy() > one["p0"].to_numpy()).astype(int) == one["true_label"].to_numpy())
                )
            target_mask = feature_bundle.subjects == int(outer_subject)
            train_mask = ~target_mask
            shift = normalized_frobenius_shift(feature_bundle.covariances[train_mask], feature_bundle.covariances[target_mask])
            decision = clean_dasf_decision(accuracies["always_nn"], accuracies["always_fuse"], shift, base_margin=0.02, alpha=0.2, max_margin=0.15)
            rows.append({
                "protocol": "loso",
                "task": task,
                "subject": int(outer_subject),
                "direction": task,
                "seed": int(seed),
                "validation_accuracy_nn": accuracies["always_nn"],
                "validation_accuracy_fuse": accuracies["always_fuse"],
                "validation_gain": float(decision.validation_gain),
                "covariance_shift": float(shift),
                "effective_margin": float(decision.effective_margin),
                "used_fusion": bool(decision.use_fusion),
                "selected_input_method": "always_fuse" if decision.use_fusion else "always_nn",
                "gate_source_recordings": int(len(np.unique(feature_bundle.recordings[train_mask]))),
            })
    return pd.DataFrame(rows)


def _final_decisions(seed_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in seed_diagnostics.groupby(["protocol", "subject", "direction"], sort=True):
        med_nn = float(group["validation_accuracy_nn"].median())
        med_fuse = float(group["validation_accuracy_fuse"].median())
        med_shift = float(group["covariance_shift"].median())
        decision = clean_dasf_decision(med_nn, med_fuse, med_shift, base_margin=0.02, alpha=0.2, max_margin=0.15)
        rows.append({
            "protocol": keys[0],
            "subject": int(keys[1]),
            "direction": keys[2],
            "validation_accuracy_nn_median": med_nn,
            "validation_accuracy_fuse_median": med_fuse,
            "validation_gain_median": float(decision.validation_gain),
            "covariance_shift_median": med_shift,
            "effective_margin": float(decision.effective_margin),
            "used_fusion": bool(decision.use_fusion),
            "selected_input_method": "always_fuse" if decision.use_fusion else "always_nn",
            "seed_count": int(group["seed"].nunique()),
        })
    final = pd.DataFrame(rows)
    if len(final) != 26 + 15 + 13:
        raise AssertionError(f"Expected 54 final DASF gate cells, got {len(final)}")
    return final


def _selected_window_rows(window: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in decisions.itertuples(index=False):
        part = window[
            (window["protocol"] == cell.protocol)
            & (window["subject"].astype(int) == int(cell.subject))
            & (window["direction"] == cell.direction)
            & (window["method"] == cell.selected_input_method)
        ].copy()
        if part.empty:
            raise AssertionError(f"Missing selected frozen input for {cell.protocol} S{cell.subject} {cell.direction}")
        part["method"] = "dasf_clean"
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    protocol = Protocol.load(POLICY_PATH)
    core = _load_core_module()
    arithmetic, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))
    source_labels = {0, 1, 2, 3}
    target_labels = {1, 2, 3}
    cross_subjects = set(int(value) for value in policy["cross_task"]["participants"])
    arithmetic_source = _bundle_subset(arithmetic, subjects=cross_subjects, raw_labels=source_labels)
    stroop_source = _bundle_subset(stroop, subjects=cross_subjects, raw_labels=source_labels)
    arithmetic_target = _bundle_subset(arithmetic, subjects=cross_subjects, raw_labels=target_labels)
    stroop_target = _bundle_subset(stroop, subjects=cross_subjects, raw_labels=target_labels)

    diagnostics = []
    for subject in sorted(cross_subjects):
        for source, target, direction in (
            (stroop_source, arithmetic_target, "stroop_to_arithmetic"),
            (arithmetic_source, stroop_target, "arithmetic_to_stroop"),
        ):
            source_cell = source.subset(np.flatnonzero(source.subjects == subject))
            target_cell = target.subset(np.flatnonzero(target.subjects == subject))
            for seed in SEEDS:
                diagnostics.append(_cross_task_seed_diagnostic(core, source_cell, target_cell, subject=subject, direction=direction, seed=seed, protocol=protocol))

    loso_arithmetic = _loso_seed_diagnostics(arithmetic, CORE_ROOT / "loso_arithmetic" / "loso_meta_recording_seed.csv", task="arithmetic", cohort=list(range(1, 16)))
    loso_stroop = _loso_seed_diagnostics(stroop, CORE_ROOT / "loso_stroop" / "loso_meta_recording_seed.csv", task="stroop", cohort=list(range(1, 14)))
    seed_diagnostics = pd.concat([pd.DataFrame(diagnostics), loso_arithmetic, loso_stroop], ignore_index=True)
    decisions = _final_decisions(seed_diagnostics)
    window = pd.read_csv(FROZEN_ROOT / "core_window_ensemble.csv")
    dasf_window = _selected_window_rows(window, decisions)
    final_window = pd.concat([window, dasf_window], ignore_index=True)
    final_recordings = aggregate_recordings(final_window)
    validate_frozen_predictions(final_recordings)
    if len(final_recordings) != 190 * 5:
        raise AssertionError(f"Expected 950 DASF/core recording rows, got {len(final_recordings)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_diagnostics.to_csv(OUTPUT_ROOT / "gate_diagnostics_seed.csv", index=False, lineterminator="\n")
    decisions.to_csv(OUTPUT_ROOT / "gate_decisions_final.csv", index=False, lineterminator="\n")
    final_window.to_csv(OUTPUT_ROOT / "window_predictions.csv", index=False, lineterminator="\n")
    final_recordings.to_csv(OUTPUT_ROOT / "recording_predictions.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "claim_scope": "secondary exploratory hard gate; no formal risk guarantee",
        "gate_order": "five-seed median diagnostics, then one cell decision",
        "cross_task_target_recordings": 3,
        "loso_tasks": {"arithmetic": 15, "stroop": 13},
        "gate_cells": int(len(decisions)),
        "used_fusion_cells": int(decisions["used_fusion"].sum()),
        "recording_rows": int(len(final_recordings)),
        "old_route_a_results_read": False,
    }
    (OUTPUT_ROOT / "73_DASF_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Route A v3 DASF completed")
    print(decisions.groupby(["protocol", "selected_input_method"]).size().to_string())
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
