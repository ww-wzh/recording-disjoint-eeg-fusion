"""编号74：Route A v3 CB-SF exploratory soft gate。

本脚本只读取已经冻结的 v3 核心预测和 nested LOSO 元预测，不重新训练
67--69 的基础模型。CB-SF 的 gate 在 participant/方向 cell 层面拟合，
然后对该 cell 的全部窗口使用同一个软融合权重。输出使用 recording-level
几何平均，便于后续统一统计。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：04_cbsf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
CORE_ROOT = HERE / "01_core"
FROZEN_ROOT = HERE / "02_frozen"
DASF_ROOT = HERE / "03_dasf"
OUTPUT_ROOT = HERE / "04_cbsf"
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "route_a"))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.risk_gate import (  # noqa: E402
    GATE_FEATURES,
    blend_probabilities,
    fit_predict_gate_robust,
    robust_feature_shift,
)
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402


def _subject_rows(bundle: FeatureBundle, subject: int, raw_labels: set[int] | None = None) -> np.ndarray:
    mask = bundle.subjects == int(subject)
    if raw_labels is not None:
        mask &= np.isin(bundle.labels_raw, sorted(raw_labels))
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"No rows for {bundle.task} subject {subject} with labels {raw_labels}")
    return rows


def _entropy(probability: np.ndarray) -> float:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    return float(np.mean(-np.sum(values * np.log(values), axis=1)))


def _cell_diagnostics(window: pd.DataFrame, protocol: str, subject: int, direction: str) -> dict[str, object]:
    cell = window[
        (window["protocol"] == protocol)
        & (window["subject"].astype(int) == int(subject))
        & (window["direction"] == direction)
    ]
    nn = cell[cell["method"] == "always_nn"]
    fuse = cell[cell["method"] == "always_fuse"]
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id"]
    if nn.empty or fuse.empty:
        raise ValueError(f"Missing NN/fusion predictions for {protocol} S{subject} {direction}")
    merged = nn[keys + ["p0", "p1"]].merge(
        fuse[keys + ["p0", "p1"]],
        on=keys,
        suffixes=("_nn", "_fuse"),
        validate="one_to_one",
    )
    nn_probability = merged[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64)
    fuse_probability = merged[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64)
    return {
        "protocol": protocol,
        "subject": int(subject),
        "direction": direction,
        "disagreement": float(np.mean(np.abs(nn_probability[:, 1] - fuse_probability[:, 1]))),
        "entropy_nn": _entropy(nn_probability),
        "entropy_fuse": _entropy(fuse_probability),
        "confidence_nn": float(np.mean(np.max(nn_probability, axis=1))),
    }


def _robust_cross_shift(
    direction: str,
    subject: int,
    arithmetic: FeatureBundle,
    stroop: FeatureBundle,
) -> float:
    source, target = (
        (arithmetic, stroop)
        if direction == "arithmetic_to_stroop"
        else (stroop, arithmetic)
    )
    source_rows = _subject_rows(source, subject, {1, 2, 3})
    target_rows = _subject_rows(target, subject, {1, 2, 3})
    return robust_feature_shift(source.features[source_rows], target.features[target_rows])


def _robust_loso_shift(
    bundle: FeatureBundle,
    target_subject: int,
    reference_exclusion: int | None = None,
    cohort: set[int] | None = None,
) -> float:
    subjects = set(int(value) for value in np.unique(bundle.subjects))
    if cohort is not None:
        subjects &= set(cohort)
    reference_subjects = subjects - {int(target_subject)}
    if reference_exclusion is not None:
        reference_subjects.discard(int(reference_exclusion))
    reference = np.isin(bundle.subjects, sorted(reference_subjects))
    target = bundle.subjects == int(target_subject)
    if not reference.any() or not target.any():
        raise ValueError("LOSO feature-shift reference or target is empty")
    return robust_feature_shift(bundle.features[reference], bundle.features[target])


def _cross_gate_frames(
    window: pd.DataFrame,
    dasf_diagnostics: pd.DataFrame,
    arithmetic: FeatureBundle,
    stroop: FeatureBundle,
    subjects: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_rows = []
    for subject in subjects:
        for direction in ("arithmetic_to_stroop", "stroop_to_arithmetic"):
            row = _cell_diagnostics(window, "cross_task", subject, direction)
            row["feature_shift"] = _robust_cross_shift(direction, subject, arithmetic, stroop)
            core_rows.append(row)
    target = pd.DataFrame(core_rows)
    gain = (
        dasf_diagnostics.groupby(["protocol", "subject", "direction"], as_index=False)[
            "validation_gain"
        ]
        .median()
        .rename(columns={"validation_gain": "benefit"})
    )
    target = target.merge(gain, on=["protocol", "subject", "direction"], validate="one_to_one")
    required = set(GATE_FEATURES + ["benefit", "protocol", "subject", "direction"])
    if required - set(target.columns):
        raise AssertionError(f"Cross-task gate frame is missing {sorted(required - set(target.columns))}")
    return target, target.copy()


def _loso_gate_frames(
    window: pd.DataFrame,
    bundle: FeatureBundle,
    core_root: Path,
    task: str,
    cohort: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction = task
    meta_path = core_root / f"loso_{task}" / "loso_meta_seed.csv"
    meta = pd.read_csv(meta_path)
    diagnostics_seed = pd.read_csv(core_root / f"loso_{task}" / "diagnostics_seed.csv")
    required = {"outer_subject", "meta_subject", "seed", *GATE_FEATURES, "benefit"}
    if required - set(meta.columns):
        raise AssertionError(f"Missing LOSO meta gate columns for {task}: {sorted(required - set(meta.columns))}")

    recalculated = meta.copy()
    robust_values = []
    cohort_set = set(int(value) for value in cohort)
    for row in recalculated.itertuples(index=False):
        robust_values.append(
            _robust_loso_shift(
                bundle,
                int(row.meta_subject),
                reference_exclusion=int(row.outer_subject),
                cohort=cohort_set,
            )
        )
    recalculated["feature_shift_original"] = recalculated["feature_shift"].astype(float)
    recalculated["feature_shift"] = robust_values
    training = (
        recalculated.groupby(["outer_subject", "meta_subject"], as_index=False)[GATE_FEATURES + ["benefit", "feature_shift_original"]]
        .median()
        .sort_values(["outer_subject", "meta_subject"])
        .reset_index(drop=True)
    )
    expected_rows = len(cohort) * (len(cohort) - 1)
    if len(training) != expected_rows:
        raise AssertionError(f"{task} gate training should contain {expected_rows} rows, got {len(training)}")

    target_rows = []
    for subject in cohort:
        row = _cell_diagnostics(window, "loso", subject, direction)
        row["feature_shift"] = _robust_loso_shift(bundle, subject, cohort=cohort_set)
        row["feature_shift_original"] = float(
            diagnostics_seed[diagnostics_seed["outer_subject"].astype(int) == int(subject)][
                "feature_shift"
            ].median()
        )
        target_rows.append(row)
    target = pd.DataFrame(target_rows)
    return training, target


def _blend_cell(window: pd.DataFrame, target: pd.Series, weight: float) -> pd.DataFrame:
    cell = window[
        (window["protocol"] == str(target["protocol"]))
        & (window["subject"].astype(int) == int(target["subject"]))
        & (window["direction"] == str(target["direction"]))
    ]
    nn = cell[cell["method"] == "always_nn"]
    fuse = cell[cell["method"] == "always_fuse"]
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "true_label"]
    paired = nn[keys + ["p0", "p1"]].merge(
        fuse[keys + ["p0", "p1"]],
        on=keys,
        suffixes=("_nn", "_fuse"),
        validate="one_to_one",
    )
    blended = blend_probabilities(
        paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64),
        paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64),
        float(weight),
    )
    output = paired[keys].copy()
    output["method"] = "cbsf"
    output["p0"] = blended[:, 0]
    output["p1"] = blended[:, 1]
    output["ensemble_size"] = 5
    return output[
        ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method", "true_label", "p0", "p1", "ensemble_size"]
    ]


def _fit_one(
    target: pd.Series,
    training: pd.DataFrame,
    window: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if int(target["subject"]) in set(training.get("meta_subject", pd.Series(dtype=int)).astype(int)):
        raise AssertionError("Target participant leaked into CB-SF gate training")
    gate = fit_predict_gate_robust(
        training,
        target,
        catastrophe_threshold=float(config["catastrophe_threshold"]),
        benefit_alpha=float(config["benefit_alpha"]),
        catastrophe_c=float(config["catastrophe_C"]),
        risk_penalty=float(config["risk_penalty"]),
        temperature=float(config["temperature"]),
        max_weight=float(config["max_weight"]),
        feature_z_cap=5.0,
        ood_z_threshold=10.0,
    )
    rows = _blend_cell(window, target, gate.weight)
    diagnostic = {
        "protocol": str(target["protocol"]),
        "subject": int(target["subject"]),
        "direction": str(target["direction"]),
        "weight": float(gate.weight),
        "predicted_benefit": float(gate.predicted_benefit),
        "predicted_benefit_raw": float(gate.predicted_benefit_raw),
        "predicted_catastrophe": float(gate.predicted_catastrophe),
        "ood_fallback": bool(gate.ood_fallback),
        "max_abs_robust_z": float(gate.max_abs_robust_z),
        "feature_shift": float(target["feature_shift"]),
        "meta_training_rows": int(len(training)),
        "meta_training_subjects": int(training["meta_subject"].nunique()) if "meta_subject" in training else int(len(training)),
    }
    return rows, diagnostic


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    seeds = [int(value) for value in policy["seeds"]]
    if seeds != [1335, 1388, 1441, 1494, 1547]:
        raise AssertionError(f"Unexpected frozen seed list: {seeds}")
    config = policy["models"]["risk_gate"]
    arithmetic, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))
    window = pd.read_csv(FROZEN_ROOT / "core_window_ensemble.csv")
    if set(window["method"].astype(str)) != {"always_nn", "always_fuse", "rf", "extra_trees"}:
        raise AssertionError("Core frozen window file has an unexpected method set")

    crosstarget, crosstraining = _cross_gate_frames(
        window,
        pd.read_csv(DASF_ROOT / "gate_diagnostics_seed.csv"),
        arithmetic,
        stroop,
        [int(value) for value in policy["cross_task"]["participants"]],
    )
    generated = []
    weights = []
    for target in crosstarget.sort_values(["subject", "direction"]).to_dict("records"):
        target_series = pd.Series(target)
        training = crosstraining[crosstraining["subject"].astype(int) != int(target["subject"])].copy()
        rows, weight = _fit_one(target_series, training, window, config)
        generated.append(rows)
        weights.append(weight)

    arithmetic_training, arithmetic_targets = _loso_gate_frames(
        window, arithmetic, CORE_ROOT, "arithmetic", list(range(1, 16))
    )
    for target in arithmetic_targets.sort_values("subject").to_dict("records"):
        target_series = pd.Series(target)
        training = arithmetic_training[arithmetic_training["outer_subject"].astype(int) == int(target["subject"])].copy()
        rows, weight = _fit_one(target_series, training, window, config)
        generated.append(rows)
        weights.append(weight)

    stroop_training, stroop_targets = _loso_gate_frames(
        window, stroop, CORE_ROOT, "stroop", list(range(1, 14))
    )
    for target in stroop_targets.sort_values("subject").to_dict("records"):
        target_series = pd.Series(target)
        training = stroop_training[stroop_training["outer_subject"].astype(int) == int(target["subject"])].copy()
        rows, weight = _fit_one(target_series, training, window, config)
        generated.append(rows)
        weights.append(weight)

    cbsf_window = pd.concat(generated, ignore_index=True)
    cbsf_recordings = aggregate_recordings(cbsf_window)
    validate_frozen_predictions(cbsf_recordings)
    if len(cbsf_recordings) != 190:
        raise AssertionError(f"Expected 190 CB-SF recording rows, got {len(cbsf_recordings)}")
    combined_window = pd.concat([window, cbsf_window], ignore_index=True)
    combined_recordings = aggregate_recordings(combined_window)
    validate_frozen_predictions(combined_recordings)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cbsf_window.to_csv(OUTPUT_ROOT / "window_predictions.csv", index=False, lineterminator="\n")
    cbsf_recordings.to_csv(OUTPUT_ROOT / "recording_predictions.csv", index=False, lineterminator="\n")
    combined_recordings.to_csv(OUTPUT_ROOT / "recording_predictions_with_core.csv", index=False, lineterminator="\n")
    pd.DataFrame(weights).to_csv(OUTPUT_ROOT / "gate_weights.csv", index=False, lineterminator="\n")
    crosstraining.to_csv(OUTPUT_ROOT / "gate_training_rows_cross_task.csv", index=False, lineterminator="\n")
    arithmetic_training.to_csv(OUTPUT_ROOT / "gate_training_rows_loso_arithmetic.csv", index=False, lineterminator="\n")
    stroop_training.to_csv(OUTPUT_ROOT / "gate_training_rows_loso_stroop.csv", index=False, lineterminator="\n")
    pd.concat([crosstarget, arithmetic_targets, stroop_targets], ignore_index=True).to_csv(
        OUTPUT_ROOT / "target_diagnostics.csv", index=False, lineterminator="\n"
    )
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "claim_scope": "exploratory conditional soft fusion; no formal risk guarantee",
        "method": "cbsf",
        "gate_model": "robust ridge benefit plus class-balanced logistic catastrophe model",
        "feature_z_cap": 5.0,
        "ood_z_threshold": 10.0,
        "seed_aggregation": "frozen five-seed median before gate; seeds are not statistical units",
        "evaluation_unit": "recording",
        "statistical_unit": "participant",
        "cross_task_target_batch_recordings": 3,
        "cross_task_gate_cells": int(len(crosstarget)),
        "arithmetic_loso_gate_cells": int(len(arithmetic_targets)),
        "stroop_loso_gate_cells": int(len(stroop_targets)),
        "recording_rows": int(len(cbsf_recordings)),
        "ood_fallback_rows": int(pd.DataFrame(weights)["ood_fallback"].sum()),
        "core_prediction_sha256": sha256_file(FROZEN_ROOT / "core_window_ensemble.csv"),
        "dasf_diagnostic_sha256": sha256_file(DASF_ROOT / "gate_diagnostics_seed.csv"),
        "old_route_a_results_read": False,
    }
    (OUTPUT_ROOT / "74_CB-SF_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CB-SF v3 completed")
    print(f"Output: {OUTPUT_ROOT}")
    print(cbsf_recordings.groupby(["protocol", "direction"], as_index=False)["correct"].mean().to_string(index=False))


if __name__ == "__main__":
    main()
