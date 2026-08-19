"""ROBUSTNESS ANALYSIS: bounded CB-SF diagnostics and gate extrapolation.

This script reuses the completed formal Route A predictions. It does not retrain
the base models and never overwrites the original CB-SF outputs. Results are
written to ``route_a/results/cbsf_robustness_r1``.

Run this file directly in PyCharm. No command-line parameters are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
REPO_ROOT = REVISION_ROOT
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from build_outputs import paired_comparisons, subject_metrics, summary_table, tail_descriptives  # noqa: E402
from freeze_predictions import (  # noqa: E402
    cross_task_gate_rows,
    final_diagnostics,
    paired_window_rows,
)
from revision_pipeline.aggregation import (  # noqa: E402
    RAW_KEYS,
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.risk_gate import (  # noqa: E402
    blend_probabilities,
    fit_predict_gate_robust,
    robust_feature_shift,
)
from revision_pipeline.protocol import Protocol  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402


PROTOCOL_PATH = HERE / "protocol_cbsf_robustness.json"
PARENT_RAW_PATH = REVISION_ROOT / "results" / "raw_seed_predictions.csv"
PARENT_DIAGNOSTIC_PATH = REVISION_ROOT / "results" / "diagnostics_seed.csv"
PARENT_RECORDING_PATH = REVISION_ROOT / "frozen" / "predictions_recording.csv"
LOSO_ROOT = HERE / "results" / "cbsf_loso"
OUTPUT_ROOT = HERE / "results" / "cbsf_robustness_r1"
SEEDS = [1335, 1388, 1441, 1494, 1547]


def _verify_inputs(protocol: dict) -> None:
    observed = {
        "parent_corrected_protocol_sha256": Protocol.load(REVISION_ROOT / "protocol.json").digest,
        "parent_recording_prediction_sha256": sha256_file(PARENT_RECORDING_PATH),
        "parent_raw_seed_prediction_sha256": sha256_file(PARENT_RAW_PATH),
        "source_cbsf_loso_recording_prediction_sha256": sha256_file(
            LOSO_ROOT / "recording_predictions.csv"
        ),
    }
    for key, digest in observed.items():
        if protocol[key] != digest:
            raise RuntimeError(f"Robustness input changed: {key} expected={protocol[key]} observed={digest}")


def _subject_rows(bundle: FeatureBundle, subject: int) -> np.ndarray:
    rows = np.flatnonzero(bundle.subjects == int(subject))
    if len(rows) == 0:
        raise ValueError(f"Missing subject {subject} in {bundle.task}")
    return rows


def _cross_task_robust_diagnostics(
    diagnostics: pd.DataFrame,
    arithmetic: FeatureBundle,
    stroop: FeatureBundle,
    shift_config: dict,
) -> pd.DataFrame:
    output = diagnostics[diagnostics["protocol"] == "cross_task"].copy()
    bundles = {
        "arithmetic_to_stroop": (arithmetic, stroop),
        "stroop_to_arithmetic": (stroop, arithmetic),
    }
    shifts = []
    for row in output.itertuples(index=False):
        source, target = bundles[str(row.direction)]
        source_rows = _subject_rows(source, int(row.subject))
        target_rows = _subject_rows(target, int(row.subject))
        shifts.append(
            robust_feature_shift(
                source.features[source_rows],
                target.features[target_rows],
                per_feature_cap=float(shift_config["per_feature_cap"]),
                scale_epsilon=float(shift_config["scale_epsilon"]),
            )
        )
    output["feature_shift_original"] = output["feature_shift"].astype(float)
    output["feature_shift"] = shifts
    return output


def _loso_meta_training_robust_shift(
    training: pd.DataFrame,
    arithmetic: FeatureBundle,
    shift_config: dict,
) -> pd.DataFrame:
    output = training.copy()
    output["feature_shift_original"] = output["feature_shift"].astype(float)
    shifts = []
    for row in output.itertuples(index=False):
        outer_subject = int(row.outer_subject)
        meta_subject = int(row.meta_subject)
        reference = (arithmetic.subjects != outer_subject) & (arithmetic.subjects != meta_subject)
        target = (arithmetic.subjects != outer_subject) & (arithmetic.subjects == meta_subject)
        shifts.append(
            robust_feature_shift(
                arithmetic.features[reference],
                arithmetic.features[target],
                per_feature_cap=float(shift_config["per_feature_cap"]),
                scale_epsilon=float(shift_config["scale_epsilon"]),
            )
        )
    output["feature_shift"] = shifts
    return output


def _loso_target_robust_diagnostics(
    diagnostics: pd.DataFrame,
    arithmetic: FeatureBundle,
    shift_config: dict,
) -> pd.DataFrame:
    output = diagnostics[diagnostics["protocol"] == "loso"].copy()
    shifts = []
    for row in output.itertuples(index=False):
        subject = int(row.subject)
        reference = arithmetic.subjects != subject
        target = arithmetic.subjects == subject
        shifts.append(
            robust_feature_shift(
                arithmetic.features[reference],
                arithmetic.features[target],
                per_feature_cap=float(shift_config["per_feature_cap"]),
                scale_epsilon=float(shift_config["scale_epsilon"]),
            )
        )
    output["feature_shift_original"] = output["feature_shift"].astype(float)
    output["feature_shift"] = shifts
    return output


def _gate_one_target(
    target: pd.Series,
    training: pd.DataFrame,
    window_ensemble: pd.DataFrame,
    config: dict,
) -> tuple[list[dict], dict]:
    gate = fit_predict_gate_robust(
        training,
        target,
        catastrophe_threshold=float(config["catastrophe_threshold"]),
        benefit_alpha=float(config["benefit_alpha"]),
        catastrophe_c=float(config["catastrophe_C"]),
        risk_penalty=float(config["risk_penalty"]),
        temperature=float(config["temperature"]),
        max_weight=float(config["max_weight"]),
        feature_z_cap=float(config["feature_z_cap"]),
        ood_z_threshold=float(config["ood_z_threshold"]),
    )
    paired = paired_window_rows(
        window_ensemble,
        str(target["protocol"]),
        int(target["subject"]),
        str(target["direction"]),
    )
    blended = blend_probabilities(
        paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64),
        paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64),
        gate.weight,
    )
    rows = []
    for index, row in paired.reset_index(drop=True).iterrows():
        rows.append(
            {
                **{key: row[key] for key in RAW_KEYS if key != "method"},
                "method": "risk_aware_robust",
                "true_label": int(row["true_label"]),
                "p0": float(blended[index, 0]),
                "p1": float(blended[index, 1]),
                "ensemble_size": 5,
            }
        )
    diagnostic = {
        "protocol": str(target["protocol"]),
        "subject": int(target["subject"]),
        "direction": str(target["direction"]),
        "weight": gate.weight,
        "predicted_benefit": gate.predicted_benefit,
        "predicted_benefit_raw": gate.predicted_benefit_raw,
        "predicted_catastrophe": gate.predicted_catastrophe,
        "ood_fallback": gate.ood_fallback,
        "max_abs_robust_z": gate.max_abs_robust_z,
        "feature_shift_original": float(target["feature_shift_original"]),
        "feature_shift_robust": float(target["feature_shift"]),
        "meta_training_rows": int(len(training)),
        "meta_training_participants": int(
            training["meta_subject"].nunique() if "meta_subject" in training else training["subject"].nunique()
        ),
    }
    return rows, diagnostic


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _verify_inputs(protocol)
    shift_config = protocol["feature_shift"]
    gate_config = {**protocol["risk_gate"], **protocol["gate_guardrails"]}

    arithmetic, stroop = load_feature_bundles(REPO_ROOT, torch.device("cpu"))
    raw = pd.read_csv(PARENT_RAW_PATH)
    diagnostics_seed = pd.read_csv(PARENT_DIAGNOSTIC_PATH)
    window_ensemble = median_seed_ensemble(raw, SEEDS)
    base_recordings = aggregate_recordings(window_ensemble)
    original_diagnostics = final_diagnostics(window_ensemble, diagnostics_seed)

    cross_diagnostics = _cross_task_robust_diagnostics(
        original_diagnostics, arithmetic, stroop, shift_config
    )
    cross_training = cross_task_gate_rows(base_recordings, cross_diagnostics)
    loso_training = _loso_meta_training_robust_shift(
        pd.read_csv(LOSO_ROOT / "gate_training_rows.csv"), arithmetic, shift_config
    )
    loso_diagnostics = _loso_target_robust_diagnostics(
        original_diagnostics, arithmetic, shift_config
    )

    generated = []
    weight_rows = []
    for target in cross_diagnostics.sort_values(["subject", "direction"]).to_dict("records"):
        target_series = pd.Series(target)
        training = cross_training[cross_training["subject"] != int(target["subject"])].copy()
        rows, gate_row = _gate_one_target(target_series, training, window_ensemble, gate_config)
        generated.extend(rows)
        weight_rows.append(gate_row)
    for target in loso_diagnostics.sort_values("subject").to_dict("records"):
        target_series = pd.Series(target)
        training = loso_training[
            loso_training["outer_subject"] == int(target["subject"])
        ].copy()
        if int(target["subject"]) in set(training["meta_subject"].astype(int)):
            raise AssertionError("Outer participant leaked into robust LOSO gate training")
        rows, gate_row = _gate_one_target(target_series, training, window_ensemble, gate_config)
        generated.extend(rows)
        weight_rows.append(gate_row)

    windows = pd.DataFrame(generated)
    recordings = aggregate_recordings(windows)
    validate_frozen_predictions(recordings)
    if len(recordings) != 180:
        raise AssertionError(f"Expected 180 robust recording rows, observed {len(recordings)}")

    original_loso = pd.read_csv(LOSO_ROOT / "recording_predictions.csv")
    original_cross = pd.read_csv(PARENT_RECORDING_PATH)
    original_cross = original_cross[
        (original_cross["protocol"] == "cross_task")
        & (original_cross["method"] == "risk_aware")
    ].copy()
    original = pd.concat([original_cross, original_loso], ignore_index=True)
    original["method"] = "risk_aware_original"
    comparator = pd.read_csv(PARENT_RECORDING_PATH)
    comparator = comparator[comparator["method"].isin(["always_nn", "always_fuse"])]
    comparison = pd.concat([comparator, original, recordings], ignore_index=True)
    metrics = subject_metrics(comparison)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    recording_path = OUTPUT_ROOT / "recording_predictions.csv"
    windows.to_csv(OUTPUT_ROOT / "window_predictions.csv", index=False, lineterminator="\n")
    recordings.to_csv(recording_path, index=False, lineterminator="\n")
    pd.DataFrame(weight_rows).to_csv(
        OUTPUT_ROOT / "gate_weights.csv", index=False, lineterminator="\n"
    )
    pd.concat([cross_diagnostics, loso_diagnostics], ignore_index=True).to_csv(
        OUTPUT_ROOT / "target_diagnostics.csv", index=False, lineterminator="\n"
    )
    loso_training.to_csv(
        OUTPUT_ROOT / "loso_gate_training_rows.csv", index=False, lineterminator="\n"
    )
    metrics.to_csv(OUTPUT_ROOT / "subject_metrics.csv", index=False, lineterminator="\n")
    summary_table(metrics).to_csv(
        OUTPUT_ROOT / "method_summary.csv", index=False, lineterminator="\n"
    )
    paired_comparisons(metrics, margin=0.05).to_csv(
        OUTPUT_ROOT / "paired_vs_always_nn.csv", index=False, lineterminator="\n"
    )
    tail_descriptives(metrics).to_csv(
        OUTPUT_ROOT / "tail_descriptives_exploratory.csv", index=False, lineterminator="\n"
    )
    weights = pd.DataFrame(weight_rows)
    audit = {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_recording_prediction_sha256": sha256_file(PARENT_RECORDING_PATH),
        "parent_raw_seed_prediction_sha256": sha256_file(PARENT_RAW_PATH),
        "source_cbsf_loso_recording_prediction_sha256": sha256_file(
            LOSO_ROOT / "recording_predictions.csv"
        ),
        "recording_prediction_sha256": sha256_file(recording_path),
        "post_hoc_robustness_analysis": True,
        "original_outputs_overwritten": False,
        "final_recording_rows": int(len(recordings)),
        "gate_rows": int(len(weights)),
        "ood_fallback_rows": int(weights["ood_fallback"].sum()),
        "max_abs_predicted_benefit": float(weights["predicted_benefit"].abs().max()),
        "max_abs_raw_predicted_benefit": float(weights["predicted_benefit_raw"].abs().max()),
        "statistical_unit": "held-out participant",
        "seed_is_statistical_unit": False,
        "claim_scope": protocol["claim_scope"],
    }
    (OUTPUT_ROOT / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary_table(metrics).to_string(index=False))
    print(f"Robustness analysis completed: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
