from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROUTE_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = ROUTE_ROOT.parent
sys.path.insert(0, str(ROUTE_ROOT))
sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.aggregation import sha256_file, validate_frozen_predictions  # noqa: E402
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.risk_gate import (  # noqa: E402
    fit_predict_gate_robust,
    robust_feature_shift,
)
from route_a_lib.probability import (  # noqa: E402
    apply_clean_dasf,
    blend_probabilities,
    clean_dasf_decision,
    crossfit_window_stacker,
)
from freeze_route_a import EXPECTED, assert_complete  # noqa: E402
from route_a_lib.cbsf_audit import (  # noqa: E402
    GATE_FEATURES,
    SEEDS,
    make_gate_training_rows,
    validate_final_cbsf_recordings,
)


def test_route_protocol_is_bound_to_parent_freeze() -> None:
    route = json.loads((ROUTE_ROOT / "protocol_route_a.json").read_text(encoding="utf-8"))
    parent_protocol = Protocol.load(REVISION_ROOT / "protocol.json")
    parent_predictions = REVISION_ROOT / "frozen" / "predictions_recording.csv"
    assert route["parent_protocol_sha256"] == parent_protocol.digest
    assert route["parent_prediction_sha256"] == sha256_file(parent_predictions)
    assert route["seeds"] == [1335, 1388, 1441, 1494, 1547]


def test_fixed_blend_formula_and_normalization() -> None:
    nn = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    fuse = np.asarray([[0.5, 0.5], [0.6, 0.4]])
    result = blend_probabilities(nn, fuse, 0.25)
    np.testing.assert_allclose(result, 0.75 * nn + 0.25 * fuse, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)


def test_clean_dasf_rejection_is_exact_nn() -> None:
    nn = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    fuse = np.asarray([[0.4, 0.6], [0.9, 0.1]], dtype=np.float64)
    decision = clean_dasf_decision(
        0.75,
        0.76,
        0.2,
        base_margin=0.02,
        alpha=0.20,
        max_margin=0.15,
    )
    assert not decision.use_fusion
    assert np.array_equal(apply_clean_dasf(nn, fuse, decision), nn)


def test_clean_dasf_acceptance_equals_no_gating_fusion() -> None:
    """Reproduces the old theoretical equality condition that failed for S4."""
    nn = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    fuse = np.asarray([[0.4, 0.6], [0.9, 0.1]], dtype=np.float64)
    decision = clean_dasf_decision(
        0.50,
        0.80,
        0.1,
        base_margin=0.02,
        alpha=0.20,
        max_margin=0.15,
    )
    assert decision.use_fusion
    dasf_output = apply_clean_dasf(nn, fuse, decision)
    no_gating_output = fuse.copy()
    assert np.array_equal(dasf_output, no_gating_output)


def test_clean_dasf_nonfinite_signal_fails_closed() -> None:
    decision = clean_dasf_decision(
        np.nan,
        1.0,
        0.1,
        base_margin=0.02,
        alpha=0.20,
        max_margin=0.15,
    )
    assert not decision.use_fusion


def test_meta_crossfit_is_deterministic_and_covers_each_group() -> None:
    labels = np.repeat([0, 0, 1, 1], 3)
    recordings = np.repeat(["r0", "r1", "r2", "r3"], 3)
    groups = recordings.copy()
    p1 = np.where(labels == 1, 0.72, 0.28)
    base = {
        "always_nn": np.column_stack([1.0 - p1, p1]),
        "rf": np.column_stack([1.0 - (0.9 * p1 + 0.05), 0.9 * p1 + 0.05]),
        "extra_trees": np.column_stack([1.0 - (0.8 * p1 + 0.10), 0.8 * p1 + 0.10]),
    }
    first = crossfit_window_stacker(base, labels, recordings, groups, c_value=1.0, seed=17)
    second = crossfit_window_stacker(base, labels, recordings, groups, c_value=1.0, seed=17)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first.sum(axis=1), 1.0)


def test_lightweight_output_is_recording_level_and_complete() -> None:
    path = ROUTE_ROOT / "results" / "lightweight_recording_predictions.csv"
    assert path.exists(), "Run build_lightweight_baselines.py"
    frame = pd.read_csv(path)
    validate_frozen_predictions(frame)
    assert "seed" not in frame
    assert "window_id" not in frame
    assert set(frame["method"]) == {
        "fixed_blend_010",
        "fixed_blend_025",
        "equal_blend_050",
        "stack_recording",
    }
    assert len(frame) == 720


def test_recording_stacker_excludes_target_participant() -> None:
    diagnostics = pd.read_csv(ROUTE_ROOT / "results" / "recording_stacker_diagnostics.csv")
    assert diagnostics["target_participant_excluded"].astype(bool).all()
    cross = diagnostics[diagnostics["protocol"] == "cross_task"]
    loso = diagnostics[diagnostics["protocol"] == "loso"]
    assert cross["meta_training_participants"].eq(14).all()
    assert loso["meta_training_participants"].eq(14).all()


def _synthetic_cbsf_meta_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    diagnostic_rows = []
    for outer_subject in range(1, 16):
        for meta_subject in sorted(set(range(1, 16)) - {outer_subject}):
            for seed in SEEDS:
                diagnostic_rows.append(
                    {
                        "outer_subject": outer_subject,
                        "meta_subject": meta_subject,
                        "seed": seed,
                        **{name: 0.01 * meta_subject for name in GATE_FEATURES},
                    }
                )
                for recording_index, true_label in enumerate((0, 0, 1, 1)):
                    for method in ("always_nn", "always_fuse"):
                        confidence = 0.70 if method == "always_nn" else 0.75
                        p1 = confidence if true_label == 1 else 1.0 - confidence
                        prediction_rows.append(
                            {
                                "outer_subject": outer_subject,
                                "meta_subject": meta_subject,
                                "recording_id": f"s{meta_subject:02d}:r{recording_index}",
                                "method": method,
                                "seed": seed,
                                "true_label": true_label,
                                "p0": 1.0 - p1,
                                "p1": p1,
                            }
                        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(diagnostic_rows)


def test_cbsf_gate_training_is_outer_subject_disjoint_and_five_seed() -> None:
    predictions, diagnostics = _synthetic_cbsf_meta_rows()
    training = make_gate_training_rows(predictions, diagnostics)
    assert len(training) == 15 * 14
    assert training.groupby("outer_subject")["meta_subject"].nunique().eq(14).all()
    assert not (training["outer_subject"] == training["meta_subject"]).any()

    incomplete = predictions.drop(index=predictions.index[0])
    with pytest.raises(ValueError, match="exactly five frozen seeds"):
        make_gate_training_rows(incomplete, diagnostics)


def test_route_a_freeze_rejects_missing_loso_cbsf() -> None:
    rows = []
    for protocol, methods in EXPECTED.items():
        included = methods - ({"cbsf"} if protocol == "loso" else set())
        for method in included:
            rows.append(
                {
                    "dataset": "synthetic",
                    "protocol": protocol,
                    "subject": 1,
                    "direction": "test",
                    "recording_id": "r0",
                    "method": method,
                }
            )
    with pytest.raises(ValueError, match="loso methods incomplete"):
        assert_complete(pd.DataFrame(rows))


def test_robust_cbsf_output_covers_both_protocols() -> None:
    path = ROUTE_ROOT / "results" / "cbsf_robustness_r1" / "recording_predictions.csv"
    frame = pd.read_csv(path)
    validate_frozen_predictions(frame)
    assert set(frame["method"].astype(str)) == {"risk_aware_robust"}
    assert set(frame["protocol"].astype(str)) == {"cross_task", "loso"}
    assert len(frame) == 180


def test_final_cbsf_schema_matches_frozen_loso_recordings() -> None:
    parent = pd.read_csv(REVISION_ROOT / "frozen" / "predictions_recording.csv")
    reference = parent[(parent["protocol"] == "loso") & (parent["method"] == "always_nn")].copy()
    candidate = reference.copy()
    candidate["method"] = "risk_aware"
    validate_final_cbsf_recordings(candidate, reference)


def test_robust_feature_shift_resists_one_pathological_dimension() -> None:
    reference = np.zeros((20, 272), dtype=np.float64)
    reference[:, 0] = np.linspace(-1.0, 1.0, len(reference))
    target = reference.copy()
    target[:, 0] += 1_000_000.0
    assert robust_feature_shift(reference, target) == 0.0


def test_robust_gate_bounds_benefit_and_fails_closed_outside_support() -> None:
    rows = []
    for index in range(14):
        rows.append(
            {
                **{name: 0.1 + 0.01 * index for name in GATE_FEATURES},
                "benefit": [-0.25, 0.0, 0.25][index % 3],
            }
        )
    training = pd.DataFrame(rows)
    target = pd.Series({name: 0.15 for name in GATE_FEATURES})
    target["feature_shift"] = 800.0
    prediction = fit_predict_gate_robust(
        training,
        target,
        catastrophe_threshold=0.05,
        benefit_alpha=1.0,
        catastrophe_c=1.0,
        risk_penalty=0.05,
        temperature=0.02,
        max_weight=0.8,
    )
    assert prediction.ood_fallback
    assert prediction.weight == 0.0
    assert -0.25 <= prediction.predicted_benefit <= 0.25
