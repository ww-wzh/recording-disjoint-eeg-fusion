from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    validate_frozen_predictions,
)
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.splits import (  # noqa: E402
    assert_disjoint_groups,
    balanced_recording_folds,
    subject_loso_folds,
)

import build_outputs  # noqa: E402
import freeze_predictions  # noqa: E402


def test_protocol_is_five_seed_eeg_only() -> None:
    protocol = Protocol.load(ROOT / "protocol.json")
    assert len(protocol.seeds) == 5
    assert len(protocol.digest) == 64


def test_balanced_recording_folds_never_split_windows() -> None:
    labels = np.repeat([0, 1, 2, 3], 3)
    recordings = np.repeat(["r0", "r1", "r2", "r3"], 3)
    folds = balanced_recording_folds(labels, recordings, seed=7)
    validation_seen = np.zeros(len(labels), dtype=int)
    for split in folds:
        assert_disjoint_groups(split.train, split.validation, recordings)
        assert len(np.unique(recordings[split.validation])) == 2
        assert set((labels[split.validation] >= 2).astype(int)) == {0, 1}
        validation_seen[split.validation] += 1
    assert np.all(validation_seen == 1)


def test_overlap_assertion_fails_closed() -> None:
    groups = np.array(["same", "same", "other"])
    with pytest.raises(AssertionError):
        assert_disjoint_groups(np.array([0]), np.array([1]), groups)


def test_subject_loso_is_subject_disjoint() -> None:
    subjects = np.repeat([1, 2, 3], 2)
    folds = subject_loso_folds(subjects)
    assert len(folds) == 3
    for split in folds:
        assert not (set(subjects[split.train]) & set(subjects[split.validation]))


def make_seed_frame() -> pd.DataFrame:
    rows = []
    seeds = [11, 12, 13, 14, 15]
    for method in ("always_nn", "always_fuse"):
        for window in ("w0", "w1"):
            for seed in seeds:
                p1 = 0.8 if method == "always_nn" else 0.7
                rows.append(
                    {
                        "dataset": "synthetic",
                        "protocol": "cross_task",
                        "subject": 1,
                        "direction": "a_to_b",
                        "recording_id": "r0",
                        "window_id": window,
                        "method": method,
                        "seed": seed,
                        "true_label": 1,
                        "p0": 1.0 - p1,
                        "p1": p1,
                    }
                )
    return pd.DataFrame(rows)


def test_final_aggregation_removes_seed_and_window_pseudoreplication() -> None:
    raw = make_seed_frame()
    windows = median_seed_ensemble(raw, [11, 12, 13, 14, 15])
    recordings = aggregate_recordings(windows)
    assert len(recordings) == 2
    assert "seed" not in recordings.columns
    assert "window_id" not in recordings.columns
    assert recordings["correct"].eq(1).all()
    validate_frozen_predictions(recordings)


def test_final_schema_rejects_seed_column() -> None:
    raw = make_seed_frame()
    recordings = aggregate_recordings(median_seed_ensemble(raw, [11, 12, 13, 14, 15]))
    recordings["seed"] = 11
    with pytest.raises(ValueError):
        validate_frozen_predictions(recordings)


def test_cross_task_freeze_and_statistics_end_to_end(tmp_path: Path) -> None:
    protocol = Protocol.load(ROOT / "protocol.json")
    raw_rows = []
    diagnostic_rows = []
    methods = ("always_nn", "rf", "extra_trees", "always_fuse")
    directions = ("a_to_b", "b_to_a")
    for subject in range(1, 7):
        for direction in directions:
            for seed_index, seed in enumerate(protocol.seeds):
                diagnostic_rows.append(
                    {
                        "protocol": "cross_task",
                        "outer_subject": subject,
                        "direction": direction,
                        "seed": seed,
                        "feature_shift": 0.1 + 0.01 * subject,
                    }
                )
                for recording in range(4):
                    true_label = int(recording >= 2)
                    for window in range(2):
                        for method in methods:
                            base = {
                                "always_nn": 0.76,
                                "rf": 0.67,
                                "extra_trees": 0.70,
                                "always_fuse": 0.78,
                            }[method]
                            correct_probability = base + 0.002 * seed_index - 0.002 * subject
                            p1 = correct_probability if true_label == 1 else 1.0 - correct_probability
                            raw_rows.append(
                                {
                                    "dataset": "synthetic",
                                    "protocol": "cross_task",
                                    "subject": subject,
                                    "direction": direction,
                                    "recording_id": f"s{subject}_{direction}_r{recording}",
                                    "window_id": f"s{subject}_{direction}_r{recording}_w{window}",
                                    "method": method,
                                    "seed": seed,
                                    "true_label": true_label,
                                    "p0": 1.0 - p1,
                                    "p1": p1,
                                }
                            )
    raw = pd.DataFrame(raw_rows)
    diagnostics_seed = pd.DataFrame(diagnostic_rows)
    windows = median_seed_ensemble(raw, protocol.seeds)
    base_recordings = aggregate_recordings(windows)
    diagnostics = freeze_predictions.final_diagnostics(windows, diagnostics_seed)
    risk_rows, weights = freeze_predictions.make_risk_aware_rows(
        windows, base_recordings, diagnostics, protocol, tmp_path
    )
    assert len(weights) == 12
    final = aggregate_recordings(pd.concat([windows, risk_rows], ignore_index=True))
    validate_frozen_predictions(final)
    build_outputs.audit_recording_resolution(final)
    metrics = build_outputs.subject_metrics(final)
    comparisons = build_outputs.paired_comparisons(metrics, margin=0.05)
    primary = comparisons[comparisons["method"] == "risk_aware"]
    assert len(primary) == 1
    assert bool(primary.iloc[0]["noninferior"])
