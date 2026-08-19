"""Pure-data audit helpers for the formal nested-LOSO CB-SF run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from revision_pipeline.aggregation import validate_frozen_predictions
from revision_pipeline.risk_gate import GATE_FEATURES


SEEDS = [1335, 1388, 1441, 1494, 1547]
EXPECTED_SUBJECTS = set(range(1, 16))
EXPECTED_META_METHODS = {"always_nn", "always_fuse"}
EXPECTED_RECORDINGS_PER_SUBJECT = 4


def make_gate_training_rows(
    meta_predictions: pd.DataFrame,
    meta_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    group_keys = ["outer_subject", "meta_subject", "recording_id", "method"]
    if meta_predictions.duplicated(group_keys + ["seed"]).any():
        raise ValueError("Duplicate meta recording/method/seed rows")
    observed = meta_predictions.groupby(group_keys)["seed"].agg(
        lambda values: (len(values), set(int(v) for v in values))
    )
    if observed.map(lambda values: values != (len(SEEDS), set(SEEDS))).any():
        raise ValueError("Every meta recording/method requires exactly five frozen seeds")

    diagnostic_keys = ["outer_subject", "meta_subject"]
    if meta_diagnostics.duplicated(diagnostic_keys + ["seed"]).any():
        raise ValueError("Duplicate meta diagnostic seed rows")
    diagnostic_seeds = meta_diagnostics.groupby(diagnostic_keys)["seed"].agg(
        lambda values: (len(values), set(int(v) for v in values))
    )
    if diagnostic_seeds.map(lambda values: values != (len(SEEDS), set(SEEDS))).any():
        raise ValueError("Every meta diagnostic row requires exactly five frozen seeds")

    labels = meta_predictions.groupby(group_keys, as_index=False)["true_label"].agg(
        lambda values: int(np.unique(values)[0]) if np.unique(values).size == 1 else -1
    )
    if (labels["true_label"] < 0).any():
        raise ValueError("Meta recording labels differ across seeds")
    probability = meta_predictions.groupby(group_keys, as_index=False)[["p0", "p1"]].median()
    median = labels.merge(probability, on=group_keys, validate="one_to_one")
    total = median[["p0", "p1"]].sum(axis=1)
    if (total <= 0.0).any() or not np.isfinite(total).all():
        raise ValueError("Meta median probabilities have invalid mass")
    median[["p0", "p1"]] = median[["p0", "p1"]].div(total, axis=0)
    median["pred_label"] = np.argmax(median[["p0", "p1"]].to_numpy(), axis=1)
    median["correct"] = (median["pred_label"] == median["true_label"]).astype(int)

    accuracy = median.groupby(["outer_subject", "meta_subject", "method"], as_index=False)[
        "correct"
    ].mean()
    pivot = accuracy.pivot(
        index=["outer_subject", "meta_subject"], columns="method", values="correct"
    ).reset_index()
    missing_methods = EXPECTED_META_METHODS - set(pivot.columns)
    if missing_methods:
        raise ValueError(f"Meta gate inputs are missing methods: {sorted(missing_methods)}")
    pivot["benefit"] = pivot["always_fuse"] - pivot["always_nn"]
    diagnostics = meta_diagnostics.groupby(diagnostic_keys, as_index=False)[GATE_FEATURES].median()
    training = diagnostics.merge(
        pivot[["outer_subject", "meta_subject", "benefit"]],
        on=diagnostic_keys,
        validate="one_to_one",
    )
    counts = training.groupby("outer_subject")["meta_subject"].nunique()
    if set(training["outer_subject"].astype(int)) != EXPECTED_SUBJECTS:
        raise ValueError("Gate training rows do not cover all 15 outer participants")
    if not counts.eq(14).all():
        raise ValueError(f"Every outer gate requires 14 meta participants; observed={counts.to_dict()}")
    if any(int(row.outer_subject) == int(row.meta_subject) for row in training.itertuples()):
        raise AssertionError("Outer test participant leaked into CB-SF gate training")
    return training.sort_values(["outer_subject", "meta_subject"]).reset_index(drop=True)


def validate_final_cbsf_recordings(recordings: pd.DataFrame, always_nn_reference: pd.DataFrame) -> None:
    validate_frozen_predictions(recordings)
    if set(recordings["method"].astype(str)) != {"risk_aware"}:
        raise AssertionError("Final CB-SF output must contain only risk_aware rows")
    if set(recordings["protocol"].astype(str)) != {"loso"}:
        raise AssertionError("Final CB-SF completion is restricted to LOSO")
    if len(recordings) != 60 or set(recordings["subject"].astype(int)) != EXPECTED_SUBJECTS:
        raise AssertionError("Final CB-SF LOSO output must contain 60 recordings from 15 participants")
    identity = ["dataset", "protocol", "subject", "direction", "recording_id", "true_label"]
    observed = recordings[identity].sort_values(identity).reset_index(drop=True)
    expected = always_nn_reference[identity].sort_values(identity).reset_index(drop=True)
    if not observed.equals(expected):
        raise AssertionError("Final CB-SF LOSO recording identities differ from always-NN")
