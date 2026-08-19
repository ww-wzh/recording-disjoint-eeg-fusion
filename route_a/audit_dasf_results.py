from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))

from build_outputs import (  # noqa: E402
    paired_comparisons,
    subject_metrics,
    summary_table,
    tail_descriptives,
)
from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)


SEEDS = [1335, 1388, 1441, 1494, 1547]
ROW_KEYS = [
    "dataset",
    "protocol",
    "subject",
    "direction",
    "recording_id",
    "window_id",
    "seed",
    "true_label",
]


def main() -> None:
    source_path = REVISION_ROOT / "results" / "raw_seed_predictions.csv"
    parent_path = REVISION_ROOT / "frozen" / "predictions_recording.csv"
    result_root = HERE / "results" / "dasf_clean"
    prediction_path = result_root / "raw_seed_predictions.csv"
    diagnostic_path = result_root / "gate_diagnostics.csv"
    seed_diagnostic_path = result_root / "gate_diagnostics_seed.csv"
    if not prediction_path.exists() or not diagnostic_path.exists() or not seed_diagnostic_path.exists():
        raise FileNotFoundError("Complete clean DASF outputs are missing")

    source = pd.read_csv(source_path)
    source_text = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    actual = pd.read_csv(prediction_path)
    actual_text = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
    diagnostics = pd.read_csv(diagnostic_path)
    seed_diagnostics = pd.read_csv(seed_diagnostic_path)
    diagnostics["used_fusion"] = diagnostics["used_fusion"].astype(bool)
    if len(seed_diagnostics) != 225:
        raise ValueError(f"Expected 225 DASF seed diagnostics, got {len(seed_diagnostics)}")
    if len(diagnostics) != 45:
        raise ValueError(f"Expected 45 final DASF gate cells, got {len(diagnostics)}")
    expected_cells = {"cross_task": 30, "loso": 15}
    if diagnostics.groupby("protocol").size().to_dict() != expected_cells:
        raise ValueError("DASF diagnostics do not have complete protocol coverage")

    choice = diagnostics.assign(
        source_method=np.where(diagnostics["used_fusion"], "always_fuse", "always_nn")
    )[["protocol", "subject", "direction", "source_method"]]
    source = source[source["method"].isin(["always_nn", "always_fuse"])].rename(
        columns={"method": "source_method"}
    )
    expected = source.merge(
        choice,
        on=["protocol", "subject", "direction", "source_method"],
        how="inner",
        validate="many_to_one",
    )
    expected = expected.sort_values(ROW_KEYS).reset_index(drop=True)
    actual = actual.sort_values(ROW_KEYS).reset_index(drop=True)
    if len(actual) != len(expected):
        raise AssertionError(f"DASF output row count differs from selected inputs: {len(actual)} vs {len(expected)}")
    if not actual[ROW_KEYS].equals(expected[ROW_KEYS]):
        raise AssertionError("DASF output row identities differ from selected frozen inputs")
    text_choice = choice.copy()
    text_choice["subject"] = text_choice["subject"].astype(int).astype(str)
    expected_text = source_text[source_text["method"].isin(["always_nn", "always_fuse"])].rename(
        columns={"method": "source_method"}
    ).merge(
        text_choice,
        on=["protocol", "subject", "direction", "source_method"],
        how="inner",
        validate="many_to_one",
    )
    expected_text["method"] = "dasf_clean"
    expected_text = expected_text[actual_text.columns].sort_values(ROW_KEYS).reset_index(drop=True)
    actual_text = actual_text.sort_values(ROW_KEYS).reset_index(drop=True)
    if not actual_text[ROW_KEYS + ["p0", "p1"]].equals(
        expected_text[ROW_KEYS + ["p0", "p1"]]
    ):
        raise AssertionError("DASF output CSV tokens are not exactly the selected frozen input tokens")
    if actual.duplicated(ROW_KEYS + ["method"]).any():
        raise AssertionError("DASF output contains duplicate seed/window rows")
    observed_seed_sets = actual.groupby(
        ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method"]
    )["seed"].agg(lambda values: set(int(value) for value in values))
    if observed_seed_sets.map(lambda values: values != set(SEEDS)).any():
        raise AssertionError("DASF final windows do not contain exactly five frozen seeds")

    recordings = aggregate_recordings(median_seed_ensemble(actual, SEEDS))
    validate_frozen_predictions(recordings)
    recording_path = result_root / "recording_predictions.csv"
    recordings.to_csv(recording_path, index=False, lineterminator="\n")
    combined = pd.concat([pd.read_csv(parent_path), recordings], ignore_index=True)
    metrics = subject_metrics(combined)
    summary = summary_table(metrics)
    paired = paired_comparisons(metrics, margin=0.05)
    tails = tail_descriptives(metrics)
    summary[summary["method"].isin(["always_nn", "always_fuse", "dasf_clean"])].to_csv(
        result_root / "method_summary.csv", index=False, lineterminator="\n"
    )
    paired[paired["method"] == "dasf_clean"].to_csv(
        result_root / "paired_vs_always_nn.csv", index=False, lineterminator="\n"
    )
    tails[tails["method"] == "dasf_clean"].to_csv(
        result_root / "tail_descriptives_exploratory.csv", index=False, lineterminator="\n"
    )

    audit = {
        "exact_seed_row_identity": True,
        "exact_probability_token_identity": True,
        "window_rows_checked": int(len(actual)),
        "seed_diagnostic_cells_checked": int(len(seed_diagnostics)),
        "final_ensemble_gate_cells_checked": int(len(diagnostics)),
        "five_seed_windows": int(len(observed_seed_sets)),
        "recording_rows": int(len(recordings)),
        "source_raw_sha256": sha256_file(source_path),
        "dasf_raw_sha256": sha256_file(prediction_path),
        "gate_diagnostics_sha256": sha256_file(diagnostic_path),
        "seed_gate_diagnostics_sha256": sha256_file(seed_diagnostic_path),
        "recording_prediction_sha256": sha256_file(recording_path),
        "protocol_sha256": sha256_file(HERE / "protocol_route_a.json"),
        "statistical_unit": "participant",
        "seed_is_statistical_unit": False,
    }
    (result_root / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print("\nDASF paired comparisons:")
    print(paired[paired["method"] == "dasf_clean"].to_string(index=False))


if __name__ == "__main__":
    main()
