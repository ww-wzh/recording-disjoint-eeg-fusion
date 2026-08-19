from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))

from build_outputs import (  # noqa: E402
    audit_recording_resolution,
    paired_comparisons,
    plot_paired_differences,
    plot_summary,
    subject_metrics,
    summary_table,
    tail_descriptives,
    write_markdown,
)
from revision_pipeline.aggregation import sha256_file, validate_frozen_predictions  # noqa: E402


def main() -> None:
    frozen = HERE / "frozen" / "predictions_recording_route_a.csv"
    if not frozen.exists():
        raise FileNotFoundError("Run freeze_route_a.py after every full Route A baseline has completed")
    predictions = pd.read_csv(frozen)
    validate_frozen_predictions(predictions)
    audit_recording_resolution(predictions)
    metrics = subject_metrics(predictions)
    summary = summary_table(metrics)
    comparisons = paired_comparisons(metrics, margin=0.05)
    tails = tail_descriptives(metrics)
    primary_metrics = metrics[metrics["method"].isin(["always_nn", "cbsf"])].copy()
    primary_comparisons = paired_comparisons(primary_metrics, margin=0.05)
    focus_methods = [
        "always_nn",
        "always_fuse",
        "dasf_clean",
        "cbsf",
        "fixed_blend_010",
        "equal_blend_050",
        "stack_recording",
    ]
    focus_metrics = metrics[metrics["method"].isin(focus_methods)].copy()
    focus_summary = summary_table(focus_metrics)
    output = HERE / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "subject_metrics.csv", index=False, lineterminator="\n")
    summary.to_csv(output / "method_summary.csv", index=False, lineterminator="\n")
    comparisons.to_csv(output / "paired_vs_always_nn.csv", index=False, lineterminator="\n")
    tails.to_csv(output / "tail_descriptives_exploratory.csv", index=False, lineterminator="\n")
    primary_comparisons.to_csv(
        output / "paired_vs_always_nn_primary.csv", index=False, lineterminator="\n"
    )
    focus_summary.to_csv(
        output / "method_summary_manuscript_core.csv", index=False, lineterminator="\n"
    )
    plot_summary(focus_summary, output / "figure_accuracy_ci_manuscript_core.png")
    plot_paired_differences(
        focus_metrics, output / "figure_paired_differences_manuscript_core.png"
    )
    write_markdown(
        output / "results_summary.md",
        frozen,
        summary,
        comparisons,
        tails,
        0.05,
    )
    generated_files = [
        "subject_metrics.csv",
        "method_summary.csv",
        "paired_vs_always_nn.csv",
        "paired_vs_always_nn_primary.csv",
        "tail_descriptives_exploratory.csv",
        "method_summary_manuscript_core.csv",
        "figure_accuracy_ci_manuscript_core.png",
        "figure_paired_differences_manuscript_core.png",
        "results_summary.md",
    ]
    manifest = {
        "prediction_file": str(frozen),
        "prediction_sha256": sha256_file(frozen),
        "protocol_bundle_manifest": str(HERE / "frozen" / "protocol_bundle_manifest.json"),
        "protocol_bundle_sha256": json.loads(
            (HERE / "frozen" / "protocol_bundle_manifest.json").read_text(encoding="utf-8")
        )["protocol_bundle_sha256"],
        "tail_status": "exploratory; n=15 is insufficient for stable CVaR10 inference",
        "generated_files": generated_files,
        "generated_sha256": {
            name: sha256_file(output / name) for name in generated_files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Wrote Route A tables to {output}")


if __name__ == "__main__":
    main()
