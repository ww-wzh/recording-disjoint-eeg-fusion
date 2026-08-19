from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))

from build_outputs import paired_comparisons, subject_metrics, summary_table  # noqa: E402
from revision_pipeline.aggregation import aggregate_recordings, median_seed_ensemble, sha256_file  # noqa: E402


SEEDS = [1335, 1388, 1441, 1494, 1547]
OPTIONAL_RAW = {
    "riemannian": HERE / "results" / "riemannian" / "raw_seed_predictions.csv",
    "dasf_clean": HERE / "results" / "dasf_clean" / "raw_seed_predictions.csv",
    "eegnet": HERE / "results" / "deep" / "eegnet" / "raw_seed_predictions.csv",
    "eeg_conformer": HERE / "results" / "deep" / "eeg_conformer" / "raw_seed_predictions.csv",
}


def main() -> None:
    frames = [
        pd.read_csv(REVISION_ROOT / "frozen" / "predictions_recording.csv"),
        pd.read_csv(HERE / "results" / "lightweight_recording_predictions.csv"),
    ]
    included = ["parent_frozen", "lightweight"]
    skipped = {}
    for name, path in OPTIONAL_RAW.items():
        if not path.exists():
            skipped[name] = "missing"
            continue
        try:
            recordings = aggregate_recordings(median_seed_ensemble(pd.read_csv(path), SEEDS))
        except (ValueError, AssertionError) as exc:
            skipped[name] = f"incomplete: {exc}"
            continue
        frames.append(recordings)
        included.append(name)
    cbsf_loso = HERE / "results" / "cbsf_loso" / "recording_predictions.csv"
    if cbsf_loso.exists():
        frames.append(pd.read_csv(cbsf_loso))
        included.append("cbsf_loso")
    else:
        skipped["cbsf_loso"] = "missing"
    final = pd.concat(frames, ignore_index=True)
    metrics = subject_metrics(final)
    summary = summary_table(metrics)
    comparisons = paired_comparisons(metrics, margin=0.05)
    output = HERE / "outputs_available"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "method_summary_available.csv", index=False, lineterminator="\n")
    comparisons.to_csv(output / "paired_vs_always_nn_available.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "provisional; do not use in the manuscript until freeze_route_a.py succeeds",
        "included": included,
        "skipped": skipped,
        "protocol_sha256": sha256_file(HERE / "protocol_route_a.json"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nIncluded:", ", ".join(included))
    if skipped:
        print("Skipped:", json.dumps(skipped, ensure_ascii=False))


if __name__ == "__main__":
    main()
