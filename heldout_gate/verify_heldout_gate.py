"""Verify the published held-out-gain audit artifacts without retraining."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> int:
    checked = 0
    manifest = ROOT / "MANIFEST.sha256"
    for raw in manifest.read_text(encoding="ascii").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"SHA-256 mismatch for {relative}: {observed}")
        checked += 1
    return checked


def read_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        "README.md",
        "heldout_gate_protocol.json",
        "heldout_gate_recording_predictions.csv",
        "heldout_gate_window_predictions.csv",
        "heldout_gate_diagnostics.csv",
        "heldout_gate_cross_task_training_rows.csv",
        "heldout_gate_event_counts.csv",
        "heldout_gate_method_comparisons_BH14.csv",
        "heldout_gate_recording_and_participant_metrics.csv",
        "heldout_gate_exact_sign_flip_wilcoxon_BH.csv",
        "heldout_gate_79_family_summary.csv",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    prediction_rows = read_rows("heldout_gate_recording_predictions.csv")
    if not prediction_rows:
        raise RuntimeError("Held-out recording prediction file is empty")
    if len(prediction_rows) != 302:
        raise RuntimeError(f"Expected 302 held-out recording rows, found {len(prediction_rows)}")
    required_prediction_fields = {
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "method",
        "true_label",
        "pred_label",
    }
    fields = set(prediction_rows[0])
    missing_fields = required_prediction_fields - fields
    if missing_fields:
        raise RuntimeError(f"Missing prediction columns: {sorted(missing_fields)}")

    comparisons = read_rows("heldout_gate_method_comparisons_BH14.csv")
    if len(comparisons) != 14:
        raise RuntimeError(f"Expected 14 held-out comparisons, found {len(comparisons)}")

    protocol = json.loads((ROOT / "heldout_gate_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("canonical_primary_prediction_sha256") != (
        "cba069dd3837e9b3894dc2649aa6c8a9ac80812de996b2dc51cc0aa056f5c5cd"
    ):
        raise RuntimeError("Held-out protocol is not linked to the v7 neutral primary prediction")
    event_rows = read_rows("heldout_gate_event_counts.csv")
    pooled = next((row for row in event_rows if row.get("direction") == "pooled"), None)
    if pooled is None or (pooled.get("training_rows"), pooled.get("loss_events"), pooled.get("non_loss_events")) != ("26", "6", "20"):
        raise RuntimeError("Unexpected held-out loss-event counts")
    family_rows = read_rows("heldout_gate_79_family_summary.csv")
    if {row.get("n_comparisons") for row in family_rows} != {"79"}:
        raise RuntimeError("The 79-comparison sensitivity summary is incomplete")

    if verify_manifest() < len(required):
        raise RuntimeError("Manifest did not cover all required files")
    print(f"heldout_gate verification passed: {len(prediction_rows)} recording rows, "
          f"{len(comparisons)} comparisons")


if __name__ == "__main__":
    main()
