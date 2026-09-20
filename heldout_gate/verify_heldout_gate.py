"""Verify the published held-out-gain audit artifacts without retraining."""

from __future__ import annotations

import csv
import hashlib
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
        "heldout_gate_method_comparisons_BH14.csv",
        "heldout_gate_recording_and_participant_metrics.csv",
        "heldout_gate_exact_sign_flip_wilcoxon_BH.csv",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    prediction_rows = read_rows("heldout_gate_recording_predictions.csv")
    if not prediction_rows:
        raise RuntimeError("Held-out recording prediction file is empty")
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

    if verify_manifest() < len(required):
        raise RuntimeError("Manifest did not cover all required files")
    print(f"heldout_gate verification passed: {len(prediction_rows)} recording rows, "
          f"{len(comparisons)} comparisons")


if __name__ == "__main__":
    main()
