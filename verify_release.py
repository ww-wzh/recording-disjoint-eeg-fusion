"""Verify the v7 neutral public release without model training."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "frozen" / "predictions_recording_route_a_v3.csv"
PROTOCOL = ROOT / "frozen" / "neutral_preprocessing_protocol.json"
MANIFEST = ROOT / "MANIFEST.sha256"
EXPECTED_SHA256 = "cba069dd3837e9b3894dc2649aa6c8a9ac80812de996b2dc51cc0aa056f5c5cd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> int:
    checked = 0
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = ROOT / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Manifest hash mismatch for {relative}: {observed}")
        checked += 1
    return checked


def main() -> None:
    required = [
        FINAL,
        PROTOCOL,
        ROOT / "frozen" / "predictions_recording_route_a_v3.source_manifest.json",
        MANIFEST,
        ROOT / "README.md",
        ROOT / "RUN_ORDER.md",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "ENVIRONMENT.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "revision_pipeline" / "risk_gate.py",
        ROOT / "route_a" / "route_a_lib" / "probability.py",
        ROOT / "manuscript_artifacts" / "v7_neutral" / "v7_neutral_all65_paired_comparisons.csv",
        ROOT / "manuscript_artifacts" / "v7_neutral" / "v7_neutral_MainTable2_matched_split_audit.csv",
        ROOT / "manuscript_artifacts" / "v7_neutral" / "v7_neutral_Figure_matched_split_audit.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release is incomplete: {missing}")

    observed_hash = sha256(FINAL)
    if observed_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Canonical prediction hash mismatch: {observed_hash}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "2026-09-neutral-v7":
        raise RuntimeError("Unexpected canonical protocol version")
    if protocol.get("canonical_prediction_sha256") != EXPECTED_SHA256:
        raise RuntimeError("Neutral protocol does not identify the canonical prediction hash")
    preprocessing = protocol.get("preprocessing", {})
    if preprocessing.get("within_window_zscore") is not False:
        raise RuntimeError("Canonical protocol enables within-window z-score")
    if preprocessing.get("recording_zscore") is not False:
        raise RuntimeError("Canonical protocol enables recording z-score")
    if preprocessing.get("bandpass", {}).get("high_hz") != 55.0:
        raise RuntimeError("Canonical protocol is not the neutral 0.5-55 Hz pipeline")
    if preprocessing.get("recording_notch", {}).get("quality_factor") != 30.0:
        raise RuntimeError("Canonical protocol is missing the Q=30 recording notch")
    if protocol.get("features", {}).get("dimension") != 272:
        raise RuntimeError("Canonical feature dimension is not 272")
    if protocol.get("features", {}).get("riemannian_tangent_space_dimension") != 36:
        raise RuntimeError("Canonical Riemannian tangent-space dimension is not 36")

    with FINAL.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required_fields = {
        "dataset", "protocol", "subject", "direction", "recording_id",
        "method", "true_label", "p0", "p1", "pred_label", "correct",
        "n_windows", "ensemble_size",
    }
    if not required_fields.issubset(fields):
        raise RuntimeError(f"Missing canonical columns: {sorted(required_fields - fields)}")
    if "seed" in fields or "window_id" in fields:
        raise RuntimeError("Canonical inference file contains seed/window pseudo-replicates")
    if len(rows) != 2660:
        raise RuntimeError(f"Expected 2660 rows, found {len(rows)}")

    methods = {row["method"] for row in rows}
    if len(methods) != 14:
        raise RuntimeError(f"Expected 14 methods, found {len(methods)}")
    method_counts = Counter(row["method"] for row in rows)
    if set(method_counts.values()) != {190}:
        raise RuntimeError(f"Every method must have 190 rows: {dict(method_counts)}")

    setting_counts = Counter((row["protocol"], row["direction"]) for row in rows)
    expected_settings = {
        ("cross_task", "arithmetic_to_stroop"): 546,
        ("cross_task", "stroop_to_arithmetic"): 546,
        ("loso", "arithmetic"): 840,
        ("loso", "stroop"): 728,
    }
    if setting_counts != expected_settings:
        raise RuntimeError(f"Unexpected setting coverage: {dict(setting_counts)}")

    expected_subjects = {
        ("cross_task", "arithmetic_to_stroop"): set(range(1, 14)),
        ("cross_task", "stroop_to_arithmetic"): set(range(1, 14)),
        ("loso", "arithmetic"): set(range(1, 16)),
        ("loso", "stroop"): set(range(1, 14)),
    }
    for setting, expected in expected_subjects.items():
        observed = {
            int(row["subject"])
            for row in rows
            if (row["protocol"], row["direction"]) == setting
        }
        if observed != expected:
            raise RuntimeError(f"Unexpected participants for {setting}: {sorted(observed)}")

    keys = [
        (
            row["dataset"], row["protocol"], row["direction"], row["subject"],
            row["recording_id"], row["method"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate method-recording keys found")
    for row in rows:
        if abs(float(row["p0"]) + float(row["p1"]) - 1.0) > 1e-6:
            raise RuntimeError("A probability row is not normalized")
        if int(float(row["ensemble_size"])) != 5:
            raise RuntimeError("A canonical row is not a five-seed ensemble")

    checked = verify_manifest()
    print("v7 neutral release verification passed")
    print(f"Manifest files checked: {checked}")
    print(f"Rows: {len(rows)}; methods: {len(methods)}; rows per method: 190")
    print("Participants: Cross-task=13, Arithmetic LOSO=15, Stroop LOSO=13")
    print(f"Prediction SHA-256: {observed_hash}")


if __name__ == "__main__":
    main()
