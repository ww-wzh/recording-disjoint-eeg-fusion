"""Verify the final JMBE release. Run directly in PyCharm."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "frozen" / "predictions_recording_jmbe_final.csv"
MANIFEST = ROOT / "MANIFEST.sha256"
EXPECTED_SHA256 = "1db8ae4b67520cf6c4ae07d4002971a3debe0bab395225c10f6265f1fab6d6d1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        MANIFEST,
        ROOT / "README.md",
        ROOT / "SUPERVISOR_REQUESTED_AUDITS.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "eeg_feature_pipeline.py",
        ROOT / "revision_pipeline" / "splits.py",
        ROOT / "revision_pipeline" / "risk_gate.py",
        ROOT / "route_a" / "frozen" / "predictions_recording_route_a.csv",
        ROOT / "supervisor_requested_audits" / "33_输出_Stroop_LOSO_14方法冻结统计表图" / "01_冻结预测_Stroop_LOSO_14方法Recording.csv",
        ROOT / "manuscript_artifacts" / "table_all_methods.csv",
        ROOT / "manuscript_artifacts" / "figure1_accuracy.png",
        ROOT / "manuscript_artifacts" / "figure2_participant_differences.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release is incomplete: {missing}")

    observed_hash = sha256(FINAL)
    if observed_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Final prediction hash mismatch: {observed_hash}")

    with FINAL.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required_fields = {
        "dataset", "protocol", "subject", "direction", "recording_id",
        "method", "true_label", "p0", "p1", "pred_label", "setting",
    }
    if not required_fields.issubset(fields):
        raise RuntimeError(f"Missing final columns: {sorted(required_fields - fields)}")
    if "seed" in fields or "window_id" in fields:
        raise RuntimeError("Final inferential file contains seed/window rows")
    if len(rows) != 3360:
        raise RuntimeError(f"Expected 3360 rows, found {len(rows)}")

    methods = {row["method"] for row in rows}
    subjects = {int(row["subject"]) for row in rows}
    setting_counts = Counter(row["setting"] for row in rows)
    if len(methods) != 14 or subjects != set(range(1, 16)):
        raise RuntimeError("Method or participant coverage is incomplete")
    if setting_counts != {
        "cross_task_bidirectional": 1680,
        "loso_arithmetic": 840,
        "loso_stroop": 840,
    }:
        raise RuntimeError(f"Unexpected setting counts: {dict(setting_counts)}")

    keys = [
        (
            row["dataset"], row["protocol"], row["subject"], row["direction"],
            row["recording_id"], row["method"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate method-recording keys found")
    for row in rows:
        total = float(row["p0"]) + float(row["p1"])
        if abs(total - 1.0) > 1e-6:
            raise RuntimeError("A probability row is not normalized")

    checked = verify_manifest()
    print("Final JMBE release verification passed")
    print(f"Manifest files checked: {checked}")
    print(f"Rows: {len(rows)}; methods: {len(methods)}; participants: {len(subjects)}")
    print(f"Prediction SHA-256: {observed_hash}")


if __name__ == "__main__":
    main()
