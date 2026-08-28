"""编号65：Route A v3 数据策略、8通道和物理文件哈希预检。

本脚本只读原始数据，不训练模型。它确认：
1. packet counter 已移除，返回矩阵严格为8个EEG信号列；
2. cross-task 使用 S01--S13，source 为四条 recording、target 为 r1--r3；
3. Arithmetic LOSO 使用15人，Stroop LOSO使用S01--S13；
4. 所有 train/validation/test 边界不存在相同原始文件 SHA-256；
5. 旧 Route A 结果不会被读取或覆盖。

请在 PyCharm 直接运行本文件。它完成后，确认输出目录中的
``65_预检_manifest.json`` 的 ``status`` 为 ``passed``，再运行编号66。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
OUTPUT_ROOT = HERE / "preflight"
MANIFEST_PATH = OUTPUT_ROOT / "65_预检_manifest.json"
RECORDING_PATH = OUTPUT_ROOT / "65_预检_recording清单.csv"

sys.path.insert(0, str(REVISION_ROOT))
from eeg_channel_selection import PARSER_SCHEMA, select_openbci_eeg_channels  # noqa: E402
from revision_pipeline.splits import balanced_recording_folds  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_legacy():
    path = PROJECT_ROOT / "1111.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_preflight_legacy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import legacy loader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record_rows(legacy, task: str) -> list[dict[str, object]]:
    subjects = legacy.normalize_subjects(legacy.SUBJECTS)
    rows: list[dict[str, object]] = []
    for record in legacy.build_records(subjects, task=task):
        source = Path(record.file_path).resolve()
        signal, sfreq = legacy.load_txt_eeg(str(source), default_sfreq=legacy.DEFAULT_SFREQ)
        eeg, report = select_openbci_eeg_channels(signal)
        if not report.packet_counter_removed:
            raise AssertionError(f"Packet counter was not removed: {source}")
        if eeg.shape[1] != 8 or not np.isfinite(eeg).all():
            raise AssertionError(f"Invalid pure EEG matrix: {source} -> {eeg.shape}")
        if abs(float(sfreq) - 250.0) > 1e-6:
            raise AssertionError(f"Unexpected sampling rate {sfreq} in {source}")
        window_samples = int(round(8.5 * float(sfreq)))
        stride_samples = int(round(0.5 * float(sfreq)))
        if len(eeg) < window_samples:
            raise AssertionError(f"Recording shorter than one analysis window: {source}")
        n_windows = 1 + (len(eeg) - window_samples) // stride_samples
        rows.append(
            {
                "task": task,
                "subject": int(record.subject_id),
                "raw_label": int(record.label),
                "recording_id": f"{task}:s{int(record.subject_id):02d}:r{int(record.label)}",
                "file": str(source),
                "file_sha256": _sha256(source),
                "samples": int(len(eeg)),
                "duration_seconds": float(len(eeg) / float(sfreq)),
                "windows": int(n_windows),
                "input_columns": int(signal.shape[1]),
                "output_eeg_columns": int(eeg.shape[1]),
                "packet_counter_removed": True,
                "parser_schema": PARSER_SCHEMA,
                "modulo_step_fraction": float(report.modulo_step_fraction),
            }
        )
    return rows


def _assert_hash_disjoint(rows_by_id: dict[str, dict[str, object]], left_ids: set[str], right_ids: set[str], label: str) -> None:
    left_hashes = {str(rows_by_id[value]["file_sha256"]) for value in left_ids}
    right_hashes = {str(rows_by_id[value]["file_sha256"]) for value in right_ids}
    overlap = left_hashes & right_hashes
    if overlap:
        raise AssertionError(f"Physical file hash leakage in {label}: {sorted(overlap)}")


def _recording_ids(task: str, subject: int, labels: list[int]) -> set[str]:
    return {f"{task}:s{int(subject):02d}:r{int(label)}" for label in labels}


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy["status"] != "prospective_before_v3_rerun":
        raise AssertionError("Route A v3 policy must remain prospective before model rerun")
    legacy = _load_legacy()
    rows = _record_rows(legacy, "arithmetic") + _record_rows(legacy, "stroop")
    if len(rows) != 120:
        raise AssertionError(f"Expected 120 physical recordings, observed {len(rows)}")
    if {row["parser_schema"] for row in rows} != {PARSER_SCHEMA}:
        raise AssertionError("Parser schema mismatch")
    if any(row["output_eeg_columns"] != 8 or not row["packet_counter_removed"] for row in rows):
        raise AssertionError("At least one recording did not produce pure 8-channel EEG")

    by_id = {str(row["recording_id"]): row for row in rows}
    if len(by_id) != 120:
        raise AssertionError("Recording IDs are not unique")
    duplicate_hashes: dict[str, list[str]] = {}
    for row in rows:
        duplicate_hashes.setdefault(str(row["file_sha256"]), []).append(str(row["recording_id"]))

    cross_subjects = {int(value) for value in policy["cross_task"]["participants"]}
    if cross_subjects != set(range(1, 14)):
        raise AssertionError("Cross-task cohort must be exactly S01-S13")
    source_labels = [int(value) for value in policy["cross_task"]["source_recordings"]]
    target_labels = [int(value) for value in policy["cross_task"]["target_recordings"]]
    split_rows = []
    for subject in sorted(cross_subjects):
        for source_task, target_task in (("arithmetic", "stroop"), ("stroop", "arithmetic")):
            source_ids = _recording_ids(source_task, subject, source_labels)
            target_ids = _recording_ids(target_task, subject, target_labels)
            _assert_hash_disjoint(by_id, source_ids, target_ids, f"cross-task S{subject:02d} {source_task}->{target_task}")
            source_labels_raw = []
            source_recordings = []
            for rid in sorted(source_ids):
                source_recordings.extend([rid] * int(by_id[rid]["windows"]))
                source_labels_raw.extend([int(by_id[rid]["raw_label"])] * int(by_id[rid]["windows"]))
            folds = balanced_recording_folds(np.asarray(source_labels_raw), np.asarray(source_recordings, dtype=object), seed=1335)
            for fold_number, split in enumerate(folds, start=1):
                train_ids = set(np.asarray(source_recordings, dtype=object)[split.train].tolist())
                validation_ids = set(np.asarray(source_recordings, dtype=object)[split.validation].tolist())
                _assert_hash_disjoint(by_id, train_ids, validation_ids, f"cross-task inner S{subject:02d} fold{fold_number}")
                split_rows.append({
                    "protocol": "cross_task",
                    "subject": subject,
                    "source_task": source_task,
                    "target_task": target_task,
                    "fold": fold_number,
                    "train_recordings": len(train_ids),
                    "validation_recordings": len(validation_ids),
                    "target_recordings": len(target_ids),
                    "target_class0_recordings": sum(int(by_id[r]["raw_label"]) < 2 for r in target_ids),
                    "target_class1_recordings": sum(int(by_id[r]["raw_label"]) >= 2 for r in target_ids),
                })

    for task, cohort in (("arithmetic", set(range(1, 16))), ("stroop", set(range(1, 14)))):
        for held_out in sorted(cohort):
            train_ids = {rid for rid, row in by_id.items() if row["task"] == task and int(row["subject"]) in cohort - {held_out}}
            test_ids = {rid for rid, row in by_id.items() if row["task"] == task and int(row["subject"]) == held_out}
            _assert_hash_disjoint(by_id, train_ids, test_ids, f"{task} LOSO held S{held_out:02d}")
            if not train_ids or not test_ids:
                raise AssertionError(f"Empty LOSO split {task} S{held_out:02d}")
            split_rows.append({
                "protocol": f"{task}_loso",
                "subject": held_out,
                "source_task": task,
                "target_task": task,
                "fold": 0,
                "train_recordings": len(train_ids),
                "validation_recordings": 0,
                "target_recordings": len(test_ids),
                "target_class0_recordings": sum(int(by_id[r]["raw_label"]) < 2 for r in test_ids),
                "target_class1_recordings": sum(int(by_id[r]["raw_label"]) >= 2 for r in test_ids),
            })

    duplicate_groups = [values for values in duplicate_hashes.values() if len(values) > 1]
    expected = {
        "cross_task_recordings": 13 * 2 * 3,
        "arithmetic_loso_recordings": 15 * 4,
        "stroop_loso_recordings": 13 * 4,
        "total_recordings_per_method": 190,
    }
    if expected["cross_task_recordings"] != int(policy["cross_task"]["expected_recordings"]):
        raise AssertionError("Cross-task expected count mismatch")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    import csv

    with RECORDING_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "script": Path(__file__).name,
        "status": "passed",
        "policy_file": POLICY_PATH.name,
        "parser_schema": PARSER_SCHEMA,
        "physical_recordings": len(rows),
        "duplicate_hash_groups_observed_but_excluded_from_cross_boundary": len(duplicate_groups),
        "duplicate_hash_groups": duplicate_groups,
        "expected": expected,
        "cross_task_source_recordings": source_labels,
        "cross_task_target_recordings": target_labels,
        "cross_task_participants": sorted(cross_subjects),
        "arithmetic_loso_participants": sorted(range(1, 16)),
        "stroop_loso_participants": sorted(range(1, 14)),
        "split_rows": split_rows,
        "old_results_read": False,
        "old_results_overwritten": False,
        "policy_sha256": _sha256(POLICY_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Route A v3 preflight passed: {len(rows)} physical recordings")
    print(f"Cross-task target recordings: {expected['cross_task_recordings']}")
    print(f"Arithmetic LOSO recordings: {expected['arithmetic_loso_recordings']}")
    print(f"Stroop LOSO recordings: {expected['stroop_loso_recordings']}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
