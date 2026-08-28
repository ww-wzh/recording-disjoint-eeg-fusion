"""编号71：汇总 67--69 已完成的核心 checkpoint 并做完整性校验。

67--69 已经分别把预测写入 checkpoint。67 的两个方向采用独立目录，
因此本脚本只负责汇总和校验，不会重新训练任何模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CORE_ROOT = HERE / "01_core"
OUTPUT = CORE_ROOT / "71_核心输出完整性_manifest.json"


def _prediction_files(checkpoint_root: Path) -> list[Path]:
    return sorted(
        path
        for path in checkpoint_root.glob("**/*.csv")
        if not path.name.endswith(".meta.csv")
        and not path.name.endswith(".meta_predictions.csv")
        and path.name not in {"raw_seed_predictions.csv", "diagnostics_seed.csv"}
    )


def _validate_frame(frame: pd.DataFrame, *, protocol: str, expected_groups: int, recordings_per_group: int) -> dict[str, object]:
    required = {"protocol", "subject", "direction", "recording_id", "window_id", "method", "seed", "true_label", "p0", "p1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"{protocol} missing columns: {missing}")
    keys = ["protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"]
    duplicate_rows = int(frame.duplicated(keys).sum())
    if duplicate_rows:
        raise AssertionError(f"{protocol} duplicate prediction rows: {duplicate_rows}")
    groups = frame.groupby(["subject", "direction", "seed"], sort=False)
    if len(groups) != expected_groups:
        raise AssertionError(f"{protocol} expected {expected_groups} subject-direction-seed groups, got {len(groups)}")
    for group_key, group in groups:
        methods = set(group["method"].astype(str))
        if methods != {"always_nn", "rf", "extra_trees", "always_fuse"}:
            raise AssertionError(f"{protocol} group {group_key} methods={sorted(methods)}")
        rec_count = int(group["recording_id"].nunique())
        if rec_count != recordings_per_group:
            raise AssertionError(f"{protocol} group {group_key} recordings={rec_count}")
        if group[["p0", "p1"]].isna().any().any():
            raise AssertionError(f"{protocol} group {group_key} has missing probabilities")
        probabilities = group[["p0", "p1"]].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError(f"{protocol} group {group_key} has invalid probabilities")
    return {
        "rows": int(len(frame)),
        "groups": int(len(groups)),
        "unique_subject_direction_cells": int(frame[["subject", "direction"]].drop_duplicates().shape[0]),
        "unique_seeds": sorted(int(value) for value in frame["seed"].unique()),
        "methods": sorted(str(value) for value in frame["method"].unique()),
        "duplicate_rows": duplicate_rows,
    }


def main() -> None:
    cross_root = CORE_ROOT / "cross_task"
    cross_files = _prediction_files(cross_root)
    if len(cross_files) != 26 * 5:
        raise AssertionError(f"Cross-task prediction checkpoint count should be 130, got {len(cross_files)}")
    cross_frame = pd.concat([pd.read_csv(path) for path in cross_files], ignore_index=True)
    cross_frame.to_csv(cross_root / "raw_seed_predictions.csv", index=False, lineterminator="\n")
    cross_diagnostics = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cross_root.glob("**/*.diagnostic.json"))]
    if len(cross_diagnostics) != 26 * 5:
        raise AssertionError(f"Cross-task diagnostic count should be 130, got {len(cross_diagnostics)}")
    pd.DataFrame(cross_diagnostics).to_csv(cross_root / "diagnostics_seed.csv", index=False, lineterminator="\n")

    summaries = {
        "cross_task": _validate_frame(cross_frame, protocol="cross_task", expected_groups=26 * 5, recordings_per_group=3),
    }
    for name, expected_groups in (("loso_arithmetic", 15 * 5), ("loso_stroop", 13 * 5)):
        path = CORE_ROOT / name / "raw_seed_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        summaries[name] = _validate_frame(frame, protocol=name, expected_groups=expected_groups, recordings_per_group=4)
        if len(pd.read_csv(CORE_ROOT / name / "diagnostics_seed.csv")) != expected_groups:
            raise AssertionError(f"{name} diagnostics count mismatch")

    manifest = {
        "status": "passed",
        "core_results_are_v3_only": True,
        "old_route_a_results_read": False,
        "summaries": summaries,
        "cross_task_checkpoint_count": len(cross_files),
        "cross_task_diagnostic_count": len(cross_diagnostics),
        "cross_task_raw_seed_path": str(cross_root / "raw_seed_predictions.csv"),
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Route A v3 core output consolidation passed")
    for name, summary in summaries.items():
        print(name, summary)
    print(f"Manifest: {OUTPUT}")


if __name__ == "__main__":
    main()
