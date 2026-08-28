"""编号72：从 67--69 的 seed predictions 生成 v3 核心冻结文件。

本脚本不训练模型，只执行：五 seed 窗口概率中位数、recording 几何平均，
并写出唯一的核心冻结预测文件。后续 DASF、CB-SF 和统计脚本只能读取这里
生成的文件，不得直接读取单个 seed 作为统计样本。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import sys

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    validate_frozen_predictions,
)

SEEDS = [1335, 1388, 1441, 1494, 1547]
OUTPUT_ROOT = HERE / "02_frozen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    core_root = HERE / "01_core"
    raw_paths = [
        core_root / "cross_task" / "raw_seed_predictions.csv",
        core_root / "loso_arithmetic" / "raw_seed_predictions.csv",
        core_root / "loso_stroop" / "raw_seed_predictions.csv",
    ]
    if any(not path.exists() for path in raw_paths):
        missing = [str(path) for path in raw_paths if not path.exists()]
        raise FileNotFoundError(f"Missing core raw seed files; run 71 first: {missing}")
    raw = pd.concat([pd.read_csv(path) for path in raw_paths], ignore_index=True)
    window = median_seed_ensemble(raw, SEEDS)
    recording = aggregate_recordings(window)
    if len(recording) != 190 * 4:
        raise AssertionError(f"Expected 760 core recording-method rows, got {len(recording)}")
    validate_frozen_predictions(recording)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    window_path = OUTPUT_ROOT / "core_window_ensemble.csv"
    recording_path = OUTPUT_ROOT / "core_recording_predictions.csv"
    window.to_csv(window_path, index=False, lineterminator="\n")
    recording.to_csv(recording_path, index=False, lineterminator="\n")
    manifest = {
        "status": "passed",
        "protocol_version": "2026-08-22-route-a-v3",
        "seed_aggregation": "per-class median at window level",
        "recording_aggregation": "geometric mean of window probabilities, then argmax",
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "seed_is_statistical_unit": False,
        "window_rows": int(len(window)),
        "recording_rows": int(len(recording)),
        "recordings_per_method": 190,
        "methods": sorted(str(value) for value in recording["method"].unique()),
        "source_raw_sha256": {str(path.relative_to(HERE)): sha256(path) for path in raw_paths},
        "window_prediction_sha256": sha256(window_path),
        "recording_prediction_sha256": sha256(recording_path),
        "old_route_a_results_read": False,
    }
    (OUTPUT_ROOT / "72_核心冻结_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Frozen v3 core windows: {window_path}")
    print(f"Frozen v3 core recordings: {recording_path}")
    print(json.dumps({"recording_rows": len(recording), "methods": manifest["methods"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
