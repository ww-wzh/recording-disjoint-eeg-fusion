"""编号77：Route A v3 Riemannian MDM 和 tangent-space logistic 基线。

旧的 route_a Riemannian runner 使用旧协议，因此本文件重新固定：
纯 8 通道、Cross-task target r1--r3、Arithmetic LOSO S01--S15、
Stroop LOSO S01--S13。Riemannian 模型是确定性的，每个 outer cell 只拟合
一次，然后把同一概率写入五个 seed 槽位；seed 不作为统计单位。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：07_riemannian
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
CORE_ROOT = HERE / "01_core"
OUTPUT_ROOT = HERE / "07_riemannian"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "route_a"))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.splits import outer_loso  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402
from run_riemannian_baselines import fit_predict_riemannian  # noqa: E402


SEEDS = [1335, 1388, 1441, 1494, 1547]
METHODS = {"riemann_ts_logreg", "riemann_mdm"}
RIEMANN_CONFIG = {
    "tangent_metric": "riemann",
    "logistic_C": 1.0,
    "logistic_max_iter": 4000,
    "mdm_metric": "riemann",
}


def _subset(bundle: FeatureBundle, subjects: set[int], raw_labels: set[int] | None = None) -> FeatureBundle:
    mask = np.isin(bundle.subjects, sorted(subjects))
    if raw_labels is not None:
        mask &= np.isin(bundle.labels_raw, sorted(raw_labels))
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"Empty {bundle.task} subset")
    return bundle.subset(rows)


def _prediction_frame(
    target: FeatureBundle,
    probability: dict[str, np.ndarray],
    *,
    protocol: str,
    subject: int,
    direction: str,
) -> pd.DataFrame:
    rows = []
    for method in sorted(METHODS):
        values = probability[method]
        if len(values) != len(target.labels):
            raise ValueError(f"{method} probability count mismatch")
        for seed in SEEDS:
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": "openbci",
                        "protocol": protocol,
                        "subject": int(subject),
                        "direction": direction,
                        "recording_id": target.recordings,
                        "window_id": target.window_ids,
                        "method": method,
                        "seed": int(seed),
                        "true_label": target.labels,
                        "p0": values[:, 0],
                        "p1": values[:, 1],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _checkpoint_path(protocol: str, subject: int, direction: str) -> Path:
    folder = CHECKPOINT_ROOT / protocol
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"s{int(subject):02d}_{direction}.csv"


def _complete(path: Path, expected_recordings: int, subject: int, direction: str, protocol: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return False
    keys = ["protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"]
    return bool(
        len(frame) > 0
        and set(frame["protocol"].astype(str)) == {protocol}
        and set(frame["subject"].astype(int)) == {int(subject)}
        and set(frame["direction"].astype(str)) == {direction}
        and set(frame["method"].astype(str)) == METHODS
        and set(frame["seed"].astype(int)) == set(SEEDS)
        and frame["recording_id"].nunique() == int(expected_recordings)
        and not frame.duplicated(keys).any()
        and np.isfinite(frame[["p0", "p1"]].to_numpy(dtype=float)).all()
        and np.allclose(frame[["p0", "p1"]].sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    )


def _run_cross_task(
    arithmetic: FeatureBundle,
    stroop: FeatureBundle,
    subjects: list[int],
) -> None:
    source_labels = {0, 1, 2, 3}
    target_labels = {1, 2, 3}
    pairs = [
        (arithmetic, stroop, "arithmetic_to_stroop"),
        (stroop, arithmetic, "stroop_to_arithmetic"),
    ]
    for subject in subjects:
        for source_all, target_all, direction in pairs:
            source = _subset(source_all, {subject}, source_labels)
            target = _subset(target_all, {subject}, target_labels)
            path = _checkpoint_path("cross_task", subject, direction)
            if _complete(path, 3, subject, direction, "cross_task"):
                print(f"[skip] Riemannian cross-task S{subject:02d} {direction}")
                continue
            probability = fit_predict_riemannian(
                source.covariances,
                source.labels,
                target.covariances,
                seed=SEEDS[0],
                config=RIEMANN_CONFIG,
            )
            _prediction_frame(
                target,
                probability,
                protocol="cross_task",
                subject=subject,
                direction=direction,
            ).to_csv(path, index=False, lineterminator="\n")
            if not _complete(path, 3, subject, direction, "cross_task"):
                raise AssertionError(f"Invalid Riemannian checkpoint: {path}")
            print(f"[done] Riemannian cross-task S{subject:02d} {direction}")


def _run_loso(bundle: FeatureBundle, task: str, subjects: list[int]) -> None:
    outer = {int(subject): (train, test) for subject, train, test in outer_loso(bundle.subjects)}
    for subject in subjects:
        path = _checkpoint_path("loso", subject, task)
        if _complete(path, 4, subject, task, "loso"):
            print(f"[skip] Riemannian {task} LOSO S{subject:02d}")
            continue
        train_index, test_index = outer[int(subject)]
        source = bundle.subset(train_index)
        target = bundle.subset(test_index)
        probability = fit_predict_riemannian(
            source.covariances,
            source.labels,
            target.covariances,
            seed=SEEDS[0],
            config=RIEMANN_CONFIG,
        )
        _prediction_frame(
            target,
            probability,
            protocol="loso",
            subject=subject,
            direction=task,
        ).to_csv(path, index=False, lineterminator="\n")
        if not _complete(path, 4, subject, task, "loso"):
            raise AssertionError(f"Invalid Riemannian checkpoint: {path}")
        print(f"[done] Riemannian {task} LOSO S{subject:02d}")


def _summarize(recordings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    participant = recordings.groupby(["protocol", "subject", "direction", "method"], as_index=False)["correct"].mean()
    for keys, group in participant.groupby(["protocol", "direction", "method"]):
        rows.append(
            {
                "protocol": keys[0],
                "direction": keys[1],
                "method": keys[2],
                "n_subjects": int(len(group)),
                "mean_participant_accuracy": float(group["correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "direction", "method"]).reset_index(drop=True)


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if [int(value) for value in policy["seeds"]] != SEEDS:
        raise AssertionError("Frozen seed list differs from Route A v3")
    try:
        import pyriemann
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 pyriemann，请先在 cuda129 环境安装后再运行 77") from exc

    arithmetic, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))
    cross_subjects = [int(value) for value in policy["cross_task"]["participants"]]
    arithmetic_subjects = [int(value) for value in policy["loso"]["arithmetic"]["participants"]]
    stroop_subjects = [int(value) for value in policy["loso"]["stroop"]["participants"]]
    _run_cross_task(arithmetic, stroop, cross_subjects)
    _run_loso(arithmetic, "arithmetic", arithmetic_subjects)
    _run_loso(stroop, "stroop", stroop_subjects)

    files = sorted(CHECKPOINT_ROOT.glob("**/*.csv"))
    if len(files) != 26 + 15 + 13:
        raise AssertionError(f"Expected 54 Riemannian cell checkpoints, got {len(files)}")
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw_keys = ["protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"]
    if raw.duplicated(raw_keys).any():
        raise AssertionError("Riemannian raw predictions contain duplicate keys")
    windows = median_seed_ensemble(raw, SEEDS)
    recordings = aggregate_recordings(windows)
    validate_frozen_predictions(recordings)
    if len(recordings) != 190 * 2:
        raise AssertionError(f"Expected 380 Riemannian recording rows, got {len(recordings)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT_ROOT / "raw_seed_predictions.csv", index=False, lineterminator="\n")
    windows.to_csv(OUTPUT_ROOT / "window_ensemble_predictions.csv", index=False, lineterminator="\n")
    recordings.to_csv(OUTPUT_ROOT / "recording_predictions.csv", index=False, lineterminator="\n")
    summary = _summarize(recordings)
    summary.to_csv(OUTPUT_ROOT / "participant_summary.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "methods": sorted(METHODS),
        "config": RIEMANN_CONFIG,
        "pyriemann": pyriemann.__version__,
        "cross_task_cells": 26,
        "arithmetic_loso_cells": 15,
        "stroop_loso_cells": 13,
        "recording_rows": int(len(recordings)),
        "deterministic_seed_policy": "one fit per cell copied to five seed slots; seeds are not statistical units",
        "evaluation_unit": "recording",
        "statistical_unit": "participant",
        "old_route_a_results_read": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    (OUTPUT_ROOT / "77_Riemannian_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Route A v3 Riemannian baselines completed")
    print(summary.to_string(index=False))
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
