"""编号78：Route A v3 EEGNet/EEG-Conformer 公共运行工具。

该文件不直接运行模型。编号79和80分别调用这里的 run_v3_model()。
所有深度基线都使用纯8通道、8.5 s/0.5 s窗口、recording-disjoint
Cross-task和participant-disjoint nested LOSO。旧 route_a/results 不会被读取。
"""

from __future__ import annotations

import importlib.util
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
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
CACHE_DIR = HERE / "08_深度基线_raw_cache_v3"
SEEDS = [1335, 1388, 1441, 1494, 1547]
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 80,
    "patience": 12,
    "selection_metric": "recording-level mean log probability",
}
ARCHITECTURES = {
    "eegnet": {
        "F1": 8,
        "D": 2,
        "F2": 16,
        "temporal_kernel": 125,
        "dropout": 0.5,
        "pool1": 4,
        "pool2": 8,
    },
    "eeg_conformer": {
        "embedding_dim": 40,
        "temporal_kernel": 25,
        "pool_kernel": 75,
        "pool_stride": 15,
        "attention_heads": 4,
        "transformer_layers": 2,
        "feedforward_dim": 160,
        "dropout": 0.5,
    },
}

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "route_a"))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.splits import (  # noqa: E402
    balanced_recording_folds,
    outer_loso,
    subject_loso_folds,
)
from route_a_lib.data import RawBundle, load_raw_bundle  # noqa: E402


def _load_deep_runner():
    path = REVISION_ROOT / "route_a" / "run_deep_baselines.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_deep_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _subset(bundle: RawBundle, *, subjects: set[int] | None = None, raw_labels: set[int] | None = None) -> RawBundle:
    mask = np.ones(len(bundle.labels), dtype=bool)
    if subjects is not None:
        mask &= np.isin(bundle.subjects, sorted(subjects))
    if raw_labels is not None:
        mask &= np.isin(bundle.labels_raw, sorted(raw_labels))
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"Filtering removed all rows from {bundle.task}")
    return bundle.subset(indices)


def _checkpoint_complete(path: Path, method: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return False
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"]
    return bool(
        not frame.empty
        and set(frame["method"].astype(str)) == {method}
        and set(frame["seed"].astype(int)) == {int(seed)}
        and not frame.duplicated(keys).any()
        and np.isfinite(frame[["p0", "p1"]].to_numpy(dtype=float)).all()
        and np.allclose(frame[["p0", "p1"]].sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    )


def _protocol_digest() -> str:
    return sha256_file(POLICY_PATH)


def _run_cross_task(
    runner,
    model_name: str,
    arithmetic: RawBundle,
    stroop: RawBundle,
    subjects: list[int],
    output_root: Path,
    device: torch.device,
) -> None:
    source_labels = {0, 1, 2, 3}
    target_labels = {1, 2, 3}
    pairs = [
        (arithmetic, stroop, "arithmetic_to_stroop"),
        (stroop, arithmetic, "stroop_to_arithmetic"),
    ]
    for subject in subjects:
        for source_all, target_all, direction in pairs:
            source = _subset(source_all, subjects={subject}, raw_labels=source_labels)
            target = _subset(target_all, subjects={subject}, raw_labels=target_labels)
            for seed in SEEDS:
                checkpoint = output_root / "checkpoints" / "cross_task" / f"s{subject:02d}_{direction}_seed{seed}.csv"
                diagnostic_path = checkpoint.with_suffix(".diagnostic.json")
                progress_path = checkpoint.with_suffix(".progress.json")
                if _checkpoint_complete(checkpoint, model_name, seed) and diagnostic_path.exists():
                    print(f"[skip] {model_name} cross-task S{subject:02d} {direction} seed={seed}")
                    continue
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                splits = balanced_recording_folds(source.labels_raw, source.recordings, seed=seed)
                probability, diagnostic = runner.nested_deep_predict(
                    source,
                    target,
                    splits,
                    model_name=model_name,
                    architecture_config=ARCHITECTURES[model_name],
                    training_config=TRAINING_CONFIG,
                    seed=seed,
                    device=device,
                    progress_path=progress_path,
                    protocol_digest=_protocol_digest(),
                )
                runner.prediction_rows(
                    target,
                    probability,
                    protocol="cross_task",
                    subject=subject,
                    direction=direction,
                    method=model_name,
                    seed=seed,
                ).to_csv(checkpoint, index=False, lineterminator="\n")
                diagnostic.update(
                    {
                        "protocol": "cross_task",
                        "subject": int(subject),
                        "direction": direction,
                        "seed": int(seed),
                        "protocol_sha256": _protocol_digest(),
                        "target_raw_recordings": [1, 2, 3],
                    }
                )
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostic_path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"[done] {model_name} cross-task S{subject:02d} {direction} seed={seed}")


def _run_loso(
    runner,
    model_name: str,
    bundle: RawBundle,
    task: str,
    cohort: list[int],
    output_root: Path,
    device: torch.device,
) -> None:
    outer = {int(subject): (train, test) for subject, train, test in outer_loso(bundle.subjects)}
    for subject in cohort:
        train_index, test_index = outer[int(subject)]
        source = bundle.subset(train_index)
        target = bundle.subset(test_index)
        splits = subject_loso_folds(source.subjects)
        for seed in SEEDS:
            checkpoint = output_root / "checkpoints" / f"loso_{task}" / f"s{subject:02d}_{task}_seed{seed}.csv"
            diagnostic_path = checkpoint.with_suffix(".diagnostic.json")
            progress_path = checkpoint.with_suffix(".progress.json")
            if _checkpoint_complete(checkpoint, model_name, seed) and diagnostic_path.exists():
                print(f"[skip] {model_name} {task} LOSO S{subject:02d} seed={seed}")
                continue
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            probability, diagnostic = runner.nested_deep_predict(
                source,
                target,
                splits,
                model_name=model_name,
                architecture_config=ARCHITECTURES[model_name],
                training_config=TRAINING_CONFIG,
                seed=seed,
                device=device,
                progress_path=progress_path,
                protocol_digest=_protocol_digest(),
            )
            runner.prediction_rows(
                target,
                probability,
                protocol="loso",
                subject=subject,
                direction=task,
                method=model_name,
                seed=seed,
            ).to_csv(checkpoint, index=False, lineterminator="\n")
            diagnostic.update(
                {
                    "protocol": "loso",
                    "subject": int(subject),
                    "direction": task,
                    "seed": int(seed),
                    "protocol_sha256": _protocol_digest(),
                }
            )
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[done] {model_name} {task} LOSO S{subject:02d} seed={seed}")


def run_v3_model(model_name: str) -> None:
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unknown deep model: {model_name}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if [int(value) for value in policy["seeds"]] != SEEDS:
        raise AssertionError("Frozen seed list differs from v3 protocol")
    runner = _load_deep_runner()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arithmetic = load_raw_bundle(
        PROJECT_ROOT,
        "arithmetic",
        list(range(1, 16)),
        CACHE_DIR,
        window_seconds=8.5,
        stride_seconds=0.5,
    )
    stroop = load_raw_bundle(
        PROJECT_ROOT,
        "stroop",
        list(range(1, 14)),
        CACHE_DIR,
        window_seconds=8.5,
        stride_seconds=0.5,
    )
    output_root = HERE / ("08_eegnet" if model_name == "eegnet" else "09_eeg_conformer")
    output_root.mkdir(parents=True, exist_ok=True)
    _run_cross_task(
        runner,
        model_name,
        arithmetic,
        stroop,
        [int(value) for value in policy["cross_task"]["participants"]],
        output_root,
        device,
    )
    _run_loso(
        runner,
        model_name,
        arithmetic,
        "arithmetic",
        [int(value) for value in policy["loso"]["arithmetic"]["participants"]],
        output_root,
        device,
    )
    _run_loso(
        runner,
        model_name,
        stroop,
        "stroop",
        [int(value) for value in policy["loso"]["stroop"]["participants"]],
        output_root,
        device,
    )
    prediction_files = sorted((output_root / "checkpoints").glob("**/*.csv"))
    expected = (26 + 15 + 13) * len(SEEDS)
    if len(prediction_files) != expected:
        raise AssertionError(f"Expected {expected} deep checkpoints, got {len(prediction_files)}")
    raw = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    raw_keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"]
    if raw.duplicated(raw_keys).any():
        raise AssertionError(f"{model_name} raw predictions contain duplicate keys")
    windows = median_seed_ensemble(raw, SEEDS)
    recordings = aggregate_recordings(windows)
    validate_frozen_predictions(recordings)
    if len(recordings) != 190:
        raise AssertionError(f"Expected 190 {model_name} recording rows, got {len(recordings)}")
    raw.to_csv(output_root / "raw_seed_predictions.csv", index=False, lineterminator="\n")
    windows.to_csv(output_root / "window_ensemble_predictions.csv", index=False, lineterminator="\n")
    recordings.to_csv(output_root / "recording_predictions.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "model": model_name,
        "architecture": ARCHITECTURES[model_name],
        "training": TRAINING_CONFIG,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "seed_aggregation": "window-level median across five seeds, then recording geometric mean",
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "cross_task_cells": 26,
        "arithmetic_loso_cells": 15,
        "stroop_loso_cells": 13,
        "recording_rows": int(len(recordings)),
        "old_route_a_results_read": False,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    (output_root / f"{model_name}_v3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed Route A v3 {model_name}: {output_root}")
    print(recordings.groupby(["protocol", "direction"], as_index=False)["correct"].mean().to_string(index=False))
