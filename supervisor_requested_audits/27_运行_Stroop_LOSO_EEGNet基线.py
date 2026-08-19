"""直接运行：补齐 Stroop 的 fully nested LOSO EEGNet 基线。

每个外层参与者内部使用完整的 14 折 participant-LOSO 选择训练 epoch，随后在
14 名外层训练参与者上重训，并预测唯一的外层目标参与者。程序覆盖 15 名参与者和
5 个冻结种子，支持断点续跑，结果写入编号 28 的独立目录。

运行方法：在 PyCharm 中直接运行本文件，不需要填写任何命令行参数。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
PROJECT_ROOT = REVISION_ROOT
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
OUTPUT_ROOT = HERE / "28_输出_Stroop_LOSO_EEGNet基线"
CHECKPOINT_ROOT = OUTPUT_ROOT / "01_检查点_可续跑"
CACHE_ROOT = ROUTE_A_ROOT / "cache" / "raw_8ch_bandpass_0p5_45_v1"

RAW_OUTPUT = OUTPUT_ROOT / "02_结果_EEGNet窗口种子预测.csv"
RECORDING_OUTPUT = OUTPUT_ROOT / "03_结果_EEGNet五种子Recording预测.csv"
SUBJECT_OUTPUT = OUTPUT_ROOT / "04_结果_EEGNet参与者准确率.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "05_结果_EEGNet汇总.csv"
MANIFEST_OUTPUT = OUTPUT_ROOT / "06_记录_运行信息和哈希.json"
PROTOCOL_PATH = ROUTE_A_ROOT / "protocol_route_a.json"

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(ROUTE_A_ROOT))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.splits import outer_loso, subject_loso_folds  # noqa: E402
from route_a.run_deep_baselines import nested_deep_predict, prediction_rows  # noqa: E402
from route_a_lib.data import load_raw_bundle  # noqa: E402


MODEL_NAME = "eegnet"
SEEDS = [1335, 1388, 1441, 1494, 1547]


def prediction_path(subject: int, seed: int) -> Path:
    return CHECKPOINT_ROOT / f"01_目标窗口预测_S{subject:02d}_Seed{seed}.csv"


def diagnostic_path(subject: int, seed: int) -> Path:
    return CHECKPOINT_ROOT / f"02_嵌套诊断_S{subject:02d}_Seed{seed}.json"


def progress_path(subject: int, seed: int) -> Path:
    return CHECKPOINT_ROOT / f"03_内层Epoch进度_S{subject:02d}_Seed{seed}.json"


def checkpoint_complete(subject: int, seed: int, protocol_digest: str) -> bool:
    prediction = prediction_path(subject, seed)
    diagnostic = diagnostic_path(subject, seed)
    if not prediction.exists() or not diagnostic.exists():
        return False
    try:
        frame = pd.read_csv(prediction)
        details = json.loads(diagnostic.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    keys = ["recording_id", "window_id", "method", "seed"]
    values = frame[["p0", "p1"]].to_numpy(dtype=np.float64)
    epochs = details.get("inner_selected_epochs", [])
    return bool(
        not frame.empty
        and set(frame["dataset"].astype(str)) == {"openbci"}
        and set(frame["protocol"].astype(str)) == {"loso"}
        and set(frame["direction"].astype(str)) == {"stroop"}
        and set(frame["subject"].astype(int)) == {int(subject)}
        and set(frame["method"].astype(str)) == {MODEL_NAME}
        and set(frame["seed"].astype(int)) == {int(seed)}
        and frame["recording_id"].nunique() == 4
        and not frame.duplicated(keys).any()
        and np.isfinite(values).all()
        and (values >= 0.0).all()
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        and details.get("protocol") == "loso"
        and details.get("direction") == "stroop"
        and details.get("subject") == int(subject)
        and details.get("seed") == int(seed)
        and details.get("protocol_sha256") == protocol_digest
        and details.get("outer_subject_excluded") is True
        and details.get("inner_participants") == 14
        and details.get("inner_folds") == 14
        and details.get("test_recordings") == 4
        and len(epochs) == 14
        and all(int(value) >= 1 for value in epochs)
    )


def combined_checkpoint_digest(paths: list[Path]) -> str:
    rows = [
        {"file": path.name, "sha256": sha256_file(path)}
        for path in sorted(paths)
        if path.exists()
    ]
    canonical = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260728)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(recordings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = (
        recordings.groupby(["dataset", "protocol", "subject", "direction", "method"])[
            "correct"
        ]
        .agg(accuracy="mean", n_recordings="count")
        .reset_index()
    )
    values = subjects["accuracy"].to_numpy(dtype=np.float64)
    low, high = bootstrap_interval(values)
    summary = pd.DataFrame(
        [
            {
                "dataset": "openbci",
                "protocol": "loso",
                "direction": "stroop",
                "method": MODEL_NAME,
                "n_subjects": int(len(values)),
                "mean_accuracy": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        ]
    )
    return subjects, summary


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("EEGNet 正式实验需要 CUDA，但当前环境 torch.cuda.is_available() 为 False")
    device = torch.device("cuda:0")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if [int(value) for value in protocol["seeds"]] != SEEDS:
        raise RuntimeError("Route A 协议中的五个种子发生变化")
    protocol_digest = sha256_file(PROTOCOL_PATH)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = load_raw_bundle(PROJECT_ROOT, "stroop", list(range(1, 16)), CACHE_ROOT)
    outer = {subject: (train, test) for subject, train, test in outer_loso(dataset.subjects)}

    print("[1/3] 运行 15 名参与者 × 5 种子的 fully nested Stroop LOSO EEGNet")
    for subject in range(1, 16):
        train_index, test_index = outer[subject]
        source = dataset.subset(train_index)
        target = dataset.subset(test_index)
        if subject in set(source.subjects.astype(int)):
            raise AssertionError("外层目标参与者进入了 EEGNet 训练数据")
        splits = subject_loso_folds(source.subjects)
        if len(splits) != 14:
            raise AssertionError("每个外层 EEGNet 单元必须包含 14 折内层 participant-LOSO")
        for seed in SEEDS:
            if checkpoint_complete(subject, seed, protocol_digest):
                print(f"[skip] EEGNet S{subject:02d} seed={seed}")
                continue
            print(f"[start] EEGNet S{subject:02d} seed={seed}")
            probability, diagnostics = nested_deep_predict(
                source,
                target,
                splits,
                model_name=MODEL_NAME,
                architecture_config=protocol[MODEL_NAME],
                training_config=protocol["deep_training"],
                seed=seed,
                device=device,
                progress_path=progress_path(subject, seed),
                protocol_digest=protocol_digest,
            )
            prediction_rows(
                target,
                probability,
                protocol="loso",
                subject=subject,
                direction="stroop",
                method=MODEL_NAME,
                seed=seed,
            ).to_csv(prediction_path(subject, seed), index=False, lineterminator="\n")
            diagnostics.update(
                {
                    "protocol": "loso",
                    "direction": "stroop",
                    "subject": subject,
                    "seed": seed,
                    "protocol_sha256": protocol_digest,
                    "outer_subject_excluded": True,
                    "inner_participants": 14,
                    "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
                }
            )
            diagnostic_path(subject, seed).write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not checkpoint_complete(subject, seed, protocol_digest):
                raise AssertionError(f"EEGNet S{subject:02d} seed={seed} 写入后未通过检查")
            print(f"[done] EEGNet S{subject:02d} seed={seed}")

    print("[2/3] 合并并生成五种子 recording-level 预测")
    prediction_files = [prediction_path(subject, seed) for subject in range(1, 16) for seed in SEEDS]
    diagnostic_files = [diagnostic_path(subject, seed) for subject in range(1, 16) for seed in SEEDS]
    if not all(
        checkpoint_complete(subject, seed, protocol_digest)
        for subject in range(1, 16)
        for seed in SEEDS
    ):
        raise RuntimeError("75 个 EEGNet 外层参与者-种子单元尚未全部完成")
    raw = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    keys = [
        "dataset",
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "window_id",
        "method",
        "seed",
    ]
    if raw.duplicated(keys).any():
        raise AssertionError("合并后的 EEGNet 窗口预测存在重复键")
    raw = raw.sort_values(keys).reset_index(drop=True)
    windows = median_seed_ensemble(raw, SEEDS)
    recordings = aggregate_recordings(windows)
    validate_frozen_predictions(recordings)
    if len(recordings) != 60:
        raise AssertionError(f"EEGNet 最终 Recording 结果应有 60 行，实际为 {len(recordings)} 行")
    subjects, summary = summarize(recordings)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT, index=False, lineterminator="\n")
    recordings.to_csv(RECORDING_OUTPUT, index=False, lineterminator="\n")
    subjects.to_csv(SUBJECT_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")

    print("[3/3] 写入哈希和运行清单")
    checkpoint_files = prediction_files + diagnostic_files + [
        progress_path(subject, seed)
        for subject in range(1, 16)
        for seed in SEEDS
    ]
    manifest = {
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "task": "stroop",
        "protocol": "fully nested participant-disjoint LOSO",
        "model": MODEL_NAME,
        "outer_folds": 15,
        "inner_folds_per_outer": 14,
        "seeds": SEEDS,
        "participants": 15,
        "recordings_per_participant": 4,
        "evaluation_unit": "recording",
        "statistical_unit": "held-out participant",
        "seed_is_statistical_unit": False,
        "ensemble_rule": "per-class window median across five seeds, then recording geometric mean",
        "checkpoint_files": len(checkpoint_files),
        "checkpoint_bundle_sha256": combined_checkpoint_digest(checkpoint_files),
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "deep_implementation": sha256_file(ROUTE_A_ROOT / "run_deep_baselines.py"),
            "deep_models": sha256_file(ROUTE_A_ROOT / "route_a_lib" / "deep.py"),
            "protocol": protocol_digest,
            "raw_loader": sha256_file(ROUTE_A_ROOT / "route_a_lib" / "data.py"),
        },
        "output_sha256": {
            RAW_OUTPUT.name: sha256_file(RAW_OUTPUT),
            RECORDING_OUTPUT.name: sha256_file(RECORDING_OUTPUT),
            SUBJECT_OUTPUT.name: sha256_file(SUBJECT_OUTPUT),
            SUMMARY_OUTPUT.name: sha256_file(SUMMARY_OUTPUT),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
        },
        "claim_scope": "baseline comparison only; no superiority claim without paired participant-level inference",
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = summary.copy()
    for column in ("mean_accuracy", "ci95_low", "ci95_high"):
        display[column] = 100.0 * display[column]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"EEGNet Stroop LOSO 完成，输出目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
