"""直接运行：补齐 Stroop LOSO 的两种 Riemannian 基线。

模型为 Riemannian MDM 和 tangent-space logistic regression。两者是固定超参数的
确定性基线，不使用外层目标标签进行调参；为统一最终五种子数据结构，同一确定性
预测写入五个冻结种子槽位，集成不会改变预测。结果写入编号 26 的独立目录。
无命令行参数，可直接在 PyCharm 运行。
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
REVISION_ROOT = HERE.parent
PROJECT_ROOT = REVISION_ROOT
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
OUTPUT_ROOT = HERE / "26_输出_Stroop_LOSO_Riemannian基线"
CHECKPOINT_ROOT = OUTPUT_ROOT / "01_检查点_可续跑"

RAW_OUTPUT = OUTPUT_ROOT / "02_结果_Riemannian窗口种子预测.csv"
RECORDING_OUTPUT = OUTPUT_ROOT / "03_结果_Riemannian五种子Recording预测.csv"
SUBJECT_OUTPUT = OUTPUT_ROOT / "04_结果_Riemannian参与者准确率.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "05_结果_Riemannian汇总.csv"
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
from revision_pipeline.splits import outer_loso  # noqa: E402
from route_a.run_riemannian_baselines import (  # noqa: E402
    METHODS,
    fit_predict_riemannian,
    prediction_rows,
)
from route_a_lib.data import load_feature_bundles  # noqa: E402


SEEDS = [1335, 1388, 1441, 1494, 1547]


def checkpoint_path(subject: int) -> Path:
    return CHECKPOINT_ROOT / f"S{subject:02d}_Stroop_LOSO_确定性预测写入五种子.csv"


def checkpoint_complete(path: Path, subject: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError):
        return False
    keys = ["recording_id", "window_id", "method", "seed"]
    return bool(
        not frame.empty
        and set(frame["subject"].astype(int)) == {int(subject)}
        and set(frame["direction"].astype(str)) == {"stroop"}
        and set(frame["protocol"].astype(str)) == {"loso"}
        and set(frame["method"].astype(str)) == set(METHODS)
        and set(frame["seed"].astype(int)) == set(SEEDS)
        and not frame.duplicated(keys).any()
        and frame.groupby(["recording_id", "window_id", "method"])["seed"].nunique().eq(5).all()
        and np.allclose(frame[["p0", "p1"]].sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    )


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
    rows = []
    for keys, group in subjects.groupby(["dataset", "protocol", "direction", "method"]):
        values = group["accuracy"].to_numpy(dtype=np.float64)
        low, high = bootstrap_interval(values)
        rows.append(
            {
                "dataset": keys[0],
                "protocol": keys[1],
                "direction": keys[2],
                "method": keys[3],
                "n_subjects": int(len(values)),
                "mean_accuracy": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return subjects, pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if [int(value) for value in protocol["seeds"]] != SEEDS:
        raise RuntimeError("Route A 协议中的五个种子发生变化")
    try:
        import pyriemann
    except ImportError as exc:
        raise RuntimeError("cuda129 环境缺少 pyriemann，请先安装 pyriemann") from exc

    _, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))
    outer = {subject: (train, test) for subject, train, test in outer_loso(stroop.subjects)}
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    for subject in range(1, 16):
        path = checkpoint_path(subject)
        if checkpoint_complete(path, subject):
            print(f"[skip] S{subject:02d} 已完成")
            continue
        train_index, test_index = outer[subject]
        source = stroop.subset(train_index)
        target = stroop.subset(test_index)
        probability = fit_predict_riemannian(
            source.covariances,
            source.labels,
            target.covariances,
            seed=SEEDS[0],
            config=protocol["riemannian"],
        )
        rows = [
            prediction_rows(
                target,
                probability,
                protocol="loso",
                subject=subject,
                direction="stroop",
                seed=seed,
            )
            for seed in SEEDS
        ]
        frame = pd.concat(rows, ignore_index=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        if not checkpoint_complete(path, subject):
            raise AssertionError(f"S{subject:02d} 的 Riemannian 检查点写入后未通过校验")
        print(f"[done] S{subject:02d} Riemannian MDM + TS-LR")

    checkpoints = [checkpoint_path(subject) for subject in range(1, 16)]
    if not all(checkpoint_complete(path, subject) for subject, path in enumerate(checkpoints, start=1)):
        raise RuntimeError("15 名参与者的 Riemannian 检查点尚未全部完成")
    raw = pd.concat([pd.read_csv(path) for path in checkpoints], ignore_index=True)
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
        raise AssertionError("合并后的 Riemannian 窗口预测存在重复键")
    raw = raw.sort_values(keys).reset_index(drop=True)
    windows = median_seed_ensemble(raw, SEEDS)
    recordings = aggregate_recordings(windows)
    validate_frozen_predictions(recordings)
    if len(recordings) != 15 * 4 * 2:
        raise AssertionError(f"Riemannian 最终结果应有 120 行，实际为 {len(recordings)} 行")
    subjects, summary = summarize(recordings)

    raw.to_csv(RAW_OUTPUT, index=False, lineterminator="\n")
    recordings.to_csv(RECORDING_OUTPUT, index=False, lineterminator="\n")
    subjects.to_csv(SUBJECT_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")
    manifest = {
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "task": "stroop",
        "protocol": "participant-disjoint LOSO",
        "methods": list(METHODS),
        "deterministic_baseline": True,
        "seed_slot_policy": "one deterministic fit copied to five seed slots; median ensemble is unchanged",
        "participants": 15,
        "recordings_per_participant": 4,
        "evaluation_unit": "recording",
        "statistical_unit": "held-out participant",
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "route_a_riemannian_implementation": sha256_file(
                ROUTE_A_ROOT / "run_riemannian_baselines.py"
            ),
            "protocol": sha256_file(PROTOCOL_PATH),
            "feature_source": sha256_file(PROJECT_ROOT / "eeg_feature_pipeline.py"),
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
            "pyriemann": pyriemann.__version__,
            "device": "cpu",
        },
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = summary.copy()
    for column in ("mean_accuracy", "ci95_low", "ci95_high"):
        display[column] = 100.0 * display[column]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"输出目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
