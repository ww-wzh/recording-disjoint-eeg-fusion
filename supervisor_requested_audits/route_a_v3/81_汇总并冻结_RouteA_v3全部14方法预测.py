"""编号81：汇总并冻结 Route A v3 全部 14 个方法的 recording 级预测。

本脚本不训练模型，只读取 72--80 已生成的 Route A v3 结果。它会：
1. 从各结果目录中只提取对应的新方法，避免重复拼入核心预测；
2. 将 Cross-task 与 LOSO 的 recording-level stacker 合并；
3. 校验 14 个方法是否覆盖完全相同的 190 个 recording；
4. 冻结唯一预测文件，并记录协议、源文件和最终文件的 SHA-256。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：10_all_methods_frozen
后续统计、表格和图片只能读取本脚本生成的冻结预测文件。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
OUTPUT_ROOT = HERE / "10_all_methods_frozen"

sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.aggregation import validate_frozen_predictions  # noqa: E402


FINAL_COLUMNS = [
    "dataset",
    "protocol",
    "subject",
    "direction",
    "recording_id",
    "method",
    "true_label",
    "p0",
    "p1",
    "pred_label",
    "correct",
    "n_windows",
    "ensemble_size",
]

RECORDING_KEYS = ["dataset", "protocol", "subject", "direction", "recording_id"]

EXPECTED_METHODS = [
    "always_nn",
    "always_fuse",
    "rf",
    "extra_trees",
    "fixed_blend_010",
    "fixed_blend_025",
    "equal_blend_050",
    "stack_recording",
    "dasf_clean",
    "cbsf",
    "riemann_ts_logreg",
    "riemann_mdm",
    "eegnet",
    "eeg_conformer",
]

# 每个源文件只允许贡献这里列出的方法。这样不会把 DASF 文件中附带的
# always-NN、always-fuse、RF 和 Extra Trees 再拼接一次。
SOURCE_SPECS = [
    (
        Path("02_frozen/core_recording_predictions.csv"),
        ["always_nn", "always_fuse", "rf", "extra_trees"],
    ),
    (Path("03_dasf/recording_predictions.csv"), ["dasf_clean"]),
    (Path("04_cbsf/recording_predictions.csv"), ["cbsf"]),
    (
        Path("05_baselines/fixed_blend_recording_predictions.csv"),
        ["fixed_blend_010", "fixed_blend_025", "equal_blend_050"],
    ),
    (Path("05_baselines/stack_recording_predictions_loso.csv"), ["stack_recording"]),
    (
        Path("06_cross_task_stacker/cross_task_stack_recording_predictions.csv"),
        ["stack_recording"],
    ),
    (
        Path("07_riemannian/recording_predictions.csv"),
        ["riemann_ts_logreg", "riemann_mdm"],
    ),
    (Path("08_eegnet/recording_predictions.csv"), ["eegnet"]),
    (Path("09_eeg_conformer/recording_predictions.csv"), ["eeg_conformer"]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected_source(relative_path: Path, methods: list[str]) -> pd.DataFrame:
    path = HERE / relative_path
    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件：{path}\n请先完成对应的 72--80 号脚本。")

    frame = pd.read_csv(path)
    missing_columns = sorted(set(FINAL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{relative_path} 缺少列：{missing_columns}")

    selected = frame[frame["method"].isin(methods)][FINAL_COLUMNS].copy()
    observed = set(str(value) for value in selected["method"].unique())
    expected = set(methods)
    if observed != expected:
        raise ValueError(
            f"{relative_path} 方法不完整：expected={sorted(expected)}, observed={sorted(observed)}"
        )
    return selected


def expected_cell_counts(policy: dict) -> dict[tuple[str, str], int]:
    return {
        ("cross_task", "arithmetic_to_stroop"): int(
            policy["cross_task"]["target_recording_count_per_cell"]
            * len(policy["cross_task"]["participants"])
        ),
        ("cross_task", "stroop_to_arithmetic"): int(
            policy["cross_task"]["target_recording_count_per_cell"]
            * len(policy["cross_task"]["participants"])
        ),
        ("loso", "arithmetic"): int(policy["loso"]["arithmetic"]["expected_recordings"]),
        ("loso", "stroop"): int(policy["loso"]["stroop"]["expected_recordings"]),
    }


def validate_probabilities(frame: pd.DataFrame) -> None:
    probabilities = frame[["p0", "p1"]].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("冻结预测中存在 NaN 或无穷概率")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("冻结预测概率超出 [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("冻结预测的 p0 + p1 不等于 1")

    calculated_prediction = np.argmax(probabilities, axis=1).astype(int)
    stored_prediction = frame["pred_label"].to_numpy(dtype=int)
    if not np.array_equal(calculated_prediction, stored_prediction):
        raise ValueError("pred_label 与 p0/p1 的 argmax 不一致")

    labels = frame["true_label"].to_numpy(dtype=int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("true_label 必须为 0/1")
    calculated_correct = (calculated_prediction == labels).astype(int)
    if not np.array_equal(calculated_correct, frame["correct"].to_numpy(dtype=int)):
        raise ValueError("correct 与 pred_label/true_label 不一致")


def validate_method_coverage(frame: pd.DataFrame, policy: dict) -> pd.DataFrame:
    expected_recordings = int(policy["expected_recordings_per_method"])
    expected_total = expected_recordings * len(EXPECTED_METHODS)
    if len(frame) != expected_total:
        raise AssertionError(f"预期 {expected_total} 行，实际 {len(frame)} 行")

    observed_methods = set(str(value) for value in frame["method"].unique())
    if observed_methods != set(EXPECTED_METHODS):
        raise AssertionError(
            f"方法集合不一致：expected={EXPECTED_METHODS}, observed={sorted(observed_methods)}"
        )

    reference = (
        frame[frame["method"] == "always_nn"]
        [RECORDING_KEYS + ["true_label", "n_windows"]]
        .sort_values(RECORDING_KEYS)
        .reset_index(drop=True)
    )
    if len(reference) != expected_recordings:
        raise AssertionError(f"always_nn 应有 {expected_recordings} 行，实际 {len(reference)} 行")

    expected_cells = expected_cell_counts(policy)
    coverage_rows: list[dict] = []
    for method in EXPECTED_METHODS:
        current = frame[frame["method"] == method].sort_values(RECORDING_KEYS).reset_index(drop=True)
        if len(current) != expected_recordings:
            raise AssertionError(f"{method} 应有 {expected_recordings} 行，实际 {len(current)} 行")

        aligned = reference.merge(
            current[RECORDING_KEYS + ["true_label", "n_windows"]],
            on=RECORDING_KEYS,
            how="outer",
            suffixes=("_reference", "_method"),
            indicator=True,
            validate="one_to_one",
        )
        if not (aligned["_merge"] == "both").all():
            raise AssertionError(f"{method} 与 always_nn 的 recording 集合不一致")
        if not (
            aligned["true_label_reference"].astype(int)
            == aligned["true_label_method"].astype(int)
        ).all():
            raise AssertionError(f"{method} 与 always_nn 的真实标签不一致")
        if not (
            aligned["n_windows_reference"].astype(int)
            == aligned["n_windows_method"].astype(int)
        ).all():
            raise AssertionError(f"{method} 与 always_nn 的窗口数量不一致")

        cell_counts = current.groupby(["protocol", "direction"]).size().to_dict()
        if cell_counts != expected_cells:
            raise AssertionError(
                f"{method} 的实验单元数量不一致：expected={expected_cells}, observed={cell_counts}"
            )
        for (protocol, direction), count in sorted(cell_counts.items()):
            cell = current[
                (current["protocol"] == protocol) & (current["direction"] == direction)
            ]
            coverage_rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "direction": direction,
                    "participants": int(cell["subject"].nunique()),
                    "recordings": int(count),
                    "class_0_recordings": int((cell["true_label"].astype(int) == 0).sum()),
                    "class_1_recordings": int((cell["true_label"].astype(int) == 1).sum()),
                }
            )
    return pd.DataFrame(coverage_rows)


def main() -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"缺少冻结协议：{POLICY_PATH}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("protocol_version") != "2026-08-22-route-a-v3":
        raise ValueError("冻结协议版本不是 Route A v3")

    source_frames = [load_selected_source(path, methods) for path, methods in SOURCE_SPECS]
    frozen = pd.concat(source_frames, ignore_index=True)
    frozen["subject"] = frozen["subject"].astype(int)
    for column in ["true_label", "pred_label", "correct", "n_windows", "ensemble_size"]:
        frozen[column] = frozen[column].astype(int)
    frozen["p0"] = frozen["p0"].astype(float)
    frozen["p1"] = frozen["p1"].astype(float)

    method_order = {method: index for index, method in enumerate(EXPECTED_METHODS)}
    frozen["_method_order"] = frozen["method"].map(method_order)
    frozen = (
        frozen.sort_values(RECORDING_KEYS + ["_method_order"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    validate_frozen_predictions(frozen)
    validate_probabilities(frozen)
    coverage = validate_method_coverage(frozen, policy)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_ROOT / "81_冻结_RouteA_v3全部14方法_recording级预测.csv"
    coverage_path = OUTPUT_ROOT / "81_校验_14方法样本覆盖.csv"
    manifest_path = OUTPUT_ROOT / "81_冻结_RouteA_v3全部方法_manifest.json"
    checksum_path = OUTPUT_ROOT / "81_SHA256_冻结预测.txt"

    frozen.to_csv(prediction_path, index=False, lineterminator="\n", float_format="%.17g")
    coverage.to_csv(coverage_path, index=False, lineterminator="\n")

    prediction_digest = sha256(prediction_path)
    source_hashes = {
        path.as_posix(): sha256(HERE / path)
        for path, _ in SOURCE_SPECS
    }
    manifest = {
        "status": "passed",
        "protocol_version": policy["protocol_version"],
        "analysis_role": policy["analysis_role"],
        "dataset": policy["dataset"],
        "eeg_channels": policy["eeg_channels"],
        "methods": EXPECTED_METHODS,
        "method_count": len(EXPECTED_METHODS),
        "rows": int(len(frozen)),
        "recordings_per_method": int(policy["expected_recordings_per_method"]),
        "evaluation_unit": "recording",
        "statistical_unit": "participant; cross-task directions clustered within participant",
        "seed_aggregation": policy["seed_aggregation"],
        "seed_is_statistical_unit": False,
        "cross_task_participants": len(policy["cross_task"]["participants"]),
        "arithmetic_loso_participants": len(policy["loso"]["arithmetic"]["participants"]),
        "stroop_loso_participants": len(policy["loso"]["stroop"]["participants"]),
        "prediction_file": prediction_path.name,
        "prediction_sha256": prediction_digest,
        "coverage_file": coverage_path.name,
        "coverage_sha256": sha256(coverage_path),
        "protocol_file": POLICY_PATH.name,
        "protocol_sha256": sha256(POLICY_PATH),
        "source_file_sha256": source_hashes,
        "old_route_a_results_read": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_path.write_text(
        f"{prediction_digest}  {prediction_path.name}\n",
        encoding="utf-8",
    )

    print("Route A v3 全部方法预测已冻结")
    print(f"预测文件：{prediction_path}")
    print(f"SHA-256：{prediction_digest}")
    print(f"方法数：{len(EXPECTED_METHODS)}")
    print(f"每个方法 recording 数：{policy['expected_recordings_per_method']}")
    print(f"总行数：{len(frozen)}")
    print("参与者：Cross-task=13，Arithmetic LOSO=15，Stroop LOSO=13")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
