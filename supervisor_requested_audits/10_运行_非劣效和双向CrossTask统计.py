"""直接运行：生成探索性非劣效敏感性和双向 cross-task 完整统计。

本脚本只读取 Route A 冻结的 recording 级预测，不训练模型、不覆盖冻结文件。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
PROTOCOL_PATH = HERE / "02_冻结方案_导师意见补充实验.json"
PREDICTION_PATH = ROUTE_A_ROOT / "frozen" / "predictions_recording_route_a.csv"

NI_OUTPUT = HERE / "11_结果_非劣效界值敏感性.csv"
DIRECTION_SUMMARY_OUTPUT = HERE / "12_结果_双向CrossTask_14方法汇总.csv"
DIRECTION_SUBJECT_OUTPUT = HERE / "13_结果_双向CrossTask_参与者级.csv"
DIRECTION_PAIRED_OUTPUT = HERE / "14_结果_双向CrossTask_配对检验和BH校正.csv"
MANIFEST_OUTPUT = HERE / "15_记录_非劣效和双向统计运行清单.json"

METHOD_LABELS = {
    "always_nn": "Always-NN",
    "always_fuse": "Always-fuse",
    "dasf_clean": "DASF",
    "cbsf": "CB-SF",
    "fixed_blend_010": "Fixed blend (lambda=0.10)",
    "fixed_blend_025": "Fixed blend (lambda=0.25)",
    "equal_blend_050": "Equal blend (lambda=0.50)",
    "stack_recording": "Recording-level stacker",
    "rf": "Random forest",
    "extra_trees": "ExtraTrees",
    "riemann_mdm": "Riemannian MDM",
    "riemann_ts_logreg": "Riemannian TS-LR",
    "eegnet": "EEGNet",
    "eeg_conformer": "EEG-Conformer",
}

DIRECTION_LABELS = {
    "arithmetic_to_stroop": "Arithmetic to Stroop",
    "stroop_to_arithmetic": "Stroop to Arithmetic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_distribution(values: np.ndarray, *, seed: int, n_resamples: int = 10000) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Participant bootstrap requires at least two one-dimensional values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(n_resamples), len(values)))
    return values[indices].mean(axis=1)


def interval_from_distribution(distribution: np.ndarray, *, coverage: float) -> tuple[float, float]:
    alpha = 1.0 - float(coverage)
    low, high = np.quantile(distribution, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def one_sided_lower(distribution: np.ndarray, *, confidence: float = 0.95) -> float:
    return float(np.quantile(distribution, 1.0 - float(confidence)))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("BH adjustment requires a non-empty one-dimensional array")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1, dtype=np.float64)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def paired_wilcoxon(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=np.float64)
    if np.allclose(difference, 0.0):
        return 1.0
    return float(
        wilcoxon(
            difference,
            zero_method="pratt",
            alternative="two-sided",
            method="auto",
        ).pvalue
    )


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "method",
        "true_label",
        "correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen prediction file is missing columns: {missing}")
    cross_task = frame[frame["protocol"] == "cross_task"].copy()
    if len(cross_task) != 1680:
        raise AssertionError(f"Expected 1680 cross-task rows, got {len(cross_task)}")
    if set(cross_task["method"].unique()) != set(METHOD_LABELS):
        raise AssertionError("Frozen cross-task methods do not match the expected 14-method set")
    if set(cross_task["direction"].unique()) != set(DIRECTION_LABELS):
        raise AssertionError("Frozen cross-task directions are incomplete")
    key = ["dataset", "protocol", "subject", "direction", "recording_id", "method"]
    if cross_task.duplicated(key).any():
        raise AssertionError("Duplicate method-recording rows found in frozen predictions")
    return cross_task


def build_subject_metrics(cross_task: pd.DataFrame) -> pd.DataFrame:
    subject = (
        cross_task.groupby(["dataset", "direction", "subject", "method"], as_index=False)
        .agg(
            n_recordings=("recording_id", "nunique"),
            accuracy=("correct", "mean"),
        )
        .sort_values(["direction", "method", "subject"])
        .reset_index(drop=True)
    )
    if set(subject["n_recordings"].unique()) != {4}:
        raise AssertionError("Every participant-direction-method cell must contain four recordings")
    subject["direction_label"] = subject["direction"].map(DIRECTION_LABELS)
    subject["method_label"] = subject["method"].map(METHOD_LABELS)
    return subject


def build_direction_summary(subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seed = 20260727
    for index, ((direction, method), group) in enumerate(subject.groupby(["direction", "method"])):
        values = group["accuracy"].to_numpy(dtype=np.float64)
        distribution = bootstrap_distribution(values, seed=seed + index)
        low, high = interval_from_distribution(distribution, coverage=0.95)
        rows.append(
            {
                "direction": direction,
                "direction_label": DIRECTION_LABELS[direction],
                "method_id": method,
                "method": METHOD_LABELS[method],
                "n_participants": int(group["subject"].nunique()),
                "mean_recording_accuracy": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["direction", "mean_recording_accuracy", "method"],
        ascending=[True, False, True],
    )


def build_direction_paired(subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seed = 20261727
    for direction_index, (direction, group) in enumerate(subject.groupby("direction")):
        pivot = group.pivot(index="subject", columns="method", values="accuracy")
        if "always_nn" not in pivot:
            raise AssertionError(f"Always-NN is missing for {direction}")
        for method_index, method in enumerate(sorted(set(METHOD_LABELS) - {"always_nn"})):
            paired = pivot[[method, "always_nn"]].dropna()
            difference = paired[method].to_numpy(dtype=np.float64) - paired["always_nn"].to_numpy(dtype=np.float64)
            distribution = bootstrap_distribution(
                difference,
                seed=seed + direction_index * 100 + method_index,
            )
            low, high = interval_from_distribution(distribution, coverage=0.95)
            rows.append(
                {
                    "direction": direction,
                    "direction_label": DIRECTION_LABELS[direction],
                    "method_id": method,
                    "method": METHOD_LABELS[method],
                    "comparator": "Always-NN",
                    "n_paired_participants": int(len(paired)),
                    "mean_difference": float(difference.mean()),
                    "paired_ci95_low": low,
                    "paired_ci95_high": high,
                    "wilcoxon_p_raw": paired_wilcoxon(difference),
                }
            )
    output = pd.DataFrame(rows)
    # One conservative family: all 13 methods in both directions (26 tests).
    output["bh_family"] = "all 26 direction-specific secondary comparisons"
    output["wilcoxon_p_bh"] = bh_adjust(output["wilcoxon_p_raw"].to_numpy(dtype=np.float64))
    output["significant_after_bh_0_05"] = output["wilcoxon_p_bh"] < 0.05
    return output.sort_values(["direction", "wilcoxon_p_bh", "method"]).reset_index(drop=True)


def build_noninferiority_sensitivity(
    cross_task: pd.DataFrame,
    margins_pp: list[float],
) -> pd.DataFrame:
    selected = cross_task[cross_task["method"].isin(["always_nn", "cbsf"])]
    subject = (
        selected.groupby(["subject", "method"], as_index=False)
        .agg(accuracy=("correct", "mean"), n_recordings=("recording_id", "nunique"))
    )
    if set(subject["n_recordings"].unique()) != {8}:
        raise AssertionError("Bidirectional cross-task accuracy must contain eight recordings per participant")
    pivot = subject.pivot(index="subject", columns="method", values="accuracy")
    difference = pivot["cbsf"].to_numpy(dtype=np.float64) - pivot["always_nn"].to_numpy(dtype=np.float64)
    distribution = bootstrap_distribution(difference, seed=20262727)
    two_sided_95_low, two_sided_95_high = interval_from_distribution(distribution, coverage=0.95)
    two_sided_90_low, two_sided_90_high = interval_from_distribution(distribution, coverage=0.90)
    one_sided_95_low = one_sided_lower(distribution, confidence=0.95)
    rows: list[dict[str, object]] = []
    for margin_pp in margins_pp:
        margin = float(margin_pp) / 100.0
        met_95 = bool(two_sided_95_low > -margin)
        met_one_sided = bool(one_sided_95_low > -margin)
        rows.append(
            {
                "analysis_status": "post_hoc exploratory margin sensitivity",
                "method": "CB-SF",
                "comparator": "Always-NN",
                "n_participants": int(len(pivot)),
                "mean_difference": float(difference.mean()),
                "margin_pp": float(margin_pp),
                "two_sided_95_ci_low": two_sided_95_low,
                "two_sided_95_ci_high": two_sided_95_high,
                "criterion_met_two_sided_95": met_95,
                "two_sided_90_ci_low": two_sided_90_low,
                "two_sided_90_ci_high": two_sided_90_high,
                "one_sided_95_lower_bound": one_sided_95_low,
                "criterion_met_one_sided_95": met_one_sided,
                "superiority_wilcoxon_p": paired_wilcoxon(difference),
                "interpretation": (
                    "Exploratory margin criterion met; this is not prospective non-inferiority evidence."
                    if met_95
                    else "Exploratory margin criterion not met."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(outputs: list[Path], *, row_counts: dict[str, int]) -> None:
    sources = {
        "protocol": PROTOCOL_PATH,
        "runner": Path(__file__).resolve(),
        "frozen_predictions": PREDICTION_PATH,
    }
    manifest = {
        "purpose": "Exploratory non-inferiority sensitivity and direction-specific cross-task statistics",
        "analysis_status": "post_hoc_exploratory",
        "bh_family": "all 26 direction-specific secondary comparisons",
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "row_counts": row_counts,
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("[1/4] 正在读取 Route A 冻结预测文件。")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTION_PATH)
    cross_task = validate_predictions(predictions)

    print("[2/4] 正在计算双向 cross-task 的14方法结果和参与者置信区间。")
    subject = build_subject_metrics(cross_task)
    summary = build_direction_summary(subject)
    paired = build_direction_paired(subject)

    print("[3/4] 正在计算 -2.5/-5/-7.5 pp 探索性界值敏感性。")
    margins = [
        float(value)
        for value in protocol["noninferiority_sensitivity"]["margins_percentage_points"]
    ]
    noninferiority = build_noninferiority_sensitivity(cross_task, margins)

    noninferiority.to_csv(NI_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(DIRECTION_SUMMARY_OUTPUT, index=False, lineterminator="\n")
    subject.to_csv(DIRECTION_SUBJECT_OUTPUT, index=False, lineterminator="\n")
    paired.to_csv(DIRECTION_PAIRED_OUTPUT, index=False, lineterminator="\n")
    outputs = [NI_OUTPUT, DIRECTION_SUMMARY_OUTPUT, DIRECTION_SUBJECT_OUTPUT, DIRECTION_PAIRED_OUTPUT]
    write_manifest(
        outputs,
        row_counts={
            "noninferiority": int(len(noninferiority)),
            "direction_summary": int(len(summary)),
            "direction_subject": int(len(subject)),
            "direction_paired": int(len(paired)),
        },
    )

    print("[4/4] 统计完成。")
    print("\n非劣效界值敏感性：")
    print(
        noninferiority[
            [
                "margin_pp",
                "mean_difference",
                "two_sided_95_ci_low",
                "two_sided_95_ci_high",
                "criterion_met_two_sided_95",
                "criterion_met_one_sided_95",
            ]
        ].to_string(index=False)
    )
    print("\n双向 cross-task 中数值最高的方法：")
    best = summary.sort_values("mean_recording_accuracy", ascending=False).groupby("direction", as_index=False).first()
    print(best[["direction_label", "method", "mean_recording_accuracy", "ci95_low", "ci95_high"]].to_string(index=False))
    print(f"\n结果文件已写入：{HERE}")
    print("下一步：把控制台输出发给我，并保留 11-15 全部文件。")


if __name__ == "__main__":
    main()

