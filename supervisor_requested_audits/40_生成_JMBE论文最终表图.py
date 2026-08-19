"""直接运行：从两份已冻结预测生成JMBE论文最终统一表图。

本程序不训练模型，不修改Route A或Stroop LOSO冻结文件，也不需要命令行参数。
输出目录：41_输出_JMBE论文最终表图。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
ROUTE_A_PREDICTIONS = REVISION_ROOT / "route_a" / "frozen" / "predictions_recording_route_a.csv"
STROOP_PREDICTIONS = (
    HERE
    / "33_输出_Stroop_LOSO_14方法冻结统计表图"
    / "01_冻结预测_Stroop_LOSO_14方法Recording.csv"
)
OUTPUT_DIR = HERE / "41_输出_JMBE论文最终表图"

FROZEN_OUTPUT = OUTPUT_DIR / "01_冻结预测_三类主评价_14方法.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "02_表_14方法三类主评价准确率和95CI.csv"
DIRECTION_OUTPUT = OUTPUT_DIR / "03_表_双向CrossTask_14方法准确率和95CI.csv"
PAIRED_OUTPUT = OUTPUT_DIR / "04_表_三类主评价相对AlwaysNN配对检验和BH校正.csv"
DIRECTION_PAIRED_OUTPUT = OUTPUT_DIR / "05_表_双向CrossTask相对AlwaysNN配对检验和BH校正.csv"
FIGURE_1_PNG = OUTPUT_DIR / "06_图1_14方法三类主评价准确率.png"
FIGURE_1_TIF = OUTPUT_DIR / "07_图1_14方法三类主评价准确率.tif"
FIGURE_2_PNG = OUTPUT_DIR / "08_图2_融合方法相对AlwaysNN参与者差值.png"
FIGURE_2_TIF = OUTPUT_DIR / "09_图2_融合方法相对AlwaysNN参与者差值.tif"
MANIFEST_OUTPUT = OUTPUT_DIR / "10_记录_冻结预测表图和哈希.json"

EXPECTED_ROUTE_A_SHA256 = "d93ecfb14705f9bbf2dd27d32fcd0e4bb1702184d2e632c5198a7d5fce408717"
EXPECTED_STROOP_SHA256 = "d3497a2ae29c048644c971327a46828a987ee3acc5fb087f3512fba082ab076f"
BOOTSTRAP_REPEATS = 10000

METHOD_DISPLAY = {
    "always_nn": "Neural comparator (Always-NN)",
    "always_fuse": "Unconditional stacked fusion",
    "dasf_clean": "DASF hard gate",
    "cbsf": "CB-SF soft gate",
    "fixed_blend_010": "Fixed 10% fusion",
    "fixed_blend_025": "Fixed 25% fusion",
    "equal_blend_050": "Equal-weight soft fusion",
    "stack_recording": "Recording-level OOF stacker",
    "rf": "Random forest",
    "extra_trees": "ExtraTrees",
    "riemann_mdm": "Riemannian MDM",
    "riemann_ts_logreg": "Riemannian tangent-space logistic regression",
    "eegnet": "EEGNet",
    "eeg_conformer": "EEG-Conformer",
}

METHOD_ORDER = [
    "always_nn",
    "always_fuse",
    "dasf_clean",
    "cbsf",
    "fixed_blend_010",
    "fixed_blend_025",
    "equal_blend_050",
    "stack_recording",
    "rf",
    "extra_trees",
    "riemann_mdm",
    "riemann_ts_logreg",
    "eegnet",
    "eeg_conformer",
]

SETTING_DISPLAY = {
    "cross_task_bidirectional": "Bidirectional cross-task",
    "loso_arithmetic": "Arithmetic LOSO",
    "loso_stroop": "Stroop LOSO",
}

DIRECTION_DISPLAY = {
    "arithmetic_to_stroop": "Arithmetic to Stroop",
    "stroop_to_arithmetic": "Stroop to Arithmetic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != 15:
        raise ValueError("Participant bootstrap expects exactly 15 one-dimensional values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def wilcoxon_p_value(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=np.float64)
    if np.allclose(difference, 0.0):
        return 1.0
    return float(
        wilcoxon(
            difference,
            zero_method="pratt",
            correction=False,
            alternative="two-sided",
            method="approx",
        ).pvalue
    )


def load_and_validate_predictions() -> pd.DataFrame:
    route_hash = sha256_file(ROUTE_A_PREDICTIONS)
    stroop_hash = sha256_file(STROOP_PREDICTIONS)
    if route_hash != EXPECTED_ROUTE_A_SHA256:
        raise RuntimeError(
            f"Route A冻结预测哈希不匹配：{route_hash}。不要继续制表，也不要覆盖原文件。"
        )
    if stroop_hash != EXPECTED_STROOP_SHA256:
        raise RuntimeError(
            f"Stroop LOSO冻结预测哈希不匹配：{stroop_hash}。不要继续制表，也不要覆盖原文件。"
        )

    route = pd.read_csv(ROUTE_A_PREDICTIONS)
    stroop = pd.read_csv(STROOP_PREDICTIONS)
    required = {
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
    }
    for name, frame in [("Route A", route), ("Stroop LOSO", stroop)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}缺少列：{sorted(missing)}")

    columns = [
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
    ]
    combined = pd.concat([route[columns], stroop[columns]], ignore_index=True)
    combined["subject"] = combined["subject"].astype(int)
    combined["true_label"] = combined["true_label"].astype(int)
    combined["pred_label"] = combined["pred_label"].astype(int)

    if len(combined) != 3360:
        raise AssertionError(f"Expected 3360 frozen rows, got {len(combined)}")
    if set(combined["method"].unique()) != set(METHOD_ORDER):
        raise AssertionError("The combined freeze does not contain the expected 14 methods")
    if combined["subject"].nunique() != 15:
        raise AssertionError("The combined freeze does not contain 15 participants")
    key = ["dataset", "protocol", "direction", "subject", "recording_id", "method"]
    if combined.duplicated(key).any():
        raise AssertionError("Duplicate method-recording key in combined freeze")
    if not np.allclose(combined["p0"] + combined["p1"], 1.0, atol=1e-8):
        raise AssertionError("Probability rows do not sum to one")
    expected_prediction = (combined["p1"].to_numpy() > combined["p0"].to_numpy()).astype(int)
    if not np.array_equal(expected_prediction, combined["pred_label"].to_numpy()):
        raise AssertionError("Stored labels do not match probability argmax")

    label_counts = combined.groupby(
        ["protocol", "direction", "subject", "recording_id"]
    )["true_label"].nunique()
    if int(label_counts.max()) != 1:
        raise AssertionError("True labels differ between methods for a recording")

    combined["method_display"] = combined["method"].map(METHOD_DISPLAY)
    if combined["method_display"].isna().any():
        raise AssertionError("A publication display name is missing")
    combined["correct"] = (
        combined["true_label"].to_numpy() == combined["pred_label"].to_numpy()
    ).astype(float)
    combined["setting"] = np.where(
        combined["protocol"] == "cross_task",
        "cross_task_bidirectional",
        np.where(combined["direction"] == "stroop", "loso_stroop", "loso_arithmetic"),
    )
    combined["setting_display"] = combined["setting"].map(SETTING_DISPLAY)
    return combined


def participant_accuracies(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(
            ["setting", "setting_display", "subject", "method", "method_display"],
            as_index=False,
        )
        .agg(accuracy=("correct", "mean"), n_recordings=("recording_id", "nunique"))
        .sort_values(["setting", "method", "subject"])
        .reset_index(drop=True)
    )


def build_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, method), group in participants.groupby(["setting", "method"], sort=True):
        values = group["accuracy"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(values, seed=stable_seed(setting, method, "summary"))
        rows.append(
            {
                "setting": setting,
                "setting_display": SETTING_DISPLAY[setting],
                "method_id": method,
                "method": METHOD_DISPLAY[method],
                "n_participants": 15,
                "recordings_per_participant": int(group["n_recordings"].iloc[0]),
                "mean_recording_accuracy": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    output = pd.DataFrame(rows)
    output["method_order"] = output["method_id"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    return output.sort_values(["setting", "method_order"]).drop(columns="method_order").reset_index(drop=True)


def build_direction_summary(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[data["protocol"] == "cross_task"].copy()
    participant = (
        selected.groupby(
            ["direction", "subject", "method", "method_display"], as_index=False
        )
        .agg(accuracy=("correct", "mean"), n_recordings=("recording_id", "nunique"))
    )
    rows = []
    for (direction, method), group in participant.groupby(["direction", "method"], sort=True):
        values = group["accuracy"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(
            values, seed=stable_seed(direction, method, "direction_summary")
        )
        rows.append(
            {
                "direction": direction,
                "direction_display": DIRECTION_DISPLAY[direction],
                "method_id": method,
                "method": METHOD_DISPLAY[method],
                "n_participants": 15,
                "recordings_per_participant": int(group["n_recordings"].iloc[0]),
                "mean_recording_accuracy": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    output = pd.DataFrame(rows)
    output["method_order"] = output["method_id"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    return output.sort_values(["direction", "method_order"]).drop(columns="method_order").reset_index(drop=True)


def paired_comparisons(
    participant: pd.DataFrame,
    *,
    group_column: str,
    group_display: dict[str, str],
    bh_family: str,
) -> pd.DataFrame:
    rows = []
    for group_value, group in participant.groupby(group_column):
        pivot = group.pivot(index="subject", columns="method", values="accuracy")
        if len(pivot) != 15 or "always_nn" not in pivot:
            raise AssertionError(f"Incomplete participant matrix for {group_value}")
        reference = pivot["always_nn"].to_numpy(dtype=np.float64)
        for method in [value for value in METHOD_ORDER if value != "always_nn"]:
            difference = pivot[method].to_numpy(dtype=np.float64) - reference
            low, high = bootstrap_mean_ci(
                difference, seed=stable_seed(str(group_value), method, "paired")
            )
            rows.append(
                {
                    group_column: group_value,
                    f"{group_column}_display": group_display[group_value],
                    "method_id": method,
                    "method": METHOD_DISPLAY[method],
                    "comparator": METHOD_DISPLAY["always_nn"],
                    "n_paired_participants": 15,
                    "mean_difference": float(difference.mean()),
                    "paired_ci95_low": low,
                    "paired_ci95_high": high,
                    "participants_improved": int(np.sum(difference > 0)),
                    "participants_tied": int(np.sum(np.isclose(difference, 0))),
                    "participants_worsened": int(np.sum(difference < 0)),
                    "wilcoxon_p_raw": wilcoxon_p_value(difference),
                    "analysis_status": "post-hoc exploratory complete-method comparison",
                    "bh_family": bh_family,
                }
            )
    output = pd.DataFrame(rows)
    output["wilcoxon_p_bh"] = bh_adjust(output["wilcoxon_p_raw"].to_numpy())
    output["significant_after_bh_0_05"] = output["wilcoxon_p_bh"] < 0.05
    return output.sort_values([group_column, "method_id"]).reset_index(drop=True)


def build_main_paired(participants: pd.DataFrame) -> pd.DataFrame:
    return paired_comparisons(
        participants,
        group_column="setting",
        group_display=SETTING_DISPLAY,
        bh_family="all 39 comparisons across bidirectional cross-task and two LOSO tasks",
    )


def build_direction_paired(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[data["protocol"] == "cross_task"]
    participant = (
        selected.groupby(["direction", "subject", "method"], as_index=False)
        .agg(accuracy=("correct", "mean"))
    )
    return paired_comparisons(
        participant,
        group_column="direction",
        group_display=DIRECTION_DISPLAY,
        bh_family="all 26 direction-specific cross-task comparisons",
    )


def save_figure(figure: plt.Figure, png_path: Path, tif_path: Path) -> None:
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(
        tif_path,
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def make_accuracy_figure(summary: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(12.8, 9.2))
    y = np.arange(len(METHOD_ORDER), dtype=float)
    offsets = {
        "cross_task_bidirectional": -0.22,
        "loso_arithmetic": 0.0,
        "loso_stroop": 0.22,
    }
    colors = {
        "cross_task_bidirectional": "#2878B5",
        "loso_arithmetic": "#D95F02",
        "loso_stroop": "#2A9D6F",
    }
    markers = {
        "cross_task_bidirectional": "o",
        "loso_arithmetic": "s",
        "loso_stroop": "D",
    }
    for setting in ["cross_task_bidirectional", "loso_arithmetic", "loso_stroop"]:
        selected = summary.set_index(["setting", "method_id"]).loc[setting].loc[METHOD_ORDER]
        mean = 100.0 * selected["mean_recording_accuracy"].to_numpy(dtype=float)
        low = 100.0 * selected["ci95_low"].to_numpy(dtype=float)
        high = 100.0 * selected["ci95_high"].to_numpy(dtype=float)
        axis.errorbar(
            mean,
            y + offsets[setting],
            xerr=np.vstack([mean - low, high - mean]),
            fmt=markers[setting],
            color=colors[setting],
            ecolor=colors[setting],
            elinewidth=1.4,
            capsize=2.5,
            markersize=5.5,
            label=SETTING_DISPLAY[setting],
        )
    axis.set_yticks(y)
    axis.set_yticklabels([METHOD_DISPLAY[method] for method in METHOD_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Recording-level accuracy (%)")
    axis.set_ylabel("Method")
    axis.set_xlim(40, 84)
    axis.grid(axis="x", color="#D6D6D6", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    save_figure(figure, FIGURE_1_PNG, FIGURE_1_TIF)


def make_difference_figure(participants: pd.DataFrame) -> None:
    core_methods = ["always_fuse", "dasf_clean", "cbsf", "stack_recording"]
    figure, axes = plt.subplots(3, 1, figsize=(12.5, 12.0), sharey=True)
    rng = np.random.default_rng(20260729)
    colors = ["#777777", "#3B6FB6", "#C84C4C", "#258A72"]

    for axis, setting in zip(
        axes,
        ["cross_task_bidirectional", "loso_arithmetic", "loso_stroop"],
    ):
        group = participants[participants["setting"] == setting]
        pivot = group.pivot(index="subject", columns="method", values="accuracy")
        reference = pivot["always_nn"].to_numpy(dtype=float)
        for index, (method, color) in enumerate(zip(core_methods, colors)):
            difference = 100.0 * (pivot[method].to_numpy(dtype=float) - reference)
            jitter = rng.uniform(-0.11, 0.11, size=len(difference))
            axis.scatter(
                np.full(len(difference), index, dtype=float) + jitter,
                difference,
                s=30,
                color=color,
                alpha=0.58,
                edgecolors="none",
            )
            axis.scatter(
                index,
                float(difference.mean()),
                s=95,
                color="black",
                marker="D",
                zorder=4,
            )
        axis.axhline(0.0, color="black", linewidth=1.1)
        axis.axhline(-5.0, color="#C84C4C", linestyle="--", linewidth=1.0)
        axis.set_title(SETTING_DISPLAY[setting], loc="left", fontsize=12)
        axis.set_xticks(range(len(core_methods)))
        axis.set_xticklabels([METHOD_DISPLAY[method] for method in core_methods])
        axis.set_ylabel("Difference from Always-NN (pp)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Fusion strategy")
    axes[0].set_ylim(-55, 55)
    figure.tight_layout(h_pad=2.0)
    save_figure(figure, FIGURE_2_PNG, FIGURE_2_TIF)


def write_manifest(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    paired: pd.DataFrame,
    direction_paired: pd.DataFrame,
) -> None:
    outputs = [
        FROZEN_OUTPUT,
        SUMMARY_OUTPUT,
        DIRECTION_OUTPUT,
        PAIRED_OUTPUT,
        DIRECTION_PAIRED_OUTPUT,
        FIGURE_1_PNG,
        FIGURE_1_TIF,
        FIGURE_2_PNG,
        FIGURE_2_TIF,
    ]
    manifest = {
        "purpose": "single-source JMBE manuscript tables and figures",
        "analysis_status": "post_hoc supervisor-requested complete reporting",
        "source_sha256": {
            "route_a_predictions": sha256_file(ROUTE_A_PREDICTIONS),
            "stroop_loso_predictions": sha256_file(STROOP_PREDICTIONS),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "combined_frozen_rows": int(len(data)),
        "methods": int(data["method"].nunique()),
        "participants": int(data["subject"].nunique()),
        "summary_rows": int(len(summary)),
        "direction_summary_rows": int(len(direction_summary)),
        "paired_rows": int(len(paired)),
        "direction_paired_rows": int(len(direction_paired)),
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "seed_is_statistical_unit": False,
        "bh_families": [
            "39 comparisons across bidirectional cross-task and two LOSO tasks",
            "26 direction-specific cross-task comparisons",
        ],
        "significant_after_bh": int(paired["significant_after_bh_0_05"].sum())
        + int(direction_paired["significant_after_bh_0_05"].sum()),
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "claim_scope": "complete exploratory reporting; no superiority, confirmatory non-inferiority, safety, no-regret or formal risk-control claim",
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/6] 核对两份冻结预测的SHA-256并合并。")
    data = load_and_validate_predictions()
    data.to_csv(FROZEN_OUTPUT, index=False, lineterminator="\n")

    print("[2/6] 生成三类主评价的14方法汇总。")
    participants = participant_accuracies(data)
    summary = build_summary(participants)
    direction_summary = build_direction_summary(data)
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")
    direction_summary.to_csv(DIRECTION_OUTPUT, index=False, lineterminator="\n")

    print("[3/6] 生成参与者级配对检验和BH校正。")
    paired = build_main_paired(participants)
    direction_paired = build_direction_paired(data)
    paired.to_csv(PAIRED_OUTPUT, index=False, lineterminator="\n")
    direction_paired.to_csv(DIRECTION_PAIRED_OUTPUT, index=False, lineterminator="\n")

    if len(summary) != 42 or len(direction_summary) != 28:
        raise AssertionError("Expected 42 main summary rows and 28 direction rows")
    if len(paired) != 39 or len(direction_paired) != 26:
        raise AssertionError("Expected 39 main paired rows and 26 direction paired rows")

    print("[4/6] 生成无代码变量名的图1和图2。")
    make_accuracy_figure(summary)
    make_difference_figure(participants)

    print("[5/6] 写入运行清单和全部文件哈希。")
    write_manifest(data, summary, direction_summary, paired, direction_paired)

    print("[6/6] JMBE论文最终统一表图已生成。")
    compact = summary.pivot(
        index="method", columns="setting_display", values="mean_recording_accuracy"
    ).loc[[METHOD_DISPLAY[method] for method in METHOD_ORDER]]
    print((100.0 * compact).round(2).to_string())
    print(f"\n输出目录：{OUTPUT_DIR}")
    print("下一步：把控制台最后的表格发给我，并保留41号目录全部文件。")


if __name__ == "__main__":
    main()
