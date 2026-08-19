"""直接运行：冻结 Stroop LOSO 的 14 方法结果并统一生成统计表和论文图。

本程序不训练模型。它读取编号 23、26、28、31 中已经完成的 recording-level
预测，先核对各自运行清单中的 SHA-256，再合并为唯一的 14 方法冻结预测文件。
所有统计以 15 名参与者为单位；五个随机种子不会被当成独立样本。

运行方法：在 PyCharm 中直接运行本文件，不需要填写任何命令行参数。
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "33_输出_Stroop_LOSO_14方法冻结统计表图"

CORE_ROOT = HERE / "23_输出_Stroop_LOSO融合和对照"
RIEMANN_ROOT = HERE / "26_输出_Stroop_LOSO_Riemannian基线"
EEGNET_ROOT = HERE / "28_输出_Stroop_LOSO_EEGNet基线"
CONFORMER_ROOT = HERE / "31_输出_Stroop_LOSO_EEGConformer基线"

CORE_PREDICTIONS = CORE_ROOT / "01_结果_十方法Recording预测.csv"
RIEMANN_PREDICTIONS = RIEMANN_ROOT / "03_结果_Riemannian五种子Recording预测.csv"
EEGNET_PREDICTIONS = EEGNET_ROOT / "03_结果_EEGNet五种子Recording预测.csv"
CONFORMER_PREDICTIONS = CONFORMER_ROOT / "03_结果_EEGConformer五种子Recording预测.csv"

INPUT_MANIFESTS = {
    CORE_PREDICTIONS: CORE_ROOT / "10_记录_运行信息和哈希.json",
    RIEMANN_PREDICTIONS: RIEMANN_ROOT / "06_记录_运行信息和哈希.json",
    EEGNET_PREDICTIONS: EEGNET_ROOT / "06_记录_运行信息和哈希.json",
    CONFORMER_PREDICTIONS: CONFORMER_ROOT / "06_记录_运行信息和哈希.json",
}

FROZEN_OUTPUT = OUTPUT_ROOT / "01_冻结预测_Stroop_LOSO_14方法Recording.csv"
SUBJECT_OUTPUT = OUTPUT_ROOT / "02_结果_14方法参与者准确率.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "03_结果_14方法准确率和95CI_显示名.csv"
PAIRED_OUTPUT = OUTPUT_ROOT / "04_结果_相对AlwaysNN配对检验和BH校正.csv"
TAIL_OUTPUT = OUTPUT_ROOT / "05_结果_尾部描述_仅探索性.csv"
ACCURACY_FIGURE = OUTPUT_ROOT / "06_图_14方法准确率和95CI.png"
DIFFERENCE_FIGURE = OUTPUT_ROOT / "07_图_融合方法相对AlwaysNN参与者差值.png"
MANIFEST_OUTPUT = OUTPUT_ROOT / "08_记录_冻结预测和统计运行清单.json"

sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.aggregation import sha256_file, validate_frozen_predictions  # noqa: E402


METHODS = [
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

DISPLAY_NAMES = {
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

FINAL_KEYS = ["dataset", "protocol", "subject", "direction", "recording_id", "method"]
RECORDING_KEYS = ["dataset", "protocol", "subject", "direction", "recording_id"]
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260729
EXPLORATORY_MARGIN = 0.05


def verify_input_hashes() -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for prediction_path, manifest_path in INPUT_MANIFESTS.items():
        if not prediction_path.exists():
            raise FileNotFoundError(f"缺少输入预测文件：{prediction_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"缺少输入运行清单：{manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("output_sha256", {}).get(prediction_path.name)
        actual = sha256_file(prediction_path)
        if expected != actual:
            raise RuntimeError(
                f"输入预测哈希与运行清单不一致：{prediction_path.name}\n"
                f"清单={expected}\n实际={actual}"
            )
        verified[prediction_path.name] = {
            "prediction_sha256": actual,
            "manifest_file": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
        }
    return verified


def load_and_validate_predictions() -> pd.DataFrame:
    frames = [
        pd.read_csv(CORE_PREDICTIONS),
        pd.read_csv(RIEMANN_PREDICTIONS),
        pd.read_csv(EEGNET_PREDICTIONS),
        pd.read_csv(CONFORMER_PREDICTIONS),
    ]
    predictions = pd.concat(frames, ignore_index=True)
    validate_frozen_predictions(predictions)

    if len(predictions) != 14 * 15 * 4:
        raise AssertionError(f"14方法冻结预测应有840行，实际为{len(predictions)}行")
    if set(predictions["method"].astype(str)) != set(METHODS):
        raise AssertionError("冻结预测的方法集合不是预期的14种方法")
    if set(predictions["dataset"].astype(str)) != {"openbci"}:
        raise AssertionError("冻结预测包含非预期数据集")
    if set(predictions["protocol"].astype(str)) != {"loso"}:
        raise AssertionError("冻结预测包含非LOSO协议")
    if set(predictions["direction"].astype(str)) != {"stroop"}:
        raise AssertionError("冻结预测不是完整的Stroop方向")
    if predictions.duplicated(FINAL_KEYS).any():
        raise AssertionError("冻结预测存在重复的方法-recording键")

    counts = predictions.groupby("method").size()
    if set(counts.to_numpy(dtype=int)) != {60}:
        raise AssertionError("每种方法必须恰好包含60条recording预测")

    reference = predictions[predictions["method"] == "always_nn"][
        RECORDING_KEYS + ["true_label", "n_windows"]
    ].copy()
    if len(reference) != 60 or reference.duplicated(RECORDING_KEYS).any():
        raise AssertionError("Always-NN不能提供唯一的60条参考recording")
    for method in METHODS:
        current = predictions[predictions["method"] == method][
            RECORDING_KEYS + ["true_label", "n_windows"]
        ]
        aligned = reference.merge(
            current,
            on=RECORDING_KEYS,
            suffixes=("_reference", "_method"),
            validate="one_to_one",
        )
        if len(aligned) != 60:
            raise AssertionError(f"{method}没有覆盖相同的60条recording")
        if not np.array_equal(
            aligned["true_label_reference"].to_numpy(),
            aligned["true_label_method"].to_numpy(),
        ):
            raise AssertionError(f"{method}的真实标签与Always-NN不一致")
        if not np.array_equal(
            aligned["n_windows_reference"].to_numpy(),
            aligned["n_windows_method"].to_numpy(),
        ):
            raise AssertionError(f"{method}的window计数与Always-NN不一致")

    probabilities = predictions[["p0", "p1"]].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise AssertionError("冻结概率包含非有限值或负值")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise AssertionError("冻结概率没有逐行归一化")
    expected_label = np.argmax(probabilities, axis=1).astype(int)
    if not np.array_equal(expected_label, predictions["pred_label"].to_numpy(dtype=int)):
        raise AssertionError("pred_label与冻结概率不一致")
    expected_correct = (
        expected_label == predictions["true_label"].to_numpy(dtype=int)
    ).astype(int)
    if not np.array_equal(expected_correct, predictions["correct"].to_numpy(dtype=int)):
        raise AssertionError("correct与冻结概率/标签不一致")

    method_order = {method: index for index, method in enumerate(METHODS)}
    predictions["_method_order"] = predictions["method"].map(method_order)
    predictions = predictions.sort_values(RECORDING_KEYS + ["_method_order"]).drop(
        columns="_method_order"
    )
    return predictions.reset_index(drop=True)


def bootstrap_distribution(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != 15:
        raise ValueError("参与者Bootstrap必须接收15个一维数值")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    return values[indices].mean(axis=1)


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        predictions.groupby(["dataset", "protocol", "direction", "subject", "method"], as_index=False)
        .agg(accuracy=("correct", "mean"), n_recordings=("recording_id", "nunique"))
    )
    if len(metrics) != 14 * 15 or set(metrics["n_recordings"].astype(int)) != {4}:
        raise AssertionError("每个参与者-方法必须恰好包含4条recording")
    metrics["method_display"] = metrics["method"].map(DISPLAY_NAMES)
    return metrics.sort_values(["method", "subject"]).reset_index(drop=True)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, method in enumerate(METHODS):
        group = metrics[metrics["method"] == method]
        values = group["accuracy"].to_numpy(dtype=np.float64)
        distribution = bootstrap_distribution(values, BOOTSTRAP_SEED + index)
        rows.append(
            {
                "dataset": "openbci",
                "protocol": "loso",
                "direction": "stroop",
                "method_id": method,
                "method": DISPLAY_NAMES[method],
                "n_participants": 15,
                "recordings_per_participant": 4,
                "mean_recording_accuracy": float(values.mean()),
                "ci95_low": float(np.quantile(distribution, 0.025)),
                "ci95_high": float(np.quantile(distribution, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_recording_accuracy", "method"], ascending=[False, True]
    ).reset_index(drop=True)


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


def paired_wilcoxon(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=np.float64)
    if np.allclose(difference, 0.0):
        return 1.0
    return float(
        wilcoxon(difference, alternative="two-sided", zero_method="pratt").pvalue
    )


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="subject", columns="method", values="accuracy")
    if set(pivot.columns) != set(METHODS) or len(pivot) != 15:
        raise AssertionError("参与者级配对矩阵不完整")
    rows: list[dict[str, object]] = []
    comparison_methods = [method for method in METHODS if method != "always_nn"]
    for index, method in enumerate(comparison_methods):
        difference = (
            pivot[method].to_numpy(dtype=np.float64)
            - pivot["always_nn"].to_numpy(dtype=np.float64)
        )
        distribution = bootstrap_distribution(difference, BOOTSTRAP_SEED + 100 + index)
        low = float(np.quantile(distribution, 0.025))
        high = float(np.quantile(distribution, 0.975))
        rows.append(
            {
                "dataset": "openbci",
                "protocol": "loso",
                "direction": "stroop",
                "method_id": method,
                "method": DISPLAY_NAMES[method],
                "comparator": DISPLAY_NAMES["always_nn"],
                "n_paired_participants": 15,
                "mean_difference": float(difference.mean()),
                "paired_ci95_low": low,
                "paired_ci95_high": high,
                "participants_improved": int(np.sum(difference > 0.0)),
                "participants_tied": int(np.sum(difference == 0.0)),
                "participants_worsened": int(np.sum(difference < 0.0)),
                "wilcoxon_p_raw": paired_wilcoxon(difference),
                "exploratory_margin": -EXPLORATORY_MARGIN,
                "lower_ci_above_minus_5pp": bool(low > -EXPLORATORY_MARGIN),
                "superior_by_ci": bool(low > 0.0),
                "analysis_status": "post-hoc exploratory; not confirmatory non-inferiority",
            }
        )
    output = pd.DataFrame(rows)
    output["bh_family"] = "all 13 Stroop LOSO comparisons versus Always-NN"
    output["wilcoxon_p_bh"] = bh_adjust(output["wilcoxon_p_raw"].to_numpy(dtype=np.float64))
    output["significant_after_bh_0_05"] = output["wilcoxon_p_bh"] < 0.05
    return output.sort_values(["wilcoxon_p_bh", "method"]).reset_index(drop=True)


def tail_descriptives(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="subject", columns="method", values="accuracy")
    rows: list[dict[str, object]] = []
    for method in METHODS:
        difference = (
            pivot[method].to_numpy(dtype=np.float64)
            - pivot["always_nn"].to_numpy(dtype=np.float64)
        )
        n_tail = max(1, int(math.ceil(0.10 * len(difference))))
        rows.append(
            {
                "dataset": "openbci",
                "protocol": "loso",
                "direction": "stroop",
                "method_id": method,
                "method": DISPLAY_NAMES[method],
                "comparator": DISPLAY_NAMES["always_nn"],
                "n_participants": 15,
                "mean_difference": float(difference.mean()),
                "worst_participant_difference": float(difference.min()),
                "empirical_cvar10": float(np.sort(difference)[:n_tail].mean()),
                "cvar_tail_participants": n_tail,
                "loss_rate": float(np.mean(difference < 0.0)),
                "loss_at_least_5pp_rate": float(np.mean(difference <= -0.05)),
                "status": "exploratory; 15 participants give only 2 observations in the 10% tail",
            }
        )
    return pd.DataFrame(rows)


def accuracy_figure(summary: pd.DataFrame) -> None:
    plot = summary.sort_values("mean_recording_accuracy", ascending=True).reset_index(drop=True)
    mean = 100.0 * plot["mean_recording_accuracy"].to_numpy(dtype=np.float64)
    low = 100.0 * plot["ci95_low"].to_numpy(dtype=np.float64)
    high = 100.0 * plot["ci95_high"].to_numpy(dtype=np.float64)
    colors = []
    for method in plot["method_id"]:
        if method == "always_nn":
            colors.append("#202020")
        elif method == "cbsf":
            colors.append("#C44E52")
        elif method == "dasf_clean":
            colors.append("#4C72B0")
        elif method in {"eegnet", "eeg_conformer"}:
            colors.append("#2A7F62")
        else:
            colors.append("#777777")

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    y = np.arange(len(plot))
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="none",
        ecolor="#555555",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    ax.scatter(mean, y, c=colors, s=42, zorder=2)
    ax.set_yticks(y, labels=plot["method"])
    ax.set_xlabel("Recording-level accuracy (%)")
    ax.set_xlim(40, 82)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(ACCURACY_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def difference_figure(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot(index="subject", columns="method", values="accuracy")
    selected = ["always_fuse", "dasf_clean", "cbsf", "stack_recording"]
    labels = [DISPLAY_NAMES[method] for method in selected]
    palette = ["#777777", "#4C72B0", "#C44E52", "#2A7F62"]
    offsets = np.linspace(-0.10, 0.10, 15)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for index, (method, color) in enumerate(zip(selected, palette)):
        difference = 100.0 * (
            pivot[method].to_numpy(dtype=np.float64)
            - pivot["always_nn"].to_numpy(dtype=np.float64)
        )
        ax.scatter(
            np.full(15, index, dtype=np.float64) + offsets,
            difference,
            color=color,
            alpha=0.75,
            s=28,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.scatter(index, difference.mean(), marker="D", color="black", s=48, zorder=4)
    ax.axhline(0.0, color="#202020", linewidth=1.0)
    ax.axhline(-5.0, color="#C44E52", linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=12, ha="right")
    ax.set_ylabel("Participant-level accuracy difference vs neural comparator (pp)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DIFFERENCE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("[1/4] 核对四个输入预测文件及运行清单哈希")
    verified_inputs = verify_input_hashes()

    print("[2/4] 合并并验证14方法、60条共同recording")
    predictions = load_and_validate_predictions()
    metrics = subject_metrics(predictions)
    summary = summary_table(metrics)
    paired = paired_comparisons(metrics)
    tail = tail_descriptives(metrics)

    print("[3/4] 写入唯一冻结预测、统计表和人类可读图")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(FROZEN_OUTPUT, index=False, lineterminator="\n")
    metrics.to_csv(SUBJECT_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")
    paired.to_csv(PAIRED_OUTPUT, index=False, lineterminator="\n")
    tail.to_csv(TAIL_OUTPUT, index=False, lineterminator="\n")
    accuracy_figure(summary)
    difference_figure(metrics)

    print("[4/4] 写入冻结哈希和运行信息")
    outputs = [
        FROZEN_OUTPUT,
        SUBJECT_OUTPUT,
        SUMMARY_OUTPUT,
        PAIRED_OUTPUT,
        TAIL_OUTPUT,
        ACCURACY_FIGURE,
        DIFFERENCE_FIGURE,
    ]
    manifest = {
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "task": "stroop",
        "protocol": "fully nested participant-disjoint LOSO",
        "methods": METHODS,
        "method_count": 14,
        "participants": 15,
        "recordings_per_participant": 4,
        "frozen_prediction_rows": int(len(predictions)),
        "evaluation_unit": "recording",
        "statistical_unit": "held-out participant",
        "seed_is_statistical_unit": False,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "paired_test": "two-sided Wilcoxon signed-rank with Pratt zero handling",
        "bh_family": "all 13 Stroop LOSO comparisons versus Always-NN",
        "tail_status": "exploratory; nominal 10% tail contains only 2 participants",
        "claim_scope": "no superiority, non-inferiority, safety, no-regret or formal risk-control claim without stated supporting inference",
        "verified_inputs": verified_inputs,
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = summary[["method", "mean_recording_accuracy", "ci95_low", "ci95_high"]].copy()
    for column in ("mean_recording_accuracy", "ci95_low", "ci95_high"):
        display[column] = 100.0 * display[column]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Stroop LOSO 14方法冻结统计完成：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
