from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from revision_pipeline.aggregation import sha256_file, validate_frozen_predictions
from revision_pipeline.protocol import Protocol


BOOTSTRAP_SEED = 20260716
BOOTSTRAP_SAMPLES = 10000


def percentile_interval(values: np.ndarray, lower: float = 0.025, upper: float = 0.975) -> tuple[float, float]:
    return float(np.quantile(values, lower)), float(np.quantile(values, upper))


def bootstrap_means(values: np.ndarray, samples: int = BOOTSTRAP_SAMPLES) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return values[indices].mean(axis=1)


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.tolist()


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(["dataset", "protocol", "subject", "method"])["correct"].agg(
        accuracy="mean", n_recordings="count"
    ).reset_index()


def audit_recording_resolution(predictions: pd.DataFrame) -> None:
    directional = predictions.groupby(
        ["dataset", "protocol", "subject", "direction", "method"]
    )["correct"].agg(mean="mean", count="count").reset_index()
    bad_count = directional[(directional["protocol"] == "cross_task") & (directional["count"] != 4)]
    if not bad_count.empty:
        raise ValueError("Cross-task direction cells must contain exactly four recordings")
    cross = directional[directional["protocol"] == "cross_task"]
    scaled = cross["mean"].to_numpy() * 4.0
    if not np.allclose(scaled, np.round(scaled), atol=1e-9):
        raise ValueError("Cross-task directional accuracies are not recording-level 25-point increments")


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "protocol", "method"], sort=True):
        values = group["accuracy"].to_numpy(dtype=np.float64)
        boot = bootstrap_means(values)
        low, high = percentile_interval(boot)
        rows.append(
            {
                "dataset": keys[0],
                "protocol": keys[1],
                "method": keys[2],
                "n_subjects": int(len(values)),
                "mean_accuracy": float(np.mean(values)),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(metrics: pd.DataFrame, margin: float) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "protocol"], sort=True):
        pivot = group.pivot(index="subject", columns="method", values="accuracy")
        if "always_nn" not in pivot:
            raise ValueError(f"Missing always_nn comparator for {keys}")
        protocol_rows = []
        for method in sorted(set(pivot.columns) - {"always_nn"}):
            paired = pivot[[method, "always_nn"]].dropna()
            difference = paired[method].to_numpy() - paired["always_nn"].to_numpy()
            boot = bootstrap_means(difference)
            low, high = percentile_interval(boot)
            if np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(difference, alternative="two-sided", zero_method="pratt").pvalue)
            protocol_rows.append(
                {
                    "dataset": keys[0],
                    "protocol": keys[1],
                    "method": method,
                    "comparator": "always_nn",
                    "n_subjects": int(len(difference)),
                    "mean_difference": float(np.mean(difference)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "noninferiority_margin": -float(margin),
                    "noninferior": bool(low > -float(margin)),
                    "superior_by_ci": bool(low > 0.0),
                    "wilcoxon_p_two_sided": p_value,
                }
            )
        adjusted = bh_adjust([row["wilcoxon_p_two_sided"] for row in protocol_rows])
        for row, value in zip(protocol_rows, adjusted):
            row["wilcoxon_p_fdr"] = value
        rows.extend(protocol_rows)
    return pd.DataFrame(rows)


def tail_descriptives(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "protocol"], sort=True):
        pivot = group.pivot(index="subject", columns="method", values="accuracy")
        for method in sorted(pivot.columns):
            differences = (pivot[method] - pivot["always_nn"]).dropna().to_numpy(dtype=np.float64)
            n_tail = max(1, int(math.ceil(0.10 * len(differences))))
            rows.append(
                {
                    "dataset": keys[0],
                    "protocol": keys[1],
                    "method": method,
                    "n_subjects": int(len(differences)),
                    "mean_benefit": float(np.mean(differences)),
                    "worst_subject_benefit": float(np.min(differences)),
                    "empirical_cvar10": float(np.mean(np.sort(differences)[:n_tail])),
                    "cvar_tail_subjects": n_tail,
                    "subject_severe_loss_rate_5pp": float(np.mean(differences <= -0.05)),
                    "subject_loss_rate": float(np.mean(differences < 0.0)),
                    "status": "exploratory_n_too_small_for_stable_10pct_tail",
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    protocols = list(summary["protocol"].unique())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.2 * len(protocols), 4.4), squeeze=False)
    colors = {
        "always_nn": "#333333",
        "always_fuse": "#C65D36",
        "risk_aware": "#1B7F79",
        "cbsf": "#1B7F79",
        "dasf_clean": "#D19A28",
        "fixed_blend_010": "#4C78A8",
        "equal_blend_050": "#8B6F9C",
        "stack_recording": "#5F6B6D",
        "rf": "#4C78A8",
        "extra_trees": "#8B6F9C",
    }
    for axis, protocol_name in zip(axes[0], protocols):
        data = summary[summary["protocol"] == protocol_name].sort_values("mean_accuracy")
        positions = np.arange(len(data))
        means = data["mean_accuracy"].to_numpy()
        errors = np.vstack([means - data["ci95_low"].to_numpy(), data["ci95_high"].to_numpy() - means])
        axis.errorbar(
            means,
            positions,
            xerr=errors,
            fmt="o",
            color="#222222",
            ecolor="#777777",
            capsize=3,
            markersize=0,
        )
        for position, (_, row) in zip(positions, data.iterrows()):
            axis.scatter(row["mean_accuracy"], position, s=56, color=colors.get(row["method"], "#555555"), zorder=3)
        axis.set_yticks(positions, data["method"].tolist())
        axis.set_xlabel("Recording-level accuracy")
        axis.set_title(protocol_name.replace("_", " ").title())
        axis.grid(axis="x", color="#dddddd", linewidth=0.8)
        axis.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_differences(metrics: pd.DataFrame, output: Path) -> None:
    protocols = list(metrics["protocol"].unique())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.2 * len(protocols), 4.4), squeeze=False)
    for axis, protocol_name in zip(axes[0], protocols):
        pivot = metrics[metrics["protocol"] == protocol_name].pivot(index="subject", columns="method", values="accuracy")
        preferred = (
            ("always_fuse", "dasf_clean", "cbsf")
            if "cbsf" in pivot
            else ("always_fuse", "risk_aware")
        )
        methods = [name for name in preferred if name in pivot]
        for position, method in enumerate(methods):
            difference = pivot[method] - pivot["always_nn"]
            jitter = np.linspace(-0.06, 0.06, len(difference))
            axis.scatter(np.full(len(difference), position) + jitter, 100.0 * difference, alpha=0.75, s=30)
            axis.plot(position, 100.0 * difference.mean(), marker="D", color="#111111", markersize=7)
        axis.axhline(0.0, color="#333333", linewidth=1)
        axis.axhline(-5.0, color="#B22222", linewidth=1, linestyle="--")
        display = {
            "always_fuse": "Always-fuse",
            "dasf_clean": "DASF",
            "cbsf": "CB-SF",
            "risk_aware": "Risk-aware",
        }
        axis.set_xticks(range(len(methods)), [display.get(name, name) for name in methods])
        axis.set_ylabel("Paired accuracy difference vs always-NN (pp)")
        axis.set_title(protocol_name.replace("_", " ").title())
        axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(
    output: Path,
    prediction_path: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    tails: pd.DataFrame,
    margin: float,
) -> None:
    primary_method = "cbsf" if "cbsf" in set(summary["method"]) else "risk_aware"
    primary_label = "CB-SF" if primary_method == "cbsf" else "risk-aware fusion"
    lines = [
        "# Corrected frozen-prediction results",
        "",
        f"Prediction SHA-256: `{sha256_file(prediction_path)}`",
        "",
        "All values below use recording-level decisions from the final five-seed ensemble. The held-out subject is the statistical unit. Seeds, directions, windows and recordings are not treated as independent replicates.",
        "",
        f"The primary comparison is {primary_label} versus always-NN with a prespecified non-inferiority margin of {100 * margin:.1f} percentage points. CVaR10 and severe-loss rates are exploratory because 15 subjects provide only two observations in the empirical 10% tail.",
        "",
        "## Accuracy",
        "",
        markdown_table(summary),
        "",
        "## Paired comparisons versus always-NN",
        "",
        markdown_table(comparisons),
        "",
        "## Exploratory subject-level tail descriptors",
        "",
        markdown_table(tails),
        "",
        "Do not copy numbers from intermediate logs or historical tables into the manuscript. Every submitted table and figure must be generated from the frozen prediction file named above.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all statistical tables and figures from one frozen file")
    base = Path(__file__).resolve().parent
    parser.add_argument("--predictions", type=Path, default=base / "frozen" / "predictions_recording.csv")
    parser.add_argument("--output", type=Path, default=base / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = Protocol.load(Path(__file__).resolve().parent / "protocol.json")
    predictions = pd.read_csv(args.predictions)
    validate_frozen_predictions(predictions)
    audit_recording_resolution(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = subject_metrics(predictions)
    summary = summary_table(metrics)
    comparisons = paired_comparisons(metrics, float(protocol.raw["noninferiority_margin"]))
    tails = tail_descriptives(metrics)
    metrics.to_csv(args.output / "table_subject_metrics.csv", index=False)
    summary.to_csv(args.output / "table_accuracy.csv", index=False)
    comparisons.to_csv(args.output / "table_paired_comparisons.csv", index=False)
    tails.to_csv(args.output / "table_tail_exploratory.csv", index=False)
    plot_summary(summary, args.output / "figure_accuracy_ci.png")
    plot_paired_differences(metrics, args.output / "figure_paired_differences.png")
    write_markdown(
        args.output / "results_summary.md",
        args.predictions,
        summary,
        comparisons,
        tails,
        float(protocol.raw["noninferiority_margin"]),
    )
    audit = {
        "prediction_sha256": sha256_file(args.predictions),
        "protocol_sha256": protocol.digest,
        "input_rows": int(len(predictions)),
        "subject_metric_rows": int(len(metrics)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (args.output / "output_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated outputs: {args.output}")


if __name__ == "__main__":
    main()
