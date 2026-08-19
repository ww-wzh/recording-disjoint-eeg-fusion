"""直接运行：由编号 18 的冻结预测生成 Stroop LOSO 融合与轻量对照。

本程序不重新训练 Residual MLP、随机森林、ExtraTrees 或窗口级 stacker。
它生成 DASF、修正后 CB-SF、三种固定权重和 recording-level OOF stacker，
结果写入编号 23 的独立目录。无命令行参数，可直接在 PyCharm 运行。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
SOURCE_ROOT = HERE / "18_输出_Stroop_LOSO核心模型"
CORE_ROOT = HERE / "20_输出_Stroop_LOSO核心模型检查"
OUTPUT_ROOT = HERE / "23_输出_Stroop_LOSO融合和对照"

RAW_PATH = SOURCE_ROOT / "02_汇总_目标窗口种子预测.csv"
TARGET_DIAGNOSTIC_PATH = SOURCE_ROOT / "03_汇总_目标诊断.csv"
META_DIAGNOSTIC_PATH = SOURCE_ROOT / "04_汇总_内层参与者诊断.csv"
META_RECORDING_PATH = SOURCE_ROOT / "05_汇总_内层Recording预测.csv"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "06_记录_运行信息和哈希.json"
CORE_RECORDING_PATH = CORE_ROOT / "01_结果_核心模型五种子Recording预测.csv"
CORE_AUDIT_PATH = CORE_ROOT / "04_记录_完整性审计和哈希.json"
ROUTE_PROTOCOL_PATH = ROUTE_A_ROOT / "protocol_route_a.json"
ROBUST_PROTOCOL_PATH = ROUTE_A_ROOT / "protocol_cbsf_robustness.json"

FINAL_PREDICTION_PATH = OUTPUT_ROOT / "01_结果_十方法Recording预测.csv"
SUMMARY_PATH = OUTPUT_ROOT / "02_结果_十方法汇总_显示名.csv"
SUBJECT_PATH = OUTPUT_ROOT / "03_结果_十方法参与者准确率.csv"
PAIRED_PATH = OUTPUT_ROOT / "04_结果_相对AlwaysNN配对检验和BH校正.csv"
TAIL_PATH = OUTPUT_ROOT / "05_结果_尾部描述_仅探索性.csv"
DASF_PATH = OUTPUT_ROOT / "06_诊断_DASF参与者决策.csv"
CBSF_PATH = OUTPUT_ROOT / "07_诊断_CBSF参与者权重.csv"
STACK_PATH = OUTPUT_ROOT / "08_诊断_Recording级Stacker.csv"
GATE_TRAINING_PATH = OUTPUT_ROOT / "09_审计_CBSF内层门控训练行.csv"
MANIFEST_PATH = OUTPUT_ROOT / "10_记录_运行信息和哈希.json"

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(ROUTE_A_ROOT))

from revision_pipeline.aggregation import (  # noqa: E402
    RAW_KEYS,
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.risk_gate import (  # noqa: E402
    GATE_FEATURES,
    blend_probabilities,
    fit_predict_gate_robust,
)
from route_a_lib.cbsf_audit import make_gate_training_rows  # noqa: E402
from route_a_lib.probability import (  # noqa: E402
    clean_dasf_decision,
    fit_recording_stacker,
    recording_rows_from_probabilities,
)


SEEDS = [1335, 1388, 1441, 1494, 1547]
BLENDS = {
    "fixed_blend_010": 0.10,
    "fixed_blend_025": 0.25,
    "equal_blend_050": 0.50,
}
DISPLAY_NAMES = {
    "always_nn": "Always-NN",
    "rf": "Random forest",
    "extra_trees": "ExtraTrees",
    "always_fuse": "Always-fuse",
    "fixed_blend_010": "Fixed soft fusion (w=0.10)",
    "fixed_blend_025": "Fixed soft fusion (w=0.25)",
    "equal_blend_050": "Equal-weight soft fusion (w=0.50)",
    "stack_recording": "Recording-level OOF stacker",
    "dasf_clean": "DASF (exploratory corrected implementation)",
    "cbsf": "CB-SF (exploratory robust implementation)",
}


def bootstrap_means(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260716)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return values[indices].mean(axis=1)


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["dataset", "protocol", "subject", "method"])["correct"]
        .agg(accuracy="mean", n_recordings="count")
        .reset_index()
    )


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "protocol", "method"], sort=True):
        values = group["accuracy"].to_numpy(dtype=np.float64)
        boot = bootstrap_means(values)
        rows.append(
            {
                "dataset": keys[0],
                "protocol": keys[1],
                "method": keys[2],
                "n_subjects": int(len(values)),
                "mean_accuracy": float(values.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output.tolist()


def paired_comparisons(metrics: pd.DataFrame, margin: float) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["dataset", "protocol"], sort=True):
        pivot = group.pivot(index="subject", columns="method", values="accuracy")
        protocol_rows = []
        for method in sorted(set(pivot.columns) - {"always_nn"}):
            paired = pivot[[method, "always_nn"]].dropna()
            difference = paired[method].to_numpy() - paired["always_nn"].to_numpy()
            boot = bootstrap_means(difference)
            p_value = (
                1.0
                if np.allclose(difference, 0.0)
                else float(wilcoxon(difference, alternative="two-sided", zero_method="pratt").pvalue)
            )
            protocol_rows.append(
                {
                    "dataset": keys[0],
                    "protocol": keys[1],
                    "method": method,
                    "comparator": "always_nn",
                    "n_subjects": int(len(difference)),
                    "mean_difference": float(difference.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "noninferiority_margin": -float(margin),
                    "noninferior": bool(np.quantile(boot, 0.025) > -float(margin)),
                    "superior_by_ci": bool(np.quantile(boot, 0.025) > 0.0),
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
                    "mean_benefit": float(differences.mean()),
                    "worst_subject_benefit": float(differences.min()),
                    "empirical_cvar10": float(np.sort(differences)[:n_tail].mean()),
                    "cvar_tail_subjects": n_tail,
                    "subject_severe_loss_rate_5pp": float(np.mean(differences <= -0.05)),
                    "subject_loss_rate": float(np.mean(differences < 0.0)),
                    "status": "exploratory_n_too_small_for_stable_10pct_tail",
                }
            )
    return pd.DataFrame(rows)


def paired_windows(window_ensemble: pd.DataFrame, subject: int) -> pd.DataFrame:
    subset = window_ensemble[
        (window_ensemble["subject"].astype(int) == int(subject))
        & window_ensemble["method"].isin(["always_nn", "always_fuse"])
    ].copy()
    keys = [key for key in RAW_KEYS if key != "method"] + ["true_label", "ensemble_size"]
    nn = subset[subset["method"] == "always_nn"][keys + ["p0", "p1"]].rename(
        columns={"p0": "p0_nn", "p1": "p1_nn"}
    )
    fuse = subset[subset["method"] == "always_fuse"][keys + ["p0", "p1"]].rename(
        columns={"p0": "p0_fuse", "p1": "p1_fuse"}
    )
    paired = nn.merge(fuse, on=keys, how="inner", validate="one_to_one")
    if len(paired) * 2 != len(subset):
        raise AssertionError(f"S{subject:02d} 的 Always-NN 与 Always-fuse 窗口不一致")
    return paired.sort_values(["recording_id", "window_id"]).reset_index(drop=True)


def window_rows(paired: pd.DataFrame, probability: np.ndarray, method: str) -> pd.DataFrame:
    output = paired[[key for key in RAW_KEYS if key != "method"] + ["true_label", "ensemble_size"]].copy()
    output["method"] = str(method)
    output["p0"] = probability[:, 0]
    output["p1"] = probability[:, 1]
    return output[RAW_KEYS + ["true_label", "p0", "p1", "ensemble_size"]]


def median_meta_recordings(meta_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["outer_subject", "meta_subject", "recording_id", "method"]
    if meta_seed.duplicated(keys + ["seed"]).any():
        raise AssertionError("内层 Recording 种子预测存在重复键")
    seeds = meta_seed.groupby(keys)["seed"].agg(lambda values: set(int(v) for v in values))
    if seeds.map(lambda values: values != set(SEEDS)).any():
        raise AssertionError("每个内层 Recording/方法必须恰有五个冻结种子")
    labels = meta_seed.groupby(keys, as_index=False)["true_label"].agg(
        lambda values: int(np.unique(values)[0]) if np.unique(values).size == 1 else -1
    )
    if (labels["true_label"] < 0).any():
        raise AssertionError("内层 Recording 标签在种子之间不一致")
    probability = meta_seed.groupby(keys, as_index=False)[["p0", "p1"]].median()
    output = labels.merge(probability, on=keys, validate="one_to_one")
    output[["p0", "p1"]] = output[["p0", "p1"]].div(output[["p0", "p1"]].sum(axis=1), axis=0)
    output["dataset"] = "openbci"
    output["protocol"] = "loso"
    output["subject"] = output["meta_subject"].astype(int)
    output["direction"] = "stroop"
    output["pred_label"] = np.argmax(output[["p0", "p1"]].to_numpy(dtype=np.float64), axis=1)
    output["correct"] = (output["pred_label"] == output["true_label"]).astype(int)
    output["n_windows"] = -1
    output["ensemble_size"] = 5
    return output


def build_fixed_blends(window_ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in range(1, 16):
        paired = paired_windows(window_ensemble, subject)
        nn = paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64)
        fuse = paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64)
        for method, weight in BLENDS.items():
            rows.append(window_rows(paired, blend_probabilities(nn, fuse, weight), method))
    return aggregate_recordings(pd.concat(rows, ignore_index=True))


def build_dasf(
    window_ensemble: pd.DataFrame,
    meta: pd.DataFrame,
    target_diagnostics: pd.DataFrame,
    route_protocol: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = route_protocol["dasf_clean"]
    windows = []
    decisions = []
    for subject in range(1, 16):
        train = meta[meta["outer_subject"].astype(int) == subject].copy()
        if subject in set(train["meta_subject"].astype(int)):
            raise AssertionError("DASF 内层验证泄漏了外层参与者")
        accuracy = train.groupby("method")["correct"].mean()
        target = target_diagnostics[target_diagnostics["subject"].astype(int) == subject]
        covariance_values = target["covariance_shift"].to_numpy(dtype=np.float64)
        if not np.allclose(covariance_values, covariance_values[0], atol=1e-12, rtol=0.0):
            raise AssertionError(f"S{subject:02d} 的 covariance shift 在种子之间不一致")
        decision = clean_dasf_decision(
            float(accuracy["always_nn"]),
            float(accuracy["always_fuse"]),
            float(covariance_values[0]),
            base_margin=float(config["base_margin"]),
            alpha=float(config["shift_alpha"]),
            max_margin=float(config["max_margin"]),
        )
        paired = paired_windows(window_ensemble, subject)
        selected = (
            paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64)
            if decision.use_fusion
            else paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64)
        )
        windows.append(window_rows(paired, selected, "dasf_clean"))
        decisions.append(
            {
                "subject": subject,
                "direction": "stroop",
                "validation_accuracy_nn": decision.validation_accuracy_nn,
                "validation_accuracy_fuse": decision.validation_accuracy_fuse,
                "validation_gain": decision.validation_gain,
                "covariance_shift": decision.shift,
                "effective_margin": decision.effective_margin,
                "selected_model": "Always-fuse" if decision.use_fusion else "Always-NN",
                "used_fusion": decision.use_fusion,
                "meta_training_participants": int(train["meta_subject"].nunique()),
                "outer_subject_excluded": True,
                "decision_unit": "final five-seed ensemble",
            }
        )
    recordings = aggregate_recordings(pd.concat(windows, ignore_index=True))
    return recordings, pd.DataFrame(decisions)


def build_cbsf(
    window_ensemble: pd.DataFrame,
    target_diagnostics: pd.DataFrame,
    gate_training: pd.DataFrame,
    robust_protocol: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_config = {**robust_protocol["risk_gate"], **robust_protocol["gate_guardrails"]}
    target = target_diagnostics.groupby("subject", as_index=False)[GATE_FEATURES].median()
    windows = []
    weights = []
    for row in target.sort_values("subject").to_dict("records"):
        subject = int(row["subject"])
        training = gate_training[gate_training["outer_subject"].astype(int) == subject].copy()
        if subject in set(training["meta_subject"].astype(int)):
            raise AssertionError("CB-SF 门控训练泄漏了外层参与者")
        gate = fit_predict_gate_robust(
            training,
            pd.Series(row),
            catastrophe_threshold=float(gate_config["catastrophe_threshold"]),
            benefit_alpha=float(gate_config["benefit_alpha"]),
            catastrophe_c=float(gate_config["catastrophe_C"]),
            risk_penalty=float(gate_config["risk_penalty"]),
            temperature=float(gate_config["temperature"]),
            max_weight=float(gate_config["max_weight"]),
            feature_z_cap=float(gate_config["feature_z_cap"]),
            ood_z_threshold=float(gate_config["ood_z_threshold"]),
        )
        paired = paired_windows(window_ensemble, subject)
        probability = blend_probabilities(
            paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64),
            paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64),
            gate.weight,
        )
        windows.append(window_rows(paired, probability, "cbsf"))
        weights.append(
            {
                "subject": subject,
                "direction": "stroop",
                "fusion_weight": gate.weight,
                "predicted_benefit": gate.predicted_benefit,
                "predicted_benefit_raw": gate.predicted_benefit_raw,
                "predicted_catastrophe": gate.predicted_catastrophe,
                "ood_fallback": gate.ood_fallback,
                "max_abs_robust_z": gate.max_abs_robust_z,
                "meta_training_participants": int(training["meta_subject"].nunique()),
                "outer_subject_excluded": True,
                "target_batch_recordings": 4,
                "usage_mode": "target-batch/transductive",
                "analysis_status": "post-hoc exploratory",
            }
        )
    recordings = aggregate_recordings(pd.concat(windows, ignore_index=True))
    return recordings, pd.DataFrame(weights)


def build_recording_stacker(
    core: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = core[core["method"].isin(["always_nn", "always_fuse"])].copy()
    rows = []
    diagnostics = []
    for subject in range(1, 16):
        train = meta[meta["outer_subject"].astype(int) == subject].copy()
        test = base[base["subject"].astype(int) == subject].copy()
        if subject in set(train["meta_subject"].astype(int)):
            raise AssertionError("Recording stacker 训练泄漏了外层参与者")
        probability = fit_recording_stacker(
            train,
            test,
            c_value=0.05,
            seed=20262728 + subject,
            methods=("always_nn", "always_fuse"),
        )
        rows.append(
            recording_rows_from_probabilities(
                test,
                probability,
                "stack_recording",
                input_methods=("always_nn", "always_fuse"),
            )
        )
        diagnostics.append(
            {
                "subject": subject,
                "direction": "stroop",
                "meta_training_participants": int(train["meta_subject"].nunique()),
                "meta_training_recordings": int(train["recording_id"].nunique()),
                "outer_subject_excluded": True,
                "meta_source": "outer-training participant-disjoint OOF recording predictions",
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


def main() -> None:
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    core_audit = json.loads(CORE_AUDIT_PATH.read_text(encoding="utf-8"))
    if sha256_file(RAW_PATH) != source_manifest["output_sha256"][RAW_PATH.name]:
        raise RuntimeError("编号 18 的目标预测哈希发生变化")
    if sha256_file(CORE_RECORDING_PATH) != core_audit["output_sha256"][CORE_RECORDING_PATH.name]:
        raise RuntimeError("编号 20 的核心 Recording 结果哈希发生变化")

    route_protocol = json.loads(ROUTE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    robust_protocol = json.loads(ROBUST_PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw = pd.read_csv(RAW_PATH)
    target_diagnostics = pd.read_csv(TARGET_DIAGNOSTIC_PATH)
    meta_diagnostics = pd.read_csv(META_DIAGNOSTIC_PATH)
    meta_seed = pd.read_csv(META_RECORDING_PATH)
    core = pd.read_csv(CORE_RECORDING_PATH)

    window_ensemble = median_seed_ensemble(raw, SEEDS)
    meta = median_meta_recordings(meta_seed)
    gate_training = make_gate_training_rows(meta_seed, meta_diagnostics)

    fixed = build_fixed_blends(window_ensemble)
    dasf, dasf_diagnostics = build_dasf(window_ensemble, meta, target_diagnostics, route_protocol)
    cbsf, cbsf_diagnostics = build_cbsf(
        window_ensemble,
        target_diagnostics,
        gate_training,
        robust_protocol,
    )
    stacker, stacker_diagnostics = build_recording_stacker(core, meta)
    final = pd.concat([core, fixed, dasf, cbsf, stacker], ignore_index=True)
    final = final.sort_values(
        ["dataset", "protocol", "subject", "direction", "recording_id", "method"]
    ).reset_index(drop=True)
    validate_frozen_predictions(final)
    if len(final) != 15 * 4 * 10:
        raise AssertionError(f"十方法结果应有 600 行，实际为 {len(final)} 行")

    metrics = subject_metrics(final)
    summary = summary_table(metrics)
    summary.insert(3, "method_display", summary["method"].map(DISPLAY_NAMES))
    paired = paired_comparisons(metrics, margin=0.05)
    paired.insert(3, "method_display", paired["method"].map(DISPLAY_NAMES))
    tail = tail_descriptives(metrics)
    tail.insert(3, "method_display", tail["method"].map(DISPLAY_NAMES))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    final.to_csv(FINAL_PREDICTION_PATH, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n")
    metrics.to_csv(SUBJECT_PATH, index=False, lineterminator="\n")
    paired.to_csv(PAIRED_PATH, index=False, lineterminator="\n")
    tail.to_csv(TAIL_PATH, index=False, lineterminator="\n")
    dasf_diagnostics.to_csv(DASF_PATH, index=False, lineterminator="\n")
    cbsf_diagnostics.to_csv(CBSF_PATH, index=False, lineterminator="\n")
    stacker_diagnostics.to_csv(STACK_PATH, index=False, lineterminator="\n")
    gate_training.to_csv(GATE_TRAINING_PATH, index=False, lineterminator="\n")

    output_files = [
        FINAL_PREDICTION_PATH,
        SUMMARY_PATH,
        SUBJECT_PATH,
        PAIRED_PATH,
        TAIL_PATH,
        DASF_PATH,
        CBSF_PATH,
        STACK_PATH,
        GATE_TRAINING_PATH,
    ]
    manifest = {
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "source_sha256": {
            RAW_PATH.name: sha256_file(RAW_PATH),
            TARGET_DIAGNOSTIC_PATH.name: sha256_file(TARGET_DIAGNOSTIC_PATH),
            META_DIAGNOSTIC_PATH.name: sha256_file(META_DIAGNOSTIC_PATH),
            META_RECORDING_PATH.name: sha256_file(META_RECORDING_PATH),
            CORE_RECORDING_PATH.name: sha256_file(CORE_RECORDING_PATH),
            ROUTE_PROTOCOL_PATH.name: sha256_file(ROUTE_PROTOCOL_PATH),
            ROBUST_PROTOCOL_PATH.name: sha256_file(ROBUST_PROTOCOL_PATH),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "output_sha256": {path.name: sha256_file(path) for path in output_files},
        "methods": sorted(final["method"].astype(str).unique().tolist()),
        "recording_rows": int(len(final)),
        "participants": int(final["subject"].nunique()),
        "recordings_per_participant": 4,
        "statistical_unit": "held-out participant",
        "evaluation_unit": "recording",
        "seed_is_statistical_unit": False,
        "ensemble_size": 5,
        "cbsf_usage_mode": "target-batch/transductive using four unlabeled target recordings",
        "claim_scope": "exploratory risk-aware fusion; no formal safety, no-regret or risk-control guarantee",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = summary[["method_display", "n_subjects", "mean_accuracy", "ci95_low", "ci95_high"]].copy()
    for column in ("mean_accuracy", "ci95_low", "ci95_high"):
        display[column] = 100.0 * display[column]
    print(display.sort_values("mean_accuracy", ascending=False).to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"DASF 选择 Always-fuse：{int(dasf_diagnostics['used_fusion'].sum())}/15 名参与者")
    print(f"CB-SF OOD 回退：{int(cbsf_diagnostics['ood_fallback'].sum())}/15 名参与者")
    print(f"输出目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
