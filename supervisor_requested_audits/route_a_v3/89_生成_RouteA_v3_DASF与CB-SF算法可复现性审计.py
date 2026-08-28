"""编号89：生成 Route A v3 DASF 与 CB-SF 算法可复现性审计。

本文件不重新训练基础模型，也不改变81号冻结预测。它从实际v3代码、协议和
已完成的73/74号输出中核对公式、超参数、目标批次和数据流，并生成投稿补充材料。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：17_algorithm_reproducibility_audit
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
DASF_RUNNER = HERE / "73_运行_RouteA_v3_DASF硬门控.py"
CBSF_RUNNER = HERE / "74_运行_RouteA_v3_CB-SF软门控.py"
RISK_GATE_SOURCE = REVISION_ROOT / "revision_pipeline" / "risk_gate.py"
MODEL_SOURCE = REVISION_ROOT / "revision_pipeline" / "models.py"
AGGREGATION_SOURCE = REVISION_ROOT / "revision_pipeline" / "aggregation.py"
PROBABILITY_SOURCE = REVISION_ROOT / "route_a" / "route_a_lib" / "probability.py"
DATA_SOURCE = REVISION_ROOT / "route_a" / "route_a_lib" / "data.py"
DASF_ROOT = HERE / "03_dasf"
CBSF_ROOT = HERE / "04_cbsf"
OUTPUT_ROOT = HERE / "17_algorithm_reproducibility_audit"

FEATURE_CSV = OUTPUT_ROOT / "89_CB-SF五个门控特征精确定义.csv"
HYPERPARAMETER_CSV = OUTPUT_ROOT / "89_DASF与CB-SF完整超参数.csv"
DATAFLOW_CSV = OUTPUT_ROOT / "89_训练验证测试数据流.csv"
CELL_AUDIT_CSV = OUTPUT_ROOT / "89_实际门控单元与训练规模审计.csv"
MACHINE_JSON = OUTPUT_ROOT / "89_DASF与CB-SF机器可读算法定义.json"
SUPPLEMENT_MD = OUTPUT_ROOT / "89_Supplementary_Exact_DASF_and_CB-SF_Algorithms.md"
CHINESE_MD = OUTPUT_ROOT / "89_中文解读_DASF与CB-SF完整流程.md"
MANIFEST_JSON = OUTPUT_ROOT / "89_DASF与CB-SF算法审计_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的行：{path.name}")
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def json_compatible(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def verify_source_implementation() -> None:
    source_requirements = {
        RISK_GATE_SOURCE: [
            "GATE_FEATURES",
            "robust_feature_shift",
            "per_feature_cap: float = 10.0",
            "scale_epsilon: float = 1e-6",
            "feature_z_cap: float = 5.0",
            "ood_z_threshold: float = 10.0",
            "catastrophe = (y_benefit <= -float(catastrophe_threshold))",
            "score = predicted_benefit - float(risk_penalty) * predicted_catastrophe",
            "scaled = np.clip(score / float(temperature), -40.0, 40.0)",
        ],
        PROBABILITY_SOURCE: [
            "p1 = np.clip",
            "np.log(p1 / (1.0 - p1))",
            "effective = float(np.clip(base_margin + alpha * shift, 0.0, max_margin))",
            "use_fusion = bool(finite and gain > effective)",
        ],
        DATA_SOURCE: [
            "def normalized_frobenius_shift",
            "np.linalg.norm(source_mean - target_mean, ord=\"fro\") / denominator",
        ],
        AGGREGATION_SOURCE: [
            "groupby(RAW_KEYS",
            "[[\"p0\", \"p1\"]].median()",
            "geom = np.exp(np.mean(np.log(probs), axis=0))",
        ],
        MODEL_SOURCE: [
            "OOF predictions inside the outer fold",
            "sample_weight=stack_weight",
            "p1 = np.clip(np.asarray(values)[:, 1], 1e-6, 1.0 - 1e-6)",
        ],
        DASF_RUNNER: [
            "base_margin=0.02, alpha=0.2, max_margin=0.15",
            "gate_source = source.labels_raw >= 1",
            "med_nn = float(group[\"validation_accuracy_nn\"].median())",
        ],
        CBSF_RUNNER: [
            "feature_z_cap=5.0",
            "ood_z_threshold=10.0",
            "training = crosstraining[crosstraining[\"subject\"].astype(int) != int(target[\"subject\"])]",
            "blended = blend_probabilities",
        ],
    }
    for path, fragments in source_requirements.items():
        text = path.read_text(encoding="utf-8")
        missing = [fragment for fragment in fragments if fragment not in text]
        if missing:
            raise AssertionError(f"{path.name}缺少冻结实现片段：{missing}")


def build_gate_feature_rows() -> list[dict]:
    return [
        {
            "feature_order": 1,
            "feature": "Robust feature shift",
            "symbol": "d_feature",
            "exact_formula": "median_j clip(z_j,0,10), z_j=abs(mean_T x_j-mean_S x_j)/std_S(x_j)",
            "zero_variance_and_outlier_handling": "If std_S<1e-6: z_j=0 when mean difference<=1e-6, otherwise 10; non-finite values become 10; all z_j are clipped to [0,10].",
            "aggregation_unit": "All windows in the source reference batch and all windows in the unlabeled target batch.",
        },
        {
            "feature_order": 2,
            "feature": "Branch disagreement",
            "symbol": "d_prob",
            "exact_formula": "mean_i abs(p_NN,i(class 1)-p_fuse,i(class 1))",
            "zero_variance_and_outlier_handling": "Frozen probabilities must be finite and row-normalized.",
            "aggregation_unit": "Mean over every window in the target participant-direction batch.",
        },
        {
            "feature_order": 3,
            "feature": "Neural entropy",
            "symbol": "H_NN",
            "exact_formula": "mean_i[-sum_k p_NN,ik log(p_NN,ik)]",
            "zero_variance_and_outlier_handling": "Each probability is clipped from below at 1e-12 before log.",
            "aggregation_unit": "Mean binary Shannon entropy over all target-batch windows; natural logarithm.",
        },
        {
            "feature_order": 4,
            "feature": "Fused entropy",
            "symbol": "H_fuse",
            "exact_formula": "mean_i[-sum_k p_fuse,ik log(p_fuse,ik)]",
            "zero_variance_and_outlier_handling": "Each probability is clipped from below at 1e-12 before log.",
            "aggregation_unit": "Mean binary Shannon entropy over all target-batch windows; natural logarithm.",
        },
        {
            "feature_order": 5,
            "feature": "Neural confidence",
            "symbol": "c_NN",
            "exact_formula": "mean_i max_k p_NN,ik",
            "zero_variance_and_outlier_handling": "Frozen probabilities must be finite and row-normalized.",
            "aggregation_unit": "Mean over every window in the target participant-direction batch.",
        },
    ]


def build_hyperparameter_rows(policy: dict) -> list[dict]:
    gate = policy["models"]["risk_gate"]
    ridge_params = Ridge(alpha=float(gate["benefit_alpha"])).get_params(deep=False)
    catastrophe_params = LogisticRegression(
        C=float(gate["catastrophe_C"]),
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    ).get_params(deep=False)
    stacker_params = LogisticRegression(
        C=float(policy["models"]["stacker"]["C"]),
        class_weight="balanced",
        max_iter=int(policy["models"]["stacker"]["max_iter"]),
        random_state=0,
    ).get_params(deep=False)

    rows = [
        {
            "component": "DASF",
            "parameter": "base margin",
            "value": 0.02,
            "meaning": "Minimum validation gain required before selecting the fused branch.",
            "status": "Inherited exploratory heuristic; frozen before v3 rerun, not externally validated.",
        },
        {
            "component": "DASF",
            "parameter": "shift coefficient",
            "value": 0.2,
            "meaning": "Adds covariance-shift-dependent conservatism to the decision margin.",
            "status": "Inherited exploratory heuristic; frozen before v3 rerun, not externally validated.",
        },
        {
            "component": "DASF",
            "parameter": "maximum margin",
            "value": 0.15,
            "meaning": "Upper cap on the effective selection margin.",
            "status": "Inherited exploratory heuristic; frozen before v3 rerun, not externally validated.",
        },
        {
            "component": "CB-SF",
            "parameter": "catastrophe threshold",
            "value": float(gate["catastrophe_threshold"]),
            "meaning": "Training row is labelled catastrophic when fused-minus-neural OOF accuracy is <= -0.05.",
            "status": "Five-percentage-point exploratory definition; not a clinical or engineering safety limit.",
        },
        {
            "component": "CB-SF",
            "parameter": "risk penalty",
            "value": float(gate["risk_penalty"]),
            "meaning": "Coefficient multiplying predicted catastrophe probability in the gate score.",
            "status": "Inherited exploratory heuristic; no formal risk constraint or guarantee.",
        },
        {
            "component": "CB-SF",
            "parameter": "temperature",
            "value": float(gate["temperature"]),
            "meaning": "Scales the benefit-minus-penalty score before the logistic weight mapping.",
            "status": "Inherited exploratory heuristic; controls numerical softness only.",
        },
        {
            "component": "CB-SF",
            "parameter": "maximum fused weight",
            "value": float(gate["max_weight"]),
            "meaning": "Caps contribution of the always-fuse branch at 0.8.",
            "status": "Inherited exploratory heuristic; keeps at least 0.2 neural contribution but is not a risk guarantee.",
        },
        {
            "component": "CB-SF robust scaling",
            "parameter": "feature z cap / OOD threshold",
            "value": "5.0 / 10.0",
            "meaning": "Clip robust-scaled gate features to +/-5; set fusion weight to zero if an unclipped target value exceeds +/-10.",
            "status": "Post-hoc robustness rule, disclosed as exploratory.",
        },
        {
            "component": "Probability processing",
            "parameter": "stacker logit clip / entropy and aggregation clip",
            "value": "1e-6 / 1e-12",
            "meaning": "Avoid infinite logits and log(0); probability rows are normalized before or after blending.",
            "status": "Numerical safeguards.",
        },
    ]
    for prefix, params in [
        ("CB-SF Ridge benefit model", ridge_params),
        ("CB-SF catastrophe logistic model", catastrophe_params),
        ("Core OOF fusion stacker", stacker_params),
    ]:
        for parameter, value in sorted(params.items()):
            rows.append(
                {
                    "component": prefix,
                    "parameter": parameter,
                    "value": json_compatible(value),
                    "meaning": "Exact scikit-learn estimator parameter recorded from the installed implementation.",
                    "status": f"scikit-learn {sklearn.__version__}",
                }
            )
    return rows


def build_dataflow_rows() -> list[dict]:
    return [
        {
            "method": "Core fused branch",
            "setting": "Every outer evaluation cell",
            "training_information": "Base-model OOF probabilities from recording-disjoint inner folds; stacker uses only OOF logits with inverse-recording-window sample weights.",
            "target_information": "Final base models generate neural, RF and Extra Trees probabilities for the untouched target recordings.",
            "target_labels_used_by_gate": False,
            "output": "Always-neural and always-fuse window probabilities for each of five seeds, then per-class window-level seed median.",
        },
        {
            "method": "DASF",
            "setting": "Bidirectional cross-task",
            "training_information": "For each source participant, two recording-disjoint folds over four source recordings; OOF gain is calculated on source levels r1-r3.",
            "target_information": "Covariance shift uses all windows from the same participant's three unlabeled target recordings r1-r3.",
            "target_labels_used_by_gate": False,
            "output": "One cell-level hard choice between the frozen five-seed-median always-neural and always-fuse probabilities.",
        },
        {
            "method": "DASF",
            "setting": "Nested LOSO",
            "training_information": "Within each outer training population, inner participant-level OOF predictions estimate fused-minus-neural validation gain.",
            "target_information": "Covariance shift uses all four unlabeled recordings of the held-out participant.",
            "target_labels_used_by_gate": False,
            "output": "One cell-level hard choice between the frozen five-seed-median branches.",
        },
        {
            "method": "CB-SF",
            "setting": "Bidirectional cross-task",
            "training_information": "Leave-one-participant-out gate training over the other 12 participants and both directions (24 participant-direction rows). Benefit labels come from source-task OOF accuracy only.",
            "target_information": "All windows from the three unlabeled target recordings jointly determine one diagnostic vector and one weight.",
            "target_labels_used_by_gate": False,
            "output": "p_CB-SF=(1-lambda)p_NN+lambda p_fuse for every target window, with one shared lambda per participant-direction cell.",
        },
        {
            "method": "CB-SF",
            "setting": "Arithmetic nested LOSO",
            "training_information": "For each held-out participant, 14 inner participant OOF rows train the gate; the outer target participant is absent.",
            "target_information": "All windows from all four unlabeled target recordings jointly determine one diagnostic vector and one weight.",
            "target_labels_used_by_gate": False,
            "output": "One transductive batch weight shared by the four recordings.",
        },
        {
            "method": "CB-SF",
            "setting": "Stroop nested LOSO",
            "training_information": "For each held-out participant, 12 inner participant OOF rows train the gate; the outer target participant is absent.",
            "target_information": "All windows from all four unlabeled target recordings jointly determine one diagnostic vector and one weight.",
            "target_labels_used_by_gate": False,
            "output": "One transductive batch weight shared by the four recordings.",
        },
    ]


def build_cell_audit() -> tuple[list[dict], dict]:
    decisions = pd.read_csv(DASF_ROOT / "gate_decisions_final.csv")
    weights = pd.read_csv(CBSF_ROOT / "gate_weights.csv")
    cross_training = pd.read_csv(CBSF_ROOT / "gate_training_rows_cross_task.csv")
    arithmetic_training = pd.read_csv(CBSF_ROOT / "gate_training_rows_loso_arithmetic.csv")
    stroop_training = pd.read_csv(CBSF_ROOT / "gate_training_rows_loso_stroop.csv")
    targets = pd.read_csv(CBSF_ROOT / "target_diagnostics.csv")

    if len(decisions) != 54 or len(weights) != 54 or len(targets) != 54:
        raise AssertionError(
            f"门控单元应各有54行，实际DASF={len(decisions)}, CB-SF={len(weights)}, target={len(targets)}"
        )
    if len(cross_training) != 26:
        raise AssertionError(f"cross-task门控候选训练表应有26行，实际{len(cross_training)}")
    if len(arithmetic_training) != 15 * 14:
        raise AssertionError(f"Arithmetic LOSO门控训练表应有210行，实际{len(arithmetic_training)}")
    if len(stroop_training) != 13 * 12:
        raise AssertionError(f"Stroop LOSO门控训练表应有156行，实际{len(stroop_training)}")
    expected_meta = {"cross_task": 24, "arithmetic": 14, "stroop": 12}
    for row in weights.itertuples(index=False):
        key = "cross_task" if row.protocol == "cross_task" else str(row.direction)
        if int(row.meta_training_rows) != expected_meta[key]:
            raise AssertionError(f"{key}门控训练行数异常：{row.meta_training_rows}")

    rows = []
    for protocol, direction, expected_cells, expected_rows, target_recordings in [
        ("cross_task", "two directions", 26, 24, 3),
        ("loso", "arithmetic", 15, 14, 4),
        ("loso", "stroop", 13, 12, 4),
    ]:
        selected = weights[
            (weights["protocol"] == protocol)
            & ((weights["direction"] == direction) if direction != "two directions" else True)
        ]
        rows.append(
            {
                "protocol": protocol,
                "direction_or_task": direction,
                "gate_cells": int(len(selected)),
                "expected_gate_cells": expected_cells,
                "training_rows_per_target_gate": expected_rows,
                "unlabeled_target_recordings_per_gate": target_recordings,
                "minimum_weight": float(selected["weight"].min()),
                "median_weight": float(selected["weight"].median()),
                "maximum_weight": float(selected["weight"].max()),
                "ood_fallback_cells": int(selected["ood_fallback"].astype(bool).sum()),
            }
        )
    summary = {
        "dasf_gate_cells": int(len(decisions)),
        "dasf_fusion_selected_cells": int(decisions["used_fusion"].astype(bool).sum()),
        "cbsf_gate_cells": int(len(weights)),
        "cbsf_ood_fallback_cells": int(weights["ood_fallback"].astype(bool).sum()),
        "cross_task_candidate_gate_rows": int(len(cross_training)),
        "arithmetic_loso_nested_gate_rows": int(len(arithmetic_training)),
        "stroop_loso_nested_gate_rows": int(len(stroop_training)),
    }
    return rows, summary


def build_pseudocode() -> str:
    return """## Algorithm S1. Frozen core probabilities

1. In each outer evaluation cell, create recording-disjoint inner folds.
2. For each of five seeds, fit the neural, random-forest, and Extra Trees branches on each inner-training fold and predict its held-out recordings.
3. Clip each OOF class-1 probability to [1e-6, 1-1e-6] and convert it to a logit.
4. Fit the class-balanced logistic stacker only on these OOF logits, weighting every window by the inverse number of windows in its recording.
5. Select the neural epoch from inner recording-level log score, refit all branches on the complete outer-training data, and predict the untouched target data.
6. At each target window and method, take the per-class median over the five seed probabilities and renormalize the two classes.

## Algorithm S2. DASF hard selection

Input: OOF neural and fused probabilities, source covariances, unlabeled target covariances, and frozen target probabilities.

1. For each seed, calculate recording-level OOF neural accuracy A_NN and fused accuracy A_fuse.
2. Calculate g = A_fuse - A_NN.
3. Calculate d_cov = ||mean(C_target)-mean(C_source)||_F / {0.5[||mean(C_target)||_F+||mean(C_source)||_F]}.
4. Across the five seeds, separately take median(A_NN), median(A_fuse), and median(d_cov).
5. Calculate m = clip(0.02 + 0.2 d_cov, 0, 0.15).
6. If all values are finite and median(A_fuse)-median(A_NN) > m, select the fused branch; otherwise select the neural branch.
7. Return the selected frozen five-seed-ensemble probability exactly; do not recalibrate or alter it.

## Algorithm S3. CB-SF transductive soft fusion

Input: OOF gate-training rows, frozen neural and fused target probabilities, source features, and the complete unlabeled target batch.

1. For every gate-training and target cell, calculate five diagnostics: robust feature shift, branch disagreement, neural entropy, fused entropy, and neural confidence.
2. Define OOF benefit b = recording_accuracy_fuse - recording_accuracy_NN. Define catastrophe y_cat = 1[b <= -0.05].
3. For each gate feature, calculate its training median and IQR. If IQR<1e-8, use training standard deviation; if that is also <1e-8, use 1. Robust-scale training and target gate features.
4. Record the largest absolute, unclipped target robust z value. Clip model inputs to [-5,5].
5. Fit Ridge(alpha=1) to predict benefit. Clip the predicted benefit to the observed training-benefit range intersected with [-1,1].
6. If both catastrophe classes occur, fit class-balanced L2 logistic regression (C=1, solver=lbfgs, max_iter=2000); otherwise use the observed catastrophe prevalence.
7. If any target diagnostic is non-finite or max_abs_target_z>10, set lambda=0. Otherwise calculate s=b_hat-0.05 p_hat_cat and lambda=0.8 sigmoid[clip(s/0.02,-40,40)].
8. Apply the same lambda to every window in the target participant-direction batch: p_CB-SF=(1-lambda)p_NN+lambda p_fuse, then renormalize.
9. Aggregate windows to recording probabilities by the class-wise geometric mean and renormalize.

No target label is used in Steps 1-9. However, target diagnostics are batch averages. Their values, and therefore lambda, can change when the number or latent class mixture of target recordings changes.
"""


def build_supplement(
    feature_rows: list[dict],
    hyperparameter_rows: list[dict],
    dataflow_rows: list[dict],
    cell_rows: list[dict],
) -> str:
    feature_lines = ["| Order | Diagnostic | Exact definition |", "|---:|---|---|"]
    for row in feature_rows:
        feature_lines.append(
            f"| {row['feature_order']} | {row['feature']} | {row['exact_formula']} |"
        )
    parameter_lines = ["| Component | Parameter | Value | Status |", "|---|---|---:|---|"]
    highlighted = hyperparameter_rows[:9]
    for row in highlighted:
        parameter_lines.append(
            f"| {row['component']} | {row['parameter']} | {row['value']} | {row['status']} |"
        )
    cell_lines = [
        "| Setting | Gate cells | Training rows per target gate | Unlabeled target recordings |",
        "|---|---:|---:|---:|",
    ]
    for row in cell_rows:
        cell_lines.append(
            f"| {row['protocol']}: {row['direction_or_task']} | {row['gate_cells']} | "
            f"{row['training_rows_per_target_gate']} | {row['unlabeled_target_recordings_per_gate']} |"
        )
    flow_lines = []
    for row in dataflow_rows:
        flow_lines.append(
            f"- **{row['method']}, {row['setting']}:** {row['training_information']} "
            f"Target use: {row['target_information']}"
        )
    return f"""# Exact DASF and CB-SF algorithms

## Scope and analysis status

DASF and CB-SF were evaluated as secondary exploratory gates in the corrected Route A v3 methodological audit. Their constants were frozen before the v3 rerun but were inherited from development conducted after inspection of earlier analyses. They were not preregistered or externally validated. Neither method optimizes CVaR, imposes a formal risk constraint, or provides a safety, no-regret, or online-deployment guarantee.

## Core fused branch

The fused branch is a class-balanced logistic stacker of neural, random-forest and Extra Trees class-1 logits. The stacker is trained only from inner out-of-fold predictions. Probabilities are clipped to [1e-6, 1-1e-6] before the logit transform. Each window receives inverse-recording-size weight, normalized to mean one. Five seeds are combined by a per-class median at each window and then renormalized; they are ensemble members, not inferential replicates.

## DASF covariance shift

Let mean(C_S) and mean(C_T) be the window-mean regularized covariance matrices in the source reference and unlabeled target batches. DASF uses

`d_cov = ||mean(C_T)-mean(C_S)||_F / (0.5[||mean(C_T)||_F+||mean(C_S)||_F])`.

If the denominator is non-finite or no larger than 1e-12, the shift is non-finite and DASF falls back to the neural branch. The effective hard-selection margin is `clip(0.02 + 0.2*d_cov, 0, 0.15)`.

## CB-SF gate diagnostics

{chr(10).join(feature_lines)}

CB-SF feature shift is not the DASF covariance shift. It is the median capped standardized difference across the 272 flat features.

## Main constants

{chr(10).join(parameter_lines)}

The complete estimator parameter list, including scikit-learn defaults, is provided in the accompanying CSV and machine-readable JSON files.

## Nested data flow

{chr(10).join(flow_lines)}

## Target-batch composition

{chr(10).join(cell_lines)}

In cross-task transfer, natural recordings were removed because the Arithmetic and Stroop natural files are exact same-participant duplicates. The gate therefore uses three target recordings (low, medium, and high). LOSO uses four target recordings. All recordings in a target cell are accessed jointly without labels. Consequently, this is transductive batch inference. Changing target-batch size or latent class composition can change feature means, probability summaries, and the gate weight. No invariance to such changes was established.

{build_pseudocode()}
"""


def build_chinese_explanation(cell_summary: dict) -> str:
    return f"""# DASF与CB-SF完整流程中文解读

## 两种方法不是同一个门控

DASF是硬门控：最后只能完整选择always-neural或always-fuse中的一个。它先看内层OOF中融合比神经分支高多少，再根据源域与目标域协方差的差异提高选择门槛。最终公式是：`融合收益 > clip(0.02 + 0.2 x 协方差偏移, 0, 0.15)`才使用融合。

CB-SF是软门控：它给出0到0.8之间的一个权重lambda，然后计算`(1-lambda) x neural + lambda x fuse`。它使用五个不需要目标标签的诊断量，包括272维特征偏移、两个分支的分歧、两个分支的熵和神经分支置信度。

## 为什么不能再说在线预测

Cross-task中，一个lambda同时查看同一参与者的low、medium、high三条目标recording；LOSO中同时查看四条目标recording。只是不读取标签，并不等于单条recording在线推断。目标批次的类别比例或recording数量变化时，这些平均诊断量也会变化。

## 0.8、0.05和0.02是什么意思

- 0.8：融合分支的最大权重，至少保留20%的神经分支。它是启发式上限，不是安全保证。
- 第一个0.05：把OOF融合收益低于或等于-5个百分点定义为“catastrophe”。这是探索性事件定义。
- 第二个0.05：在门控分数中惩罚预测灾难概率的系数。
- 0.02：把门控分数变成软权重时的temperature；DASF还单独使用0.02作为基础硬门槛。

这些常数是在v3重跑前冻结的，但它们来自看过更早实验之后的开发过程，因此只能标为post-hoc exploratory，不能写成预注册参数或形式化风险约束。

## 实际审计数量

- DASF门控单元：{cell_summary['dasf_gate_cells']}，选择融合的单元：{cell_summary['dasf_fusion_selected_cells']}。
- CB-SF门控单元：{cell_summary['cbsf_gate_cells']}，OOD强制退回神经分支的单元：{cell_summary['cbsf_ood_fallback_cells']}。
- Cross-task每个目标gate用其他参与者的24个participant-direction训练行。
- Arithmetic LOSO每个gate用14个内层参与者行；Stroop LOSO用12个。

论文最终应把DASF和CB-SF作为被审计的探索性融合方法，而不是已经证明准确率优势或风险保证的主算法。
"""


def main() -> None:
    required = [
        POLICY_PATH,
        DASF_RUNNER,
        CBSF_RUNNER,
        RISK_GATE_SOURCE,
        MODEL_SOURCE,
        AGGREGATION_SOURCE,
        PROBABILITY_SOURCE,
        DATA_SOURCE,
        DASF_ROOT / "gate_decisions_final.csv",
        CBSF_ROOT / "gate_weights.csv",
        CBSF_ROOT / "gate_training_rows_cross_task.csv",
        CBSF_ROOT / "gate_training_rows_loso_arithmetic.csv",
        CBSF_ROOT / "gate_training_rows_loso_stroop.csv",
        CBSF_ROOT / "target_diagnostics.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"缺少算法审计输入：{path}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    gate = policy["models"]["risk_gate"]
    expected_gate = {
        "catastrophe_threshold": 0.05,
        "benefit_alpha": 1.0,
        "catastrophe_C": 1.0,
        "risk_penalty": 0.05,
        "temperature": 0.02,
        "max_weight": 0.8,
    }
    for key, value in expected_gate.items():
        if not np.isclose(float(gate[key]), value, rtol=0.0, atol=1e-12):
            raise AssertionError(f"风险门控参数不匹配：{key}={gate[key]}")

    print("[1/4] 核对DASF、CB-SF、概率聚合和OOF stacker源代码。")
    verify_source_implementation()
    feature_rows = build_gate_feature_rows()
    hyperparameter_rows = build_hyperparameter_rows(policy)
    dataflow_rows = build_dataflow_rows()
    print("[2/4] 核对54个实际gate cell和嵌套训练行数。")
    cell_rows, cell_summary = build_cell_audit()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(FEATURE_CSV, feature_rows)
    write_csv(HYPERPARAMETER_CSV, hyperparameter_rows)
    write_csv(DATAFLOW_CSV, dataflow_rows)
    write_csv(CELL_AUDIT_CSV, cell_rows)

    machine = {
        "analysis_version": "2026-08-27-route-a-v3-algorithm-reproducibility-audit-r1",
        "status": "source_and_output_verified",
        "claim_scope": "secondary post-hoc exploratory gates; no formal risk guarantee",
        "gate_features": feature_rows,
        "hyperparameters": hyperparameter_rows,
        "data_flow": dataflow_rows,
        "cell_audit": cell_rows,
        "cell_summary": cell_summary,
        "formulas": {
            "dasf_covariance_shift": "||mean(C_T)-mean(C_S)||_F / [0.5*(||mean(C_T)||_F+||mean(C_S)||_F)]",
            "dasf_margin": "clip(0.02+0.2*d_cov,0,0.15)",
            "dasf_choice": "always-fuse iff OOF gain > margin; otherwise always-neural",
            "cbsf_catastrophe": "1[OOF benefit <= -0.05]",
            "cbsf_score": "predicted_benefit - 0.05*predicted_catastrophe_probability",
            "cbsf_weight": "0.8*sigmoid(clip(score/0.02,-40,40)); zero under OOD fallback",
            "cbsf_blend": "(1-weight)*p_neural + weight*p_fused, followed by row normalization",
        },
        "target_batch": {
            "cross_task_recordings": 3,
            "cross_task_levels": ["low", "medium", "high"],
            "loso_recordings": 4,
            "inference_type": "transductive batch",
            "single_recording_online": False,
            "class_mix_invariance_tested": False,
        },
    }
    MACHINE_JSON.write_text(
        json.dumps(machine, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SUPPLEMENT_MD.write_text(
        build_supplement(feature_rows, hyperparameter_rows, dataflow_rows, cell_rows),
        encoding="utf-8",
    )
    CHINESE_MD.write_text(build_chinese_explanation(cell_summary), encoding="utf-8")

    print("[3/4] 生成英文补充材料、完整伪代码、超参数表和中文解读。")
    source_files = {
        "policy": POLICY_PATH,
        "dasf_runner": DASF_RUNNER,
        "cbsf_runner": CBSF_RUNNER,
        "risk_gate": RISK_GATE_SOURCE,
        "models": MODEL_SOURCE,
        "aggregation": AGGREGATION_SOURCE,
        "probability": PROBABILITY_SOURCE,
        "data": DATA_SOURCE,
        "audit_runner": Path(__file__).resolve(),
    }
    output_files = [
        FEATURE_CSV,
        HYPERPARAMETER_CSV,
        DATAFLOW_CSV,
        CELL_AUDIT_CSV,
        MACHINE_JSON,
        SUPPLEMENT_MD,
        CHINESE_MD,
    ]
    manifest = {
        "analysis_version": "2026-08-27-route-a-v3-algorithm-reproducibility-audit-r1",
        "status": "passed",
        "source_verified": True,
        "actual_output_verified": True,
        "claim_scope": "secondary post-hoc exploratory gates; no formal risk guarantee",
        "cell_summary": cell_summary,
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256(path) for path in output_files},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "prohibited_claims": [
            "formal risk control",
            "risk guarantee",
            "no-regret guarantee",
            "single-recording online prediction",
            "class-mixture invariant target-batch gate",
        ],
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[4/4] Route A v3 DASF与CB-SF算法可复现性审计完成。")
    print(f"DASF gate cells：{cell_summary['dasf_gate_cells']}；选择融合：{cell_summary['dasf_fusion_selected_cells']}")
    print(f"CB-SF gate cells：{cell_summary['cbsf_gate_cells']}；OOD fallback：{cell_summary['cbsf_ood_fallback_cells']}")
    print("Cross-task每个目标gate训练行：24；Arithmetic LOSO：14；Stroop LOSO：12")
    print("推断类型：transductive target-batch，不是single-recording online prediction")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
