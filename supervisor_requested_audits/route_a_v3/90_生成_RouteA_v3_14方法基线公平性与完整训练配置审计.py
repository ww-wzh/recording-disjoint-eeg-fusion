"""编号90：生成 Route A v3 十四种方法的公平性与完整训练配置审计。

本文件不重新训练模型，也不改变81号冻结预测。它核对64号冻结协议、81号
冻结预测、模型清单和实际源代码，生成可直接用于论文补充材料的训练配置、
方法角色、种子策略和比较公平性说明。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：18_baseline_fairness_audit
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
FROZEN_ROOT = HERE / "10_all_methods_frozen"
FROZEN_PREDICTION = FROZEN_ROOT / "81_冻结_RouteA_v3全部14方法_recording级预测.csv"
FROZEN_MANIFEST = FROZEN_ROOT / "81_冻结_RouteA_v3全部方法_manifest.json"
CORE_MANIFEST = HERE / "01_core" / "71_核心输出完整性_manifest.json"
CORE_RUNTIME = HERE / "01_core" / "cross_task" / "run_metadata.json"
BASELINE_MANIFEST = HERE / "05_baselines" / "75_baselines_manifest.json"
CROSS_STACK_MANIFEST = HERE / "06_cross_task_stacker" / "76_cross_task_stacker_manifest.json"
RIEMANN_MANIFEST = HERE / "07_riemannian" / "77_Riemannian_manifest.json"
EEGNET_MANIFEST = HERE / "08_eegnet" / "eegnet_v3_manifest.json"
CONFORMER_MANIFEST = HERE / "09_eeg_conformer" / "eeg_conformer_v3_manifest.json"

MODEL_SOURCE = REVISION_ROOT / "revision_pipeline" / "models.py"
DEEP_SOURCE = REVISION_ROOT / "route_a" / "route_a_lib" / "deep.py"
DEEP_RUNNER = HERE / "78_公共_RouteA_v3深度基线工具.py"
RIEMANN_RUNNER = HERE / "77_运行_RouteA_v3_Riemannian基线.py"
FIXED_RUNNER = HERE / "75_运行_RouteA_v3固定权重与recording级stacking基线.py"
CROSS_STACK_RUNNER = HERE / "76_运行_RouteA_v3_CrossTask_sourceOOF与recording级stacking.py"

OUTPUT_ROOT = HERE / "18_baseline_fairness_audit"
ROLE_CSV = OUTPUT_ROOT / "90_14方法角色种子策略与公平性.csv"
CONFIG_CSV = OUTPUT_ROOT / "90_全部模型与融合方法完整配置.csv"
SOFTWARE_CSV = OUTPUT_ROOT / "90_软件硬件与确定性设置.csv"
MACHINE_JSON = OUTPUT_ROOT / "90_基线公平性机器可读定义.json"
SUPPLEMENT_MD = OUTPUT_ROOT / "90_Supplementary_Training_Protocols_and_Baseline_Fairness.md"
CHINESE_MD = OUTPUT_ROOT / "90_中文解读_14方法怎样公平比较.md"
MANIFEST_JSON = OUTPUT_ROOT / "90_基线公平性与配置审计_manifest.json"

SEEDS = [1335, 1388, 1441, 1494, 1547]
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def verify_inputs() -> tuple[dict, dict, dict, dict, dict]:
    required = [
        POLICY_PATH,
        FROZEN_PREDICTION,
        FROZEN_MANIFEST,
        CORE_MANIFEST,
        CORE_RUNTIME,
        BASELINE_MANIFEST,
        CROSS_STACK_MANIFEST,
        RIEMANN_MANIFEST,
        EEGNET_MANIFEST,
        CONFORMER_MANIFEST,
        MODEL_SOURCE,
        DEEP_SOURCE,
        DEEP_RUNNER,
        RIEMANN_RUNNER,
        FIXED_RUNNER,
        CROSS_STACK_RUNNER,
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少90号审计输入：{missing}")

    policy = load_json(POLICY_PATH)
    frozen = load_json(FROZEN_MANIFEST)
    eegnet = load_json(EEGNET_MANIFEST)
    conformer = load_json(CONFORMER_MANIFEST)
    riemann = load_json(RIEMANN_MANIFEST)

    if policy["protocol_version"] != "2026-08-22-route-a-v3":
        raise AssertionError("64号协议版本不是Route A v3")
    if sha256(POLICY_PATH) != frozen["protocol_sha256"]:
        raise AssertionError("64号协议SHA-256与81号冻结清单不一致")
    if sha256(FROZEN_PREDICTION) != frozen["prediction_sha256"]:
        raise AssertionError("81号冻结预测SHA-256不一致")
    if frozen["prediction_sha256"] != "1c0fd57fda8042293f23303499638b0f76d17c38c6a0e346b86613b5aa03aea0":
        raise AssertionError("读取的不是已确认的81号主冻结预测")

    predictions = pd.read_csv(FROZEN_PREDICTION, usecols=["method", "recording_id"])
    actual_methods = sorted(predictions["method"].astype(str).unique().tolist())
    if actual_methods != sorted(EXPECTED_METHODS):
        raise AssertionError(f"81号方法集合不正确：{actual_methods}")
    counts = predictions.groupby("method")["recording_id"].size()
    if len(predictions) != 2660 or not (counts == 190).all():
        raise AssertionError("81号冻结预测不是14种方法各190条recording")

    for name, manifest in {
        "EEGNet": eegnet,
        "EEG-Conformer": conformer,
        "Riemannian": riemann,
        "core": load_json(CORE_MANIFEST),
        "fixed baseline": load_json(BASELINE_MANIFEST),
        "cross-task stacker": load_json(CROSS_STACK_MANIFEST),
    }.items():
        if manifest.get("status") != "passed":
            raise AssertionError(f"{name}清单没有通过校验")

    if eegnet["training"] != conformer["training"]:
        raise AssertionError("EEGNet与EEG-Conformer没有使用相同训练配置")
    if eegnet["recording_rows"] != 190 or conformer["recording_rows"] != 190:
        raise AssertionError("深度基线没有覆盖全部190条recording")
    if riemann["recording_rows"] != 380:
        raise AssertionError("两个Riemannian方法没有各覆盖190条recording")

    source_requirements = {
        MODEL_SOURCE: [
            "class FeatureMLP",
            "class_weight=\"balanced\"",
            "sample_weight=stack_weight",
            "nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
        ],
        DEEP_SOURCE: [
            "class EEGNet",
            "class EEGConformer",
            "nn.CrossEntropyLoss(",
            "nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
        ],
        DEEP_RUNNER: [
            '"batch_size": 32',
            '"max_epochs": 80',
            '"patience": 12',
            '"selection_metric": "recording-level mean log probability"',
        ],
        RIEMANN_RUNNER: [
            '"tangent_metric": "riemann"',
            '"logistic_C": 1.0',
            '"logistic_max_iter": 4000',
            '"mdm_metric": "riemann"',
        ],
        FIXED_RUNNER: [
            'model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000',
            '_fixed_rows(window, "equal_blend_050", 0.50)',
        ],
        CROSS_STACK_RUNNER: [
            'model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000',
            'target_labels_used_for_fit": False',
        ],
    }
    for path, fragments in source_requirements.items():
        text = path.read_text(encoding="utf-8")
        absent = [fragment for fragment in fragments if fragment not in text]
        if absent:
            raise AssertionError(f"{path.name}缺少冻结配置片段：{absent}")
    return policy, frozen, eegnet, conformer, riemann


def count_parameters(policy: dict, eegnet_manifest: dict, conformer_manifest: dict) -> dict[str, int]:
    sys.path.insert(0, str(REVISION_ROOT))
    sys.path.insert(0, str(REVISION_ROOT / "route_a"))
    from revision_pipeline.models import FeatureMLP
    from route_a_lib.deep import build_model, parameter_count

    feature_model = FeatureMLP(
        input_dim=272,
        width=int(policy["models"]["mlp"]["width"]),
        blocks=int(policy["models"]["mlp"]["blocks"]),
        dropout=float(policy["models"]["mlp"]["dropout"]),
    )
    eegnet_model = build_model("eegnet", 8, 2125, eegnet_manifest["architecture"])
    conformer_model = build_model("eeg_conformer", 8, 2125, conformer_manifest["architecture"])
    return {
        "feature_mlp": parameter_count(feature_model),
        "eegnet": parameter_count(eegnet_model),
        "eeg_conformer": parameter_count(conformer_model),
    }


def role_rows() -> list[dict]:
    common = {
        "outer_test_partition": "Identical Route A v3 participant/direction cells and recordings",
        "evaluation": "Window probabilities aggregated to recording; participant is statistical unit",
        "v3_hyperparameter_search": "None; fixed before the v3 rerun",
    }
    rows = [
        ("always_nn", "Feature MLP", "Primary comparator", "272 engineered features; 0.5-55 Hz", 5,
         "Median window probability across five independent stochastic fits", "Same nested inner folds", "Directly protocol-aligned"),
        ("always_fuse", "Window-level heterogeneous stack", "Learned fusion comparator", "OOF logits from Feature MLP, RF and Extra Trees", 5,
         "One OOF logistic stacker per seed, followed by five-seed median", "Same nested inner folds; OOF only", "Protocol-aligned derived model"),
        ("rf", "Random forest", "Classical feature baseline", "272 engineered features; 0.5-55 Hz", 5,
         "Median window probability across five seeded fits", "Same nested inner folds for OOF fusion evidence", "Directly protocol-aligned"),
        ("extra_trees", "Extra Trees", "Classical feature baseline", "272 engineered features; 0.5-55 Hz", 5,
         "Median window probability across five seeded fits", "Same nested inner folds for OOF fusion evidence", "Directly protocol-aligned"),
        ("fixed_blend_010", "Fixed 10% blend", "Secondary fixed-weight baseline", "Frozen neural and heterogeneous-stack probabilities", 0,
         "Deterministic 0.90 neural + 0.10 fused probability", "No fitted parameters", "Derived from the same frozen branches"),
        ("fixed_blend_025", "Fixed 25% blend", "Secondary fixed-weight baseline", "Frozen neural and heterogeneous-stack probabilities", 0,
         "Deterministic 0.75 neural + 0.25 fused probability", "No fitted parameters", "Derived from the same frozen branches"),
        ("equal_blend_050", "Equal 50% blend", "Secondary fixed-weight baseline", "Frozen neural and heterogeneous-stack probabilities", 0,
         "Deterministic 0.50 neural + 0.50 fused probability", "No fitted parameters", "Derived from the same frozen branches"),
        ("stack_recording", "Recording-level OOF stacker", "Secondary learned meta-baseline", "Recording-level OOF branch logits", 1,
         "One deterministic logistic meta-model per outer cell", "Source-recording OOF for cross-task; nested participant OOF for LOSO", "Protocol-aligned but trained at recording level"),
        ("dasf_clean", "DASF hard gate", "Exploratory proposed selector", "Frozen neural/fused probabilities plus label-free shift", 0,
         "One batch-level hard decision after five-seed aggregation", "OOF validation evidence only", "Exploratory derived decision rule"),
        ("cbsf", "CB-SF soft gate", "Post-hoc exploratory proposed gate", "Five target-batch diagnostics from frozen branches and features", 0,
         "One transductive batch weight after five-seed aggregation", "Cross-fitted gate training; target labels excluded", "Exploratory derived decision rule"),
        ("riemann_ts_logreg", "Riemannian tangent-space logistic regression", "Conventional geometric baseline", "8-channel covariance; 0.5-55 Hz", 1,
         "One deterministic fit; copied to seed slots only for file compatibility", "No inner model selection", "Same outer test set; deterministic and differently represented"),
        ("riemann_mdm", "Riemannian MDM", "Conventional geometric baseline", "8-channel covariance; 0.5-55 Hz", 1,
         "One deterministic fit; copied to seed slots only for file compatibility", "No inner model selection", "Same outer test set; deterministic and differently represented"),
        ("eegnet", "EEGNet", "Raw-EEG deep baseline", "8 x 2125 raw window; 0.5-45 Hz", 5,
         "Median window probability across five independent stochastic fits", "Same recording-/participant-disjoint nested folds", "Same split/ensemble budget; representation-specific front end"),
        ("eeg_conformer", "EEG-Conformer", "Raw-EEG deep baseline", "8 x 2125 raw window; 0.5-45 Hz", 5,
         "Median window probability across five independent stochastic fits", "Same recording-/participant-disjoint nested folds", "Same split/ensemble budget; representation-specific front end"),
    ]
    output = []
    for method_id, display, role, representation, fits, aggregation, selection, fairness in rows:
        output.append(
            {
                "internal_method_id": method_id,
                "manuscript_display_name": display,
                "method_role": role,
                "input_representation": representation,
                "independent_stochastic_fits_per_outer_cell": fits,
                "seed_or_fit_aggregation": aggregation,
                "model_selection": selection,
                "outer_test_partition": common["outer_test_partition"],
                "evaluation_and_statistics": common["evaluation"],
                "v3_hyperparameter_search": common["v3_hyperparameter_search"],
                "fairness_interpretation": fairness,
            }
        )
    return output


def add_config(rows: list[dict], component: str, parameter: str, value, scope: str, evidence: str) -> None:
    rows.append(
        {
            "component": component,
            "parameter": parameter,
            "value": value,
            "scope": scope,
            "evidence_source": evidence,
        }
    )


def config_rows(policy: dict, eegnet: dict, conformer: dict, riemann: dict, counts: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    mlp = policy["models"]["mlp"]
    for parameter, value in {
        "input dimension": 272,
        "trainable parameters": counts["feature_mlp"],
        "hidden width": mlp["width"],
        "residual blocks": mlp["blocks"],
        "dropout": mlp["dropout"],
        "optimizer": "AdamW",
        "learning rate": mlp["learning_rate"],
        "weight decay": mlp["weight_decay"],
        "batch size": mlp["batch_size"],
        "maximum epochs": mlp["max_epochs"],
        "early-stopping patience": mlp["patience"],
        "loss": "inverse-frequency class-weighted cross-entropy",
        "gradient clipping": "L2 norm 1.0",
        "feature scaling": "StandardScaler fitted only on the corresponding training fold",
        "epoch selection": "recording-level mean true-class log probability; final epoch is median inner-fold best epoch",
    }.items():
        add_config(rows, "Feature MLP", parameter, value, "all v3 settings", "64 protocol and models.py")

    for name, display in [("rf", "Random forest"), ("extra_trees", "Extra Trees")]:
        for parameter, value in policy["models"][name].items():
            add_config(rows, display, parameter, value, "all v3 settings", "64 protocol")
        add_config(rows, display, "class weight", "balanced", "all v3 settings", "models.py")
        add_config(rows, display, "outer-cell seed offset", "+37 for RF; +74 for Extra Trees", "each ensemble seed", "models.py")

    stacker = policy["models"]["stacker"]
    for parameter, value in {
        "inputs": "class-1 logits from Feature MLP, RF and Extra Trees",
        "C": stacker["C"],
        "class weight": "balanced",
        "maximum iterations": stacker["max_iter"],
        "training predictions": stacker["training_predictions"],
        "sample weight": stacker["sample_weight"],
        "probability clipping before logit": "[1e-6, 1-1e-6]",
    }.items():
        add_config(rows, "Window-level heterogeneous stack", parameter, value, "each inner/outer cell", "64 protocol and models.py")

    deep_training = eegnet["training"]
    for display, manifest, count_key in [
        ("EEGNet", eegnet, "eegnet"),
        ("EEG-Conformer", conformer, "eeg_conformer"),
    ]:
        for parameter, value in manifest["architecture"].items():
            add_config(rows, display, parameter, value, "all v3 settings", f"{display} v3 manifest")
        add_config(rows, display, "input shape", "8 x 2125", "every window", "78 runner")
        add_config(rows, display, "trainable parameters", counts[count_key], "fixed architecture", "deep.py instantiated audit")
        add_config(rows, display, "band-pass", "0.5-45 Hz, fourth-order zero-phase Butterworth", "each recording before windowing", "78 runner data loader")
        add_config(rows, display, "normalization", "per-window per-channel z-score", "each completed 8.5-s window", "78 runner data loader")
        for parameter, value in deep_training.items():
            add_config(rows, display, parameter.replace("_", " "), value, "all v3 settings", f"{display} v3 manifest")
        add_config(rows, display, "loss", "inverse-frequency class-weighted cross-entropy", "each training fold", "deep.py")
        add_config(rows, display, "optimizer", "AdamW", "each fit", "deep.py")
        add_config(rows, display, "gradient clipping", "L2 norm 1.0", "each update", "deep.py")
        add_config(rows, display, "final epoch", "median inner-fold selected epoch", "each outer cell and seed", "78 runner")

    for component in ["Riemannian tangent-space logistic regression", "Riemannian MDM"]:
        add_config(rows, component, "input", "8 x 8 regularized covariance per window", "all v3 settings", "feature loader and 77 runner")
        add_config(rows, component, "reference fitting", "outer-training windows only", "each outer cell", "77 runner")
    for parameter, value in riemann["config"].items():
        component = "Riemannian tangent-space logistic regression" if parameter.startswith(("tangent", "logistic")) else "Riemannian MDM"
        add_config(rows, component, parameter.replace("_", " "), value, "all v3 settings", "77 manifest")
    add_config(rows, "Riemannian tangent-space logistic regression", "class weight", "balanced", "all v3 settings", "run_riemannian_baselines.py")
    add_config(rows, "Riemannian tangent-space logistic regression", "tangent scaling", "StandardScaler fitted on outer training", "each outer cell", "run_riemannian_baselines.py")

    for component, weight in [("Fixed 10% blend", 0.10), ("Fixed 25% blend", 0.25), ("Equal 50% blend", 0.50)]:
        add_config(rows, component, "formula", f"(1-{weight:.2f})*P_neural + {weight:.2f}*P_heterogeneous", "window probability", "75 runner")
        add_config(rows, component, "fitted parameters", 0, "all settings", "75 runner")

    for parameter, value in {
        "inputs": "recording-level class-1 branch logits",
        "C": 1.0,
        "class weight": "balanced",
        "maximum iterations": 4000,
        "probability clipping before logit": "[1e-6, 1-1e-6]",
        "cross-task fit data": "four source-task recording-level OOF rows in the target participant cell",
        "LOSO fit data": "inner participant-level OOF recording predictions from outer-training participants",
        "target labels used": False,
    }.items():
        add_config(rows, "Recording-level OOF stacker", parameter, value, "protocol-specific", "75 and 76 runners")
    return rows


def software_rows() -> list[dict]:
    gpu = "CPU only"
    cudnn = "not available"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cudnn = str(torch.backends.cudnn.version())
    values = [
        ("Operating system", platform.platform()),
        ("Python", platform.python_version()),
        ("NumPy", np.__version__),
        ("pandas", pd.__version__),
        ("scikit-learn", sklearn.__version__),
        ("SciPy", package_version("scipy")),
        ("PyTorch", torch.__version__),
        ("PyTorch CUDA runtime", str(torch.version.cuda)),
        ("cuDNN", cudnn),
        ("GPU", gpu),
        ("pyRiemann", package_version("pyriemann")),
        ("Random seeds", ", ".join(str(seed) for seed in SEEDS)),
        ("PyTorch deterministic setting", "cudnn.deterministic=True; cudnn.benchmark=False"),
        ("Seed statistical role", "ensemble members only; never statistical replicates"),
    ]
    return [{"item": key, "value": value} for key, value in values]


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(display for _, display in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [str(row[key]).replace("|", "\\|").replace("\n", " ") for key, _ in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def build_supplement(role: list[dict], config: list[dict], software: list[dict], counts: dict[str, int]) -> str:
    role_table = markdown_table(
        role,
        [
            ("manuscript_display_name", "Method"),
            ("method_role", "Role"),
            ("input_representation", "Input"),
            ("independent_stochastic_fits_per_outer_cell", "Independent fits"),
            ("model_selection", "Selection"),
            ("fairness_interpretation", "Comparison status"),
        ],
    )
    software_table = markdown_table(software, [("item", "Item"), ("value", "Value")])
    config_groups = []
    frame = pd.DataFrame(config)
    for component, group in frame.groupby("component", sort=False):
        config_groups.append(f"### {component}\n\n" + markdown_table(group.to_dict("records"), [("parameter", "Parameter"), ("value", "Value"), ("scope", "Scope")]))
    return f"""# Exact training protocols and baseline-comparison audit

## Common evaluation rules

All 14 methods were evaluated on the same frozen Route A v3 outer test recordings. Cross-task analysis used 13 participants in both transfer directions, Arithmetic LOSO used 15 participants, and Stroop LOSO used 13 participants. No test label was used for fitting, early stopping, stacking, or gate diagnostics. Window probabilities were aggregated by a geometric mean within recording, and participants, rather than windows or random seeds, were the statistical units. The five fixed seeds ({', '.join(str(seed) for seed in SEEDS)}) were ensemble members and were never treated as repeated observations.

The protocol was aligned with respect to outer partitions, disjoint inner partitions where model selection was required, target recordings, label mapping, probability aggregation, and statistical analysis. It was not computationally budget-matched: deterministic Riemannian methods were fitted once, fixed blends had no fitted parameters, and the raw-signal deep models used a representation-specific 0.5-45 Hz front end whereas engineered-feature and covariance methods used 0.5-55 Hz. Accordingly, the comparison is described as protocol-aligned, not as an equal-compute benchmark.

## Method roles and fit policies

{role_table}

## Exact configurations

{chr(10).join(config_groups)}

## Architecture sizes

The fixed implementations contained {counts['feature_mlp']:,} trainable parameters for the 272-input Feature MLP, {counts['eegnet']:,} for EEGNet, and {counts['eeg_conformer']:,} for EEG-Conformer. No architecture or hyperparameter sweep was conducted during the v3 rerun.

## Reproducibility environment

{software_table}

## Interpretation of fairness

The Feature MLP, RF, Extra Trees, EEGNet, and EEG-Conformer used five genuine seeded fits and identical outer/inner split identities. The Riemannian models are deterministic; their probabilities were copied into five seed slots solely to preserve a common file schema, so this does not constitute a five-model ensemble. Fixed blends, DASF, and CB-SF are deterministic transformations of frozen branch predictions. The recording-level stacker is a separately fitted OOF meta-baseline. These distinctions must accompany performance tables and prevent seed slots from being counted as independent evidence.
"""


def build_chinese(role: list[dict], counts: dict[str, int]) -> str:
    return f"""# Route A v3 十四种方法公平性审计中文解读

## 这一步回答审稿人的什么问题

审稿人担心不同基线使用了不同的数据划分、调参次数或随机种子，导致比较不公平。90号核对后的准确结论是：14种方法使用同一批外层测试recording、同一标签定义、同一recording聚合方式和同一参与者级统计，但它们不是完全相同的计算模型，不能写成“计算预算完全一致”。

## 哪些地方是公平对齐的

1. 所有方法最终都覆盖81号冻结文件中的190条recording。
2. Cross-task、Arithmetic LOSO和Stroop LOSO的外层测试对象完全一致。
3. 需要选模型的随机模型使用recording-disjoint或participant-disjoint内层划分，测试标签不参与训练。
4. Feature MLP、RF、Extra Trees、EEGNet和EEG-Conformer都使用5个固定seed，并先在窗口层取概率中位数。
5. 所有方法最终都在recording层评价，并在participant层做统计；seed不能当成样本量。

## 哪些地方不能假装完全一样

1. Riemannian方法本身是确定性的，每个外层单元只训练1次。文件中复制成5个seed槽只是统一数据格式，不是5次独立实验。
2. 固定权重融合不训练模型；DASF和CB-SF是在冻结分支上做决策，也不是重新训练5个基础模型。
3. EEGNet和EEG-Conformer读取0.5-45 Hz原始窗口；272维特征和协方差分支读取0.5-55 Hz。两者数据划分相同，但输入表示不同。
4. v3阶段没有给任何方法做额外超参数搜索。因此可以写“protocol-aligned comparison”，不能写“identical tuning budget”或“equal-compute comparison”。

## 模型规模

- Feature MLP：{counts['feature_mlp']:,}个可训练参数。
- EEGNet：{counts['eegnet']:,}个可训练参数。
- EEG-Conformer：{counts['eeg_conformer']:,}个可训练参数。

## 论文应怎样写

正文需要公开优化器、学习率、batch size、最大epoch、早停规则、输入尺寸、模型结构、5个seed的用途以及Riemannian只拟合1次的事实。结果只能表述为“在统一外层测试协议下比较”，不能把不同表示方法说成完全相同的训练方案，也不能把五个seed当成统计重复。

本审计不会改变81号结果。它解决的是实验说明和比较公平性的透明度，而不是提高准确率。
"""


def main() -> None:
    print("[1/4] 核对64号协议、81号冻结预测和各模型清单。")
    policy, frozen, eegnet, conformer, riemann = verify_inputs()
    print("[2/4] 从实际模型类计算参数量并整理14种方法配置。")
    counts = count_parameters(policy, eegnet, conformer)
    roles = role_rows()
    configs = config_rows(policy, eegnet, conformer, riemann, counts)
    software = software_rows()

    if [row["internal_method_id"] for row in roles] != EXPECTED_METHODS:
        raise AssertionError("90号方法角色表与81号冻结方法顺序不一致")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(roles).to_csv(ROLE_CSV, index=False, lineterminator="\n")
    pd.DataFrame(configs).to_csv(CONFIG_CSV, index=False, lineterminator="\n")
    pd.DataFrame(software).to_csv(SOFTWARE_CSV, index=False, lineterminator="\n")

    machine = {
        "analysis_version": "2026-08-28-route-a-v3-baseline-fairness-r1",
        "status": "source_verified",
        "protocol_version": policy["protocol_version"],
        "frozen_prediction_sha256": frozen["prediction_sha256"],
        "method_count": len(roles),
        "recordings_per_method": 190,
        "seeds": SEEDS,
        "seed_is_statistical_unit": False,
        "parameter_counts": counts,
        "comparison_scope": {
            "aligned": [
                "outer test cells and recordings",
                "disjoint split identities when inner selection is required",
                "label mapping",
                "recording aggregation",
                "participant-level statistics",
            ],
            "not_identical": [
                "input representation and representation-specific band-pass",
                "deterministic versus stochastic fit count",
                "computational cost",
                "presence or absence of fitted hyperparameters",
            ],
            "approved_wording": "protocol-aligned comparison; not an equal-compute benchmark",
        },
        "methods": roles,
        "configurations": configs,
        "software": software,
    }
    MACHINE_JSON.write_text(json.dumps(machine, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SUPPLEMENT_MD.write_text(build_supplement(roles, configs, software, counts), encoding="utf-8")
    CHINESE_MD.write_text(build_chinese(roles, counts), encoding="utf-8")

    print("[3/4] 生成英文补充材料、中文解读和机器可读配置。")
    source_files = {
        "protocol": POLICY_PATH,
        "frozen_prediction": FROZEN_PREDICTION,
        "model_source": MODEL_SOURCE,
        "deep_source": DEEP_SOURCE,
        "deep_runner": DEEP_RUNNER,
        "riemann_runner": RIEMANN_RUNNER,
        "fixed_runner": FIXED_RUNNER,
        "cross_task_stacker_runner": CROSS_STACK_RUNNER,
    }
    outputs = [ROLE_CSV, CONFIG_CSV, SOFTWARE_CSV, MACHINE_JSON, SUPPLEMENT_MD, CHINESE_MD]
    manifest = {
        "analysis_version": "2026-08-28-route-a-v3-baseline-fairness-r1",
        "status": "passed",
        "source_verified": True,
        "frozen_prediction_sha256": frozen["prediction_sha256"],
        "method_count": len(roles),
        "configuration_rows": len(configs),
        "parameter_counts": counts,
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "fairness_conclusion": "Protocol-aligned outer evaluation; not an equal-compute or identical-representation benchmark.",
        "publication_requirements": [
            "Report all optimizer, architecture, early-stopping, input-shape and seed details.",
            "State that five seeds are ensemble members and not statistical replicates.",
            "State that deterministic Riemannian fits were copied to seed slots only for schema compatibility.",
            "Disclose the 0.5-55 Hz engineered/covariance versus 0.5-45 Hz raw-deep front-end difference.",
            "Do not claim identical tuning budgets or equal computational budgets.",
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[4/4] Route A v3十四方法公平性与完整训练配置审计完成。")
    print(f"方法数：{len(roles)}；完整配置项：{len(configs)}")
    print(
        f"参数量：Feature MLP={counts['feature_mlp']:,}，"
        f"EEGNet={counts['eegnet']:,}，EEG-Conformer={counts['eeg_conformer']:,}"
    )
    print("公平性结论：外层测试协议对齐，但不是等计算量或完全相同输入表示的比较。")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
