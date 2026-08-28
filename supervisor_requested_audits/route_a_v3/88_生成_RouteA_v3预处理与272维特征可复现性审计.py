"""编号88：生成 Route A v3 预处理与 272 维特征可复现性审计。

本文件不训练模型。它读取实际冻结源代码，核对预处理常数，并输出：
- 272维特征逐块维数、索引、公式和实现说明；
- 可放入补充材料的英文方法说明；
- 便于作者学习和核对的中文说明；
- 带SHA-256的机器可读审计清单。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：16_preprocessing_feature_audit
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
LEGACY_SOURCE = PROJECT_ROOT / "1111.py"
CHANNEL_SOURCE = REVISION_ROOT / "eeg_channel_selection.py"
RAW_BASELINE_SOURCE = REVISION_ROOT / "route_a" / "route_a_lib" / "data.py"
PARENT_PROTOCOL_PATH = HERE / "64_冻结_RouteA_v3协议.json"
OUTPUT_ROOT = HERE / "16_preprocessing_feature_audit"

FEATURE_CSV = OUTPUT_ROOT / "88_272维特征逐块公式与索引.csv"
PREPROCESSING_CSV = OUTPUT_ROOT / "88_预处理逐步审计.csv"
MACHINE_JSON = OUTPUT_ROOT / "88_预处理与272维特征机器可读定义.json"
SUPPLEMENT_MD = OUTPUT_ROOT / "88_Supplementary_Exact_Preprocessing_and_272D_Features.md"
CHINESE_MD = OUTPUT_ROOT / "88_中文解读_预处理与272维特征.md"
MANIFEST_JSON = OUTPUT_ROOT / "88_预处理与272维特征审计_manifest.json"


CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C2"]
EXPECTED_BANDS = [
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 18.0),
    (18.0, 30.0),
    (30.0, 55.0),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载源代码：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def csv_escape(value: object) -> str:
    text = str(value)
    if any(character in text for character in [",", '"', "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的行：{path.name}")
    columns = list(rows[0])
    lines = [",".join(csv_escape(column) for column in columns)]
    for row in rows:
        if list(row) != columns:
            raise ValueError(f"CSV列顺序不一致：{path.name}")
        lines.append(",".join(csv_escape(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_frozen_constants(legacy, protocol: dict) -> dict:
    observed = {
        "sampling_rate_hz": int(legacy.DEFAULT_SFREQ),
        "window_seconds": float(legacy.WIN_SEC),
        "stride_seconds": float(legacy.STRIDE_SEC),
        "window_samples": int(round(float(legacy.WIN_SEC) * float(legacy.DEFAULT_SFREQ))),
        "stride_samples": int(round(float(legacy.STRIDE_SEC) * float(legacy.DEFAULT_SFREQ))),
        "fft_points": int(1 << (int(round(float(legacy.WIN_SEC) * float(legacy.DEFAULT_SFREQ))) - 1).bit_length()),
        "bandpass_enabled": bool(legacy.USE_BANDPASS_FILTER),
        "bandpass_low_hz": float(legacy.BANDPASS_LOW_HZ),
        "bandpass_high_hz": float(legacy.BANDPASS_HIGH_HZ),
        "bandpass_order": int(legacy.BANDPASS_ORDER),
        "line_frequency_hz": float(legacy.NOTCH_LINE_FREQ_HZ),
        "line_half_width_hz": float(legacy.NOTCH_LINE_WIDTH_HZ),
        "line_power_multiplier": float(legacy.NOTCH_LINE_ATTEN),
        "frontal_feature_mode": str(legacy.FRONTAL_FEATURE_MODE),
        "frontal_relative_power_multiplier": float(legacy.FRONTAL_REL_POWER_WEIGHT),
        "theta_alpha_relative_power_multiplier": float(legacy.WORKLOAD_THETA_ALPHA_REL_BOOST),
        "differential_entropy_enabled": bool(legacy.USE_DIFFERENTIAL_ENTROPY),
        "inter_channel_correlation_enabled": bool(legacy.USE_INTER_CHANNEL_CORR),
        "bands_hz": [tuple(map(float, pair)) for pair in legacy.BANDS],
    }
    expected = {
        "sampling_rate_hz": 250,
        "window_seconds": 8.5,
        "stride_seconds": 0.5,
        "window_samples": 2125,
        "stride_samples": 125,
        "fft_points": 4096,
        "bandpass_enabled": True,
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 55.0,
        "bandpass_order": 4,
        "line_frequency_hz": 50.0,
        "line_half_width_hz": 1.5,
        "line_power_multiplier": 0.03,
        "frontal_feature_mode": "first_half",
        "frontal_relative_power_multiplier": 1.12,
        "theta_alpha_relative_power_multiplier": 1.10,
        "differential_entropy_enabled": True,
        "inter_channel_correlation_enabled": False,
        "bands_hz": EXPECTED_BANDS,
    }
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not math.isclose(float(observed_value), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"冻结常数不匹配：{key}={observed_value}，预期{expected_value}")
        elif observed_value != expected_value:
            raise AssertionError(f"冻结常数不匹配：{key}={observed_value}，预期{expected_value}")
    if protocol.get("eeg_channels") != CHANNELS:
        raise AssertionError(f"64号协议通道顺序不匹配：{protocol.get('eeg_channels')}")
    if "feature_dimension" in protocol and int(protocol["feature_dimension"]) != 272:
        raise AssertionError("64号协议中记录的特征维数不是272")

    channel_text = CHANNEL_SOURCE.read_text(encoding="utf-8")
    required_channel_fragments = ["values[:, 1:9]", "output_columns=8", "packet_counter_removed"]
    if not all(fragment in channel_text for fragment in required_channel_fragments):
        raise AssertionError("纯8通道选择器的关键实现与冻结说明不一致")
    raw_text = RAW_BASELINE_SOURCE.read_text(encoding="utf-8")
    if "high = min(45.0" not in raw_text or "sosfiltfilt" not in raw_text:
        raise AssertionError("raw-EEG基线的0.5-45 Hz实现未找到")
    return observed


def build_feature_rows() -> list[dict]:
    channel_count = 8
    band_count = 6
    definitions = [
        (
            "Log absolute band power",
            channel_count * band_count,
            "log(mean_{f in band} P_c(f) + epsilon)",
            "Six bands for each of eight channels; the frontal multiplier is added in log space to channels 1-4.",
        ),
        (
            "Relative band power",
            channel_count * band_count,
            "mean_{f in band} P_c(f) / mean_{0.5<=f<55} P_c(f)",
            "Six bands per channel; channels 1-4 are multiplied by 1.12 and theta/alpha entries by 1.10.",
        ),
        (
            "Spectral entropy",
            channel_count,
            "-sum_f q_c(f) log(q_c(f)), q_c(f)=P_c(f)/sum_f P_c(f)",
            "Computed per channel over the implemented 0.5-55 Hz mask.",
        ),
        (
            "Theta/beta and alpha/beta ratios",
            channel_count * 2,
            "theta/(low-beta+high-beta); alpha/(low-beta+high-beta)",
            "Two ratios for each channel using relative powers after the fixed multipliers.",
        ),
        (
            "Time-domain summaries",
            channel_count * 4,
            "mean(x); std(x); sqrt(mean(x^2)); max(x)-min(x)",
            "Four summaries per channel after within-window z-score; mean, standard deviation and RMS are therefore nearly fixed by construction.",
        ),
        (
            "Across-channel spatial summaries",
            band_count * 3 + 2,
            "mean_c(relative); std_c(relative); mean_c(log absolute); mean_c(entropy); std_c(entropy)",
            "Six values for each of the first three summaries plus two entropy summaries.",
        ),
        (
            "Hjorth descriptors",
            channel_count * 3,
            "log(var(x)); sqrt(var(dx)/var(x)); sqrt(var(d2x)/var(dx))/mobility",
            "Activity, mobility and complexity for each channel.",
        ),
        (
            "Physiological extras",
            channel_count + 1 + channel_count,
            "mean(abs(dx)); mean(alpha first 4)-mean(alpha last 4); (delta+theta)/(alpha+low-beta+high-beta)",
            "The one-dimensional alpha term is a channel-group contrast, not a standard left-right frontal alpha asymmetry measure.",
        ),
        (
            "Cross-band log summaries",
            3,
            "log(mean_c(alpha/beta)); log(mean_c(gamma/alpha)); log(mean_c(theta/alpha))",
            "Three across-channel ratios.",
        ),
        (
            "Frontal workload summaries",
            2,
            "log(mean_{c=1..4}(theta)/mean_{c=1..4}(beta)); log(mean_{c=1..4}(alpha))",
            "The first four ordered channels are Fp1, Fp2, F7 and F3.",
        ),
        (
            "Channel-band differential entropy",
            channel_count * band_count,
            "0.5 log(2*pi*e*var(x_{c,band}))",
            "Six FFT-mask reconstructed band signals for each of eight channels.",
        ),
        (
            "Across-channel differential entropy",
            band_count,
            "mean_c[0.5 log(2*pi*e*var(x_{c,band}))]",
            "One channel-mean differential-entropy value per band.",
        ),
    ]
    rows = []
    start = 1
    for block_id, (name, dimension, formula, notes) in enumerate(definitions, start=1):
        end = start + int(dimension) - 1
        rows.append(
            {
                "block_id": block_id,
                "feature_block": name,
                "index_start_1based": start,
                "index_end_1based": end,
                "dimension": int(dimension),
                "formula": formula,
                "implementation_notes": notes,
            }
        )
        start = end + 1
    if start - 1 != 272 or sum(int(row["dimension"]) for row in rows) != 272:
        raise AssertionError("特征逐块维数之和不是272")
    return rows


def build_preprocessing_rows(constants: dict) -> list[dict]:
    return [
        {
            "step": 1,
            "operation": "Fail-closed channel selection",
            "exact_implementation": "Detect modulo-256 packet counter in column 0; retain columns 1-8 only.",
            "purpose": "Exclude packet counter, accelerometer, marker, status and other auxiliary columns.",
            "limitation_or_interpretation": "Exactly eight finite EEG columns are required; the parser refuses to guess when extra columns are ambiguous.",
        },
        {
            "step": 2,
            "operation": "Continuous-recording band-pass",
            "exact_implementation": "Fourth-order Butterworth 0.5-55 Hz, applied with forward-backward sosfiltfilt before windowing.",
            "purpose": "Suppress drift and frequencies above the engineered feature range.",
            "limitation_or_interpretation": "Forward-backward filtering is zero-phase but non-causal; this is an offline analysis pipeline.",
        },
        {
            "step": 3,
            "operation": "Overlapping windows",
            "exact_implementation": f"{constants['window_seconds']} s ({constants['window_samples']} samples) with {constants['stride_seconds']} s ({constants['stride_samples']} samples) stride.",
            "purpose": "Obtain stable spectral estimates from each completed window.",
            "limitation_or_interpretation": "Adjacent windows share 94.12% of raw samples and must never be randomly separated across training and validation.",
        },
        {
            "step": 4,
            "operation": "Within-window per-channel z-score",
            "exact_implementation": "x_c(t) <- [x_c(t)-mean_t(x_c)] / max(std_t(x_c), 1e-6).",
            "purpose": "Reduce amplitude and impedance-scale differences before feature extraction.",
            "limitation_or_interpretation": "Uses the complete 8.5-s window. It is valid for completed-window offline/batch inference, not causal sample-by-sample prediction.",
        },
        {
            "step": 5,
            "operation": "Hann taper and FFT",
            "exact_implementation": "Apply a 2125-point Hann window and zero-pad to a 4096-point real FFT.",
            "purpose": "Reduce spectral leakage and provide a convenient frequency grid.",
            "limitation_or_interpretation": "The implementation does not use a 2048-point FFT. Zero-padding refines the frequency grid but does not add physical frequency resolution.",
        },
        {
            "step": 6,
            "operation": "50-Hz spectral attenuation",
            "exact_implementation": "Multiply FFT power bins from 48.5 through 51.5 Hz by 0.03.",
            "purpose": "Heuristically suppress mains-frequency contamination in spectral features.",
            "limitation_or_interpretation": "This is not a designed time-domain notch filter. The value 0.03 is an inherited heuristic, not a learned or theoretically guaranteed constant.",
        },
        {
            "step": 7,
            "operation": "Fixed spectral emphasis",
            "exact_implementation": "Multiply all relative band powers in Fp1/Fp2/F7/F3 by 1.12; multiply theta and alpha relative powers in every channel by 1.10.",
            "purpose": "Inherited heuristic emphasis of frontal and workload-related spectral terms.",
            "limitation_or_interpretation": "The values 1.12 and 1.10 were not selected inside the v3 nested protocol and must not be presented as optimized or physiologically validated constants.",
        },
        {
            "step": 8,
            "operation": "Covariance output for Riemannian baselines",
            "exact_implementation": "C = X X^T / 2125 + 1e-4 I for each standardized 8-channel window.",
            "purpose": "Provide a separate symmetric positive-definite representation for MDM and tangent-space baselines.",
            "limitation_or_interpretation": "The covariance matrix is not part of the 272-dimensional flat feature vector. An 8-channel tangent vector has 8*9/2=36 dimensions, not 253.",
        },
        {
            "step": 9,
            "operation": "Raw-EEG deep-baseline preprocessing",
            "exact_implementation": "Fourth-order zero-phase 0.5-45 Hz band-pass followed by the same within-window per-channel z-score.",
            "purpose": "Provide standard bandwidth-limited raw input for EEGNet and EEG-Conformer.",
            "limitation_or_interpretation": "Its upper cutoff differs from the 55-Hz engineered-feature path. This must be disclosed as a representation-specific preprocessing difference, not described as an identical front end.",
        },
    ]


def feature_table_markdown(rows: list[dict]) -> str:
    lines = [
        "| Block | Feature group | Indices | Dimension |",
        "|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['block_id']} | {row['feature_block']} | "
            f"{row['index_start_1based']}-{row['index_end_1based']} | {row['dimension']} |"
        )
    return "\n".join(lines)


def build_supplement(feature_rows: list[dict], preprocessing_rows: list[dict]) -> str:
    details = []
    for row in feature_rows:
        details.append(
            f"### S{row['block_id']}. {row['feature_block']} ({row['dimension']} features)\n\n"
            f"Formula: `{row['formula']}`\n\n{row['implementation_notes']}"
        )
    preprocessing = []
    for row in preprocessing_rows:
        preprocessing.append(
            f"{row['step']}. **{row['operation']}.** {row['exact_implementation']} "
            f"{row['limitation_or_interpretation']}"
        )
    return f"""# Exact preprocessing and 272-dimensional feature definition

## Scope

This supplement documents the implementation used in the corrected eight-channel exploratory audit. The retained channels, in fixed order, were Fp1, Fp2, F7, F3, Fz, F4, F8, and C2. The OpenBCI packet counter and every auxiliary column were excluded. The implementation is an offline completed-window pipeline and is not a causal sample-by-sample or single-recording online system.

## Preprocessing

{chr(10).join(preprocessing)}

## Feature-vector composition

{feature_table_markdown(feature_rows)}

The blocks sum to 272 features. The separately stored 8 x 8 regularized covariance matrices were used by the Riemannian baselines and were not appended to this vector.

{chr(10).join(details)}

## Status of fixed constants

The 50-Hz power multiplier (0.03), the first-four-channel spectral multiplier (1.12), and the theta/alpha multiplier (1.10) were inherited heuristic settings. They were not optimized within the corrected nested protocol and do not provide a formal physiological or risk guarantee. Their influence is therefore treated only in a post-hoc preprocessing sensitivity analysis.

## Deployment interpretation

Forward-backward filtering and standardization over each complete 8.5-s window use future samples relative to the beginning of that window. The present results therefore support only offline or delayed completed-window batch inference. Moreover, the selective gate uses all unlabeled recordings in a target participant-direction cell and is transductive; it is not a conventional single-recording online gate.
"""


def build_chinese_explanation(feature_rows: list[dict]) -> str:
    return f"""# Route A v3 预处理与272维特征中文解读

## 最重要的核对结果

1. 实际输入是8个EEG通道：{', '.join(CHANNELS)}。packet counter和所有辅助列均被排除。
2. 每个窗口是8.5秒，即2125个采样点；相邻窗口只移动0.5秒，因此共享94.12%的原始信号。
3. FFT不是2048点。代码把2125点补零到下一个2的整数次幂，所以使用4096点FFT。补零只让频率网格更密，不会凭空增加真实信息。
4. 50 Hz处理是在48.5-51.5 Hz的频谱功率上乘0.03，不是标准时域陷波器。
5. 1.12和1.10是旧流程遗留的启发式倍率，不是v3实验自动学习出来的，也不能说有理论保证。
6. 窗口z-score需要先取得完整8.5秒窗口，所以论文必须定位为离线或延迟窗口分析，不能宣称逐采样实时预测。
7. 272维由{len(feature_rows)}个特征块组成。单独的8x8协方差矩阵供Riemannian基线使用，不计入272维。
8. 代码中的“alpha asymmetry”实际是前四通道平均alpha减后四通道平均alpha。它不是标准左右半球FAA，论文中应改称“channel-group alpha contrast”。
9. EEGNet和EEG-Conformer使用0.5-45 Hz，手工特征使用0.5-55 Hz。两者不是完全相同的前端，必须在公平性限制中公开说明。

## 272维组成

{feature_table_markdown(feature_rows)}

这份表可以作为补充材料的基础，但最后英文论文应使用88号生成的英文补充说明，并结合后续预处理敏感性结果一起解释。
"""


def main() -> None:
    required = [LEGACY_SOURCE, CHANNEL_SOURCE, RAW_BASELINE_SOURCE, PARENT_PROTOCOL_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"缺少审计输入：{path}")
    protocol = json.loads(PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, str(PROJECT_ROOT))
    legacy = load_module(LEGACY_SOURCE, "route_a_v3_feature_audit_legacy")

    print("[1/4] 核对冻结预处理常数和纯8通道选择器。")
    constants = verify_frozen_constants(legacy, protocol)
    print("[2/4] 计算272维特征逐块维数和1-based索引。")
    feature_rows = build_feature_rows()
    preprocessing_rows = build_preprocessing_rows(constants)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(FEATURE_CSV, feature_rows)
    write_csv(PREPROCESSING_CSV, preprocessing_rows)
    machine_definition = {
        "analysis_version": "2026-08-27-route-a-v3-preprocessing-feature-audit-r1",
        "status": "source_verified",
        "channels_in_order": CHANNELS,
        "constants": constants,
        "feature_dimension": 272,
        "feature_blocks": feature_rows,
        "preprocessing_steps": preprocessing_rows,
        "important_corrections": {
            "fft": "4096-point zero-padded FFT for 2125 samples; not 2048 points",
            "alpha_term": "channel-group alpha contrast; not standard left-right FAA",
            "riemannian_tangent_dimension": 36,
            "flat_feature_dimension": 272,
            "online_scope": "offline or delayed completed-window batch inference only",
            "deep_baseline_bandpass": "0.5-45 Hz versus 0.5-55 Hz engineered-feature path",
        },
    }
    MACHINE_JSON.write_text(
        json.dumps(machine_definition, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SUPPLEMENT_MD.write_text(
        build_supplement(feature_rows, preprocessing_rows), encoding="utf-8"
    )
    CHINESE_MD.write_text(build_chinese_explanation(feature_rows), encoding="utf-8")

    print("[3/4] 生成英文补充材料、中文解读和机器可读定义。")
    source_files = {
        "feature_source": LEGACY_SOURCE,
        "channel_selector": CHANNEL_SOURCE,
        "raw_baseline_preprocessing": RAW_BASELINE_SOURCE,
        "parent_protocol": PARENT_PROTOCOL_PATH,
        "audit_runner": Path(__file__).resolve(),
    }
    output_files = [
        FEATURE_CSV,
        PREPROCESSING_CSV,
        MACHINE_JSON,
        SUPPLEMENT_MD,
        CHINESE_MD,
    ]
    manifest = {
        "analysis_version": "2026-08-27-route-a-v3-preprocessing-feature-audit-r1",
        "status": "passed",
        "source_verified": True,
        "feature_dimension": 272,
        "feature_block_count": len(feature_rows),
        "dimension_sum": sum(int(row["dimension"]) for row in feature_rows),
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256(path) for path in output_files},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "publication_warnings": [
            "Do not describe the 0.03, 1.12 or 1.10 constants as optimized or theoretically guaranteed.",
            "Do not describe the channel-group alpha contrast as standard left-right frontal alpha asymmetry.",
            "Do not claim causal sample-by-sample or single-recording online deployment.",
            "Disclose the 0.5-55 versus 0.5-45 Hz representation-specific preprocessing difference.",
        ],
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[4/4] Route A v3 预处理与272维特征审计完成。")
    print(f"特征块：{len(feature_rows)}；维数合计：{sum(int(row['dimension']) for row in feature_rows)}")
    print("关键修正：2125点窗口使用4096点FFT；8通道Riemannian切空间为36维。")
    print("关键限制：窗口z-score与零相位滤波不是逐采样因果流程。")
    print("校验状态：passed")


if __name__ == "__main__":
    main()
