"""直接运行：审计 17 号实验，并生成五种子 recording-level 核心结果。

本程序只读取编号 18 的已完成输出，不修改任何训练结果。检查通过后，
结果写入编号 20 的新目录。程序不需要命令行参数，可以直接在 PyCharm 运行。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "18_输出_Stroop_LOSO核心模型"
CHECKPOINT_ROOT = SOURCE_ROOT / "01_检查点_可续跑"
OUTPUT_ROOT = HERE / "20_输出_Stroop_LOSO核心模型检查"

PREDICTION_PATH = SOURCE_ROOT / "02_汇总_目标窗口种子预测.csv"
TARGET_DIAGNOSTIC_PATH = SOURCE_ROOT / "03_汇总_目标诊断.csv"
META_DIAGNOSTIC_PATH = SOURCE_ROOT / "04_汇总_内层参与者诊断.csv"
META_RECORDING_PATH = SOURCE_ROOT / "05_汇总_内层Recording预测.csv"
SOURCE_MANIFEST_PATH = SOURCE_ROOT / "06_记录_运行信息和哈希.json"

RECORDING_OUTPUT = OUTPUT_ROOT / "01_结果_核心模型五种子Recording预测.csv"
SUBJECT_OUTPUT = OUTPUT_ROOT / "02_结果_核心模型参与者准确率.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "03_结果_核心模型汇总.csv"
AUDIT_OUTPUT = OUTPUT_ROOT / "04_记录_完整性审计和哈希.json"

SEEDS = (1335, 1388, 1441, 1494, 1547)
METHODS = ("always_nn", "rf", "extra_trees", "always_fuse")
GATE_FEATURES = (
    "feature_shift",
    "disagreement",
    "entropy_nn",
    "entropy_fuse",
    "confidence_nn",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_file_digest(paths: list[Path]) -> str:
    rows = [{"file": path.name, "sha256": sha256_file(path)} for path in sorted(paths)]
    canonical = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not bool(condition):
        raise AssertionError(message)
    checks.append(message)


def validate_probabilities(frame: pd.DataFrame, name: str, checks: list[str]) -> None:
    values = frame[["p0", "p1"]].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all(), f"{name}: 概率全部为有限数", checks)
    require((values >= 0.0).all(), f"{name}: 概率全部非负", checks)
    require(
        np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0),
        f"{name}: 每行概率和为 1",
        checks,
    )


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260728)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def five_seed_recordings(predictions: pd.DataFrame) -> pd.DataFrame:
    window_keys = [
        "dataset",
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "window_id",
        "method",
    ]
    labels = predictions.groupby(window_keys, as_index=False)["true_label"].agg(
        lambda values: int(np.unique(values)[0]) if np.unique(values).size == 1 else -1
    )
    if (labels["true_label"] < 0).any():
        raise AssertionError("同一窗口在不同种子之间出现不一致标签")
    probability = predictions.groupby(window_keys, as_index=False)[["p0", "p1"]].median()
    windows = labels.merge(probability, on=window_keys, validate="one_to_one")
    mass = windows[["p0", "p1"]].sum(axis=1)
    windows[["p0", "p1"]] = windows[["p0", "p1"]].div(mass, axis=0)

    recording_keys = [
        "dataset",
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "method",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in windows.groupby(recording_keys, sort=True, dropna=False):
        unique_labels = np.unique(group["true_label"].to_numpy(dtype=np.int64))
        if unique_labels.size != 1:
            raise AssertionError(f"Recording {keys!r} 内存在不一致标签")
        values = np.clip(group[["p0", "p1"]].to_numpy(dtype=np.float64), 1e-12, 1.0)
        aggregate = np.exp(np.mean(np.log(values), axis=0))
        aggregate = aggregate / aggregate.sum()
        prediction = int(np.argmax(aggregate))
        true_label = int(unique_labels[0])
        rows.append(
            {
                **dict(zip(recording_keys, keys)),
                "true_label": true_label,
                "p0": float(aggregate[0]),
                "p1": float(aggregate[1]),
                "pred_label": prediction,
                "correct": int(prediction == true_label),
                "n_windows": int(len(group)),
                "ensemble_size": len(SEEDS),
            }
        )
    return pd.DataFrame(rows).sort_values(recording_keys).reset_index(drop=True)


def summarize(recordings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = (
        recordings.groupby(["dataset", "protocol", "subject", "direction", "method"])[
            "correct"
        ]
        .agg(accuracy="mean", n_recordings="count")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
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
                "recordings_per_subject": int(group["n_recordings"].iloc[0]),
                "mean_accuracy": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    return subjects, summary


def main() -> None:
    checks: list[str] = []
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_outputs = [
        PREDICTION_PATH,
        TARGET_DIAGNOSTIC_PATH,
        META_DIAGNOSTIC_PATH,
        META_RECORDING_PATH,
    ]
    for path in source_outputs:
        observed = sha256_file(path)
        expected = manifest["output_sha256"][path.name]
        require(observed == expected, f"哈希一致: {path.name}", checks)

    checkpoint_groups = {
        "目标窗口预测": sorted(CHECKPOINT_ROOT.glob("01_目标窗口预测_*.csv")),
        "目标诊断": sorted(CHECKPOINT_ROOT.glob("02_目标诊断_*.json")),
        "内层参与者诊断": sorted(CHECKPOINT_ROOT.glob("03_内层参与者诊断_*.csv")),
        "内层Recording预测": sorted(CHECKPOINT_ROOT.glob("04_内层Recording预测_*.csv")),
    }
    for name, paths in checkpoint_groups.items():
        require(len(paths) == 75, f"{name}: 75 个检查点", checks)
    all_checkpoints = [path for paths in checkpoint_groups.values() for path in paths]
    require(len(all_checkpoints) == 300, "检查点总数为 300", checks)
    require(
        combined_file_digest(all_checkpoints) == manifest["checkpoint_bundle_sha256"],
        "300 个检查点的组合哈希一致",
        checks,
    )

    predictions = pd.read_csv(PREDICTION_PATH)
    target_diagnostics = pd.read_csv(TARGET_DIAGNOSTIC_PATH)
    meta_diagnostics = pd.read_csv(META_DIAGNOSTIC_PATH)
    meta_recordings = pd.read_csv(META_RECORDING_PATH)

    require(len(predictions) == 189020, "目标窗口种子预测为 189020 行", checks)
    require(len(target_diagnostics) == 75, "目标诊断为 75 行", checks)
    require(len(meta_diagnostics) == 1050, "内层参与者诊断为 1050 行", checks)
    require(len(meta_recordings) == 8400, "内层 Recording 预测为 8400 行", checks)

    expected_subjects = set(range(1, 16))
    expected_seeds = set(SEEDS)
    target_keys = [
        "dataset",
        "protocol",
        "subject",
        "direction",
        "recording_id",
        "window_id",
        "method",
        "seed",
    ]
    require(not predictions.duplicated(target_keys).any(), "目标窗口预测无重复键", checks)
    require(set(predictions["subject"].astype(int)) == expected_subjects, "覆盖 15 名外层参与者", checks)
    require(set(predictions["seed"].astype(int)) == expected_seeds, "覆盖冻结的 5 个种子", checks)
    require(set(predictions["method"].astype(str)) == set(METHODS), "覆盖 4 个核心方法", checks)
    require(set(predictions["direction"].astype(str)) == {"stroop"}, "目标任务仅为 Stroop", checks)
    validate_probabilities(predictions, "目标窗口预测", checks)

    seed_keys = [key for key in target_keys if key != "seed"]
    seed_counts = predictions.groupby(seed_keys)["seed"].nunique()
    require(seed_counts.eq(5).all(), "每个方法和窗口恰有 5 个种子", checks)
    method_keys = [key for key in target_keys if key not in {"method", "seed"}] + ["seed"]
    method_counts = predictions.groupby(method_keys)["method"].nunique()
    require(method_counts.eq(4).all(), "每个种子和窗口恰有 4 个核心方法", checks)
    recording_counts = predictions.groupby("subject")["recording_id"].nunique()
    require(recording_counts.eq(4).all(), "每名目标参与者恰有 4 个 Stroop recordings", checks)

    diagnostic_keys = ["subject", "seed"]
    require(not target_diagnostics.duplicated(diagnostic_keys).any(), "75 个目标诊断单元均唯一", checks)
    require(set(target_diagnostics["subject"].astype(int)) == expected_subjects, "目标诊断覆盖 15 名参与者", checks)
    require(set(target_diagnostics["seed"].astype(int)) == expected_seeds, "目标诊断覆盖 5 个种子", checks)
    require(target_diagnostics["outer_subject_excluded"].astype(bool).all(), "目标诊断确认外层参与者已排除", checks)
    require(target_diagnostics["inner_participants"].eq(14).all(), "每个外层单元含 14 名内层参与者", checks)
    require(target_diagnostics["target_batch_recordings"].eq(4).all(), "门控诊断使用 4 个未标注目标 recordings", checks)
    require(
        np.isfinite(target_diagnostics[list(GATE_FEATURES)].to_numpy(dtype=np.float64)).all(),
        "目标门控特征全部有限",
        checks,
    )

    meta_keys = ["outer_subject", "meta_subject", "seed"]
    require(not meta_diagnostics.duplicated(meta_keys).any(), "内层参与者诊断无重复键", checks)
    require(
        (meta_diagnostics["outer_subject"].astype(int) != meta_diagnostics["meta_subject"].astype(int)).all(),
        "内层参与者诊断中外层参与者始终排除",
        checks,
    )
    meta_counts = meta_diagnostics.groupby(["outer_subject", "seed"])["meta_subject"].nunique()
    require(meta_counts.eq(14).all(), "每个外层参与者和种子恰有 14 个 meta-participants", checks)
    require(
        np.isfinite(meta_diagnostics[list(GATE_FEATURES) + ["benefit"]].to_numpy(dtype=np.float64)).all(),
        "内层门控特征和收益全部有限",
        checks,
    )

    meta_recording_keys = ["outer_subject", "meta_subject", "recording_id", "method", "seed"]
    require(not meta_recordings.duplicated(meta_recording_keys).any(), "内层 Recording 预测无重复键", checks)
    require(set(meta_recordings["method"].astype(str)) == {"always_nn", "always_fuse"}, "内层 Recording 包含 NN 与融合", checks)
    validate_probabilities(meta_recordings, "内层 Recording 预测", checks)
    meta_recording_counts = meta_recordings.groupby(
        ["outer_subject", "meta_subject", "seed", "method"]
    )["recording_id"].nunique()
    require(meta_recording_counts.eq(4).all(), "每个 meta-participant 每种方法恰有 4 个 recordings", checks)

    verify = meta_recordings.copy()
    verify["pred_label"] = np.argmax(verify[["p0", "p1"]].to_numpy(dtype=np.float64), axis=1)
    verify["correct"] = (verify["pred_label"] == verify["true_label"]).astype(float)
    accuracy = verify.groupby(["outer_subject", "meta_subject", "seed", "method"])["correct"].mean()
    benefit = accuracy.unstack("method")
    benefit["recomputed_benefit"] = benefit["always_fuse"] - benefit["always_nn"]
    benefit = benefit.reset_index()
    aligned = meta_diagnostics.merge(
        benefit[["outer_subject", "meta_subject", "seed", "recomputed_benefit"]],
        on=["outer_subject", "meta_subject", "seed"],
        validate="one_to_one",
    )
    require(
        np.allclose(aligned["benefit"], aligned["recomputed_benefit"], atol=1e-12, rtol=0.0),
        "内层 benefit 可由 recording 预测精确复算",
        checks,
    )

    recordings = five_seed_recordings(predictions)
    require(len(recordings) == 240, "五种子核心结果为 240 行（15×4×4）", checks)
    require(recordings["ensemble_size"].eq(5).all(), "最终结果只含五种子集成模型", checks)
    require(recordings.groupby(["subject", "method"])["recording_id"].nunique().eq(4).all(), "每名参与者每种方法均有 4 个 recordings", checks)
    validate_probabilities(recordings, "五种子 Recording 结果", checks)
    subjects, summary = summarize(recordings)
    require(subjects["n_recordings"].eq(4).all(), "参与者准确率均由 4 个 recordings 计算", checks)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    recordings.to_csv(RECORDING_OUTPUT, index=False, lineterminator="\n")
    subjects.to_csv(SUBJECT_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")
    audit = {
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "source_output_sha256_verified": manifest["output_sha256"],
        "checkpoint_bundle_sha256_verified": manifest["checkpoint_bundle_sha256"],
        "checks_passed": len(checks),
        "check_messages": checks,
        "statistical_unit": "held-out participant",
        "evaluation_unit": "recording",
        "seed_is_statistical_unit": False,
        "ensemble_rule": "per-class window median across five seeds, renormalize, then recording geometric mean",
        "target_batch_transductive": True,
        "output_sha256": {
            RECORDING_OUTPUT.name: sha256_file(RECORDING_OUTPUT),
            SUBJECT_OUTPUT.name: sha256_file(SUBJECT_OUTPUT),
            SUMMARY_OUTPUT.name: sha256_file(SUMMARY_OUTPUT),
        },
        "claim_scope": "exploratory nested Stroop LOSO completion; no formal safety or risk-control guarantee",
    }
    AUDIT_OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = summary.copy()
    for column in ("mean_accuracy", "ci95_low", "ci95_high"):
        display[column] = 100.0 * display[column]
    print(f"完整性检查通过：{len(checks)} 项")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"输出目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
