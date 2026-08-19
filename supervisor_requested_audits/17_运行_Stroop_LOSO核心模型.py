"""直接运行：完成 Stroop 的 fully nested LOSO 核心模型预测。

核心模型包括 Residual MLP、Random forest、ExtraTrees 和 OOF stacker。
同时保存后续 DASF、CB-SF、固定融合和 recording stacker 所需的严格内层 OOF 信息。

特点：
- 不需要命令行参数；
- 使用与 Arithmetic LOSO 相同的冻结模型和超参数；
- 每个外层参与者和种子均保存检查点，中断后重新运行会自动续跑；
- 结果写入编号 18 的独立目录，不覆盖 Route A 或 Arithmetic 结果。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
PROJECT_ROOT = REVISION_ROOT
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
OUTPUT_ROOT = HERE / "18_输出_Stroop_LOSO核心模型"
CHECKPOINT_ROOT = OUTPUT_ROOT / "01_检查点_可续跑"

PREDICTION_OUTPUT = OUTPUT_ROOT / "02_汇总_目标窗口种子预测.csv"
TARGET_DIAGNOSTIC_OUTPUT = OUTPUT_ROOT / "03_汇总_目标诊断.csv"
META_DIAGNOSTIC_OUTPUT = OUTPUT_ROOT / "04_汇总_内层参与者诊断.csv"
META_RECORDING_OUTPUT = OUTPUT_ROOT / "05_汇总_内层Recording预测.csv"
MANIFEST_OUTPUT = OUTPUT_ROOT / "06_记录_运行信息和哈希.json"

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(ROUTE_A_ROOT))

from revision_pipeline.models import _make_forest, fit_mlp  # noqa: E402
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.risk_gate import label_free_diagnostics, robust_feature_shift  # noqa: E402
from revision_pipeline.splits import outer_loso, subject_loso_folds  # noqa: E402
from route_a_lib.data import (  # noqa: E402
    FeatureBundle,
    load_feature_bundles,
    normalized_frobenius_shift,
)
from route_a_lib.probability import (  # noqa: E402
    BASE_METHODS,
    clean_dasf_decision,
    crossfit_window_stacker,
    logit_features,
)


TARGET_METHODS = ("always_nn", "rf", "extra_trees", "always_fuse")
SEEDS = (1335, 1388, 1441, 1494, 1547)


@dataclass
class CellResult:
    test: dict[str, np.ndarray]
    base_oof: dict[str, np.ndarray]
    fusion_meta_oof: np.ndarray
    selected_epoch: int
    inner_best_epochs: list[int]


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


def recording_accuracy(
    probability: np.ndarray,
    labels: np.ndarray,
    recordings: np.ndarray,
) -> float:
    outcomes: list[int] = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        values = np.unique(labels[mask])
        if values.size != 1:
            raise ValueError(f"Recording {recording!r} contains inconsistent labels")
        aggregate = np.exp(np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0))
        aggregate = aggregate / aggregate.sum()
        outcomes.append(int(int(np.argmax(aggregate)) == int(values[0])))
    return float(np.mean(outcomes))


def fit_global_stacker(
    base_oof: dict[str, np.ndarray],
    labels: np.ndarray,
    recordings: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LogisticRegression:
    unique_recordings, counts = np.unique(recordings, return_counts=True)
    count_by_recording = dict(zip(unique_recordings.tolist(), counts.tolist()))
    sample_weight = np.asarray(
        [1.0 / count_by_recording[value] for value in recordings],
        dtype=np.float64,
    )
    sample_weight = sample_weight / sample_weight.mean()
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    model.fit(
        logit_features(*(base_oof[method] for method in BASE_METHODS)),
        labels,
        sample_weight=sample_weight,
    )
    return model


def fit_outer_cell(
    source: FeatureBundle,
    target: FeatureBundle,
    *,
    outer_subject: int,
    seed: int,
    protocol: Protocol,
    device: torch.device,
) -> CellResult:
    configs = {name: protocol.model(name) for name in ("mlp", "rf", "extra_trees", "stacker")}
    splits = subject_loso_folds(source.subjects)
    base_oof = {
        method: np.full((len(source.labels), 2), np.nan, dtype=np.float64)
        for method in BASE_METHODS
    }
    selected_epochs: list[int] = []

    for fold_number, split in enumerate(splits, start=1):
        held = np.unique(source.subjects[split.validation])
        if held.size != 1:
            raise AssertionError("Each inner LOSO fold must hold one participant")
        fold_seed = int(seed) + 1009 * fold_number
        mlp, selected_epoch = fit_mlp(
            source.features[split.train],
            source.labels[split.train],
            configs["mlp"],
            fold_seed,
            device,
            x_validation=source.features[split.validation],
            y_validation=source.labels[split.validation],
            validation_recordings=source.recordings[split.validation],
        )
        selected_epochs.append(int(selected_epoch))
        base_oof["always_nn"][split.validation] = mlp.predict_proba(
            source.features[split.validation]
        )
        for offset, method in enumerate(("rf", "extra_trees"), start=1):
            forest = _make_forest(method, configs[method], fold_seed + offset * 37)
            forest.fit(source.features[split.train], source.labels[split.train])
            base_oof[method][split.validation] = forest.predict_proba(
                source.features[split.validation]
            )
        del mlp
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"    [inner done] outer=S{outer_subject:02d} seed={seed} "
            f"fold={fold_number:02d}/{len(splits)} held=S{int(held[0]):02d} epoch={selected_epoch}"
        )

    for method, values in base_oof.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"Incomplete OOF probabilities for {method}")

    fusion_meta_oof = crossfit_window_stacker(
        base_oof,
        source.labels,
        source.recordings,
        source.subjects,
        c_value=float(configs["stacker"]["C"]),
        seed=seed,
    )
    global_stacker = fit_global_stacker(
        base_oof,
        source.labels,
        source.recordings,
        c_value=float(configs["stacker"]["C"]),
        max_iter=int(configs["stacker"]["max_iter"]),
        seed=seed,
    )

    selected_epoch = max(1, int(np.median(selected_epochs)))
    final_mlp, _ = fit_mlp(
        source.features,
        source.labels,
        configs["mlp"],
        int(seed),
        device,
        fixed_epochs=selected_epoch,
    )
    test = {"always_nn": final_mlp.predict_proba(target.features)}
    for offset, method in enumerate(("rf", "extra_trees"), start=1):
        forest = _make_forest(method, configs[method], int(seed) + offset * 37)
        forest.fit(source.features, source.labels)
        test[method] = forest.predict_proba(target.features)
    test["always_fuse"] = global_stacker.predict_proba(
        logit_features(*(test[method] for method in BASE_METHODS))
    )
    del final_mlp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return CellResult(
        test=test,
        base_oof=base_oof,
        fusion_meta_oof=fusion_meta_oof,
        selected_epoch=selected_epoch,
        inner_best_epochs=selected_epochs,
    )


def target_prediction_rows(
    target: FeatureBundle,
    probabilities: dict[str, np.ndarray],
    *,
    subject: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method in TARGET_METHODS:
        probability = probabilities[method]
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "openbci",
                    "protocol": "loso",
                    "subject": int(subject),
                    "direction": "stroop",
                    "recording_id": target.recordings,
                    "window_id": target.window_ids,
                    "method": method,
                    "seed": int(seed),
                    "true_label": target.labels,
                    "p0": probability[:, 0],
                    "p1": probability[:, 1],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def meta_recording_rows(
    source: FeatureBundle,
    result: CellResult,
    *,
    outer_subject: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meta_subject in sorted(np.unique(source.subjects).astype(int).tolist()):
        subject_mask = source.subjects == meta_subject
        for method, probability in (
            ("always_nn", result.base_oof["always_nn"]),
            ("always_fuse", result.fusion_meta_oof),
        ):
            for recording in np.unique(source.recordings[subject_mask]):
                mask = subject_mask & (source.recordings == recording)
                labels = np.unique(source.labels[mask])
                if labels.size != 1:
                    raise ValueError(f"Recording {recording!r} contains inconsistent labels")
                aggregate = np.exp(
                    np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0)
                )
                aggregate = aggregate / aggregate.sum()
                rows.append(
                    {
                        "outer_subject": int(outer_subject),
                        "meta_subject": int(meta_subject),
                        "recording_id": str(recording),
                        "method": method,
                        "seed": int(seed),
                        "true_label": int(labels[0]),
                        "p0": float(aggregate[0]),
                        "p1": float(aggregate[1]),
                    }
                )
    return pd.DataFrame(rows)


def meta_diagnostic_rows(
    source: FeatureBundle,
    result: CellResult,
    *,
    outer_subject: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for meta_subject in sorted(np.unique(source.subjects).astype(int).tolist()):
        target_mask = source.subjects == meta_subject
        reference_mask = ~target_mask
        diagnostics = label_free_diagnostics(
            result.base_oof["always_nn"][target_mask],
            result.fusion_meta_oof[target_mask],
            source.features[reference_mask],
            source.features[target_mask],
        )
        diagnostics.update(
            {
                "feature_shift_original": float(diagnostics["feature_shift"]),
                "feature_shift_robust": robust_feature_shift(
                    source.features[reference_mask],
                    source.features[target_mask],
                ),
                "outer_subject": int(outer_subject),
                "meta_subject": int(meta_subject),
                "seed": int(seed),
                "benefit": recording_accuracy(
                    result.fusion_meta_oof[target_mask],
                    source.labels[target_mask],
                    source.recordings[target_mask],
                )
                - recording_accuracy(
                    result.base_oof["always_nn"][target_mask],
                    source.labels[target_mask],
                    source.recordings[target_mask],
                ),
            }
        )
        diagnostics["feature_shift"] = diagnostics["feature_shift_robust"]
        rows.append(diagnostics)
    return pd.DataFrame(rows)


def target_diagnostic(
    source: FeatureBundle,
    target: FeatureBundle,
    result: CellResult,
    *,
    subject: int,
    seed: int,
    route_protocol: dict,
) -> dict[str, object]:
    diagnostics = label_free_diagnostics(
        result.test["always_nn"],
        result.test["always_fuse"],
        source.features,
        target.features,
    )
    original_shift = float(diagnostics["feature_shift"])
    robust_shift = robust_feature_shift(source.features, target.features)
    covariance_shift = normalized_frobenius_shift(source.covariances, target.covariances)
    nn_accuracy = recording_accuracy(
        result.base_oof["always_nn"],
        source.labels,
        source.recordings,
    )
    fuse_accuracy = recording_accuracy(
        result.fusion_meta_oof,
        source.labels,
        source.recordings,
    )
    gate = route_protocol["dasf_clean"]
    dasf = clean_dasf_decision(
        nn_accuracy,
        fuse_accuracy,
        covariance_shift,
        base_margin=float(gate["base_margin"]),
        alpha=float(gate["shift_alpha"]),
        max_margin=float(gate["max_margin"]),
    )
    diagnostics.update(
        {
            "protocol": "loso",
            "subject": int(subject),
            "direction": "stroop",
            "seed": int(seed),
            "feature_shift_original": original_shift,
            "feature_shift_robust": robust_shift,
            "feature_shift": robust_shift,
            "covariance_shift": covariance_shift,
            "validation_accuracy_nn": dasf.validation_accuracy_nn,
            "validation_accuracy_fuse_meta_crossfit": dasf.validation_accuracy_fuse,
            "validation_gain": dasf.validation_gain,
            "dasf_effective_margin": dasf.effective_margin,
            "dasf_used_fusion_seed_diagnostic": dasf.use_fusion,
            "selected_epoch": int(result.selected_epoch),
            "inner_selected_epochs": "|".join(str(value) for value in result.inner_best_epochs),
            "inner_participants": int(np.unique(source.subjects).size),
            "outer_subject_excluded": True,
            "target_batch_recordings": int(np.unique(target.recordings).size),
        }
    )
    return diagnostics


def checkpoint_paths(subject: int, seed: int) -> tuple[Path, Path, Path, Path]:
    stem = f"S{subject:02d}_Seed{seed}"
    return (
        CHECKPOINT_ROOT / f"01_目标窗口预测_{stem}.csv",
        CHECKPOINT_ROOT / f"02_目标诊断_{stem}.json",
        CHECKPOINT_ROOT / f"03_内层参与者诊断_{stem}.csv",
        CHECKPOINT_ROOT / f"04_内层Recording预测_{stem}.csv",
    )


def checkpoint_complete(subject: int, seed: int) -> bool:
    prediction_path, diagnostic_path, meta_path, meta_recording_path = checkpoint_paths(subject, seed)
    if not all(path.exists() for path in (prediction_path, diagnostic_path, meta_path, meta_recording_path)):
        return False
    try:
        prediction = pd.read_csv(prediction_path)
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        meta = pd.read_csv(meta_path)
        meta_recording = pd.read_csv(meta_recording_path)
    except (ValueError, json.JSONDecodeError):
        return False
    prediction_keys = ["recording_id", "window_id", "method", "seed"]
    return bool(
        not prediction.empty
        and set(prediction["method"].astype(str)) == set(TARGET_METHODS)
        and set(prediction["subject"].astype(int)) == {int(subject)}
        and set(prediction["seed"].astype(int)) == {int(seed)}
        and set(prediction["direction"].astype(str)) == {"stroop"}
        and not prediction.duplicated(prediction_keys).any()
        and diagnostic.get("subject") == int(subject)
        and diagnostic.get("seed") == int(seed)
        and diagnostic.get("direction") == "stroop"
        and len(meta) == 14
        and meta["meta_subject"].nunique() == 14
        and int(subject) not in set(meta["meta_subject"].astype(int))
        and len(meta_recording) == 14 * 4 * 2
        and set(meta_recording["method"].astype(str)) == {"always_nn", "always_fuse"}
    )


def write_checkpoint(
    subject: int,
    seed: int,
    target_rows: pd.DataFrame,
    target_diagnostics: dict[str, object],
    meta_rows: pd.DataFrame,
    meta_recording_rows_frame: pd.DataFrame,
) -> None:
    prediction_path, diagnostic_path, meta_path, meta_recording_path = checkpoint_paths(subject, seed)
    target_rows.to_csv(prediction_path, index=False, lineterminator="\n")
    diagnostic_path.write_text(
        json.dumps(target_diagnostics, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    meta_rows.to_csv(meta_path, index=False, lineterminator="\n")
    meta_recording_rows_frame.to_csv(meta_recording_path, index=False, lineterminator="\n")


def consolidate() -> None:
    prediction_files = sorted(CHECKPOINT_ROOT.glob("01_目标窗口预测_*.csv"))
    diagnostic_files = sorted(CHECKPOINT_ROOT.glob("02_目标诊断_*.json"))
    meta_files = sorted(CHECKPOINT_ROOT.glob("03_内层参与者诊断_*.csv"))
    meta_recording_files = sorted(CHECKPOINT_ROOT.glob("04_内层Recording预测_*.csv"))
    expected = len(SEEDS) * 15
    observed = [len(prediction_files), len(diagnostic_files), len(meta_files), len(meta_recording_files)]
    if observed != [expected] * 4:
        raise RuntimeError(f"Incomplete checkpoint collection: expected {expected} per type, got {observed}")

    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    diagnostics = pd.DataFrame(
        [json.loads(path.read_text(encoding="utf-8")) for path in diagnostic_files]
    )
    meta = pd.concat([pd.read_csv(path) for path in meta_files], ignore_index=True)
    meta_recordings = pd.concat(
        [pd.read_csv(path) for path in meta_recording_files],
        ignore_index=True,
    )

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
    if predictions.duplicated(target_keys).any():
        raise AssertionError("Consolidated target predictions contain duplicate rows")
    predictions = predictions.sort_values(target_keys).reset_index(drop=True)
    diagnostics = diagnostics.sort_values(["subject", "seed"]).reset_index(drop=True)
    meta = meta.sort_values(["outer_subject", "meta_subject", "seed"]).reset_index(drop=True)
    meta_recordings = meta_recordings.sort_values(
        ["outer_subject", "meta_subject", "seed", "recording_id", "method"]
    ).reset_index(drop=True)

    predictions.to_csv(PREDICTION_OUTPUT, index=False, lineterminator="\n")
    diagnostics.to_csv(TARGET_DIAGNOSTIC_OUTPUT, index=False, lineterminator="\n")
    meta.to_csv(META_DIAGNOSTIC_OUTPUT, index=False, lineterminator="\n")
    meta_recordings.to_csv(META_RECORDING_OUTPUT, index=False, lineterminator="\n")

    source_files = {
        "runner": Path(__file__).resolve(),
        "model_protocol": REVISION_ROOT / "protocol.json",
        "route_a_protocol": ROUTE_A_ROOT / "protocol_route_a.json",
        "feature_source": PROJECT_ROOT / "eeg_feature_pipeline.py",
    }
    output_files = [
        PREDICTION_OUTPUT,
        TARGET_DIAGNOSTIC_OUTPUT,
        META_DIAGNOSTIC_OUTPUT,
        META_RECORDING_OUTPUT,
    ]
    all_checkpoints = prediction_files + diagnostic_files + meta_files + meta_recording_files
    manifest = {
        "purpose": "Fully nested Stroop LOSO core model predictions",
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "task": "stroop",
        "outer_folds": 15,
        "inner_folds_per_outer": 14,
        "seeds": list(SEEDS),
        "models": list(TARGET_METHODS),
        "statistical_unit": "held-out participant",
        "evaluation_unit": "recording",
        "target_batch_recordings": 4,
        "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256_file(path) for path in output_files},
        "checkpoint_file_count": len(all_checkpoints),
        "checkpoint_bundle_sha256": combined_file_digest(all_checkpoints),
        "row_counts": {
            "target_window_seed_predictions": int(len(predictions)),
            "target_diagnostics": int(len(diagnostics)),
            "meta_participant_diagnostics": int(len(meta)),
            "meta_recording_predictions": int(len(meta_recordings)),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": "cuda:0",
        },
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stroop LOSO core models require CUDA, but CUDA is unavailable")
    device = torch.device("cuda:0")
    protocol = Protocol.load(REVISION_ROOT / "protocol.json")
    route_protocol = json.loads(
        (ROUTE_A_ROOT / "protocol_route_a.json").read_text(encoding="utf-8")
    )
    if tuple(int(value) for value in protocol.seeds) != SEEDS:
        raise RuntimeError("Model protocol seed list differs from the frozen five-seed policy")
    if tuple(int(value) for value in route_protocol["seeds"]) != SEEDS:
        raise RuntimeError("Route A seed list differs from the frozen five-seed policy")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/3] 正在读取冻结的 8 通道 Stroop 特征。")
    _, stroop = load_feature_bundles(PROJECT_ROOT, torch.device("cpu"))
    outer = {subject: (train, test) for subject, train, test in outer_loso(stroop.subjects)}

    print("[2/3] 正在运行 15 个外层参与者 × 5 个种子。")
    for subject in range(1, 16):
        train_index, test_index = outer[subject]
        source = stroop.subset(train_index)
        target = stroop.subset(test_index)
        for seed in SEEDS:
            if checkpoint_complete(subject, seed):
                print(f"[skip] Stroop LOSO outer=S{subject:02d} seed={seed}")
                continue
            print(f"[start] Stroop LOSO outer=S{subject:02d} seed={seed}")
            result = fit_outer_cell(
                source,
                target,
                outer_subject=subject,
                seed=seed,
                protocol=protocol,
                device=device,
            )
            target_rows = target_prediction_rows(
                target,
                result.test,
                subject=subject,
                seed=seed,
            )
            target_diagnostics = target_diagnostic(
                source,
                target,
                result,
                subject=subject,
                seed=seed,
                route_protocol=route_protocol,
            )
            meta_rows = meta_diagnostic_rows(
                source,
                result,
                outer_subject=subject,
                seed=seed,
            )
            meta_recording = meta_recording_rows(
                source,
                result,
                outer_subject=subject,
                seed=seed,
            )
            write_checkpoint(
                subject,
                seed,
                target_rows,
                target_diagnostics,
                meta_rows,
                meta_recording,
            )
            print(f"[done] Stroop LOSO outer=S{subject:02d} seed={seed}")

    print("[3/3] 正在合并检查点并生成哈希。")
    consolidate()
    print(f"Stroop LOSO 核心模型完成：{OUTPUT_ROOT}")
    print("下一步：把最后的控制台输出发给我，并保留编号 18 的整个目录。")


if __name__ == "__main__":
    main()

