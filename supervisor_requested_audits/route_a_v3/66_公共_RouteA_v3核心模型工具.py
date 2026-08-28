"""编号66：Route A v3 核心模型重跑的公共工具。

该文件只服务于编号67--69三个可直接运行的脚本。它复用已经审计过的
纯8通道加载器和 nested_fit_predict 实现，但所有输出根目录、协议和任务
都由 v3 固定，不读取旧 route_a/results。
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
PROJECT_ROOT = REVISION_ROOT
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
V3_ROOT = HERE / "01_core"

sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.splits import (  # noqa: E402
    balanced_recording_folds,
    outer_loso,
    subject_loso_folds,
)


def load_policy_and_protocol() -> tuple[dict, Protocol]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    protocol = Protocol.load(POLICY_PATH)
    return policy, protocol


def load_core_module():
    path = REVISION_ROOT / "run_corrected_experiments.py"
    spec = importlib.util.spec_from_file_location("route_a_v3_core_engine", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import core engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def subset_bundle(bundle, *, subjects: set[int] | None = None, raw_labels: set[int] | None = None):
    mask = np.ones(len(bundle.labels), dtype=bool)
    if subjects is not None:
        mask &= np.isin(bundle.subjects, sorted(subjects))
    if raw_labels is not None:
        mask &= np.isin(bundle.labels_raw, sorted(raw_labels))
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"Filtering removed every row from {bundle.task}")
    return bundle.subset(indices)


def ensure_core_metadata(output: Path, policy: dict, protocol: Protocol, *, task: str, cohort: list[int], target_labels: list[int] | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol_version": policy["protocol_version"],
        "protocol_sha256": protocol.digest,
        "policy_file": POLICY_PATH.name,
        "task": task,
        "cohort": cohort,
        "target_raw_labels": target_labels,
        "parser_schema": policy["parser_schema"],
        "eeg_channels": policy["eeg_channels"],
        "window_seconds": policy["window_seconds"],
        "stride_seconds": policy["stride_seconds"],
        "device": str(choose_device()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warning": "This directory is the v3 rerun only; old route_a/results is never read.",
    }
    path = output / "run_metadata.json"
    if not path.exists():
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_cell(core, output: Path, source, target, *, protocol_name: str, subject: int, direction: str, seed: int, protocol: Protocol, device: torch.device) -> None:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"s{int(subject):02d}_{direction}_seed{int(seed)}.csv"
    diagnostic = checkpoint.with_suffix(".diagnostic.json")
    if core.checkpoint_exists(checkpoint) and diagnostic.exists():
        print(f"[skip] {checkpoint.name}")
        return
    folds = balanced_recording_folds(source.labels_raw, source.recordings, seed=int(seed))
    configs = core.model_config(protocol)
    result = core.nested_fit_predict(
        source.features,
        source.labels,
        source.recordings,
        target.features,
        folds,
        configs,
        int(seed),
        device,
    )
    pd.DataFrame(core.prediction_rows("openbci", protocol_name, int(subject), direction, int(seed), target, result)).to_csv(
        checkpoint, index=False, lineterminator="\n"
    )
    diagnostic.write_text(
        json.dumps(
            core.diagnostic_row(protocol_name, int(subject), direction, int(seed), source, target, result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] core {protocol_name} S{int(subject):02d} {direction} seed={int(seed)}")


def run_cross_task(core, arithmetic_source, stroop_source, arithmetic_target, stroop_target, *, output: Path, protocol: Protocol, device: torch.device) -> None:
    cohort = sorted(set(int(x) for x in np.unique(arithmetic_source.subjects)) & set(int(x) for x in np.unique(stroop_source.subjects)))
    for subject in cohort:
        _write_cell(
            core,
            output / "stroop_to_arithmetic",
            stroop_source.subset(np.flatnonzero(stroop_source.subjects == subject)),
            arithmetic_target.subset(np.flatnonzero(arithmetic_target.subjects == subject)),
            protocol_name="cross_task",
            subject=subject,
            direction="stroop_to_arithmetic",
            seed=protocol.seeds[0],
            protocol=protocol,
            device=device,
        )
        for seed in protocol.seeds[1:]:
            _write_cell(
                core,
                output / "stroop_to_arithmetic",
                stroop_source.subset(np.flatnonzero(stroop_source.subjects == subject)),
                arithmetic_target.subset(np.flatnonzero(arithmetic_target.subjects == subject)),
                protocol_name="cross_task",
                subject=subject,
                direction="stroop_to_arithmetic",
                seed=seed,
                protocol=protocol,
                device=device,
            )
        for seed in protocol.seeds:
            _write_cell(
                core,
                output / "arithmetic_to_stroop",
                arithmetic_source.subset(np.flatnonzero(arithmetic_source.subjects == subject)),
                stroop_target.subset(np.flatnonzero(stroop_target.subjects == subject)),
                protocol_name="cross_task",
                subject=subject,
                direction="arithmetic_to_stroop",
                seed=seed,
                protocol=protocol,
                device=device,
            )
    consolidate(output)


def run_loso_task(core, dataset, *, task: str, cohort: list[int], output: Path, protocol: Protocol, device: torch.device) -> None:
    output.mkdir(parents=True, exist_ok=True)
    outer = {subject: (train, test) for subject, train, test in outer_loso(dataset.subjects)}
    configs = core.model_config(protocol)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for subject in cohort:
        train_index, test_index = outer[int(subject)]
        source = dataset.subset(train_index)
        target = dataset.subset(test_index)
        folds = subject_loso_folds(source.subjects)
        for seed in protocol.seeds:
            checkpoint = checkpoint_dir / f"s{int(subject):02d}_{task}_seed{int(seed)}.csv"
            diagnostic = checkpoint.with_suffix(".diagnostic.json")
            meta_path = checkpoint.with_suffix(".meta.csv")
            meta_prediction_path = checkpoint.with_suffix(".meta_predictions.csv")
            if core.checkpoint_exists(checkpoint) and diagnostic.exists() and meta_path.exists() and meta_prediction_path.exists():
                print(f"[skip] {checkpoint.name}")
                continue
            result = core.nested_fit_predict(
                source.features,
                source.labels,
                source.recordings,
                target.features,
                folds,
                configs,
                int(seed),
                device,
            )
            pd.DataFrame(core.prediction_rows("openbci", "loso", int(subject), task, int(seed), target, result)).to_csv(
                checkpoint, index=False, lineterminator="\n"
            )
            diagnostic.write_text(
                json.dumps(core.diagnostic_row("loso", int(subject), task, int(seed), source, target, result), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            meta_rows = []
            meta_prediction_rows = []
            for meta_subject in np.unique(source.subjects):
                mask = source.subjects == meta_subject
                reference = ~mask
                diag = core.label_free_diagnostics(
                    result.oof["always_nn"][mask],
                    result.oof["always_fuse"][mask],
                    source.features[reference],
                    source.features[mask],
                )
                diag.update(
                    {
                        "outer_subject": int(subject),
                        "meta_subject": int(meta_subject),
                        "seed": int(seed),
                        "benefit": core.recording_accuracy(result.oof["always_fuse"][mask], source.labels[mask], source.recordings[mask])
                        - core.recording_accuracy(result.oof["always_nn"][mask], source.labels[mask], source.recordings[mask]),
                    }
                )
                meta_rows.append(diag)
                meta_prediction_rows.extend(core.recording_probability_rows(int(subject), int(meta_subject), int(seed), source, mask, result))
            pd.DataFrame(meta_rows).to_csv(meta_path, index=False, lineterminator="\n")
            pd.DataFrame(meta_prediction_rows).to_csv(meta_prediction_path, index=False, lineterminator="\n")
            print(f"[done] core loso {task} S{int(subject):02d} seed={int(seed)}")
    consolidate(output)


def consolidate(output: Path) -> None:
    checkpoint_files = sorted((output / "checkpoints").glob("**/*.csv"))
    prediction_files = [path for path in checkpoint_files if not path.name.endswith(".meta.csv") and not path.name.endswith(".meta_predictions.csv")]
    if prediction_files:
        frame = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
        frame.to_csv(output / "raw_seed_predictions.csv", index=False, lineterminator="\n")
    diagnostics = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output.glob("**/*.diagnostic.json"))]
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(output / "diagnostics_seed.csv", index=False, lineterminator="\n")
    meta_files = sorted(output.glob("**/*.meta.csv"))
    if meta_files:
        pd.concat([pd.read_csv(path) for path in meta_files], ignore_index=True).to_csv(output / "loso_meta_seed.csv", index=False, lineterminator="\n")
    meta_prediction_files = sorted(output.glob("**/*.meta_predictions.csv"))
    if meta_prediction_files:
        pd.concat([pd.read_csv(path) for path in meta_prediction_files], ignore_index=True).to_csv(output / "loso_meta_recording_seed.csv", index=False, lineterminator="\n")
