from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from eeg_channel_selection import PARSER_SCHEMA, select_openbci_eeg_channels
from revision_pipeline.models import NestedPrediction, nested_fit_predict
from revision_pipeline.aggregation import sha256_file
from revision_pipeline.protocol import Protocol
from revision_pipeline.risk_gate import label_free_diagnostics
from revision_pipeline.splits import balanced_recording_folds, binary_labels, outer_loso, subject_loso_folds


@dataclass
class Bundle:
    features: np.ndarray
    labels_raw: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray
    window_ids: np.ndarray
    task: str

    def subset(self, indices: np.ndarray) -> "Bundle":
        return Bundle(
            features=self.features[indices],
            labels_raw=self.labels_raw[indices],
            labels=self.labels[indices],
            subjects=self.subjects[indices],
            recordings=self.recordings[indices],
            window_ids=self.window_ids[indices],
            task=self.task,
        )


def load_legacy_module(repo_root: Path):
    path = repo_root / "1111.py"
    spec = importlib.util.spec_from_file_location("legacy_eeg_pipeline_readonly", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_openbci_eeg_only(
    repo_root: Path, device: torch.device
) -> tuple[Bundle, Bundle, dict[str, int], pd.DataFrame]:
    legacy = load_legacy_module(repo_root)
    original_loader = legacy.load_txt_eeg

    def eeg_only_loader(path: str, default_sfreq: float = legacy.DEFAULT_SFREQ):
        signal, sfreq = original_loader(path, default_sfreq=default_sfreq)
        eeg, report = select_openbci_eeg_channels(signal)
        if not report.packet_counter_removed:
            raise ValueError(f"Expected and failed to remove the OpenBCI packet counter in {path}")
        return eeg, sfreq

    legacy.load_txt_eeg = eeg_only_loader
    subjects = legacy.normalize_subjects(legacy.SUBJECTS)
    cache = repo_root / ".openbci_cache_eeg_only_recording_disjoint_v2_packet_counter_fixed"
    cache.mkdir(parents=True, exist_ok=True)
    feature_device = device if device.type == "cuda" else torch.device("cpu")

    manifest_rows = []

    def build(task: str) -> Bundle:
        records = legacy.build_records(subjects, task=task)
        for record in records:
            file_path = Path(record.file_path).resolve()
            try:
                public_path = str(file_path.relative_to(repo_root))
            except ValueError:
                public_path = file_path.name
            manifest_rows.append(
                {
                    "task": task,
                    "subject": int(record.subject_id),
                    "raw_level": int(record.label),
                    "binary_label": int(record.label >= 2),
                    "source_file": public_path,
                    "file_bytes": int(file_path.stat().st_size),
                    "source_sha256": sha256_file(file_path),
                    "included": True,
                    "exclusion_reason": "",
                }
            )
        dataset = legacy.EEGWindowDataset(
            records=records,
            win_sec=legacy.WIN_SEC,
            stride_sec=legacy.STRIDE_SEC,
            bands=legacy.BANDS,
            default_sfreq=legacy.DEFAULT_SFREQ,
            cache_dir=str(cache),
            device_for_feature=feature_device,
        )
        if int(dataset.cov_n_channels) != 8:
            raise AssertionError(f"EEG-only extraction failed: cov_n_channels={dataset.cov_n_channels}")
        segment = dataset.segment_ids.cpu().numpy().astype(np.int64)
        record_ids = np.asarray(
            [f"{task}:s{records[int(value)].subject_id:02d}:r{records[int(value)].label}" for value in segment],
            dtype=object,
        )
        window_counter: dict[object, int] = {}
        window_ids = []
        for record_id in record_ids:
            offset = window_counter.get(record_id, 0)
            window_ids.append(f"{record_id}:w{offset:05d}")
            window_counter[record_id] = offset + 1
        raw = dataset.labels.cpu().numpy().astype(np.int64)
        return Bundle(
            features=dataset.features.cpu().numpy().astype(np.float32),
            labels_raw=raw,
            labels=binary_labels(raw),
            subjects=dataset.subject_ids.cpu().numpy().astype(np.int64),
            recordings=record_ids,
            window_ids=np.asarray(window_ids, dtype=object),
            task=task,
        )

    arithmetic = build("arithmetic")
    stroop = build("stroop")
    if arithmetic.features.shape[1] != stroop.features.shape[1]:
        raise ValueError("Feature dimensions differ between tasks")
    metadata = {
        "feature_dimension": int(arithmetic.features.shape[1]),
        "eeg_channels": 8,
        "arithmetic_windows": int(len(arithmetic.labels)),
        "stroop_windows": int(len(stroop.labels)),
        "subjects": int(len(np.intersect1d(np.unique(arithmetic.subjects), np.unique(stroop.subjects)))),
        "parser_schema": PARSER_SCHEMA,
        "packet_counter_removed": True,
    }
    return arithmetic, stroop, metadata, pd.DataFrame(manifest_rows)


def model_config(protocol: Protocol) -> dict[str, dict]:
    return {name: protocol.model(name) for name in ("mlp", "rf", "extra_trees", "stacker")}


def recording_accuracy(probabilities: np.ndarray, labels: np.ndarray, recordings: np.ndarray) -> float:
    scores = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        y = np.unique(labels[mask])
        if y.size != 1:
            raise ValueError(f"Recording {recording!r} contains multiple labels")
        p = np.exp(np.mean(np.log(np.clip(probabilities[mask], 1e-12, 1.0)), axis=0))
        scores.append(int(int(np.argmax(p)) == int(y[0])))
    return float(np.mean(scores))


def recording_probability_rows(
    outer_subject: int,
    meta_subject: int,
    seed: int,
    source: Bundle,
    mask: np.ndarray,
    result: NestedPrediction,
) -> list[dict]:
    rows = []
    for method in ("always_nn", "always_fuse"):
        probabilities = result.oof[method][mask]
        labels = source.labels[mask]
        recordings = source.recordings[mask]
        for recording in np.unique(recordings):
            rec_mask = recordings == recording
            values = np.unique(labels[rec_mask])
            if values.size != 1:
                raise ValueError(f"Recording {recording!r} contains multiple labels")
            p = np.exp(np.mean(np.log(np.clip(probabilities[rec_mask], 1e-12, 1.0)), axis=0))
            p = p / p.sum()
            rows.append(
                {
                    "outer_subject": int(outer_subject),
                    "meta_subject": int(meta_subject),
                    "recording_id": str(recording),
                    "method": method,
                    "seed": int(seed),
                    "true_label": int(values[0]),
                    "p0": float(p[0]),
                    "p1": float(p[1]),
                }
            )
    return rows


def prediction_rows(
    dataset: str,
    protocol_name: str,
    subject: int,
    direction: str,
    seed: int,
    target: Bundle,
    result: NestedPrediction,
) -> list[dict]:
    rows = []
    for method, probabilities in result.test.items():
        for index in range(len(target.labels)):
            rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol_name,
                    "subject": int(subject),
                    "direction": direction,
                    "recording_id": str(target.recordings[index]),
                    "window_id": str(target.window_ids[index]),
                    "method": method,
                    "seed": int(seed),
                    "true_label": int(target.labels[index]),
                    "p0": float(probabilities[index, 0]),
                    "p1": float(probabilities[index, 1]),
                }
            )
    return rows


def diagnostic_row(
    protocol_name: str,
    subject: int,
    direction: str,
    seed: int,
    source: Bundle,
    target: Bundle,
    result: NestedPrediction,
) -> dict:
    values = label_free_diagnostics(
        result.test["always_nn"], result.test["always_fuse"], source.features, target.features
    )
    values.update(
        {
            "protocol": protocol_name,
            "outer_subject": int(subject),
            "direction": direction,
            "seed": int(seed),
            "selected_epoch": int(result.selected_epoch),
            "inner_best_epochs": "|".join(str(x) for x in result.inner_best_epochs),
        }
    )
    return values


def checkpoint_exists(path: Path) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    return not frame.empty and set(frame["method"].unique()) == {"always_nn", "rf", "extra_trees", "always_fuse"}


def run_cross_task(
    arithmetic: Bundle,
    stroop: Bundle,
    protocol: Protocol,
    output: Path,
    device: torch.device,
    subjects_limit: int | None,
) -> None:
    common = np.intersect1d(np.unique(arithmetic.subjects), np.unique(stroop.subjects))
    if subjects_limit is not None:
        common = common[:subjects_limit]
    configs = model_config(protocol)
    checkpoint_dir = output / "checkpoints" / "cross_task"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for subject in common:
        pairs = [(stroop, arithmetic, "stroop_to_arithmetic"), (arithmetic, stroop, "arithmetic_to_stroop")]
        for source_all, target_all, direction in pairs:
            source = source_all.subset(np.flatnonzero(source_all.subjects == subject))
            target = target_all.subset(np.flatnonzero(target_all.subjects == subject))
            for seed in protocol.seeds:
                checkpoint = checkpoint_dir / f"s{int(subject):02d}_{direction}_seed{seed}.csv"
                diagnostic = checkpoint.with_suffix(".diagnostic.json")
                if checkpoint_exists(checkpoint) and diagnostic.exists():
                    print(f"[skip] {checkpoint.name}")
                    continue
                folds = balanced_recording_folds(source.labels_raw, source.recordings, seed=seed)
                result = nested_fit_predict(
                    source.features,
                    source.labels,
                    source.recordings,
                    target.features,
                    folds,
                    configs,
                    seed,
                    device,
                )
                pd.DataFrame(
                    prediction_rows("openbci", "cross_task", int(subject), direction, seed, target, result)
                ).to_csv(checkpoint, index=False)
                diagnostic.write_text(
                    json.dumps(
                        diagnostic_row("cross_task", int(subject), direction, seed, source, target, result),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[done] cross_task subject={subject} direction={direction} seed={seed}")


def run_loso(
    dataset: Bundle,
    protocol: Protocol,
    output: Path,
    device: torch.device,
    subjects_limit: int | None,
) -> None:
    outer = list(outer_loso(dataset.subjects))
    if subjects_limit is not None:
        outer = outer[:subjects_limit]
    configs = model_config(protocol)
    checkpoint_dir = output / "checkpoints" / "loso"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for subject, train_index, test_index in outer:
        source = dataset.subset(train_index)
        target = dataset.subset(test_index)
        for seed in protocol.seeds:
            checkpoint = checkpoint_dir / f"s{subject:02d}_arithmetic_seed{seed}.csv"
            diagnostic = checkpoint.with_suffix(".diagnostic.json")
            meta_path = checkpoint.with_suffix(".meta.csv")
            meta_prediction_path = checkpoint.with_suffix(".meta_predictions.csv")
            if checkpoint_exists(checkpoint) and diagnostic.exists() and meta_path.exists() and meta_prediction_path.exists():
                print(f"[skip] {checkpoint.name}")
                continue
            folds = subject_loso_folds(source.subjects)
            result = nested_fit_predict(
                source.features,
                source.labels,
                source.recordings,
                target.features,
                folds,
                configs,
                seed,
                device,
            )
            pd.DataFrame(
                prediction_rows("openbci", "loso", subject, "arithmetic", seed, target, result)
            ).to_csv(checkpoint, index=False)
            diagnostic.write_text(
                json.dumps(
                    diagnostic_row("loso", subject, "arithmetic", seed, source, target, result),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            meta_rows = []
            meta_prediction_rows = []
            for meta_subject in np.unique(source.subjects):
                mask = source.subjects == meta_subject
                reference = ~mask
                diag = label_free_diagnostics(
                    result.oof["always_nn"][mask],
                    result.oof["always_fuse"][mask],
                    source.features[reference],
                    source.features[mask],
                )
                diag.update(
                    {
                        "outer_subject": subject,
                        "meta_subject": int(meta_subject),
                        "seed": int(seed),
                        "benefit": recording_accuracy(
                            result.oof["always_fuse"][mask], source.labels[mask], source.recordings[mask]
                        )
                        - recording_accuracy(
                            result.oof["always_nn"][mask], source.labels[mask], source.recordings[mask]
                        ),
                    }
                )
                meta_rows.append(diag)
                meta_prediction_rows.extend(
                    recording_probability_rows(subject, int(meta_subject), seed, source, mask, result)
                )
            pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
            pd.DataFrame(meta_prediction_rows).to_csv(meta_prediction_path, index=False)
            print(f"[done] loso outer_subject={subject} seed={seed}")


def consolidate(output: Path) -> None:
    prediction_files = sorted((output / "checkpoints").glob("**/*.csv"))
    prediction_files = [
        path for path in prediction_files
        if not path.name.endswith(".meta.csv") and not path.name.endswith(".meta_predictions.csv")
    ]
    if prediction_files:
        pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True).to_csv(
            output / "raw_seed_predictions.csv", index=False
        )
    diagnostics = []
    for path in sorted((output / "checkpoints").glob("**/*.diagnostic.json")):
        diagnostics.append(json.loads(path.read_text(encoding="utf-8")))
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(output / "diagnostics_seed.csv", index=False)
    meta_files = sorted((output / "checkpoints" / "loso").glob("*.meta.csv"))
    if meta_files:
        pd.concat([pd.read_csv(path) for path in meta_files], ignore_index=True).to_csv(
            output / "loso_meta_seed.csv", index=False
        )
    meta_prediction_files = sorted((output / "checkpoints" / "loso").glob("*.meta_predictions.csv"))
    if meta_prediction_files:
        pd.concat([pd.read_csv(path) for path in meta_prediction_files], ignore_index=True).to_csv(
            output / "loso_meta_recording_seed.csv", index=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-resistant EEG-only nested reanalysis")
    parser.add_argument("--protocol", choices=("cross_task", "loso", "both"), default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Use four subjects for a structural smoke run")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    default_name = "results_smoke" if args.smoke else "results"
    output = (args.output or (Path(__file__).resolve().parent / default_name)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = Protocol.load(Path(__file__).resolve().parent / "protocol.json")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    arithmetic, stroop, metadata, data_manifest = load_openbci_eeg_only(root, device)
    metadata.update(
        {
            "protocol_sha256": protocol.digest,
            "device": str(device),
            "smoke": bool(args.smoke),
            "warning": "Smoke outputs are structural checks and must never enter the manuscript" if args.smoke else "",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "legacy_feature_source_sha256": sha256_file(root / "1111.py"),
        }
    )
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_manifest.to_csv(output / "data_manifest.csv", index=False)
    subject_limit = 4 if args.smoke else None
    if args.protocol in ("cross_task", "both"):
        run_cross_task(arithmetic, stroop, protocol, output, device, subject_limit)
    if args.protocol in ("loso", "both"):
        run_loso(arithmetic, protocol, output, device, subject_limit)
    consolidate(output)
    print(f"Consolidated outputs: {output}")


if __name__ == "__main__":
    main()
