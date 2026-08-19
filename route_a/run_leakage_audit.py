from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from revision_pipeline.aggregation import sha256_file  # noqa: E402
from revision_pipeline.splits import Split, balanced_recording_folds  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402


def recording_accuracy(probability: np.ndarray, labels: np.ndarray, recordings: np.ndarray) -> float:
    outcomes = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        recording_labels = np.unique(labels[mask])
        if recording_labels.size != 1:
            raise ValueError("Recording labels are inconsistent")
        aggregate = np.exp(np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0))
        outcomes.append(int(int(np.argmax(aggregate)) == int(recording_labels[0])))
    return float(np.mean(outcomes))


def evaluate_split(
    bundle: FeatureBundle,
    split: Split,
    *,
    strategy: str,
    subject: int,
    seed: int,
    fold: int,
) -> dict:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=int(seed) + int(fold),
        ),
    )
    model.fit(bundle.features[split.train], bundle.labels[split.train])
    probability = model.predict_proba(bundle.features[split.validation])
    prediction = np.argmax(probability, axis=1)
    train_recordings = set(bundle.recordings[split.train])
    validation_recordings = set(bundle.recordings[split.validation])
    overlap = train_recordings & validation_recordings
    return {
        "task": bundle.task,
        "subject": int(subject),
        "seed": int(seed),
        "fold": int(fold),
        "strategy": strategy,
        "train_windows": int(len(split.train)),
        "validation_windows": int(len(split.validation)),
        "train_recordings": int(len(train_recordings)),
        "validation_recordings": int(len(validation_recordings)),
        "overlapping_recordings": int(len(overlap)),
        "recording_overlap_rate": float(len(overlap) / max(1, len(validation_recordings))),
        "adjacent_window_raw_overlap": float(8.0 / 8.5),
        "validation_window_accuracy": float(accuracy_score(bundle.labels[split.validation], prediction)),
        "validation_window_log_loss": float(
            log_loss(bundle.labels[split.validation], probability, labels=[0, 1])
        ),
        "validation_recording_accuracy": recording_accuracy(
            probability,
            bundle.labels[split.validation],
            bundle.recordings[split.validation],
        ),
    }


def audit_bundle(bundle: FeatureBundle, subjects: list[int], seeds: list[int]) -> list[dict]:
    rows = []
    for subject in subjects:
        subset = bundle.subset(np.flatnonzero(bundle.subjects == subject))
        for seed in seeds:
            invalid = StratifiedShuffleSplit(n_splits=2, test_size=0.12, random_state=int(seed))
            for fold, (train, validation) in enumerate(invalid.split(subset.features, subset.labels), start=1):
                rows.append(
                    evaluate_split(
                        subset,
                        Split(train=np.asarray(train), validation=np.asarray(validation)),
                        strategy="invalid_random_window_88_12",
                        subject=subject,
                        seed=seed,
                        fold=fold,
                    )
                )
            corrected = balanced_recording_folds(subset.labels_raw, subset.recordings, seed=seed)
            for fold, split in enumerate(corrected, start=1):
                rows.append(
                    evaluate_split(
                        subset,
                        split,
                        strategy="corrected_recording_disjoint",
                        subject=subject,
                        seed=seed,
                        fold=fold,
                    )
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-window leakage methodological audit")
    parser.add_argument("--repo-root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--smoke", action="store_true", help="Audit four participants and one seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads((HERE / "protocol_route_a.json").read_text(encoding="utf-8"))
    subjects = list(range(1, 5)) if args.smoke else list(range(1, 16))
    seeds = [int(protocol["seeds"][0])] if args.smoke else [int(value) for value in protocol["seeds"]]
    arithmetic, stroop = load_feature_bundles(args.repo_root.resolve(), torch.device("cpu"))
    rows = audit_bundle(arithmetic, subjects, seeds) + audit_bundle(stroop, subjects, seeds)
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["task", "strategy"], as_index=False)
        .agg(
            folds=("fold", "size"),
            mean_recording_overlap_rate=("recording_overlap_rate", "mean"),
            mean_window_accuracy=("validation_window_accuracy", "mean"),
            mean_recording_accuracy=("validation_recording_accuracy", "mean"),
            mean_window_log_loss=("validation_window_log_loss", "mean"),
        )
        .sort_values(["task", "strategy"])
    )
    output = HERE / ("results_smoke" if args.smoke else "results") / "leakage_audit"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "leakage_audit_folds.csv", index=False, lineterminator="\n")
    summary.to_csv(output / "leakage_audit_summary.csv", index=False, lineterminator="\n")
    metadata = {
        "protocol_route_a_sha256": sha256_file(HERE / "protocol_route_a.json"),
        "purpose": "methodological audit only; not used for model or hyperparameter selection",
        "smoke": bool(args.smoke),
        "rows": int(len(frame)),
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Wrote leakage audit to {output}")


if __name__ == "__main__":
    main()
