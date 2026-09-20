from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from revision_pipeline.aggregation import sha256_file  # noqa: E402
from revision_pipeline.splits import (  # noqa: E402
    Split,
    balanced_recording_folds,
    outer_loso,
    subject_loso_folds,
)
from route_a_lib.data import RawBundle, load_raw_bundle  # noqa: E402
from route_a_lib.deep import build_model, fit_deep_model, parameter_count  # noqa: E402


def load_protocol() -> dict:
    return json.loads((HERE / "protocol_route_a.json").read_text(encoding="utf-8"))


def prediction_rows(
    bundle: RawBundle,
    probability: np.ndarray,
    *,
    protocol: str,
    subject: int,
    direction: str,
    method: str,
    seed: int,
) -> pd.DataFrame:
    if len(bundle.labels) != len(probability):
        raise ValueError("Deep prediction count does not match target windows")
    return pd.DataFrame(
        {
            "dataset": "openbci",
            "protocol": protocol,
            "subject": int(subject),
            "direction": direction,
            "recording_id": bundle.recordings,
            "window_id": bundle.window_ids,
            "method": method,
            "seed": int(seed),
            "true_label": bundle.labels,
            "p0": probability[:, 0],
            "p1": probability[:, 1],
        }
    )


def _read_progress(path: Path, protocol_digest: str) -> dict[str, int]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("protocol_sha256") != protocol_digest:
        raise RuntimeError(f"Stale progress file has a different protocol digest: {path}")
    return {str(key): int(value) for key, value in raw.get("inner_epochs", {}).items()}


def _write_progress(path: Path, protocol_digest: str, epochs: dict[str, int]) -> None:
    path.write_text(
        json.dumps(
            {"protocol_sha256": protocol_digest, "inner_epochs": epochs},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def nested_deep_predict(
    source: RawBundle,
    target: RawBundle,
    splits: list[Split],
    *,
    model_name: str,
    architecture_config: dict,
    training_config: dict,
    seed: int,
    device: torch.device,
    progress_path: Path,
    protocol_digest: str,
) -> tuple[np.ndarray, dict]:
    n_channels, n_samples = source.windows.shape[1:]
    if target.windows.shape[1:] != (n_channels, n_samples):
        raise ValueError("Source and target raw-window shapes differ")

    def factory():
        return build_model(model_name, n_channels, n_samples, architecture_config)

    epoch_by_fold = _read_progress(progress_path, protocol_digest)
    for fold_number, split in enumerate(splits):
        fold_key = f"fold_{fold_number:02d}"
        if fold_key in epoch_by_fold:
            print(f"    [resume] {fold_key} selected_epoch={epoch_by_fold[fold_key]}")
            continue
        fitted = fit_deep_model(
            factory,
            source.windows[split.train],
            source.labels[split.train],
            training_config,
            int(seed) + 1009 * (fold_number + 1),
            device,
            validation_windows=source.windows[split.validation],
            validation_labels=source.labels[split.validation],
            validation_recordings=source.recordings[split.validation],
        )
        epoch_by_fold[fold_key] = int(fitted.selected_epoch)
        _write_progress(progress_path, protocol_digest, epoch_by_fold)
        print(f"    [inner done] {fold_key} selected_epoch={fitted.selected_epoch}")
        del fitted
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(epoch_by_fold) != len(splits):
        raise AssertionError("Not every inner fold has a selected epoch")
    ordered_epochs = [epoch_by_fold[f"fold_{number:02d}"] for number in range(len(splits))]
    selected_epoch = max(1, int(np.rint(np.median(ordered_epochs))))
    final_model = fit_deep_model(
        factory,
        source.windows,
        source.labels,
        training_config,
        int(seed),
        device,
        fixed_epochs=selected_epoch,
    )
    probability = final_model.predict(
        target.windows,
        device,
        batch_size=int(training_config["batch_size"]),
    )
    diagnostics = {
        "selected_epoch": selected_epoch,
        "inner_selected_epochs": ordered_epochs,
        "parameter_count": int(final_model.parameter_count),
        "training_windows": int(len(source.labels)),
        "test_windows": int(len(target.labels)),
        "training_recordings": int(np.unique(source.recordings).size),
        "test_recordings": int(np.unique(target.recordings).size),
        "inner_folds": int(len(splits)),
    }
    del final_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability, diagnostics


def checkpoint_complete(path: Path, method: str, seed: int) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    return bool(
        not frame.empty
        and set(frame["method"]) == {method}
        and set(frame["seed"].astype(int)) == {int(seed)}
        and not frame.duplicated(["recording_id", "window_id", "method", "seed"]).any()
    )


def run_cross_task(
    repo_root: Path,
    cache_dir: Path,
    result_dir: Path,
    model_name: str,
    protocol: dict,
    device: torch.device,
    subjects: list[int],
    seeds: list[int],
) -> None:
    checkpoint_dir = result_dir / model_name / "cross_task"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(HERE / "protocol_route_a.json")
    for subject in subjects:
        arithmetic = load_raw_bundle(repo_root, "arithmetic", [subject], cache_dir)
        stroop = load_raw_bundle(repo_root, "stroop", [subject], cache_dir)
        pairs = [
            (stroop, arithmetic, "stroop_to_arithmetic"),
            (arithmetic, stroop, "arithmetic_to_stroop"),
        ]
        for source, target, direction in pairs:
            for seed in seeds:
                checkpoint = checkpoint_dir / f"s{subject:02d}_{direction}_seed{seed}.csv"
                diagnostic_path = checkpoint.with_suffix(".diagnostic.json")
                if checkpoint_complete(checkpoint, model_name, seed) and diagnostic_path.exists():
                    print(f"[skip] {checkpoint.name}")
                    continue
                splits = balanced_recording_folds(source.labels_raw, source.recordings, seed=seed)
                probability, diagnostic = nested_deep_predict(
                    source,
                    target,
                    splits,
                    model_name=model_name,
                    architecture_config=protocol[model_name],
                    training_config=protocol["deep_training"],
                    seed=seed,
                    device=device,
                    progress_path=checkpoint.with_suffix(".progress.json"),
                    protocol_digest=digest,
                )
                prediction_rows(
                    target,
                    probability,
                    protocol="cross_task",
                    subject=subject,
                    direction=direction,
                    method=model_name,
                    seed=seed,
                ).to_csv(checkpoint, index=False, lineterminator="\n")
                diagnostic.update(
                    {
                        "protocol": "cross_task",
                        "subject": subject,
                        "direction": direction,
                        "seed": seed,
                        "protocol_sha256": digest,
                    }
                )
                diagnostic_path.write_text(
                    json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                print(f"[done] {model_name} cross_task S{subject:02d} {direction} seed={seed}")
        del arithmetic, stroop


def run_loso(
    repo_root: Path,
    cache_dir: Path,
    result_dir: Path,
    model_name: str,
    protocol: dict,
    device: torch.device,
    cohort_subjects: list[int],
    held_subjects: list[int],
    seeds: list[int],
) -> None:
    dataset = load_raw_bundle(repo_root, "arithmetic", cohort_subjects, cache_dir)
    checkpoint_dir = result_dir / model_name / "loso"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(HERE / "protocol_route_a.json")
    outer = {subject: (train, test) for subject, train, test in outer_loso(dataset.subjects)}
    for subject in held_subjects:
        if subject not in outer:
            raise ValueError(f"Held participant {subject} is not in the loaded cohort")
        train_index, test_index = outer[subject]
        source = dataset.subset(train_index)
        target = dataset.subset(test_index)
        splits = subject_loso_folds(source.subjects)
        for seed in seeds:
            checkpoint = checkpoint_dir / f"s{subject:02d}_arithmetic_seed{seed}.csv"
            diagnostic_path = checkpoint.with_suffix(".diagnostic.json")
            if checkpoint_complete(checkpoint, model_name, seed) and diagnostic_path.exists():
                print(f"[skip] {checkpoint.name}")
                continue
            probability, diagnostic = nested_deep_predict(
                source,
                target,
                splits,
                model_name=model_name,
                architecture_config=protocol[model_name],
                training_config=protocol["deep_training"],
                seed=seed,
                device=device,
                progress_path=checkpoint.with_suffix(".progress.json"),
                protocol_digest=digest,
            )
            prediction_rows(
                target,
                probability,
                protocol="loso",
                subject=subject,
                direction="arithmetic",
                method=model_name,
                seed=seed,
            ).to_csv(checkpoint, index=False, lineterminator="\n")
            diagnostic.update(
                {
                    "protocol": "loso",
                    "subject": subject,
                    "direction": "arithmetic",
                    "seed": seed,
                    "protocol_sha256": digest,
                }
            )
            diagnostic_path.write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"[done] {model_name} LOSO S{subject:02d} seed={seed}")


def consolidate(result_dir: Path, model_name: str) -> None:
    files = sorted((result_dir / model_name).glob("**/*.csv"))
    if not files:
        return
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = frame.drop_duplicates(
        ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"],
        keep="last",
    ).sort_values(["protocol", "subject", "direction", "seed", "recording_id", "window_id"])
    frame.to_csv(result_dir / model_name / "raw_seed_predictions.csv", index=False, lineterminator="\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route A EEGNet/EEG-Conformer nested baseline runner")
    parser.add_argument("--model", choices=("eegnet", "eeg_conformer"), required=True)
    parser.add_argument("--protocol", choices=("cross_task", "loso", "both"), default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REVISION_ROOT,
        help="Repository root containing 1111.py and the public data entry point",
    )
    parser.add_argument("--subject", type=int, default=None, help="Run only this outer/target participant")
    parser.add_argument("--smoke", action="store_true", help="One seed and four participants; never use in paper")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = load_protocol()
    if args.smoke:
        protocol["deep_training"] = dict(protocol["deep_training"])
        protocol["deep_training"]["max_epochs"] = 3
        protocol["deep_training"]["patience"] = 2
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    all_subjects = list(range(1, 16))
    cohort = all_subjects[:4] if args.smoke else all_subjects
    requested = [args.subject] if args.subject is not None else cohort
    if any(subject not in cohort for subject in requested):
        raise ValueError(f"Requested participants {requested} are outside the active cohort {cohort}")
    seeds = [int(protocol["seeds"][0])] if args.smoke else [int(value) for value in protocol["seeds"]]
    result_dir = HERE / ("results_smoke" if args.smoke else "results") / "deep"
    cache_dir = HERE / "cache" / "raw_8ch_neutral_notch50_bandpass_0p5_55_v1"
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "protocol_sha256": sha256_file(HERE / "protocol_route_a.json"),
        "smoke": bool(args.smoke),
        "warning": "Smoke results must never enter the manuscript" if args.smoke else "",
    }
    model_for_count = build_model(args.model, 8, 2125, protocol[args.model])
    metadata["parameter_count_at_250hz"] = parameter_count(model_for_count)
    del model_for_count
    model_root = result_dir / args.model
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.protocol in ("cross_task", "both"):
        run_cross_task(
            args.repo_root.resolve(), cache_dir, result_dir, args.model, protocol, device, requested, seeds
        )
    if args.protocol in ("loso", "both"):
        run_loso(
            args.repo_root.resolve(), cache_dir, result_dir, args.model, protocol, device, cohort, requested, seeds
        )
    consolidate(result_dir, args.model)
    print(f"Completed available {args.model} checkpoints under {model_root}")


if __name__ == "__main__":
    main()
