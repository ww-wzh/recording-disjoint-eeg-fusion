from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
)
from revision_pipeline.models import _make_forest, fit_mlp  # noqa: E402
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.splits import (  # noqa: E402
    Split,
    balanced_recording_folds,
    outer_loso,
    subject_loso_folds,
)
from route_a_lib.data import FeatureBundle, load_feature_bundles, normalized_frobenius_shift  # noqa: E402
from route_a_lib.probability import (  # noqa: E402
    BASE_METHODS,
    DasfDecision,
    apply_clean_dasf,
    clean_dasf_decision,
    crossfit_window_stacker,
)


RAW_ID_KEYS = [
    "dataset",
    "protocol",
    "subject",
    "direction",
    "recording_id",
    "window_id",
    "seed",
    "true_label",
]


def fit_base_oof(
    source: FeatureBundle,
    splits: list[Split],
    model_config: dict[str, dict],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[int]]:
    n_rows = len(source.labels)
    oof = {method: np.full((n_rows, 2), np.nan, dtype=np.float64) for method in BASE_METHODS}
    selected_epochs = []
    for fold_number, split in enumerate(splits):
        fold_seed = int(seed) + 1009 * (fold_number + 1)
        mlp, selected_epoch = fit_mlp(
            source.features[split.train],
            source.labels[split.train],
            model_config["mlp"],
            fold_seed,
            device,
            x_validation=source.features[split.validation],
            y_validation=source.labels[split.validation],
            validation_recordings=source.recordings[split.validation],
        )
        selected_epochs.append(int(selected_epoch))
        oof["always_nn"][split.validation] = mlp.predict_proba(source.features[split.validation])
        for offset, method in enumerate(("rf", "extra_trees"), start=1):
            forest = _make_forest(method, model_config[method], fold_seed + offset * 37)
            forest.fit(source.features[split.train], source.labels[split.train])
            oof[method][split.validation] = forest.predict_proba(source.features[split.validation])
        del mlp
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    [inner done] fold={fold_number + 1}/{len(splits)} epoch={selected_epoch}")
    for method, probability in oof.items():
        if not np.isfinite(probability).all():
            raise AssertionError(f"Base OOF predictions are incomplete for {method}")
    return oof, selected_epochs


def recording_accuracy(probability: np.ndarray, labels: np.ndarray, recordings: np.ndarray) -> float:
    correct = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        values = np.unique(labels[mask])
        if values.size != 1:
            raise ValueError(f"Recording {recording!r} has inconsistent labels")
        aggregate = np.exp(np.mean(np.log(np.clip(probability[mask], 1e-12, 1.0)), axis=0))
        aggregate = aggregate / aggregate.sum()
        correct.append(int(int(np.argmax(aggregate)) == int(values[0])))
    return float(np.mean(correct))


def select_frozen_rows(
    raw_seed: pd.DataFrame,
    *,
    protocol: str,
    subject: int,
    direction: str,
    seed: int,
    decision: DasfDecision,
) -> pd.DataFrame:
    cell = raw_seed[
        (raw_seed["protocol"] == protocol)
        & (raw_seed["subject"].astype(int) == int(subject))
        & (raw_seed["direction"] == direction)
        & (raw_seed["seed"].astype(int) == int(seed))
        & raw_seed["method"].isin(["always_nn", "always_fuse"])
    ].copy()
    nn = cell[cell["method"] == "always_nn"].sort_values(RAW_ID_KEYS).reset_index(drop=True)
    fuse = cell[cell["method"] == "always_fuse"].sort_values(RAW_ID_KEYS).reset_index(drop=True)
    if len(nn) == 0 or len(nn) != len(fuse):
        raise ValueError("Frozen NN and fusion inputs do not cover the same target cell")
    if not nn[RAW_ID_KEYS].equals(fuse[RAW_ID_KEYS]):
        raise ValueError("Frozen NN and fusion input row identities differ")
    selected = fuse if decision.use_fusion else nn
    chosen = apply_clean_dasf(
        nn[["p0", "p1"]].to_numpy(dtype=np.float64),
        fuse[["p0", "p1"]].to_numpy(dtype=np.float64),
        decision,
    )
    # The synthetic decision above tests the selector branch; the real decision was
    # already made outside this function. Output values must remain byte-for-byte
    # equal to one frozen input after CSV parsing.
    if not np.array_equal(chosen, selected[["p0", "p1"]].to_numpy(dtype=np.float64)):
        raise AssertionError("Clean DASF did not return the selected frozen input exactly")
    out = selected.copy()
    out["method"] = "dasf_clean"
    return out


def run_cell(
    source: FeatureBundle,
    target: FeatureBundle,
    splits: list[Split],
    crossfit_groups: np.ndarray,
    raw_seed: pd.DataFrame,
    *,
    protocol_name: str,
    subject: int,
    direction: str,
    seed: int,
    parent_protocol: Protocol,
    route_protocol: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    config = {name: parent_protocol.model(name) for name in ("mlp", "rf", "extra_trees", "stacker")}
    base_oof, selected_epochs = fit_base_oof(source, splits, config, seed, device)
    fusion_oof = crossfit_window_stacker(
        base_oof,
        source.labels,
        source.recordings,
        crossfit_groups,
        c_value=float(config["stacker"]["C"]),
        seed=seed,
    )
    nn_accuracy = recording_accuracy(base_oof["always_nn"], source.labels, source.recordings)
    fusion_accuracy = recording_accuracy(fusion_oof, source.labels, source.recordings)
    shift = normalized_frobenius_shift(source.covariances, target.covariances)
    gate = route_protocol["dasf_clean"]
    decision = clean_dasf_decision(
        nn_accuracy,
        fusion_accuracy,
        shift,
        base_margin=float(gate["base_margin"]),
        alpha=float(gate["shift_alpha"]),
        max_margin=float(gate["max_margin"]),
    )
    output = select_frozen_rows(
        raw_seed,
        protocol=protocol_name,
        subject=subject,
        direction=direction,
        seed=seed,
        decision=decision,
    )
    diagnostic = {
        "protocol": protocol_name,
        "subject": int(subject),
        "direction": direction,
        "seed": int(seed),
        "validation_accuracy_nn": decision.validation_accuracy_nn,
        "validation_accuracy_fuse_meta_crossfit": decision.validation_accuracy_fuse,
        "validation_gain": decision.validation_gain,
        "covariance_shift": decision.shift,
        "effective_margin": decision.effective_margin,
        "used_fusion": bool(decision.use_fusion),
        "selected_input_method": "always_fuse" if decision.use_fusion else "always_nn",
        "inner_selected_epochs": "|".join(str(value) for value in selected_epochs),
        "validation_crossfit_groups": int(np.unique(crossfit_groups).size),
        "output_exactly_selected_input": True,
    }
    return output, diagnostic


def validate_raw_inputs_against_parent(raw_seed: pd.DataFrame, parent_recordings: pd.DataFrame) -> None:
    methods = ["always_nn", "always_fuse"]
    raw_base = raw_seed[raw_seed["method"].isin(methods)].copy()
    reconstructed = aggregate_recordings(
        median_seed_ensemble(raw_base, [1335, 1388, 1441, 1494, 1547])
    )
    frozen = parent_recordings[parent_recordings["method"].isin(methods)].copy()
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "method"]
    left = reconstructed.sort_values(keys).reset_index(drop=True)
    right = frozen.sort_values(keys).reset_index(drop=True)
    if not left[keys + ["true_label"]].equals(right[keys + ["true_label"]]):
        raise RuntimeError("Raw seed inputs do not reconstruct the parent recording identities")
    if not np.allclose(
        left[["p0", "p1"]].to_numpy(dtype=np.float64),
        right[["p0", "p1"]].to_numpy(dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("Raw seed inputs do not reconstruct the parent frozen probabilities")


def checkpoint_complete(path: Path, seed: int) -> bool:
    if not path.exists() or not path.with_suffix(".diagnostic.json").exists():
        return False
    frame = pd.read_csv(path)
    return bool(not frame.empty and set(frame["method"]) == {"dasf_clean"} and set(frame["seed"]) == {seed})


def consolidate(root: Path, raw_seed_text: pd.DataFrame, route_protocol: dict) -> None:
    diagnostics = []
    for path in sorted((root / "checkpoints").glob("**/*.diagnostic.json")):
        diagnostics.append(json.loads(path.read_text(encoding="utf-8")))
    if diagnostics:
        seed_diagnostics = pd.DataFrame(diagnostics)
        seed_diagnostics.to_csv(root / "gate_diagnostics_seed.csv", index=False, lineterminator="\n")
        cell_keys = ["protocol", "subject", "direction"]
        ensemble_diagnostics = seed_diagnostics.groupby(cell_keys, as_index=False).agg(
            validation_accuracy_nn=("validation_accuracy_nn", "median"),
            validation_accuracy_fuse_meta_crossfit=("validation_accuracy_fuse_meta_crossfit", "median"),
            covariance_shift=("covariance_shift", "median"),
            seed_fusion_votes=("used_fusion", "sum"),
            seed_count=("seed", "nunique"),
        )
        if not ensemble_diagnostics["seed_count"].eq(5).all():
            raise ValueError("Every final DASF outer cell requires five seed diagnostics")
        gate = route_protocol["dasf_clean"]
        ensemble_diagnostics["validation_gain"] = (
            ensemble_diagnostics["validation_accuracy_fuse_meta_crossfit"]
            - ensemble_diagnostics["validation_accuracy_nn"]
        )
        ensemble_diagnostics["effective_margin"] = np.clip(
            float(gate["base_margin"])
            + float(gate["shift_alpha"]) * ensemble_diagnostics["covariance_shift"],
            0.0,
            float(gate["max_margin"]),
        )
        finite = np.isfinite(
            ensemble_diagnostics[
                [
                    "validation_accuracy_nn",
                    "validation_accuracy_fuse_meta_crossfit",
                    "covariance_shift",
                    "effective_margin",
                ]
            ]
        ).all(axis=1)
        ensemble_diagnostics["used_fusion"] = finite & (
            ensemble_diagnostics["validation_gain"] > ensemble_diagnostics["effective_margin"]
        )
        ensemble_diagnostics["selected_input_method"] = np.where(
            ensemble_diagnostics["used_fusion"], "always_fuse", "always_nn"
        )
        ensemble_diagnostics["gate_stage"] = "after_five_seed_median_ensemble"
        ensemble_diagnostics.to_csv(root / "gate_diagnostics.csv", index=False, lineterminator="\n")
        choice = ensemble_diagnostics.assign(
            source_method=ensemble_diagnostics["selected_input_method"],
            subject=ensemble_diagnostics["subject"].astype(int).astype(str),
        )[["protocol", "subject", "direction", "source_method"]]
        source = raw_seed_text[raw_seed_text["method"].isin(["always_nn", "always_fuse"])].rename(
            columns={"method": "source_method"}
        )
        frame = source.merge(
            choice,
            on=["protocol", "subject", "direction", "source_method"],
            how="inner",
            validate="many_to_one",
        )
        frame["method"] = "dasf_clean"
        frame = frame[raw_seed_text.columns]
        keys = RAW_ID_KEYS[:-1] + ["method", "true_label"]
        if frame.duplicated(keys).any():
            raise AssertionError("Reconstructed DASF output contains duplicate rows")
        frame.sort_values(keys).to_csv(
            root / "raw_seed_predictions.csv",
            index=False,
            lineterminator="\n",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route A clean-room DASF runner")
    parser.add_argument("--protocol", choices=("cross_task", "loso", "both"), default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--repo-root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--smoke", action="store_true", help="Four-participant structural run; never use in paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    route_protocol = json.loads((HERE / "protocol_route_a.json").read_text(encoding="utf-8"))
    parent_protocol = Protocol.load(REVISION_ROOT / "protocol.json")
    if parent_protocol.digest != route_protocol["parent_protocol_sha256"]:
        raise RuntimeError("Parent protocol digest changed; refusing to mix predictions")
    frozen_path = REVISION_ROOT / "frozen" / "predictions_recording.csv"
    if sha256_file(frozen_path) != route_protocol["parent_prediction_sha256"]:
        raise RuntimeError("Parent frozen prediction digest changed; refusing to run DASF")
    raw_seed_path = REVISION_ROOT / "results" / "raw_seed_predictions.csv"
    raw_seed = pd.read_csv(raw_seed_path)
    raw_seed_text = pd.read_csv(raw_seed_path, dtype=str, keep_default_na=False)
    validate_raw_inputs_against_parent(raw_seed, pd.read_csv(frozen_path))
    arithmetic, stroop = load_feature_bundles(args.repo_root.resolve(), torch.device("cpu"))
    cohort = list(range(1, 5)) if args.smoke else list(range(1, 16))
    if args.smoke:
        arithmetic = arithmetic.subset(np.flatnonzero(np.isin(arithmetic.subjects, cohort)))
        stroop = stroop.subset(np.flatnonzero(np.isin(stroop.subjects, cohort)))
    requested = [args.subject] if args.subject is not None else cohort
    seeds = [int(route_protocol["seeds"][0])] if args.smoke else [int(value) for value in route_protocol["seeds"]]
    root = HERE / ("results_smoke" if args.smoke else "results") / "dasf_clean"

    if args.protocol in ("cross_task", "both"):
        checkpoint_dir = root / "checkpoints" / "cross_task"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for subject in requested:
            pairs = [
                (stroop, arithmetic, "stroop_to_arithmetic"),
                (arithmetic, stroop, "arithmetic_to_stroop"),
            ]
            for source_all, target_all, direction in pairs:
                source = source_all.subset(np.flatnonzero(source_all.subjects == subject))
                target = target_all.subset(np.flatnonzero(target_all.subjects == subject))
                for seed in seeds:
                    path = checkpoint_dir / f"s{subject:02d}_{direction}_seed{seed}.csv"
                    if checkpoint_complete(path, seed):
                        print(f"[skip] {path.name}")
                        continue
                    splits = balanced_recording_folds(source.labels_raw, source.recordings, seed=seed)
                    output, diagnostic = run_cell(
                        source,
                        target,
                        splits,
                        source.recordings,
                        raw_seed,
                        protocol_name="cross_task",
                        subject=subject,
                        direction=direction,
                        seed=seed,
                        parent_protocol=parent_protocol,
                        route_protocol=route_protocol,
                        device=device,
                    )
                    output.to_csv(path, index=False, lineterminator="\n")
                    path.with_suffix(".diagnostic.json").write_text(
                        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    print(f"[done] clean DASF cross_task S{subject:02d} {direction} seed={seed}")

    if args.protocol in ("loso", "both"):
        checkpoint_dir = root / "checkpoints" / "loso"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        outer = {subject: (train, test) for subject, train, test in outer_loso(arithmetic.subjects)}
        for subject in requested:
            train_index, test_index = outer[subject]
            source = arithmetic.subset(train_index)
            target = arithmetic.subset(test_index)
            for seed in seeds:
                path = checkpoint_dir / f"s{subject:02d}_arithmetic_seed{seed}.csv"
                if checkpoint_complete(path, seed):
                    print(f"[skip] {path.name}")
                    continue
                splits = subject_loso_folds(source.subjects)
                output, diagnostic = run_cell(
                    source,
                    target,
                    splits,
                    source.subjects,
                    raw_seed,
                    protocol_name="loso",
                    subject=subject,
                    direction="arithmetic",
                    seed=seed,
                    parent_protocol=parent_protocol,
                    route_protocol=route_protocol,
                    device=device,
                )
                output.to_csv(path, index=False, lineterminator="\n")
                path.with_suffix(".diagnostic.json").write_text(
                    json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                print(f"[done] clean DASF LOSO S{subject:02d} seed={seed}")
    consolidate(root, raw_seed_text, route_protocol)
    print(f"Completed available clean DASF checkpoints under {root}")


if __name__ == "__main__":
    main()
