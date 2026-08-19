"""Internal engine for the formal nested-LOSO CB-SF completion.

Users should run ``run_cbsf_loso.py`` rather than launching this module directly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
REPO_ROOT = REVISION_ROOT
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from build_outputs import paired_comparisons, subject_metrics, summary_table, tail_descriptives  # noqa: E402
from freeze_predictions import final_diagnostics, paired_window_rows  # noqa: E402
from revision_pipeline.aggregation import (  # noqa: E402
    RAW_KEYS,
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from revision_pipeline.models import _make_forest, fit_mlp  # noqa: E402
from revision_pipeline.protocol import Protocol  # noqa: E402
from revision_pipeline.risk_gate import (  # noqa: E402
    GATE_FEATURES,
    blend_probabilities,
    fit_predict_gate,
    label_free_diagnostics,
)
from revision_pipeline.splits import outer_loso, subject_loso_folds  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402
from route_a_lib.cbsf_audit import (  # noqa: E402
    EXPECTED_META_METHODS,
    EXPECTED_RECORDINGS_PER_SUBJECT,
    EXPECTED_SUBJECTS,
    SEEDS,
    make_gate_training_rows,
    validate_final_cbsf_recordings,
)
from route_a_lib.probability import BASE_METHODS, crossfit_window_stacker  # noqa: E402


PROTOCOL_PATH = HERE / "protocol_cbsf_loso.json"
ROUTE_PROTOCOL_PATH = HERE / "protocol_route_a.json"
PARENT_PROTOCOL_PATH = REVISION_ROOT / "protocol.json"
PARENT_RAW_PATH = REVISION_ROOT / "results" / "raw_seed_predictions.csv"
PARENT_DIAGNOSTIC_PATH = REVISION_ROOT / "results" / "diagnostics_seed.csv"
PARENT_RECORDING_PATH = REVISION_ROOT / "frozen" / "predictions_recording.csv"
DASF_DIAGNOSTIC_ROOT = HERE / "results" / "dasf_clean" / "checkpoints" / "loso"
RESULT_ROOT = HERE / "results" / "cbsf_loso"


def _combined_digest(rows: list[dict[str, str]]) -> str:
    canonical = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_frozen_inputs() -> tuple[dict, Protocol, str]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent_protocol = Protocol.load(PARENT_PROTOCOL_PATH)
    checks = {
        "parent_route_protocol_sha256": sha256_file(ROUTE_PROTOCOL_PATH),
        "parent_corrected_protocol_sha256": parent_protocol.digest,
        "parent_recording_prediction_sha256": sha256_file(PARENT_RECORDING_PATH),
        "parent_raw_seed_prediction_sha256": sha256_file(PARENT_RAW_PATH),
        "parent_diagnostic_sha256": sha256_file(PARENT_DIAGNOSTIC_PATH),
    }
    for key, observed in checks.items():
        if protocol[key] != observed:
            raise RuntimeError(f"Frozen input changed: {key} expected={protocol[key]} observed={observed}")
    if [int(value) for value in protocol["seeds"]] != SEEDS:
        raise RuntimeError("CB-SF LOSO seed list changed")

    epoch_sources = []
    for outer_subject in range(1, 16):
        for seed in SEEDS:
            path = DASF_DIAGNOSTIC_ROOT / f"s{outer_subject:02d}_arithmetic_seed{seed}.diagnostic.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing clean-DASF epoch source: {path}")
            epoch_sources.append(
                {
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
            )
    epoch_source_digest = _combined_digest(epoch_sources)
    expected_epoch_digest = protocol["clean_dasf_loso_epoch_source_bundle_sha256"]
    if epoch_source_digest != expected_epoch_digest:
        raise RuntimeError(
            "Frozen clean-DASF epoch sources changed: "
            f"expected={expected_epoch_digest} observed={epoch_source_digest}"
        )
    return protocol, parent_protocol, epoch_source_digest


def selected_epochs(outer_subject: int, seed: int, expected_folds: int) -> tuple[list[int], Path]:
    path = DASF_DIAGNOSTIC_ROOT / f"s{outer_subject:02d}_arithmetic_seed{seed}.diagnostic.json"
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    epochs = [int(value) for value in str(diagnostic["inner_selected_epochs"]).split("|")]
    if len(epochs) != expected_folds or any(value < 1 for value in epochs):
        raise ValueError(f"Invalid selected epochs in {path}: {epochs}")
    if int(diagnostic["subject"]) != outer_subject or int(diagnostic["seed"]) != seed:
        raise ValueError(f"DASF epoch source identity mismatch: {path}")
    return epochs, path


def _load_fold_prediction(
    path: Path,
    validation_index: np.ndarray,
    selected_epoch: int,
    protocol_digest: str,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["protocol_digest"].item()) != protocol_digest:
            raise RuntimeError(f"Progress file belongs to another protocol: {path}")
        if int(saved["selected_epoch"].item()) != int(selected_epoch):
            raise RuntimeError(f"Progress epoch differs from frozen selected epoch: {path}")
        if not np.array_equal(saved["validation_index"], validation_index):
            raise RuntimeError(f"Progress validation indices changed: {path}")
        result = {method: saved[method].astype(np.float64) for method in BASE_METHODS}
    if any(values.shape != (len(validation_index), 2) for values in result.values()):
        raise RuntimeError(f"Progress probability shape is invalid: {path}")
    if any(not np.isfinite(values).all() for values in result.values()):
        raise RuntimeError(f"Progress probabilities are non-finite: {path}")
    return result


def reconstruct_base_oof(
    source: FeatureBundle,
    outer_subject: int,
    seed: int,
    epochs: list[int],
    parent_protocol: Protocol,
    protocol_digest: str,
    device: torch.device,
    fold_root: Path,
) -> dict[str, np.ndarray]:
    splits = subject_loso_folds(source.subjects)
    if len(splits) != len(epochs):
        raise ValueError("Frozen epoch count does not match current inner folds")
    configs = {name: parent_protocol.model(name) for name in ("mlp", "rf", "extra_trees")}
    oof = {method: np.full((len(source.labels), 2), np.nan, dtype=np.float64) for method in BASE_METHODS}
    fold_root.mkdir(parents=True, exist_ok=True)

    for fold_number, (split, epoch) in enumerate(zip(splits, epochs)):
        held_subjects = np.unique(source.subjects[split.validation])
        if held_subjects.size != 1:
            raise AssertionError("An inner LOSO fold must hold exactly one participant")
        fold_path = fold_root / f"fold_{fold_number:02d}_held_s{int(held_subjects[0]):02d}.npz"
        prediction = _load_fold_prediction(
            fold_path,
            split.validation,
            epoch,
            protocol_digest,
        )
        if prediction is None:
            fold_seed = int(seed) + 1009 * (fold_number + 1)
            mlp, _ = fit_mlp(
                source.features[split.train],
                source.labels[split.train],
                configs["mlp"],
                fold_seed,
                device,
                fixed_epochs=int(epoch),
            )
            prediction = {
                "always_nn": mlp.predict_proba(source.features[split.validation]),
            }
            for offset, method in enumerate(("rf", "extra_trees"), start=1):
                forest = _make_forest(method, configs[method], fold_seed + offset * 37)
                forest.fit(source.features[split.train], source.labels[split.train])
                prediction[method] = forest.predict_proba(source.features[split.validation])
            np.savez_compressed(
                fold_path,
                protocol_digest=np.asarray(protocol_digest),
                outer_subject=np.int64(outer_subject),
                seed=np.int64(seed),
                held_subject=np.int64(held_subjects[0]),
                selected_epoch=np.int64(epoch),
                validation_index=split.validation,
                **prediction,
            )
            del mlp
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"    [inner done] outer=S{outer_subject:02d} seed={seed} "
                f"fold={fold_number + 1:02d}/{len(splits)} held=S{int(held_subjects[0]):02d} epoch={epoch}"
            )
        else:
            print(
                f"    [inner resume] outer=S{outer_subject:02d} seed={seed} "
                f"fold={fold_number + 1:02d}/{len(splits)} held=S{int(held_subjects[0]):02d}"
            )
        for method in BASE_METHODS:
            oof[method][split.validation] = prediction[method]

    for method, values in oof.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"Reconstructed OOF predictions are incomplete for {method}")
    return oof


def _recording_probability_rows(
    source: FeatureBundle,
    outer_subject: int,
    seed: int,
    nn_probability: np.ndarray,
    fusion_probability: np.ndarray,
) -> list[dict]:
    rows = []
    for meta_subject in np.unique(source.subjects):
        subject_mask = source.subjects == meta_subject
        for method, probability in (
            ("always_nn", nn_probability),
            ("always_fuse", fusion_probability),
        ):
            for recording in np.unique(source.recordings[subject_mask]):
                mask = subject_mask & (source.recordings == recording)
                labels = np.unique(source.labels[mask])
                if labels.size != 1:
                    raise ValueError(f"Recording {recording!r} contains inconsistent labels")
                values = np.clip(probability[mask], 1e-12, 1.0)
                aggregate = np.exp(np.mean(np.log(values), axis=0))
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
    return rows


def _meta_diagnostic_rows(
    source: FeatureBundle,
    outer_subject: int,
    seed: int,
    nn_probability: np.ndarray,
    fusion_probability: np.ndarray,
) -> list[dict]:
    rows = []
    for meta_subject in np.unique(source.subjects):
        target = source.subjects == meta_subject
        reference = ~target
        diagnostics = label_free_diagnostics(
            nn_probability[target],
            fusion_probability[target],
            source.features[reference],
            source.features[target],
        )
        diagnostics.update(
            {
                "outer_subject": int(outer_subject),
                "meta_subject": int(meta_subject),
                "seed": int(seed),
            }
        )
        rows.append(diagnostics)
    return rows


def checkpoint_paths(outer_subject: int, seed: int) -> tuple[Path, Path, Path]:
    root = RESULT_ROOT / "checkpoints"
    stem = root / f"s{outer_subject:02d}_seed{seed}"
    return (
        stem.with_suffix(".meta_predictions.csv"),
        stem.with_suffix(".meta_diagnostics.csv"),
        stem.with_suffix(".manifest.json"),
    )


def checkpoint_complete(outer_subject: int, seed: int, protocol_digest: str) -> bool:
    prediction_path, diagnostic_path, manifest_path = checkpoint_paths(outer_subject, seed)
    if not (prediction_path.exists() and diagnostic_path.exists() and manifest_path.exists()):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != protocol_digest:
        raise RuntimeError(f"Checkpoint belongs to another protocol: {manifest_path}")
    expected_meta_subjects = EXPECTED_SUBJECTS - {int(outer_subject)}
    required_predictions = {
        "outer_subject", "meta_subject", "recording_id", "method", "seed",
        "true_label", "p0", "p1",
    }
    required_diagnostics = {"outer_subject", "meta_subject", "seed", *GATE_FEATURES}
    predictions = pd.read_csv(prediction_path)
    diagnostics = pd.read_csv(diagnostic_path)
    if required_predictions - set(predictions) or required_diagnostics - set(diagnostics):
        return False
    prediction_keys = ["outer_subject", "meta_subject", "recording_id", "method", "seed"]
    diagnostic_keys = ["outer_subject", "meta_subject", "seed"]
    probability = predictions[["p0", "p1"]].to_numpy(dtype=np.float64)
    recording_counts = predictions.groupby(["meta_subject", "method"])["recording_id"].nunique()
    expected_epoch_source = f"s{outer_subject:02d}_arithmetic_seed{seed}.diagnostic.json"
    return bool(
        len(predictions) == 14 * EXPECTED_RECORDINGS_PER_SUBJECT * 2
        and len(diagnostics) == 14
        and not predictions.duplicated(prediction_keys).any()
        and not diagnostics.duplicated(diagnostic_keys).any()
        and set(predictions["outer_subject"].astype(int)) == {outer_subject}
        and set(diagnostics["outer_subject"].astype(int)) == {outer_subject}
        and set(predictions["seed"].astype(int)) == {seed}
        and set(diagnostics["seed"].astype(int)) == {seed}
        and set(predictions["meta_subject"].astype(int)) == expected_meta_subjects
        and set(diagnostics["meta_subject"].astype(int)) == expected_meta_subjects
        and set(predictions["method"].astype(str)) == EXPECTED_META_METHODS
        and recording_counts.eq(EXPECTED_RECORDINGS_PER_SUBJECT).all()
        and set(predictions["true_label"].astype(int)).issubset({0, 1})
        and np.isfinite(probability).all()
        and (probability >= 0.0).all()
        and np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        and manifest.get("outer_subject") == outer_subject
        and manifest.get("seed") == seed
        and manifest.get("epoch_source") == expected_epoch_source
        and manifest.get("meta_subjects") == sorted(expected_meta_subjects)
        and len(manifest.get("selected_epochs", [])) == 14
        and manifest.get("meta_prediction_sha256") == sha256_file(prediction_path)
        and manifest.get("meta_diagnostic_sha256") == sha256_file(diagnostic_path)
        and manifest.get("stacker_crossfit_group") == "participant"
    )


def build_checkpoint(
    source: FeatureBundle,
    outer_subject: int,
    seed: int,
    parent_protocol: Protocol,
    protocol_digest: str,
    device: torch.device,
) -> None:
    prediction_path, diagnostic_path, manifest_path = checkpoint_paths(outer_subject, seed)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    epochs, epoch_source = selected_epochs(outer_subject, seed, len(np.unique(source.subjects)))
    base_oof = reconstruct_base_oof(
        source,
        outer_subject,
        seed,
        epochs,
        parent_protocol,
        protocol_digest,
        device,
        RESULT_ROOT / "fold_progress" / f"s{outer_subject:02d}_seed{seed}",
    )
    fusion_oof = crossfit_window_stacker(
        base_oof,
        source.labels,
        source.recordings,
        source.subjects,
        c_value=float(parent_protocol.model("stacker")["C"]),
        seed=seed,
    )
    predictions = pd.DataFrame(
        _recording_probability_rows(
            source,
            outer_subject,
            seed,
            base_oof["always_nn"],
            fusion_oof,
        )
    )
    diagnostics = pd.DataFrame(
        _meta_diagnostic_rows(
            source,
            outer_subject,
            seed,
            base_oof["always_nn"],
            fusion_oof,
        )
    )
    predictions.to_csv(prediction_path, index=False, lineterminator="\n")
    diagnostics.to_csv(diagnostic_path, index=False, lineterminator="\n")
    manifest = {
        "outer_subject": int(outer_subject),
        "seed": int(seed),
        "protocol_sha256": protocol_digest,
        "epoch_source": epoch_source.name,
        "epoch_source_sha256": sha256_file(epoch_source),
        "selected_epochs": epochs,
        "meta_subjects": sorted(int(value) for value in predictions["meta_subject"].unique()),
        "meta_prediction_sha256": sha256_file(prediction_path),
        "meta_diagnostic_sha256": sha256_file(diagnostic_path),
        "stacker_crossfit_group": "participant",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] CB-SF meta OOF outer=S{outer_subject:02d} seed={seed}")


def consolidate_checkpoints(protocol_digest: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_files = []
    diagnostic_files = []
    for outer_subject in range(1, 16):
        for seed in SEEDS:
            if not checkpoint_complete(outer_subject, seed, protocol_digest):
                raise RuntimeError(f"Incomplete CB-SF checkpoint outer={outer_subject} seed={seed}")
            prediction_path, diagnostic_path, _ = checkpoint_paths(outer_subject, seed)
            prediction_files.append(prediction_path)
            diagnostic_files.append(diagnostic_path)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in diagnostic_files], ignore_index=True)
    if len(predictions) != 15 * 5 * 14 * EXPECTED_RECORDINGS_PER_SUBJECT * 2:
        raise AssertionError("Consolidated CB-SF meta predictions have an unexpected row count")
    if len(diagnostics) != 15 * 5 * 14:
        raise AssertionError("Consolidated CB-SF meta diagnostics have an unexpected row count")
    predictions.to_csv(RESULT_ROOT / "meta_recording_seed.csv", index=False, lineterminator="\n")
    diagnostics.to_csv(RESULT_ROOT / "meta_diagnostics_seed.csv", index=False, lineterminator="\n")
    return predictions, diagnostics


def derive_final_cbsf(
    protocol: dict,
    parent_protocol: Protocol,
    gate_training: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(PARENT_RAW_PATH)
    diagnostics_seed = pd.read_csv(PARENT_DIAGNOSTIC_PATH)
    window_ensemble = median_seed_ensemble(raw, SEEDS)
    target_diagnostics = final_diagnostics(window_ensemble, diagnostics_seed)
    target_diagnostics = target_diagnostics[target_diagnostics["protocol"] == "loso"].copy()
    if len(target_diagnostics) != 15 or set(target_diagnostics["subject"].astype(int)) != EXPECTED_SUBJECTS:
        raise AssertionError("Final CB-SF target diagnostics must cover all 15 outer participants")
    config = protocol["risk_gate"]
    generated = []
    weights = []

    for target in target_diagnostics.sort_values("subject").itertuples(index=False):
        outer_subject = int(target.subject)
        training = gate_training[gate_training["outer_subject"] == outer_subject].copy()
        if outer_subject in set(training["meta_subject"].astype(int)):
            raise AssertionError("Outer target participant is present in gate training")
        gate = fit_predict_gate(
            training,
            pd.Series(target._asdict()),
            catastrophe_threshold=float(config["catastrophe_threshold"]),
            benefit_alpha=float(config["benefit_alpha"]),
            catastrophe_c=float(config["catastrophe_C"]),
            risk_penalty=float(config["risk_penalty"]),
            temperature=float(config["temperature"]),
            max_weight=float(config["max_weight"]),
        )
        paired = paired_window_rows(window_ensemble, "loso", outer_subject, str(target.direction))
        blended = blend_probabilities(
            paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64),
            paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64),
            gate.weight,
        )
        for index, row in paired.reset_index(drop=True).iterrows():
            generated.append(
                {
                    **{key: row[key] for key in RAW_KEYS if key != "method"},
                    "method": "risk_aware",
                    "true_label": int(row["true_label"]),
                    "p0": float(blended[index, 0]),
                    "p1": float(blended[index, 1]),
                    "ensemble_size": 5,
                }
            )
        weights.append(
            {
                "protocol": "loso",
                "subject": outer_subject,
                "direction": target.direction,
                "weight": gate.weight,
                "predicted_benefit": gate.predicted_benefit,
                "predicted_catastrophe": gate.predicted_catastrophe,
                "meta_training_participants": int(training["meta_subject"].nunique()),
                "outer_subject_excluded": True,
            }
        )

    windows = pd.DataFrame(generated)
    recordings = aggregate_recordings(windows)
    reference = pd.read_csv(PARENT_RECORDING_PATH)
    reference = reference[(reference["protocol"] == "loso") & (reference["method"] == "always_nn")]
    validate_final_cbsf_recordings(recordings, reference)
    if len(weights) != 15 or not pd.DataFrame(weights)["meta_training_participants"].eq(14).all():
        raise AssertionError("Final CB-SF must produce one strictly nested gate per outer participant")
    return windows, recordings, pd.DataFrame(weights)


def write_final_outputs(
    protocol: dict,
    epoch_source_digest: str,
    meta_predictions: pd.DataFrame,
    meta_diagnostics: pd.DataFrame,
    gate_training: pd.DataFrame,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    gate_training.to_csv(RESULT_ROOT / "gate_training_rows.csv", index=False, lineterminator="\n")
    windows.to_csv(RESULT_ROOT / "window_ensemble_predictions.csv", index=False, lineterminator="\n")
    recording_path = RESULT_ROOT / "recording_predictions.csv"
    recordings.to_csv(recording_path, index=False, lineterminator="\n")
    weights.to_csv(RESULT_ROOT / "gate_weights.csv", index=False, lineterminator="\n")

    parent = pd.read_csv(PARENT_RECORDING_PATH)
    comparison_frame = pd.concat(
        [
            parent[(parent["protocol"] == "loso") & (parent["method"] == "always_nn")],
            recordings,
        ],
        ignore_index=True,
    )
    metrics = subject_metrics(comparison_frame)
    summary_table(metrics).to_csv(RESULT_ROOT / "method_summary.csv", index=False, lineterminator="\n")
    paired_comparisons(metrics, margin=float(protocol["noninferiority_margin"])).to_csv(
        RESULT_ROOT / "paired_vs_always_nn.csv", index=False, lineterminator="\n"
    )
    tail_descriptives(metrics).to_csv(
        RESULT_ROOT / "tail_descriptives_exploratory.csv", index=False, lineterminator="\n"
    )

    training_counts = gate_training.groupby("outer_subject")["meta_subject"].nunique()
    audit = {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_route_protocol_sha256": sha256_file(ROUTE_PROTOCOL_PATH),
        "parent_raw_seed_prediction_sha256": sha256_file(PARENT_RAW_PATH),
        "epoch_source_manifest_sha256": epoch_source_digest,
        "meta_seed_rows": int(len(meta_predictions)),
        "meta_diagnostic_rows": int(len(meta_diagnostics)),
        "gate_training_rows": int(len(gate_training)),
        "gate_training_participants_per_outer": sorted(int(value) for value in training_counts.unique()),
        "outer_subject_excluded_from_every_gate": bool(weights["outer_subject_excluded"].all()),
        "five_seed_ensemble_before_gate": True,
        "actual_probability_blending": True,
        "final_recording_rows": int(len(recordings)),
        "recording_prediction_sha256": sha256_file(recording_path),
        "statistical_unit": "held-out participant",
        "seed_is_statistical_unit": False,
        "claim_scope": protocol["claim_scope"],
    }
    (RESULT_ROOT / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_formal() -> None:
    protocol, parent_protocol, epoch_source_digest = verify_frozen_inputs()
    protocol_digest = sha256_file(PROTOCOL_PATH)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal CB-SF LOSO requires CUDA, but torch.cuda.is_available() is false")
    arithmetic, _ = load_feature_bundles(REPO_ROOT, torch.device("cpu"))
    outer = {subject: (train, test) for subject, train, test in outer_loso(arithmetic.subjects)}
    metadata = {
        "protocol_sha256": protocol_digest,
        "epoch_source_manifest_sha256": epoch_source_digest,
        "device": str(device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "user_entrypoint": "run_cbsf_loso.py",
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for outer_subject in range(1, 16):
        train_index, _ = outer[outer_subject]
        source = arithmetic.subset(train_index)
        for seed in SEEDS:
            if checkpoint_complete(outer_subject, seed, protocol_digest):
                print(f"[skip] CB-SF outer=S{outer_subject:02d} seed={seed}")
                continue
            build_checkpoint(
                source,
                outer_subject,
                seed,
                parent_protocol,
                protocol_digest,
                device,
            )

    meta_predictions, meta_diagnostics = consolidate_checkpoints(protocol_digest)
    gate_training = make_gate_training_rows(meta_predictions, meta_diagnostics)
    windows, recordings, weights = derive_final_cbsf(protocol, parent_protocol, gate_training)
    write_final_outputs(
        protocol,
        epoch_source_digest,
        meta_predictions,
        meta_diagnostics,
        gate_training,
        windows,
        recordings,
        weights,
    )
    print(f"Completed formal nested-LOSO CB-SF outputs under {RESULT_ROOT}")
