from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from revision_pipeline.aggregation import (
    RAW_KEYS,
    aggregate_recordings,
    freeze_csv,
    median_seed_ensemble,
)
from revision_pipeline.protocol import Protocol
from revision_pipeline.risk_gate import GATE_FEATURES, blend_probabilities, fit_predict_gate


def entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(np.mean(-np.sum(values * np.log(values), axis=1)))


def final_diagnostics(window_ensemble: pd.DataFrame, diagnostics_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["protocol", "subject", "direction"]
    nn = window_ensemble[window_ensemble["method"] == "always_nn"].copy()
    fu = window_ensemble[window_ensemble["method"] == "always_fuse"].copy()
    join_keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id"]
    merged = nn[join_keys + ["p0", "p1"]].merge(
        fu[join_keys + ["p0", "p1"]], on=join_keys, suffixes=("_nn", "_fuse"), validate="one_to_one"
    )
    feature_shift = (
        diagnostics_seed.groupby(["protocol", "outer_subject", "direction"], as_index=False)["feature_shift"]
        .median()
        .rename(columns={"outer_subject": "subject"})
    )
    rows = []
    for values, group in merged.groupby(keys, sort=True):
        nn_prob = group[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64)
        fu_prob = group[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64)
        rows.append(
            {
                "protocol": values[0],
                "subject": int(values[1]),
                "direction": values[2],
                "disagreement": float(np.mean(np.abs(nn_prob[:, 1] - fu_prob[:, 1]))),
                "entropy_nn": entropy(nn_prob),
                "entropy_fuse": entropy(fu_prob),
                "confidence_nn": float(np.mean(np.max(nn_prob, axis=1))),
            }
        )
    return pd.DataFrame(rows).merge(feature_shift, on=keys, validate="one_to_one")


def method_accuracy(recordings: pd.DataFrame) -> pd.DataFrame:
    return (
        recordings.groupby(["protocol", "subject", "direction", "method"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )


def cross_task_gate_rows(recordings: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    accuracy = method_accuracy(recordings[recordings["protocol"] == "cross_task"])
    pivot = accuracy.pivot(index=["protocol", "subject", "direction"], columns="method", values="accuracy").reset_index()
    pivot["benefit"] = pivot["always_fuse"] - pivot["always_nn"]
    return diagnostics[diagnostics["protocol"] == "cross_task"].merge(
        pivot[["protocol", "subject", "direction", "benefit"]],
        on=["protocol", "subject", "direction"],
        validate="one_to_one",
    )


def loso_gate_training_rows(meta_seed_path: Path, meta_diag_path: Path, seeds: list[int]) -> dict[int, pd.DataFrame]:
    raw = pd.read_csv(meta_seed_path)
    expected = set(seeds)
    group_keys = ["outer_subject", "meta_subject", "recording_id", "method"]
    observed = raw.groupby(group_keys)["seed"].agg(lambda x: set(int(v) for v in x))
    if observed.map(lambda x: x != expected).any():
        raise ValueError("LOSO meta-recording predictions do not contain exactly the frozen five seeds")
    med = raw.groupby(group_keys + ["true_label"], as_index=False)[["p0", "p1"]].median()
    total = med[["p0", "p1"]].sum(axis=1)
    med["p0"] = med["p0"] / total
    med["p1"] = med["p1"] / total
    med["pred"] = (med["p1"] > med["p0"]).astype(int)
    med["correct"] = (med["pred"] == med["true_label"]).astype(int)

    diag_seed = pd.read_csv(meta_diag_path)
    diag = diag_seed.groupby(["outer_subject", "meta_subject"], as_index=False)[["feature_shift"]].median()
    outputs: dict[int, pd.DataFrame] = {}
    for outer_subject, outer in med.groupby("outer_subject"):
        accuracy = outer.groupby(["meta_subject", "method"], as_index=False)["correct"].mean()
        pivot = accuracy.pivot(index="meta_subject", columns="method", values="correct").reset_index()
        pivot["benefit"] = pivot["always_fuse"] - pivot["always_nn"]
        feature_rows = []
        for meta_subject, subject_rows in outer.groupby("meta_subject"):
            nn = subject_rows[subject_rows["method"] == "always_nn"].sort_values("recording_id")
            fu = subject_rows[subject_rows["method"] == "always_fuse"].sort_values("recording_id")
            if nn["recording_id"].tolist() != fu["recording_id"].tolist():
                raise ValueError("LOSO meta methods do not share recordings")
            nn_prob = nn[["p0", "p1"]].to_numpy()
            fu_prob = fu[["p0", "p1"]].to_numpy()
            feature_rows.append(
                {
                    "outer_subject": int(outer_subject),
                    "meta_subject": int(meta_subject),
                    "disagreement": float(np.mean(np.abs(nn_prob[:, 1] - fu_prob[:, 1]))),
                    "entropy_nn": entropy(nn_prob),
                    "entropy_fuse": entropy(fu_prob),
                    "confidence_nn": float(np.mean(np.max(nn_prob, axis=1))),
                }
            )
        frame = pd.DataFrame(feature_rows)
        frame = frame.merge(
            diag[diag["outer_subject"] == outer_subject],
            on=["outer_subject", "meta_subject"],
            validate="one_to_one",
        ).merge(pivot[["meta_subject", "benefit"]], on="meta_subject", validate="one_to_one")
        outputs[int(outer_subject)] = frame
    return outputs


def paired_window_rows(window_ensemble: pd.DataFrame, protocol_name: str, subject: int, direction: str) -> pd.DataFrame:
    subset = window_ensemble[
        (window_ensemble["protocol"] == protocol_name)
        & (window_ensemble["subject"] == subject)
        & (window_ensemble["direction"] == direction)
    ]
    nn = subset[subset["method"] == "always_nn"].copy()
    fu = subset[subset["method"] == "always_fuse"].copy()
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "true_label"]
    return nn[keys + ["p0", "p1", "ensemble_size"]].merge(
        fu[keys + ["p0", "p1"]], on=keys, suffixes=("_nn", "_fuse"), validate="one_to_one"
    )


def make_risk_aware_rows(
    window_ensemble: pd.DataFrame,
    base_recordings: pd.DataFrame,
    diagnostics: pd.DataFrame,
    protocol: Protocol,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = protocol.model("risk_gate")
    cross_rows = cross_task_gate_rows(base_recordings, diagnostics)
    generated = []
    weights = []
    for target in diagnostics.itertuples(index=False):
        target_series = pd.Series(target._asdict())
        if target.protocol == "cross_task":
            training = cross_rows[cross_rows["subject"] != target.subject]
        else:
            continue
        gate = fit_predict_gate(
            training,
            target_series,
            catastrophe_threshold=float(config["catastrophe_threshold"]),
            benefit_alpha=float(config["benefit_alpha"]),
            catastrophe_c=float(config["catastrophe_C"]),
            risk_penalty=float(config["risk_penalty"]),
            temperature=float(config["temperature"]),
            max_weight=float(config["max_weight"]),
        )
        paired = paired_window_rows(
            window_ensemble, target.protocol, int(target.subject), str(target.direction)
        )
        blended = blend_probabilities(
            paired[["p0_nn", "p1_nn"]].to_numpy(),
            paired[["p0_fuse", "p1_fuse"]].to_numpy(),
            gate.weight,
        )
        for idx, row in paired.reset_index(drop=True).iterrows():
            generated.append(
                {
                    **{key: row[key] for key in RAW_KEYS if key != "method"},
                    "method": "risk_aware",
                    "true_label": int(row["true_label"]),
                    "p0": float(blended[idx, 0]),
                    "p1": float(blended[idx, 1]),
                    "ensemble_size": 5,
                }
            )
        weights.append(
            {
                "protocol": target.protocol,
                "subject": int(target.subject),
                "direction": target.direction,
                "weight": gate.weight,
                "predicted_benefit": gate.predicted_benefit,
                "predicted_catastrophe": gate.predicted_catastrophe,
                "meta_training_rows": int(len(training)),
            }
        )
    return pd.DataFrame(generated), pd.DataFrame(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the one auditable frozen recording-prediction file")
    base = Path(__file__).resolve().parent
    parser.add_argument("--results", type=Path, default=base / "results")
    parser.add_argument("--output", type=Path, default=base / "frozen" / "predictions_recording.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = Protocol.load(Path(__file__).resolve().parent / "protocol.json")
    raw = pd.read_csv(args.results / "raw_seed_predictions.csv")
    diagnostics_seed = pd.read_csv(args.results / "diagnostics_seed.csv")
    window_ensemble = median_seed_ensemble(raw, protocol.seeds)
    base_recordings = aggregate_recordings(window_ensemble)
    diagnostics = final_diagnostics(window_ensemble, diagnostics_seed)
    risk_rows, weights = make_risk_aware_rows(
        window_ensemble, base_recordings, diagnostics, protocol, args.results
    )
    if risk_rows.empty:
        raise RuntimeError("No risk-aware rows were generated")
    final_windows = pd.concat([window_ensemble, risk_rows], ignore_index=True)
    final_recordings = aggregate_recordings(final_windows)
    output = freeze_csv(final_recordings, args.output, protocol.digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output.parent / "gate_weights.csv", index=False)
    diagnostics.to_csv(output.parent / "diagnostics_final_ensemble.csv", index=False)
    print(f"Frozen predictions: {output}")


if __name__ == "__main__":
    main()
