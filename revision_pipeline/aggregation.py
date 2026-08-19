from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_KEYS = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method"]
FINAL_KEYS = ["dataset", "protocol", "subject", "direction", "recording_id", "method"]


def _normalise_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected an (n, 2) probability matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    row_sum = values.sum(axis=1, keepdims=True)
    if (row_sum <= 0).any():
        raise ValueError("Probability rows must have positive mass")
    return values / row_sum


def validate_seed_predictions(frame: pd.DataFrame, expected_seeds: list[int]) -> None:
    required = set(RAW_KEYS + ["seed", "true_label", "p0", "p1"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Seed prediction file is missing columns: {missing}")
    if frame.duplicated(RAW_KEYS + ["seed"]).any():
        raise ValueError("Duplicate seed/window prediction rows")
    if not set(frame["true_label"].unique()).issubset({0, 1}):
        raise ValueError("true_label must be binary")
    expected = set(int(x) for x in expected_seeds)
    observed = frame.groupby(RAW_KEYS, dropna=False)["seed"].agg(lambda x: set(int(v) for v in x))
    bad = observed[observed.map(lambda x: x != expected)]
    if not bad.empty:
        raise ValueError(f"Every final window requires exactly the five frozen seeds; bad cells={len(bad)}")
    probs = _normalise_probabilities(frame[["p0", "p1"]].to_numpy())
    if not np.allclose(probs, frame[["p0", "p1"]].to_numpy(), atol=1e-6):
        raise ValueError("Input probabilities are not row-normalised")


def median_seed_ensemble(frame: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    validate_seed_predictions(frame, expected_seeds)
    metadata = frame.groupby(RAW_KEYS, as_index=False, dropna=False)["true_label"].agg(
        lambda x: int(np.unique(x)[0]) if np.unique(x).size == 1 else -1
    )
    if (metadata["true_label"] < 0).any():
        raise ValueError("A window has inconsistent labels across seeds")
    med = frame.groupby(RAW_KEYS, as_index=False, dropna=False)[["p0", "p1"]].median()
    out = metadata.merge(med, on=RAW_KEYS, validate="one_to_one")
    out[["p0", "p1"]] = _normalise_probabilities(out[["p0", "p1"]].to_numpy())
    out["ensemble_size"] = len(expected_seeds)
    return out


def aggregate_recordings(window_ensemble: pd.DataFrame) -> pd.DataFrame:
    required = set(RAW_KEYS + ["true_label", "p0", "p1", "ensemble_size"])
    missing = sorted(required - set(window_ensemble.columns))
    if missing:
        raise ValueError(f"Window ensemble is missing columns: {missing}")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        labels = group["true_label"].unique()
        if labels.size != 1:
            raise ValueError("A recording contains multiple true labels")
        probs = np.clip(group[["p0", "p1"]].to_numpy(dtype=np.float64), 1e-12, 1.0)
        geom = np.exp(np.mean(np.log(probs), axis=0))
        geom = geom / geom.sum()
        prediction = int(np.argmax(geom))
        return pd.Series(
            {
                "true_label": int(labels[0]),
                "p0": float(geom[0]),
                "p1": float(geom[1]),
                "pred_label": prediction,
                "correct": int(prediction == int(labels[0])),
                "n_windows": int(len(group)),
                "ensemble_size": int(group["ensemble_size"].iloc[0]),
            }
        )

    rows = []
    for keys, group in window_ensemble.groupby(FINAL_KEYS, sort=True, dropna=False):
        values = aggregate(group).to_dict()
        rows.append({**dict(zip(FINAL_KEYS, keys)), **values})
    final = pd.DataFrame(rows)
    if final.duplicated(FINAL_KEYS).any():
        raise AssertionError("Recording aggregation did not produce unique rows")
    return final.sort_values(FINAL_KEYS).reset_index(drop=True)


def validate_frozen_predictions(frame: pd.DataFrame) -> None:
    required = set(FINAL_KEYS + ["true_label", "p0", "p1", "pred_label", "correct", "ensemble_size"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen prediction file is missing columns: {missing}")
    forbidden = {"seed", "fold", "window_id"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"Frozen recording predictions cannot expose pseudo-replicates: {sorted(forbidden)}")
    if frame.duplicated(FINAL_KEYS).any():
        raise ValueError("Frozen prediction rows are not unique")
    if (frame["ensemble_size"] != 5).any():
        raise ValueError("Risk/statistical analysis is restricted to the final five-seed ensemble")

    cell_keys = ["dataset", "protocol", "subject", "direction", "recording_id"]
    method_sets = frame.groupby(cell_keys)["method"].agg(set)
    for (dataset, protocol), expected_rows in frame.groupby(["dataset", "protocol"]):
        expected = set(expected_rows["method"].unique())
        subset = method_sets.loc[(dataset, protocol)]
        if subset.map(lambda values: values != expected).any():
            raise ValueError(f"Methods are not evaluated on identical recordings in {dataset}/{protocol}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_csv(frame: pd.DataFrame, output: str | Path, protocol_digest: str) -> Path:
    validate_frozen_predictions(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n")
    manifest = {
        "prediction_file": output.name,
        "prediction_sha256": sha256_file(output),
        "protocol_sha256": protocol_digest,
        "rows": int(len(frame)),
        "statistical_unit": "held-out subject",
        "evaluation_unit": "recording",
        "ensemble_size": 5,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
