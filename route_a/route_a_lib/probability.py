from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


BASE_METHODS = ("always_nn", "rf", "extra_trees")
CELL_KEYS = ["dataset", "protocol", "subject", "direction", "recording_id"]


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected an (n, 2) probability matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    mass = values.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Probability rows must have positive mass")
    return values / mass


def blend_probabilities(nn: np.ndarray, fuse: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("Blend weight must lie in [0, 1]")
    nn = normalize_probabilities(nn)
    fuse = normalize_probabilities(fuse)
    if nn.shape != fuse.shape:
        raise ValueError("NN and fusion probability shapes differ")
    return normalize_probabilities((1.0 - float(weight)) * nn + float(weight) * fuse)


def logit_features(*probabilities: np.ndarray) -> np.ndarray:
    columns = []
    for values in probabilities:
        p1 = np.clip(normalize_probabilities(values)[:, 1], 1e-6, 1.0 - 1e-6)
        columns.append(np.log(p1 / (1.0 - p1)))
    return np.column_stack(columns)


def method_matrix(frame: pd.DataFrame, methods: tuple[str, ...] = BASE_METHODS) -> tuple[pd.DataFrame, np.ndarray]:
    required = set(CELL_KEYS + ["method", "true_label", "p0", "p1"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Recording predictions are missing columns: {missing}")
    subset = frame[frame["method"].isin(methods)].copy()
    key_labels = subset.groupby(CELL_KEYS, as_index=False)["true_label"].agg(
        lambda x: int(np.unique(x)[0]) if np.unique(x).size == 1 else -1
    )
    if (key_labels["true_label"] < 0).any():
        raise ValueError("A recording has inconsistent labels across methods")
    matrices = []
    for method in methods:
        part = subset[subset["method"] == method][CELL_KEYS + ["p0", "p1"]]
        if part.duplicated(CELL_KEYS).any():
            raise ValueError(f"Duplicate {method} recording rows")
        aligned = key_labels.merge(part, on=CELL_KEYS, how="left", validate="one_to_one")
        if aligned[["p0", "p1"]].isna().any().any():
            raise ValueError(f"Method {method} does not cover every recording")
        matrices.append(aligned[["p0", "p1"]].to_numpy(dtype=np.float64))
    return key_labels, logit_features(*matrices)


def fit_recording_stacker(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    c_value: float,
    seed: int,
    methods: tuple[str, ...] = BASE_METHODS,
) -> np.ndarray:
    train_keys, train_x = method_matrix(train, methods)
    test_keys, test_x = method_matrix(test, methods)
    if np.unique(train_keys["true_label"]).size != 2:
        raise ValueError("Recording stacker training data must contain both classes")
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=4000,
        random_state=int(seed),
    )
    model.fit(train_x, train_keys["true_label"].to_numpy(dtype=np.int64))
    probability = model.predict_proba(test_x)
    return normalize_probabilities(probability)


def recording_rows_from_probabilities(
    reference: pd.DataFrame,
    probabilities: np.ndarray,
    method: str,
    *,
    input_methods: tuple[str, ...] = BASE_METHODS,
) -> pd.DataFrame:
    keys, _ = method_matrix(reference, input_methods)
    probabilities = normalize_probabilities(probabilities)
    if len(keys) != len(probabilities):
        raise ValueError("Probability count does not match reference recordings")
    metadata_columns = CELL_KEYS + ["true_label"]
    out = keys[metadata_columns].copy()
    out["method"] = str(method)
    out["p0"] = probabilities[:, 0]
    out["p1"] = probabilities[:, 1]
    out["pred_label"] = np.argmax(probabilities, axis=1).astype(int)
    out["correct"] = (out["pred_label"].to_numpy() == out["true_label"].to_numpy()).astype(int)
    counts = reference[reference["method"] == "always_nn"][CELL_KEYS + ["n_windows", "ensemble_size"]]
    out = out.merge(counts, on=CELL_KEYS, how="left", validate="one_to_one")
    return out[
        CELL_KEYS
        + ["method", "true_label", "p0", "p1", "pred_label", "correct", "n_windows", "ensemble_size"]
    ]


def crossfit_window_stacker(
    base_probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    recordings: np.ndarray,
    crossfit_groups: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> np.ndarray:
    """Create meta-OOF fusion probabilities with no held group used to fit its stacker."""
    labels = np.asarray(labels, dtype=np.int64)
    recordings = np.asarray(recordings)
    crossfit_groups = np.asarray(crossfit_groups)
    if not (len(labels) == len(recordings) == len(crossfit_groups)):
        raise ValueError("Labels, recordings and cross-fit groups must have equal length")
    missing = sorted(set(BASE_METHODS) - set(base_probabilities))
    if missing:
        raise ValueError(f"Missing base OOF probabilities: {missing}")
    features = logit_features(*(base_probabilities[method] for method in BASE_METHODS))
    output = np.full((len(labels), 2), np.nan, dtype=np.float64)
    for fold_number, held_group in enumerate(np.unique(crossfit_groups)):
        validation = crossfit_groups == held_group
        train = ~validation
        if np.unique(labels[train]).size != 2:
            raise ValueError(f"Meta-training fold excluding {held_group!r} is missing a class")
        unique_recordings, counts = np.unique(recordings[train], return_counts=True)
        count_by_recording = dict(zip(unique_recordings.tolist(), counts.tolist()))
        sample_weight = np.asarray(
            [1.0 / count_by_recording[value] for value in recordings[train]], dtype=np.float64
        )
        sample_weight = sample_weight / sample_weight.mean()
        stacker = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=4000,
            random_state=int(seed) + 7919 * (fold_number + 1),
        )
        stacker.fit(features[train], labels[train], sample_weight=sample_weight)
        output[validation] = stacker.predict_proba(features[validation])
    if not np.isfinite(output).all():
        raise AssertionError("Meta-cross-fitting did not cover every training window")
    return normalize_probabilities(output)


@dataclass(frozen=True)
class DasfDecision:
    use_fusion: bool
    validation_accuracy_nn: float
    validation_accuracy_fuse: float
    validation_gain: float
    shift: float
    effective_margin: float


def clean_dasf_decision(
    validation_accuracy_nn: float,
    validation_accuracy_fuse: float,
    shift: float,
    *,
    base_margin: float,
    alpha: float,
    max_margin: float,
) -> DasfDecision:
    values = np.asarray([validation_accuracy_nn, validation_accuracy_fuse, shift], dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    effective = float(np.clip(base_margin + alpha * shift, 0.0, max_margin)) if finite else float(max_margin)
    gain = float(validation_accuracy_fuse - validation_accuracy_nn) if finite else float("nan")
    use_fusion = bool(finite and gain > effective)
    return DasfDecision(
        use_fusion=use_fusion,
        validation_accuracy_nn=float(validation_accuracy_nn),
        validation_accuracy_fuse=float(validation_accuracy_fuse),
        validation_gain=gain,
        shift=float(shift),
        effective_margin=effective,
    )


def apply_clean_dasf(nn: np.ndarray, fuse: np.ndarray, decision: DasfDecision) -> np.ndarray:
    nn = np.asarray(nn, dtype=np.float64)
    fuse = np.asarray(fuse, dtype=np.float64)
    if nn.shape != fuse.shape:
        raise ValueError("NN and fusion probability shapes differ")
    if nn.ndim != 2 or nn.shape[1] != 2:
        raise ValueError("DASF inputs must be (n, 2) probability matrices")
    for name, values in (("NN", nn), ("fusion", fuse)):
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"{name} probabilities are not normalized within tolerance")
    chosen = fuse if decision.use_fusion else nn
    out = chosen.copy()
    if not np.array_equal(out, chosen):
        raise AssertionError("DASF output is not exactly the selected input")
    return out
