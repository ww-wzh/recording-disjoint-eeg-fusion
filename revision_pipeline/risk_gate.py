from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GATE_FEATURES = [
    "feature_shift",
    "disagreement",
    "entropy_nn",
    "entropy_fuse",
    "confidence_nn",
]


def robust_feature_shift(
    x_train: np.ndarray,
    x_target: np.ndarray,
    *,
    per_feature_cap: float = 10.0,
    scale_epsilon: float = 1e-6,
) -> float:
    """Return a bounded median standardized shift across feature dimensions."""
    reference = np.asarray(x_train, dtype=np.float64)
    target = np.asarray(x_target, dtype=np.float64)
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("Feature-shift inputs must be two-dimensional")
    if reference.shape[1] != target.shape[1] or reference.shape[1] == 0:
        raise ValueError("Feature-shift inputs must have the same non-zero feature dimension")
    if len(reference) == 0 or len(target) == 0:
        raise ValueError("Feature-shift inputs must contain at least one row")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("Feature-shift inputs must be finite")
    if per_feature_cap <= 0.0 or scale_epsilon <= 0.0:
        raise ValueError("Feature-shift bounds must be positive")

    delta = np.abs(np.mean(target, axis=0) - np.mean(reference, axis=0))
    scale = np.std(reference, axis=0, ddof=0)
    standardized = np.zeros_like(delta, dtype=np.float64)
    stable = scale >= float(scale_epsilon)
    standardized[stable] = delta[stable] / scale[stable]
    standardized[~stable] = np.where(
        delta[~stable] <= float(scale_epsilon),
        0.0,
        float(per_feature_cap),
    )
    standardized = np.clip(
        np.nan_to_num(
            standardized,
            nan=float(per_feature_cap),
            posinf=float(per_feature_cap),
            neginf=float(per_feature_cap),
        ),
        0.0,
        float(per_feature_cap),
    )
    return float(np.median(standardized))


def binary_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(np.mean(-np.sum(values * np.log(values), axis=1)))


def label_free_diagnostics(
    nn_prob: np.ndarray,
    fuse_prob: np.ndarray,
    x_train: np.ndarray,
    x_target: np.ndarray,
) -> dict[str, float]:
    scale = np.std(x_train, axis=0, ddof=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    shift = np.mean(np.abs((np.mean(x_target, axis=0) - np.mean(x_train, axis=0)) / scale))
    return {
        "feature_shift": float(shift),
        "disagreement": float(np.mean(np.abs(nn_prob[:, 1] - fuse_prob[:, 1]))),
        "entropy_nn": binary_entropy(nn_prob),
        "entropy_fuse": binary_entropy(fuse_prob),
        "confidence_nn": float(np.mean(np.max(nn_prob, axis=1))),
    }


@dataclass(frozen=True)
class GatePrediction:
    weight: float
    predicted_benefit: float
    predicted_catastrophe: float


@dataclass(frozen=True)
class RobustGatePrediction:
    weight: float
    predicted_benefit: float
    predicted_benefit_raw: float
    predicted_catastrophe: float
    ood_fallback: bool
    max_abs_robust_z: float


def fit_predict_gate(
    training_rows: pd.DataFrame,
    target_row: pd.Series,
    catastrophe_threshold: float,
    benefit_alpha: float,
    catastrophe_c: float,
    risk_penalty: float,
    temperature: float,
    max_weight: float,
) -> GatePrediction:
    missing = sorted(set(GATE_FEATURES + ["benefit"]) - set(training_rows.columns))
    if missing:
        raise ValueError(f"Gate training rows are missing: {missing}")
    if len(training_rows) < 5:
        raise ValueError("At least five independent training subjects/directions are required for the gate")

    x_train = training_rows[GATE_FEATURES].to_numpy(dtype=np.float64)
    y_benefit = training_rows["benefit"].to_numpy(dtype=np.float64)
    x_target = target_row[GATE_FEATURES].to_numpy(dtype=np.float64).reshape(1, -1)

    benefit_model = make_pipeline(StandardScaler(), Ridge(alpha=float(benefit_alpha)))
    benefit_model.fit(x_train, y_benefit)
    predicted_benefit = float(benefit_model.predict(x_target)[0])

    catastrophe = (y_benefit <= -float(catastrophe_threshold)).astype(np.int64)
    if np.unique(catastrophe).size == 2:
        catastrophe_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(catastrophe_c), class_weight="balanced", max_iter=2000, random_state=0
            ),
        )
        catastrophe_model.fit(x_train, catastrophe)
        predicted_catastrophe = float(catastrophe_model.predict_proba(x_target)[0, 1])
    else:
        predicted_catastrophe = float(np.mean(catastrophe))

    score = predicted_benefit - float(risk_penalty) * predicted_catastrophe
    scaled = np.clip(score / float(temperature), -40.0, 40.0)
    weight = float(max_weight) * float(1.0 / (1.0 + np.exp(-scaled)))
    return GatePrediction(
        weight=float(np.clip(weight, 0.0, float(max_weight))),
        predicted_benefit=predicted_benefit,
        predicted_catastrophe=predicted_catastrophe,
    )


def fit_predict_gate_robust(
    training_rows: pd.DataFrame,
    target_row: pd.Series,
    catastrophe_threshold: float,
    benefit_alpha: float,
    catastrophe_c: float,
    risk_penalty: float,
    temperature: float,
    max_weight: float,
    *,
    feature_z_cap: float = 5.0,
    ood_z_threshold: float = 10.0,
) -> RobustGatePrediction:
    """Fit the gate with bounded extrapolation and a fail-closed OOD guard."""
    missing = sorted(set(GATE_FEATURES + ["benefit"]) - set(training_rows.columns))
    if missing:
        raise ValueError(f"Gate training rows are missing: {missing}")
    if len(training_rows) < 5:
        raise ValueError("At least five independent training subjects/directions are required for the gate")
    if feature_z_cap <= 0.0 or ood_z_threshold <= feature_z_cap:
        raise ValueError("The OOD threshold must be greater than the positive feature z cap")

    x_train_raw = training_rows[GATE_FEATURES].to_numpy(dtype=np.float64)
    y_benefit = training_rows["benefit"].to_numpy(dtype=np.float64)
    x_target_raw = target_row[GATE_FEATURES].to_numpy(dtype=np.float64).reshape(1, -1)
    if not np.isfinite(x_train_raw).all() or not np.isfinite(y_benefit).all():
        raise ValueError("Gate training values must be finite")
    if not np.isfinite(x_target_raw).all():
        return RobustGatePrediction(
            weight=0.0,
            predicted_benefit=0.0,
            predicted_benefit_raw=float("nan"),
            predicted_catastrophe=1.0,
            ood_fallback=True,
            max_abs_robust_z=float("inf"),
        )

    center = np.median(x_train_raw, axis=0)
    q1, q3 = np.quantile(x_train_raw, [0.25, 0.75], axis=0)
    robust_scale = q3 - q1
    fallback_scale = np.std(x_train_raw, axis=0, ddof=0)
    robust_scale = np.where(robust_scale >= 1e-8, robust_scale, fallback_scale)
    robust_scale = np.where(robust_scale >= 1e-8, robust_scale, 1.0)

    x_train_z = (x_train_raw - center) / robust_scale
    x_target_z = (x_target_raw - center) / robust_scale
    max_abs_robust_z = float(np.max(np.abs(x_target_z)))
    ood_fallback = bool(max_abs_robust_z > float(ood_z_threshold))
    x_train = np.clip(x_train_z, -float(feature_z_cap), float(feature_z_cap))
    x_target = np.clip(x_target_z, -float(feature_z_cap), float(feature_z_cap))

    benefit_model = Ridge(alpha=float(benefit_alpha))
    benefit_model.fit(x_train, y_benefit)
    predicted_benefit_raw = float(benefit_model.predict(x_target)[0])
    benefit_low = max(-1.0, float(np.min(y_benefit)))
    benefit_high = min(1.0, float(np.max(y_benefit)))
    predicted_benefit = float(np.clip(predicted_benefit_raw, benefit_low, benefit_high))

    catastrophe = (y_benefit <= -float(catastrophe_threshold)).astype(np.int64)
    if np.unique(catastrophe).size == 2:
        catastrophe_model = LogisticRegression(
            C=float(catastrophe_c), class_weight="balanced", max_iter=2000, random_state=0
        )
        catastrophe_model.fit(x_train, catastrophe)
        predicted_catastrophe = float(catastrophe_model.predict_proba(x_target)[0, 1])
    else:
        predicted_catastrophe = float(np.mean(catastrophe))

    if ood_fallback:
        weight = 0.0
    else:
        score = predicted_benefit - float(risk_penalty) * predicted_catastrophe
        scaled = np.clip(score / float(temperature), -40.0, 40.0)
        weight = float(max_weight) * float(1.0 / (1.0 + np.exp(-scaled)))
    return RobustGatePrediction(
        weight=float(np.clip(weight, 0.0, float(max_weight))),
        predicted_benefit=predicted_benefit,
        predicted_benefit_raw=predicted_benefit_raw,
        predicted_catastrophe=predicted_catastrophe,
        ood_fallback=ood_fallback,
        max_abs_robust_z=max_abs_robust_z,
    )


def blend_probabilities(nn_prob: np.ndarray, fuse_prob: np.ndarray, weight: float) -> np.ndarray:
    blended = (1.0 - float(weight)) * np.asarray(nn_prob) + float(weight) * np.asarray(fuse_prob)
    return blended / blended.sum(axis=1, keepdims=True)
