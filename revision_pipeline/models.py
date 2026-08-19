from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .splits import Split


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(inputs)
        hidden = torch.nn.functional.gelu(self.linear1(hidden))
        hidden = self.dropout(hidden)
        return inputs + self.linear2(hidden)


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


@dataclass
class FittedMLP:
    model: FeatureMLP
    scaler: StandardScaler
    device: torch.device

    def predict_proba(self, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float32)).astype(np.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(transformed)), batch_size=batch_size, shuffle=False)
        rows: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                logits = self.model(batch.to(self.device))
                rows.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(rows, axis=0).astype(np.float64)


def _recording_log_score(probabilities: np.ndarray, labels: np.ndarray, recordings: np.ndarray) -> float:
    log_likelihood: list[float] = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        y_values = np.unique(labels[mask])
        if y_values.size != 1:
            raise ValueError(f"Recording {recording!r} has inconsistent labels")
        p = np.clip(probabilities[mask], 1e-12, 1.0)
        p_recording = np.exp(np.mean(np.log(p), axis=0))
        p_recording = p_recording / p_recording.sum()
        log_likelihood.append(float(np.log(np.clip(p_recording[int(y_values[0])], 1e-12, 1.0))))
    return float(np.mean(log_likelihood))


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
    validation_recordings: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[FittedMLP, int]:
    set_deterministic_seed(seed)
    scaler = StandardScaler().fit(x_train)
    train_x = scaler.transform(x_train).astype(np.float32)
    train_y = np.asarray(y_train, dtype=np.int64)
    model = FeatureMLP(
        input_dim=train_x.shape[1],
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        dropout=float(config["dropout"]),
    ).to(device)

    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("MLP training fold is missing a class")
    weights = counts.sum() / (2.0 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )

    max_epochs = int(fixed_epochs or config["max_epochs"])
    patience_limit = int(config["patience"])
    best_score = -math.inf
    best_epoch = max_epochs
    best_state: dict[str, torch.Tensor] | None = None
    patience = patience_limit

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if fixed_epochs is not None:
            continue
        if x_validation is None or y_validation is None or validation_recordings is None:
            raise ValueError("Early stopping requires recording-aware validation inputs")
        fitted = FittedMLP(model=model, scaler=scaler, device=device)
        val_prob = fitted.predict_proba(x_validation)
        score = _recording_log_score(val_prob, y_validation, validation_recordings)
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = patience_limit
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return FittedMLP(model=model, scaler=scaler, device=device), int(best_epoch)


def _make_forest(name: str, config: dict[str, Any], seed: int):
    common = dict(
        n_estimators=int(config["n_estimators"]),
        max_depth=None if config.get("max_depth") is None else int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=int(config.get("n_jobs", -1)),
    )
    if name == "rf":
        return RandomForestClassifier(**common)
    if name == "extra_trees":
        return ExtraTreesClassifier(**common)
    raise KeyError(name)


def _stack_features(*probabilities: np.ndarray) -> np.ndarray:
    columns = []
    for values in probabilities:
        p1 = np.clip(np.asarray(values)[:, 1], 1e-6, 1.0 - 1e-6)
        columns.append(np.log(p1 / (1.0 - p1)))
    return np.column_stack(columns)


@dataclass
class NestedPrediction:
    test: dict[str, np.ndarray]
    oof: dict[str, np.ndarray]
    selected_epoch: int
    inner_best_epochs: list[int]


def nested_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_recordings: np.ndarray,
    x_test: np.ndarray,
    inner_splits: list[Split],
    model_config: dict[str, dict[str, Any]],
    seed: int,
    device: torch.device,
) -> NestedPrediction:
    """Fit all selection components from OOF predictions inside the outer fold."""

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    train_recordings = np.asarray(train_recordings)
    n = len(y_train)
    oof = {name: np.full((n, 2), np.nan, dtype=np.float64) for name in ("always_nn", "rf", "extra_trees")}
    epochs: list[int] = []

    for fold_number, split in enumerate(inner_splits):
        fold_seed = int(seed) + 1009 * (fold_number + 1)
        mlp, best_epoch = fit_mlp(
            x_train[split.train],
            y_train[split.train],
            model_config["mlp"],
            fold_seed,
            device,
            x_validation=x_train[split.validation],
            y_validation=y_train[split.validation],
            validation_recordings=train_recordings[split.validation],
        )
        epochs.append(best_epoch)
        oof["always_nn"][split.validation] = mlp.predict_proba(x_train[split.validation])

        for offset, name in enumerate(("rf", "extra_trees"), start=1):
            forest = _make_forest(name, model_config[name], fold_seed + offset * 37)
            forest.fit(x_train[split.train], y_train[split.train])
            oof[name][split.validation] = forest.predict_proba(x_train[split.validation])

    for name, values in oof.items():
        if not np.isfinite(values).all():
            missing = int(np.isnan(values).any(axis=1).sum())
            raise AssertionError(f"{name} has {missing} non-OOF training predictions")

    stacker = LogisticRegression(
        C=float(model_config["stacker"]["C"]),
        class_weight="balanced",
        max_iter=int(model_config["stacker"]["max_iter"]),
        random_state=int(seed),
    )
    unique_recordings, recording_counts = np.unique(train_recordings, return_counts=True)
    count_by_recording = dict(zip(unique_recordings.tolist(), recording_counts.tolist()))
    stack_weight = np.asarray([1.0 / count_by_recording[value] for value in train_recordings], dtype=np.float64)
    stack_weight = stack_weight / np.mean(stack_weight)
    stacker.fit(
        _stack_features(oof["always_nn"], oof["rf"], oof["extra_trees"]),
        y_train,
        sample_weight=stack_weight,
    )
    oof["always_fuse"] = stacker.predict_proba(
        _stack_features(oof["always_nn"], oof["rf"], oof["extra_trees"])
    )

    selected_epoch = max(1, int(np.median(epochs)))
    final_mlp, _ = fit_mlp(
        x_train,
        y_train,
        model_config["mlp"],
        int(seed),
        device,
        fixed_epochs=selected_epoch,
    )
    test = {"always_nn": final_mlp.predict_proba(x_test)}
    for offset, name in enumerate(("rf", "extra_trees"), start=1):
        forest = _make_forest(name, model_config[name], int(seed) + offset * 37)
        forest.fit(x_train, y_train)
        test[name] = forest.predict_proba(x_test)
    test["always_fuse"] = stacker.predict_proba(
        _stack_features(test["always_nn"], test["rf"], test["extra_trees"])
    )
    return NestedPrediction(test=test, oof=oof, selected_epoch=selected_epoch, inner_best_epochs=epochs)
