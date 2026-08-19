from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        *,
        f1: int,
        depth_multiplier: int,
        f2: int,
        temporal_kernel: int,
        dropout: float,
        pool1: int,
        pool2: int,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f1 * depth_multiplier, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                (1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flattened = self.separable(self.spatial(self.temporal(dummy))).flatten(1).shape[1]
        self.classifier = nn.Linear(flattened, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal(inputs.unsqueeze(1))
        hidden = self.spatial(hidden)
        hidden = self.separable(hidden)
        return self.classifier(hidden.flatten(1))


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        *,
        embedding_dim: int,
        temporal_kernel: int,
        pool_kernel: int,
        pool_stride: int,
        attention_heads: int,
        transformer_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                1,
                embedding_dim,
                (1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.Conv2d(embedding_dim, embedding_dim, (n_channels, 1), bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            token_count = int(self.patch_embedding(dummy).shape[-1])
        self.position = nn.Parameter(torch.zeros(1, token_count, embedding_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True already disables PyTorch's nested-tensor fast path. Make
        # that explicit to avoid emitting the same non-actionable warning per fit.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embedding(inputs.unsqueeze(1)).squeeze(2).transpose(1, 2)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        hidden = self.encoder(hidden)
        return self.output(hidden.mean(dim=1))


def build_model(name: str, n_channels: int, n_samples: int, config: dict[str, Any]) -> nn.Module:
    if name == "eegnet":
        return EEGNet(
            n_channels,
            n_samples,
            f1=int(config["F1"]),
            depth_multiplier=int(config["D"]),
            f2=int(config["F2"]),
            temporal_kernel=int(config["temporal_kernel"]),
            dropout=float(config["dropout"]),
            pool1=int(config["pool1"]),
            pool2=int(config["pool2"]),
        )
    if name == "eeg_conformer":
        return EEGConformer(
            n_channels,
            n_samples,
            embedding_dim=int(config["embedding_dim"]),
            temporal_kernel=int(config["temporal_kernel"]),
            pool_kernel=int(config["pool_kernel"]),
            pool_stride=int(config["pool_stride"]),
            attention_heads=int(config["attention_heads"]),
            transformer_layers=int(config["transformer_layers"]),
            feedforward_dim=int(config["feedforward_dim"]),
            dropout=float(config["dropout"]),
        )
    raise KeyError(name)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def recording_log_score(probabilities: np.ndarray, labels: np.ndarray, recordings: np.ndarray) -> float:
    scores = []
    for recording in np.unique(recordings):
        mask = recordings == recording
        recording_labels = np.unique(labels[mask])
        if recording_labels.size != 1:
            raise ValueError(f"Recording {recording!r} has inconsistent labels")
        probability = np.clip(probabilities[mask], 1e-12, 1.0)
        aggregate = np.exp(np.mean(np.log(probability), axis=0))
        aggregate = aggregate / aggregate.sum()
        scores.append(float(np.log(np.clip(aggregate[int(recording_labels[0])], 1e-12, 1.0))))
    if not scores:
        raise ValueError("No validation recordings")
    return float(np.mean(scores))


def predict_probabilities(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(windows, dtype=np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


@dataclass
class FittedDeepModel:
    model: nn.Module
    selected_epoch: int
    parameter_count: int

    def predict(self, windows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
        return predict_probabilities(self.model, windows, device, batch_size=batch_size)


def fit_deep_model(
    model_factory: Callable[[], nn.Module],
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    training_config: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    validation_windows: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    validation_recordings: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> FittedDeepModel:
    set_deterministic_seed(seed)
    model = model_factory().to(device)
    count = parameter_count(model)
    labels = np.asarray(train_labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    if (class_counts == 0).any():
        raise ValueError("Deep training fold is missing a class")
    class_weights = class_counts.sum() / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(train_windows, dtype=np.float32)),
            torch.from_numpy(labels),
        ),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    max_epochs = int(fixed_epochs or training_config["max_epochs"])
    patience_remaining = int(training_config["patience"])
    best_score = -np.inf
    best_epoch = max_epochs
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_windows, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_windows.to(device, non_blocking=True))
            loss = criterion(logits, batch_labels.to(device, non_blocking=True))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if fixed_epochs is not None:
            continue
        if validation_windows is None or validation_labels is None or validation_recordings is None:
            raise ValueError("Early stopping requires recording-aware validation data")
        probability = predict_probabilities(
            model,
            validation_windows,
            device,
            batch_size=int(training_config["batch_size"]),
        )
        score = recording_log_score(probability, validation_labels, validation_recordings)
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_remaining = int(training_config["patience"])
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return FittedDeepModel(model=model, selected_epoch=int(best_epoch), parameter_count=count)
