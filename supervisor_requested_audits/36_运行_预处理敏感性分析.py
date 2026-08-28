"""直接运行：完成导师要求的预处理敏感性分析。

用途：在同一固定学习器下比较参考预处理与四个冻结变体。
运行方式：在 PyCharm 中直接运行本文件，不需要填写命令行参数。
输出位置：37_输出_预处理敏感性分析。

本程序不会修改 Route A、33号冻结预测或旧特征缓存。每个预处理变体
都使用独立缓存，并在拟合模型前核对独立重建的参考特征与冻结参考特征。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
PROJECT_ROOT = REVISION_ROOT
ROUTE_A_ROOT = REVISION_ROOT / "route_a"
PROTOCOL_PATH = HERE / "35_冻结方案_预处理敏感性分析.json"
OUTPUT_DIR = HERE / "37_输出_预处理敏感性分析"
CACHE_ROOT = OUTPUT_DIR / "00_特征缓存_各变体"

FOLD_OUTPUT = OUTPUT_DIR / "01_结果_逐折.csv"
PARTICIPANT_OUTPUT = OUTPUT_DIR / "02_结果_参与者级.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "03_结果_预处理变体汇总.csv"
PAIRED_OUTPUT = OUTPUT_DIR / "04_结果_相对参考配对差值和BH校正.csv"
FIGURE_OUTPUT = OUTPUT_DIR / "05_图_预处理变体相对参考准确率差值.png"
MANIFEST_OUTPUT = OUTPUT_DIR / "06_记录_运行清单和哈希.json"

BOOTSTRAP_REPEATS = 10000
FEATURE_ATOL = 2e-4
FEATURE_RTOL = 2e-4
RUNNER_VERSION = "2026-07-29-preprocessing-sensitivity-r1"

sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(ROUTE_A_ROOT))

from revision_pipeline.splits import balanced_recording_folds  # noqa: E402
from route_a_lib.data import FeatureBundle, _load_legacy, load_feature_bundles  # noqa: E402


VARIANT_DISPLAY = {
    "reference": "Reference preprocessing",
    "no_window_zscore": "No within-window z-score",
    "recording_zscore": "Recording-level z-score",
    "no_50hz_attenuation": "No 50-Hz attenuation",
    "neutral_legacy_multipliers": "Neutral legacy multipliers",
}

EVALUATION_DISPLAY = {
    "recording_disjoint": "Within-participant recording-disjoint",
    "participant_disjoint_loso": "Participant-disjoint LOSO",
    "cross_task": "Cross-task transfer",
}

CONDITION_DISPLAY = {
    "arithmetic": "Arithmetic",
    "stroop": "Stroop",
    "arithmetic_to_stroop": "Arithmetic to Stroop",
    "stroop_to_arithmetic": "Stroop to Arithmetic",
}


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    normalization: str
    line_attenuation_50hz: float
    frontal_multiplier: float
    theta_alpha_multiplier: float

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "variant_id": self.variant_id,
                "normalization": self.normalization,
                "line_attenuation_50hz": self.line_attenuation_50hz,
                "frontal_multiplier": self.frontal_multiplier,
                "theta_alpha_multiplier": self.theta_alpha_multiplier,
                "runner_version": RUNNER_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def load_protocol() -> tuple[dict[str, Any], list[VariantConfig]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    variants = []
    for row in protocol["variants"]:
        normalization_text = str(row["normalization"])
        if normalization_text.startswith("within-window"):
            normalization = "window_zscore"
        elif normalization_text.startswith("none"):
            normalization = "none"
        elif normalization_text.startswith("continuous-recording"):
            normalization = "recording_zscore"
        else:
            raise ValueError(f"Unknown normalization in protocol: {normalization_text}")
        variants.append(
            VariantConfig(
                variant_id=str(row["id"]),
                normalization=normalization,
                line_attenuation_50hz=float(row["line_attenuation_50hz"]),
                frontal_multiplier=float(row["frontal_relative_power_multiplier"]),
                theta_alpha_multiplier=float(row["theta_alpha_relative_power_multiplier"]),
            )
        )
    expected = set(VARIANT_DISPLAY)
    observed = {variant.variant_id for variant in variants}
    if observed != expected:
        raise AssertionError(f"Protocol variants differ from runner: {observed} versus {expected}")
    return protocol, variants


def cache_path_for_recording(
    config: VariantConfig,
    task: str,
    subject: int,
    raw_label: int,
) -> Path:
    return CACHE_ROOT / config.variant_id / f"{task}_s{subject:02d}_r{raw_label}.npz"


def recording_id(task: str, subject: int, raw_label: int) -> str:
    return f"{task}:s{subject:02d}:r{raw_label}"


def extract_features(
    eeg: np.ndarray,
    *,
    sfreq: float,
    win_sec: float,
    stride_sec: float,
    bands: list[tuple[float, float]],
    config: VariantConfig,
    device: torch.device,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the frozen 272-D extractor with one explicit variant configuration."""
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[1] != 8:
        raise ValueError(f"Expected raw EEG shaped (samples, 8), got {eeg.shape}")

    nyquist = float(sfreq) / 2.0
    if not (0.5 > 0.0 and 55.0 < nyquist):
        raise ValueError(f"Sampling rate {sfreq} does not support the frozen 0.5-55 Hz bandpass")
    sos = butter(4, [0.5 / nyquist, 55.0 / nyquist], btype="band", output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=0).astype(np.float32)

    if config.normalization == "recording_zscore":
        mean = eeg.mean(axis=0, keepdims=True)
        std = np.clip(eeg.std(axis=0, keepdims=True), 1e-6, None)
        eeg = ((eeg - mean) / std).astype(np.float32)
    elif config.normalization not in {"window_zscore", "none"}:
        raise ValueError(f"Unsupported normalization mode: {config.normalization}")

    win_samples = int(round(float(win_sec) * float(sfreq)))
    stride_samples = int(round(float(stride_sec) * float(sfreq)))
    if win_samples != 2125 or stride_samples != 125:
        raise AssertionError(
            f"Frozen sampling geometry expected 2125/125 samples, got {win_samples}/{stride_samples}"
        )
    if len(eeg) < win_samples:
        raise ValueError("Recording is shorter than one frozen analysis window")
    n_windows = 1 + (len(eeg) - win_samples) // stride_samples
    n_fft = next_power_of_two(win_samples)
    if n_fft != 4096:
        raise AssertionError(f"2125-sample windows must use a 4096-point FFT, got {n_fft}")

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sfreq))
    band_masks = [(freqs >= low) & (freqs < high) for low, high in bands]
    hann = torch.from_numpy(np.hanning(win_samples).astype(np.float32)).to(device)
    eps = 1e-12
    feature_parts: list[np.ndarray] = []
    covariance_parts: list[np.ndarray] = []
    starts = [index * stride_samples for index in range(n_windows)]

    for chunk_start in range(0, n_windows, int(chunk_size)):
        chunk_starts = starts[chunk_start : chunk_start + int(chunk_size)]
        chunk = np.stack(
            [eeg[start : start + win_samples].T for start in chunk_starts],
            axis=0,
        ).astype(np.float32)
        x_raw = torch.from_numpy(chunk).to(device)
        if config.normalization == "window_zscore":
            channel_mean = x_raw.mean(dim=-1, keepdim=True)
            channel_std = x_raw.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
            x = (x_raw - channel_mean) / channel_std
        else:
            x = x_raw

        windowed = x * hann[None, None, :]
        spectrum = torch.fft.rfft(windowed, n=n_fft, dim=-1)
        power = (spectrum.real**2 + spectrum.imag**2) / float(n_fft)

        if config.line_attenuation_50hz < 1.0:
            notch_mask = (freqs >= 48.5) & (freqs <= 51.5)
            notch_tensor = torch.from_numpy(notch_mask.astype(np.bool_)).to(device)
            power[:, :, notch_tensor] *= float(config.line_attenuation_50hz)

        total_mask = (freqs >= 0.5) & (freqs < 55.0)
        total_tensor = torch.from_numpy(total_mask.astype(np.bool_)).to(device)
        total_power = power[:, :, total_tensor].mean(dim=-1, keepdim=True).clamp_min(eps)

        absolute_bands = []
        relative_bands = []
        for mask in band_masks:
            mask_tensor = torch.from_numpy(mask.astype(np.bool_)).to(device)
            band_power = power[:, :, mask_tensor].mean(dim=-1)
            absolute_bands.append(torch.log(band_power + eps))
            relative_bands.append(band_power / total_power.squeeze(-1))
        absolute_band_power = torch.stack(absolute_bands, dim=-1)
        relative_band_power = torch.stack(relative_bands, dim=-1)

        if abs(config.frontal_multiplier - 1.0) > 1e-6:
            frontal_indices = torch.arange(
                0,
                relative_band_power.shape[1] // 2,
                device=device,
                dtype=torch.long,
            )
            log_weight = float(np.log(config.frontal_multiplier))
            absolute_band_power = absolute_band_power.clone()
            relative_band_power = relative_band_power.clone()
            absolute_band_power[:, frontal_indices, :] += log_weight
            relative_band_power[:, frontal_indices, :] *= float(config.frontal_multiplier)

        if config.theta_alpha_multiplier > 1.0:
            relative_band_power = relative_band_power.clone()
            relative_band_power[:, :, 1] *= float(config.theta_alpha_multiplier)
            relative_band_power[:, :, 2] *= float(config.theta_alpha_multiplier)

        normalized_power = power[:, :, total_tensor]
        normalized_power = normalized_power / normalized_power.sum(dim=-1, keepdim=True).clamp_min(eps)
        spectral_entropy = -(normalized_power * torch.log(normalized_power + eps)).sum(dim=-1)

        theta = relative_band_power[:, :, 1]
        alpha = relative_band_power[:, :, 2]
        beta = (relative_band_power[:, :, 3] + relative_band_power[:, :, 4]).clamp_min(eps)
        ratio_theta_beta = (theta / beta).unsqueeze(-1)
        ratio_alpha_beta = (alpha / beta).unsqueeze(-1)

        time_mean = x.mean(dim=-1)
        time_std = x.std(dim=-1, unbiased=False)
        time_rms = torch.sqrt((x**2).mean(dim=-1) + eps)
        time_peak_to_peak = x.amax(dim=-1) - x.amin(dim=-1)

        absolute_flat = absolute_band_power.reshape(absolute_band_power.shape[0], -1)
        relative_flat = relative_band_power.reshape(relative_band_power.shape[0], -1)
        entropy_flat = spectral_entropy.reshape(spectral_entropy.shape[0], -1)
        ratio_flat = torch.cat([ratio_theta_beta, ratio_alpha_beta], dim=-1).reshape(
            spectral_entropy.shape[0], -1
        )
        time_flat = torch.cat([time_mean, time_std, time_rms, time_peak_to_peak], dim=-1)

        line_length = torch.abs(x[..., 1:] - x[..., :-1]).mean(dim=-1)
        alpha_channels = relative_band_power[:, :, 2]
        half_channels = alpha_channels.shape[1] // 2
        alpha_asymmetry = (
            alpha_channels[:, :half_channels].mean(dim=1)
            - alpha_channels[:, half_channels:].mean(dim=1)
        ).unsqueeze(-1)
        delta_theta = (relative_band_power[:, :, 0] + relative_band_power[:, :, 1]).clamp_min(eps)
        alpha_beta = (
            relative_band_power[:, :, 2]
            + relative_band_power[:, :, 3]
            + relative_band_power[:, :, 4]
        ).clamp_min(eps)
        slow_to_fast = (delta_theta / alpha_beta).unsqueeze(-1).reshape(x.shape[0], -1)
        physiological_extra = torch.cat([line_length, alpha_asymmetry, slow_to_fast], dim=-1)

        first_difference = x[..., 1:] - x[..., :-1]
        second_difference = first_difference[..., 1:] - first_difference[..., :-1]
        variance_zero = x.var(dim=-1, unbiased=False).clamp_min(eps)
        variance_one = first_difference.var(dim=-1, unbiased=False).clamp_min(eps)
        variance_two = second_difference.var(dim=-1, unbiased=False).clamp_min(eps)
        mobility = torch.sqrt(variance_one / variance_zero)
        complexity = torch.sqrt(variance_two / variance_one) / mobility.clamp_min(eps)
        activity = torch.log(variance_zero + eps)
        hjorth_flat = torch.cat([activity, mobility, complexity], dim=-1).reshape(x.shape[0], -1)

        gamma_channels = relative_band_power[:, :, -1].clamp_min(eps)
        beta_channels = (
            relative_band_power[:, :, 3] + relative_band_power[:, :, 4]
        ).clamp_min(eps)
        alpha_channels_safe = relative_band_power[:, :, 2].clamp_min(eps)
        theta_channels_safe = relative_band_power[:, :, 1].clamp_min(eps)
        log_channel_ratios = torch.stack(
            [
                torch.log((alpha_channels_safe / beta_channels).mean(dim=1)),
                torch.log((gamma_channels / alpha_channels_safe).mean(dim=1)),
                torch.log((theta_channels_safe / alpha_channels_safe).mean(dim=1)),
            ],
            dim=-1,
        )

        relative_mean = relative_band_power.mean(dim=1)
        relative_std = relative_band_power.std(dim=1, unbiased=False)
        absolute_mean = absolute_band_power.mean(dim=1)
        entropy_mean = spectral_entropy.mean(dim=1, keepdim=True)
        entropy_std = spectral_entropy.std(dim=1, unbiased=False, keepdim=True)
        spatial_flat = torch.cat(
            [relative_mean, relative_std, absolute_mean, entropy_mean, entropy_std],
            dim=-1,
        )

        frontal_theta = relative_band_power[:, :half_channels, 1].mean(dim=1, keepdim=True)
        frontal_beta = (
            relative_band_power[:, :half_channels, 3]
            + relative_band_power[:, :half_channels, 4]
        ).mean(dim=1, keepdim=True).clamp_min(eps)
        frontal_alpha = relative_band_power[:, :half_channels, 2].mean(
            dim=1, keepdim=True
        ).clamp_min(eps)
        workload_features = torch.cat(
            [
                torch.log((frontal_theta / frontal_beta).clamp_min(eps)),
                torch.log(frontal_alpha),
            ],
            dim=-1,
        )

        differential_entropy = []
        for mask in band_masks:
            mask_tensor = torch.from_numpy(mask.astype(np.bool_)).to(device)
            band_signal = torch.fft.irfft(
                spectrum * mask_tensor[None, None, :].float(),
                n=n_fft,
                dim=-1,
            )[..., :win_samples]
            band_variance = band_signal.var(dim=-1, unbiased=False).clamp_min(eps)
            differential_entropy.append(
                0.5 * torch.log(2.0 * math.pi * math.e * band_variance)
            )
        differential_entropy_all = torch.stack(differential_entropy, dim=-1)

        features = torch.cat(
            [
                absolute_flat,
                relative_flat,
                entropy_flat,
                ratio_flat,
                time_flat,
                spatial_flat,
                hjorth_flat,
                physiological_extra,
                log_channel_ratios,
                workload_features,
                differential_entropy_all.reshape(x.shape[0], -1),
                differential_entropy_all.mean(dim=1),
            ],
            dim=-1,
        )
        if features.shape[1] != 272:
            raise AssertionError(f"Expected 272 features, got {features.shape[1]}")

        covariance = torch.matmul(x, x.transpose(1, 2)) / float(x.shape[-1])
        identity = torch.eye(8, device=device, dtype=x.dtype).unsqueeze(0)
        covariance = covariance + 1e-4 * identity

        feature_parts.append(features.detach().cpu().numpy().astype(np.float32))
        covariance_parts.append(covariance.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feature_parts), np.concatenate(covariance_parts)


def cache_metadata(
    source: Path,
    *,
    sfreq: float,
    config: VariantConfig,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    stat = source.stat()
    data = protocol["data"]
    return {
        "source_path": str(source.resolve()),
        "source_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "sfreq": float(sfreq),
        "window_seconds": float(data["window_seconds"]),
        "stride_seconds": float(data["stride_seconds"]),
        "bands_hz": protocol["shared_feature_pipeline"]["bands_hz"],
        "variant": json.loads(config.canonical_json()),
    }


def load_or_extract_recording(
    *,
    source: Path,
    eeg: np.ndarray,
    sfreq: float,
    task: str,
    subject: int,
    raw_label: int,
    config: VariantConfig,
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_path_for_recording(config, task, subject, raw_label)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata(source, sfreq=sfreq, config=config, protocol=protocol)
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_metadata = str(cached["metadata_json"].item())
                if cached_metadata == metadata_json:
                    features = cached["features"].astype(np.float32)
                    covariances = cached["covariances"].astype(np.float32)
                    if features.ndim == 2 and features.shape[1] == 272:
                        return features, covariances
        except (KeyError, OSError, ValueError):
            pass

    bands = [tuple(map(float, pair)) for pair in protocol["shared_feature_pipeline"]["bands_hz"]]
    features, covariances = extract_features(
        eeg,
        sfreq=sfreq,
        win_sec=float(protocol["data"]["window_seconds"]),
        stride_sec=float(protocol["data"]["stride_seconds"]),
        bands=bands,
        config=config,
        device=device,
    )
    np.savez_compressed(
        cache_path,
        features=features,
        covariances=covariances,
        metadata_json=np.asarray(metadata_json),
    )
    return features, covariances


def build_variant_bundles(
    legacy: Any,
    protocol: dict[str, Any],
    config: VariantConfig,
    device: torch.device,
) -> dict[str, FeatureBundle]:
    normalized_subjects = legacy.normalize_subjects(legacy.SUBJECTS)
    bundles: dict[str, FeatureBundle] = {}
    for task in ["arithmetic", "stroop"]:
        print(f"    {VARIANT_DISPLAY[config.variant_id]}: extracting {CONDITION_DISPLAY[task]}")
        records = legacy.build_records(normalized_subjects, task=task)
        features_all = []
        covariances_all = []
        labels_raw_all = []
        subjects_all = []
        recordings_all = []
        window_ids_all = []
        for record in records:
            source = Path(record.file_path).resolve()
            eeg, sfreq = legacy.load_txt_eeg(
                str(source), default_sfreq=float(protocol["data"]["sampling_rate_hz"])
            )
            features, covariances = load_or_extract_recording(
                source=source,
                eeg=eeg,
                sfreq=float(sfreq),
                task=task,
                subject=int(record.subject_id),
                raw_label=int(record.label),
                config=config,
                protocol=protocol,
                device=device,
            )
            if not np.isfinite(features).all() or not np.isfinite(covariances).all():
                raise RuntimeError(
                    f"Non-finite feature output for {config.variant_id}, {task}, "
                    f"S{int(record.subject_id):02d}, recording {int(record.label)}"
                )
            count = len(features)
            rec_id = recording_id(task, int(record.subject_id), int(record.label))
            features_all.append(features)
            covariances_all.append(covariances)
            labels_raw_all.append(np.full(count, int(record.label), dtype=np.int64))
            subjects_all.append(np.full(count, int(record.subject_id), dtype=np.int64))
            recordings_all.append(np.full(count, rec_id, dtype=object))
            window_ids_all.append(
                np.asarray([f"{rec_id}:w{index:05d}" for index in range(count)], dtype=object)
            )

        labels_raw = np.concatenate(labels_raw_all)
        bundles[task] = FeatureBundle(
            features=np.concatenate(features_all).astype(np.float32),
            covariances=np.concatenate(covariances_all).astype(np.float64),
            labels_raw=labels_raw,
            labels=(labels_raw >= 2).astype(np.int64),
            subjects=np.concatenate(subjects_all),
            recordings=np.concatenate(recordings_all),
            window_ids=np.concatenate(window_ids_all),
            task=task,
        )
    return bundles


def check_reference_parity(
    rebuilt: dict[str, FeatureBundle],
    frozen: dict[str, FeatureBundle],
) -> dict[str, dict[str, float | bool]]:
    results: dict[str, dict[str, float | bool]] = {}
    for task in ["arithmetic", "stroop"]:
        current = rebuilt[task]
        expected = frozen[task]
        if not np.array_equal(current.window_ids, expected.window_ids):
            raise AssertionError(f"Reference parity failed: {task} window identifiers differ")
        if current.features.shape != expected.features.shape:
            raise AssertionError(
                f"Reference parity failed: {task} shape {current.features.shape} versus {expected.features.shape}"
            )
        difference = np.abs(current.features.astype(np.float64) - expected.features.astype(np.float64))
        finite = bool(np.isfinite(current.features).all() and np.isfinite(expected.features).all())
        passed = bool(
            finite
            and np.allclose(
                current.features,
                expected.features,
                atol=FEATURE_ATOL,
                rtol=FEATURE_RTOL,
            )
        )
        results[task] = {
            "passed": passed,
            "finite": finite,
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "atol": FEATURE_ATOL,
            "rtol": FEATURE_RTOL,
        }
        if not passed:
            raise RuntimeError(
                f"Reference feature parity failed for {task}: max abs difference "
                f"{difference.max():.6g}. Stop before fitting models; do not delete old caches."
            )
    return results


def build_model(protocol: dict[str, Any]) -> Any:
    learner = protocol["learner"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(learner["C"]),
            class_weight=str(learner["class_weight"]),
            solver=str(learner["solver"]),
            max_iter=int(learner["max_iter"]),
            random_state=int(learner["random_state"]),
        ),
    )


def aggregate_recordings(
    probabilities: np.ndarray,
    labels: np.ndarray,
    recordings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels = []
    aggregate_probabilities = []
    recording_names = []
    for rec_id in np.unique(recordings):
        mask = recordings == rec_id
        unique_labels = np.unique(labels[mask])
        if unique_labels.size != 1:
            raise ValueError(f"Recording {rec_id} contains inconsistent labels")
        aggregate = np.exp(np.mean(np.log(np.clip(probabilities[mask], 1e-12, 1.0)), axis=0))
        aggregate /= aggregate.sum()
        true_labels.append(int(unique_labels[0]))
        aggregate_probabilities.append(aggregate)
        recording_names.append(str(rec_id))
    return (
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(aggregate_probabilities, dtype=np.float64),
        np.asarray(recording_names, dtype=object),
    )


def evaluate_indices(
    *,
    train_bundle: FeatureBundle,
    train_indices: np.ndarray,
    validation_bundle: FeatureBundle,
    validation_indices: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, float | int]:
    model = build_model(protocol)
    model.fit(train_bundle.features[train_indices], train_bundle.labels[train_indices])
    probability = model.predict_proba(validation_bundle.features[validation_indices])
    y_true, p_recording, _ = aggregate_recordings(
        probability,
        validation_bundle.labels[validation_indices],
        validation_bundle.recordings[validation_indices],
    )
    prediction = np.argmax(p_recording, axis=1)
    return {
        "train_windows": int(len(train_indices)),
        "validation_windows": int(len(validation_indices)),
        "validation_recordings": int(len(y_true)),
        "recording_accuracy": float(accuracy_score(y_true, prediction)),
        "recording_log_loss": float(log_loss(y_true, p_recording, labels=[0, 1])),
    }


def evaluate_recording_disjoint(
    bundle: FeatureBundle,
    *,
    config: VariantConfig,
    split_seeds: list[int],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for subject in sorted(np.unique(bundle.subjects).astype(int).tolist()):
        subject_indices = np.flatnonzero(bundle.subjects == subject)
        subset = bundle.subset(subject_indices)
        for split_seed in split_seeds:
            for fold, split in enumerate(
                balanced_recording_folds(subset.labels_raw, subset.recordings, seed=split_seed),
                start=1,
            ):
                metrics = evaluate_indices(
                    train_bundle=subset,
                    train_indices=np.asarray(split.train, dtype=np.int64),
                    validation_bundle=subset,
                    validation_indices=np.asarray(split.validation, dtype=np.int64),
                    protocol=protocol,
                )
                train_recordings = set(subset.recordings[split.train])
                validation_recordings = set(subset.recordings[split.validation])
                if train_recordings & validation_recordings:
                    raise AssertionError("Recording-disjoint split contains recording overlap")
                rows.append(
                    {
                        "variant_id": config.variant_id,
                        "variant": VARIANT_DISPLAY[config.variant_id],
                        "evaluation": "recording_disjoint",
                        "evaluation_display": EVALUATION_DISPLAY["recording_disjoint"],
                        "condition": bundle.task,
                        "condition_display": CONDITION_DISPLAY[bundle.task],
                        "participant": subject,
                        "split_seed": split_seed,
                        "fold": fold,
                        **metrics,
                    }
                )
    return rows


def evaluate_loso(
    bundle: FeatureBundle,
    *,
    config: VariantConfig,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for fold, subject in enumerate(sorted(np.unique(bundle.subjects).astype(int).tolist()), start=1):
        train = np.flatnonzero(bundle.subjects != subject)
        validation = np.flatnonzero(bundle.subjects == subject)
        metrics = evaluate_indices(
            train_bundle=bundle,
            train_indices=train,
            validation_bundle=bundle,
            validation_indices=validation,
            protocol=protocol,
        )
        rows.append(
            {
                "variant_id": config.variant_id,
                "variant": VARIANT_DISPLAY[config.variant_id],
                "evaluation": "participant_disjoint_loso",
                "evaluation_display": EVALUATION_DISPLAY["participant_disjoint_loso"],
                "condition": bundle.task,
                "condition_display": CONDITION_DISPLAY[bundle.task],
                "participant": subject,
                "split_seed": "",
                "fold": fold,
                **metrics,
            }
        )
    return rows


def evaluate_cross_task(
    source: FeatureBundle,
    target: FeatureBundle,
    *,
    config: VariantConfig,
    condition: str,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    model = build_model(protocol)
    model.fit(source.features, source.labels)
    rows = []
    for fold, subject in enumerate(sorted(np.unique(target.subjects).astype(int).tolist()), start=1):
        validation = np.flatnonzero(target.subjects == subject)
        probability = model.predict_proba(target.features[validation])
        y_true, p_recording, _ = aggregate_recordings(
            probability,
            target.labels[validation],
            target.recordings[validation],
        )
        rows.append(
            {
                "variant_id": config.variant_id,
                "variant": VARIANT_DISPLAY[config.variant_id],
                "evaluation": "cross_task",
                "evaluation_display": EVALUATION_DISPLAY["cross_task"],
                "condition": condition,
                "condition_display": CONDITION_DISPLAY[condition],
                "participant": subject,
                "split_seed": "",
                "fold": fold,
                "train_windows": int(len(source.features)),
                "validation_windows": int(len(validation)),
                "validation_recordings": int(len(y_true)),
                "recording_accuracy": float(
                    accuracy_score(y_true, np.argmax(p_recording, axis=1))
                ),
                "recording_log_loss": float(log_loss(y_true, p_recording, labels=[0, 1])),
            }
        )
    return rows


def bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def build_participant_results(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby(
            [
                "variant_id",
                "variant",
                "evaluation",
                "evaluation_display",
                "condition",
                "condition_display",
                "participant",
            ],
            as_index=False,
        )
        .agg(
            n_evaluations=("fold", "size"),
            mean_recording_accuracy=("recording_accuracy", "mean"),
            mean_recording_log_loss=("recording_log_loss", "mean"),
        )
        .sort_values(["evaluation", "condition", "variant_id", "participant"])
        .reset_index(drop=True)
    )


def build_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = participants.groupby(
        [
            "variant_id",
            "variant",
            "evaluation",
            "evaluation_display",
            "condition",
            "condition_display",
        ],
        sort=True,
    )
    for keys, group in groups:
        accuracy = group["mean_recording_accuracy"].to_numpy(dtype=np.float64)
        logloss = group["mean_recording_log_loss"].to_numpy(dtype=np.float64)
        accuracy_low, accuracy_high = bootstrap_mean_ci(
            accuracy, seed=stable_seed(*map(str, keys), "accuracy")
        )
        logloss_low, logloss_high = bootstrap_mean_ci(
            logloss, seed=stable_seed(*map(str, keys), "logloss")
        )
        rows.append(
            {
                "variant_id": keys[0],
                "variant": keys[1],
                "evaluation": keys[2],
                "evaluation_display": keys[3],
                "condition": keys[4],
                "condition_display": keys[5],
                "n_participants": int(group["participant"].nunique()),
                "mean_recording_accuracy": float(accuracy.mean()),
                "accuracy_ci95_low": accuracy_low,
                "accuracy_ci95_high": accuracy_high,
                "mean_recording_log_loss": float(logloss.mean()),
                "log_loss_ci95_low": logloss_low,
                "log_loss_ci95_high": logloss_high,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evaluation", "condition", "variant_id"]
    ).reset_index(drop=True)


def wilcoxon_p_value(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=np.float64)
    if np.allclose(difference, 0.0):
        return 1.0
    return float(
        wilcoxon(
            difference,
            zero_method="pratt",
            correction=False,
            alternative="two-sided",
            method="approx",
        ).pvalue
    )


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def build_paired_results(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (evaluation, condition), group in participants.groupby(["evaluation", "condition"]):
        reference = group[group["variant_id"] == "reference"].set_index("participant")
        for variant_id in [value for value in VARIANT_DISPLAY if value != "reference"]:
            comparison = group[group["variant_id"] == variant_id].set_index("participant")
            common = reference.index.intersection(comparison.index)
            accuracy_difference = (
                comparison.loc[common, "mean_recording_accuracy"].to_numpy(dtype=np.float64)
                - reference.loc[common, "mean_recording_accuracy"].to_numpy(dtype=np.float64)
            )
            logloss_difference = (
                comparison.loc[common, "mean_recording_log_loss"].to_numpy(dtype=np.float64)
                - reference.loc[common, "mean_recording_log_loss"].to_numpy(dtype=np.float64)
            )
            low, high = bootstrap_mean_ci(
                accuracy_difference,
                seed=stable_seed(evaluation, condition, variant_id, "paired"),
            )
            rows.append(
                {
                    "evaluation": evaluation,
                    "evaluation_display": EVALUATION_DISPLAY[evaluation],
                    "condition": condition,
                    "condition_display": CONDITION_DISPLAY[condition],
                    "variant_id": variant_id,
                    "variant": VARIANT_DISPLAY[variant_id],
                    "comparator": VARIANT_DISPLAY["reference"],
                    "n_paired_participants": int(len(common)),
                    "mean_accuracy_difference": float(accuracy_difference.mean()),
                    "paired_accuracy_ci95_low": low,
                    "paired_accuracy_ci95_high": high,
                    "participants_improved": int(np.sum(accuracy_difference > 0)),
                    "participants_tied": int(np.sum(np.isclose(accuracy_difference, 0))),
                    "participants_worsened": int(np.sum(accuracy_difference < 0)),
                    "mean_log_loss_difference": float(logloss_difference.mean()),
                    "wilcoxon_p_raw": wilcoxon_p_value(accuracy_difference),
                    "analysis_status": "post-hoc supervisor-requested exploratory sensitivity",
                }
            )
    output = pd.DataFrame(rows)
    output["wilcoxon_p_bh"] = bh_adjust(output["wilcoxon_p_raw"].to_numpy())
    output["significant_after_bh_0_05"] = output["wilcoxon_p_bh"] < 0.05
    output["bh_family"] = "all 24 non-reference preprocessing comparisons"
    return output.sort_values(["evaluation", "condition", "variant_id"]).reset_index(drop=True)


def build_figure(paired: pd.DataFrame) -> None:
    variant_ids = [value for value in VARIANT_DISPLAY if value != "reference"]
    conditions = [
        ("recording_disjoint", "arithmetic"),
        ("recording_disjoint", "stroop"),
        ("participant_disjoint_loso", "arithmetic"),
        ("participant_disjoint_loso", "stroop"),
        ("cross_task", "arithmetic_to_stroop"),
        ("cross_task", "stroop_to_arithmetic"),
    ]
    matrix = np.full((len(conditions), len(variant_ids)), np.nan, dtype=np.float64)
    for row_index, (evaluation, condition) in enumerate(conditions):
        for column_index, variant_id in enumerate(variant_ids):
            selected = paired[
                (paired["evaluation"] == evaluation)
                & (paired["condition"] == condition)
                & (paired["variant_id"] == variant_id)
            ]
            if len(selected) != 1:
                raise AssertionError("Figure requires one paired row per condition and variant")
            matrix[row_index, column_index] = 100.0 * float(
                selected.iloc[0]["mean_accuracy_difference"]
            )

    limit = max(5.0, float(np.nanmax(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(12.5, 7.2))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if abs(value) > 0.55 * limit else "black"
            axis.text(column, row, f"{value:+.1f}", ha="center", va="center", color=text_color)

    axis.set_xticks(range(len(variant_ids)))
    axis.set_xticklabels([VARIANT_DISPLAY[value] for value in variant_ids], rotation=20, ha="right")
    axis.set_yticks(range(len(conditions)))
    axis.set_yticklabels(
        [
            f"{EVALUATION_DISPLAY[evaluation]}: {CONDITION_DISPLAY[condition]}"
            for evaluation, condition in conditions
        ]
    )
    axis.set_xlabel("Preprocessing sensitivity variant")
    axis.set_ylabel("Evaluation setting")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Accuracy difference from reference (pp)")
    figure.tight_layout()
    figure.savefig(FIGURE_OUTPUT, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    folds: pd.DataFrame,
    participants: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    parity: dict[str, dict[str, float | bool]],
    device: torch.device,
) -> None:
    folds.to_csv(FOLD_OUTPUT, index=False, lineterminator="\n")
    participants.to_csv(PARTICIPANT_OUTPUT, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_OUTPUT, index=False, lineterminator="\n")
    paired.to_csv(PAIRED_OUTPUT, index=False, lineterminator="\n")
    build_figure(paired)

    source_files = {
        "runner": Path(__file__).resolve(),
        "protocol": PROTOCOL_PATH,
        "legacy_feature_source": PROJECT_ROOT / "1111.py",
        "route_a_data_adapter": ROUTE_A_ROOT / "route_a_lib" / "data.py",
        "split_source": REVISION_ROOT / "revision_pipeline" / "splits.py",
    }
    output_files = [FOLD_OUTPUT, PARTICIPANT_OUTPUT, SUMMARY_OUTPUT, PAIRED_OUTPUT, FIGURE_OUTPUT]
    cache_counts = {
        variant_id: len(list((CACHE_ROOT / variant_id).glob("*.npz")))
        for variant_id in VARIANT_DISPLAY
    }
    if any(count != 120 for count in cache_counts.values()):
        raise AssertionError(f"Each variant must contain 120 recording caches, got {cache_counts}")
    manifest = {
        "purpose": "Supervisor-requested preprocessing sensitivity analysis",
        "analysis_status": "post_hoc_supervisor_requested_exploratory_extension",
        "protocol_version": RUNNER_VERSION,
        "statistical_unit": "participant",
        "evaluation_unit": "recording",
        "seed_is_statistical_unit": False,
        "reference_feature_parity": parity,
        "fold_rows": int(len(folds)),
        "participant_rows": int(len(participants)),
        "summary_rows": int(len(summary)),
        "paired_rows": int(len(paired)),
        "cache_file_counts": cache_counts,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "output_sha256": {path.name: sha256_file(path) for path in output_files},
        "claim_scope": "descriptive sensitivity only; no superiority, non-inferiority or safety claim",
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol, variants = load_protocol()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/7] 使用设备：{device}")
    print("      特征提取可使用GPU；固定LogisticRegression仍在CPU上运行。")

    legacy = _load_legacy(PROJECT_ROOT)
    print("[2/7] 正在读取冻结Route A参考特征，只用于一致性核对。")
    frozen_arithmetic, frozen_stroop = load_feature_bundles(PROJECT_ROOT, device)
    frozen = {"arithmetic": frozen_arithmetic, "stroop": frozen_stroop}

    all_rows: list[dict[str, Any]] = []
    parity: dict[str, dict[str, float | bool]] | None = None
    split_seeds = [
        int(value)
        for value in protocol["evaluations"]["recording_disjoint"]["split_seeds"]
    ]

    for variant_index, config in enumerate(variants, start=1):
        print(f"[3/7] 特征变体 {variant_index}/{len(variants)}：{VARIANT_DISPLAY[config.variant_id]}")
        bundles = build_variant_bundles(legacy, protocol, config, device)
        if config.variant_id == "reference":
            parity = check_reference_parity(bundles, frozen)
            print("      参考特征一致性检查通过。")

        print("      运行within-participant recording-disjoint评价。")
        for task in ["arithmetic", "stroop"]:
            all_rows.extend(
                evaluate_recording_disjoint(
                    bundles[task],
                    config=config,
                    split_seeds=split_seeds,
                    protocol=protocol,
                )
            )

        print("      运行participant-disjoint LOSO评价。")
        for task in ["arithmetic", "stroop"]:
            all_rows.extend(evaluate_loso(bundles[task], config=config, protocol=protocol))

        print("      运行双向cross-task评价。")
        all_rows.extend(
            evaluate_cross_task(
                bundles["arithmetic"],
                bundles["stroop"],
                config=config,
                condition="arithmetic_to_stroop",
                protocol=protocol,
            )
        )
        all_rows.extend(
            evaluate_cross_task(
                bundles["stroop"],
                bundles["arithmetic"],
                config=config,
                condition="stroop_to_arithmetic",
                protocol=protocol,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if parity is None:
        raise AssertionError("Reference preprocessing was not evaluated")

    print("[4/7] 汇总逐折和参与者级结果。")
    folds = pd.DataFrame(all_rows)
    participants = build_participant_results(folds)
    summary = build_summary(participants)
    if len(folds) != 1800 or len(participants) != 450 or len(summary) != 30:
        raise AssertionError(
            "Incomplete sensitivity result: expected 1800 fold rows, 450 participant rows "
            f"and 30 summary rows; got {len(folds)}, {len(participants)} and {len(summary)}"
        )

    print("[5/7] 计算相对参考流程的配对区间、Wilcoxon检验和BH校正。")
    paired = build_paired_results(participants)
    if len(paired) != 24:
        raise AssertionError(f"Expected 24 paired comparisons, got {len(paired)}")

    print("[6/7] 写入37号输出、图片和SHA-256运行清单。")
    write_outputs(folds, participants, summary, paired, parity, device)

    print("[7/7] 预处理敏感性分析完成。")
    display = summary[
        [
            "evaluation_display",
            "condition_display",
            "variant",
            "mean_recording_accuracy",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
        ]
    ].copy()
    for column in ["mean_recording_accuracy", "accuracy_ci95_low", "accuracy_ci95_high"]:
        display[column] = display[column].map(lambda value: f"{100.0 * float(value):.2f}%")
    print(display.to_string(index=False))
    print(f"\n结果目录：{OUTPUT_DIR}")
    print("下一步：把控制台最后的汇总结果发给我，并保留37号目录全部文件。")


if __name__ == "__main__":
    main()
