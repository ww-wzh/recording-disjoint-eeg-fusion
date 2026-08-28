from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from eeg_channel_selection import PARSER_SCHEMA, select_openbci_eeg_channels


@dataclass
class FeatureBundle:
    features: np.ndarray
    covariances: np.ndarray
    labels_raw: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray
    window_ids: np.ndarray
    task: str

    def subset(self, indices: np.ndarray) -> "FeatureBundle":
        indices = np.asarray(indices)
        return FeatureBundle(
            features=self.features[indices],
            covariances=self.covariances[indices],
            labels_raw=self.labels_raw[indices],
            labels=self.labels[indices],
            subjects=self.subjects[indices],
            recordings=self.recordings[indices],
            window_ids=self.window_ids[indices],
            task=self.task,
        )


@dataclass
class RawBundle:
    windows: np.ndarray
    labels_raw: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray
    window_ids: np.ndarray
    task: str
    sfreq: float

    def subset(self, indices: np.ndarray) -> "RawBundle":
        indices = np.asarray(indices)
        return RawBundle(
            windows=self.windows[indices],
            labels_raw=self.labels_raw[indices],
            labels=self.labels[indices],
            subjects=self.subjects[indices],
            recordings=self.recordings[indices],
            window_ids=self.window_ids[indices],
            task=self.task,
            sfreq=self.sfreq,
        )


def _load_legacy(repo_root: Path):
    path = repo_root / "1111.py"
    module_name = "route_a_legacy_eeg_readonly"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    original_loader = module.load_txt_eeg

    def eeg_only_loader(path: str, default_sfreq: float = module.DEFAULT_SFREQ):
        signal, sfreq = original_loader(path, default_sfreq=default_sfreq)
        eeg, report = select_openbci_eeg_channels(signal)
        if not report.packet_counter_removed:
            raise ValueError(f"Expected and failed to remove the OpenBCI packet counter in {path}")
        return eeg, float(sfreq)

    module.load_txt_eeg = eeg_only_loader
    return module


def _recording_ids(task: str, records: list, segment_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{task}:s{records[int(value)].subject_id:02d}:r{records[int(value)].label}" for value in segment_ids],
        dtype=object,
    )


def _window_ids(recordings: np.ndarray) -> np.ndarray:
    counter: dict[object, int] = {}
    output = []
    for recording in recordings:
        offset = counter.get(recording, 0)
        output.append(f"{recording}:w{offset:05d}")
        counter[recording] = offset + 1
    return np.asarray(output, dtype=object)


def load_feature_bundles(repo_root: Path, device: torch.device | None = None) -> tuple[FeatureBundle, FeatureBundle]:
    repo_root = Path(repo_root).resolve()
    legacy = _load_legacy(repo_root)
    subjects = legacy.normalize_subjects(legacy.SUBJECTS)
    cache = repo_root / ".openbci_cache_eeg_only_recording_disjoint_v2_packet_counter_fixed"
    feature_device = device or torch.device("cpu")

    def build(task: str) -> FeatureBundle:
        records = legacy.build_records(subjects, task=task)
        dataset = legacy.EEGWindowDataset(
            records=records,
            win_sec=legacy.WIN_SEC,
            stride_sec=legacy.STRIDE_SEC,
            bands=legacy.BANDS,
            default_sfreq=legacy.DEFAULT_SFREQ,
            cache_dir=str(cache),
            device_for_feature=feature_device,
        )
        if int(dataset.cov_n_channels) != 8:
            raise AssertionError(f"Expected 8-channel covariances, got {dataset.cov_n_channels}")
        segments = dataset.segment_ids.cpu().numpy().astype(np.int64)
        recordings = _recording_ids(task, records, segments)
        labels_raw = dataset.labels.cpu().numpy().astype(np.int64)
        return FeatureBundle(
            features=dataset.features.cpu().numpy().astype(np.float32),
            covariances=dataset.covs.cpu().numpy().astype(np.float64),
            labels_raw=labels_raw,
            labels=(labels_raw >= 2).astype(np.int64),
            subjects=dataset.subject_ids.cpu().numpy().astype(np.int64),
            recordings=recordings,
            window_ids=_window_ids(recordings),
            task=task,
        )

    return build("arithmetic"), build("stroop")


def normalized_frobenius_shift(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 3 or target.ndim != 3 or source.shape[1:] != target.shape[1:]:
        raise ValueError("Source and target covariance arrays must be compatible 3-D arrays")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    denominator = 0.5 * (
        np.linalg.norm(source_mean, ord="fro") + np.linalg.norm(target_mean, ord="fro")
    )
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(source_mean - target_mean, ord="fro") / denominator)


def _raw_cache_path(cache_dir: Path, task: str, subject: int, label: int) -> Path:
    return cache_dir / f"{PARSER_SCHEMA}_{task}_s{subject:02d}_r{label}.npz"


def _window_recording(
    signal: np.ndarray,
    sfreq: float,
    *,
    window_seconds: float,
    stride_seconds: float,
) -> np.ndarray:
    if signal.ndim != 2 or signal.shape[1] != 8:
        raise ValueError(f"Raw recording must be (samples, 8), got {signal.shape}")
    nyquist = 0.5 * float(sfreq)
    high = min(45.0, nyquist * 0.95)
    if high <= 0.5:
        raise ValueError(f"Sampling rate {sfreq} is too low for the frozen bandpass")
    sos = butter(4, [0.5 / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, signal, axis=0).astype(np.float32)
    window = int(round(window_seconds * sfreq))
    stride = int(round(stride_seconds * sfreq))
    starts = np.arange(0, len(filtered) - window + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("Recording is shorter than one analysis window")
    chunks = np.stack([filtered[start : start + window].T for start in starts], axis=0)
    mean = chunks.mean(axis=2, keepdims=True)
    std = np.clip(chunks.std(axis=2, keepdims=True), 1e-6, None)
    return ((chunks - mean) / std).astype(np.float32)


def load_raw_bundle(
    repo_root: Path,
    task: str,
    subjects: Iterable[int],
    cache_dir: Path,
    *,
    window_seconds: float = 8.5,
    stride_seconds: float = 0.5,
) -> RawBundle:
    repo_root = Path(repo_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    subject_set = {int(value) for value in subjects}
    legacy = _load_legacy(repo_root)
    normalized = legacy.normalize_subjects(legacy.SUBJECTS)
    records = [record for record in legacy.build_records(normalized, task=task) if int(record.subject_id) in subject_set]
    if not records:
        raise ValueError(f"No {task} records for subjects {sorted(subject_set)}")

    windows_all = []
    labels_raw = []
    subject_rows = []
    recording_rows = []
    sfreq_seen = []
    for record in records:
        path = _raw_cache_path(cache_dir, task, int(record.subject_id), int(record.label))
        source = Path(record.file_path)
        source_stat = source.stat()
        use_cache = False
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                use_cache = bool(
                    int(cached["source_bytes"]) == int(source_stat.st_size)
                    and int(cached["source_mtime_ns"]) == int(source_stat.st_mtime_ns)
                    and float(cached["window_seconds"]) == float(window_seconds)
                    and float(cached["stride_seconds"]) == float(stride_seconds)
                    and str(cached["parser_schema"].item()) == PARSER_SCHEMA
                )
                if use_cache:
                    windows = cached["windows"].astype(np.float32)
                    sfreq = float(cached["sfreq"])
        if not use_cache:
            signal, sfreq = legacy.load_txt_eeg(str(source), default_sfreq=legacy.DEFAULT_SFREQ)
            windows = _window_recording(
                signal,
                sfreq,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            )
            np.savez_compressed(
                path,
                windows=windows,
                sfreq=np.float64(sfreq),
                source_bytes=np.int64(source_stat.st_size),
                source_mtime_ns=np.int64(source_stat.st_mtime_ns),
                window_seconds=np.float64(window_seconds),
                stride_seconds=np.float64(stride_seconds),
                parser_schema=np.asarray(PARSER_SCHEMA),
            )
        n_windows = len(windows)
        recording_id = f"{task}:s{int(record.subject_id):02d}:r{int(record.label)}"
        windows_all.append(windows)
        labels_raw.append(np.full(n_windows, int(record.label), dtype=np.int64))
        subject_rows.append(np.full(n_windows, int(record.subject_id), dtype=np.int64))
        recording_rows.append(np.full(n_windows, recording_id, dtype=object))
        sfreq_seen.append(float(sfreq))

    if not np.allclose(sfreq_seen, sfreq_seen[0], atol=1e-6):
        raise ValueError(f"Deep baselines require a common sampling rate; observed {sorted(set(sfreq_seen))}")
    recordings = np.concatenate(recording_rows)
    raw_labels = np.concatenate(labels_raw)
    output = RawBundle(
        windows=np.concatenate(windows_all, axis=0),
        labels_raw=raw_labels,
        labels=(raw_labels >= 2).astype(np.int64),
        subjects=np.concatenate(subject_rows),
        recordings=recordings,
        window_ids=_window_ids(recordings),
        task=task,
        sfreq=float(sfreq_seen[0]),
    )
    if output.windows.shape[1] != 8:
        raise AssertionError("Deep baseline cache contains non-EEG channels")
    return output
