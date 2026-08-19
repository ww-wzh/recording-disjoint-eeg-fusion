from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray


def binary_labels(labels: Iterable[int]) -> np.ndarray:
    y = np.asarray(list(labels), dtype=np.int64)
    return (y >= 2).astype(np.int64)


def assert_disjoint_groups(
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    groups: Iterable[object],
) -> None:
    g = np.asarray(list(groups))
    overlap = set(g[train_idx].tolist()) & set(g[validation_idx].tolist())
    if overlap:
        raise AssertionError(f"Train/validation group leakage: {sorted(overlap, key=str)}")


def _recording_labels(labels: np.ndarray, recordings: np.ndarray) -> dict[object, int]:
    result: dict[object, int] = {}
    y = binary_labels(labels)
    for recording in np.unique(recordings):
        values = np.unique(y[recordings == recording])
        if values.size != 1:
            raise ValueError(f"Recording {recording!r} contains multiple labels")
        result[recording] = int(values[0])
    return result


def balanced_recording_folds(
    labels: Iterable[int],
    recordings: Iterable[object],
    seed: int,
) -> list[Split]:
    """Two balanced, recording-disjoint folds for the four-recording source task.

    Each validation fold contains one low-load and one high-load recording. The
    complementary two recordings form training. No overlapping windows can cross
    the boundary because recording IDs, rather than windows, are assigned.
    """

    labels_arr = np.asarray(list(labels), dtype=np.int64)
    rec_arr = np.asarray(list(recordings))
    if labels_arr.shape[0] != rec_arr.shape[0]:
        raise ValueError("labels and recordings must have the same length")
    mapping = _recording_labels(labels_arr, rec_arr)
    by_class = {
        cls: np.asarray(sorted([r for r, value in mapping.items() if value == cls], key=str), dtype=object)
        for cls in (0, 1)
    }
    if any(len(by_class[cls]) < 2 for cls in (0, 1)):
        raise ValueError("Recording-disjoint validation requires at least two recordings per binary class")

    rng = np.random.default_rng(int(seed))
    for cls in (0, 1):
        rng.shuffle(by_class[cls])

    folds: list[Split] = []
    for fold_idx in range(2):
        val_groups = {by_class[0][fold_idx % len(by_class[0])], by_class[1][fold_idx % len(by_class[1])]}
        val_mask = np.isin(rec_arr, list(val_groups))
        train = np.flatnonzero(~val_mask).astype(np.int64)
        validation = np.flatnonzero(val_mask).astype(np.int64)
        assert_disjoint_groups(train, validation, rec_arr)
        if set(binary_labels(labels_arr[train]).tolist()) != {0, 1}:
            raise ValueError("A recording fold lost a training class")
        if set(binary_labels(labels_arr[validation]).tolist()) != {0, 1}:
            raise ValueError("A recording fold lost a validation class")
        folds.append(Split(train=train, validation=validation))
    return folds


def subject_loso_folds(subjects: Iterable[int]) -> list[Split]:
    subject_arr = np.asarray(list(subjects), dtype=np.int64)
    unique = np.unique(subject_arr)
    if unique.size < 3:
        raise ValueError("Nested subject-level LOSO requires at least three subjects")
    folds: list[Split] = []
    for held_out in unique:
        validation = np.flatnonzero(subject_arr == held_out).astype(np.int64)
        train = np.flatnonzero(subject_arr != held_out).astype(np.int64)
        assert_disjoint_groups(train, validation, subject_arr)
        folds.append(Split(train=train, validation=validation))
    return folds


def outer_loso(subjects: Iterable[int]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    subject_arr = np.asarray(list(subjects), dtype=np.int64)
    for held_out in np.unique(subject_arr):
        train = np.flatnonzero(subject_arr != held_out).astype(np.int64)
        test = np.flatnonzero(subject_arr == held_out).astype(np.int64)
        if set(subject_arr[train].tolist()) & set(subject_arr[test].tolist()):
            raise AssertionError("Outer LOSO subject leakage")
        yield int(held_out), train, test
