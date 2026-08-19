from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eeg_feature_pipeline as feature_pipeline  # noqa: E402


def test_frozen_feature_settings_are_explicit() -> None:
    assert feature_pipeline.NOTCH_LINE_FREQ_HZ == 50.0
    assert feature_pipeline.NOTCH_LINE_WIDTH_HZ == 1.5
    assert feature_pipeline.NOTCH_LINE_ATTEN == 0.03
    assert feature_pipeline.USE_BANDPASS_FILTER is True
    assert feature_pipeline.BANDPASS_LOW_HZ == 0.5
    assert feature_pipeline.BANDPASS_HIGH_HZ == 55.0
    assert feature_pipeline.BANDPASS_ORDER == 4
    assert feature_pipeline.WORKLOAD_THETA_ALPHA_REL_BOOST == 1.10
    assert feature_pipeline.USE_DIFFERENTIAL_ENTROPY is True
    assert feature_pipeline.USE_INTER_CHANNEL_CORR is False
    assert feature_pipeline.FRONTAL_FEATURE_MODE == "first_half"
    assert feature_pipeline.PREFRONTAL_CHANNEL_INDICES is None
    assert feature_pipeline.FAA_LEFT_RIGHT_ALPHA is None
    assert feature_pipeline.FRONTAL_REL_POWER_WEIGHT == 1.12


def test_frozen_eight_channel_input_produces_272_features(tmp_path: Path) -> None:
    sfreq = 250.0
    n_samples = 2250
    time = np.arange(n_samples, dtype=np.float32) / sfreq
    rng = np.random.default_rng(20260722)
    channels = []
    for channel in range(8):
        signal = (
            np.sin(2.0 * np.pi * (4.0 + channel) * time)
            + 0.25 * np.sin(2.0 * np.pi * (10.0 + 0.5 * channel) * time)
            + 0.02 * rng.standard_normal(n_samples)
        )
        channels.append(signal.astype(np.float32))
    eeg = np.stack(channels, axis=1)

    features, covariances = feature_pipeline.extract_features_from_eeg_windows(
        eeg=eeg,
        sfreq=sfreq,
        win_sec=feature_pipeline.WIN_SEC,
        stride_sec=feature_pipeline.STRIDE_SEC,
        bands=feature_pipeline.BANDS,
        device=torch.device("cpu"),
        cache_key="deterministic_frozen_feature_test",
        cache_dir=str(tmp_path),
        chunk_size=4,
    )

    assert features.shape[1] == feature_pipeline.EXPECTED_FEATURE_DIM == 272
    assert covariances.shape[1:] == (8, 8)
    assert torch.isfinite(features).all()
    assert torch.isfinite(covariances).all()
