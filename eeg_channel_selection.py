"""Fail-closed OpenBCI EEG channel selection used by the corrected Route A rerun."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PARSER_SCHEMA = "openbci_packet_counter_removed_v2"


@dataclass(frozen=True)
class ChannelSelectionReport:
    input_columns: int
    output_columns: int
    packet_counter_removed: bool
    modulo_step_fraction: float
    reset_count: int
    schema: str = PARSER_SCHEMA


def packet_counter_diagnostics(values: np.ndarray) -> tuple[bool, float, int]:
    """Detect a 0..255 OpenBCI packet counter, including modulo resets."""
    column = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(column) < 256 or not np.isfinite(column).all():
        return False, 0.0, 0
    if not np.allclose(column, np.round(column), atol=1e-6):
        return False, 0.0, 0
    if float(column.min()) < 0.0 or float(column.max()) > 255.0:
        return False, 0.0, 0
    unique_count = int(np.unique(column).size)
    differences = np.diff(column)
    modulo_steps = np.mod(differences, 256.0)
    step_fraction = float(np.mean(np.isclose(modulo_steps, 1.0, atol=1e-6)))
    reset_count = int(np.sum(differences < 0.0))
    # Some deposited files contain packet loss, repeated blocks, or out-of-order
    # rows, so a strict +1 modulo sequence is not reliable. The bounded integer
    # support with broad 0..255 coverage is the fail-closed identifying signal;
    # EEG amplitudes in these files are not bounded integers in this range.
    detected = bool(unique_count >= 128 and reset_count >= 1)
    return detected, step_fraction, reset_count


def select_openbci_eeg_channels(signal: np.ndarray) -> tuple[np.ndarray, ChannelSelectionReport]:
    """Remove the packet counter and return exactly the next eight EEG columns.

    The deposited TXT layout starts with a modulo-256 packet counter, followed by
    eight EEG values and then accelerometer/status/timestamp fields. The legacy
    loader retained the counter because it was not strictly monotonic. This
    selector fails closed when extra columns are present but the expected counter
    cannot be identified.
    """
    values = np.asarray(signal)
    if values.ndim != 2:
        raise ValueError(f"OpenBCI signal must be two-dimensional; got {values.shape}")
    if values.shape[1] < 8:
        raise ValueError(f"Expected at least eight signal columns; got {values.shape[1]}")

    detected, step_fraction, reset_count = packet_counter_diagnostics(values[:, 0])
    if detected:
        if values.shape[1] < 9:
            raise ValueError("Packet counter detected but fewer than eight following EEG columns remain")
        eeg = values[:, 1:9]
    elif values.shape[1] == 8:
        eeg = values
    else:
        raise ValueError(
            "Extra OpenBCI columns are present, but column 0 is not the expected modulo-256 "
            "packet counter. Refusing to guess the EEG column boundary."
        )

    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.shape[1] != 8 or not np.isfinite(eeg).all():
        raise ValueError("Selected EEG matrix must contain eight finite columns")
    report = ChannelSelectionReport(
        input_columns=int(values.shape[1]),
        output_columns=8,
        packet_counter_removed=detected,
        modulo_step_fraction=step_fraction,
        reset_count=reset_count,
    )
    return eeg, report
