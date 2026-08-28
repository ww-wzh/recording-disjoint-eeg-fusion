from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT))

from eeg_channel_selection import select_openbci_eeg_channels  # noqa: E402


class ChannelSelectionTests(unittest.TestCase):
    def test_removes_regular_modulo_packet_counter(self) -> None:
        counter = np.arange(1024, dtype=np.float64) % 256
        eeg = np.arange(1024 * 8, dtype=np.float64).reshape(1024, 8) + 1000.25
        auxiliary = np.zeros((1024, 3), dtype=np.float64)
        selected, report = select_openbci_eeg_channels(np.column_stack([counter, eeg, auxiliary]))
        self.assertTrue(report.packet_counter_removed)
        self.assertTrue(np.array_equal(selected, eeg.astype(np.float32)))

    def test_removes_irregular_but_bounded_packet_counter(self) -> None:
        counter = np.tile(np.arange(256, dtype=np.float64), 5)
        counter[300:500] = counter[100:300]
        eeg = np.random.default_rng(7).normal(30_000.0, 20.0, size=(len(counter), 8))
        selected, report = select_openbci_eeg_channels(np.column_stack([counter, eeg]))
        self.assertTrue(report.packet_counter_removed)
        self.assertTrue(np.array_equal(selected, eeg.astype(np.float32)))

    def test_fails_closed_for_ambiguous_extra_columns(self) -> None:
        signal = np.random.default_rng(8).normal(size=(1024, 12))
        with self.assertRaisesRegex(ValueError, "Refusing to guess"):
            select_openbci_eeg_channels(signal)

    def test_accepts_already_selected_eight_columns(self) -> None:
        eeg = np.random.default_rng(9).normal(size=(1024, 8)).astype(np.float32)
        selected, report = select_openbci_eeg_channels(eeg)
        self.assertFalse(report.packet_counter_removed)
        self.assertTrue(np.array_equal(selected, eeg))


if __name__ == "__main__":
    unittest.main()
