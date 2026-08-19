"""FORMAL EXPERIMENT: nested participant-level LOSO evaluation of CB-SF.

Input:
    The frozen 8-channel corrected OpenBCI predictions and clean-DASF inner epochs.
Output:
    ``route_a/results/cbsf_loso`` with resumable OOF checkpoints, gate weights,
    recording predictions, paired statistics and an audit manifest.

Run this file directly in PyCharm. No command-line parameters are required.
Do not run the internal ``cbsf_loso_engine.py`` module directly.
"""

from cbsf_loso_engine import run_formal


if __name__ == "__main__":
    run_formal()
