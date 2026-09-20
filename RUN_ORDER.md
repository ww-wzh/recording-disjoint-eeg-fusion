# v7 neutral reproducibility order

The canonical frozen predictions are already included. Verification does not
retrain models:

1. Run `python verify_release.py`.
2. Run `python heldout_gate/verify_heldout_gate.py`.
3. Run `pytest -q`.

For a full refit from the public raw EEG, set `CBSF_DATA_ROOT` and run
`run_corrected_experiments.py` with the v7 protocol. The repository contains
maintained model adapters under `route_a/`; private numbered audit runners and
local output directories are intentionally not part of the public release.

The matched three-split audit protocol is documented in
`manuscript_artifacts/v7_neutral/v7_neutral_matched_three_split_protocol.md`.
