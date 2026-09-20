"""Portable compatibility entry point for the frozen neutral EEG extractor.

The historical runners import ``1111.py``.  Keeping this small forwarding module
preserves that interface while the maintained implementation lives in
``eeg_feature_pipeline.py``.
"""

from eeg_feature_pipeline import *  # noqa: F401,F403

