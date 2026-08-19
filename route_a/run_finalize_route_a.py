"""FINALIZATION: freeze Route A and generate all submission tables and figures.

Run this file directly in PyCharm after ``run_cbsf_robustness_analysis.py``.
No command-line parameters are required. The script fails closed if any selected
prediction or protocol hash has changed.
"""

from build_route_a_outputs import main as build_outputs
from freeze_route_a import main as freeze_predictions


if __name__ == "__main__":
    freeze_predictions()
    build_outputs()
