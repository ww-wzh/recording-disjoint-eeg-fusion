"""Run the formal EEGNet nested-LOSO experiment. No model/protocol arguments are needed."""

from __future__ import annotations

import sys

from run_deep_baselines import main


if __name__ == "__main__":
    main(["--model", "eegnet", "--protocol", "loso", "--device", "cuda:0", *sys.argv[1:]])
