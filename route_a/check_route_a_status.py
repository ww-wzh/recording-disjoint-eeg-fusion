from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def checkpoint_count(path: Path) -> int:
    return len(list(path.glob("*.csv"))) if path.exists() else 0


def raw_methods(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        frame = pd.read_csv(path, usecols=["method"])
    except Exception as exc:
        return f"unreadable ({exc})"
    return ",".join(sorted(frame["method"].unique()))


def main() -> None:
    rows = [
        {
            "component": "lightweight recording baselines",
            "protocol": "both",
            "completed": int((HERE / "results" / "lightweight_recording_predictions.csv").exists()),
            "expected": 1,
        },
        {
            "component": "leakage audit",
            "protocol": "both tasks",
            "completed": int((HERE / "results" / "leakage_audit" / "leakage_audit_summary.csv").exists()),
            "expected": 1,
        },
    ]
    checkpoint_specs = [
        ("riemannian", "cross_task", HERE / "results" / "riemannian" / "checkpoints" / "cross_task", 150),
        ("riemannian", "loso", HERE / "results" / "riemannian" / "checkpoints" / "loso", 75),
        ("dasf_clean", "cross_task", HERE / "results" / "dasf_clean" / "checkpoints" / "cross_task", 150),
        ("dasf_clean", "loso", HERE / "results" / "dasf_clean" / "checkpoints" / "loso", 75),
        ("eegnet", "cross_task", HERE / "results" / "deep" / "eegnet" / "cross_task", 150),
        ("eegnet", "loso", HERE / "results" / "deep" / "eegnet" / "loso", 75),
        (
            "eeg_conformer",
            "cross_task",
            HERE / "results" / "deep" / "eeg_conformer" / "cross_task",
            150,
        ),
        ("eeg_conformer", "loso", HERE / "results" / "deep" / "eeg_conformer" / "loso", 75),
    ]
    for component, protocol, path, expected in checkpoint_specs:
        rows.append(
            {
                "component": component,
                "protocol": protocol,
                "completed": checkpoint_count(path),
                "expected": expected,
            }
        )
    cbsf_checkpoint_root = HERE / "results" / "cbsf_loso" / "checkpoints"
    rows.append(
        {
            "component": "cbsf",
            "protocol": "loso_meta_oof",
            "completed": len(list(cbsf_checkpoint_root.glob("*.manifest.json")))
            if cbsf_checkpoint_root.exists()
            else 0,
            "expected": 75,
        }
    )
    robust_root = HERE / "results" / "cbsf_robustness_r1"
    rows.append(
        {
            "component": "cbsf",
            "protocol": "robustness_final",
            "completed": int(
                (robust_root / "recording_predictions.csv").exists()
                and (robust_root / "audit_manifest.json").exists()
            ),
            "expected": 1,
        }
    )
    rows.append(
        {
            "component": "cbsf",
            "protocol": "loso_final",
            "completed": int((HERE / "results" / "cbsf_loso" / "recording_predictions.csv").exists()),
            "expected": 1,
        }
    )
    status = pd.DataFrame(rows)
    status["ready"] = status["completed"] == status["expected"]
    print(status.to_string(index=False))
    print("\nConsolidated raw methods:")
    raw_files = {
        "riemannian": HERE / "results" / "riemannian" / "raw_seed_predictions.csv",
        "dasf_clean": HERE / "results" / "dasf_clean" / "raw_seed_predictions.csv",
        "eegnet": HERE / "results" / "deep" / "eegnet" / "raw_seed_predictions.csv",
        "eeg_conformer": HERE / "results" / "deep" / "eeg_conformer" / "raw_seed_predictions.csv",
        "cbsf_loso_audit": HERE / "results" / "cbsf_loso" / "recording_predictions.csv",
        "cbsf_robust": robust_root / "recording_predictions.csv",
    }
    for name, path in raw_files.items():
        print(f"  {name:16s} {raw_methods(path)}")
    print("\nRoute A freeze is allowed only when every row above is ready=True.")


if __name__ == "__main__":
    main()
