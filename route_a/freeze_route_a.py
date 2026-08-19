from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))

from revision_pipeline.aggregation import (  # noqa: E402
    aggregate_recordings,
    freeze_csv,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)


SEEDS = [1335, 1388, 1441, 1494, 1547]
RAW_SOURCES = {
    "dasf_clean": HERE / "results" / "dasf_clean" / "raw_seed_predictions.csv",
    "riemannian": HERE / "results" / "riemannian" / "raw_seed_predictions.csv",
    "eegnet": HERE / "results" / "deep" / "eegnet" / "raw_seed_predictions.csv",
    "eeg_conformer": HERE / "results" / "deep" / "eeg_conformer" / "raw_seed_predictions.csv",
}
EXPECTED = {
    "cross_task": {
        "always_nn",
        "rf",
        "extra_trees",
        "always_fuse",
        "cbsf",
        "fixed_blend_010",
        "fixed_blend_025",
        "equal_blend_050",
        "stack_recording",
        "dasf_clean",
        "riemann_ts_logreg",
        "riemann_mdm",
        "eegnet",
        "eeg_conformer",
    },
    "loso": {
        "always_nn",
        "rf",
        "extra_trees",
        "always_fuse",
        "fixed_blend_010",
        "fixed_blend_025",
        "equal_blend_050",
        "stack_recording",
        "dasf_clean",
        "riemann_ts_logreg",
        "riemann_mdm",
        "eegnet",
        "eeg_conformer",
        "cbsf",
    },
}


def protocol_bundle_digest(output_dir: Path) -> str:
    payload = {
        "route_a_protocol": {
            "file": "protocol_route_a.json",
            "sha256": sha256_file(HERE / "protocol_route_a.json"),
        },
        "cbsf_loso_protocol": {
            "file": "protocol_cbsf_loso.json",
            "sha256": sha256_file(HERE / "protocol_cbsf_loso.json"),
        },
        "cbsf_robustness_protocol": {
            "file": "protocol_cbsf_robustness.json",
            "sha256": sha256_file(HERE / "protocol_cbsf_robustness.json"),
        },
        "final_submission_protocol": {
            "file": "protocol_final_submission.json",
            "sha256": sha256_file(HERE / "protocol_final_submission.json"),
        },
        "parent_corrected_protocol": {
            "file": "../protocol.json",
            "canonical_sha256": json.loads(
                (HERE / "protocol_cbsf_loso.json").read_text(encoding="utf-8")
            )["parent_corrected_protocol_sha256"],
        },
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest = {**payload, "protocol_bundle_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return digest


def assert_complete(final: pd.DataFrame) -> None:
    for protocol, expected_methods in EXPECTED.items():
        subset = final[final["protocol"] == protocol]
        observed = set(subset["method"].unique())
        if observed != expected_methods:
            missing = sorted(expected_methods - observed)
            extra = sorted(observed - expected_methods)
            raise ValueError(f"{protocol} methods incomplete; missing={missing}, extra={extra}")
        expected_recordings = set(
            subset[subset["method"] == "always_nn"][
                ["dataset", "subject", "direction", "recording_id"]
            ].itertuples(index=False, name=None)
        )
        for method in expected_methods:
            observed_recordings = set(
                subset[subset["method"] == method][
                    ["dataset", "subject", "direction", "recording_id"]
                ].itertuples(index=False, name=None)
            )
            if observed_recordings != expected_recordings:
                raise ValueError(
                    f"{protocol}/{method} does not cover the same recordings as always_nn "
                    f"({len(observed_recordings)} vs {len(expected_recordings)})"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the complete Route A recording prediction table")
    parser.add_argument("--force", action="store_true", help="Replace an existing Route A freeze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_protocol = json.loads((HERE / "protocol_route_a.json").read_text(encoding="utf-8"))
    parent_path = REVISION_ROOT / "frozen" / "predictions_recording.csv"
    if sha256_file(parent_path) != route_protocol["parent_prediction_sha256"]:
        raise RuntimeError("The parent frozen prediction file changed")
    missing_files = [str(path) for path in RAW_SOURCES.values() if not path.exists()]
    lightweight_path = HERE / "results" / "lightweight_recording_predictions.csv"
    cbsf_loso_path = HERE / "results" / "cbsf_loso" / "recording_predictions.csv"
    robust_root = HERE / "results" / "cbsf_robustness_r1"
    robust_path = robust_root / "recording_predictions.csv"
    robust_audit_path = robust_root / "audit_manifest.json"
    if not lightweight_path.exists():
        missing_files.append(str(lightweight_path))
    if not cbsf_loso_path.exists():
        missing_files.append(str(cbsf_loso_path))
    if not robust_path.exists():
        missing_files.append(str(robust_path))
    if not robust_audit_path.exists():
        missing_files.append(str(robust_audit_path))
    if missing_files:
        raise FileNotFoundError("Route A runs are incomplete. Missing:\n" + "\n".join(missing_files))

    final_protocol = json.loads((HERE / "protocol_final_submission.json").read_text(encoding="utf-8"))
    robust_protocol = json.loads((HERE / "protocol_cbsf_robustness.json").read_text(encoding="utf-8"))
    robust_audit = json.loads(robust_audit_path.read_text(encoding="utf-8"))
    if sha256_file(robust_path) != final_protocol["robust_prediction_sha256"]:
        raise RuntimeError("The selected robust CB-SF prediction file changed")
    if sha256_file(HERE / "protocol_cbsf_robustness.json") != final_protocol["robust_protocol_sha256"]:
        raise RuntimeError("The selected robust CB-SF protocol changed")
    if sha256_file(cbsf_loso_path) != robust_protocol["source_cbsf_loso_recording_prediction_sha256"]:
        raise RuntimeError("The nested-LOSO source used by the robustness analysis changed")
    if not robust_audit.get("post_hoc_robustness_analysis"):
        raise RuntimeError("The robust CB-SF audit does not disclose its post-hoc status")

    raw = pd.concat([pd.read_csv(path) for path in RAW_SOURCES.values()], ignore_index=True)
    route_windows = median_seed_ensemble(raw, SEEDS)
    route_recordings = aggregate_recordings(route_windows)
    parent = pd.read_csv(parent_path)
    parent = parent[parent["method"] != "risk_aware"].copy()
    robust = pd.read_csv(robust_path)
    if set(robust["method"].astype(str)) != {"risk_aware_robust"}:
        raise ValueError("The selected CB-SF robustness file has an unexpected method label")
    if set(robust["protocol"].astype(str)) != {"cross_task", "loso"} or len(robust) != 180:
        raise ValueError("The selected CB-SF robustness file must contain 180 rows for both protocols")
    robust["method"] = "cbsf"
    final = pd.concat(
        [
            parent,
            pd.read_csv(lightweight_path),
            route_recordings,
            robust,
        ],
        ignore_index=True,
    )
    duplicate_keys = ["dataset", "protocol", "subject", "direction", "recording_id", "method"]
    if final.duplicated(duplicate_keys).any():
        duplicates = final[final.duplicated(duplicate_keys, keep=False)][duplicate_keys]
        raise ValueError(f"Duplicate final Route A rows:\n{duplicates.head().to_string(index=False)}")
    final = final.sort_values(duplicate_keys).reset_index(drop=True)
    validate_frozen_predictions(final)
    assert_complete(final)

    frozen_dir = HERE / "frozen"
    output = frozen_dir / "predictions_recording_route_a.csv"
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; use --force only after documenting why")
    bundle_digest = protocol_bundle_digest(frozen_dir)
    freeze_csv(final, output, bundle_digest)
    print(f"Frozen Route A predictions: {output}")
    print(f"SHA-256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
