from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from revision_pipeline.aggregation import sha256_file  # noqa: E402
from revision_pipeline.splits import outer_loso  # noqa: E402
from route_a_lib.data import FeatureBundle, load_feature_bundles  # noqa: E402
from route_a_lib.probability import normalize_probabilities  # noqa: E402


METHODS = ("riemann_ts_logreg", "riemann_mdm")


def _mdm_probabilities(model, covariances: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return normalize_probabilities(model.predict_proba(covariances))
    distances = np.asarray(model.transform(covariances), dtype=np.float64)
    shifted = -distances - np.max(-distances, axis=1, keepdims=True)
    return normalize_probabilities(np.exp(shifted))


def fit_predict_riemannian(
    train_covariances: np.ndarray,
    train_labels: np.ndarray,
    test_covariances: np.ndarray,
    *,
    seed: int,
    config: dict,
) -> dict[str, np.ndarray]:
    try:
        from pyriemann.classification import MDM
        from pyriemann.tangentspace import TangentSpace
    except ImportError as exc:
        raise RuntimeError("pyriemann is required: pip install pyriemann") from exc

    train_labels = np.asarray(train_labels, dtype=np.int64)
    if np.unique(train_labels).size != 2:
        raise ValueError("Riemannian training fold is missing a class")
    tangent = TangentSpace(metric=str(config["tangent_metric"]))
    train_tangent = tangent.fit_transform(train_covariances)
    test_tangent = tangent.transform(test_covariances)
    scaler = StandardScaler().fit(train_tangent)
    logistic = LogisticRegression(
        C=float(config["logistic_C"]),
        class_weight="balanced",
        max_iter=int(config["logistic_max_iter"]),
        random_state=int(seed),
    )
    logistic.fit(scaler.transform(train_tangent), train_labels)
    tangent_probability = normalize_probabilities(
        logistic.predict_proba(scaler.transform(test_tangent))
    )

    mdm = MDM(metric=str(config["mdm_metric"]))
    mdm.fit(train_covariances, train_labels)
    mdm_probability = _mdm_probabilities(mdm, test_covariances)
    return {
        "riemann_ts_logreg": tangent_probability,
        "riemann_mdm": mdm_probability,
    }


def prediction_rows(
    target: FeatureBundle,
    probabilities: dict[str, np.ndarray],
    *,
    protocol: str,
    subject: int,
    direction: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for method, probability in probabilities.items():
        if len(probability) != len(target.labels):
            raise ValueError(f"{method} probability count differs from target windows")
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "openbci",
                    "protocol": protocol,
                    "subject": int(subject),
                    "direction": direction,
                    "recording_id": target.recordings,
                    "window_id": target.window_ids,
                    "method": method,
                    "seed": int(seed),
                    "true_label": target.labels,
                    "p0": probability[:, 0],
                    "p1": probability[:, 1],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def checkpoint_complete(path: Path, seed: int) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    return bool(
        not frame.empty
        and set(frame["method"]) == set(METHODS)
        and set(frame["seed"].astype(int)) == {int(seed)}
        and not frame.duplicated(["recording_id", "window_id", "method", "seed"]).any()
    )


def run_cross_task(
    arithmetic: FeatureBundle,
    stroop: FeatureBundle,
    subjects: list[int],
    seeds: list[int],
    config: dict,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        pairs = [
            (stroop, arithmetic, "stroop_to_arithmetic"),
            (arithmetic, stroop, "arithmetic_to_stroop"),
        ]
        for source_all, target_all, direction in pairs:
            source = source_all.subset(np.flatnonzero(source_all.subjects == subject))
            target = target_all.subset(np.flatnonzero(target_all.subjects == subject))
            missing = [
                seed
                for seed in seeds
                if not checkpoint_complete(
                    checkpoint_dir / f"s{subject:02d}_{direction}_seed{seed}.csv", seed
                )
            ]
            if not missing:
                print(f"[skip] Riemannian S{subject:02d} {direction} all seed slots")
                continue
            probability = fit_predict_riemannian(
                source.covariances,
                source.labels,
                target.covariances,
                seed=seeds[0],
                config=config,
            )
            for seed in missing:
                path = checkpoint_dir / f"s{subject:02d}_{direction}_seed{seed}.csv"
                prediction_rows(
                    target,
                    probability,
                    protocol="cross_task",
                    subject=subject,
                    direction=direction,
                    seed=seed,
                ).to_csv(path, index=False, lineterminator="\n")
            print(
                f"[done] Riemannian cross_task S{subject:02d} {direction} "
                f"seed_slots={missing} (one deterministic fit)"
            )


def run_loso(
    arithmetic: FeatureBundle,
    held_subjects: list[int],
    seeds: list[int],
    config: dict,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outer = {subject: (train, test) for subject, train, test in outer_loso(arithmetic.subjects)}
    for subject in held_subjects:
        train_index, test_index = outer[subject]
        source = arithmetic.subset(train_index)
        target = arithmetic.subset(test_index)
        missing = [
            seed
            for seed in seeds
            if not checkpoint_complete(checkpoint_dir / f"s{subject:02d}_arithmetic_seed{seed}.csv", seed)
        ]
        if not missing:
            print(f"[skip] Riemannian LOSO S{subject:02d} all seed slots")
            continue
        probability = fit_predict_riemannian(
            source.covariances,
            source.labels,
            target.covariances,
            seed=seeds[0],
            config=config,
        )
        for seed in missing:
            path = checkpoint_dir / f"s{subject:02d}_arithmetic_seed{seed}.csv"
            prediction_rows(
                target,
                probability,
                protocol="loso",
                subject=subject,
                direction="arithmetic",
                seed=seed,
            ).to_csv(path, index=False, lineterminator="\n")
        print(
            f"[done] Riemannian LOSO S{subject:02d} "
            f"seed_slots={missing} (one deterministic fit)"
        )


def consolidate(root: Path) -> None:
    files = sorted(root.glob("checkpoints/**/*.csv"))
    if not files:
        return
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "method", "seed"]
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(root / "raw_seed_predictions.csv", index=False, lineterminator="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route A 8-channel Riemannian baselines")
    parser.add_argument("--protocol", choices=("cross_task", "loso", "both"), default="both")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REVISION_ROOT,
        help="Repository root containing 1111.py and the public data entry point",
    )
    parser.add_argument("--smoke", action="store_true", help="Four participants and one seed; never use in paper")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads((HERE / "protocol_route_a.json").read_text(encoding="utf-8"))
    seeds = [int(protocol["seeds"][0])] if args.smoke else [int(value) for value in protocol["seeds"]]
    cohort = list(range(1, 5)) if args.smoke else list(range(1, 16))
    requested = [args.subject] if args.subject is not None else cohort
    if any(subject not in cohort for subject in requested):
        raise ValueError(f"Requested participants {requested} are outside cohort {cohort}")
    arithmetic, stroop = load_feature_bundles(args.repo_root.resolve(), torch.device("cpu"))
    if args.smoke:
        arithmetic = arithmetic.subset(np.flatnonzero(np.isin(arithmetic.subjects, cohort)))
        stroop = stroop.subset(np.flatnonzero(np.isin(stroop.subjects, cohort)))
    result_root = HERE / ("results_smoke" if args.smoke else "results") / "riemannian"
    result_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol_sha256": sha256_file(HERE / "protocol_route_a.json"),
        "channels": 8,
        "coral": False,
        "smoke": bool(args.smoke),
        "warning": "Smoke results must never enter the manuscript" if args.smoke else "",
    }
    try:
        import pyriemann

        metadata["pyriemann"] = pyriemann.__version__
    except ImportError as exc:
        raise RuntimeError("pyriemann is required: pip install pyriemann") from exc
    (result_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.protocol in ("cross_task", "both"):
        run_cross_task(
            arithmetic,
            stroop,
            requested,
            seeds,
            protocol["riemannian"],
            result_root / "checkpoints" / "cross_task",
        )
    if args.protocol in ("loso", "both"):
        run_loso(
            arithmetic,
            requested,
            seeds,
            protocol["riemannian"],
            result_root / "checkpoints" / "loso",
        )
    consolidate(result_root)
    print(f"Completed available Riemannian checkpoints under {result_root}")


if __name__ == "__main__":
    main()
