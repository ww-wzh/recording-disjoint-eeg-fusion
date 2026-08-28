"""编号75：Route A v3 fixed-blend and recording-level stacking baselines。

固定权重基线只使用已经冻结的 always-NN/always-fuse window probabilities。
LOSO 的 recording-level stacker 使用 01_core 中保存的 nested meta-recording
OOF 预测；它只覆盖 LOSO，不伪造 cross-task 的 OOF 训练数据。

运行方式：直接在 PyCharm 中运行本文件，不需要命令行参数。
输出目录：05_baselines
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parents[1]
CORE_ROOT = HERE / "01_core"
FROZEN_ROOT = HERE / "02_frozen"
OUTPUT_ROOT = HERE / "05_baselines"
POLICY_PATH = HERE / "64_冻结_RouteA_v3协议.json"
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(REVISION_ROOT / "route_a"))

from revision_pipeline.aggregation import aggregate_recordings, validate_frozen_predictions  # noqa: E402
from route_a_lib.probability import blend_probabilities  # noqa: E402


def _paired_window(window: pd.DataFrame, protocol: str, subject: int, direction: str) -> pd.DataFrame:
    cell = window[
        (window["protocol"] == protocol)
        & (window["subject"].astype(int) == int(subject))
        & (window["direction"] == direction)
    ]
    keys = ["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "true_label"]
    nn = cell[cell["method"] == "always_nn"]
    fuse = cell[cell["method"] == "always_fuse"]
    return nn[keys + ["p0", "p1"]].merge(
        fuse[keys + ["p0", "p1"]],
        on=keys,
        suffixes=("_nn", "_fuse"),
        validate="one_to_one",
    )


def _fixed_rows(window: pd.DataFrame, method: str, weight: float) -> pd.DataFrame:
    rows = []
    for (protocol, subject, direction), _ in window.groupby(["protocol", "subject", "direction"], sort=True):
        paired = _paired_window(window, str(protocol), int(subject), str(direction))
        probability = blend_probabilities(
            paired[["p0_nn", "p1_nn"]].to_numpy(dtype=np.float64),
            paired[["p0_fuse", "p1_fuse"]].to_numpy(dtype=np.float64),
            float(weight),
        )
        out = paired[["dataset", "protocol", "subject", "direction", "recording_id", "window_id", "true_label"]].copy()
        out["method"] = method
        out["p0"] = probability[:, 0]
        out["p1"] = probability[:, 1]
        out["ensemble_size"] = 5
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _logit(p: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def _median_meta_recordings(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    keys = ["outer_subject", "meta_subject", "recording_id", "method"]
    labels = raw.groupby(keys, as_index=False)["true_label"].agg(lambda x: int(np.unique(x)[0]))
    probabilities = raw.groupby(keys, as_index=False)[["p0", "p1"]].median()
    output = labels.merge(probabilities, on=keys, validate="one_to_one")
    mass = output[["p0", "p1"]].sum(axis=1)
    output[["p0", "p1"]] = output[["p0", "p1"]].div(mass, axis=0)
    return output


def _stacker_rows(core_recordings: pd.DataFrame, task: str, cohort: list[int]) -> pd.DataFrame:
    meta_path = CORE_ROOT / f"loso_{task}" / "loso_meta_recording_seed.csv"
    meta = _median_meta_recordings(meta_path)
    rows = []
    for outer_subject in cohort:
        train = meta[meta["outer_subject"].astype(int) == int(outer_subject)].copy()
        pivot = train.pivot_table(
            index=["meta_subject", "recording_id", "true_label"],
            columns="method",
            values=["p0", "p1"],
            aggfunc="first",
        ).reset_index()
        required = {("p1", "always_nn"), ("p1", "always_fuse")}
        if not required.issubset(set(pivot.columns)):
            raise AssertionError(f"Missing NN/fuse meta probabilities for {task} outer S{outer_subject}")
        x_train = np.column_stack([
            _logit(pivot[("p1", "always_nn")].to_numpy()),
            _logit(pivot[("p1", "always_fuse")].to_numpy()),
        ])
        y_train = pivot["true_label"].to_numpy(dtype=np.int64)
        if np.unique(y_train).size != 2:
            raise AssertionError(f"Stacker training has one class for {task} outer S{outer_subject}")
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=1335 + int(outer_subject))
        model.fit(x_train, y_train)

        target = core_recordings[
            (core_recordings["protocol"] == "loso")
            & (core_recordings["subject"].astype(int) == int(outer_subject))
            & (core_recordings["direction"] == task)
        ].copy()
        nn = target[target["method"] == "always_nn"]
        fuse = target[target["method"] == "always_fuse"]
        keys = ["dataset", "protocol", "subject", "direction", "recording_id", "true_label"]
        paired = nn[keys + ["p0", "p1", "n_windows"]].merge(
            fuse[keys + ["p0", "p1"]],
            on=keys,
            suffixes=("_nn", "_fuse"),
            validate="one_to_one",
        )
        x_test = np.column_stack([_logit(paired["p1_nn"].to_numpy()), _logit(paired["p1_fuse"].to_numpy())])
        probability = model.predict_proba(x_test)
        output = paired[keys].copy()
        output["method"] = "stack_recording"
        output["p0"] = probability[:, 0]
        output["p1"] = probability[:, 1]
        output["pred_label"] = np.argmax(probability, axis=1).astype(int)
        output["correct"] = (
            output["pred_label"].to_numpy(dtype=int) == output["true_label"].to_numpy(dtype=int)
        ).astype(int)
        output["n_windows"] = paired["n_windows"].to_numpy(dtype=int)
        output["ensemble_size"] = 5
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    window = pd.read_csv(FROZEN_ROOT / "core_window_ensemble.csv")
    core_recordings = aggregate_recordings(window)
    generated = [
        _fixed_rows(window, "fixed_blend_010", 0.10),
        _fixed_rows(window, "fixed_blend_025", 0.25),
        _fixed_rows(window, "equal_blend_050", 0.50),
    ]
    arithmetic_stack = _stacker_rows(core_recordings, "arithmetic", list(range(1, 16)))
    stroop_stack = _stacker_rows(core_recordings, "stroop", list(range(1, 14)))
    stack_recordings = pd.concat([arithmetic_stack, stroop_stack], ignore_index=True)
    baseline_window = pd.concat(generated, ignore_index=True)
    baseline_recordings = aggregate_recordings(baseline_window)
    validate_frozen_predictions(baseline_recordings)
    validate_frozen_predictions(stack_recordings)
    combined = pd.concat([core_recordings, baseline_recordings, stack_recordings], ignore_index=True)
    validate_frozen_predictions(combined)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_window.to_csv(OUTPUT_ROOT / "fixed_blend_window_predictions.csv", index=False, lineterminator="\n")
    baseline_recordings.to_csv(OUTPUT_ROOT / "fixed_blend_recording_predictions.csv", index=False, lineterminator="\n")
    stack_recordings.to_csv(OUTPUT_ROOT / "stack_recording_predictions_loso.csv", index=False, lineterminator="\n")
    combined.to_csv(OUTPUT_ROOT / "recording_predictions_with_core.csv", index=False, lineterminator="\n")
    manifest = {
        "status": "passed",
        "fixed_methods": ["fixed_blend_010", "fixed_blend_025", "equal_blend_050"],
        "stacker_method": "stack_recording",
        "stacker_scope": "nested LOSO only; two-branch recording-level OOF stacker",
        "cross_task_stacker_scope": "not generated because v3 cross-task checkpoints do not store source-task OOF predictions",
        "fixed_recording_rows": int(len(baseline_recordings)),
        "stack_recording_rows": int(len(stack_recordings)),
        "combined_recording_rows": int(len(combined)),
        "old_route_a_results_read": False,
        "protocol_version": policy["protocol_version"],
    }
    (OUTPUT_ROOT / "75_baselines_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Fixed blends and LOSO recording stacker completed")
    print(baseline_recordings.groupby(["protocol", "method"], as_index=False)["correct"].mean().to_string(index=False))
    print(stack_recordings.groupby(["protocol", "method"], as_index=False)["correct"].mean().to_string(index=False))


if __name__ == "__main__":
    main()
