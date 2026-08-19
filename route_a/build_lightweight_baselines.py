from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION_ROOT = HERE.parent
sys.path.insert(0, str(REVISION_ROOT))
sys.path.insert(0, str(HERE))

from revision_pipeline.aggregation import (  # noqa: E402
    RAW_KEYS,
    aggregate_recordings,
    median_seed_ensemble,
    sha256_file,
    validate_frozen_predictions,
)
from route_a_lib.probability import (  # noqa: E402
    CELL_KEYS,
    blend_probabilities,
    fit_recording_stacker,
    recording_rows_from_probabilities,
)


BLENDS = {
    "fixed_blend_010": 0.10,
    "fixed_blend_025": 0.25,
    "equal_blend_050": 0.50,
}
SEEDS = [1335, 1388, 1441, 1494, 1547]
STACK_INPUTS = ("always_nn", "always_fuse")


def fixed_blend_rows(raw_seed: pd.DataFrame) -> pd.DataFrame:
    base = raw_seed[raw_seed["method"].isin(["always_nn", "always_fuse"])].copy()
    windows = median_seed_ensemble(base, SEEDS)
    join_keys = [key for key in RAW_KEYS if key != "method"] + ["true_label", "ensemble_size"]
    nn = windows[windows["method"] == "always_nn"][join_keys + ["p0", "p1"]].rename(
        columns={"p0": "nn_p0", "p1": "nn_p1"}
    )
    fuse = windows[windows["method"] == "always_fuse"][join_keys + ["p0", "p1"]].rename(
        columns={"p0": "fuse_p0", "p1": "fuse_p1"}
    )
    paired = nn.merge(fuse, on=join_keys, how="inner", validate="one_to_one")
    if len(paired) * 2 != len(windows):
        raise ValueError("always_nn and always_fuse do not cover identical windows")

    outputs = []
    nn_probability = paired[["nn_p0", "nn_p1"]].to_numpy(dtype=np.float64)
    fuse_probability = paired[["fuse_p0", "fuse_p1"]].to_numpy(dtype=np.float64)
    for method, weight in BLENDS.items():
        probability = blend_probabilities(nn_probability, fuse_probability, weight)
        out = paired[join_keys].copy()
        out["method"] = method
        out["p0"] = probability[:, 0]
        out["p1"] = probability[:, 1]
        outputs.append(out[RAW_KEYS + ["true_label", "p0", "p1", "ensemble_size"]])
    return aggregate_recordings(pd.concat(outputs, ignore_index=True))


def _median_meta_recordings(meta_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["outer_subject", "meta_subject", "recording_id", "method"]
    expected = set(SEEDS)
    observed = meta_seed.groupby(keys)["seed"].agg(lambda x: set(int(v) for v in x))
    if observed.map(lambda x: x != expected).any():
        raise ValueError("LOSO meta predictions do not contain exactly five seeds per recording/method")
    labels = meta_seed.groupby(keys, as_index=False)["true_label"].agg(
        lambda x: int(np.unique(x)[0]) if np.unique(x).size == 1 else -1
    )
    if (labels["true_label"] < 0).any():
        raise ValueError("LOSO meta prediction labels are inconsistent")
    probability = meta_seed.groupby(keys, as_index=False)[["p0", "p1"]].median()
    out = labels.merge(probability, on=keys, validate="one_to_one")
    mass = out[["p0", "p1"]].sum(axis=1)
    out[["p0", "p1"]] = out[["p0", "p1"]].div(mass, axis=0)
    out["dataset"] = "openbci"
    out["protocol"] = "loso"
    out["subject"] = out["meta_subject"].astype(int)
    out["direction"] = "arithmetic"
    return out


def recording_stacker_rows(
    frozen: pd.DataFrame,
    loso_meta_seed: pd.DataFrame,
    *,
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frozen[frozen["method"].isin(STACK_INPUTS)].copy()
    outputs = []
    diagnostics = []

    cross = base[base["protocol"] == "cross_task"]
    for direction in sorted(cross["direction"].unique()):
        direction_rows = cross[cross["direction"] == direction]
        for subject in sorted(direction_rows["subject"].unique()):
            train = direction_rows[direction_rows["subject"] != subject]
            test = direction_rows[direction_rows["subject"] == subject]
            probability = fit_recording_stacker(
                train, test, c_value=c_value, seed=20260717 + int(subject), methods=STACK_INPUTS
            )
            outputs.append(
                recording_rows_from_probabilities(
                    test, probability, "stack_recording", input_methods=STACK_INPUTS
                )
            )
            diagnostics.append(
                {
                    "protocol": "cross_task",
                    "subject": int(subject),
                    "direction": direction,
                    "meta_training_participants": int(train["subject"].nunique()),
                    "meta_training_recordings": int(train["recording_id"].nunique()),
                    "target_participant_excluded": True,
                    "meta_source": "participant-cross-fitted outer recording predictions",
                }
            )

    meta = _median_meta_recordings(loso_meta_seed)
    loso = base[base["protocol"] == "loso"]
    for subject in sorted(loso["subject"].unique()):
        train = meta[meta["outer_subject"] == subject]
        test = loso[loso["subject"] == subject]
        probability = fit_recording_stacker(
            train, test, c_value=c_value, seed=20261717 + int(subject), methods=STACK_INPUTS
        )
        outputs.append(
            recording_rows_from_probabilities(
                test, probability, "stack_recording", input_methods=STACK_INPUTS
            )
        )
        diagnostics.append(
            {
                "protocol": "loso",
                "subject": int(subject),
                "direction": "arithmetic",
                "meta_training_participants": int(train["meta_subject"].nunique()),
                "meta_training_recordings": int(train["recording_id"].nunique()),
                "target_participant_excluded": bool(subject not in set(train["meta_subject"])),
                "meta_source": "outer-training inner-OOF recording predictions",
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(diagnostics)


def main() -> None:
    source_results = REVISION_ROOT / "results"
    source_frozen = REVISION_ROOT / "frozen" / "predictions_recording.csv"
    output = HERE / "results"
    output.mkdir(parents=True, exist_ok=True)

    raw_seed = pd.read_csv(source_results / "raw_seed_predictions.csv")
    frozen = pd.read_csv(source_frozen)
    meta = pd.read_csv(source_results / "loso_meta_recording_seed.csv")
    blends = fixed_blend_rows(raw_seed)
    stacker, diagnostics = recording_stacker_rows(frozen, meta, c_value=0.05)
    final = pd.concat([blends, stacker], ignore_index=True).sort_values(CELL_KEYS + ["method"])
    validate_frozen_predictions(final)

    result_path = output / "lightweight_recording_predictions.csv"
    final.to_csv(result_path, index=False, lineterminator="\n")
    diagnostics.to_csv(output / "recording_stacker_diagnostics.csv", index=False, lineterminator="\n")
    manifest = {
        "file": result_path.name,
        "sha256": sha256_file(result_path),
        "rows": int(len(final)),
        "methods": sorted(final["method"].unique()),
        "source_frozen_sha256": sha256_file(source_frozen),
        "source_raw_seed_sha256": sha256_file(source_results / "raw_seed_predictions.csv"),
        "protocol_route_a_sha256": sha256_file(HERE / "protocol_route_a.json"),
        "confirmatory_status": "secondary/exploratory on OpenBCI; freeze before external validation",
    }
    (result_path.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {result_path} ({len(final)} rows)")
    print(f"SHA-256: {manifest['sha256']}")


if __name__ == "__main__":
    main()
