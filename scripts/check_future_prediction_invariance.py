"""Compare real validation-prefix predictions before and after future edits.

This check uses only the observed prefix at each saved cutoff. It never trains
a model or reads the separate FD001 test answers.
"""

from __future__ import annotations

import hashlib
import json
import math

import sklearn

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.services import predict_rul

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_future_and_append(history, cutoff):
    changed = [dict(row) for row in history]
    for row in changed:
        if row["cycle"] > cutoff:
            extreme = 1e200 if row["cycle"] % 2 == 0 else -1e200
            row.update(dict.fromkeys(READING_COLUMNS, extreme))
    latest = max(row["cycle"] for row in changed)
    for cycle in range(latest + 1, latest + 6):
        extreme = 1e200 if cycle % 2 == 0 else -1e200
        changed.append(
            {
                "unit_id": changed[0]["unit_id"],
                "cycle": cycle,
                **dict.fromkeys(READING_COLUMNS, extreme),
            }
        )
    return changed


def run_invariance_check():
    """Return actual model/data provenance and per-prefix equality evidence."""
    source = PROJECT_ROOT / "data/raw/train_FD001.txt"
    split_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    manifest_path = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    model_path = manifest_path.parent / "decision_tree_model.joblib"
    if _sha256(source) != TRAIN_SHA256:
        raise ValueError("FD001/train source hash does not match the reviewed source.")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))["payload"]
    fitting_ids = {
        (item["dataset_partition"], item["unit_id"]) for item in split["fitting_engines"]
    }
    validation_ids = {
        (item["dataset_partition"], item["unit_id"])
        for item in split["validation_prefixes"]
    }
    if fitting_ids & validation_ids:
        raise ValueError("The saved development fitting and validation engines overlap.")
    if len(fitting_ids) != bundle["training"]["fitting_engine_count"]:
        raise ValueError("Saved fitting-engine count differs from the reviewed split.")
    if len(split["validation_prefixes"]) != 20:
        raise ValueError("Expected the reviewed 20-engine validation prefix set.")

    by_unit = {}
    for row in load_engine_history(source):
        by_unit.setdefault(row["unit_id"], []).append(row)

    compared = []
    for entry in split["validation_prefixes"]:
        unit, cutoff = entry["unit_id"], entry["cutoff_cycle"]
        history = by_unit[unit]
        original_prefix = [row for row in history if row["cycle"] <= cutoff]
        altered = _replace_future_and_append(history, cutoff)
        altered_prefix = [row for row in altered if row["cycle"] <= cutoff]
        if len(original_prefix) != entry["prefix_row_count"]:
            raise ValueError(f"Saved prefix count mismatch for engine {unit}.")

        common = {
            "dataset_partition": "FD001/train",
            "unit_id": unit,
            "manifest_path": manifest_path,
        }
        original = predict_rul(original_prefix, **common).raw_rul_cycles
        changed = predict_rul(altered_prefix, **common).raw_rul_cycles
        difference = abs(original - changed)
        if not math.isfinite(difference):
            raise ValueError(f"Nonfinite prediction difference for engine {unit}.")
        compared.append(
            {
                "unit_id": unit,
                "cutoff_cycle": cutoff,
                "raw_rul_cycles": original,
                "changed_future_raw_rul_cycles": changed,
                "abs_difference_cycles": difference,
            }
        )

    if any(row["abs_difference_cycles"] != 0.0 for row in compared):
        raise AssertionError("An earlier prediction changed after future readings were edited.")
    return {
        "source_sha256": _sha256(source),
        "split_manifest_sha256": _sha256(split_path),
        "model_name": bundle["model"]["name"],
        "model_version": bundle["model"]["version"],
        "model_sha256": _sha256(model_path),
        "feature_version": bundle["training"]["feature_config"]["version"],
        "window_cycles": bundle["training"]["feature_config"]["window_cycles"],
        "random_seed": bundle["training"]["random_seed"],
        "fitting_engine_count": len(fitting_ids),
        "validation_engine_count": len(validation_ids),
        "fit_validation_overlap": len(fitting_ids & validation_ids),
        "scikit_learn_version": sklearn.__version__,
        "compared_prefixes": len(compared),
        "exact_matches": sum(row["abs_difference_cycles"] == 0.0 for row in compared),
        "max_abs_difference_cycles": max(row["abs_difference_cycles"] for row in compared),
        "prediction_min_cycles": min(row["raw_rul_cycles"] for row in compared),
        "prediction_max_cycles": max(row["raw_rul_cycles"] for row in compared),
        "examples": compared,
    }


def main():
    print(json.dumps(run_invariance_check(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
