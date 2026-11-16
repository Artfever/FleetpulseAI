"""Compare every saved development replay point with standalone prefix inference.

This check reuses the reviewed P06 development model and its saved split. It
does not fit a model or read the separate FD001 test answers.
"""

from __future__ import annotations

import hashlib
import json
import math

import sklearn

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.services import predict_rul

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
EXPECTED_VALIDATION_COUNT = 20


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row):
    return (row["dataset_partition"], row["unit_id"], row["cutoff_cycle"])


def run_replay_check():
    """Compare each saved prediction to a fresh inference on its raw prefix."""
    source = PROJECT_ROOT / "data/raw/train_FD001.txt"
    split_path = PROJECT_ROOT / "artifacts/split_manifest.json"
    manifest_path = PROJECT_ROOT / "artifacts/development/decision_tree_validation.json"
    model_path = manifest_path.parent / "decision_tree_model.joblib"

    if not source.is_file() or not split_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Install the reviewed FD001/train, split, and P06 development bundle.")
    source_sha256 = _sha256(source)
    if source_sha256 != TRAIN_SHA256:
        raise ValueError("FD001/train source hash does not match the reviewed source.")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))["payload"]
    saved_rows = bundle["evaluation"]["predictions"]
    split_rows = split["validation_prefixes"]
    expected = {_identity(row): row for row in saved_rows}
    if len(expected) != len(saved_rows):
        raise ValueError("Saved evaluation predictions contain duplicate replay identities.")
    split_by_id = {_identity(row): row for row in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("Split manifest contains duplicate validation replay identities.")
    if set(expected) != set(split_by_id):
        raise ValueError("Saved predictions and split manifest replay identities do not match.")
    if not split_rows or len(split_rows) != EXPECTED_VALIDATION_COUNT:
        raise ValueError("Expected the reviewed 20-engine development validation split.")

    by_unit = {}
    for row in load_engine_history(source):
        by_unit.setdefault(row["unit_id"], []).append(row)

    compared = []
    for identity, entry in sorted(split_by_id.items()):
        partition, unit, cutoff = identity
        if partition != "FD001/train":
            raise ValueError(f"Unexpected replay partition: {partition}.")
        history = by_unit.get(unit)
        if not history:
            raise ValueError(f"Missing training history for engine {unit}.")
        prefix = [row for row in history if row["cycle"] <= cutoff]
        if len(prefix) != entry["prefix_row_count"]:
            raise ValueError(f"Saved prefix count mismatch for engine {unit}.")
        if not prefix or prefix[-1]["cycle"] != cutoff:
            raise ValueError(f"Standalone prefix does not end at saved cycle {cutoff}.")

        actual = predict_rul(
            prefix,
            dataset_partition=partition,
            unit_id=unit,
            manifest_path=manifest_path,
        ).raw_rul_cycles
        recorded = expected[identity]["prediction_cycles"]
        difference = abs(actual - recorded)
        if not math.isfinite(difference):
            raise ValueError(f"Nonfinite prediction difference at engine {unit}, cycle {cutoff}.")
        compared.append(
            {
                "dataset_partition": partition,
                "unit_id": unit,
                "cutoff_cycle": cutoff,
                "standalone_raw_rul_cycles": actual,
                "saved_raw_rul_cycles": recorded,
                "abs_difference_cycles": difference,
            }
        )

    mismatches = [row for row in compared if row["abs_difference_cycles"] != 0.0]
    if mismatches:
        first = mismatches[0]
        raise AssertionError(
            "Standalone prefix replay differs from saved output for "
            f"engine {first['unit_id']} cycle {first['cutoff_cycle']}: "
            f"{first['abs_difference_cycles']} cycles."
        )

    return {
        "source_sha256": source_sha256,
        "split_manifest_sha256": _sha256(split_path),
        "model_name": bundle["model"]["name"],
        "model_version": bundle["model"]["version"],
        "model_sha256": _sha256(model_path),
        "scikit_learn_version": sklearn.__version__,
        "compared_replay_points": len(compared),
        "exact_matches": len(compared) - len(mismatches),
        "max_abs_difference_cycles": max(row["abs_difference_cycles"] for row in compared),
        "examples": compared,
    }


def main():
    print(json.dumps(run_replay_check(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
