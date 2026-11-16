"""Read-only P09-A1 preflight of common development tables; never train a model."""

import hashlib
import json

from fleetpulse import PROJECT_ROOT
from fleetpulse.data.loader import load_engine_history
from fleetpulse.data.split_audit import audit_split_manifest
from fleetpulse.features import FeatureConfig, engineer_features
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset
from fleetpulse.models.training_validation import validate_training_datasets

TRAIN_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"


def require(condition, message):
    if not condition:
        raise ValueError(f"XGBoost input audit failed: {message}")


def main():
    source = PROJECT_ROOT / "data/raw/train_FD001.txt"
    require(hashlib.sha256(source.read_bytes()).hexdigest() == TRAIN_SHA256, "source SHA mismatch")
    records = load_engine_history(source)
    manifest = json.loads(
        (PROJECT_ROOT / "artifacts/split_manifest.json").read_text(encoding="utf-8")
    )
    audit_split_manifest(manifest, records, source_sha256=TRAIN_SHA256)
    tables = {
        role: load_feature_dataset(
            PROJECT_ROOT / f"artifacts/development/{role}_features.json",
            expected_source_sha256=TRAIN_SHA256,
            expected_manifest_sha256=manifest_digest(manifest),
            expected_config=FeatureConfig(),
        )
        for role in ("fitting", "validation")
    }
    fitting, validation = tables["fitting"], tables["validation"]
    validate_training_datasets(fitting, validation)
    by_unit = {}
    for record in records:
        by_unit.setdefault(record["unit_id"], []).append(record)
    fitting_units = {entry["unit_id"] for entry in manifest["fitting_engines"]}
    expected_keys = sorted(
        ("FD001/train", r["unit_id"], r["cycle"]) for r in records if r["unit_id"] in fitting_units
    )
    require(list(fitting.row_keys) == expected_keys, "fitting engine/cycle membership differs")
    for row in fitting.rows:
        require(
            row.true_rul == by_unit[row.vector.unit_id][-1]["cycle"] - row.vector.selected_cycle,
            "fitting target differs from complete lifetime minus cycle",
        )
    entries = sorted(manifest["validation_prefixes"], key=lambda entry: entry["unit_id"])
    expected_validation = [
        ("FD001/train", entry["unit_id"], entry["cutoff_cycle"]) for entry in entries
    ]
    require(
        list(validation.row_keys) == expected_validation, "validation keys differ from manifest"
    )
    require(
        list(validation.y) == [entry["target_rul"] for entry in entries],
        "validation targets differ",
    )
    fitting_samples = [fitting.rows[index] for index in (0, len(fitting.rows) // 2, -1)]
    for row in (*validation.rows, *fitting_samples):
        prefix = [r for r in by_unit[row.vector.unit_id] if r["cycle"] <= row.vector.selected_cycle]
        expected = engineer_features(
            prefix,
            dataset_partition="FD001/train",
            unit_id=row.vector.unit_id,
            config=FeatureConfig(),
        )
        require(row.vector == expected, "saved features differ from the raw observed prefix")
    print(
        json.dumps(
            {
                "source_sha256": TRAIN_SHA256,
                "split_manifest_sha256": manifest_digest(manifest),
                "fitting_rows": len(fitting.rows),
                "fitting_engines": len(fitting_units),
                "validation_rows": len(validation.rows),
                "feature_count": len(fitting.feature_names),
                "fitting_target_min_cycles": min(fitting.y),
                "fitting_target_max_cycles": max(fitting.y),
                "fitting_labels_checked": len(fitting.rows),
                "validation_vectors_checked": len(validation.rows),
                "fitting_vectors_sampled": len(fitting_samples),
                "model_fit_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
