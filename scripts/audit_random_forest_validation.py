"""Audit saved forest validation rows without deserializing the model binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fleetpulse import PROJECT_ROOT
from fleetpulse.evaluation.metrics_artifact import validate_metrics_record
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import manifest_digest
from fleetpulse.features.dataset_io import load_feature_dataset

MANIFEST_NAME = "random_forest_validation.json"
METRICS_NAME = "random_forest_metrics.json"
MODEL_NAME = "random_forest_model.joblib"


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} mismatch: saved {expected!r}, recomputed {actual!r}.")


def recompute_prediction_audit(
    validation_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check one-to-one validation identity/target coverage and recompute errors."""
    expected: dict[tuple[str, int, int], float] = {}
    for row in validation_rows:
        key_values = row.get("key")
        if (
            not isinstance(key_values, list)
            or len(key_values) != 3
            or not isinstance(key_values[0], str)
            or type(key_values[1]) is not int
            or type(key_values[2]) is not int
        ):
            raise ValueError("Validation row has an invalid partition/engine/cycle key.")
        key = (key_values[0], key_values[1], key_values[2])
        if key in expected:
            raise ValueError(f"Duplicate validation identity {key}.")
        expected[key] = _finite_number(row.get("true_rul"), f"validation target {key}")

    if not expected or len(prediction_rows) != len(expected):
        raise ValueError("Saved prediction count does not match validation row count.")

    seen: set[tuple[str, int, int]] = set()
    audited_rows = []
    absolute_errors = []
    squared_errors = []
    for row in prediction_rows:
        key = (row.get("dataset_partition"), row.get("unit_id"), row.get("cutoff_cycle"))
        if (
            not isinstance(key[0], str)
            or type(key[1]) is not int
            or type(key[2]) is not int
            or key not in expected
        ):
            raise ValueError(f"Saved prediction has an unknown identity {key}.")
        if key in seen:
            raise ValueError(f"Duplicate saved prediction identity {key}.")
        seen.add(key)

        prediction = _finite_number(row.get("prediction_cycles"), f"prediction {key}")
        target = _finite_number(row.get("true_rul_cycles"), f"saved target {key}")
        if target != expected[key]:
            raise ValueError(f"Saved target differs from validation table for {key}.")
        absolute_error = abs(prediction - target)
        squared_error = (prediction - target) ** 2
        _close(
            _finite_number(row.get("absolute_error_cycles"), f"absolute error {key}"),
            absolute_error,
            f"absolute error {key}",
        )
        _close(
            _finite_number(row.get("squared_error_cycles"), f"squared error {key}"),
            squared_error,
            f"squared error {key}",
        )
        absolute_errors.append(absolute_error)
        squared_errors.append(squared_error)
        audited_rows.append(
            {
                "dataset_partition": key[0],
                "unit_id": key[1],
                "cutoff_cycle": key[2],
                "true_rul_cycles": target,
                "prediction_cycles": prediction,
                "absolute_error_cycles": absolute_error,
            }
        )

    if seen != set(expected):
        raise ValueError("Saved prediction identities do not cover the validation table.")
    count = len(audited_rows)
    return {
        "sample_count": count,
        "engine_ids": sorted(row["unit_id"] for row in audited_rows),
        "mae_cycles": math.fsum(absolute_errors) / count,
        "rmse_cycles": math.sqrt(math.fsum(squared_errors) / count),
        "rows": sorted(audited_rows, key=lambda row: row["unit_id"]),
    }


def audit_saved_validation(
    manifest_path: Path,
    metrics_path: Path,
    fitting_path: Path,
    validation_path: Path,
    split_manifest_path: Path,
    training_source_path: Path,
) -> dict[str, Any]:
    """Verify provenance and independently audit metrics using JSON only."""
    document = _read_json(manifest_path)
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("sha256") != _canonical_sha256(payload):
        raise ValueError("Forest validation manifest payload hash mismatch.")
    if payload.get("schema_version") != "random-forest-validation-v1":
        raise ValueError("Unsupported forest validation manifest schema.")

    model = payload.get("model", {})
    if (
        model.get("name") != "random-forest-rul"
        or model.get("version") != "random-forest-rul-v1"
        or model.get("filename") != MODEL_NAME
    ):
        raise ValueError("Unsupported forest model identity or filename.")
    model_path = manifest_path.parent / MODEL_NAME
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != model.get("sha256"):
        raise ValueError("Forest model file SHA-256 differs from the manifest.")

    training = payload.get("training", {})
    config_values = training.get("feature_config")
    if not isinstance(config_values, dict):
        raise TypeError("Forest manifest has no valid feature configuration.")
    config = FeatureConfig(
        channels=tuple(config_values["channels"]),
        window_cycles=config_values["window_cycles"],
    )
    source_hash = hashlib.sha256(training_source_path.read_bytes()).hexdigest()
    split_document = _read_json(split_manifest_path)
    split_hash = manifest_digest(split_document)
    fitting_document = _read_json(fitting_path)
    validation_document = _read_json(validation_path)
    expected_hashes = {
        "source_sha256": source_hash,
        "split_manifest_sha256": split_hash,
        "fitting_dataset_sha256": fitting_document.get("sha256"),
        "validation_dataset_sha256": validation_document.get("sha256"),
    }
    for name, actual in expected_hashes.items():
        if training.get(name) != actual:
            raise ValueError(f"Forest manifest {name} does not match the local input.")

    fitting = load_feature_dataset(
        fitting_path,
        expected_source_sha256=source_hash,
        expected_manifest_sha256=split_hash,
        expected_config=config,
    )
    validation = load_feature_dataset(
        validation_path,
        expected_source_sha256=source_hash,
        expected_manifest_sha256=split_hash,
        expected_config=config,
    )
    if (
        fitting.role != "fitting"
        or validation.role != "validation"
        or list(validation.feature_names) != training.get("feature_names")
    ):
        raise ValueError("Saved feature tables do not match the forest roles/schema.")
    fitting_engine_count = len({row.vector.engine_key for row in fitting.rows})
    if (
        len(fitting.rows) != training.get("fitting_row_count")
        or fitting_engine_count != training.get("fitting_engine_count")
    ):
        raise ValueError("Fitting table counts differ from the forest manifest.")

    validation_rows = [
        {"key": list(row.key), "true_rul": row.true_rul} for row in validation.rows
    ]
    evaluation = payload.get("evaluation", {})
    audit = recompute_prediction_audit(
        validation_rows, evaluation.get("predictions", [])
    )
    if (
        audit["sample_count"] != evaluation.get("sample_count")
        or evaluation.get("partition") != "FD001/train"
        or evaluation.get("sample_unit") != "engine"
    ):
        raise ValueError("Forest evaluation identity/count differs from the audited rows.")
    _close(audit["mae_cycles"], _finite_number(evaluation.get("mae_cycles"), "manifest MAE"), "manifest MAE")
    _close(
        audit["rmse_cycles"],
        _finite_number(evaluation.get("rmse_cycles"), "manifest RMSE"),
        "manifest RMSE",
    )

    metrics = _read_json(metrics_path)
    validate_metrics_record(metrics)
    metrics_evaluation = metrics["evaluation"]
    expected_parameters = {
        **training["parameters"],
        "fitting_row_count": len(fitting.rows),
        "fitting_engine_count": fitting_engine_count,
    }
    expected_input_hashes = {
        "training_source_sha256": source_hash,
        "split_manifest_sha256": split_hash,
        "fitting_dataset_sha256": fitting_document["sha256"],
        "validation_dataset_sha256": validation_document["sha256"],
    }
    if (
        metrics["model"] != {"name": model["name"], "version": model["version"]}
        or metrics_evaluation["partition"] != evaluation["partition"]
        or metrics_evaluation["sample_unit"] != evaluation["sample_unit"]
        or metrics_evaluation["sample_count"] != audit["sample_count"]
        or metrics_evaluation["feature_config"] != config_values
        or metrics_evaluation["random_seed"] != training["random_seed"]
        or metrics_evaluation["parameters"] != expected_parameters
        or metrics_evaluation["runtime_seconds"] != training["runtime_seconds"]
        or metrics_evaluation["input_hashes"] != expected_input_hashes
    ):
        raise ValueError("Common metrics record identity/provenance differs from the manifest.")
    _close(
        audit["mae_cycles"], metrics_evaluation["metrics"]["mae_cycles"], "metrics-v1 MAE"
    )
    _close(
        audit["rmse_cycles"], metrics_evaluation["metrics"]["rmse_cycles"], "metrics-v1 RMSE"
    )

    return {
        "manifest_sha256": document["sha256"],
        "model_sha256_verified_without_deserialization": model_hash,
        "source_sha256": source_hash,
        "split_manifest_sha256": split_hash,
        "fitting_dataset_sha256": fitting_document["sha256"],
        "validation_dataset_sha256": validation_document["sha256"],
        "random_seed": training["random_seed"],
        "parameters": training["parameters"],
        "runtime_seconds_recorded": training["runtime_seconds"],
        **audit,
    }


def main() -> None:
    default_folder = PROJECT_ROOT / "artifacts/development"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_folder / MANIFEST_NAME)
    parser.add_argument("--metrics", type=Path, default=default_folder / METRICS_NAME)
    parser.add_argument("--fitting", type=Path, default=default_folder / "fitting_features.json")
    parser.add_argument("--validation", type=Path, default=default_folder / "validation_features.json")
    parser.add_argument(
        "--split-manifest", type=Path, default=PROJECT_ROOT / "artifacts/split_manifest.json"
    )
    parser.add_argument(
        "--training-source", type=Path, default=PROJECT_ROOT / "data/raw/train_FD001.txt"
    )
    args = parser.parse_args()
    result = audit_saved_validation(
        args.manifest,
        args.metrics,
        args.fitting,
        args.validation,
        args.split_manifest,
        args.training_source,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
