"""Independently recompute P09 model metrics on identical saved validation rows."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from fleetpulse import PROJECT_ROOT

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "development"
MODEL_FILES = {
    "tree": "decision_tree_validation.json",
    "forest": "random_forest_validation.json",
    "xgboost": "xgboost_validation.json",
}
IDENTITY_FIELDS = ("dataset_partition", "unit_id", "cutoff_cycle", "true_rul_cycles")
SHARED_TRAINING_FIELDS = (
    "source_sha256",
    "split_manifest_sha256",
    "validation_dataset_sha256",
    "feature_names",
    "feature_config",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _load_model_record(label: str, filename: str) -> dict[str, Any]:
    path = ARTIFACT_DIR / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    actual_digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if document.get("sha256") != actual_digest:
        raise ValueError(f"{label} manifest content hash mismatch: {path}")
    evaluation = payload["evaluation"]
    rows = evaluation["predictions"]
    if (
        evaluation.get("partition") != "FD001/train"
        or evaluation.get("sample_unit") != "engine"
        or evaluation.get("sample_count") != 20
        or len(rows) != 20
    ):
        raise ValueError(f"{label} must contain 20 FD001/train engine validation rows.")
    identities = [tuple(row[field] for field in IDENTITY_FIELDS) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} contains duplicate validation identities.")
    for row in rows:
        if row["dataset_partition"] != "FD001/train":
            raise ValueError(f"{label} contains a row outside FD001/train.")
        prediction = float(row["prediction_cycles"])
        target = float(row["true_rul_cycles"])
        if not math.isfinite(prediction) or not math.isfinite(target):
            raise ValueError(f"{label} contains a nonfinite prediction or target.")
    return {
        "payload": payload,
        "model_name": payload["model"]["name"],
        "model_version": payload["model"]["version"],
        "manifest_sha256": actual_digest,
        "rows": rows,
    }


def _recompute(record: dict[str, Any], label: str) -> dict[str, float]:
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    for row in record["rows"]:
        error = float(row["prediction_cycles"]) - float(row["true_rul_cycles"])
        absolute_error = abs(error)
        squared_error = error * error
        if not math.isclose(
            absolute_error, float(row["absolute_error_cycles"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} saved absolute error differs from independent calculation.")
        if not math.isclose(
            squared_error, float(row["squared_error_cycles"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} saved squared error differs from independent calculation.")
        absolute_errors.append(absolute_error)
        squared_errors.append(squared_error)
    count = len(absolute_errors)
    metrics = {
        "mae_cycles": sum(absolute_errors) / count,
        "rmse_cycles": math.sqrt(sum(squared_errors) / count),
    }
    saved = record["payload"]["evaluation"]
    for metric, actual in metrics.items():
        if not math.isclose(actual, float(saved[metric]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} manifest {metric} differs from independent calculation.")
    return metrics


def audit() -> dict[str, Any]:
    records = {
        label: _load_model_record(label, filename) for label, filename in MODEL_FILES.items()
    }
    reference = records["tree"]
    reference_ids = [tuple(row[field] for field in IDENTITY_FIELDS) for row in reference["rows"]]
    reference_training = reference["payload"]["training"]
    for label, record in records.items():
        identities = [tuple(row[field] for field in IDENTITY_FIELDS) for row in record["rows"]]
        if identities != reference_ids:
            raise ValueError(f"{label} validation identities/order/targets differ from tree.")
        training = record["payload"]["training"]
        if any(training.get(field) != reference_training.get(field) for field in SHARED_TRAINING_FIELDS):
            raise ValueError(f"{label} validation source/split/schema differs from tree.")

    metrics = {label: _recompute(record, label) for label, record in records.items()}
    per_engine_comparison = {}
    for comparator in ("tree", "forest"):
        outcomes = {"xgboost_lower_error": 0, "xgboost_equal_error": 0, "xgboost_higher_error": 0}
        for boosted, other in zip(
            records["xgboost"]["rows"], records[comparator]["rows"], strict=True
        ):
            boosted_error = abs(
                float(boosted["prediction_cycles"]) - float(boosted["true_rul_cycles"])
            )
            other_error = abs(
                float(other["prediction_cycles"]) - float(other["true_rul_cycles"])
            )
            outcome = (
                "xgboost_lower_error"
                if boosted_error < other_error
                else "xgboost_higher_error"
                if boosted_error > other_error
                else "xgboost_equal_error"
            )
            outcomes[outcome] += 1
        per_engine_comparison[comparator] = outcomes

    xgboost_rows = records["xgboost"]["rows"]
    example = next(
        row for row in xgboost_rows if row["unit_id"] == 4 and row["cutoff_cycle"] == 164
    )
    example_results = {}
    for label, record in records.items():
        row = next(
            item for item in record["rows"] if item["unit_id"] == 4 and item["cutoff_cycle"] == 164
        )
        example_results[label] = {
            "prediction_cycles": float(row["prediction_cycles"]),
            "target_cycles": float(row["true_rul_cycles"]),
            "absolute_error_cycles": abs(
                float(row["prediction_cycles"]) - float(row["true_rul_cycles"])
            ),
        }
    return {
        "dataset_partition": "FD001/train",
        "sample_unit": "engine",
        "sample_count": len(reference_ids),
        "unique_validation_identities": len(set(reference_ids)),
        "source_sha256": reference_training["source_sha256"],
        "split_manifest_sha256": reference_training["split_manifest_sha256"],
        "validation_dataset_sha256": reference_training["validation_dataset_sha256"],
        "manifest_sha256": {
            label: record["manifest_sha256"] for label, record in records.items()
        },
        "models": {
            label: {
                "name": records[label]["model_name"],
                "version": records[label]["model_version"],
                **metrics[label],
            }
            for label in records
        },
        "xgboost_minus_tree_cycles": {
            metric: metrics["xgboost"][metric] - metrics["tree"][metric]
            for metric in ("mae_cycles", "rmse_cycles")
        },
        "xgboost_minus_forest_cycles": {
            metric: metrics["xgboost"][metric] - metrics["forest"][metric]
            for metric in ("mae_cycles", "rmse_cycles")
        },
        "engine_4_cycle_164": example_results,
        "validation_source_scope": "FD001/train development-validation only; not official test.",
        "model_bundle_sha256": {
            label: records[label]["payload"]["model"]["sha256"] for label in records
        },
        "engine_4_partition": example["dataset_partition"],
        "xgboost_per_engine_absolute_error_counts": per_engine_comparison,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2, allow_nan=False))
