"""Versioned persistence and validation replay for the P09 XGBoost candidate."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from math import isclose
from pathlib import Path
from typing import Any

import joblib
import sklearn

from fleetpulse.evaluation import (
    ValidationPrediction,
    build_metrics_record,
    calculate_validation_metrics,
    save_metrics_record,
    validate_metrics_record,
)
from fleetpulse.features.build_dataset import FeatureDataset
from fleetpulse.models.train_xgboost import MODEL_VERSION, XGBoostValidationResult
from fleetpulse.models.training_validation import validate_training_datasets

XGBOOST_ARTIFACT_SCHEMA = "xgboost-validation-v1"
MODEL_FILENAME = "xgboost_model.joblib"
MANIFEST_FILENAME = "xgboost_validation.json"
METRICS_FILENAME = "xgboost_metrics.json"
RELOAD_DISTRIBUTIONS = ("xgboost", "scikit-learn", "numpy", "scipy", "joblib")


def runtime_software_versions() -> dict[str, str]:
    """Return runtime and installed package versions recorded in candidate manifests."""
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in RELOAD_DISTRIBUTIONS},
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_xgboost_artifact(
    result: XGBoostValidationResult,
    fitting: FeatureDataset,
    validation: FeatureDataset,
    output_dir: str | Path,
    *,
    fitting_dataset_sha256: str,
    validation_dataset_sha256: str,
) -> dict[str, Any]:
    """Save the trusted estimator, lineage, predictions and common metrics record."""
    validate_training_datasets(fitting, validation)
    _validate_result(result, fitting, validation)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    temporary_path = destination / f"{MODEL_FILENAME}.tmp"
    joblib.dump(result.model, temporary_path)
    temporary_path.replace(model_path)
    model_sha256 = sha256_file(model_path)

    rows = [
        {
            "dataset_partition": prediction.dataset_partition,
            "unit_id": prediction.unit_id,
            "cutoff_cycle": prediction.cutoff_cycle,
            "prediction_cycles": prediction.prediction,
            "true_rul_cycles": prediction.true_rul,
            "absolute_error_cycles": metric.absolute_error,
            "squared_error_cycles": metric.squared_error,
        }
        for prediction, metric in zip(
            result.predictions, result.metrics.per_engine, strict=True
        )
    ]
    payload = {
        "schema_version": XGBOOST_ARTIFACT_SCHEMA,
        "model": {
            "name": "xgboost-rul",
            "version": MODEL_VERSION,
            "xgboost_version": version("xgboost"),
            "scikit_learn_version": sklearn.__version__,
            "runtime_versions": runtime_software_versions(),
            "filename": MODEL_FILENAME,
            "sha256": model_sha256,
        },
        "training": {
            "fitting_row_count": len(fitting.rows),
            "fitting_engine_count": len({row.vector.engine_key for row in fitting.rows}),
            "feature_names": list(fitting.feature_names),
            "feature_config": asdict(fitting.config),
            "random_seed": result.random_seed,
            "parameters": dict(result.parameters),
            "runtime_seconds": result.runtime_seconds,
            "source_sha256": fitting.source_sha256,
            "split_manifest_sha256": fitting.split_manifest_sha256,
            "fitting_dataset_sha256": fitting_dataset_sha256,
            "validation_dataset_sha256": validation_dataset_sha256,
        },
        "evaluation": {
            "partition": "FD001/train",
            "sample_unit": "engine",
            "sample_count": result.metrics.engine_count,
            "mae_cycles": result.metrics.mae,
            "rmse_cycles": result.metrics.rmse,
            "predictions": rows,
        },
    }
    manifest_path = destination / MANIFEST_FILENAME
    _write_hashed_json(payload, manifest_path)

    metrics = build_metrics_record(
        model_name="xgboost-rul",
        model_version=MODEL_VERSION,
        partition="FD001/train",
        sample_unit="engine",
        sample_count=result.metrics.engine_count,
        feature_config=asdict(fitting.config),
        random_seed=result.random_seed,
        parameters={
            **result.parameters,
            "fitting_row_count": len(fitting.rows),
            "fitting_engine_count": len({row.vector.engine_key for row in fitting.rows}),
        },
        mae_cycles=result.metrics.mae,
        rmse_cycles=result.metrics.rmse,
        runtime_seconds=result.runtime_seconds,
        input_hashes={
            "training_source_sha256": fitting.source_sha256,
            "split_manifest_sha256": fitting.split_manifest_sha256,
            "fitting_dataset_sha256": fitting_dataset_sha256,
            "validation_dataset_sha256": validation_dataset_sha256,
        },
    )
    metrics_path = destination / METRICS_FILENAME
    save_metrics_record(metrics, metrics_path)
    return {
        "model_path": model_path,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "model_sha256": model_sha256,
        "metrics": metrics,
    }


def verify_xgboost_artifact(
    manifest_path: str | Path,
    validation: FeatureDataset,
) -> dict[str, Any]:
    """Reload the candidate and independently reproduce all validation records."""
    document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("XGBoost manifest content hash mismatch.")
    if payload.get("schema_version") != XGBOOST_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported XGBoost artifact schema.")
    info = payload["model"]
    if (
        info.get("name") != "xgboost-rul"
        or info.get("version") != MODEL_VERSION
        or info.get("filename") != MODEL_FILENAME
    ):
        raise ValueError("Unsupported XGBoost model identity/version/filename.")
    saved_versions = info.get("runtime_versions")
    if not isinstance(saved_versions, dict):
        raise TypeError("Saved candidate has no complete runtime version metadata.")
    actual_versions = runtime_software_versions()
    if any(saved_versions.get(name) != actual_versions[name] for name in RELOAD_DISTRIBUTIONS):
        raise ValueError("Saved package versions required for reload do not match the runtime.")
    if saved_versions.get("python", "").split(".")[:2] != actual_versions["python"].split(".")[:2]:
        raise ValueError("Saved Python major/minor version does not match the project runtime.")
    if info.get("xgboost_version") != saved_versions["xgboost"]:
        raise ValueError("Saved XGBoost version differs from runtime metadata.")
    if info.get("scikit_learn_version") != saved_versions["scikit-learn"]:
        raise ValueError("Saved scikit-learn version differs from runtime metadata.")
    model_path = Path(manifest_path).parent / MODEL_FILENAME
    if sha256_file(model_path) != info.get("sha256"):
        raise ValueError("Saved XGBoost model file hash mismatch.")
    training = payload["training"]
    if (
        validation.role != "validation"
        or validation.source_sha256 != training["source_sha256"]
        or validation.split_manifest_sha256 != training["split_manifest_sha256"]
        or list(validation.feature_names) != training["feature_names"]
        or json.loads(_canonical(asdict(validation.config))) != training["feature_config"]
    ):
        raise ValueError("Validation dataset does not match saved provenance/schema.")

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "Loading this candidate requires `python -m pip install -e '.[xgboost]'`."
        ) from error
    model = joblib.load(model_path)
    if type(model) is not XGBRegressor:
        raise TypeError("Saved candidate is not an XGBRegressor.")
    if model.n_features_in_ != len(training["feature_names"]):
        raise ValueError("Saved model feature count differs from its manifest.")
    expected_parameters = {**training["parameters"], "random_state": training["random_seed"]}
    if any(model.get_params().get(key) != value for key, value in expected_parameters.items()):
        raise ValueError("Saved model parameters differ from their manifest.")

    actual = model.predict(validation.X)
    saved_rows = payload["evaluation"]["predictions"]
    if len(saved_rows) != len(validation.rows):
        raise ValueError("Saved prediction count differs from validation rows.")
    replayed: list[ValidationPrediction] = []
    for row, value, saved in zip(validation.rows, actual, saved_rows, strict=True):
        identity = {
            "dataset_partition": row.vector.dataset_partition,
            "unit_id": row.vector.unit_id,
            "cutoff_cycle": row.vector.selected_cycle,
            "true_rul_cycles": row.true_rul,
        }
        if any(saved.get(key) != expected for key, expected in identity.items()):
            raise ValueError("Saved prediction identity/label differs from validation table.")
        if float(value) != saved["prediction_cycles"]:
            raise ValueError("Reloaded XGBoost candidate differs from saved prediction.")
        replayed.append(
            ValidationPrediction(
                row.vector.dataset_partition,
                row.vector.unit_id,
                row.vector.selected_cycle,
                float(value),
                row.true_rul,
            )
        )
    metrics = calculate_validation_metrics(replayed)
    evaluation = payload["evaluation"]
    for metric, saved in zip(metrics.per_engine, saved_rows, strict=True):
        if (
            saved["absolute_error_cycles"] != metric.absolute_error
            or saved["squared_error_cycles"] != metric.squared_error
        ):
            raise ValueError("Saved per-engine errors differ from reloaded predictions.")
    if not isclose(metrics.mae, evaluation["mae_cycles"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved MAE differs from reloaded per-engine predictions.")
    if not isclose(metrics.rmse, evaluation["rmse_cycles"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved RMSE differs from reloaded per-engine predictions.")

    metrics_record = json.loads(
        (Path(manifest_path).parent / METRICS_FILENAME).read_text(encoding="utf-8")
    )
    validate_metrics_record(metrics_record)
    expected_metrics = build_metrics_record(
        model_name="xgboost-rul",
        model_version=MODEL_VERSION,
        partition="FD001/train",
        sample_unit="engine",
        sample_count=metrics.engine_count,
        feature_config=training["feature_config"],
        random_seed=training["random_seed"],
        parameters={
            **training["parameters"],
            "fitting_row_count": training["fitting_row_count"],
            "fitting_engine_count": training["fitting_engine_count"],
        },
        mae_cycles=metrics.mae,
        rmse_cycles=metrics.rmse,
        runtime_seconds=training["runtime_seconds"],
        input_hashes={
            "training_source_sha256": training["source_sha256"],
            "split_manifest_sha256": training["split_manifest_sha256"],
            "fitting_dataset_sha256": training["fitting_dataset_sha256"],
            "validation_dataset_sha256": training["validation_dataset_sha256"],
        },
    )
    if metrics_record != expected_metrics or evaluation["sample_count"] != metrics.engine_count:
        raise ValueError("Common metrics record differs from the verified candidate.")
    return {
        "prediction_count": len(saved_rows),
        "mae_cycles": metrics.mae,
        "rmse_cycles": metrics.rmse,
    }


def _validate_result(result, fitting, validation) -> None:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Saving this candidate requires the optional XGBoost extra.") from error
    if not isinstance(result, XGBoostValidationResult) or type(result.model) is not XGBRegressor:
        raise TypeError("Requires a fitted XGBoostValidationResult.")
    if result.feature_names != fitting.feature_names:
        raise ValueError("Result feature schema differs from fitting table.")
    expected = [
        (row.vector.dataset_partition, row.vector.unit_id, row.vector.selected_cycle, row.true_rul)
        for row in validation.rows
    ]
    actual = [
        (row.dataset_partition, row.unit_id, row.cutoff_cycle, row.true_rul)
        for row in result.predictions
    ]
    if actual != expected or result.metrics != calculate_validation_metrics(result.predictions):
        raise ValueError("Result predictions/labels/metrics differ from validation table.")
    if tuple(map(float, result.model.predict(validation.X))) != tuple(
        row.prediction for row in result.predictions
    ):
        raise ValueError("Fitted model does not reproduce its recorded predictions.")
    parameters = {**result.parameters, "random_state": result.random_seed}
    if any(result.model.get_params().get(key) != value for key, value in parameters.items()):
        raise ValueError("Fitted estimator parameters differ from result record.")


def _write_hashed_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(
            {"payload": payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
