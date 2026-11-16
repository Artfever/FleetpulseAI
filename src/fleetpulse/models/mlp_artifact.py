"""Versioned, CPU-reloadable storage for a validation-selected MLP checkpoint."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

from fleetpulse.features.build_dataset import FeatureDataset
from fleetpulse.models.mlp import DenseRULRegressor, MLPTrainingResult, evaluate_mlp
from fleetpulse.models.training_validation import validate_training_datasets

MLP_ARTIFACT_SCHEMA = "mlp-validation-checkpoint-v1"
MODEL_VERSION = "dense-rul-mlp-v1"
WEIGHTS_FILENAME = "mlp_best_state.pt"
MANIFEST_FILENAME = "mlp_training_manifest.json"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_manifest(payload: dict[str, Any], path: Path) -> None:
    document = {"payload": payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _scaler_payload(scaler: StandardScaler) -> dict[str, Any]:
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("The training scaler must be fitted before saving.")
    return {
        "type": "StandardScaler",
        "with_mean": bool(scaler.with_mean),
        "with_std": bool(scaler.with_std),
        "feature_count": int(scaler.n_features_in_),
        "fitting_row_count": int(scaler.n_samples_seen_),
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "variance": [float(value) for value in scaler.var_],
    }


def _restore_scaler(saved: dict[str, Any]) -> StandardScaler:
    if saved.get("type") != "StandardScaler" or saved.get("with_mean") is not True or saved.get(
        "with_std"
    ) is not True:
        raise ValueError("Unsupported saved scaler configuration.")
    count = saved["feature_count"]
    if type(count) is not int or count <= 0:
        raise ValueError("Saved scaler feature count must be positive.")
    arrays = [np.asarray(saved[name], dtype=np.float64) for name in ("mean", "scale", "variance")]
    if any(values.shape != (count,) or not np.isfinite(values).all() for values in arrays):
        raise ValueError("Saved scaler statistics are invalid.")
    if np.any(arrays[1] <= 0) or np.any(arrays[2] < 0):
        raise ValueError("Saved scaler scale/variance is invalid.")
    rows = saved["fitting_row_count"]
    if type(rows) is not int or rows <= 0:
        raise ValueError("Saved scaler fitting row count must be positive.")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = arrays
    scaler.n_features_in_ = count
    scaler.n_samples_seen_ = rows
    return scaler


def save_mlp_artifact(
    result: MLPTrainingResult,
    fitting: FeatureDataset,
    validation: FeatureDataset,
    output_dir: str | Path,
    *,
    fitting_dataset_sha256: str,
    validation_dataset_sha256: str,
) -> dict[str, Any]:
    """Save CPU weights plus architecture, scaler, learning curves and predictions."""
    validate_training_datasets(fitting, validation)
    if not isinstance(result, MLPTrainingResult) or not isinstance(result.model, DenseRULRegressor):
        raise TypeError("result must be an MLPTrainingResult.")
    if result.feature_names != fitting.feature_names:
        raise ValueError("Training result feature order differs from the fitting data.")
    if result.model.input_dim != len(result.feature_names):
        raise ValueError("Model input width differs from the recorded feature schema.")
    if result.scaler.n_features_in_ != len(result.feature_names):
        raise ValueError("Scaler feature width differs from the recorded feature schema.")
    if len(result.training_losses) != len(result.validation_losses):
        raise ValueError("Training and validation curves must have equal lengths.")
    if not 1 <= result.best_epoch <= len(result.validation_losses):
        raise ValueError("Best epoch is outside the recorded validation curve.")
    if len(result.validation_predictions) != len(validation.rows):
        raise ValueError("Saved validation prediction count differs from the validation data.")
    numbers = (*result.training_losses, *result.validation_losses, *result.validation_predictions,
               result.runtime_seconds)
    if not all(isfinite(value) for value in numbers):
        raise ValueError("MLP losses, predictions, and runtime must all be finite.")
    replay = evaluate_mlp(result.model, validation, scaler=result.scaler)
    if replay.predictions != result.validation_predictions:
        raise ValueError("Training result model does not reproduce its selected predictions.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / WEIGHTS_FILENAME
    temporary_weights = destination / f"{WEIGHTS_FILENAME}.tmp"
    cpu_state = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
    torch.save(cpu_state, temporary_weights)
    temporary_weights.replace(weights_path)

    predictions = [
        {
            "dataset_partition": row.vector.dataset_partition,
            "unit_id": row.vector.unit_id,
            "cutoff_cycle": row.vector.selected_cycle,
            "true_rul_cycles": row.true_rul,
            "prediction_cycles": float(prediction),
        }
        for row, prediction in zip(validation.rows, result.validation_predictions, strict=True)
    ]
    payload = {
        "schema_version": MLP_ARTIFACT_SCHEMA,
        "model": {
            "name": "dense-rul-mlp",
            "version": MODEL_VERSION,
            "torch_version": torch.__version__,
            "filename": WEIGHTS_FILENAME,
            "sha256": sha256_file(weights_path),
            "device": "cpu",
            "architecture": {
                "class": "DenseRULRegressor",
                "input_dim": result.model.input_dim,
                "hidden_dims": list(result.model.hidden_dims),
                "hidden_activation": "ReLU",
                "output_dim": 1,
            },
        },
        "training": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "feature_names": list(result.feature_names),
            "feature_config": asdict(fitting.config),
            "fitting_row_count": len(fitting.rows),
            "validation_row_count": len(validation.rows),
            "random_seed": result.random_seed,
            "parameters": result.parameters,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "runtime_seconds": result.runtime_seconds,
            "scaler": _scaler_payload(result.scaler),
            "training_mse_cycles_squared": list(result.training_losses),
            "validation_mse_cycles_squared": list(result.validation_losses),
            "source_sha256": fitting.source_sha256,
            "split_manifest_sha256": fitting.split_manifest_sha256,
            "fitting_dataset_sha256": fitting_dataset_sha256,
            "validation_dataset_sha256": validation_dataset_sha256,
        },
        "evaluation": {
            "partition": "FD001/train",
            "sample_unit": "engine-prefix",
            "sample_count": len(predictions),
            "predictions": predictions,
        },
    }
    manifest_path = destination / MANIFEST_FILENAME
    _write_manifest(payload, manifest_path)
    return {"weights_path": weights_path, "manifest_path": manifest_path, "payload": payload}


def verify_mlp_artifact(
    manifest_path: str | Path,
    fitting: FeatureDataset,
    validation: FeatureDataset,
    *,
    fitting_dataset_sha256: str,
    validation_dataset_sha256: str,
) -> dict[str, Any]:
    """Reload the bundle on CPU and check provenance, curves and exact predictions."""
    document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if document.get("sha256") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("MLP manifest content hash mismatch.")
    if payload.get("schema_version") != MLP_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported MLP artifact schema.")
    training = payload["training"]
    model_info = payload["model"]
    if model_info.get("version") != MODEL_VERSION or model_info.get("device") != "cpu":
        raise ValueError("Unsupported MLP model version or device.")
    if (
        training["feature_names"] != list(fitting.feature_names)
        or training["feature_names"] != list(validation.feature_names)
        or training["feature_config"] != json.loads(_canonical(asdict(fitting.config)))
        or training["source_sha256"] != fitting.source_sha256
        or training["source_sha256"] != validation.source_sha256
        or training["split_manifest_sha256"] != fitting.split_manifest_sha256
        or training["split_manifest_sha256"] != validation.split_manifest_sha256
        or training["fitting_dataset_sha256"] != fitting_dataset_sha256
        or training["validation_dataset_sha256"] != validation_dataset_sha256
    ):
        raise ValueError("Saved MLP training provenance differs from the supplied datasets.")
    weights_path = Path(manifest_path).parent / model_info["filename"]
    if sha256_file(weights_path) != model_info["sha256"]:
        raise ValueError("MLP weights hash mismatch.")
    architecture = model_info["architecture"]
    if architecture.get("class") != "DenseRULRegressor" or architecture.get(
        "hidden_activation"
    ) != "ReLU":
        raise ValueError("Unsupported MLP architecture.")
    model = DenseRULRegressor(
        architecture["input_dim"], tuple(architecture["hidden_dims"])
    ).to("cpu")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.training or any(value.device.type != "cpu" for value in model.parameters()):
        raise RuntimeError("Reloaded MLP must be in evaluation mode on CPU.")
    scaler = _restore_scaler(training["scaler"])
    replay = evaluate_mlp(model, validation, scaler=scaler)
    saved_predictions = payload["evaluation"]["predictions"]
    expected_rows = [
        (row.vector.dataset_partition, row.vector.unit_id, row.vector.selected_cycle, row.true_rul)
        for row in validation.rows
    ]
    actual_rows = [
        (row["dataset_partition"], row["unit_id"], row["cutoff_cycle"], row["true_rul_cycles"])
        for row in saved_predictions
    ]
    if expected_rows != actual_rows or len(replay.predictions) != len(saved_predictions):
        raise ValueError("Saved MLP prediction identities differ from validation data.")
    if replay.predictions != tuple(float(row["prediction_cycles"]) for row in saved_predictions):
        raise ValueError("Reloaded MLP prediction parity failed.")
    return {
        "prediction_count": len(replay.predictions),
        "best_epoch": training["best_epoch"],
        "eval_mode": not model.training,
        "device": "cpu",
        "max_prediction_difference_cycles": max(
            abs(actual - row["prediction_cycles"])
            for actual, row in zip(replay.predictions, saved_predictions, strict=True)
        ),
    }
