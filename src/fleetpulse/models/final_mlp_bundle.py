"""Standalone, versioned bundle contract for the refitted final MLP."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from fleetpulse.evaluation.selection_manifest import load_selection_manifest
from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_scaler import load_final_scaler_manifest
from fleetpulse.models.mlp import DenseRULRegressor

FINAL_MLP_BUNDLE_SCHEMA = "p12-final-mlp-bundle-v1"
FINAL_MLP_VERSION = "dense-rul-mlp-final-v1"
FINAL_MLP_NAME = "dense-rul-mlp-final"
FINAL_MLP_WEIGHTS_FILENAME = "final_mlp_state.pt"
FINAL_MLP_MANIFEST_FILENAME = "final_mlp_manifest.json"
_SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class FinalMLPBundle:
    """Loaded CPU model, exact input scaler, and verified bundle provenance."""

    model: DenseRULRegressor
    scaler: StandardScaler
    payload: dict[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validated_scaler(saved: dict[str, Any], feature_count: int) -> StandardScaler:
    if (
        saved.get("type") != "StandardScaler"
        or saved.get("with_mean") is not True
        or saved.get("with_std") is not True
        or type(saved.get("feature_count")) is not int
        or saved["feature_count"] != feature_count
    ):
        raise ValueError("Final MLP scaler parameters do not match its feature schema.")
    if type(saved.get("fitting_row_count")) is not int or saved["fitting_row_count"] < 1:
        raise ValueError("Final MLP scaler fitting-row count is invalid.")
    arrays = [np.asarray(saved[name], dtype=np.float64) for name in ("mean", "scale", "variance")]
    if any(values.shape != (feature_count,) or not np.isfinite(values).all() for values in arrays):
        raise ValueError("Final MLP scaler statistics are malformed or non-finite.")
    if np.any(arrays[1] <= 0) or np.any(arrays[2] < 0):
        raise ValueError("Final MLP scaler scale or variance is invalid.")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = arrays
    scaler.n_features_in_ = feature_count
    scaler.n_samples_seen_ = saved["fitting_row_count"]
    return scaler


def load_final_mlp_bundle(manifest_path: str | Path) -> FinalMLPBundle:
    """Load final weights and scaler only after hashes/schema are verified."""
    path = Path(manifest_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("sha256") != _sha256_payload(payload):
        raise ValueError("Final MLP bundle manifest hash mismatch.")
    if payload.get("schema_version") != FINAL_MLP_BUNDLE_SCHEMA or payload.get(
        "artifact_role"
    ) != "final-training-model-bundle":
        raise ValueError("Unsupported final MLP bundle schema or artifact role.")

    model_info = payload.get("model", {})
    architecture = model_info.get("architecture", {})
    training = payload.get("training", {})
    parameters = training.get("architecture_and_training", {})
    feature_names = training.get("feature_names")
    if (
        model_info.get("name") != FINAL_MLP_NAME
        or model_info.get("version") != FINAL_MLP_VERSION
        or model_info.get("device") != "cpu"
        or model_info.get("weights_file") != FINAL_MLP_WEIGHTS_FILENAME
        or architecture.get("class") != "DenseRULRegressor"
        or architecture.get("hidden_activation") != "ReLU"
        or architecture.get("output_dim") != 1
        or architecture.get("hidden_dims") != [128, 64]
        or architecture.get("input_dim") != len(feature_names or [])
        or len(feature_names or []) != len(set(feature_names or []))
        or type(training.get("validation_used")) is not bool
        or training["validation_used"] is not False
        or type(training.get("early_stopping")) is not bool
        or training["early_stopping"] is not False
        or training.get("epochs_trained") != 8
        or training.get("random_seed") != 42
        or parameters.get("hidden_dims") != [128, 64]
        or parameters.get("batch_size") != 128
        or parameters.get("learning_rate") != 0.001
        or parameters.get("optimizer") != "Adam"
        or parameters.get("loss") != "mean_squared_error"
        or parameters.get("device") != "cpu"
    ):
        raise ValueError("Final MLP architecture or frozen training rule is invalid.")

    config_document = training.get("feature_config", {})
    config = FeatureConfig(
        channels=tuple(config_document["channels"]),
        window_cycles=config_document["window_cycles"],
    )
    if (
        config_document.get("version") != config.version
        or feature_names != list(config.output_schema)
        or architecture["input_dim"] != len(config.output_schema)
    ):
        raise ValueError("Final MLP feature configuration/order is invalid.")
    for field in ("training_source_sha256", "selection_manifest_sha256", "scaler_payload_sha256"):
        if not isinstance(training.get(field), str) or not _SHA256_PATTERN.fullmatch(training[field]):
            raise ValueError(f"Final MLP {field} is invalid.")
    if training.get("dataset_partition") != "train":
        raise ValueError("Final MLP training partition must be the original train histories.")
    row_count = training.get("training_row_count")
    engine_count = training.get("training_engine_count")
    per_engine = training.get("per_engine_row_counts")
    if (
        type(row_count) is not int or row_count < 1
        or type(engine_count) is not int or engine_count < 1
        or not isinstance(per_engine, list) or len(per_engine) != engine_count
    ):
        raise ValueError("Final MLP training row/engine counts are invalid.")
    engine_ids = [item.get("unit_id") for item in per_engine]
    if (
        len(set(engine_ids)) != engine_count
        or any(type(item.get("unit_id")) is not int or item["unit_id"] < 1 for item in per_engine)
        or any(type(item.get("row_count")) is not int or item["row_count"] < 1 for item in per_engine)
        or sum(item["row_count"] for item in per_engine) != row_count
    ):
        raise ValueError("Final MLP per-engine training row counts are invalid.")

    scaler_info = payload.get("scaler", {})
    scaler = _validated_scaler(scaler_info, len(feature_names))
    if scaler_info.get("selection_manifest_sha256") != training["selection_manifest_sha256"]:
        raise ValueError("Final MLP scaler selection lineage differs from the model.")
    if scaler_info.get("training_source_sha256") != training["training_source_sha256"]:
        raise ValueError("Final MLP scaler source lineage differs from the model.")
    if scaler_info.get("payload_sha256") != training["scaler_payload_sha256"]:
        raise ValueError("Final MLP scaler payload hash differs from model provenance.")
    if scaler.n_samples_seen_ != row_count:
        raise ValueError("Final MLP scaler and model training row counts differ.")

    weights_path = path.parent / model_info["weights_file"]
    expected_weights_hash = model_info.get("weights_sha256")
    if not isinstance(expected_weights_hash, str) or not _SHA256_PATTERN.fullmatch(
        expected_weights_hash
    ) or _sha256_file(weights_path) != expected_weights_hash:
        raise ValueError("Final MLP weights hash mismatch.")
    model = DenseRULRegressor(architecture["input_dim"], tuple(architecture["hidden_dims"]))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("Final MLP weights contain non-finite parameters.")
    model.to("cpu")
    model.eval()
    return FinalMLPBundle(model, scaler, payload)


def package_final_mlp_bundle(
    refit_weights_path: str | Path,
    refit_manifest_path: str | Path,
    scaler_manifest_path: str | Path,
    selection_manifest_path: str | Path,
    output_dir: str | Path,
) -> FinalMLPBundle:
    """Package verified P12-S1/S2 outputs into a separately versioned bundle."""
    weights_path = Path(refit_weights_path)
    refit_document = json.loads(Path(refit_manifest_path).read_text(encoding="utf-8"))
    refit = refit_document.get("payload")
    if not isinstance(refit, dict) or refit_document.get("sha256") != _sha256_payload(refit):
        raise ValueError("P12-S2 refit manifest hash mismatch.")
    if refit.get("checkpoint_sha256") != _sha256_file(weights_path):
        raise ValueError("P12-S2 checkpoint hash differs from its manifest.")

    scaler_doc = json.loads(Path(scaler_manifest_path).read_text(encoding="utf-8"))
    scaler, scaler_payload = load_final_scaler_manifest(
        scaler_manifest_path, expected_source_sha256=refit["training_source_sha256"]
    )
    if scaler_doc.get("sha256") != _sha256_payload(scaler_payload):
        raise ValueError("P12-S1 scaler manifest hash mismatch.")
    selection_doc = json.loads(Path(selection_manifest_path).read_text(encoding="utf-8"))
    selection = load_selection_manifest(selection_manifest_path)
    if selection["training_source_sha256"] != refit["training_source_sha256"]:
        raise ValueError("P12-S2 refit source lineage differs from the P11 selection.")
    if refit["selection_manifest_sha256"] != selection_doc.get("sha256"):
        raise ValueError("P12-S2 refit selection lineage differs from P11.")
    if scaler_payload["selection_manifest_sha256"] != selection_doc.get("sha256"):
        raise ValueError("P12-S1 scaler selection lineage differs from P11.")
    config_doc = refit["feature_config"]
    config = FeatureConfig(tuple(config_doc["channels"]), config_doc["window_cycles"])
    if config_doc != {
        "version": config.version,
        "channels": list(config.channels),
        "window_cycles": config.window_cycles,
    }:
        raise ValueError("P12-S2 feature configuration is malformed.")
    if scaler_payload["feature_config"] != {
        "version": config.version,
        "channels": list(config.channels),
        "window_cycles": config.window_cycles,
    }:
        raise ValueError("P12-S1 scaler and P12-S2 checkpoint feature configurations differ.")
    selection_rule = selection["decision"]["mlp_refit_rule"]
    if (
        selection_rule.get("hidden_dims") != [128, 64]
        or selection_rule.get("frozen_epoch_count") != 8
        or selection_rule.get("seed") != refit["random_seed"]
        or selection_rule.get("feature_config") != config_doc
    ):
        raise ValueError("P12-S2 checkpoint differs from the frozen P11 MLP selection rule.")
    if refit["scaler_payload_sha256"] != _sha256_payload(scaler_payload):
        raise ValueError("P12-S2 refit used a different scaler payload.")
    parameters = refit["architecture_and_training"]
    if (
        parameters.get("hidden_dims") != [128, 64]
        or parameters.get("epochs_trained") != 8
        or parameters.get("batch_size") != 128
        or parameters.get("learning_rate") != 0.001
        or parameters.get("optimizer") != "Adam"
        or parameters.get("loss") != "mean_squared_error"
        or parameters.get("device") != "cpu"
        or parameters.get("validation_used") is not False
        or parameters.get("early_stopping") is not False
        or refit["random_seed"] != 42
        or refit.get("official_test_data_opened") is not False
        or refit.get("official_test_labels_used") is not False
    ):
        raise ValueError("P12-S2 refit does not satisfy the frozen final-MPL rule.")

    model = DenseRULRegressor(len(config.output_schema), (128, 64)).to("cpu")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("P12-S2 checkpoint contains non-finite parameters.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_weights = destination / FINAL_MLP_WEIGHTS_FILENAME
    temporary_weights = bundle_weights.with_suffix(bundle_weights.suffix + ".tmp")
    shutil.copyfile(weights_path, temporary_weights)
    temporary_weights.replace(bundle_weights)
    scaler_info = {
        **scaler_payload["scaler"],
        "training_source_sha256": scaler_payload["training_source_sha256"],
        "selection_manifest_sha256": scaler_payload["selection_manifest_sha256"],
        "payload_sha256": _sha256_payload(scaler_payload),
    }
    payload = {
        "schema_version": FINAL_MLP_BUNDLE_SCHEMA,
        "artifact_role": "final-training-model-bundle",
        "model": {
            "name": FINAL_MLP_NAME,
            "version": FINAL_MLP_VERSION,
            "device": "cpu",
            "weights_file": FINAL_MLP_WEIGHTS_FILENAME,
            "weights_sha256": _sha256_file(bundle_weights),
            "architecture": {
                "class": "DenseRULRegressor",
                "input_dim": len(config.output_schema),
                "hidden_dims": [128, 64],
                "hidden_activation": "ReLU",
                "output_dim": 1,
            },
        },
        "training": {
            "dataset_partition": "train",
            "training_source_sha256": refit["training_source_sha256"],
            "selection_manifest_sha256": refit["selection_manifest_sha256"],
            "feature_config": {
                "version": config.version,
                "channels": list(config.channels),
                "window_cycles": config.window_cycles,
            },
            "feature_names": list(config.output_schema),
            "training_engine_count": refit["training_engine_count"],
            "training_row_count": refit["training_row_count"],
            "per_engine_row_counts": refit["per_engine_row_counts"],
            "architecture_and_training": parameters,
            "random_seed": refit["random_seed"],
            "epochs_trained": parameters["epochs_trained"],
            "epoch_training_mse_cycles_squared": refit["epoch_training_mse_cycles_squared"],
            "runtime_seconds": refit["runtime_seconds"],
            "python_version": refit["python_version"],
            "torch_version": refit["torch_version"],
            "scikit_learn_version": refit["scikit_learn_version"],
            "validation_used": False,
            "early_stopping": False,
            "official_test_data_opened": False,
            "official_test_labels_used": False,
            "refit_manifest_sha256": refit_document["sha256"],
            "scaler_payload_sha256": scaler_info["payload_sha256"],
        },
        "scaler": scaler_info,
    }
    manifest_path = destination / FINAL_MLP_MANIFEST_FILENAME
    document = {"payload": payload, "sha256": _sha256_payload(payload)}
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_manifest.replace(manifest_path)
    bundle = load_final_mlp_bundle(manifest_path)
    if not (
        np.array_equal(bundle.scaler.mean_, scaler.mean_)
        and np.array_equal(bundle.scaler.scale_, scaler.scale_)
        and np.array_equal(bundle.scaler.var_, scaler.var_)
    ):
        raise ValueError("Packaged final scaler differs from the verified P12-S1 scaler.")
    return bundle


def predict_final_mlp(bundle: FinalMLPBundle, features: np.ndarray) -> tuple[float, ...]:
    """Apply the saved scaler and final CPU model to raw ordered feature rows."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != bundle.model.input_dim:
        raise ValueError("features must be a 2D matrix in the saved feature order.")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values.")
    transformed = torch.tensor(bundle.scaler.transform(matrix), dtype=torch.float32, device="cpu")
    bundle.model.eval()
    with torch.no_grad():
        prediction = bundle.model(transformed)
    if not torch.isfinite(prediction).all().item():
        raise FloatingPointError("Final MLP produced a non-finite prediction.")
    return tuple(float(value) for value in prediction.tolist())
