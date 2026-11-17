"""Persistence and integrity tests for the standalone final MLP bundle."""

import hashlib
import json

import numpy as np
import pytest
import torch

from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.final_mlp_bundle import load_final_mlp_bundle, predict_final_mlp
from fleetpulse.models.mlp import DenseRULRegressor


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _bundle(tmp_path):
    config = FeatureConfig()
    model = DenseRULRegressor(96, (128, 64))
    weights_path = tmp_path / "final_mlp_state.pt"
    torch.save(model.state_dict(), weights_path)
    source_sha = "1" * 64
    selection_sha = "2" * 64
    scaler_payload_sha = "3" * 64
    scaler_info = {
        "type": "StandardScaler",
        "with_mean": True,
        "with_std": True,
        "feature_count": 96,
        "fitting_row_count": 4,
        "mean": [0.0] * 96,
        "scale": [1.0] * 96,
        "variance": [0.0] * 96,
        "training_source_sha256": source_sha,
        "selection_manifest_sha256": selection_sha,
        "payload_sha256": scaler_payload_sha,
    }
    payload = {
        "schema_version": "p12-final-mlp-bundle-v1",
        "artifact_role": "final-training-model-bundle",
        "model": {
            "name": "dense-rul-mlp-final",
            "version": "dense-rul-mlp-final-v1",
            "device": "cpu",
            "weights_file": weights_path.name,
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "architecture": {
                "class": "DenseRULRegressor",
                "input_dim": 96,
                "hidden_dims": [128, 64],
                "hidden_activation": "ReLU",
                "output_dim": 1,
            },
        },
        "training": {
            "dataset_partition": "train",
            "training_source_sha256": source_sha,
            "selection_manifest_sha256": selection_sha,
            "feature_config": {
                "version": config.version,
                "channels": list(config.channels),
                "window_cycles": config.window_cycles,
            },
            "feature_names": list(config.output_schema),
            "training_engine_count": 1,
            "training_row_count": 4,
            "per_engine_row_counts": [{"unit_id": 1, "row_count": 4}],
            "architecture_and_training": {
                "hidden_dims": [128, 64],
                "batch_size": 128,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "loss": "mean_squared_error",
                "device": "cpu",
            },
            "random_seed": 42,
            "epochs_trained": 8,
            "epoch_training_mse_cycles_squared": [1.0] * 8,
            "runtime_seconds": 1.0,
            "python_version": "3.12",
            "torch_version": torch.__version__,
            "scikit_learn_version": "1.9",
            "validation_used": False,
            "early_stopping": False,
            "official_test_data_opened": False,
            "official_test_labels_used": False,
            "refit_manifest_sha256": "4" * 64,
            "scaler_payload_sha256": scaler_payload_sha,
        },
        "scaler": scaler_info,
    }
    manifest_path = tmp_path / "final_mlp_manifest.json"
    manifest_path.write_text(
        json.dumps({"payload": payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}),
        encoding="utf-8",
    )
    return manifest_path, weights_path


def test_final_mlp_bundle_reloads_on_cpu_and_predicts(tmp_path):
    manifest_path, _ = _bundle(tmp_path)

    bundle = load_final_mlp_bundle(manifest_path)
    prediction = predict_final_mlp(bundle, np.zeros((1, 96), dtype=np.float64))

    assert bundle.model.training is False
    assert all(parameter.device.type == "cpu" for parameter in bundle.model.parameters())
    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_final_mlp_bundle_rejects_weight_hash_mismatch(tmp_path):
    manifest_path, weights_path = _bundle(tmp_path)
    weights_path.write_bytes(weights_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="weights hash mismatch"):
        load_final_mlp_bundle(manifest_path)
