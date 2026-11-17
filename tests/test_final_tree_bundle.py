"""Integrity and inference checks for standalone P12 tree bundles."""

import hashlib
import json

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from fleetpulse.features.config import FeatureConfig
from fleetpulse.models.constant_baseline import ConstantRULBaseline
from fleetpulse.models.final_tree_bundle import (
    FINAL_TREE_BUNDLE_SCHEMA,
    load_final_tree_bundle,
    predict_final_tree_bundle,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _training(config):
    return {
        "dataset_partition": "train",
        "training_source_sha256": "1" * 64,
        "selection_manifest_sha256": "2" * 64,
        "refit_manifest_sha256": "3" * 64,
        "feature_config": {
            "version": config.version,
            "channels": list(config.channels),
            "window_cycles": config.window_cycles,
        },
        "feature_names": list(config.output_schema),
        "training_engine_count": 100,
        "training_row_count": 20631,
        "per_engine_row_counts": [
            {"unit_id": engine, "row_count": 207 if engine <= 31 else 206}
            for engine in range(1, 101)
        ],
        "parameters": {"max_depth": 8, "min_samples_leaf": 5},
        "random_seed": 42,
    }


def _write_manifest(path, payload):
    path.write_text(
        json.dumps({"payload": payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}),
        encoding="utf-8",
    )


def _tree_bundle(tmp_path):
    config = FeatureConfig()
    model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42)
    model.fit(np.zeros((2, 96)), np.array([2.0, 3.0]))
    weights = tmp_path / "decision_tree_final_v1.joblib"
    import joblib

    joblib.dump(model, weights)
    payload = {
        "schema_version": FINAL_TREE_BUNDLE_SCHEMA,
        "artifact_role": "final-training-model-bundle",
        "name": "decision-tree-rul",
        "model": {
            "version": "decision-tree-rul-v1",
            "weights_file": weights.name,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
        "training": _training(config),
    }
    manifest = tmp_path / "decision-tree-rul_manifest.json"
    _write_manifest(manifest, payload)
    return manifest, weights


def test_final_tree_bundle_checks_hash_schema_and_predicts(tmp_path):
    manifest, _ = _tree_bundle(tmp_path)

    bundle = load_final_tree_bundle(manifest)
    predictions = predict_final_tree_bundle(bundle, np.zeros((3, 96)))

    assert bundle.version == "decision-tree-rul-v1"
    assert len(bundle.feature_names) == 96
    assert predictions == (2.5, 2.5, 2.5)


def test_final_tree_bundle_rejects_weight_tampering(tmp_path):
    manifest, weights = _tree_bundle(tmp_path)
    weights.write_bytes(weights.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="weights hash mismatch"):
        load_final_tree_bundle(manifest)


def test_constant_bundle_predicts_selected_mean(tmp_path):
    config = FeatureConfig()
    baseline = ConstantRULBaseline(statistic="mean", value_cycles=107.8, fitting_label_count=20631)
    payload = {
        "schema_version": FINAL_TREE_BUNDLE_SCHEMA,
        "artifact_role": "final-training-model-bundle",
        "name": "constant-rul",
        "model": {
            "version": "constant-rul-v1",
            "weights_file": None,
            "weights_sha256": None,
            "baseline": baseline.to_artifact(),
        },
        "training": _training(config),
    }
    manifest = tmp_path / "constant-rul_manifest.json"
    _write_manifest(manifest, payload)

    bundle = load_final_tree_bundle(manifest)

    assert predict_final_tree_bundle(bundle, np.zeros((2, 96))) == (107.8, 107.8)
