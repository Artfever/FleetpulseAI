"""Synthetic P17 comparison boundaries; fixture metadata is schema-only, not NASA evidence."""

import hashlib
import json
from dataclasses import asdict

import joblib
import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.features import FeatureConfig, engineer_features
from fleetpulse.models.registry import FINAL_MODEL_REGISTRY
from fleetpulse.services import PredictionInputError, compare_final_models


def row(cycle, unit=99):
    return {
        "unit_id": unit,
        "cycle": cycle,
        **dict.fromkeys(READING_COLUMNS, 0.5),
        "sensor_1": 2 * cycle,
    }


def rewrite(path, payload):
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path.write_text(
        json.dumps({"payload": payload, "sha256": hashlib.sha256(content).hexdigest()}),
        encoding="utf-8",
    )


@pytest.fixture
def synthetic_tree(tmp_path):
    config = FeatureConfig()
    folder = tmp_path / "artifacts/final/tree/bundles"
    folder.mkdir(parents=True)
    model = DecisionTreeRegressor(random_state=42).fit(
        np.array([[0.0] * 96, [10.0] * 96]), [-2.5, 17.5]
    )
    weights = folder / "decision_tree_final_v1.joblib"
    joblib.dump(model, weights)
    payload = {
        "schema_version": "p12-final-tree-bundle-v1",
        "artifact_role": "final-training-model-bundle",
        "name": "decision-tree-rul",
        "model": {
            "version": "decision-tree-rul-v1",
            "weights_file": weights.name,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
        "training": {
            "dataset_partition": "train",
            "training_source_sha256": "1" * 64,
            "selection_manifest_sha256": "2" * 64,
            "refit_manifest_sha256": "3" * 64,
            "feature_config": json.loads(json.dumps(asdict(config))),
            "feature_names": list(config.output_schema),
            # Required loader shape, wholly invented for this isolated fixture.
            "training_engine_count": 100,
            "training_row_count": 20631,
            "per_engine_row_counts": [
                {"unit_id": unit, "row_count": 207 if unit <= 31 else 206} for unit in range(1, 101)
            ],
            "parameters": {},
            "random_seed": 42,
        },
    }
    path = folder / "decision-tree-rul_manifest.json"
    rewrite(path, payload)
    return tmp_path, path, weights, payload


def compare(history, root, **kwargs):
    return compare_final_models(
        history, dataset_partition="synthetic/P17", unit_id=99, artifact_root=root, **kwargs
    )


def test_real_tree_loader_preserves_raw_output_and_exact_input_features(
    synthetic_tree, monkeypatch
):
    root, _, _, _ = synthetic_tree
    history = [row(c) for c in (1, 2, 3)]
    original = [dict(r) for r in history]
    monkeypatch.setattr(
        DecisionTreeRegressor, "fit", lambda *a, **kw: pytest.fail("Inference must not fit")
    )
    result = compare(iter(history), root)
    assert history == original
    assert result.observed_cycle == 3
    assert result.dataset_partition == "synthetic/P17" and result.unit_id == 99
    assert len(result.predictions) == 1 and len(result.failures) == 4
    value = result.predictions[0]
    assert value.raw_rul_cycles == -2.5  # No display clipping.
    assert value.stage == "final" and value.model_version == "decision-tree-rul-v1"
    assert value.vector == engineer_features(history, dataset_partition="synthetic/P17", unit_id=99)
    assert len(value.vector.values) == 96
    assert (
        len(value.manifest_payload_sha256)
        == len(value.weights_sha256)
        == len(value.feature_config_sha256)
        == 64
    )
    assert all(f.code == "artifact_unavailable" for f in result.failures)
    changed = [{**r, **dict.fromkeys(READING_COLUMNS, 10.0)} for r in history]
    assert compare(changed, root).input_history_sha256 != result.input_history_sha256
    assert compare(changed, root).predictions[0].raw_rul_cycles == 17.5


@pytest.mark.parametrize(
    "case", ["empty", "mixed", "target", "nonfinite", "wrong_unit", "unsorted"]
)
def test_invalid_input_fails_before_artifact_access(tmp_path, monkeypatch, case):
    import fleetpulse.services.final_comparison as service

    history = [row(1), row(2)]
    if case == "empty":
        history = []
    elif case == "mixed":
        history[1]["unit_id"] = 100
    elif case == "target":
        history[0]["true_rul"] = 5
    elif case == "nonfinite":
        history[0]["sensor_1"] = float("nan")
    elif case == "wrong_unit":
        history = [row(1, 100)]
    else:
        history.reverse()
    monkeypatch.setattr(
        service, "_predict_bundle", lambda *a: pytest.fail("Invalid input reached a model")
    )
    with pytest.raises(PredictionInputError):
        compare(history, tmp_path)


def test_all_models_missing_is_explicit_and_has_no_fake_estimates(tmp_path):
    result = compare([row(1)], tmp_path)
    assert result.predictions == ()
    assert tuple(f.model_name for f in result.failures) == tuple(FINAL_MODEL_REGISTRY)
    assert all(
        f.code == "artifact_unavailable" and "manifest unavailable" in f.reason
        for f in result.failures
    )


@pytest.mark.parametrize("case", ["hash", "development", "version", "feature_order", "weights"])
def test_incompatible_bundle_is_reported_without_fallback(synthetic_tree, case):
    root, path, weights, payload = synthetic_tree
    if case == "weights":
        weights.write_bytes(b"corrupt")
    elif case == "hash":
        path.write_text('{"payload": {}, "sha256": "wrong"}')
    else:
        if case == "development":
            payload["artifact_role"] = "development"
        elif case == "version":
            payload["model"]["version"] = "unsupported"
        else:
            payload["training"]["feature_names"].reverse()
        rewrite(path, payload)
    result = compare([row(1)], root)
    assert result.predictions == ()
    assert result.failures[0].code == "incompatible_artifact"
    assert result.failures[0].reason


def test_every_model_receives_same_normalized_observed_history(tmp_path, monkeypatch):
    import fleetpulse.services.final_comparison as service
    from fleetpulse.services.final_comparison import FinalModelPrediction

    seen = []

    def observed(path, spec, rows, partition, unit):
        seen.append((rows, partition, unit))
        vector = engineer_features(rows, dataset_partition=partition, unit_id=unit)
        return FinalModelPrediction(
            spec.name,
            spec.version,
            spec.stage,
            5.0,
            "a" * 64,
            None,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            vector,
        )

    monkeypatch.setattr(service, "_predict_bundle", observed)
    result = compare([row(1), row(2)], tmp_path)
    assert len(result.predictions) == 5 and result.failures == ()
    assert all(request is seen[0][0] for request, _, _ in seen)
    assert all((partition, unit) == ("synthetic/P17", 99) for _, partition, unit in seen)
    assert len({p.vector for p in result.predictions}) == 1


def test_mlp_adapter_delegates_raw_features_to_saved_preprocessing(tmp_path, monkeypatch):
    # Adapter double, not a trained MLP result. Native loader/scaler tests are
    # in test_final_mlp_bundle.py and run in CI with the optional torch extra.
    import sys
    from types import ModuleType, SimpleNamespace

    from sklearn.preprocessing import StandardScaler

    module = ModuleType("fleetpulse.models.final_mlp_bundle")
    config = FeatureConfig()
    payload = {
        "artifact_role": "final-training-model-bundle",
        "model": {
            "name": "dense-rul-mlp-final",
            "version": "dense-rul-mlp-final-v1",
            "weights_sha256": "a" * 64,
        },
        "training": {
            "feature_config": json.loads(json.dumps(asdict(config))),
            "feature_names": list(config.output_schema),
            "training_source_sha256": "b" * 64,
            "selection_manifest_sha256": "c" * 64,
        },
    }
    scaler = StandardScaler().fit(np.array([[8.0] * 96, [12.0] * 96]))
    bundle = SimpleNamespace(payload=payload, scaler=scaler)
    module.load_final_mlp_bundle = lambda path: bundle
    received = []

    def predict(loaded, matrix):
        assert loaded is bundle
        received.append(matrix.copy())
        return (float(loaded.scaler.transform(matrix)[0, 0]),)

    module.predict_final_mlp = predict
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / FINAL_MODEL_REGISTRY["dense-rul-mlp-final"].manifest_path
    path.parent.mkdir(parents=True)
    path.write_text("adapter fixture", encoding="utf-8")
    result = compare([row(1), row(2)], tmp_path)
    prediction = result.predictions[0]
    assert prediction.model_name == "dense-rul-mlp-final"
    assert prediction.raw_rul_cycles == -4.75  # (raw first feature .5 - mean10) / scale2.
    assert tuple(received[0][0]) == prediction.vector.values
    assert len(received) == 1


def test_numeric_feature_overflow_is_an_input_error(synthetic_tree):
    root, _, _, _ = synthetic_tree
    history = [row(1), row(2)]
    history[0]["sensor_1"], history[1]["sensor_1"] = -1e308, 1e308
    with pytest.raises(PredictionInputError, match="finite saved-schema features"):
        compare(history, root)


def test_one_estimator_failure_does_not_stop_other_models(tmp_path, monkeypatch):
    import fleetpulse.services.final_comparison as service

    called = []

    def failing(path, spec, rows, partition, unit):
        called.append(spec.name)
        if spec.name == "decision-tree-rul":
            raise ArithmeticError("synthetic estimator failure")
        raise FileNotFoundError("synthetic missing fixture")

    monkeypatch.setattr(service, "_predict_bundle", failing)
    result = compare([row(1)], tmp_path)
    assert called == list(FINAL_MODEL_REGISTRY)
    assert result.predictions == ()
    assert result.failures[0].code == "prediction_failed"
    assert result.failures[0].reason == "synthetic estimator failure"
