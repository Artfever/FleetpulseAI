"""Synthetic registered-forest inference and artifact-boundary regression tests."""

import hashlib
import json
from dataclasses import asdict

import joblib
import pytest
from sklearn.tree import DecisionTreeRegressor

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig, engineer_features
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.models.forest_artifact import save_random_forest_artifact
from fleetpulse.models.train_forest import train_random_forest
from fleetpulse.services import PredictionArtifactError, predict_rul, predict_rul_with_context


def row(cycle, unit=99):
    return {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, cycle * 2)}


@pytest.fixture
def forest(tmp_path):
    rows = [row(c, u) for u in range(1, 5) for c in range(1, u + 5)]
    split = build_split_manifest(
        rows,
        dataset_partition="FD001/train",
        source_path="synthetic/train.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
    )
    config = FeatureConfig(channels=("sensor_2", "sensor_1"), window_cycles=2)
    options = {"manifest": split, "source_sha256": "a" * 64, "config": config}
    fitting = build_fitting_dataset(rows, **options)
    validation = build_validation_dataset(rows, **options)
    fitted = train_random_forest(
        fitting, validation, n_estimators=8, max_depth=4, min_samples_leaf=1
    )
    paths = save_random_forest_artifact(
        fitted,
        fitting,
        validation,
        tmp_path,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )
    return paths, fitted.model, config


def predict(history, path, contextual=False):
    function = predict_rul_with_context if contextual else predict_rul
    return function(history, dataset_partition="synthetic/forest", unit_id=99, manifest_path=path)


def rewrite(path, document):
    content = json.dumps(
        document["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    document["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")


def test_forest_uses_shared_saved_features_and_unchanged_return_contract(forest):
    paths, model, config = forest
    history = [row(c) for c in (1, 2, 3)]
    vector = engineer_features(
        history, dataset_partition="synthetic/forest", unit_id=99, config=config
    )
    expected = float(model.predict([vector.values])[0])
    assert asdict(predict(history, paths["manifest_path"])) == {"raw_rul_cycles": expected}
    result = predict(history, paths["manifest_path"], contextual=True)
    assert result.raw_rul_cycles == expected
    assert result.displayed_rul_cycles == max(0, expected)
    assert result.model_name == "random-forest-rul"
    assert result.model_version == "random-forest-rul-v1"
    assert result.observed_cycle == 3
    assert result.input_window_rows == 2
    assert result.warnings == ()


@pytest.mark.parametrize(
    "change", ["version", "filename", "schema", "parameters", "features", "binary", "type"]
)
def test_forest_registry_rejects_incompatible_artifact_even_after_cache_use(forest, change):
    paths, _, _ = forest
    path = paths["manifest_path"]
    predict([row(1)], path)  # Populate the deserialization cache first.
    document = json.loads(path.read_text())
    payload = document["payload"]
    if change == "version":
        payload["model"]["version"] = "unsupported"
    elif change == "filename":
        payload["model"]["filename"] = "../elsewhere.joblib"
    elif change == "schema":
        payload["schema_version"] = "decision-tree-validation-v1"
    elif change == "parameters":
        payload["training"]["parameters"]["n_estimators"] = 999
    elif change == "features":
        payload["training"]["feature_names"].reverse()
    elif change == "binary":
        paths["model_path"].write_bytes(b"broken")
    else:
        wrong = DecisionTreeRegressor().fit([[0] * 8, [1] * 8], [1, 2])
        joblib.dump(wrong, paths["model_path"])
        payload["model"]["sha256"] = hashlib.sha256(paths["model_path"].read_bytes()).hexdigest()
    rewrite(path, document)
    with pytest.raises(PredictionArtifactError):
        predict([row(1)], path)
