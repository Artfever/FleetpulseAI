"""Saved XGBoost candidates reload through the common registry/service contract."""

import hashlib
import json
from unittest.mock import Mock

import pytest

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig, engineer_features
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.models.train_xgboost import train_xgboost
from fleetpulse.models.xgboost_artifact import (
    RELOAD_DISTRIBUTIONS,
    runtime_software_versions,
    save_xgboost_artifact,
    verify_xgboost_artifact,
)
from fleetpulse.services import (
    PredictionArtifactError,
    PredictionInputError,
    predict_rul,
    predict_rul_with_context,
)
from fleetpulse.services import predictor as predictor_service


def _row(cycle, unit):
    return {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, cycle * unit / 10)}


@pytest.fixture
def candidate(tmp_path):
    histories = [_row(cycle, unit) for unit in range(1, 5) for cycle in range(1, unit + 5)]
    split = build_split_manifest(
        histories,
        dataset_partition="FD001/train",
        source_path="synthetic/xgboost.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
    )
    config = FeatureConfig(channels=("sensor_2", "sensor_1"), window_cycles=2)
    fitting = build_fitting_dataset(
        histories, manifest=split, source_sha256="a" * 64, config=config
    )
    validation = build_validation_dataset(
        histories, manifest=split, source_sha256="a" * 64, config=config
    )
    result = train_xgboost(
        fitting, validation, n_estimators=8, max_depth=2, random_seed=42
    )
    saved = save_xgboost_artifact(
        result,
        fitting,
        validation,
        tmp_path,
        fitting_dataset_sha256="b" * 64,
        validation_dataset_sha256="c" * 64,
    )
    return histories, validation, result, saved


def test_candidate_artifact_and_service_replay_predictions(candidate):
    histories, validation, result, saved = candidate

    verified = verify_xgboost_artifact(saved["manifest_path"], validation)
    assert verified["prediction_count"] == len(validation.rows)
    assert verified["mae_cycles"] == result.metrics.mae
    assert verified["rmse_cycles"] == result.metrics.rmse

    engine = validation.rows[0].vector.unit_id
    cutoff = validation.rows[0].vector.selected_cycle
    prefix = [row for row in histories if row["unit_id"] == engine and row["cycle"] <= cutoff]
    expected_vector = engineer_features(
        prefix,
        dataset_partition="FD001/train",
        unit_id=engine,
        config=validation.config,
    )
    expected = float(result.model.predict((expected_vector.values,))[0])
    raw = predict_rul(
        prefix,
        dataset_partition="FD001/train",
        unit_id=engine,
        manifest_path=saved["manifest_path"],
    )
    contextual = predict_rul_with_context(
        prefix,
        dataset_partition="FD001/train",
        unit_id=engine,
        manifest_path=saved["manifest_path"],
    )
    assert raw.raw_rul_cycles == expected
    assert contextual.raw_rul_cycles == expected
    assert contextual.model_name == "xgboost-rul"
    assert contextual.model_version == "xgboost-rul-v1"
    document = json.loads(saved["manifest_path"].read_text(encoding="utf-8"))
    metadata = document["payload"]["model"]["runtime_versions"]
    assert set(metadata) == {"python", *RELOAD_DISTRIBUTIONS}
    assert metadata == runtime_software_versions()


def test_registry_rejects_xgboost_version_mismatch_after_manifest_rehash(candidate):
    _, _, _, saved = candidate
    path = saved["manifest_path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    payload["model"]["xgboost_version"] = "0.0.0"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PredictionArtifactError, match="XGBoost version"):
        predict_rul(
            [_row(1, 1)],
            dataset_partition="FD001/train",
            unit_id=1,
            manifest_path=path,
        )


def test_registry_rejects_missing_runtime_versions_after_manifest_rehash(candidate):
    _, _, _, saved = candidate
    path = saved["manifest_path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    del payload["model"]["runtime_versions"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PredictionArtifactError, match="complete runtime version metadata"):
        predict_rul(
            [_row(1, 1)],
            dataset_partition="FD001/train",
            unit_id=1,
            manifest_path=path,
        )


def test_registry_rejects_reload_package_mismatch_after_manifest_rehash(candidate):
    _, _, _, saved = candidate
    path = saved["manifest_path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    payload["model"]["runtime_versions"]["numpy"] = "0.0.0"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PredictionArtifactError, match="package versions required"):
        predict_rul(
            [_row(1, 1)],
            dataset_partition="FD001/train",
            unit_id=1,
            manifest_path=path,
        )


@pytest.mark.parametrize("invalid_reading", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_xgboost_reading_is_rejected_before_model_load_or_predict(
    candidate, invalid_reading, monkeypatch
):
    histories, validation, _, saved = candidate
    validation_row = validation.rows[0]
    engine = validation_row.vector.unit_id
    cutoff = validation_row.vector.selected_cycle
    prefix = [
        dict(row)
        for row in histories
        if row["unit_id"] == engine and row["cycle"] <= cutoff
    ]
    prefix[0]["sensor_1"] = invalid_reading

    model_load = Mock()
    estimator = Mock()
    deserializer = Mock(return_value=estimator)
    monkeypatch.setattr(predictor_service, "_load_model", model_load)
    monkeypatch.setattr(predictor_service, "_deserialize_model", deserializer)

    with pytest.raises(PredictionInputError, match="non_finite_reading"):
        predict_rul(
            prefix,
            dataset_partition="FD001/train",
            unit_id=engine,
            manifest_path=saved["manifest_path"],
        )

    model_load.assert_not_called()
    deserializer.assert_not_called()
    estimator.predict.assert_not_called()
