"""Independent P09-A1 audit; all readings and targets here are synthetic."""

import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
from xgboost import XGBRegressor

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.features.dataset_io import load_feature_dataset, save_feature_dataset
from fleetpulse.models.train_xgboost import train_xgboost


@pytest.fixture
def audit_tables():
    records = [
        {
            "unit_id": unit,
            "cycle": cycle,
            **dict.fromkeys(READING_COLUMNS, 0.5),
            "sensor_1": 2 * cycle,
            "sensor_2": 10 * cycle,
        }
        for unit, last in ((1, 4), (2, 6), (3, 8), (4, 10))
        for cycle in range(1, last + 1)
    ]
    manifest = build_split_manifest(
        records,
        dataset_partition="FD001/train",
        source_path="synthetic/train.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
        minimum_history_rows=2,
    )
    config = FeatureConfig(channels=("sensor_2", "sensor_1"), window_cycles=2)
    options = {"manifest": manifest, "source_sha256": "a" * 64, "config": config}
    fitting = build_fitting_dataset(records, **options)
    validation = build_validation_dataset(records, **options)
    return records, options, fitting, validation


def test_independent_numeric_order_targets_and_exact_estimator_boundary(audit_tables):
    records, _, fitting, validation = audit_tables
    sample_index = next(i for i, row in enumerate(fitting.rows) if row.vector.selected_cycle == 2)
    sample = fitting.rows[sample_index]
    lifetime = max(row["cycle"] for row in records if row["unit_id"] == sample.vector.unit_id)
    # Hand calculation: sensor2 first (20,15,5,10), then sensor1 (4,3,1,2).
    expected = (20, 15, 5, 10, 4, 3, 1, 2)
    assert fitting.X[sample_index] == expected
    assert fitting.y[sample_index] == lifetime - 2
    assert fitting.feature_names == (
        "sensor_2_latest",
        "sensor_2_mean",
        "sensor_2_std",
        "sensor_2_slope",
        "sensor_1_latest",
        "sensor_1_mean",
        "sensor_1_std",
        "sensor_1_slope",
    )
    original_fit, original_predict = XGBRegressor.fit, XGBRegressor.predict
    fitting_calls, prediction_calls = [], []

    def observed_fit(model, X, y, **kwargs):
        fitting_calls.append((X, y, kwargs))
        return original_fit(model, X, y, **kwargs)

    def observed_predict(model, X, **kwargs):
        prediction_calls.append((X, kwargs))
        return original_predict(model, X, **kwargs)

    with (
        patch.object(XGBRegressor, "fit", observed_fit),
        patch.object(XGBRegressor, "predict", observed_predict),
    ):
        result = train_xgboost(fitting, validation, n_estimators=8, max_depth=2)
    assert fitting_calls == [(fitting.X, fitting.y, {})]  # No eval_set or validation labels.
    assert prediction_calls == [(validation.X, {})]
    assert result.feature_names == fitting.feature_names
    assert [row.true_rul for row in result.predictions] == list(validation.y)
    print(
        json.dumps(
            {
                "synthetic": True,
                "unit_id": sample.vector.unit_id,
                "cycle": 2,
                "complete_lifetime": lifetime,
                "X": expected,
                "target_rul_cycles": fitting.y[sample_index],
                "validation_raw_predictions": [row.prediction for row in result.predictions],
            }
        )
    )


@pytest.mark.parametrize("changed", ["labels", "features"])
def test_validation_cannot_change_fitted_booster_or_parameters(audit_tables, changed):
    _, _, fitting, validation = audit_tables
    baseline = train_xgboost(fitting, validation, n_estimators=8, max_depth=2)
    rows = tuple(
        replace(row, true_rul=row.true_rul + 1000)
        if changed == "labels"
        else replace(
            row, vector=replace(row.vector, values=tuple(v + 1000 for v in row.vector.values))
        )
        for row in validation.rows
    )
    mutated = train_xgboost(fitting, replace(validation, rows=rows), n_estimators=8, max_depth=2)
    assert mutated.parameters == baseline.parameters
    assert mutated.model.get_booster().get_dump() == baseline.model.get_booster().get_dump()
    assert list(mutated.model.predict(validation.X)) == list(baseline.model.predict(validation.X))
    if changed == "labels":
        assert [row.prediction for row in mutated.predictions] == [
            row.prediction for row in baseline.predictions
        ]
        assert mutated.metrics.mae != baseline.metrics.mae


def test_hidden_validation_readings_do_not_enter_features(audit_tables):
    records, options, fitting, validation = audit_tables
    cutoffs = {row.vector.unit_id: row.vector.selected_cycle for row in validation.rows}
    poisoned = [
        {**row, "sensor_1": 1e6, "sensor_2": -1e6}
        if row["unit_id"] in cutoffs and row["cycle"] > cutoffs[row["unit_id"]]
        else row
        for row in records
    ]
    rebuilt = build_validation_dataset(poisoned, **options)
    assert rebuilt.X == validation.X
    assert rebuilt.y == validation.y
    assert rebuilt.row_keys == validation.row_keys
    assert build_fitting_dataset(poisoned, **options) == fitting


@pytest.mark.parametrize("changed", ["official_partition", "feature_order"])
def test_checked_table_loader_rejects_mislabelled_inputs_before_training(
    audit_tables, tmp_path, changed
):
    _, _, fitting, _ = audit_tables
    path = tmp_path / "fitting.json"
    save_feature_dataset(fitting, path)
    document = json.loads(path.read_text())
    if changed == "official_partition":
        for row in document["payload"]["rows"]:
            row["metadata"]["dataset_partition"] = "FD001/test"
            row["key"][0] = "FD001/test"
    else:
        document["payload"]["feature_names"].reverse()
    content = json.dumps(
        document["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    document["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")
    with patch.object(XGBRegressor, "fit") as fit:
        with pytest.raises(ValueError, match="source partition|feature order/schema"):
            load_feature_dataset(path)
        fit.assert_not_called()
