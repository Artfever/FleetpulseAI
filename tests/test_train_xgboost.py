"""XGBoost respects the shared engine-separated development table contract."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from xgboost import XGBRegressor

from fleetpulse.data.quality import READING_COLUMNS
from fleetpulse.data.split_manifest import build_split_manifest
from fleetpulse.features import FeatureConfig
from fleetpulse.features.build_dataset import build_fitting_dataset, build_validation_dataset
from fleetpulse.models.train_xgboost import train_xgboost


@pytest.fixture
def xgboost_tables():
    records = [
        {"unit_id": unit, "cycle": cycle, **dict.fromkeys(READING_COLUMNS, cycle + unit)}
        for unit, lifetime in ((1, 5), (2, 7), (3, 9), (4, 11))
        for cycle in range(1, lifetime + 1)
    ]
    manifest = build_split_manifest(
        records,
        dataset_partition="FD001/train",
        source_path="synthetic/train.txt",
        source_sha256="a" * 64,
        validation_fraction=0.5,
    )
    options = {
        "manifest": manifest,
        "source_sha256": "a" * 64,
        "config": FeatureConfig(channels=("sensor_1",)),
    }
    return build_fitting_dataset(records, **options), build_validation_dataset(records, **options)


def test_xgboost_fits_fitting_rows_and_scores_saved_validation_rows(xgboost_tables):
    fitting, validation = xgboost_tables
    original_fit = XGBRegressor.fit
    fit_calls = []

    def observed_fit(estimator, X, y, **kwargs):
        fit_calls.append((X, y))
        return original_fit(estimator, X, y, **kwargs)

    with patch.object(XGBRegressor, "fit", autospec=True, side_effect=observed_fit):
        result = train_xgboost(fitting, validation, n_estimators=8, max_depth=3)

    assert len(fit_calls) == 1
    assert [list(row) for row in fit_calls[0][0]] == [list(row) for row in fitting.X]
    assert list(fit_calls[0][1]) == list(fitting.y)
    assert result.feature_names == fitting.feature_names
    assert result.random_seed == 42
    assert result.parameters["n_jobs"] == 1
    assert result.parameters["device"] == "cpu"
    assert result.parameters["n_estimators"] == len(result.model.get_booster().get_dump()) == 8
    assert [
        (row.dataset_partition, row.unit_id, row.cutoff_cycle) for row in result.predictions
    ] == list(validation.row_keys)
    assert [row.true_rul for row in result.predictions] == list(validation.y)
    assert [row.prediction for row in result.predictions] == pytest.approx(
        result.model.predict(validation.X)
    )
    assert result.metrics.engine_count == len(validation.rows)
    assert result.runtime_seconds >= 0


@pytest.mark.parametrize("kind", ["overlap", "source", "schema", "role"])
def test_invalid_tables_are_rejected_before_xgboost_fits(xgboost_tables, kind):
    fitting, validation = xgboost_tables
    if kind == "overlap":
        validation = replace(validation, rows=(fitting.rows[0],))
    elif kind == "source":
        validation = replace(validation, source_sha256="b" * 64)
    elif kind == "schema":
        validation = replace(validation, config=FeatureConfig(channels=("sensor_2",)))
    else:
        fitting = replace(fitting, role="validation")

    with patch.object(XGBRegressor, "fit", autospec=True) as fit:
        with pytest.raises(ValueError):
            train_xgboost(fitting, validation)
        fit.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"max_depth": 0},
        {"learning_rate": 0},
        {"min_child_weight": -1},
        {"subsample": 1.1},
        {"colsample_bytree": 0},
        {"reg_lambda": float("nan")},
        {"random_seed": True},
    ],
)
def test_xgboost_rejects_invalid_candidate_parameters_before_fit(xgboost_tables, kwargs):
    with patch.object(XGBRegressor, "fit", autospec=True) as fit:
        with pytest.raises(ValueError):
            train_xgboost(*xgboost_tables, **kwargs)
        fit.assert_not_called()
