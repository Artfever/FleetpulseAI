"""Fleet input extraction preserves engine identity and validates peers independently."""

import math

import pytest

from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.services import (
    PredictionArtifactError,
    PredictionInputError,
    RULPrediction,
    extract_latest_engine_histories,
    predict_fleet,
)
from fleetpulse.services import fleet as fleet_service


def _row(unit_id, cycle, *, sensor_1=0.25):
    return {
        field: unit_id if field == "unit_id" else cycle if field == "cycle" else sensor_1 if field == "sensor_1" else 0.5
        for field in HISTORY_COLUMNS
    }


def test_interleaved_histories_are_extracted_per_dataset_qualified_engine():
    rows = [_row(5, 1), _row(4, 1), _row(5, 2), _row(4, 2)]

    fleet = extract_latest_engine_histories(rows, dataset_partition="FD001/train")

    assert not fleet.errors
    assert [history.identity.dataset_partition for history in fleet.histories] == [
        "FD001/train",
        "FD001/train",
    ]
    assert [history.identity.unit_id for history in fleet.histories] == [4, 5]
    assert [history.observed_cycle for history in fleet.histories] == [2, 2]
    assert [
        [row["cycle"] for row in history.records] for history in fleet.histories
    ] == [[1, 2], [1, 2]]
    assert [row["unit_id"] for row in rows] == [5, 4, 5, 4]


def test_invalid_engine_is_reported_by_original_identity_without_dropping_valid_peer():
    rows = [_row(4, 1), _row(5, 1, sensor_1=math.nan), _row(4, 2)]

    fleet = extract_latest_engine_histories(rows, dataset_partition="FD001/train")

    assert [(item.identity.unit_id, item.observed_cycle) for item in fleet.histories] == [(4, 2)]
    assert len(fleet.errors) == 1
    assert fleet.errors[0].identity.dataset_partition == "FD001/train"
    assert fleet.errors[0].identity.unit_id == 5
    assert fleet.errors[0].codes == ("non_finite_reading",)


def test_bad_cycle_order_is_engine_local_and_rows_are_not_reordered():
    rows = [_row(4, 1), _row(5, 1), _row(4, 2), _row(5, 3), _row(5, 2)]

    fleet = extract_latest_engine_histories(rows, dataset_partition="FD001/train")

    assert [item.identity.unit_id for item in fleet.histories] == [4]
    assert [row["cycle"] for row in fleet.histories[0].records] == [1, 2]
    assert fleet.errors[0].identity.unit_id == 5
    assert fleet.errors[0].codes == ("cycle_order",)


@pytest.mark.parametrize(
    ("records", "dataset_partition", "message"),
    [
        ([], "FD001/train", "at least one"),
        ([_row(1, 1)], " ", "dataset_partition"),
        ([{"cycle": 1}], "FD001/train", "unit_id"),
        ([None], "FD001/train", "must be a mapping"),
    ],
)
def test_unidentifiable_fleet_input_is_rejected(records, dataset_partition, message):
    with pytest.raises(PredictionInputError, match=message):
        extract_latest_engine_histories(records, dataset_partition=dataset_partition)


def test_fleet_predictions_keep_dataset_engine_cycle_and_model_context_attached(monkeypatch):
    rows = [_row(5, 1), _row(4, 1), _row(5, 2), _row(4, 2)]
    calls = []

    def controlled_prediction(history, **kwargs):
        unit_id = kwargs["unit_id"]
        cycles = [row["cycle"] for row in history]
        calls.append((unit_id, cycles, kwargs))
        estimate = float(unit_id + cycles[-1])
        return RULPrediction(
            estimate,
            estimate,
            "test-model",
            "test-v1",
            cycles[-1],
            len(cycles),
            (),
        )

    monkeypatch.setattr(fleet_service, "predict_rul_with_context", controlled_prediction)

    results = predict_fleet(
        rows,
        dataset_partition="FD001/train",
        manifest_path="test-manifest.json",
    )

    assert [(result.identity.dataset_partition, result.identity.unit_id) for result in results] == [
        ("FD001/train", 4),
        ("FD001/train", 5),
    ]
    assert [(result.observed_cycle, result.raw_rul_cycles) for result in results] == [(2, 6.0), (2, 7.0)]
    assert [result.model_version for result in results] == ["test-v1", "test-v1"]
    assert [result.input_window_rows for result in results] == [2, 2]
    assert all(result.error is None and not result.error_codes for result in results)
    assert [(unit_id, cycles) for unit_id, cycles, _ in calls] == [(4, [1, 2]), (5, [1, 2])]
    assert all(kwargs["dataset_partition"] == "FD001/train" for _, _, kwargs in calls)


def test_fleet_prediction_retains_invalid_history_error_and_valid_peer(monkeypatch):
    rows = [_row(4, 1), _row(5, 1, sensor_1=math.nan), _row(4, 2)]
    predicted_ids = []

    def controlled_prediction(history, **kwargs):
        predicted_ids.append(kwargs["unit_id"])
        return RULPrediction(8.0, 8.0, "test-model", "test-v1", 2, 2, ())

    monkeypatch.setattr(fleet_service, "predict_rul_with_context", controlled_prediction)

    results = predict_fleet(
        rows,
        dataset_partition="FD001/train",
        manifest_path="test-manifest.json",
    )

    assert [result.identity.unit_id for result in results] == [4, 5]
    assert results[0].raw_rul_cycles == 8.0
    assert results[0].error is None
    assert results[1].raw_rul_cycles is None
    assert results[1].identity.unit_id == 5
    assert results[1].error_codes == ("non_finite_reading",)
    assert "non_finite_reading" in results[1].error
    assert predicted_ids == [4]


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    [
        (PredictionInputError("invalid prefix"), "prediction_input_error"),
        (PredictionArtifactError("bundle unavailable"), "prediction_artifact_error"),
    ],
)
def test_fleet_prediction_errors_stay_attached_to_the_engine(monkeypatch, exception, expected_code):
    rows = [_row(4, 1), _row(5, 1)]

    def fails_for_engine_five(history, **kwargs):
        if kwargs["unit_id"] == 5:
            raise exception
        return RULPrediction(9.0, 9.0, "test-model", "test-v1", 1, 1, ())

    monkeypatch.setattr(fleet_service, "predict_rul_with_context", fails_for_engine_five)

    results = predict_fleet(
        rows,
        dataset_partition="FD001/train",
        manifest_path="test-manifest.json",
    )

    assert [result.identity.unit_id for result in results] == [4, 5]
    assert results[0].raw_rul_cycles == 9.0
    assert results[0].error is None
    assert results[1].identity.unit_id == 5
    assert results[1].observed_cycle == 1
    assert results[1].error_codes == (expected_code,)
    assert str(exception) == results[1].error
