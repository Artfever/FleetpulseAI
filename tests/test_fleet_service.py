"""Fleet input extraction preserves engine identity and validates peers independently."""

import math

import pytest

from fleetpulse.data.loader import HISTORY_COLUMNS
from fleetpulse.services import PredictionInputError, extract_latest_engine_histories


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
